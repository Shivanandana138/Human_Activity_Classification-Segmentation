"""
Single-HMM Human Activity Recognition (no ASM, no per-activity HMMs)
====================================================================

ONE HMM whose hidden-state space covers ALL activities:

    hidden states  = 6 activities x K sub-states   (N = 6*K)
    state s        -> activity  s // K
    A (N x N)      = one shared transition matrix  (within- AND between-activity moves)
    pi (N)         = initial state distribution
    B              = diagonal-Gaussian emission per state (mean[s], var[s])

Algorithms inside the HMM
-------------------------
    forward / backward (log-space)     -> likelihood, posteriors
    Viterbi                            -> most probable state path (offline decoding)
    forward filtering                  -> online (causal) decoding
    Baum-Welch (EM)                    -> optional unsupervised refinement
    supervised MLE initialisation      -> counts from labelled windows

Prediction = map decoded hidden state -> activity (s // K).

Evaluation (same style as the 6-HMM+ASM paper): accuracy, per-class
precision / recall / F1, macro-F1, confusion matrix (counts + row-normalised).

Data: UCI HAR (X_*.txt, y_*.txt, subject_*.txt). If the folder is missing, a
synthetic stand-in is generated so the script runs end to end.

Usage:
    python single_hmm_har.py --data /path/to/UCI_HAR_Dataset --k 3 --pca 30
    python single_hmm_har.py                      # synthetic demo
"""
import argparse
import os
import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)

ACTIVITIES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
EPS = 1e-12


# --------------------------------------------------------------------------
# 1. The single HMM
# --------------------------------------------------------------------------
class SingleHMM:
    def __init__(self, n_activities=6, k_sub=1, var_floor=1e-3, seed=0):
        self.C = n_activities
        self.K = k_sub
        self.N = n_activities * k_sub
        self.var_floor = var_floor
        self.seed = seed

    # ---------- emission model ----------
    def _log_emission(self, X):
        """log b_s(x_t) for every state -> (T, N); diagonal Gaussian."""
        d = X.shape[1]
        diff2 = (X[:, None, :] - self.mu[None]) ** 2 / self.var[None]
        return -0.5 * (diff2.sum(2) + np.log(self.var).sum(1)[None] + d * np.log(2 * np.pi))

    # ---------- supervised initialisation (hidden states from labels) ----------
    def fit_supervised(self, seqs, labels, smooth=1.0):
        """
        seqs   : list of (T_i, d) feature arrays (one per subject/recording)
        labels : list of (T_i,) activity ids 0..C-1
        Sub-states of each activity come from k-means on that activity's windows.
        """
        X_all = np.vstack(seqs)
        y_all = np.concatenate(labels)
        d = X_all.shape[1]
        self.mu = np.zeros((self.N, d))
        self.var = np.ones((self.N, d))
        km_models = {}
        for c in range(self.C):
            Xc = X_all[y_all == c]
            if self.K == 1:
                assign = np.zeros(len(Xc), int)
            else:
                km = KMeans(self.K, n_init=5, random_state=self.seed).fit(Xc)
                assign = km.labels_
            for k in range(self.K):
                s = c * self.K + k
                pts = Xc[assign == k]
                self.mu[s] = pts.mean(0)
                self.var[s] = pts.var(0) + self.var_floor
            km_models[c] = assign

        # hidden state sequence: nearest sub-state (by likelihood) within the true activity
        state_seqs = []
        for X, y in zip(seqs, labels):
            logB = self._log_emission(X)
            mask = np.full_like(logB, -np.inf)
            for c in range(self.C):
                mask[y == c, c * self.K:(c + 1) * self.K] = 0.0
            state_seqs.append((logB + mask).argmax(1))

        # transition / initial counts (+ Laplace smoothing)
        A = np.full((self.N, self.N), smooth)
        pi = np.full(self.N, smooth)
        for s_seq in state_seqs:
            pi[s_seq[0]] += 1
            np.add.at(A, (s_seq[:-1], s_seq[1:]), 1)
        self.A = A / A.sum(1, keepdims=True)
        self.pi = pi / pi.sum()
        return self

    # ---------- forward / backward ----------
    def forward(self, X):
        logB = self._log_emission(X)
        logA, T = np.log(self.A + EPS), len(X)
        la = np.zeros((T, self.N))
        la[0] = np.log(self.pi + EPS) + logB[0]
        for t in range(1, T):
            la[t] = logsumexp(la[t - 1][:, None] + logA, axis=0) + logB[t]
        return la, logB

    def backward(self, logB):
        logA, T = np.log(self.A + EPS), len(logB)
        lb = np.zeros((T, self.N))
        for t in range(T - 2, -1, -1):
            lb[t] = logsumexp(logA + (logB[t + 1] + lb[t + 1])[None], axis=1)
        return lb

    def log_likelihood(self, X):
        la, _ = self.forward(X)
        return logsumexp(la[-1])

    # ---------- Baum-Welch (optional refinement) ----------
    def baum_welch(self, seqs, n_iter=5, verbose=True):
        for it in range(n_iter):
            N, d = self.N, self.mu.shape[1]
            A_num = np.zeros((N, N)); pi_num = np.zeros(N)
            g_sum = np.zeros(N); gx = np.zeros((N, d)); gx2 = np.zeros((N, d))
            total_ll = 0.0
            logA = np.log(self.A + EPS)
            for X in seqs:
                la, logB = self.forward(X)
                lb = self.backward(logB)
                ll = logsumexp(la[-1]); total_ll += ll
                gamma = np.exp(la + lb - ll)
                pi_num += gamma[0]
                for t in range(len(X) - 1):
                    xi = la[t][:, None] + logA + (logB[t + 1] + lb[t + 1])[None] - ll
                    A_num += np.exp(xi)
                g_sum += gamma.sum(0)
                gx += gamma.T @ X
                gx2 += gamma.T @ (X ** 2)
            self.A = (A_num + EPS) / (A_num + EPS).sum(1, keepdims=True)
            self.pi = (pi_num + EPS) / (pi_num + EPS).sum()
            self.mu = gx / (g_sum[:, None] + EPS)
            self.var = np.maximum(gx2 / (g_sum[:, None] + EPS) - self.mu ** 2, self.var_floor)
            if verbose:
                print(f"  Baum-Welch iter {it + 1}/{n_iter}  total log-lik = {total_ll:.1f}")
        return self

    # ---------- decoding ----------
    def viterbi(self, X):
        logB = self._log_emission(X)
        logA, T = np.log(self.A + EPS), len(X)
        delta = np.zeros((T, self.N)); psi = np.zeros((T, self.N), int)
        delta[0] = np.log(self.pi + EPS) + logB[0]
        for t in range(1, T):
            cand = delta[t - 1][:, None] + logA
            psi[t] = cand.argmax(0)
            delta[t] = cand.max(0) + logB[t]
        path = np.zeros(T, int)
        path[-1] = delta[-1].argmax()
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def filter_decode(self, X):
        """Online/causal decoding: argmax_s P(s_t | x_1..t)."""
        la, _ = self.forward(X)
        return (la - logsumexp(la, axis=1, keepdims=True)).argmax(1)

    def predict(self, seqs, method="viterbi"):
        fn = self.viterbi if method == "viterbi" else self.filter_decode
        return [fn(X) // self.K for X in seqs]      # hidden state -> activity


# --------------------------------------------------------------------------
# 2. Data
# --------------------------------------------------------------------------
def group_by_subject(X, y, subj):
    seqs, labs = [], []
    for s in np.unique(subj):
        idx = np.where(subj == s)[0]
        seqs.append(X[idx]); labs.append(y[idx])
    return seqs, labs


def load_uci_har(root):
    def rd(split):
        X = np.loadtxt(os.path.join(root, split, f"X_{split}.txt"))
        y = np.loadtxt(os.path.join(root, split, f"y_{split}.txt"), dtype=int) - 1
        s = np.loadtxt(os.path.join(root, split, f"subject_{split}.txt"), dtype=int)
        return X, y, s
    return rd("train"), rd("test")


def synthetic_har(n_subj=10, d=40, seed=0):
    """Stand-in data: each activity = 2 Gaussian modes, sticky activity bouts."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 2.0, (6, 2, d))
    def make(n_sub, off):
        X, y, s = [], [], []
        for sub in range(n_sub):
            act = rng.integers(6)
            for _ in range(60):                      # 60 bouts per subject
                if rng.random() < 0.6: act = rng.integers(6)
                for _ in range(rng.integers(5, 15)):
                    m = rng.integers(2)
                    X.append(centers[act, m] + rng.normal(0, 1.2, d)); y.append(act); s.append(sub + off)
        return np.array(X), np.array(y), np.array(s)
    return make(n_subj, 0), make(n_subj // 2 + 1, 100)


# --------------------------------------------------------------------------
# 3. Evaluation (accuracy, P/R/F1, confusion matrix)
# --------------------------------------------------------------------------
def evaluate(y_true, y_pred, names=ACTIVITIES, title=""):
    labels = list(range(len(names)))
    acc = accuracy_score(y_true, y_pred)
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    print(f"\n===== {title} =====")
    print(f"Accuracy : {acc:.4f}   Macro-F1 : {f.mean():.4f}   "
          f"Macro-P : {p.mean():.4f}   Macro-R : {r.mean():.4f}")
    print(f"{'Activity':<12}{'Prec':>8}{'Recall':>8}{'F1':>8}{'Support':>9}")
    for i, n in enumerate(names):
        print(f"{n:<12}{p[i]:>8.3f}{r[i]:>8.3f}{f[i]:>8.3f}{sup[i]:>9d}")
    print("Confusion matrix (counts):\n", cm)
    return dict(acc=acc, p=p, r=r, f1=f, cm=cm, cm_norm=cmn)


def plot_confusion(cmn, names, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > .5 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Single HMM - confusion matrix")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------
# 4. Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="UCI HAR dataset root")
    ap.add_argument("--k", type=int, default=2, help="hidden sub-states per activity")
    ap.add_argument("--pca", type=int, default=30, help="PCA dims (0 = none)")
    ap.add_argument("--em", type=int, default=0, help="Baum-Welch refinement iterations")
    ap.add_argument("--out", default="/mnt/user-data/outputs")
    a = ap.parse_args()

    (Xtr, ytr, str_), (Xte, yte, ste) = (load_uci_har(a.data) if a.data and os.path.isdir(a.data)
                                         else synthetic_har())
    if not (a.data and os.path.isdir(a.data)):
        print("[info] dataset folder not found -> using synthetic demo data")

    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if a.pca:
        pca = PCA(a.pca, random_state=0).fit(Xtr)
        Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)

    tr_seqs, tr_labs = group_by_subject(Xtr, ytr, str_)
    te_seqs, te_labs = group_by_subject(Xte, yte, ste)

    hmm = SingleHMM(k_sub=a.k).fit_supervised(tr_seqs, tr_labs)
    if a.em:
        hmm.baum_welch(tr_seqs, n_iter=a.em)

    y_true = np.concatenate(te_labs)
    res = {}
    for method in ("viterbi", "filter"):
        y_pred = np.concatenate(hmm.predict(te_seqs, method))
        res[method] = evaluate(y_true, y_pred, title=f"Single HMM (K={a.k}) - {method} decoding")
    os.makedirs(a.out, exist_ok=True)
    plot_confusion(res["viterbi"]["cm_norm"], ACTIVITIES, os.path.join(a.out, "single_hmm_confusion.png"))
    np.savez(os.path.join(a.out, "single_hmm_results.npz"),
             **{f"{m}_{k}": v for m, r in res.items() for k, v in r.items()})
    print("\nTransition matrix shape:", hmm.A.shape, "| saved confusion plot + results.npz")


if __name__ == "__main__":
    main()
