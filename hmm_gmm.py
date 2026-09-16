import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp

class GMMHMM:
    """
    Left-to-right-ish HMM with 8 states and a 10-component diagonal GMM in each state.

    Training uses an EM/Baum-Welch implementation:
      - initialize each state's GMM using frame chunks / global GMM fallback
      - forward-backward
      - update state mixture responsibilities
      - update transition matrix and initial state probabilities

    The implementation is deliberately written in this project so the HMM is inspectable.
    """

    def __init__(self, n_states=8, n_mix=10, random_state=42, max_iter=25,
                 gmm_max_iter=50, reg_covar=1e-4, smoothing=1e-2):
        self.n_states = n_states
        self.n_mix = n_mix
        self.random_state = random_state
        self.max_iter = max_iter
        self.gmm_max_iter = gmm_max_iter
        self.reg_covar = reg_covar
        self.smoothing = smoothing
        self.pi_ = None
        self.A_ = None
        self.gmms_ = []
        self.fitted_ = False

    def _log_emission(self, X):
        # [T, S]
        return np.column_stack([g.score_samples(X) for g in self.gmms_])

    def _forward(self, logB):
        T, S = logB.shape
        alpha = np.full((T, S), -np.inf)
        alpha[0] = np.log(self.pi_) + logB[0]
        for t in range(1, T):
            alpha[t] = logB[t] + logsumexp(
                alpha[t-1][:, None] + np.log(self.A_), axis=0
            )
        return alpha

    def _backward(self, logB):
        T, S = logB.shape
        beta = np.full((T, S), -np.inf)
        beta[-1] = 0.0
        for t in range(T-2, -1, -1):
            beta[t] = logsumexp(
                np.log(self.A_) + logB[t+1][None, :] + beta[t+1][None, :],
                axis=1
            )
        return beta

    def _initialize(self, sequences):
        rng = np.random.default_rng(self.random_state)
        allX = np.vstack(sequences)
        # Initial state distribution: start in state 0.
        #self.pi_ = np.zeros(self.n_states)
        #self.pi_[0] = 1.0
        self.pi_ = np.full(self.n_states, self.smoothing)
        self.pi_[0] = 1.0
        self.pi_ /= self.pi_.sum()

        # Initialize transitions as a left-to-right topology with self-loop.
        A = np.full((self.n_states, self.n_states),self.smoothing)
        for i in range(self.n_states):
             A[i, i] = 0.65
             if i < self.n_states - 1:
                  A[i, i + 1] = 0.35
        A[-1, :] = self.smoothing
        A[-1, -1] = 1.0
        A /= A.sum(axis=1, keepdims=True)
        self.A_ = A

        self.gmms_ = []
        # Each state's initialization gets a different region of each activity sequence.
        # If a state does not have enough samples, fall back to random samples from allX.
        for s in range(self.n_states):
            chunks = []
            for seq in sequences:
                idx = np.linspace(0, len(seq)-1, self.n_states).astype(int)[s]
                lo = max(0, idx - max(1, len(seq)//self.n_states//2))
                hi = min(len(seq), idx + max(2, len(seq)//self.n_states//2))
                if hi > lo:
                    chunks.append(seq[lo:hi])
            Xs = np.vstack(chunks) if chunks else allX
            if len(Xs) < self.n_mix * 2:
                ids = rng.choice(len(allX), size=min(len(allX), self.n_mix*20), replace=False)
                Xs = allX[ids]
            # sklearn GMM cannot use more components than samples.
            k = min(self.n_mix, max(1, len(Xs)))
            g = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                max_iter=self.gmm_max_iter,
                reg_covar=self.reg_covar,
                random_state=self.random_state + s,
                init_params="kmeans",
            )
            g.fit(Xs)
            self.gmms_.append(g)

    def fit(self, sequences, verbose=True):
        self._initialize(sequences)
        prev_total = -np.inf

        for it in range(self.max_iter):
            gamma_sum = np.zeros(self.n_states)
            xi_sum = np.zeros((self.n_states, self.n_states))
            total_ll = 0.0
            state_samples = [[] for _ in range(self.n_states)]

            for X in sequences:
                if len(X) < 2:
                    continue
                logB = self._log_emission(X)
                alpha = self._forward(logB)
                beta = self._backward(logB)
                ll = logsumexp(alpha[-1])
                total_ll += ll

                log_gamma = alpha + beta - ll
                gamma = np.exp(log_gamma)
                gamma_sum += gamma.sum(axis=0)

                for s in range(self.n_states):
                    # Weighted samples are not directly supported by all GMM versions
                    # for refitting, so retain samples and use responsibilities below.
                    state_samples[s].append((X, gamma[:, s]))

                if len(X) > 1:
                    log_xi = (
                        alpha[:-1, :, None]
                        + np.log(self.A_)[None, :, :]
                        + logB[1:, None, :]
                        + beta[1:, None, :]
                        - ll
                    )
                    xi_sum += np.exp(log_xi).sum(axis=0)

            # Transition re-estimation.
            A_new = xi_sum + self.smoothing
            A_new /= A_new.sum(axis=1, keepdims=True)
            self.A_ = A_new

            # Initial state distribution.
            # Preserve a strong start-state prior while allowing data to update it.
            pi_acc = np.zeros(self.n_states)
            for X in sequences:
                if len(X) < 2:
                    continue
                logB = self._log_emission(X)
                a = self._forward(logB)
                b = self._backward(logB)
                ll = logsumexp(a[-1])
                pi_acc += np.exp(a[0] + b[0] - ll)
            if pi_acc.sum() > 0:
                self.pi_ = (pi_acc + self.smoothing)
                self.pi_ /= self.pi_.sum()

            # Refit each state's GMM approximately using posterior-weighted resampling.
            rng = np.random.default_rng(self.random_state + it)
            new_gmms = []
            for s in range(self.n_states):
                Xparts, Wparts = [], []
                for X, w in state_samples[s]:
                    Xparts.append(X)
                    Wparts.append(np.maximum(w, 1e-12))
                Xs = np.vstack(Xparts)
                ws = np.concatenate(Wparts)
                # Posterior-weighted resampling avoids requiring sample_weight support.
                n_draw = min(max(self.n_mix * 100, 500), max(500, len(Xs)))
                p = ws / ws.sum()
                ids = rng.choice(len(Xs), size=n_draw, replace=True, p=p)
                Xdraw = Xs[ids]
                k = min(self.n_mix, max(1, len(Xdraw)))
                g = GaussianMixture(
                    n_components=k,
                    covariance_type="diag",
                    max_iter=self.gmm_max_iter,
                    reg_covar=self.reg_covar,
                    random_state=self.random_state + it + s + 100,
                    init_params="kmeans",
                )
                g.fit(Xdraw)
                new_gmms.append(g)
            self.gmms_ = new_gmms

            if verbose:
                print(f"    HMM EM iter {it+1:02d}: log-likelihood={total_ll:.3f}")
            if total_ll <= prev_total + 1e-3:
                break
            prev_total = total_ll

        self.fitted_ = True
        return self
    def score(self, X):
        if not self.fitted_:
            raise RuntimeError("HMM is not fitted.")
        logB = self._log_emission(X)
        alpha = self._forward(logB)
        return float(logsumexp(alpha[-1]))

    def score_samples(self, X):
        if not self.fitted_:
            raise RuntimeError("HMM is not fitted.")
        logB = self._log_emission(X)
        return logsumexp(logB - np.log(self.n_states), axis=1)

    def state_log_likelihoods(self, X):
        return self._log_emission(X)

    def viterbi_states(self, X):
        logB = self._log_emission(X)
        T, S = logB.shape
        delta = np.full((T, S), -np.inf)
        psi = np.zeros((T, S), dtype=int)
        delta[0] = np.log(self.pi_) + logB[0]
        for t in range(1, T):
            vals = delta[t-1][:, None] + np.log(self.A_)
            psi[t] = np.argmax(vals, axis=0)
            delta[t] = logB[t] + vals[psi[t], np.arange(S)]
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states

    def save(self, path):
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        import joblib
        return joblib.load(path)
