import numpy as np
from collections import Counter, defaultdict

class ActivitySequenceModel:
    """
    Three modes:
      - none: raw per-frame HMM likelihood classification
      - bigram: learned P(activity_t | activity_{t-1})
      - trigram: learned P(activity_t | activity_{t-2}, activity_{t-1})
      - graph: same idea as a first-order graph with transition probabilities.

    The paper evaluates graph, 2-sequence and 3-sequence statistical models.
    This implementation provides all three in an inspectable form.
    """

    def __init__(self, n_activities=6, smoothing=1.0):
        self.n = n_activities
        self.smoothing = smoothing
        self.bigram = np.ones((n_activities, n_activities)) * smoothing
        self.trigram = np.ones((n_activities, n_activities, n_activities)) * smoothing
        self.start = np.ones(n_activities) * smoothing

    def fit(self, label_sequences):
        self.bigram[:] = self.smoothing
        self.trigram[:] = self.smoothing
        self.start[:] = self.smoothing
        for seq in label_sequences:
            z = np.asarray(seq, dtype=int) - 1
            if len(z) == 0:
                continue
            self.start[z[0]] += 1
            for i in range(1, len(z)):
                self.bigram[z[i-1], z[i]] += 1
            for i in range(2, len(z)):
                self.trigram[z[i-2], z[i-1], z[i]] += 1
        self.bigram /= self.bigram.sum(axis=1, keepdims=True)
        self.trigram /= self.trigram.sum(axis=2, keepdims=True)
        self.start /= self.start.sum()
        return self

    def log_prior(self, prev=None, prev2=None, curr=None, mode="bigram"):
        if curr is None:
            raise ValueError
        c = curr - 1
        if mode == "trigram" and prev2 is not None and prev is not None:
            return np.log(self.trigram[prev2-1, prev-1, c])
        if prev is not None:
            return np.log(self.bigram[prev-1, c])
        return np.log(self.start[c])

    def smooth_scores(self, log_scores, mode="bigram", beam=64):
        """
        Decode a sequence of per-frame log scores.
        log_scores[t,a] = log P(observation_t | activity=a).
        """
        T, A = log_scores.shape
        if mode == "none":
            return np.argmax(log_scores, axis=1) + 1

        if mode in ("bigram", "graph"):
            dp = np.full((T, A), -np.inf)
            back = np.zeros((T, A), dtype=int)
            dp[0] = log_scores[0] + np.log(self.start)
            for t in range(1, T):
                vals = dp[t-1][:, None] + np.log(self.bigram)
                back[t] = np.argmax(vals, axis=0)
                dp[t] = log_scores[t] + vals[back[t], np.arange(A)]
            z = np.zeros(T, dtype=int)
            z[-1] = np.argmax(dp[-1])
            for t in range(T-2, -1, -1):
                z[t] = back[t+1, z[t+1]]
            return z + 1

        if mode == "trigram":
            # State is the ordered pair (a_{t-1}, a_t).
            pairs = [(i, j) for i in range(A) for j in range(A)]
            P = len(pairs)
            dp = np.full((T, P), -np.inf)
            back = np.full((T, P), -1, dtype=int)
            if T == 1:
                return np.array([np.argmax(log_scores[0]) + 1])
            for p, (i, j) in enumerate(pairs):
                dp[1, p] = (
                    log_scores[0, i] + log_scores[1, j]
                    + np.log(self.start[i])
                    + np.log(self.bigram[i, j])
                )
            for t in range(2, T):
                for p, (i, j) in enumerate(pairs):
                    prev_vals = np.array([
                        dp[t-1, q] + np.log(self.trigram[k, i, j])
                        for q, (k, jj) in enumerate(pairs) if jj == i
                    ])
                    qs = [q for q, (k, jj) in enumerate(pairs) if jj == i]
                    if len(qs):
                        qbest = qs[int(np.argmax(prev_vals))]
                        dp[t, p] = log_scores[t, j] + prev_vals.max()
                        back[t, p] = qbest
            z = np.zeros(T, dtype=int)
            p = int(np.argmax(dp[T-1]))
            z[-1] = pairs[p][1]
            z[-2] = pairs[p][0]
            for t in range(T-1, 1, -1):
                p = back[t, p]
                if p < 0:
                    break
                z[t-2] = pairs[p][0]
            return z + 1

        raise ValueError(f"Unknown ASM mode: {mode}")

def labels_to_segments(labels, activity_names):
    labels = list(labels)
    segments = []
    if not labels:
        return segments
    start = 0
    for i in range(1, len(labels)+1):
        if i == len(labels) or labels[i] != labels[start]:
            lab = int(labels[start])
            segments.append({
                "start_frame": start,
                "end_frame": i-1,
                "start_time_sec": start * 1.28,
                "end_time_sec": (i) * 1.28,
                "activity_id": lab,
                "activity": activity_names[lab],
            })
            start = i
    return segments
