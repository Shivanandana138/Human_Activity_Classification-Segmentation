import numpy as np
from collections import Counter, defaultdict

class ActivitySequenceModel:
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

