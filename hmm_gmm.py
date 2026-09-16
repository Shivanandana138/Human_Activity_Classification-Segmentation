import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp

class GMMHMM:
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

    def _initialize(self, sequences):
        rng = np.random.default_rng(self.random_state)
        allX = np.vstack(sequences)
        self.pi_ = np.full(self.n_states, self.smoothing)
        self.pi_[0] = 1.0
        self.pi_ /= self.pi_.sum()
        A = np.full((self.n_states, self.n_states), self.smoothing)
        for i in range(self.n_states):
            A[i, i] = 0.65
            if i < self.n_states - 1:
                A[i, i + 1] = 0.35
        A[-1, :] = self.smoothing
        A[-1, -1] = 1.0
        A /= A.sum(axis=1, keepdims=True)
        self.A_ = A

