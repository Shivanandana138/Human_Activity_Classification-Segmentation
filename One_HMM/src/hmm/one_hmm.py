import numpy as np
from hmmlearn.hmm import GaussianHMM


N_STATES = 6
N_ITER = 100
TOL = 1e-3
RANDOM_STATE = 42


def create_hmm():
    """
    Create the single Gaussian HMM.

    One model is used for all six activities.
    """

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="diag",
        n_iter=N_ITER,
        tol=TOL,
        random_state=RANDOM_STATE,
        verbose=False,
    )

    return model


def train_one_hmm(X_train):
    """
    Train one Gaussian HMM using all training observations.

    Parameters
    ----------
    X_train : ndarray
        Shape: (number_of_samples, 561)

    Returns
    -------
    model : GaussianHMM
        Trained single HMM.
    """

    if X_train is None or len(X_train) == 0:
        raise ValueError("Training data is empty.")

    model = create_hmm()

    model.fit(X_train)

    return model


def predict_states(model, X):
    """
    Predict hidden states for observations.

    Returns
    -------
    states : ndarray
        Hidden state assigned to each observation.
    """

    if X is None or len(X) == 0:
        raise ValueError("Input data is empty.")

    return model.predict(X)


def score_sequence(model, X):
    """
    Calculate log likelihood of a sequence.
    """

    if X is None or len(X) == 0:
        raise ValueError("Input sequence is empty.")

    return model.score(X)