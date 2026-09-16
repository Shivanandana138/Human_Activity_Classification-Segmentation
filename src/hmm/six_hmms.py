import numpy as np
from hmmlearn.hmm import GMMHMM

from .preprocessing import ACTIVITIES, split_by_activity


# ---------------------------------------------------------
# Base-paper configuration
# ---------------------------------------------------------

N_STATES = 8
N_MIXTURES = 10

RANDOM_STATE = 42
N_ITER = 100
TOL = 1e-3


# ---------------------------------------------------------
# Create one GMM-HMM
# ---------------------------------------------------------

def create_hmm():
    """
    Create a single activity-specific GMM-HMM.

    Configuration:
        8 hidden states
        10 Gaussian mixtures per state
    """

    model = GMMHMM(
        n_components=N_STATES,
        n_mix=N_MIXTURES,
        covariance_type="diag",
        n_iter=N_ITER,
        tol=TOL,
        random_state=RANDOM_STATE,
        verbose=False
    )

    return model


# ---------------------------------------------------------
# Train one activity HMM
# ---------------------------------------------------------

def train_activity_hmm(X_activity):
    """
    Train one HMM for a single activity.

    Parameters
    ----------
    X_activity : numpy.ndarray
        Shape: (number_of_observations, 561)

    Returns
    -------
    model : GMMHMM
        Trained activity-specific HMM.
    """

    if len(X_activity) == 0:
        raise ValueError("No training data available for this activity.")

    model = create_hmm()

    model.fit(X_activity)

    return model


# ---------------------------------------------------------
# Train all six HMMs
# ---------------------------------------------------------

def train_six_hmms(X_train, y_train):
    """
    Train six separate HMMs, one for each activity.

    Returns
    -------
    models : dict
        Dictionary containing six trained HMMs.
    """

    activity_data = split_by_activity(X_train, y_train)

    models = {}

    for activity_name in ACTIVITIES.values():

        print(f"Training HMM for: {activity_name}")

        X_activity = activity_data[activity_name]

        model = train_activity_hmm(X_activity)

        models[activity_name] = model

        print(f"Finished: {activity_name}")

    return models


# ---------------------------------------------------------
# Calculate likelihood under all six HMMs
# ---------------------------------------------------------

def get_activity_likelihoods(models, X):
    """
    Calculate the log-likelihood of observations under
    each of the six activity-specific HMMs.

    Parameters
    ----------
    models : dict
        Six trained HMMs.

    X : numpy.ndarray
        Observation sequence.
        Shape: (number_of_observations, 561)

    Returns
    -------
    likelihoods : dict
        Log-likelihood for each activity.
    """

    likelihoods = {}

    for activity_name, model in models.items():

        try:
            log_likelihood = model.score(X)

        except ValueError:
            log_likelihood = float("-inf")

        likelihoods[activity_name] = log_likelihood

    return likelihoods


# ---------------------------------------------------------
# Predict activity
# ---------------------------------------------------------

def predict_activity(models, X):
    """
    Predict the activity with the highest HMM likelihood.

    Note:
        This is the basic six-HMM decision.
        ASM is NOT included here.
    """

    likelihoods = get_activity_likelihoods(models, X)

    predicted_activity = max(
        likelihoods,
        key=likelihoods.get
    )

    return predicted_activity, likelihoods