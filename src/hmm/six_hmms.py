import numpy as np
from hmmlearn.hmm import GaussianHMM

from .preprocessing import ACTIVITIES, split_by_activity


# ---------------------------------------------------------
# HMM configuration
# ---------------------------------------------------------

N_STATES = 8
RANDOM_STATE = 42
N_ITER = 100
TOL = 1e-3


# ---------------------------------------------------------
# Create one normal HMM
# ---------------------------------------------------------

def create_hmm():
    """
    Create a standard Gaussian HMM.

    Each hidden state has one Gaussian emission
    distribution.
    """

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="diag",
        n_iter=N_ITER,
        tol=TOL,
        random_state=RANDOM_STATE,
        verbose=False
    )

    return model


# ---------------------------------------------------------
# Train one activity-specific HMM
# ---------------------------------------------------------

def train_activity_hmm(X_activity):
    """
    Train one HMM using data belonging to one activity.
    """

    if len(X_activity) == 0:
        raise ValueError(
            "No training data available for this activity."
        )

    model = create_hmm()

    model.fit(X_activity)

    return model


# ---------------------------------------------------------
# Train all six HMMs
# ---------------------------------------------------------

def train_six_hmms(X_train, y_train):
    """
    Train one separate HMM for each of the six activities.
    """

    activity_data = split_by_activity(
        X_train,
        y_train
    )

    models = {}

    for activity_name in ACTIVITIES.values():

        print(
            f"Training HMM for: {activity_name}"
        )

        X_activity = activity_data[activity_name]

        print(
            f"  Training samples: {X_activity.shape}"
        )

        model = train_activity_hmm(
            X_activity
        )

        models[activity_name] = model

        print(
            f"Finished: {activity_name}"
        )

    return models


# ---------------------------------------------------------
# Calculate likelihood under every HMM
# ---------------------------------------------------------

def get_activity_likelihoods(models, X):
    """
    Calculate the log-likelihood of X under
    each activity-specific HMM.
    """

    likelihoods = {}

    for activity_name, model in models.items():

        try:

            score = model.score(X)

            if np.isfinite(score):
                likelihoods[activity_name] = score
            else:
                likelihoods[activity_name] = float("-inf")

        except (
            ValueError,
            np.linalg.LinAlgError
        ):

            likelihoods[activity_name] = float("-inf")

    return likelihoods


# ---------------------------------------------------------
# Predict activity
# ---------------------------------------------------------

def predict_activity(models, X):
    """
    Predict the activity whose HMM gives
    the highest log-likelihood.
    """

    likelihoods = get_activity_likelihoods(
        models,
        X
    )

    predicted_activity = max(
        likelihoods,
        key=likelihoods.get
    )

    return predicted_activity, likelihoods