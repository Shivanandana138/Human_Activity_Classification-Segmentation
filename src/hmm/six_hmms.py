import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from .preprocessing import ACTIVITIES, split_by_activity


# ---------------------------------------------------------
# HMM configuration
# ---------------------------------------------------------

N_STATES = 8
RANDOM_STATE = 42
N_ITER = 100
TOL = 1e-3
MIN_COVAR = 1e-2


# ---------------------------------------------------------
# Feature scaler
# ---------------------------------------------------------

def create_scaler(X):
    """
    Create a StandardScaler using the training data.
    """
    scaler = StandardScaler()

    X = np.asarray(X, dtype=np.float64)

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "Training data contains NaN or infinite values."
        )

    scaler.fit(X)

    return scaler


# ---------------------------------------------------------
# Create HMM
# ---------------------------------------------------------

def create_hmm():
    """
    Create a standard Gaussian HMM.

    One Gaussian emission distribution is used
    for each hidden state.
    """

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="diag",
        n_iter=N_ITER,
        tol=TOL,
        min_covar=MIN_COVAR,
        random_state=RANDOM_STATE,
        init_params="mc",
        params="stmc",
        verbose=False,
    )

    # Uniform initial state probabilities
    model.startprob_ = np.full(
        N_STATES,
        1.0 / N_STATES
    )

    # Initialize transition matrix
    transmat = np.full(
        (N_STATES, N_STATES),
        0.05 / (N_STATES - 1)
    )

    np.fill_diagonal(
        transmat,
        0.95
    )

    model.transmat_ = transmat

    return model


# ---------------------------------------------------------
# Train one activity HMM
# ---------------------------------------------------------

def train_activity_hmm(X_activity):
    """
    Train one Gaussian HMM for one activity.
    """

    if X_activity is None or len(X_activity) == 0:
        raise ValueError(
            "No training data available for this activity."
        )

    X_activity = np.asarray(
        X_activity,
        dtype=np.float64
    )

    if not np.all(np.isfinite(X_activity)):
        raise ValueError(
            "Training data contains NaN or infinite values."
        )

    # -----------------------------------------------------
    # Standardize features
    # -----------------------------------------------------

    scaler = create_scaler(X_activity)

    X_scaled = scaler.transform(X_activity)

    # Avoid extremely large values
    X_scaled = np.clip(
        X_scaled,
        -10,
        10
    )

    # -----------------------------------------------------
    # Create HMM
    # -----------------------------------------------------

    model = create_hmm()

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    model.fit(X_scaled)

    # -----------------------------------------------------
    # Check parameters
    # -----------------------------------------------------

    if not np.all(np.isfinite(model.startprob_)):
        raise ValueError(
            "HMM training produced invalid start probabilities."
        )

    if not np.all(np.isfinite(model.transmat_)):
        raise ValueError(
            "HMM training produced invalid transition probabilities."
        )

    if not np.all(np.isfinite(model.means_)):
        raise ValueError(
            "HMM training produced invalid means."
        )

    if not np.all(np.isfinite(model.covars_)):
        raise ValueError(
            "HMM training produced invalid covariance values."
        )

    # -----------------------------------------------------
    # Normalize start probabilities
    # -----------------------------------------------------

    start_sum = model.startprob_.sum()

    if start_sum <= 0 or not np.isfinite(start_sum):
        raise ValueError(
            "Invalid start probability sum after training."
        )

    model.startprob_ /= start_sum

    # -----------------------------------------------------
    # Fix transition matrix
    # -----------------------------------------------------

    row_sums = model.transmat_.sum(axis=1)

    for i in range(N_STATES):

        if (
            row_sums[i] <= 0
            or not np.isfinite(row_sums[i])
        ):
            # Fall back to a stable transition row
            model.transmat_[i] = np.full(
                N_STATES,
                0.05 / (N_STATES - 1)
            )

            model.transmat_[i, i] = 0.95

    # Normalize rows
    model.transmat_ /= (
        model.transmat_.sum(
            axis=1,
            keepdims=True
        )
    )

    # -----------------------------------------------------
    # Store scaler inside model
    # -----------------------------------------------------

    model.activity_scaler = scaler

    return model


# ---------------------------------------------------------
# Train all six HMMs
# ---------------------------------------------------------

def train_six_hmms(X_train, y_train):

    activity_data = split_by_activity(
        X_train,
        y_train
    )

    models = {}

    for activity_name in ACTIVITIES.values():

        print(
            f"Training HMM for: {activity_name}"
        )

        X_activity = activity_data[
            activity_name
        ]

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
# Calculate activity likelihoods
# ---------------------------------------------------------

def get_activity_likelihoods(models, X):

    X = np.asarray(
        X,
        dtype=np.float64
    )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "Input contains NaN or infinite values."
        )

    likelihoods = {}

    for activity_name, model in models.items():

        try:

            # Apply the same scaler used during training
            if hasattr(model, "activity_scaler"):
                X_scaled = model.activity_scaler.transform(X)

                X_scaled = np.clip(
                    X_scaled,
                    -10,
                    10
                )
            else:
                X_scaled = X

            score = model.score(X_scaled)

            if np.isfinite(score):
                likelihoods[activity_name] = score
            else:
                likelihoods[activity_name] = float("-inf")

        except (
            ValueError,
            np.linalg.LinAlgError,
            FloatingPointError
        ):

            likelihoods[activity_name] = float("-inf")

    return likelihoods


# ---------------------------------------------------------
# Predict activity
# ---------------------------------------------------------

def predict_activity(models, X):

    likelihoods = get_activity_likelihoods(
        models,
        X
    )

    if all(
        score == float("-inf")
        for score in likelihoods.values()
    ):
        raise ValueError(
            "All HMMs returned invalid likelihoods."
        )

    predicted_activity = max(
        likelihoods,
        key=likelihoods.get
    )

    return predicted_activity, likelihoods