import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from .preprocessing import (
    ACTIVITIES,
    split_by_activity,
    create_activity_sequences,
)


N_STATES = 8
RANDOM_STATE = 42
N_ITER = 100
TOL = 1e-3
MIN_COVAR = 1e-2


def create_hmm():
    """Create a standard Gaussian HMM."""

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

    return model


def train_activity_hmm(X_activity, sequence_lengths):
    """
    Train one activity-specific HMM using multiple
    independent sequences.
    """

    if X_activity is None or len(X_activity) == 0:
        raise ValueError(
            "No training data available."
        )

    X_activity = np.asarray(
        X_activity,
        dtype=np.float64
    )

    if not np.all(np.isfinite(X_activity)):
        raise ValueError(
            "Training data contains NaN or infinity."
        )

    if sum(sequence_lengths) != len(X_activity):
        raise ValueError(
            "Sequence lengths do not match X_activity."
        )

    # -----------------------------------------------------
    # Scale features
    # -----------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_activity
    )

    X_scaled = np.clip(
        X_scaled,
        -10,
        10
    )

    # -----------------------------------------------------
    # Create model
    # -----------------------------------------------------

    model = create_hmm()

    # -----------------------------------------------------
    # Train using independent sequences
    # -----------------------------------------------------

    model.fit(
        X_scaled,
        lengths=sequence_lengths
    )

    # -----------------------------------------------------
    # Validate parameters
    # -----------------------------------------------------

    if not np.all(np.isfinite(model.startprob_)):
        raise ValueError(
            "Invalid start probabilities."
        )

    if not np.all(np.isfinite(model.transmat_)):
        raise ValueError(
            "Invalid transition probabilities."
        )

    if not np.all(np.isfinite(model.means_)):
        raise ValueError(
            "Invalid means."
        )

    if not np.all(np.isfinite(model.covars_)):
        raise ValueError(
            "Invalid covariance values."
        )

    # -----------------------------------------------------
    # Normalize start probabilities
    # -----------------------------------------------------

    start_sum = model.startprob_.sum()

    if start_sum <= 0:
        raise ValueError(
            "Invalid start probability sum."
        )

    model.startprob_ /= start_sum

    # -----------------------------------------------------
    # Repair invalid transition rows
    # -----------------------------------------------------

    row_sums = model.transmat_.sum(axis=1)

    for i in range(N_STATES):

        if (
            row_sums[i] <= 0
            or not np.isfinite(row_sums[i])
        ):

            model.transmat_[i] = np.full(
                N_STATES,
                0.05 / (N_STATES - 1)
            )

            model.transmat_[i, i] = 0.95

    model.transmat_ /= (
        model.transmat_.sum(
            axis=1,
            keepdims=True
        )
    )

    # Store scaler and sequence information
    model.activity_scaler = scaler

    return model


def train_six_hmms(X_train, y_train, subject_train):
    """
    Train six activity-specific HMMs using
    subject/activity-based sequences.
    """

    sequences = create_activity_sequences(
        X_train,
        y_train,
        subject_train
    )

    models = {}

    for activity_name in ACTIVITIES.values():

        activity_label = next(
            label
            for label, name in ACTIVITIES.items()
            if name == activity_name
        )

        # -------------------------------------------------
        # Select sequences belonging to this activity
        # -------------------------------------------------

        activity_sequences = [
            seq
            for seq in sequences
            if seq["activity"] == activity_label
        ]

        if not activity_sequences:
            raise ValueError(
                f"No sequences found for {activity_name}."
            )

        # -------------------------------------------------
        # Combine sequences
        # -------------------------------------------------

        X_parts = [
            seq["X"]
            for seq in activity_sequences
        ]

        X_activity = np.concatenate(
            X_parts,
            axis=0
        )

        sequence_lengths = [
            len(seq["X"])
            for seq in activity_sequences
        ]

        print(
            f"  Number of sequences: "
            f"{len(activity_sequences)}"
        )

        print(
            f"  Combined samples: "
            f"{X_activity.shape}"
        )

        # -------------------------------------------------
        # Train
        # -------------------------------------------------

        print(
            f"Training HMM for: {activity_name}"
        )

        model = train_activity_hmm(
            X_activity,
            sequence_lengths
        )

        models[activity_name] = model

        print(
            f"Finished: {activity_name}"
        )

    return models


def get_activity_likelihoods(models, X):
    """
    Calculate log-likelihood for each activity HMM.
    """

    X = np.asarray(
        X,
        dtype=np.float64
    )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "Input contains NaN or infinity."
        )

    likelihoods = {}

    for activity_name, model in models.items():

        try:

            if hasattr(model, "activity_scaler"):

                X_scaled = (
                    model.activity_scaler
                    .transform(X)
                )

                X_scaled = np.clip(
                    X_scaled,
                    -10,
                    10
                )

            else:
                X_scaled = X

            score = model.score(
                X_scaled
            )

            if np.isfinite(score):
                likelihoods[activity_name] = score
            else:
                likelihoods[activity_name] = float(
                    "-inf"
                )

        except (
            ValueError,
            np.linalg.LinAlgError,
            FloatingPointError,
        ):

            likelihoods[activity_name] = float(
                "-inf"
            )

    return likelihoods


def predict_activity(models, X):
    """
    Predict activity using the HMM with
    highest log-likelihood.
    """

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

    return predicted_activity, likelihoodss