from pathlib import Path

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import precision_score, recall_score


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_DIR / "Data" / "processed" / "one_hmm_data.npz"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"


# ============================================================
# HMM configuration
# ============================================================

N_STATES = 6
N_ITER = 100
TOL = 1e-3
RANDOM_STATE = 42


# ============================================================
# Load preprocessed data
# ============================================================

def load_data(data_file=DATA_FILE):
    data = np.load(data_file, allow_pickle=False)

    return {
        "X_train": data["X_train"],
        "y_train": data["y_train"],
        "train_lengths": data["train_lengths"],

        "X_test": data["X_test"],
        "y_test": data["y_test"],
        "test_lengths": data["test_lengths"],

        "activity_names": data["activity_names"],
    }


# ============================================================
# Create HMM
# ============================================================

def create_hmm():
    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="diag",
        n_iter=N_ITER,
        tol=TOL,
        random_state=RANDOM_STATE,
        verbose=True,
    )

    return model


# ============================================================
# Train HMM
# ============================================================

def train_hmm(X_train, train_lengths):
    model = create_hmm()

    print("\nTraining HMM...")
    print(f"Observations: {X_train.shape}")
    print(f"States: {N_STATES}")
    print(f"Sequences: {len(train_lengths)}")

    model.fit(
        X_train,
        lengths=train_lengths,
    )

    print("\nHMM training complete.")

    return model


# ============================================================
# Map latent HMM states to activities
# ============================================================

def create_state_activity_mapping(
    model,
    X_train,
    y_train,
    train_lengths,
):
    """
    HMM states are anonymous.

    Example:
        HMM state 0 might mostly represent Walking
        HMM state 1 might mostly represent Sitting
        etc.

    We decode the training data and find the most common
    true activity associated with each hidden state.

    This is done AFTER HMM training.
    """

    hidden_states = model.predict(
        X_train,
        lengths=train_lengths,
    )

    mapping = {}

    for state in range(N_STATES):

        state_mask = hidden_states == state

        if not np.any(state_mask):
            mapping[state] = -1
            continue

        state_labels = y_train[state_mask]

        # Most common activity label for this hidden state
        labels, counts = np.unique(
            state_labels,
            return_counts=True,
        )

        mapping[state] = int(labels[np.argmax(counts)])

    return mapping


# ============================================================
# Convert hidden states -> activity labels
# ============================================================

def states_to_activities(hidden_states, mapping):
    predicted_activities = np.array(
        [mapping[state] for state in hidden_states],
        dtype=np.int64,
    )

    return predicted_activities


# ============================================================
# Predict test sequence
# ============================================================

def predict_test(
    model,
    X_test,
    test_lengths,
    state_activity_mapping,
):
    """
    Decode the test observations using Viterbi.

    Then convert the six latent HMM states into the six
    activity labels.
    """

    hidden_states = model.predict(
        X_test,
        lengths=test_lengths,
    )

    predicted_activities = states_to_activities(
        hidden_states,
        state_activity_mapping,
    )

    return hidden_states, predicted_activities


# ============================================================
# Convert frame labels into segments
# ============================================================

def get_segments(activity_sequence):
    """
    Convert:

        1 1 1 1 2 2 2 4 4 1 1

    into:

        (1, 0, 4)
        (2, 4, 7)
        (4, 7, 9)
        (1, 9, 11)

    Format:
        (activity, start_frame, end_frame)
    """

    if len(activity_sequence) == 0:
        return []

    segments = []

    start = 0
    current_activity = activity_sequence[0]

    for i in range(1, len(activity_sequence)):

        if activity_sequence[i] != current_activity:

            segments.append(
                (
                    int(current_activity),
                    start,
                    i,
                )
            )

            start = i
            current_activity = activity_sequence[i]

    # Add final segment
    segments.append(
        (
            int(current_activity),
            start,
            len(activity_sequence),
        )
    )

    return segments


# ============================================================
# Evaluation
# ============================================================

def evaluate_predictions(y_true, y_pred):
    """
    Calculate activity recognition metrics.

    Accuracy/error is calculated per window.

    Precision and recall use macro averaging so that
    all six activities are treated equally.
    """

    accuracy = np.mean(y_true == y_pred)

    error_rate = 1.0 - accuracy

    precision = precision_score(
        y_true,
        y_pred,
        labels=np.arange(1, 7),
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        labels=np.arange(1, 7),
        average="macro",
        zero_division=0,
    )

    return {
        "error_rate": error_rate * 100,
        "precision": precision * 100,
        "recall": recall * 100,
    }


# ============================================================
# Save model
# ============================================================

def save_model(model, model_dir=MODEL_DIR):
    model_dir.mkdir(parents=True, exist_ok=True)

    model_file = model_dir / "one_hmm.pkl"

    import joblib

    joblib.dump(model, model_file)

    print(f"\nSaved model: {model_file}")

    return model_file


# ============================================================
# Save results
# ============================================================

def save_results(
    y_test,
    predicted_activities,
    hidden_states,
    state_activity_mapping,
    results,
    results_dir=RESULTS_DIR,
):
    results_dir.mkdir(parents=True, exist_ok=True)

    # Window-level predictions
    prediction_file = results_dir / "one_hmm_predictions.npz"

    np.savez_compressed(
        prediction_file,

        y_true=y_test,
        y_pred=predicted_activities,
        hidden_states=hidden_states,

        state_activity_mapping=np.array(
            [
                state_activity_mapping[state]
                for state in range(N_STATES)
            ],
            dtype=np.int64,
        ),

        error_rate=results["error_rate"],
        precision=results["precision"],
        recall=results["recall"],
    )

    # Human-readable summary
    summary_file = results_dir / "one_hmm_results.txt"

    with summary_file.open("w", encoding="utf-8") as f:

        f.write("ONE HMM RESULTS\n")
        f.write("================\n\n")

        f.write(f"Number of states: {N_STATES}\n")
        f.write(f"Training iterations: {N_ITER}\n\n")

        f.write("State -> Activity mapping\n")
        f.write("-------------------------\n")

        for state in range(N_STATES):
            activity = state_activity_mapping[state]
            f.write(
                f"State {state} -> Activity {activity}\n"
            )

        f.write("\nMetrics\n")
        f.write("-------\n")

        f.write(
            f"Error rate: {results['error_rate']:.4f}%\n"
        )

        f.write(
            f"Precision:  {results['precision']:.4f}%\n"
        )

        f.write(
            f"Recall:     {results['recall']:.4f}%\n"
        )

    print(f"Saved predictions: {prediction_file}")
    print(f"Saved results: {summary_file}")


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load preprocessing output
    # --------------------------------------------------------

    print("Loading preprocessed data...")

    data = load_data()

    X_train = data["X_train"]
    y_train = data["y_train"]
    train_lengths = data["train_lengths"]

    X_test = data["X_test"]
    y_test = data["y_test"]
    test_lengths = data["test_lengths"]

    activity_names = data["activity_names"]

    print(f"Train: {X_train.shape}")
    print(f"Test:  {X_test.shape}")

    # --------------------------------------------------------
    # 2. Train 6-state HMM
    # --------------------------------------------------------

    model = train_hmm(
        X_train,
        train_lengths,
    )

    # --------------------------------------------------------
    # 3. Interpret the learned hidden states
    # --------------------------------------------------------

    state_activity_mapping = create_state_activity_mapping(
        model,
        X_train,
        y_train,
        train_lengths,
    )

    print("\nHidden state -> activity mapping:")

    for state in range(N_STATES):

        activity_id = state_activity_mapping[state]

        if activity_id == -1:
            name = "UNUSED"
        else:
            name = activity_names[activity_id - 1]

        print(
            f"State {state} -> "
            f"Activity {activity_id} ({name})"
        )

    # --------------------------------------------------------
    # 4. Predict test activity sequence
    # --------------------------------------------------------

    hidden_states, predicted_activities = predict_test(
        model,
        X_test,
        test_lengths,
        state_activity_mapping,
    )

    # --------------------------------------------------------
    # 5. Generate segmentation
    # --------------------------------------------------------

    segments = get_segments(predicted_activities)

    print("\nFirst 10 predicted segments:")

    for segment in segments[:10]:
        activity, start, end = segment

        print(
            f"Activity {activity}: "
            f"frames {start} -> {end}"
        )

    # --------------------------------------------------------
    # 6. Evaluate
    # --------------------------------------------------------

    results = evaluate_predictions(
        y_test,
        predicted_activities,
    )

    print("\n==============================")
    print("RESULTS")
    print("==============================")

    print(
        f"Error rate: {results['error_rate']:.4f}%"
    )

    print(
        f"Precision:  {results['precision']:.4f}%"
    )

    print(
        f"Recall:     {results['recall']:.4f}%"
    )

    # --------------------------------------------------------
    # 7. Save
    # --------------------------------------------------------

    save_model(model)

    save_results(
        y_test,
        predicted_activities,
        hidden_states,
        state_activity_mapping,
        results,
    )


if __name__ == "__main__":
    main()