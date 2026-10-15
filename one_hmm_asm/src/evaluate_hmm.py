from pathlib import Path
import argparse

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .train_hmm import (
    DATA_FILE,
    MODEL_FILE,
    N_MIXTURES,
    N_STATES,
    PROJECT_DIR,
    load_preprocessed_data,
)


RESULTS_DIR = PROJECT_DIR / "results"


def load_model(model_path=MODEL_FILE):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def create_state_activity_mapping(model, X_train, y_train, train_lengths):
    y_train = np.asarray(y_train, dtype=np.int64)
    hidden_states = model.predict(X_train, lengths=np.asarray(train_lengths).tolist())
    if len(hidden_states) != len(y_train):
        raise ValueError("Training labels and decoded states have different lengths.")
    activity_labels = np.unique(y_train)
    if len(activity_labels) != model.n_components:
        raise ValueError("One-to-one state mapping requires one activity per HMM state.")

    counts = np.zeros((model.n_components, len(activity_labels)), dtype=np.int64)
    np.add.at(counts, (hidden_states, np.searchsorted(activity_labels, y_train)), 1)
    states, activities = linear_sum_assignment(-counts)
    return {
        int(state): int(activity_labels[activity])
        for state, activity in zip(states, activities)
    }


def predict_hmm(model, X_test, test_lengths, state_activity_mapping):
    hidden_states = model.predict(X_test, lengths=np.asarray(test_lengths).tolist())
    predicted_activities = np.asarray(
        [state_activity_mapping[int(state)] for state in hidden_states],
        dtype=np.int64,
    )
    return hidden_states, predicted_activities


def evaluate_predictions(y_true, y_pred, activity_labels=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape or len(y_true) == 0:
        raise ValueError("Expected equal, non-empty one-dimensional label arrays.")

    if activity_labels is None:
        activity_labels = np.unique(y_true)
    labels = np.asarray(activity_labels)
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "accuracy": accuracy * 100,
        "error_rate": (1 - accuracy) * 100,
        "precision": precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ) * 100,
        "recall": recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ) * 100,
        "f1": f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ) * 100,
    }


def get_segments(activity_sequence, sequence_lengths=None):
    activity_sequence = np.asarray(activity_sequence)
    if activity_sequence.ndim != 1:
        raise ValueError("Activity sequence must be one-dimensional.")
    if sequence_lengths is None:
        if len(activity_sequence) == 0:
            return []
        sequence_lengths = [len(activity_sequence)]
    sequence_lengths = np.asarray(sequence_lengths, dtype=np.int64)
    if (
        sequence_lengths.ndim != 1
        or np.any(sequence_lengths <= 0)
        or int(sequence_lengths.sum()) != len(activity_sequence)
    ):
        raise ValueError("Sequence lengths do not match the activity sequence.")

    segments = []
    offset = 0
    for length in sequence_lengths:
        end = offset + int(length)
        sequence = activity_sequence[offset:end]
        start = 0
        for index in range(1, len(sequence)):
            if sequence[index] != sequence[start]:
                segments.append((int(sequence[start]), offset + start, offset + index))
                start = index
        segments.append((int(sequence[start]), offset + start, end))
        offset = end
    return segments


def save_results(
    y_true,
    hmm_predictions,
    hidden_states,
    state_activity_mapping,
    hmm_metrics,
    results_dir=RESULTS_DIR,
    test_lengths=None,
    asm_predictions=None,
    asm_metrics=None,
    hmm_segments=None,
    asm_segments=None,
    activity_labels=None,
    graph=None,
):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    prediction_file = results_dir / "one_hmm_predictions.npz"
    summary_file = results_dir / "one_hmm_results.txt"

    arrays = {
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(hmm_predictions),
        "hidden_states": np.asarray(hidden_states),
        "state_activity_mapping": np.asarray(
            [state_activity_mapping[state] for state in range(len(state_activity_mapping))],
            dtype=np.int64,
        ),
        "accuracy": hmm_metrics["accuracy"],
        "error_rate": hmm_metrics["error_rate"],
        "precision": hmm_metrics["precision"],
        "recall": hmm_metrics["recall"],
        "f1": hmm_metrics["f1"],
    }
    if activity_labels is None:
        activity_labels = np.arange(1, N_STATES + 1)
    activity_labels = np.asarray(activity_labels, dtype=np.int64)
    arrays["activity_labels"] = activity_labels
    arrays["hmm_confusion_matrix"] = confusion_matrix(
        y_true, hmm_predictions, labels=activity_labels
    )
    if test_lengths is not None:
        arrays["test_lengths"] = np.asarray(test_lengths, dtype=np.int64)
    if hmm_segments is not None:
        arrays["hmm_segments"] = np.asarray(hmm_segments, dtype=np.int64).reshape(-1, 3)
    if asm_predictions is not None:
        arrays["asm_y_pred"] = np.asarray(asm_predictions, dtype=np.int64)
    if asm_metrics is not None:
        arrays.update(
            {
                f"asm_{name}": float(value)
                for name, value in asm_metrics.items()
            }
        )
        arrays["asm_confusion_matrix"] = confusion_matrix(
            y_true, asm_predictions, labels=activity_labels
        )
    if asm_segments is not None:
        arrays["asm_segments"] = np.asarray(asm_segments, dtype=np.int64).reshape(-1, 3)

    np.savez_compressed(prediction_file, **arrays)
    with summary_file.open("w", encoding="utf-8") as summary:
        summary.write("ONE HMM RESULTS\n")
        summary.write("================\n\n")
        summary.write(
            f"Model: hmmlearn GMMHMM, {N_STATES} states, "
            f"{N_MIXTURES} diagonal Gaussian mixtures per state\n\n"
        )
        summary.write("State -> Activity mapping\n")
        summary.write("-------------------------\n")
        for state, activity in state_activity_mapping.items():
            summary.write(f"State {state} -> Activity {activity}\n")

        for title, metrics in (("HMM only", hmm_metrics), ("HMM + graph ASM", asm_metrics)):
            if metrics is None:
                continue
            summary.write(f"\n{title}\n{'-' * len(title)}\n")
            summary.write("Frame-level classification metrics (percent)\n")
            summary.write(f"Accuracy:        {metrics['accuracy']:.4f}%\n")
            summary.write(f"Error rate:      {metrics['error_rate']:.4f}%\n")
            summary.write(f"Macro precision: {metrics['precision']:.4f}%\n")
            summary.write(f"Macro recall:    {metrics['recall']:.4f}%\n")
            summary.write(f"Macro F1:        {metrics['f1']:.4f}%\n")
        if hmm_segments is not None:
            summary.write(
                f"\nSegmentation output count (HMM only): {len(hmm_segments)}\n"
            )
        if asm_segments is not None:
            summary.write(
                f"Segmentation output count (HMM + graph ASM): {len(asm_segments)}\n"
            )
        if graph is not None:
            summary.write("\nConfigured graph edges\n")
            summary.write("----------------------\n")
            for activity, destinations in zip(graph.labels, graph.allowed_transitions):
                summary.write(f"{activity}: {', '.join(map(str, destinations))}\n")

    print(f"Saved predictions: {prediction_file}")
    print(f"Saved results: {summary_file}")
    return prediction_file, summary_file


def evaluate_model(model, data):
    mapping = create_state_activity_mapping(
        model, data["X_train"], data["y_train"], data["train_lengths"]
    )
    hidden_states, predictions = predict_hmm(
        model, data["X_test"], data["test_lengths"], mapping
    )
    labels = data.get("activity_labels", np.arange(1, 7))
    metrics = evaluate_predictions(data["y_test"], predictions, labels)
    segments = get_segments(predictions, data["test_lengths"])
    return mapping, hidden_states, predictions, metrics, segments


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a saved one-HMM model.")
    parser.add_argument("--data-file", type=Path, default=DATA_FILE)
    parser.add_argument("--model-path", type=Path, default=MODEL_FILE)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    data = load_preprocessed_data(args.data_file)
    model = load_model(args.model_path)
    mapping, states, predictions, metrics, segments = evaluate_model(model, data)
    print("HMM-only metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}%")
    print(f"HMM-only segments: {len(segments)}")
    save_results(
        data["y_test"],
        predictions,
        states,
        mapping,
        metrics,
        results_dir=args.results_dir,
        test_lengths=data["test_lengths"],
        hmm_segments=segments,
    )


if __name__ == "__main__":
    main()
