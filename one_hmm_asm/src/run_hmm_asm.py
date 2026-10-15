from pathlib import Path
import argparse
from datetime import datetime
import json

import hmmlearn
from .evaluate_hmm import (
    RESULTS_DIR,
    evaluate_model,
    evaluate_predictions,
    get_segments,
    save_results,
)
from .graph_asm import (
    ACTIVITY_LABELS,
    NON_SELF_TRANSITIONS,
    activity_log_evidence,
    build_activity_graph,
    decode_activity_sequence,
)
from .train_hmm import (
    DATA_FILE,
    N_ITER,
    N_MIXTURES,
    N_STATES,
    PROJECT_DIR,
    RANDOM_STATE,
    TOL,
    load_preprocessed_data,
    save_model,
    train_hmm,
)


def _new_results_dir(results_dir=None):
    if results_dir is None:
        results_dir = RESULTS_DIR / datetime.now().strftime("hmm_asm_%Y%m%d_%H%M%S")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=False)
    return results_dir


def run_pipeline(data_file=DATA_FILE, results_dir=None):
    results_dir = _new_results_dir(results_dir)
    model_path = results_dir / "model" / "one_hmm.pkl"
    data = load_preprocessed_data(data_file)
    model = train_hmm(data["X_train"], data["train_lengths"])
    save_model(model, model_path)

    mapping, hidden_states, hmm_predictions, hmm_metrics, hmm_segments = evaluate_model(
        model, data
    )
    activity_labels = tuple(int(label) for label in data["activity_labels"])
    if activity_labels != ACTIVITY_LABELS:
        raise ValueError(f"Expected activity IDs {ACTIVITY_LABELS}, got {activity_labels}.")
    graph = build_activity_graph(activity_labels)
    log_evidence = activity_log_evidence(
        model,
        data["X_test"],
        data["test_lengths"],
        mapping,
        graph,
    )
    asm_predictions = decode_activity_sequence(
        log_evidence,
        graph,
        data["test_lengths"],
    )
    asm_metrics = evaluate_predictions(
        data["y_test"], asm_predictions, activity_labels
    )
    asm_segments = get_segments(asm_predictions, data["test_lengths"])

    save_results(
        data["y_test"],
        hmm_predictions,
        hidden_states,
        mapping,
        hmm_metrics,
        results_dir=results_dir,
        test_lengths=data["test_lengths"],
        asm_predictions=asm_predictions,
        asm_metrics=asm_metrics,
        hmm_segments=hmm_segments,
        asm_segments=asm_segments,
        activity_labels=activity_labels,
        graph=graph,
    )
    config = {
        "model": "hmmlearn.hmm.GMMHMM",
        "hmmlearn_version": hmmlearn.__version__,
        "n_components": N_STATES,
        "n_mix": N_MIXTURES,
        "covariance_type": "diag",
        "weights_prior": 1.01,
        "means_weight": 1e-3,
        "covars_prior": 0.0,
        "covars_weight": 1.0,
        "n_iter": N_ITER,
        "tol": TOL,
        "random_state": RANDOM_STATE,
        "state_mapping_method": (
            "One-to-one assignment maximizing state/activity overlap on decoded "
            "training data only."
        ),
        "activity_labels": {
            str(label): str(name)
            for label, name in zip(activity_labels, data["activity_names"])
        },
        "state_to_activity_mapping": {
            str(state): int(activity)
            for state, activity in mapping.items()
        },
        "graph_self_transitions": True,
        "graph_non_self_transitions": {
            str(activity): list(destinations)
            for activity, destinations in NON_SELF_TRANSITIONS.items()
        },
        "graph_assumption": (
            "Project-specific application graph; not claimed to reproduce the "
            "paper's unavailable edge list."
        ),
        "graph_decoding": (
            "Viterbi maximizes summed log HMM-mapped activity posterior under "
            "hard graph constraints; no learned transition probabilities."
        ),
        "classification_metrics": "Frame-level; macro precision, recall, and F1.",
        "segmentation_metrics": (
            "No interval ground-truth metric computed; segment counts are output counts."
        ),
        "training_samples": int(len(data["X_train"])),
        "test_samples": int(len(data["X_test"])),
        "train_recordings": [int(length) for length in data["train_lengths"]],
        "test_recordings": [int(length) for length in data["test_lengths"]],
    }
    config_path = results_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")

    print("\nComparison on the same test split:")
    for title, metrics in (("HMM only", hmm_metrics), ("HMM + graph ASM", asm_metrics)):
        print(f"{title}:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}%")
    print("\nAllowed activity transitions (configured graph):")
    for activity, destinations in zip(graph.labels, graph.allowed_transitions):
        print(f"  {activity}: {list(destinations)}")
    print(f"HMM-only segments: {len(hmm_segments)}")
    print(f"HMM + graph ASM segments: {len(asm_segments)}")
    print(f"Run directory: {results_dir}")

    return {
        "results_dir": results_dir,
        "hmm_metrics": hmm_metrics,
        "asm_metrics": asm_metrics,
        "hmm_predictions": hmm_predictions,
        "asm_predictions": asm_predictions,
        "hmm_segments": hmm_segments,
        "asm_segments": asm_segments,
        "graph": graph,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the one-HMM and graph-ASM activity-recognition pipeline."
    )
    parser.add_argument("--data-file", type=Path, default=DATA_FILE)
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="New, non-existing directory for model and results.",
    )
    args = parser.parse_args(argv)
    run_pipeline(args.data_file, args.results_dir)


if __name__ == "__main__":
    main()
