"""
Activity prediction using only the six activity HMMs.

Every test frame is scored by all six HMMs (one per activity); the activity whose
HMM gives the highest log-likelihood is the prediction.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from config import ACTIVITIES
from data import load_data, group_by_subject
from train import load_models
from metrics import frame_metrics, boundary_error_rate, segment_error_diagnostic

METRIC_COLS = [
    "accuracy", "precision_macro", "recall_macro", "f1_macro",
    "boundary_error", "segment_error_diagnostic",
]


def get_log_scores(models, X):
    """[T, 6] matrix: log-likelihood of every frame under each activity HMM."""
    return np.column_stack([models[aid].score_samples(X) for aid in ACTIVITIES])


def normalize_scores(scores):
    return scores - scores.max(axis=1, keepdims=True)


def predict_activities(models, X):
    """Predict an activity id (1..6) for every frame of X."""
    return np.argmax(normalize_scores(get_log_scores(models, X)), axis=1) + 1


def evaluate(cfg):
    _, (Xte, yte, ste) = load_data(cfg.data_dir)
    models = load_models(cfg)

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    cfg.predictions_dir.mkdir(parents=True, exist_ok=True)

    rows, yt_all, yp_all = [], [], []
    for item in group_by_subject(Xte, yte, ste, cfg.max_subjects):
        X, y = item["X"], item["y"]
        pred = predict_activities(models, X)

        m = frame_metrics(y, pred)
        rows.append({
            "subject": item["subject"],
            "accuracy": m["accuracy"],
            "precision_macro": m["precision_macro"],
            "recall_macro": m["recall_macro"],
            "f1_macro": m["f1_macro"],
            "boundary_error": boundary_error_rate(y, pred),
            "segment_error_diagnostic": segment_error_diagnostic(y, pred),
        })
        np.savetxt(
            cfg.predictions_dir / f"subject_{item['subject']}.txt",
            np.column_stack([y, pred]), fmt="%d", header="true predicted",
        )
        yt_all.extend(y.tolist())
        yp_all.extend(pred.tolist())

    df = pd.DataFrame(rows)
    summary = df[METRIC_COLS].mean().to_frame("6_HMM").T
    print("\n=== RESULTS (6 HMMs) ===")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    df.to_csv(cfg.results_dir / "per_subject_results.csv", index=False)
    summary.to_csv(cfg.results_dir / "summary.csv")

    # Confusion matrix aggregated across all test subjects.
    fig, ax = plt.subplots(figsize=(7, 7))
    ConfusionMatrixDisplay.from_predictions(
        yt_all, yp_all, labels=list(ACTIVITIES.keys()),
        display_labels=list(ACTIVITIES.values()), xticks_rotation=45, ax=ax,
    )
    ax.set_title("UCI-HAR — 6 HMM/GMM models")
    fig.tight_layout()
    fig.savefig(cfg.plots_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    return summary
