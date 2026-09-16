from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from data import load_data, group_by_subject
from config import ACTIVITIES
from train import load_models
from asm import ActivitySequenceModel, labels_to_segments
from metrics import frame_metrics, boundary_error_rate, segment_error_diagnostic

def get_log_scores(models, X):
    return np.column_stack([models[aid].score_samples(X) for aid in ACTIVITIES])

def normalize_scores(scores):
    # Per-frame log-score normalization makes the six model scores comparable
    # enough for sequence decoding.
    scores = scores - scores.max(axis=1, keepdims=True)
    return scores

def build_asm_training_sequences(X, y, subject, cfg, models):
    seqs = []
    for item in group_by_subject(X,y,subject,cfg.max_subjects):
        if len(item["X"]) < 2:
            continue
        raw = get_log_scores(models, item["X"])
        pred = np.argmax(raw, axis=1) + 1
        # Use the known training labels to learn activity transition statistics.
        seqs.append(item["y"])
    return seqs

def evaluate(cfg, asm_modes=("none","graph","bigram","trigram")):
    _, (Xte, yte, ste) = load_data(cfg.data_dir)
    models = load_models(cfg)

    # ASM is trained from the training activity-label sequences.
    (Xtr,ytr,str_), _ = load_data(cfg.data_dir)
    asm_sequences = build_asm_training_sequences(Xtr,ytr,str_,cfg,models)
    asms = {}
    for mode in asm_modes:
        if mode == "none":
            asms[mode] = None
        else:
            asms[mode] = ActivitySequenceModel(6, smoothing=1.0).fit(asm_sequences)

    all_results = []
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    cfg.predictions_dir.mkdir(parents=True, exist_ok=True)

    for item in group_by_subject(Xte,yte,ste,cfg.max_subjects):
        X, y = item["X"], item["y"]
        raw = normalize_scores(get_log_scores(models, X))
        for mode in asm_modes:
            pred = (np.argmax(raw,axis=1)+1) if mode=="none" else asms[mode].smooth_scores(raw,mode)
            m = frame_metrics(y,pred)
            result = {
                "subject": item["subject"],
                "asm": mode,
                "accuracy": m["accuracy"],
                "precision_macro": m["precision_macro"],
                "recall_macro": m["recall_macro"],
                "f1_macro": m["f1_macro"],
                "boundary_error": boundary_error_rate(y,pred),
                "segment_error_diagnostic": segment_error_diagnostic(y,pred),
            }
            all_results.append(result)
            np.savetxt(
                cfg.predictions_dir / f"subject_{item['subject']}_{mode}.txt",
                np.column_stack([y,pred]), fmt="%d", header="true predicted"
            )

    df = pd.DataFrame(all_results)
    summary = df.groupby("asm")[[
        "accuracy","precision_macro","recall_macro","f1_macro",
        "boundary_error","segment_error_diagnostic"
    ]].mean()
    print("\n=== RESULTS ===")
    print(summary.to_string(float_format=lambda x:f"{x:.4f}"))
    df.to_csv(cfg.results_dir/"per_subject_results.csv", index=False)
    summary.to_csv(cfg.results_dir/"summary.csv")

    # Confusion matrix for each mode, aggregated across subjects.
    for mode in asm_modes:
        yt, yp = [], []
        for item in group_by_subject(Xte,yte,ste,cfg.max_subjects):
            raw = normalize_scores(get_log_scores(models,item["X"]))
            pred = np.argmax(raw,axis=1)+1 if mode=="none" else asms[mode].smooth_scores(raw,mode)
            yt.extend(item["y"]); yp.extend(pred)
        fig, ax = plt.subplots(figsize=(7,7))
        ConfusionMatrixDisplay.from_predictions(
            yt, yp, labels=list(ACTIVITIES.keys()),
            display_labels=list(ACTIVITIES.values()), xticks_rotation=45, ax=ax
        )
        ax.set_title(f"UCI-HAR HMM/GMM — ASM: {mode}")
        fig.tight_layout()
        fig.savefig(cfg.plots_dir/f"confusion_{mode}.png", dpi=180)
        plt.close(fig)

    return summary
