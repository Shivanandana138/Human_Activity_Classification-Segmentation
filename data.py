from pathlib import Path
import numpy as np
from config import ACTIVITIES

def read_matrix(path):
    return np.loadtxt(path)

def load_split(data_dir, split):
    split_dir = Path(data_dir) / split
    X = read_matrix(split_dir / f"X_{split}.txt")
    y = np.loadtxt(split_dir / f"y_{split}.txt", dtype=int)
    subject = np.loadtxt(split_dir / f"subject_{split}.txt", dtype=int)
    return X, y, subject

def load_data(data_dir):
    Xtr, ytr, str_ = load_split(data_dir, "train")
    Xte, yte, ste = load_split(data_dir, "test")
    return (Xtr, ytr, str_), (Xte, yte, ste)

def load_features(data_dir):
    path = Path(data_dir) / "features.txt"
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            names.append(parts[1] if len(parts) == 2 else parts[0])
    return names

def group_by_subject(X, y, subject, max_subjects=None):
    """Return chronological frame sequences per subject, preserving file order."""
    ids = np.unique(subject)
    if max_subjects:
        ids = ids[:max_subjects]
    out = []
    for sid in ids:
        idx = np.where(subject == sid)[0]
        # UCI rows are already ordered in the source files; keep that order.
        out.append({
            "subject": int(sid),
            "X": X[idx],
            "y": y[idx],
        })
    return out

def make_activity_training_sequences(X, y, subject, cfg):
    """Collect contiguous same-activity runs for training one HMM per activity."""
    sequences = {a: [] for a in ACTIVITIES}
    for item in group_by_subject(X, y, subject, cfg.max_subjects):
        ys = item["y"]
        xs = item["X"]
        start = 0
        for i in range(1, len(ys) + 1):
            if i == len(ys) or ys[i] != ys[start]:
                label = int(ys[start])
                run = xs[start:i]
                if len(run) >= cfg.min_activity_frames:
                    sequences[label].append(run)
                start = i
    return sequences

def make_continuous_test_sequences(X, y, subject, cfg):
    return group_by_subject(X, y, subject, cfg.max_subjects)
