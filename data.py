from pathlib import Path
import numpy as np
from config import ACTIVITIES

def read_matrix(path):
    return np.loadtxt(path)

def load_split(data_dir, split):
    split_dir = Path(data_dir) / split
    X = read_matrix(split_dir / f" X_  \)
 y = np.loadtxt(split_dir / f\y_  \, dtype=int)
 subject = np.loadtxt(split_dir / f\subject_  \, dtype=int)
 return X, y, subject

def load_data(data_dir):
 Xtr, ytr, str_ = load_split(data_dir, \train\)
 Xte, yte, ste = load_split(data_dir, \test\)
 return (Xtr, ytr, str_), (Xte, yte, ste)

