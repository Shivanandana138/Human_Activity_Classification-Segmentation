from pathlib import Path
import numpy as np
import pandas as pd


ACTIVITIES = {
    1: "Walking",
    2: "Walking Upstairs",
    3: "Walking Downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Lying Down",
}


def load_split(data_dir, split):
    """
    Load one UCI HAR split.

    Parameters
    ----------
    data_dir : Path
        Path to 'UCI HAR Dataset'.

    split : str
        Either 'train' or 'test'.

    Returns
    -------
    X : ndarray
        Shape (N, 561)

    y : ndarray
        Activity labels.

    subjects : ndarray
        Subject IDs.
    """

    split_dir = Path(data_dir) / split

    X = np.loadtxt(split_dir / f"X_{split}.txt")
    y = np.loadtxt(
        split_dir / f"y_{split}.txt",
        dtype=int
    )
    subjects = np.loadtxt(
        split_dir / f"subject_{split}.txt",
        dtype=int
    )

    return X, y, subjects


def load_uci_har(data_dir):
    """
    Load both train and test splits.
    """

    X_train, y_train, subjects_train = load_split(
        data_dir,
        "train"
    )

    X_test, y_test, subjects_test = load_split(
        data_dir,
        "test"
    )

    return (
        X_train,
        y_train,
        subjects_train,
        X_test,
        y_test,
        subjects_test,
    )


def create_dataframe(X, y, subjects):
    """
    Create a convenient dataframe while preserving
    the original row order.
    """

    df = pd.DataFrame(X)

    df["activity"] = y
    df["subject"] = subjects

    df["activity_name"] = df["activity"].map(ACTIVITIES)

    df["row_order"] = np.arange(len(df))

    return df

# Sequence constructing next