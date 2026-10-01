from pathlib import Path
import numpy as np


ACTIVITIES = {
    1: "Walking",
    2: "Walking Upstairs",
    3: "Walking Downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Lying Down",
}


def load_uci_har(data_dir):
    """
    Load the UCI HAR feature data.

    Returns
    -------
    X_train : ndarray
        Training features, shape (7352, 561)

    y_train : ndarray
        Training labels

    X_test : ndarray
        Testing features, shape (2947, 561)

    y_test : ndarray
        Testing labels
    """

    data_dir = Path(data_dir)

    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    X_train = np.loadtxt(train_dir / "X_train.txt")
    y_train = np.loadtxt(
        train_dir / "y_train.txt",
        dtype=int
    )

    X_test = np.loadtxt(test_dir / "X_test.txt")
    y_test = np.loadtxt(
        test_dir / "y_test.txt",
        dtype=int
    )

    return X_train, y_train, X_test, y_test


def load_subject_data(data_dir):
    """
    Load subject IDs for train and test data.
    """

    data_dir = Path(data_dir)

    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    subjects_train = np.loadtxt(
        train_dir / "subject_train.txt",
        dtype=int
    )

    subjects_test = np.loadtxt(
        test_dir / "subject_test.txt",
        dtype=int
    )

    return subjects_train, subjects_test


def get_activity_name(label):
    return ACTIVITIES[int(label)]