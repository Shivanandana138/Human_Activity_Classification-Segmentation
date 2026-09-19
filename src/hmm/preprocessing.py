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
    Load the UCI HAR train and test feature/label data.

    Expected structure:
        data_dir/
            train/X_train.txt
            train/y_train.txt
            test/X_test.txt
            test/y_test.txt
    """

    data_dir = Path(data_dir)

    X_train = np.loadtxt(data_dir / "train" / "X_train.txt")
    y_train = np.loadtxt(data_dir / "train" / "y_train.txt", dtype=int)

    X_test = np.loadtxt(data_dir / "test" / "X_test.txt")
    y_test = np.loadtxt(data_dir / "test" / "y_test.txt", dtype=int)

    return X_train, y_train, X_test, y_test


def get_activity_name(label):
    """Convert numeric activity label to activity name."""
    return ACTIVITIES[int(label)]


def split_by_activity(X, y):
    activity_data = {}

    for label, activity_name in ACTIVITIES.items():
        activity_data[activity_name] = X[y == label]

    return activity_data
def load_subject_data(data_dir):
    """
    Load subject IDs for train and test sets.
    """

    data_dir = Path(data_dir)

    subject_train = np.loadtxt(
        data_dir / "train" / "subject_train.txt",
        dtype=int
    )

    subject_test = np.loadtxt(
        data_dir / "test" / "subject_test.txt",
        dtype=int
    )

    return subject_train, subject_test
def create_activity_sequences(X, y, subjects):
    """
    Create contiguous sequences using subject and activity labels.

    A new sequence starts whenever:
    - the subject changes, or
    - the activity changes.
    """

    sequences = []

    if not (
        len(X) == len(y) == len(subjects)
    ):
        raise ValueError(
            "X, y and subjects must have the same length."
        )

    start = 0

    for i in range(1, len(X)):

        subject_changed = (
            subjects[i] != subjects[i - 1]
        )

        activity_changed = (
            y[i] != y[i - 1]
        )

        if subject_changed or activity_changed:

            sequences.append(
                {
                    "X": X[start:i],
                    "y": y[start:i],
                    "subject": subjects[start],
                    "activity": y[start],
                }
            )

            start = i

    # Add final sequence
    sequences.append(
        {
            "X": X[start:],
            "y": y[start:],
            "subject": subjects[start],
            "activity": y[start],
        }
    )

    return sequences