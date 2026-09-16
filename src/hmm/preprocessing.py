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
    """
    Split observations into six activity-specific datasets.

    Returns:
        {
            "Walking": X_walking,
            ...
        }
    """

    activity_data = {}

    for label, activity_name in ACTIVITIES.items():
        activity_data[activity_name] = X[y == label]

    return activity_data