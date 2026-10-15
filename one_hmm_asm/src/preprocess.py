from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "Data" / "UCI HAR Dataset"
OUTPUT_DIR = PROJECT_DIR / "Data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "one_hmm_data.npz"


SIGNAL_CHANNELS = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)


def _read_feature_names(data_dir):
    feature_names = []

    with (data_dir / "features.txt").open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                feature_names.append(parts[1])

    if not feature_names:
        raise ValueError("No feature names found.")

    return np.asarray(feature_names, dtype=np.str_)


def _read_activity_names(data_dir):
    activities = {}

    with (data_dir / "activity_labels.txt").open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                activities[int(parts[0])] = parts[1]

    if set(activities) != set(range(1, 7)):
        raise ValueError("Expected activity labels 1 to 6.")

    return activities


def _load_split(data_dir, split, feature_count):
    split_dir = data_dir / split

    X = np.loadtxt(
        split_dir / f"X_{split}.txt",
        dtype=np.float32,
        ndmin=2,
    )

    y = np.loadtxt(
        split_dir / f"y_{split}.txt",
        dtype=np.int64,
        ndmin=1,
    )

    subjects = np.loadtxt(
        split_dir / f"subject_{split}.txt",
        dtype=np.int64,
        ndmin=1,
    )

    # Basic checks
    if X.shape[1] != feature_count:
        raise ValueError(
            f"{split}: expected {feature_count} features, got {X.shape[1]}"
        )

    if not (len(X) == len(y) == len(subjects)):
        raise ValueError(
            f"{split}: X, y and subject lengths do not match."
        )

    if not np.isfinite(X).all():
        raise ValueError(f"{split}: X contains NaN or infinity.")

    if not np.isin(y, np.arange(1, 7)).all():
        raise ValueError(f"{split}: invalid activity labels.")

    return X, y, subjects


def _get_sequence_lengths(subjects):
    """
    Find consecutive subject sequences.

    Example:
        subjects = [1,1,1,2,2,3,3,3]

        lengths = [3,2,3]

    These lengths are later passed to hmmlearn so that it does
    not treat the boundary between subjects as a real transition.
    """

    if len(subjects) == 0:
        raise ValueError("Subject array is empty.")

    boundaries = np.flatnonzero(subjects[1:] != subjects[:-1]) + 1

    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(subjects)]

    lengths = ends - starts
    sequence_subjects = subjects[starts]

    return lengths.astype(np.int64), sequence_subjects.astype(np.int64)


def _load_inertial_signals(data_dir, split, window_count):
    """
    Load the 9 raw inertial signal channels.

    Shape:
        (number_of_windows, 128, 9)
    """

    signal_dir = data_dir / split / "Inertial Signals"

    signals = []

    for channel in SIGNAL_CHANNELS:
        signal = np.loadtxt(
            signal_dir / f"{channel}_{split}.txt",
            dtype=np.float32,
            ndmin=2,
        )

        if signal.shape[0] != window_count:
            raise ValueError(
                f"{split}/{channel}: expected {window_count} windows, "
                f"got {signal.shape[0]}"
            )

        if not np.isfinite(signal).all():
            raise ValueError(
                f"{split}/{channel}: contains NaN or infinity."
            )

        signals.append(signal)

    # (9, windows, 128) -> (windows, 128, 9)
    return np.stack(signals, axis=-1)


def preprocess_and_save(data_dir=DATA_DIR, output_file=OUTPUT_FILE):
    data_dir = Path(data_dir)
    output_file = Path(output_file)

    # ---------------------------------------------------------
    # 1. Read metadata
    # ---------------------------------------------------------

    feature_names = _read_feature_names(data_dir)
    activity_names = _read_activity_names(data_dir)

    # ---------------------------------------------------------
    # 2. Load train/test data
    # ---------------------------------------------------------

    X_train, y_train, subjects_train = _load_split(
        data_dir,
        "train",
        len(feature_names),
    )

    X_test, y_test, subjects_test = _load_split(
        data_dir,
        "test",
        len(feature_names),
    )

    # ---------------------------------------------------------
    # 3. Find sequence boundaries
    # ---------------------------------------------------------

    train_lengths, train_sequence_subjects = _get_sequence_lengths(
        subjects_train
    )

    test_lengths, test_sequence_subjects = _get_sequence_lengths(
        subjects_test
    )

    # ---------------------------------------------------------
    # 4. Load raw inertial signals
    # ---------------------------------------------------------

    signals_train = _load_inertial_signals(
        data_dir,
        "train",
        len(X_train),
    )

    signals_test = _load_inertial_signals(
        data_dir,
        "test",
        len(X_test),
    )

    # ---------------------------------------------------------
    # 5. Save everything needed for experiments
    # ---------------------------------------------------------

    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_file,

        # Main HMM observations
        X_train=X_train,
        X_test=X_test,

        # Ground-truth labels
        y_train=y_train,
        y_test=y_test,

        # Subject information
        subjects_train=subjects_train,
        subjects_test=subjects_test,

        # Sequence information for hmmlearn
        train_lengths=train_lengths,
        test_lengths=test_lengths,

        train_sequence_subjects=train_sequence_subjects,
        test_sequence_subjects=test_sequence_subjects,

        # Metadata
        feature_names=feature_names,
        activity_labels=np.arange(1, 7),
        activity_names=np.asarray(
            [activity_names[i] for i in range(1, 7)],
            dtype=np.str_,
        ),

        # Raw inertial data
        signal_channels=np.asarray(
            SIGNAL_CHANNELS,
            dtype=np.str_,
        ),
        signals_train=signals_train,
        signals_test=signals_test,
    )

    # ---------------------------------------------------------
    # 6. Print summary
    # ---------------------------------------------------------

    print(f"Saved: {output_file}")
    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test:  {y_test.shape}")

    print(
        f"Train sequences: {len(train_lengths)} "
        f"({train_sequence_subjects.tolist()})"
    )

    print(
        f"Test sequences: {len(test_lengths)} "
        f"({test_sequence_subjects.tolist()})"
    )

    print(f"Raw signals train: {signals_train.shape}")
    print(f"Raw signals test:  {signals_test.shape}")

    return output_file


if __name__ == "__main__":
    preprocess_and_save()