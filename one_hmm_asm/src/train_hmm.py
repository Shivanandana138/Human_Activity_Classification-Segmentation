from pathlib import Path
import argparse

import joblib
import numpy as np
from hmmlearn.hmm import GMMHMM


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "Data" / "processed" / "one_hmm_data.npz"
MODEL_FILE = PROJECT_DIR / "models" / "one_hmm.pkl"

N_STATES = 6
N_MIXTURES = 10
N_ITER = 100
TOL = 1e-3
RANDOM_STATE = 42

PREPROCESSED_KEYS = (
    "X_train",
    "y_train",
    "train_lengths",
    "X_test",
    "y_test",
    "test_lengths",
    "activity_labels",
    "activity_names",
)


def load_preprocessed_data(data_file=DATA_FILE):
    with np.load(data_file, allow_pickle=False) as archive:
        missing = set(PREPROCESSED_KEYS).difference(archive.files)
        if missing:
            raise ValueError(
                f"Preprocessed archive is missing required arrays: {sorted(missing)}"
            )
        return {key: archive[key] for key in PREPROCESSED_KEYS}


def create_hmm():
    return GMMHMM(
        n_components=N_STATES,
        n_mix=N_MIXTURES,
        covariance_type="diag",
        weights_prior=1.01,
        means_weight=1e-3,
        covars_prior=0.0,
        covars_weight=1.0,
        n_iter=N_ITER,
        tol=TOL,
        random_state=RANDOM_STATE,
        verbose=True,
    )


def train_hmm(X_train, train_lengths):
    X_train = np.asarray(X_train)
    train_lengths = np.asarray(train_lengths, dtype=np.int64)

    if X_train.ndim != 2 or X_train.shape[0] == 0:
        raise ValueError("Training observations must be a non-empty matrix.")
    if (
        train_lengths.ndim != 1
        or np.any(train_lengths <= 0)
        or int(train_lengths.sum()) != len(X_train)
    ):
        raise ValueError("Training sequence lengths do not match observations.")

    model = create_hmm()
    print("\nTraining HMM...")
    print(f"Observations: {X_train.shape}")
    print(f"States: {N_STATES}")
    print(f"Sequences: {len(train_lengths)}")
    model.fit(X_train, lengths=train_lengths.tolist())
    print("\nHMM training complete.")
    return model


def save_model(model, model_path=MODEL_FILE):
    model_path = Path(model_path)
    if model_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing model: {model_path}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved model: {model_path}")
    return model_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and save the one-HMM model.")
    parser.add_argument("--data-file", type=Path, default=DATA_FILE)
    parser.add_argument("--model-path", type=Path, default=MODEL_FILE)
    args = parser.parse_args(argv)

    data = load_preprocessed_data(args.data_file)
    model = train_hmm(data["X_train"], data["train_lengths"])
    save_model(model, args.model_path)


if __name__ == "__main__":
    main()
