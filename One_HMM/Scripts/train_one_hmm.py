from pathlib import Path

from One_HMM.src.hmm import (
    load_uci_har,
    train_one_hmm,
    save_model,
)


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "Data" / "UCI HAR Dataset"
MODEL_DIR = ROOT_DIR / "models"


def main():

    print("Loading UCI HAR dataset...")

    X_train, y_train, X_test, y_test = load_uci_har(
        DATA_DIR
    )

    print(f"Training samples : {X_train.shape}")
    print(f"Testing samples  : {X_test.shape}")

    print("\nTraining single HMM...")

    model = train_one_hmm(X_train)

    print("\nSaving model...")

    save_model(
        model,
        MODEL_DIR
    )

    print("\nSingle HMM trained successfully.")


if __name__ == "__main__":
    main()