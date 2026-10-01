from Six_HMM.src.hmm import (
    load_uci_har,
    train_six_hmms,
    save_models,
)

from Six_HMM.src.hmm  import load_subject_data


DATA_DIR = (
    r"C:\Users\Shivananda"
    r"\Human_Activity_Classification-Segmentation"
    r"\Six_HMM"
    r"\Data\UCI HAR Dataset"
)

MODEL_DIR = (
    r"C:\Users\Shivananda"
    r"\Human_Activity_Classification-Segmentation"
    r"\Six_HMM"
    r"\models"
)


def main():

    print("Loading UCI HAR dataset...")

    X_train, y_train, X_test, y_test = load_uci_har(
        DATA_DIR
    )

    print(f"Training samples : {X_train.shape}")
    print(f"Testing samples  : {X_test.shape}")

    print("\nLoading subject information...")

    subject_train, subject_test = load_subject_data(
        DATA_DIR
    )

    print(
        f"Training subjects: {subject_train.shape}"
    )

    print(
        f"Testing subjects : {subject_test.shape}"
    )

    print(
        "\nTraining six activity-specific HMMs "
        "using sequences..."
    )

    models = train_six_hmms(
        X_train,
        y_train,
        subject_train
    )

    print("\nSaving trained models...")

    save_models(
        models,
        MODEL_DIR
    )

    print(
        "\nAll six sequence-aware HMMs "
        "trained successfully."
    )


if __name__ == "__main__":
    main()