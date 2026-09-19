from src.hmm import (
    load_uci_har,
    train_six_hmms,
    save_models,
)


DATA_DIR = (
    r"C:\Users\Shivananda"
    r"\Human_Activity_Classification-Segmentation"
    r"\Data\UCI HAR Dataset"
)

MODEL_DIR = (
    r"C:\Users\Shivananda"
    r"\Human_Activity_Classification-Segmentation"
    r"\models"
)


def main():

    print("Loading UCI HAR dataset...")

    X_train, y_train, X_test, y_test = load_uci_har(
        DATA_DIR
    )

    print(
        f"Training samples : {X_train.shape}"
    )

    print(
        f"Testing samples  : {X_test.shape}"
    )

    print(
        "\nTraining six activity-specific HMMs..."
    )

    models = train_six_hmms(
        X_train,
        y_train
    )

    print(
        "\nSaving trained models..."
    )

    save_models(
        models,
        MODEL_DIR
    )

    print(
        "\nAll six HMMs trained successfully."
    )


if __name__ == "__main__":
    main()