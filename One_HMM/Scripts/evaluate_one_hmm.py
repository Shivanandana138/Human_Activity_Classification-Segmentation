from pathlib import Path
import numpy as np

from One_HMM.src.hmm import (
    load_uci_har,
    load_model,
    predict_states,
)


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "Data" / "UCI HAR Dataset"
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"


def main():

    print("Loading UCI HAR test dataset...")

    X_train, y_train, X_test, y_test = load_uci_har(
        DATA_DIR
    )

    print(f"Test samples: {X_test.shape}")

    print("\nLoading one HMM...")

    model = load_model(MODEL_DIR)

    print("One HMM loaded.")

    print("\nPredicting hidden states...")

    states = predict_states(
        model,
        X_test
    )

    print(f"Predicted states: {states.shape}")

    print("\nHidden-state distribution:")

    unique, counts = np.unique(
        states,
        return_counts=True
    )

    for state, count in zip(unique, counts):
        print(
            f"State {state}: {count} samples"
        )

    # Save predictions
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prediction_file = RESULTS_DIR / "predicted_states.npy"
    np.save(prediction_file, states)

    print(f"\nPredictions saved to: {prediction_file}")

    print("\nOne-HMM evaluation completed.")


if __name__ == "__main__":
    main()