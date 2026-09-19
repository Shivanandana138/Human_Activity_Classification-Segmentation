import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.hmm import (
    load_uci_har,
    load_models,
)

from src.hmm.preprocessing import ACTIVITIES


DATA_DIR = r"C:\Users\Shivananda\Human_Activity_Classification-Segmentation\Data\UCI HAR Dataset"

MODEL_DIR = r"C:\Users\Shivananda\Human_Activity_Classification-Segmentation\models"


def main():

    # -----------------------------------------------------
    # Load test data
    # -----------------------------------------------------

    print("Loading UCI HAR test dataset...")

    X_train, y_train, X_test, y_test = load_uci_har(
        DATA_DIR
    )

    print(f"Test samples: {X_test.shape}")

    # -----------------------------------------------------
    # Load six trained HMMs
    # -----------------------------------------------------

    print("\nLoading trained HMMs...")

    models = load_models(MODEL_DIR)

    print(
        f"Loaded {len(models)} activity models."
    )

    # -----------------------------------------------------
    # Predict each test sample
    # -----------------------------------------------------

    print("\nEvaluating six HMMs...")

    activity_names = list(ACTIVITIES.values())

    predictions = []

    for i, X_sample in enumerate(X_test):

        # Each UCI HAR row is one feature vector.
        # Convert it into a single-observation sequence.

        X_sample = X_sample.reshape(1, -1)

        likelihoods = {}

        for activity_name, model in models.items():

            try:

                # Apply the scaler saved with the model
                if hasattr(model, "activity_scaler"):

                    X_scaled = (
                        model.activity_scaler
                        .transform(X_sample)
                    )

                    X_scaled = np.clip(
                        X_scaled,
                        -10,
                        10
                    )

                else:
                    X_scaled = X_sample

                score = model.score(X_scaled)

                if np.isfinite(score):
                    likelihoods[activity_name] = score
                else:
                    likelihoods[activity_name] = float("-inf")

            except (
                ValueError,
                np.linalg.LinAlgError,
                FloatingPointError,
            ):
                likelihoods[activity_name] = float("-inf")

        predicted_activity = max(
            likelihoods,
            key=likelihoods.get
        )

        predictions.append(
            predicted_activity
        )

        if (i + 1) % 500 == 0:
            print(
                f"Processed {i + 1}/{len(X_test)} samples..."
            )

    # -----------------------------------------------------
    # Convert numerical labels to activity names
    # -----------------------------------------------------

    true_labels = [
        ACTIVITIES[int(label)]
        for label in y_test
    ]

    # -----------------------------------------------------
    # Overall accuracy
    # -----------------------------------------------------

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    print("\n" + "=" * 60)
    print("SIX-HMM EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"\nOverall Accuracy: {accuracy * 100:.2f}%"
    )

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            true_labels,
            predictions,
            labels=activity_names,
            target_names=activity_names,
            zero_division=0,
        )
    )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    cm = confusion_matrix(
        true_labels,
        predictions,
        labels=activity_names
    )

    print("\nConfusion Matrix:")

    print("Rows = Actual")
    print("Columns = Predicted\n")

    print(
        "                  "
        + " ".join(
            f"{name[:8]:>10}"
            for name in activity_names
        )
    )

    for i, row in enumerate(cm):

        print(
            f"{activity_names[i]:<20}"
            + " ".join(
                f"{value:>10}"
                for value in row
            )
        )


if __name__ == "__main__":
    main()