import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.hmm.preprocessing import ACTIVITIES


def main():

    # -----------------------------------------------------
    # Load saved predictions
    # -----------------------------------------------------

    true_labels = np.load(
        "true_labels.npy",
        allow_pickle=True
    )

    predicted_labels = np.load(
        "predicted_labels.npy",
        allow_pickle=True
    )

    activity_names = list(ACTIVITIES.values())

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=activity_names
    )

    print("\nConfusion Matrix:")
    print(cm)

    # -----------------------------------------------------
    # Per-activity results
    # -----------------------------------------------------

    print("\nPer-Activity Results:")
    print("-" * 60)

    for i, activity in enumerate(activity_names):

        total = cm[i].sum()
        correct = cm[i, i]

        accuracy = (
            correct / total
            if total > 0
            else 0
        )

        print(
            f"{activity:<20}"
            f"Correct: {correct:<5}"
            f"Total: {total:<5}"
            f"Accuracy: {accuracy * 100:.2f}%"
        )

    # -----------------------------------------------------
    # Plot confusion matrix
    # -----------------------------------------------------

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=activity_names
    )

    disp.plot(
        xticks_rotation=45
    )

    plt.title(
        "Six-HMM Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        "six_hmm_confusion_matrix.png",
        dpi=300
    )

    plt.show()

    print(
        "\nSaved: six_hmm_confusion_matrix.png"
    )


if __name__ == "__main__":
    main()