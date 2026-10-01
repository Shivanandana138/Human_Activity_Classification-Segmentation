from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT_DIR / "results"

STATES_FILE = RESULTS_DIR / "predicted_states.npy"

PLOT_FILE = RESULTS_DIR / "one_hmm_state_distribution.png"


# ---------------------------------------------------------
# Analyze hidden-state predictions
# ---------------------------------------------------------

def main():

    print("Analyzing One-HMM results...")

    if not STATES_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{STATES_FILE}\n\n"
            "Run evaluate_one_hmm.py first."
        )

    states = np.load(STATES_FILE)

    print()
    print("=" * 60)
    print("ONE-HMM RESULT ANALYSIS")
    print("=" * 60)

    print(f"\nTotal observations: {len(states)}")
    print(f"Number of unique states: {len(np.unique(states))}")

    # -----------------------------------------------------
    # State distribution
    # -----------------------------------------------------

    unique_states, counts = np.unique(
        states,
        return_counts=True
    )

    print("\nHidden-State Distribution:")
    print("-" * 60)

    for state, count in zip(unique_states, counts):

        percentage = (
            count / len(states)
        ) * 100

        print(
            f"State {state:<3} "
            f"Count: {count:<6} "
            f"Percentage: {percentage:.2f}%"
        )

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        unique_states,
        counts
    )

    plt.xlabel("Hidden State")
    plt.ylabel("Number of Observations")
    plt.title("One-HMM Hidden State Distribution")

    plt.xticks(unique_states)

    plt.tight_layout()

    plt.savefig(
        PLOT_FILE,
        dpi=300
    )

    plt.close()

    print(
        f"\nSaved: {PLOT_FILE}"
    )

    print("\nAnalysis completed.")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()