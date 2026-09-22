from pathlib import Path
import joblib


def save_model(model, model_dir):
    """
    Save the single HMM.
    """

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "one_hmm.pkl"

    joblib.dump(model, model_path)

    print(f"Saved: {model_path}")


def load_model(model_dir):
    """
    Load the single HMM.
    """

    model_path = Path(model_dir) / "one_hmm.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return joblib.load(model_path)