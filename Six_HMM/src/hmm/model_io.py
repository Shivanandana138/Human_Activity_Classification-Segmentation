from pathlib import Path
import joblib


def save_models(models, model_dir):
    """
    Save all six trained HMMs.

    Parameters
    ----------
    models : dict
        Dictionary containing six trained models.

    model_dir : str or Path
        Directory where models will be saved.
    """

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    for activity_name, model in models.items():

        filename = (
            activity_name
            .lower()
            .replace(" ", "_")
            + ".pkl"
        )

        filepath = model_dir / filename

        joblib.dump(model, filepath)

        print(f"Saved: {filepath}")


def load_models(model_dir):
    """
    Load the six trained HMMs from disk.
    """

    model_dir = Path(model_dir)

    activity_names = [
        "Walking",
        "Walking Upstairs",
        "Walking Downstairs",
        "Sitting",
        "Standing",
        "Lying Down",
    ]

    models = {}

    for activity_name in activity_names:

        filename = (
            activity_name
            .lower()
            .replace(" ", "_")
            + ".pkl"
        )

        filepath = model_dir / filename

        models[activity_name] = joblib.load(filepath)

    return models