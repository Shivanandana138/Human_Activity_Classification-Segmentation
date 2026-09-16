from dataclasses import dataclass
from pathlib import Path

ACTIVITIES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}
N_ACTIVITIES = 6

@dataclass
class Config:
    data_dir: Path
    work_dir: Path = Path("artifacts")
    n_states: int = 8
    n_mix: int = 10
    covariance_type: str = "diag"
    random_state: int = 42
    hmm_max_iter: int = 25
    gmm_max_iter: int = 50
    gmm_reg_covar: float = 1e-4
    transition_smoothing: float = 1e-2
    min_activity_frames: int = 2
    max_subjects: int | None = None
    quick: bool = False

    @property
    def models_dir(self):
        return self.work_dir / "models"

    @property
    def results_dir(self):
        return self.work_dir / "results"

    @property
    def plots_dir(self):
        return self.work_dir / "plots"

    @property
    def predictions_dir(self):
        return self.work_dir / "predictions"
