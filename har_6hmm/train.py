from pathlib import Path
import numpy as np
from data import load_data, make_activity_training_sequences, group_by_subject
from config import ACTIVITIES
from hmm_gmm import GMMHMM

def train_models(cfg):
    (Xtr, ytr, str_), _ = load_data(cfg.data_dir)
    sequences = make_activity_training_sequences(Xtr, ytr, str_, cfg)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)

    models = {}
    for aid, name in ACTIVITIES.items():
        print(f"\n=== Training HMM for {aid}: {name} ===")
        seqs = sequences[aid]
        if not seqs:
            raise RuntimeError(f"No training sequences for activity {aid}")
        # Quick mode reduces the amount of data only for a smoke test.
        if cfg.quick:
            seqs = seqs[:min(20, len(seqs))]
        model = GMMHMM(
            n_states=cfg.n_states,
            n_mix=cfg.n_mix,
            random_state=cfg.random_state + aid,
            max_iter=3 if cfg.quick else cfg.hmm_max_iter,
            gmm_max_iter=10 if cfg.quick else cfg.gmm_max_iter,
            reg_covar=cfg.gmm_reg_covar,
            smoothing=cfg.transition_smoothing,
        )
        model.fit(seqs, verbose=True)
        model.save(cfg.models_dir / f"hmm_{aid}_{name}.joblib")
        models[aid] = model
    return models

def load_models(cfg):
    from hmm_gmm import GMMHMM
    return {aid: GMMHMM.load(cfg.models_dir / f"hmm_{aid}_{name}.joblib")
            for aid,name in ACTIVITIES.items()}
