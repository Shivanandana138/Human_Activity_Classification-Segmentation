import argparse
from pathlib import Path

from config import Config
from train import train_models
from evaluate_hmm_only import evaluate_hmm_only


def main():
    p = argparse.ArgumentParser(description="Activity prediction using only the 6 activity HMMs")
    p.add_argument("--data_dir", required=True, help="Path to UCI_HAR_Dataset")
    p.add_argument("--work_dir", default="artifacts_hmm_only",
                   help="Where results/plots/predictions (and models, if trained) are written")
    p.add_argument("--reuse_models_from", default=None,
                   help="Work dir of an earlier run (e.g. 'artifacts') whose trained HMMs "
                        "should be reused. Skips training.")
    p.add_argument("--skip_train", action="store_true",
                   help="Skip training and load models from --work_dir/models")
    p.add_argument("--quick", action="store_true", help="Smoke test (less data, fewer EM iterations)")
    args = p.parse_args()

    cfg = Config(data_dir=Path(args.data_dir), work_dir=Path(args.work_dir), quick=args.quick)
    for d in [cfg.work_dir, cfg.models_dir, cfg.results_dir, cfg.plots_dir, cfg.predictions_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if args.reuse_models_from:
        models_cfg = Config(data_dir=Path(args.data_dir), work_dir=Path(args.reuse_models_from))
        print(f"Reusing trained HMMs from {models_cfg.models_dir}")
    elif args.skip_train:
        models_cfg = cfg
        print(f"Skipping training, loading HMMs from {cfg.models_dir}")
    else:
        print("Training six activity HMM/GMM models...")
        train_models(cfg)
        models_cfg = cfg

    print("\nPredicting activities with the 6 HMMs only (no ASM)...")
    evaluate_hmm_only(cfg, models_cfg)
    print(f"\nDone. Results are in: {cfg.work_dir.resolve()}")


if __name__ == "__main__":
    main()
