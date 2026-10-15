import argparse
from pathlib import Path

from config import Config
from train import train_models
from evaluate import evaluate


def main():
    p = argparse.ArgumentParser(description="Human activity recognition with six activity HMMs")
    p.add_argument("--data_dir", required=True, help="Path to UCI_HAR_Dataset")
    p.add_argument("--work_dir", default="artifacts")
    p.add_argument("--skip_train", action="store_true",
                   help="Skip training and load HMMs from <work_dir>/models")
    p.add_argument("--quick", action="store_true",
                   help="Smoke test (less data, fewer EM iterations)")
    args = p.parse_args()

    cfg = Config(
        data_dir=Path(args.data_dir),
        work_dir=Path(args.work_dir),
        quick=args.quick,
    )
    for d in [cfg.work_dir, cfg.models_dir, cfg.results_dir, cfg.plots_dir, cfg.predictions_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if args.skip_train:
        print(f"Skipping training, loading HMMs from {cfg.models_dir}")
    else:
        print("Training six activity HMM/GMM models...")
        train_models(cfg)

    print("\nPredicting activities with the six HMMs...")
    evaluate(cfg)
    print(f"\nDone. Results are in: {cfg.work_dir.resolve()}")


if __name__ == "__main__":
    main()
