import argparse
from pathlib import Path
from config import Config
from train import train_models
from evaluate import evaluate

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="Path to UCI_HAR_Dataset")
    p.add_argument("--work_dir", default="artifacts")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    cfg = Config(
        data_dir=Path(args.data_dir),
        work_dir=Path(args.work_dir),
        quick=args.quick,
    )
    for d in [cfg.work_dir,cfg.models_dir,cfg.results_dir,cfg.plots_dir,cfg.predictions_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("Training six activity HMM/GMM models...")
    train_models(cfg)
    print("\nEvaluating continuous sequences + Activity Sequence Models...")
    evaluate(cfg)
    print(f"\nDone. Results are in: {cfg.work_dir.resolve()}")

if __name__ == "__main__":
    main()
