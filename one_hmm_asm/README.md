# One-HMM + Graph ASM

This folder contains a human-activity recognition and segmentation pipeline built around one Gaussian mixture HMM and a graph-constrained activity decoder.

## Files

- `Data/UCI HAR Dataset/` — Source UCI HAR data.
- `Data/processed/one_hmm_data.npz` — Preprocessed train/test features, labels, activity names, and recording lengths.
- `src/preprocess.py` — Builds the preprocessing archive from the source dataset.
- `src/train_hmm.py` — Loads the archive, trains one 6-state HMM with 10 diagonal Gaussian mixtures per state, and saves the model.
- `src/evaluate_hmm.py` — Maps hidden states to activity labels using training data, evaluates HMM-only predictions, and creates activity segments.
- `src/graph_asm.py` — Defines the configurable activity graph and Viterbi-style decoder. Self-transitions are allowed; non-self edges are project-specific assumptions, not a reproduction of the paper's unavailable edge list.
- `src/run_hmm_asm.py` — Runs training, HMM-only evaluation, graph-constrained decoding, comparison, and result saving.
- `src/copilot_preprocess.py` — Additional preprocessing script; the main pipeline uses `preprocess.py` and the existing archive.
- `tests/test_graph_asm.py` — Tests the model configuration, activity mapping, and graph decoder behavior.
- `models/` — Default location used by standalone HMM training/evaluation.
- `results/` — Evaluation outputs. Full pipeline runs use a new timestamped subfolder.

## Data flow

`UCI HAR Dataset` → `preprocess.py` → `Data/processed/one_hmm_data.npz` → `run_hmm_asm.py`

The runner trains one HMM on training features, learns a one-to-one hidden-state/activity mapping from training data only, and evaluates HMM-only predictions on the test recordings. It then turns HMM state posteriors into activity evidence and decodes each test recording independently under the configured graph. It saves both predictions, frame-level metrics, confusion matrices, and predicted segments.

## Run sequence

Use the `pyS3` environment, with the terminal's working directory set to this folder:

```powershell
conda activate pyS3
```

The preprocessed archive is already included, so normally run the complete pipeline directly:

```powershell
python -m src.run_hmm_asm
```

To regenerate the archive first, then run the pipeline:

```powershell
python -m src.preprocess
python -m src.run_hmm_asm
```

Run the focused tests with:

```powershell
python -m unittest discover -s tests
```

`train_hmm.py` and `evaluate_hmm.py` can also be run separately for standalone HMM training/evaluation; they are not required before the full runner.

## Architecture and results

The model is `hmmlearn.GMMHMM` with 6 hidden states, 10 diagonal Gaussian mixtures per state, and the existing preprocessed 561-feature data. Hidden-state IDs are not treated as activity IDs: a one-to-one assignment based on decoded training data supplies the mapping. The graph decoder uses hard transition constraints, with all self-transitions permitted. Its non-self transitions are explicitly configured in `src/graph_asm.py`.

Recorded full-run results are in `results/hmm_asm_20261015_221737/`. On the same test split:

| Frame-level metric | HMM only | HMM + graph ASM |
|---|---:|---:|
| Accuracy | 59.1449% | 59.4164% |
| Macro precision | 46.3409% | 49.1924% |
| Macro recall | 58.2177% | 58.5029% |
| Macro F1 | 48.0883% | 49.1156% |
| Predicted segment count | 241 | 280 |

The segment counts are output counts, not segmentation-quality metrics. The run directory also contains the trained model, predictions, configuration, and evaluation summary.
