# UCI-HAR — Activity Recognition with Six HMMs

One GMM-HMM per activity (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING,
STANDING, LAYING). Each test frame is scored by all six models and the activity
with the highest log-likelihood is predicted. No sequence or transition model is
used on top of the HMMs.

    UCI-HAR 561-D feature vectors
        -> one HMM per activity (8 states, 10 diagonal Gaussians per state)
        -> per-frame log-likelihood under each of the 6 HMMs
        -> argmax = predicted activity

## Expected dataset

    UCI_HAR_Dataset/
      features.txt
      train/  X_train.txt  y_train.txt  subject_train.txt
      test/   X_test.txt   y_test.txt   subject_test.txt

## Run

    pip install -r requirements.txt
    python main.py --data_dir /path/to/UCI_HAR_Dataset --work_dir artifacts

    python main.py --data_dir /path/to/UCI_HAR_Dataset --quick         # smoke test
    python main.py --data_dir /path/to/UCI_HAR_Dataset --skip_train    # reuse saved HMMs
    python inspect_dataset.py --data_dir /path/to/UCI_HAR_Dataset      # dataset sanity check

## Metrics (per test subject, then averaged)

accuracy, macro precision, macro recall, macro F1, boundary_error,
segment_error_diagnostic. A confusion matrix is aggregated over all test subjects.

## Outputs

    artifacts/models/        six trained HMMs (.joblib)
    artifacts/results/       per_subject_results.csv, summary.csv
    artifacts/plots/         confusion_matrix.png
    artifacts/predictions/   subject_<id>.txt  (true, predicted per frame)