# UCI-HAR HMM + GMM + Activity Sequence Model

This implementation is an original, runnable reproduction-oriented implementation of the
architecture described by San-Segundo et al. (2016):

UCI-HAR -> 128-sample windows / 50% overlap -> 561-D feature vectors
         -> one HMM per activity
         -> 8 states / HMM
         -> 10 Gaussian components / state
         -> continuous activity decoding
         -> Activity Sequence Model (bigram/trigram/graph-style smoothing)
         -> segmentation + ASER / precision / recall

Important:
- The UCI HAR archive already provides the 561-D feature vectors and 128-sample inertial
  windows. We use those directly rather than reimplementing UCI's feature extraction.
- The paper reports the best configuration as 128 samples, 50% overlap, 8 states/HMM,
  and 10 Gaussians/state.
- The exact original HTK implementation and every internal parameter are not publicly
  specified in the article page, so this is a faithful engineering reproduction of the
  published architecture/configuration, not a claim of bit-for-bit reproduction.

## Expected dataset

Extract UCI HAR so the directory looks like:

UCI_HAR_Dataset/
  activity_labels.txt
  features.txt
  train/
    X_train.txt
    y_train.txt
    subject_train.txt
    Inertial Signals/
      total_acc_x_train.txt
      ...
  test/
    X_test.txt
    y_test.txt
    subject_test.txt
    Inertial Signals/
      ...

## Run

1. Install:
   pip install -r requirements.txt

2. Train + evaluate:
   python main.py --data_dir /path/to/UCI_HAR_Dataset --work_dir artifacts

3. For a quick smoke test:
   python main.py --data_dir /path/to/UCI_HAR_Dataset --work_dir artifacts --quick

Outputs:
  artifacts/models/
  artifacts/results/
  artifacts/plots/
  artifacts/predictions/
