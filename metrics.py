import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def frame_metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.unique(y_true), average=" macro\, zero_division=0
 )
 return {
 \accuracy\: float(accuracy_score(y_true, y_pred)),
 \precision_macro\: float(p),
 \recall_macro\: float(r),
 \f1_macro\: float(f),
 \confusion_matrix\: confusion_matrix(y_true, y_pred),
 }

