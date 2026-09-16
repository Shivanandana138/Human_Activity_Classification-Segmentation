import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def frame_metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.unique(y_true), average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

def boundary_error_rate(y_true, y_pred):
    """
    Boundary-based diagnostic: fraction of true change points that do not have a
    predicted change point at the same frame. This is not the paper's ASER formula.
    Use the paper's reported ASER as the external reference point.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    true_b = set(np.where(yt[1:] != yt[:-1])[0] + 1)
    pred_b = set(np.where(yp[1:] != yp[:-1])[0] + 1)
    if not true_b:
        return 0.0
    return len(true_b - pred_b) / len(true_b)

def segment_iou(a0, a1, b0, b1):
    inter = max(0, min(a1,b1) - max(a0,b0) + 1)
    union = max(a1,b1) - min(a0,b0) + 1
    return inter / union if union else 0.0

def segment_error_diagnostic(y_true, y_pred):
    """
    A practical segmentation diagnostic: a true activity segment is considered
    matched when the predicted segment with the same label has IoU >= 0.5.
    This is intentionally labeled a diagnostic rather than the paper's ASER.
    """
    def segs(y):
        out=[]
        s=0
        for i in range(1,len(y)+1):
            if i==len(y) or y[i]!=y[s]:
                out.append((int(y[s]),s,i-1))
                s=i
        return out
    T, P = segs(y_true), segs(y_pred)
    matched=0
    for lab,s,e in T:
        if any(pl==lab and segment_iou(s,e,ps,pe)>=0.5 for pl,ps,pe in P):
            matched += 1
    return 1 - matched / max(1,len(T))
