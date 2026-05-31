import numpy as np

def accuracy_at_k(phi, true_S, k, d):
    """
    phi:   (d,) array of attributions
    true_S: array of indices for true important features
    k:     number of predicted important features (usually len(true_S))
    d:     total number of features
    """
    phi = np.asarray(phi)
    true_S = np.asarray(true_S)

    # Predicted important = top-k
    pred_topk = np.argsort(-np.abs(phi))[:k]

    # Build binary masks
    true_mask = np.zeros(d, dtype=bool)
    pred_mask = np.zeros(d, dtype=bool)

    true_mask[true_S] = True
    pred_mask[pred_topk] = True

    # Compute confusion terms
    TP = np.sum(true_mask & pred_mask)
    TN = np.sum(~true_mask & ~pred_mask)

    return (TP + TN) / d
