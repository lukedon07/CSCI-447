"""Evaluation metrics. All functions take plain label / value arrays.

For regression, convert predictions back to the original target units first
(y_pred * std + mean) so RMSE is reported in meaningful units.
"""
import numpy as np


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def macro_f1(y_true, y_pred, n_classes):
    """Unweighted mean of per-class F1, over classes that occur in y_true or y_pred."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        if tp + fp + fn == 0:
            continue                      # class absent from both -> not counted
        scores.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def mse(y_true, y_pred):
    d = np.asarray(y_true, dtype=float).ravel() - np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean(d ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true, y_pred):
    """1 - SS_res / SS_tot. Returns 0.0 if y_true is constant."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot)


def classification_metrics(y_true, y_pred, n_classes):
    acc = accuracy(y_true, y_pred)
    return {"error": 1.0 - acc, "accuracy": acc,
            "macro_f1": macro_f1(y_true, y_pred, n_classes)}


def regression_metrics(y_true, y_pred):
    return {"error": rmse(y_true, y_pred), "mse": mse(y_true, y_pred),
            "rmse": rmse(y_true, y_pred), "r2": r2(y_true, y_pred)}
