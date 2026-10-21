"""Positive-class (vulnerable) Accuracy / Recall / Precision / F1."""

from __future__ import annotations

import numpy as np


def scores(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    return {
        "accuracy": float(acc),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
