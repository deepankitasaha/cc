"""Evaluation metrics suited to extreme class imbalance (0.17% positives).

Plain accuracy is reported for completeness but is NOT informative here: predicting
"never fraud" already scores 99.83%. AUPRC (average precision) is the primary metric.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from .model import bce_loss, predict_proba


def evaluate(params, X, y, threshold: float = 0.5, pos_weight: float = 10.0) -> dict:
    p = predict_proba(params, X)
    yhat = (p >= threshold).astype(int)
    return {
        "loss": bce_loss(params, X, y, pos_weight),
        "accuracy": float(np.mean(yhat == y)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "auprc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
    }
