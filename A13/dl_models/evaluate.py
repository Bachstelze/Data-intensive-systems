"""Evaluation helpers (confusion matrix, metrics tables, plots)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRIC_KEYS = ["tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "auc"]


def predict_proba(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    return model.predict(X, verbose=0).reshape(-1)


def metrics_from_predictions(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    auc = float("nan")
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_proba)
    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "auc": auc,
    }


def confusion(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return confusion_matrix(y_true, (y_proba >= threshold).astype(int), labels=[0, 1])


def plot_confusion(cm: np.ndarray, title: str, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["bad", "good"]); ax.set_yticklabels(["bad", "good"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    return ax


def metrics_table(rows: Iterable[dict], index: Iterable[str]):
    import pandas as pd
    df = pd.DataFrame(list(rows), index=list(index))
    cols = [c for c in METRIC_KEYS if c in df.columns]
    return df[cols].round(4)
