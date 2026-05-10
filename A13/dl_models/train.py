"""Training and cross-validation utilities for Issue #10.

Supports:

* a single train/test fit (``train_final``)
* 10-fold *grouped* cross-validation that keeps all augmentations of the same
  original clip in the same fold (``cross_validate``)
* small grid search over a few hyper-parameter combinations (``grid_search``)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold

from .data_loader import Dataset
from . import models as _models


# --------------------------------------------------------------------------- #
# Common training helpers                                                     #
# --------------------------------------------------------------------------- #
def _callbacks(patience: int = 15) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(3, patience // 3), min_lr=1e-5
        ),
    ]


def class_weight(y: np.ndarray) -> dict[int, float]:
    pos = float(y.sum())
    neg = float(len(y) - pos)
    if pos == 0 or neg == 0:
        return {0: 1.0, 1: 1.0}
    total = pos + neg
    return {0: total / (2 * neg), 1: total / (2 * pos)}


def _evaluate(model: tf.keras.Model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    out = model.evaluate(X, y, verbose=0, return_dict=True)
    return {k: float(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Grouped cross-validation                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class CVResult:
    fold_metrics: list[dict[str, float]]
    mean: dict[str, float]
    std: dict[str, float]


def cross_validate(
    dataset: Dataset,
    build_fn: Callable[[], tf.keras.Model],
    n_splits: int = 10,
    epochs: int = 80,
    batch_size: int = 32,
    use_class_weight: bool = True,
    verbose: int = 0,
) -> CVResult:
    """Run grouped K-fold CV on ``dataset.X_train_aug``.

    Splits use ``dataset.train_groups`` so all augmented copies of one
    original clip stay in the same fold, as required by issue #10.
    """

    X, y, groups = dataset.X_train_aug, dataset.y_train_aug, dataset.train_groups
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    fold_metrics: list[dict[str, float]] = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), start=1):
        tf.keras.backend.clear_session()
        model = build_fn()
        cw = class_weight(y[train_idx]) if use_class_weight else None
        model.fit(
            X[train_idx], y[train_idx],
            validation_data=(X[val_idx], y[val_idx]),
            epochs=epochs, batch_size=batch_size,
            callbacks=_callbacks(),
            class_weight=cw,
            verbose=verbose,
        )
        m = _evaluate(model, X[val_idx], y[val_idx])
        m["fold"] = fold
        fold_metrics.append(m)
        if verbose:
            print(f"  fold {fold:2d}: auc={m['auc']:.3f} acc={m['accuracy']:.3f}")

    keys = [k for k in fold_metrics[0] if k != "fold"]
    mean = {k: float(np.mean([f[k] for f in fold_metrics])) for k in keys}
    std = {k: float(np.std([f[k] for f in fold_metrics])) for k in keys}
    return CVResult(fold_metrics=fold_metrics, mean=mean, std=std)


# --------------------------------------------------------------------------- #
# Final fit on all augmented training data                                    #
# --------------------------------------------------------------------------- #
@dataclass
class TrainResult:
    model: tf.keras.Model
    history: dict[str, list[float]]
    test_metrics: dict[str, float]


def train_final(
    dataset: Dataset,
    build_fn: Callable[[], tf.keras.Model],
    epochs: int = 120,
    batch_size: int = 32,
    val_fraction: float = 0.15,
    use_class_weight: bool = True,
    verbose: int = 0,
    save_path: Path | str | None = None,
) -> TrainResult:
    X, y, groups = dataset.X_train_aug, dataset.y_train_aug, dataset.train_groups

    # Hold out a single grouped validation split for early stopping.
    n_groups = len(np.unique(groups))
    n_val = max(1, int(round(n_groups * val_fraction)))
    gkf = GroupKFold(n_splits=max(2, n_groups // n_val))
    train_idx, val_idx = next(iter(gkf.split(X, y, groups)))

    tf.keras.backend.clear_session()
    model = build_fn()
    cw = class_weight(y[train_idx]) if use_class_weight else None
    history = model.fit(
        X[train_idx], y[train_idx],
        validation_data=(X[val_idx], y[val_idx]),
        epochs=epochs, batch_size=batch_size,
        callbacks=_callbacks(),
        class_weight=cw,
        verbose=verbose,
    )
    test_metrics = _evaluate(model, dataset.X_test, dataset.y_test)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
    return TrainResult(model=model, history={k: list(map(float, v)) for k, v in history.history.items()},
                       test_metrics=test_metrics)


# --------------------------------------------------------------------------- #
# Tiny grid search                                                            #
# --------------------------------------------------------------------------- #
def grid_search(
    dataset: Dataset,
    build_fn_factory: Callable[..., Callable[[], tf.keras.Model]],
    grid: dict[str, Iterable],
    n_splits: int = 5,
    epochs: int = 60,
    batch_size: int = 32,
    verbose: int = 0,
) -> list[dict]:
    """Simple grid search using grouped CV.

    ``build_fn_factory(**hp)`` must return a zero-arg builder of a fresh model.
    Returns a list of dicts sorted by mean validation AUC (best first).
    """

    keys = list(grid.keys())
    results = []
    for combo in product(*[grid[k] for k in keys]):
        hp = dict(zip(keys, combo))
        if verbose:
            print(f"-> {hp}")
        cv = cross_validate(
            dataset, build_fn_factory(**hp),
            n_splits=n_splits, epochs=epochs, batch_size=batch_size,
            verbose=0,
        )
        results.append({"hp": hp, "mean": cv.mean, "std": cv.std})
    results.sort(key=lambda r: r["mean"]["auc"], reverse=True)
    return results
