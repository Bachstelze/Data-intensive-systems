"""Hyper-parameter sweep for Issue #13.

Goal: find a smaller Dense (target ~10k params) and a CNN (=<20% Dense params)
that match or beat the current saved champions while still respecting the
"CNN test metrics no more than 10% worse than Dense" rule.

Runs grouped 5-fold CV (suffix-stripped clip id) on the augmented train split
for each configuration, then refits the best config on the full train_aug and
evaluates on the held-out test set. NOTHING is overwritten in saved/; results
go to A13/dl_models/sweep_results/.
"""

from __future__ import annotations

import json
import os
import sys
import time
from itertools import product
from pathlib import Path

# Silence TF info logs but keep warnings.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from A13.dl_models import models as M
from A13.dl_models.data_loader import load_dataset

OUT_DIR = THIS_DIR / "sweep_results"
OUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# Grids                                                                       #
# --------------------------------------------------------------------------- #
DENSE_GRID = [
    # (hidden_units, dropout, lr)
    ((64, 32),       0.3, 1e-3),
    ((48, 24),       0.3, 1e-3),
    ((32, 16),       0.3, 1e-3),
    ((24, 12),       0.3, 1e-3),
    ((64, 32),       0.5, 1e-3),
    ((32, 16),       0.5, 1e-3),
    ((48, 24),       0.3, 5e-4),
    ((24, 12, 8),    0.3, 1e-3),
]

CNN_GRID = [
    # (filters, kernel, dense_units, dropout, lr)
    ((8, 16),  (3, 3), 16, 0.3, 1e-3),
    ((4, 8),   (3, 3),  8, 0.3, 1e-3),
    ((4, 8),   (3, 3), 16, 0.3, 1e-3),
    ((8, 8),   (3, 3),  8, 0.3, 1e-3),
    ((4, 8),   (5, 3),  8, 0.3, 1e-3),
    ((4, 8),   (3, 3),  8, 0.5, 1e-3),
    ((4, 8),   (3, 3),  8, 0.3, 5e-4),
    ((6, 12),  (3, 3), 12, 0.3, 1e-3),
]

CV_FOLDS = 5
CV_EPOCHS = 60
FINAL_EPOCHS = 120
BATCH = 32


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _callbacks(patience: int = 12):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(3, patience // 3), min_lr=1e-5
        ),
    ]


def _class_weight(y):
    pos = float(y.sum()); neg = float(len(y) - pos)
    if pos == 0 or neg == 0:
        return {0: 1.0, 1: 1.0}
    total = pos + neg
    return {0: total / (2 * neg), 1: total / (2 * pos)}


def _cv_score(build_fn, ds, n_splits=CV_FOLDS, epochs=CV_EPOCHS):
    X, y, g = ds.X_train_aug, ds.y_train_aug, ds.train_groups
    n_splits = min(n_splits, len(np.unique(g)))
    gkf = GroupKFold(n_splits=n_splits)
    fold_aucs, fold_accs = [], []
    for tr, va in gkf.split(X, y, g):
        tf.keras.backend.clear_session()
        model = build_fn()
        model.fit(
            X[tr], y[tr],
            validation_data=(X[va], y[va]),
            epochs=epochs, batch_size=BATCH,
            callbacks=_callbacks(),
            class_weight=_class_weight(y[tr]),
            verbose=0,
        )
        m = model.evaluate(X[va], y[va], verbose=0, return_dict=True)
        fold_aucs.append(float(m["auc"]))
        fold_accs.append(float(m["accuracy"]))
    return {
        "auc_mean": float(np.mean(fold_aucs)), "auc_std": float(np.std(fold_aucs)),
        "acc_mean": float(np.mean(fold_accs)), "acc_std": float(np.std(fold_accs)),
        "fold_aucs": fold_aucs, "fold_accs": fold_accs,
    }


def _refit_test(build_fn, ds):
    tf.keras.backend.clear_session()
    model = build_fn()
    model.fit(
        ds.X_train_aug, ds.y_train_aug,
        validation_split=0.0,
        epochs=FINAL_EPOCHS, batch_size=BATCH,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=15, restore_best_weights=True)],
        class_weight=_class_weight(ds.y_train_aug),
        verbose=0,
    )
    m = model.evaluate(ds.X_test, ds.y_test, verbose=0, return_dict=True)
    return {k: float(v) for k, v in m.items()}, M.count_params(model)


# --------------------------------------------------------------------------- #
# Sweep                                                                       #
# --------------------------------------------------------------------------- #
def sweep_dense(problem: str):
    ds = load_dataset(problem, "Dense")
    input_dim = ds.X_train_aug.shape[1]
    print(f"\n=== Dense sweep, problem {problem} (input_dim={input_dim}) ===", flush=True)
    results = []
    for i, (hu, dr, lr) in enumerate(DENSE_GRID, 1):
        build = lambda hu=hu, dr=dr, lr=lr: M.build_dense(
            input_dim=input_dim, hidden_units=hu, dropout=dr, learning_rate=lr,
        )
        params = M.count_params(build())
        t0 = time.time()
        cv = _cv_score(build, ds)
        dt = time.time() - t0
        row = {
            "problem": problem, "kind": "Dense", "config": {
                "hidden_units": list(hu), "dropout": dr, "lr": lr,
            },
            "params": params,
            **cv,
            "cv_seconds": round(dt, 1),
        }
        results.append(row)
        print(f"  [{i:2d}/{len(DENSE_GRID)}] hu={hu} dr={dr} lr={lr} "
              f"params={params} auc={cv['auc_mean']:.3f}\u00b1{cv['auc_std']:.3f} "
              f"acc={cv['acc_mean']:.3f} ({dt:.0f}s)", flush=True)
    return results


def sweep_cnn(problem: str):
    ds = load_dataset(problem, "CNN")
    shp = tuple(ds.X_train_aug.shape[1:])
    print(f"\n=== CNN sweep, problem {problem} (input_shape={shp}) ===", flush=True)
    results = []
    for i, (filt, ks, du, dr, lr) in enumerate(CNN_GRID, 1):
        build = lambda filt=filt, ks=ks, du=du, dr=dr, lr=lr: M.build_cnn(
            input_shape=shp, filters=filt, kernel_size=ks, dense_units=du,
            dropout=dr, learning_rate=lr,
        )
        params = M.count_params(build())
        t0 = time.time()
        cv = _cv_score(build, ds)
        dt = time.time() - t0
        row = {
            "problem": problem, "kind": "CNN", "config": {
                "filters": list(filt), "kernel_size": list(ks),
                "dense_units": du, "dropout": dr, "lr": lr,
            },
            "params": params,
            **cv,
            "cv_seconds": round(dt, 1),
        }
        results.append(row)
        print(f"  [{i:2d}/{len(CNN_GRID)}] f={filt} k={ks} du={du} dr={dr} lr={lr} "
              f"params={params} auc={cv['auc_mean']:.3f}\u00b1{cv['auc_std']:.3f} "
              f"acc={cv['acc_mean']:.3f} ({dt:.0f}s)", flush=True)
    return results


def pick_champion(results):
    # Rank by mean AUC, tiebreak by smaller params.
    return sorted(results, key=lambda r: (-r["auc_mean"], r["params"]))[0]


def main():
    overall = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": [],
               "champions": {}, "test_metrics": {}}

    for problem in ("A", "B"):
        dense_results = sweep_dense(problem)
        cnn_results   = sweep_cnn(problem)
        overall["results"].extend(dense_results)
        overall["results"].extend(cnn_results)

        d_champ = pick_champion(dense_results)
        c_champ = pick_champion(cnn_results)

        # Refit champion and report test metrics.
        print(f"\n-- Refit champions for problem {problem} --", flush=True)
        ds_d = load_dataset(problem, "Dense")
        ds_c = load_dataset(problem, "CNN")
        d_build = lambda c=d_champ["config"], dim=ds_d.X_train_aug.shape[1]: M.build_dense(
            input_dim=dim, hidden_units=tuple(c["hidden_units"]),
            dropout=c["dropout"], learning_rate=c["lr"])
        c_build = lambda c=c_champ["config"], shp=tuple(ds_c.X_train_aug.shape[1:]): M.build_cnn(
            input_shape=shp, filters=tuple(c["filters"]),
            kernel_size=tuple(c["kernel_size"]), dense_units=c["dense_units"],
            dropout=c["dropout"], learning_rate=c["lr"])
        d_metrics, d_params = _refit_test(d_build, ds_d)
        c_metrics, c_params = _refit_test(c_build, ds_c)
        ratio = c_params / max(d_params, 1)
        acc_gap = (d_metrics["accuracy"] - c_metrics["accuracy"]) / max(d_metrics["accuracy"], 1e-9)
        print(f"  Dense champ: {d_champ['config']}  params={d_params}  "
              f"test_auc={d_metrics['auc']:.3f} test_acc={d_metrics['accuracy']:.3f}",
              flush=True)
        print(f"  CNN   champ: {c_champ['config']}  params={c_params}  "
              f"test_auc={c_metrics['auc']:.3f} test_acc={c_metrics['accuracy']:.3f}",
              flush=True)
        print(f"  ratio CNN/Dense params = {ratio:.2%}   "
              f"acc gap (Dense-CNN)/Dense = {acc_gap:+.1%}", flush=True)
        overall["champions"][problem] = {"Dense": d_champ, "CNN": c_champ}
        overall["test_metrics"][problem] = {
            "Dense": {"params": d_params, **d_metrics},
            "CNN":   {"params": c_params, **c_metrics},
            "param_ratio_cnn_over_dense": ratio,
            "acc_gap_dense_minus_cnn_rel": acc_gap,
        }

    out_path = OUT_DIR / f"sweep_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(overall, indent=2))
    latest = OUT_DIR / "latest.json"
    latest.write_text(json.dumps(overall, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
