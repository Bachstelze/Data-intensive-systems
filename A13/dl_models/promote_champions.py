"""Promote sweep champions into A13/dl_models/saved/ — the files app.py loads.

What this does:
  1. Read champion configs from A13/dl_models/sweep_results/latest.json.
  2. Back up existing saved/ models to saved/legacy_<timestamp>/.
  3. Retrain each champion (A_Dense, A_CNN, B_Dense, B_CNN) on the full
     augmented training split using the same protocol as sweep._refit_test.
  4. Save each model as saved/{P}_{kind}.keras.
  5. Evaluate on the held-out test split and write training_summary.json.
  6. Run the smoke test from the report (zero, good, bad inputs) on A_CNN.

Safe to re-run; idempotent except that each promotion bumps the legacy
backup directory.

Run:
    .venv/bin/python A13/dl_models/promote_champions.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

THIS_DIR  = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from A13.dl_models import models as M
from A13.dl_models.data_loader import load_dataset

SAVED_DIR  = THIS_DIR / "saved"
SWEEP_PATH = THIS_DIR / "sweep_results" / "latest.json"

FINAL_EPOCHS = 120
BATCH        = 32


def class_weight(y: np.ndarray) -> dict[int, float]:
    pos = float(y.sum()); neg = float(len(y) - pos)
    if pos == 0 or neg == 0:
        return {0: 1.0, 1: 1.0}
    total = pos + neg
    return {0: total / (2 * neg), 1: total / (2 * pos)}


def build_champion(problem: str, kind: str, ds, cfg: dict) -> tf.keras.Model:
    if kind == "Dense":
        return M.build_dense(
            input_dim=ds.input_shape[0],
            hidden_units=tuple(cfg["hidden_units"]),
            dropout=cfg["dropout"],
            learning_rate=cfg["lr"],
            name=f"{problem}_Dense",
        )
    return M.build_cnn(
        input_shape=ds.input_shape,
        filters=tuple(cfg["filters"]),
        kernel_size=tuple(cfg["kernel_size"]),
        dense_units=cfg["dense_units"],
        dropout=cfg["dropout"],
        learning_rate=cfg["lr"],
        name=f"{problem}_CNN",
    )


def fit_and_eval(model: tf.keras.Model, ds) -> dict:
    model.fit(
        ds.X_train_aug, ds.y_train_aug,
        validation_split=0.0,
        epochs=FINAL_EPOCHS,
        batch_size=BATCH,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=15, restore_best_weights=True)],
        class_weight=class_weight(ds.y_train_aug),
        verbose=0,
    )
    metrics = model.evaluate(ds.X_test, ds.y_test, verbose=0, return_dict=True)
    return {k: float(v) for k, v in metrics.items()}


def smoke_test(model: tf.keras.Model, ds_cnn) -> dict:
    """Same checks as A14_Report_v2.ipynb §5.3."""
    X, y, files = ds_cnn.X_test, ds_cnn.y_test, ds_cnn.test_filenames
    good_idx = next((i for i in range(len(y)) if int(y[i]) == 1), None)
    bad_idx  = next((i for i in range(len(y)) if int(y[i]) == 0), None)

    p_zero = float(model(np.zeros((1, *ds_cnn.input_shape), dtype="float32"),
                         training=False).numpy()[0, 0])
    p_good = float(model(X[good_idx:good_idx + 1], training=False).numpy()[0, 0]) if good_idx is not None else float("nan")
    p_bad  = float(model(X[bad_idx:bad_idx + 1],  training=False).numpy()[0, 0])  if bad_idx  is not None else float("nan")
    spread = p_good - p_bad
    return {
        "p_zero": p_zero, "p_good": p_good, "p_bad": p_bad,
        "spread": spread,
        "discriminates": bool(p_good > 0.5 and p_bad < 0.5),
        "good_clip": str(files[good_idx]) if good_idx is not None else None,
        "bad_clip":  str(files[bad_idx])  if bad_idx  is not None else None,
    }


def main() -> None:
    print(f"Loading sweep results from {SWEEP_PATH}")
    sweep = json.loads(SWEEP_PATH.read_text())

    # Backup existing saved/ folder
    stamp     = time.strftime("%Y%m%d_%H%M%S")
    legacy    = SAVED_DIR / f"legacy_{stamp}"
    if SAVED_DIR.exists():
        legacy.mkdir(parents=True)
        for f in SAVED_DIR.iterdir():
            if f.is_file():
                shutil.copy2(f, legacy / f.name)
        print(f"Backed up existing saved/ models -> {legacy}")
    SAVED_DIR.mkdir(exist_ok=True)

    summary: dict = {"promoted_at": stamp, "from": str(SWEEP_PATH), "models": {}}

    for problem in ("A", "B"):
        for kind in ("Dense", "CNN"):
            cfg = sweep["champions"][problem][kind]["config"]
            print(f"\n=== Promoting {problem}_{kind}  cfg={cfg} ===")
            tf.keras.backend.clear_session()
            ds    = load_dataset(problem, kind)
            model = build_champion(problem, kind, ds, cfg)
            t0    = time.perf_counter()
            metrics = fit_and_eval(model, ds)
            dt    = time.perf_counter() - t0
            params = M.count_params(model)

            out_path = SAVED_DIR / f"{problem}_{kind}.keras"
            model.save(out_path)
            print(f"  -> saved {out_path}  ({params} params, "
                  f"test auc={metrics.get('auc', float('nan')):.3f}, "
                  f"acc={metrics.get('accuracy', float('nan')):.3f}, "
                  f"trained in {dt:.1f}s)")

            summary["models"][f"{problem}_{kind}"] = {
                "config":        cfg,
                "params":        params,
                "test_metrics":  metrics,
                "training_seconds": round(dt, 1),
            }

    # Smoke test against the just-promoted A_CNN
    print("\n=== Smoke test on promoted A_CNN ===")
    ds_cnn_A = load_dataset("A", "CNN")
    a_cnn    = tf.keras.models.load_model(SAVED_DIR / "A_CNN.keras", compile=False)
    smoke    = smoke_test(a_cnn, ds_cnn_A)
    print(json.dumps(smoke, indent=2))
    summary["smoke_test"] = smoke

    out_summary = SAVED_DIR / "training_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_summary}")

    print("\nDone.")


if __name__ == "__main__":
    main()
