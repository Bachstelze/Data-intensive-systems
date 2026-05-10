"""Inference helpers and a small CLI so the model is easy to (re)use.

Examples
--------
Train + save all four models::

    python -m A13.dl_models.predict train --out A13/dl_models/saved

Predict on a NumPy array of features::

    python -m A13.dl_models.predict run --model A13/dl_models/saved/A_Dense.keras \\
                                        --X my_features.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .data_loader import load_all, load_dataset
from .models import build_dense, build_cnn, count_params, assert_param_budget
from .train import train_final
from .evaluate import predict_proba, metrics_from_predictions


SAVED_DIR = Path(__file__).resolve().parent / "saved"


def _builders(dataset):
    if dataset.model_kind == "Dense":
        return lambda: build_dense(input_dim=dataset.input_shape[0], name=f"{dataset.problem}_Dense")
    return lambda: build_cnn(input_shape=dataset.input_shape, name=f"{dataset.problem}_CNN")


def train_all(out_dir: Path = SAVED_DIR, epochs: int = 120, verbose: int = 1) -> dict:
    """Train Dense + CNN for both problems and save them.

    Also asserts the CNN parameter budget (<= 20% of Dense) per problem.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    datasets = load_all()
    summary: dict[str, dict] = {}

    # --- parameter budget check ------------------------------------------------
    for problem in ("A", "B"):
        d = build_dense(input_dim=datasets[(problem, "Dense")].input_shape[0])
        c = build_cnn(input_shape=datasets[(problem, "CNN")].input_shape)
        assert_param_budget(d, c, ratio=0.20)
        summary[f"{problem}_param_counts"] = {
            "dense": count_params(d), "cnn": count_params(c),
            "ratio": count_params(c) / count_params(d),
        }

    # --- train + save ----------------------------------------------------------
    for (problem, kind), dataset in datasets.items():
        if verbose:
            print(f"== training {problem} / {kind} ==  {dataset.summary()}")
        result = train_final(
            dataset, _builders(dataset), epochs=epochs, verbose=verbose,
            save_path=out_dir / f"{problem}_{kind}.keras",
        )
        summary[f"{problem}_{kind}_test_metrics"] = result.test_metrics

    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def predict(model_path: Path | str, X: np.ndarray, threshold: float = 0.5):
    model = tf.keras.models.load_model(model_path)
    proba = predict_proba(model, X)
    return proba, (proba >= threshold).astype(int)


def evaluate_saved(model_path: Path | str, problem: str, model_kind: str) -> dict:
    """Re-evaluate a saved model on the official held-out test set."""
    ds = load_dataset(problem, model_kind)
    proba, _ = predict(model_path, ds.X_test)
    return metrics_from_predictions(ds.y_test, proba)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train / use Issue #10 models.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train all four models.")
    p_train.add_argument("--out", default=str(SAVED_DIR))
    p_train.add_argument("--epochs", type=int, default=120)

    p_eval = sub.add_parser("eval", help="Evaluate a saved model on its test set.")
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--problem", required=True, choices=["A", "B"])
    p_eval.add_argument("--kind", required=True, choices=["Dense", "CNN"])

    p_run = sub.add_parser("run", help="Run inference on a .npy feature array.")
    p_run.add_argument("--model", required=True)
    p_run.add_argument("--X", required=True)
    p_run.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()
    if args.cmd == "train":
        summary = train_all(Path(args.out), epochs=args.epochs)
        print(json.dumps(summary, indent=2))
    elif args.cmd == "eval":
        print(json.dumps(evaluate_saved(args.model, args.problem, args.kind), indent=2))
    elif args.cmd == "run":
        X = np.load(args.X)
        proba, pred = predict(args.model, X, threshold=args.threshold)
        for p, q in zip(proba, pred):
            print(f"{p:.4f}\t{int(q)}")


if __name__ == "__main__":
    _cli()
