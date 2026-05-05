"""Prediction service for A12 start/stop exercise classifiers."""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np

from A12.service.contracts import LABEL_NAMES, normalize_problem, read_pose_csv, validate_pose_dataframe

MODEL_VERSION = "A12 Dense classifiers from A12_results"

MODEL_FILES = {
    "A": {
        "name": "A_Kinect_Dense_relu_adam_bs64",
        "weights": "A_Kinect_Dense_relu_adam_bs64.weights.h5",
        "scaler": "A_Kinect_Dense_relu_adam_bs64_scaler.pkl",
    },
    "B": {
        "name": "B_PoseNet_Dense_relu_adam_bs64",
        "weights": "B_PoseNet_Dense_relu_adam_bs64.weights.h5",
        "scaler": "B_PoseNet_Dense_relu_adam_bs64_scaler.pkl",
    },
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _model_dir() -> Path:
    """Prefer A12/A12_results, but also support A12_results at repo root."""
    root = _project_root()
    candidates = [root / "A12" / "A12_results", root / "A12_results"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _build_dense_model(input_dim: int):
    """Build the same dense architecture used by A12_classifier.py."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    # Keeping seed stable avoids warnings about unseeded initialisers before
    # loading trained weights.
    tf.random.set_seed(42)
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_1",
    )(inputs)
    x = layers.Dropout(0.2, name="drop_1")(x)
    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_2",
    )(x)
    x = layers.Dropout(0.2, name="drop_2")(x)
    outputs = layers.Dense(2, activation="softmax", name="output")(x)
    return keras.Model(inputs, outputs, name="Dense")


@lru_cache(maxsize=2)
def load_classifier(problem: str) -> Dict[str, Any]:
    """Load model weights and scaler for Problem A or B."""
    problem_key = normalize_problem(problem)
    info = MODEL_FILES[problem_key]
    directory = _model_dir()
    scaler_path = directory / info["scaler"]
    weights_path = directory / info["weights"]

    if not scaler_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            "Missing A12 model files. Expected files in A12/A12_results/: "
            f"{info['scaler']} and {info['weights']}"
        )

    scaler = joblib.load(scaler_path)
    input_dim = int(getattr(scaler, "n_features_in_"))
    model = _build_dense_model(input_dim)
    model.load_weights(weights_path)
    return {"model": model, "scaler": scaler, "model_name": info["name"]}


def predict_pose_csv(csv_path: str, problem: str = "B") -> Dict[str, Any]:
    """Predict exercise/non-exercise frames from an uploaded pose CSV."""
    started = time.perf_counter()
    problem_key = normalize_problem(problem)
    df = read_pose_csv(csv_path)
    features, feature_names = validate_pose_dataframe(df, problem_key)
    bundle = load_classifier(problem_key)

    scaler = bundle["scaler"]
    model = bundle["model"]
    x_scaled = scaler.transform(features.values.astype(np.float32))
    probabilities = model.predict(x_scaled, verbose=0)
    frame_predictions = np.argmax(probabilities, axis=1)
    exercise_confidence = probabilities[:, 1]

    exercise_ratio = float(np.mean(frame_predictions == 1))
    overall_label_index = int(exercise_ratio >= 0.5)
    overall_confidence = float(np.mean(exercise_confidence))

    frame_preview = []
    for idx in range(min(10, len(frame_predictions))):
        frame_preview.append(
            {
                "frame_index": idx,
                "label": LABEL_NAMES[int(frame_predictions[idx])],
                "confidence": float(np.max(probabilities[idx])),
                "probabilities": {
                    LABEL_NAMES[0]: float(probabilities[idx, 0]),
                    LABEL_NAMES[1]: float(probabilities[idx, 1]),
                },
            }
        )

    return {
        "status": "ok",
        "endpoint": "Gradio tab inside app.py",
        "problem": problem_key,
        "model_name": bundle["model_name"],
        "model_version": MODEL_VERSION,
        "input_contract": "Pose feature CSV with the same feature columns used by A12_classifier.py.",
        "metadata": {
            "rows": int(len(df)),
            "features": int(len(feature_names)),
            "inference_time_ms": round((time.perf_counter() - started) * 1000, 2),
        },
        "prediction": {
            "label": LABEL_NAMES[overall_label_index],
            "confidence": overall_confidence,
            "exercise_frame_ratio": exercise_ratio,
        },
        "frame_preview": frame_preview,
    }


def safe_predict_pose_csv(csv_path: str | None, problem: str = "B") -> Dict[str, Any]:
    """UI-safe wrapper that returns structured errors instead of crashing Gradio."""
    try:
        return predict_pose_csv(str(csv_path), problem)
    except Exception as exc:  # Gradio should show JSON instead of runtime crash.
        return {
            "status": "error",
            "endpoint": "Gradio tab inside app.py",
            "problem": str(problem),
            "message": str(exc),
        }
