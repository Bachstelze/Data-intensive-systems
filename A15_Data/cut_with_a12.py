#!/usr/bin/env python3
"""
A15 — Cut each "good" clip between start→stop frames using the A12 model.

Pipeline:
  1. For each clip in a15_good_rescaled.csv, load its Kinect CSV.
  2. Engineer the same feature columns used in A12 training (distances,
     velocities, accelerations).
  3. Run the A12 Dense classifier (A_Kinect_Dense_relu_adam_bs64) frame by
     frame → binary prediction (non-exercise = 0, exercise = 1).
  4. Find the first 0→1 transition → START, last 1→0 transition → STOP.
  5. Cut the original CSV to frames [START, STOP].
  6. If the resulting sequence is shorter than MIN_FRAMES, mark it as too
     short and skip the output.
  7. Save cut CSVs + a summary CSV.

Output directories:
  A15/a15_cut/               — cut CSV files (only those meeting min length)
  A15/a15_cut_summary.csv    — per-clip metadata
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError as exc:
    raise ImportError("Install joblib: pip install joblib") from exc

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError as exc:
    raise ImportError("Install TensorFlow: pip install tensorflow") from exc

# --------------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent
REPO = HERE.parent

GOOD_CSV       = HERE / "a15_good_rescaled.csv"
KINECT_DIR     = REPO / "kinect_good_preprocessed"
A12_RESULTS    = REPO / "A12" / "A12_results"
OUT_DIR        = HERE / "a15_cut"

WEIGHTS_PATH   = A12_RESULTS / "A_Kinect_Dense_relu_adam_bs64.weights.h5"
SCALER_PATH    = A12_RESULTS / "A_Kinect_Dense_relu_adam_bs64_scaler.pkl"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------- consts
JOINTS = [
    "head", "left_shoulder", "left_elbow", "right_shoulder", "right_elbow",
    "left_hand", "right_hand", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_foot", "right_foot",
]

KINECT_COLS = [f"{j}_{d}" for j in JOINTS for d in ["x", "y", "z"]]  # 39 cols

EXTRA_COLS = [
    "left_hand_to_left_shoulder", "right_hand_to_right_shoulder",
    "left_hand_to_left_hip", "right_hand_to_right_hip",
    "left_elbow_to_left_shoulder", "right_elbow_to_right_shoulder",
    "head_to_hip",
    "head_vx", "head_vy", "head_vz", "head_speed",
    "left_hand_vx", "left_hand_vy", "left_hand_vz", "left_hand_speed",
    "right_hand_vx", "right_hand_vy", "right_hand_vz", "right_hand_speed",
    "head_ax", "head_ay", "head_az", "head_accel",
    "left_hand_ax", "left_hand_ay", "left_hand_az", "left_hand_accel",
    "right_hand_ax", "right_hand_ay", "right_hand_az", "right_hand_accel",
]

FEATURE_COLS = KINECT_COLS + EXTRA_COLS  # 70 cols, matching A12 training

# Minimum acceptable cut length in frames (≈ 1 s at 30 fps).
MIN_FRAMES = 30

LABEL_NAMES = ["non-exercise", "exercise"]


# --------------------------------------------------------------- feature engineering

def distance(df: pd.DataFrame, a: str, b: str) -> np.ndarray:
    z_part = (df[f"{a}_z"] - df[f"{b}_z"]) ** 2 if f"{a}_z" in df.columns and f"{b}_z" in df.columns else 0
    return np.sqrt(
        (df[f"{a}_x"] - df[f"{b}_x"]) ** 2 +
        (df[f"{a}_y"] - df[f"{b}_y"]) ** 2 +
        z_part
    )


def calculate_joint_distances(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["left_hand_to_left_shoulder"] = distance(df, "left_hand", "left_shoulder")
    df["right_hand_to_right_shoulder"] = distance(df, "right_hand", "right_shoulder")
    df["left_hand_to_left_hip"] = distance(df, "left_hand", "left_hip")
    df["right_hand_to_right_hip"] = distance(df, "right_hand", "right_hip")
    df["left_elbow_to_left_shoulder"] = distance(df, "left_elbow", "left_shoulder")
    df["right_elbow_to_right_shoulder"] = distance(df, "right_elbow", "right_shoulder")
    hip_mid_x = (df["left_hip_x"] + df["right_hip_x"]) / 2
    hip_mid_y = (df["left_hip_y"] + df["right_hip_y"]) / 2
    hip_mid_z = (df["left_hip_z"] + df["right_hip_z"]) / 2
    df["head_to_hip"] = np.sqrt(
        (df["head_x"] - hip_mid_x) ** 2 +
        (df["head_y"] - hip_mid_y) ** 2 +
        (df["head_z"] - hip_mid_z) ** 2
    )
    return df


def calculate_velocity_features(df: pd.DataFrame, fps: float = 30.0) -> pd.DataFrame:
    df = df.copy()
    for joint in ["head", "left_hand", "right_hand"]:
        for dim in ["x", "y", "z"]:
            col = f"{joint}_{dim}"
            if col in df.columns:
                df[f"{joint}_v{dim}"] = np.diff(df[col], prepend=df[col].iloc[0]) * fps
            else:
                df[f"{joint}_v{dim}"] = 0.0
        df[f"{joint}_speed"] = np.sqrt(
            df[f"{joint}_vx"] ** 2 + df[f"{joint}_vy"] ** 2 + df[f"{joint}_vz"] ** 2
        )
    return df


def calculate_acceleration_features(df: pd.DataFrame, fps: float = 30.0) -> pd.DataFrame:
    df = df.copy()
    for joint in ["head", "left_hand", "right_hand"]:
        for dim in ["x", "y", "z"]:
            vcol = f"{joint}_v{dim}"
            df[f"{joint}_a{dim}"] = np.diff(df[vcol], prepend=df[vcol].iloc[0]) * fps
        df[f"{joint}_accel"] = np.sqrt(
            df[f"{joint}_ax"] ** 2 + df[f"{joint}_ay"] ** 2 + df[f"{joint}_az"] ** 2
        )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer the same feature set used during A12 model training."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = calculate_joint_distances(df)
    df = calculate_velocity_features(df)
    df = calculate_acceleration_features(df)
    return df


# --------------------------------------------------------------------- model

def build_dense_model(input_dim: int) -> keras.Model:
    """Rebuild the architecture of A_Kinect_Dense_relu_adam_bs64."""
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-4),
                     name="dense_1")(inputs)
    x = layers.Dropout(0.2, name="drop_1")(x)
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-4),
                     name="dense_2")(x)
    x = layers.Dropout(0.2, name="drop_2")(x)
    outputs = layers.Dense(2, activation="softmax", name="output")(x)
    return keras.Model(inputs, outputs, name="Dense")


def load_model_and_scaler() -> tuple[keras.Model, object]:
    """Load the A12 best model weights and its StandardScaler."""
    scaler = joblib.load(str(SCALER_PATH))
    input_dim = int(getattr(scaler, "n_features_in_", len(getattr(scaler, "mean_", []))))
    print(f"  Scaler expects {input_dim} features.")

    model = build_dense_model(input_dim)
    model.load_weights(str(WEIGHTS_PATH))
    print(f"  Model loaded from {WEIGHTS_PATH}")
    return model, scaler


# --------------------------------------------------------------------- prediction

def predict_exercise_frames(
    df_raw: pd.DataFrame,
    model: keras.Model,
    scaler: object,
) -> np.ndarray:
    """Predict exercise (1) vs non-exercise (0) for every frame.

    Returns integer array of shape (n_frames,) with {0, 1}.
    """
    df_feat = engineer_features(df_raw)
    available = [c for c in FEATURE_COLS if c in df_feat.columns]
    X = df_feat[available].values.astype(np.float32)

    X_scaled = scaler.transform(X)
    probs = model.predict(X_scaled, verbose=0)  # (n_frames, 2)
    preds = np.argmax(probs, axis=1)            # 0 = non-exercise, 1 = exercise
    return preds


def find_start_stop(preds: np.ndarray, frames: np.ndarray) -> tuple[int | None, int | None]:
    """Find the first 0→1 transition (start) and last 1→0 transition (stop)."""
    starts = np.where((preds[:-1] == 0) & (preds[1:] == 1))[0] + 1
    stops  = np.where((preds[:-1] == 1) & (preds[1:] == 0))[0] + 1

    start_frame = None
    stop_frame  = None

    if len(starts) > 0 and len(stops) > 0:
        # Best pair: first start, last stop (ensures start < stop)
        if starts[0] < stops[-1]:
            start_frame = int(frames[starts[0]])
            stop_frame  = int(frames[stops[-1]])
        else:
            # Fallback to first start and first stop after it
            for s in starts:
                valid_stops = stops[stops > s]
                if len(valid_stops) > 0:
                    start_frame = int(frames[s])
                    stop_frame  = int(frames[valid_stops[-1]])
                    break
    elif len(starts) > 0:
        start_frame = int(frames[starts[0]])
        if len(stops) == 0:
            stop_frame = int(frames[-1])
    elif len(stops) > 0:
        stop_frame = int(frames[stops[-1]])
        start_frame = int(frames[0])

    return start_frame, stop_frame


# --------------------------------------------------------------------- main

def main() -> None:
    print("=" * 65)
    print("A15 — Cut good clips using A12 start/stop model")
    print("=" * 65)

    # 1. Load good clips list
    print(f"\n[1] Loading good clips from {GOOD_CSV}")
    good_df = pd.read_csv(GOOD_CSV)
    good_df.columns = [c.strip() for c in good_df.columns]
    clips = good_df["clip"].tolist()
    print(f"    {len(clips)} clips to process")

    # 2. Load model
    print(f"\n[2] Loading A12 model and scaler")
    model, scaler = load_model_and_scaler()

    # 3. Process each clip
    print(f"\n[3] Processing clips")
    results: list[dict] = []
    n_too_short = 0
    n_ok = 0

    for idx, clip_name in enumerate(clips):
        kinect_path = KINECT_DIR / f"{clip_name}.csv"
        if not kinect_path.exists():
            print(f"  [{idx+1}/{len(clips)}] {clip_name}: Kinect CSV not found, skipping")
            results.append({
                "clip": clip_name,
                "status": "missing_kinect_csv",
                "n_frames_input": 0,
                "n_frames_cut": 0,
                "start_frame": None,
                "stop_frame": None,
                "cut_path": "",
            })
            continue

        df_raw = pd.read_csv(kinect_path)
        df_raw.columns = [c.strip() for c in df_raw.columns]
        n_input = len(df_raw)
        frames = df_raw["FrameNo"].values if "FrameNo" in df_raw.columns else np.arange(n_input)

        preds = predict_exercise_frames(df_raw, model, scaler)
        start_f, stop_f = find_start_stop(preds, frames)

        if start_f is None or stop_f is None:
            print(f"  [{idx+1}/{len(clips)}] {clip_name}: no exercise segment found, keeping full clip")
            cut_df = df_raw.copy()
            start_f = int(frames[0])
            stop_f  = int(frames[-1])
        else:
            mask = (frames >= start_f) & (frames <= stop_f)
            cut_df = df_raw.loc[mask].copy()

        n_cut = len(cut_df)
        too_short = n_cut < MIN_FRAMES

        if too_short:
            n_too_short += 1
            status = f"too_short_{n_cut}_frames"
            cut_path = ""
            print(f"  [{idx+1}/{len(clips)}] {clip_name}: cut={n_cut} frames "
                  f"(< {MIN_FRAMES}) → SKIPPED")
        else:
            n_ok += 1
            status = "ok"
            out_path = OUT_DIR / f"{clip_name}.csv"
            cut_df.to_csv(out_path, index=False)
            cut_path = str(out_path)
            print(f"  [{idx+1}/{len(clips)}] {clip_name}: "
                  f"in={n_input} frames, cut={n_cut} frames "
                  f"[{start_f}–{stop_f}] → saved")

        results.append({
            "clip": clip_name,
            "status": status,
            "n_frames_input": n_input,
            "n_frames_cut": n_cut,
            "start_frame": start_f,
            "stop_frame": stop_f,
            "cut_path": cut_path,
        })

    # 4. Save summary
    print(f"\n[4] Saving summary")
    result_df = pd.DataFrame(results)
    summary_path = HERE / "a15_cut_summary.csv"
    result_df.to_csv(summary_path, index=False)
    print(f"    Summary saved to {summary_path}")

    # 5. Print final report
    print(f"\n{'=' * 65}")
    print("FINAL REPORT")
    print(f"{'=' * 65}")
    print(f"  Total clips in good list:  {len(clips)}")
    print(f"  Processed:                 {len(results)}")
    print(f"  Cut OK:                    {n_ok}")
    print(f"  Too short (< {MIN_FRAMES} fr): {n_too_short}")
    print(f"  Missing Kinect CSV:        {len(clips) - len(results)}")
    print(f"  Output directory:          {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
