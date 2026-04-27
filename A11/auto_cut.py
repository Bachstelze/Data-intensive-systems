#!/usr/bin/env python3
"""
A11 auto-cutting pipeline.

Input: full uncut Kinect/PoseNet sequence CSV
Output: predicted cut CSV + probability plots + trajectory plots + GIF/MP4 animations

Default paths assume this file is placed inside:
    /Users/reemothman/Downloads/DIS/Data-intensive-systems/A11/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    raise ImportError("Install TensorFlow before running auto_cut.py") from exc

from visualize import (
    plot_probabilities,
    plot_joint_trajectories,
    animate_sequence,
    animate_side_by_side,
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "A11_results"
DEFAULT_INPUT_DIR = BASE_DIR / "A11_kinect_good_preprocessed_not_cut"
CUT_DIR = BASE_DIR / "cut_sequences"
PLOT_DIR = BASE_DIR / "plots"
ANIM_DIR = BASE_DIR / "animations"
METRICS_DIR = BASE_DIR / "metrics"

JOINTS = [
    "head", "left_shoulder", "left_elbow", "right_shoulder", "right_elbow",
    "left_hand", "right_hand", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_foot", "right_foot",
]

KINECT_COLS = [f"{j}_{d}" for j in JOINTS for d in ["x", "y", "z"]]
POSENET_COLS = [f"{j}_{d}" for j in JOINTS for d in ["x", "y"]]

# Must match A11_classifier.py. That file used these engineered columns in addition
# to raw Kinect/PoseNet coordinates.
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

MODEL_CONFIGS = {
    "A": {
        "name": "A_Kinect_LSTM_adam_bs64",
        "arch": "LSTM",
        "weights": RESULTS_DIR / "A_Kinect_LSTM_adam_bs64.weights.h5",
        "scaler": RESULTS_DIR / "A_Kinect_LSTM_adam_bs64_scaler.pkl",
    },
    "B": {
        "name": "B_PoseNet_GRU_rmsprop_bs64",
        "arch": "GRU",
        "weights": RESULTS_DIR / "B_PoseNet_GRU_rmsprop_bs64.weights.h5",
        "scaler": RESULTS_DIR / "B_PoseNet_GRU_rmsprop_bs64_scaler.pkl",
    },
}

LABEL_NAMES = {0: "neutral", 1: "start", 2: "stop"}


@dataclass
class CutResult:
    file_id: str
    modality: str
    input_path: str
    output_cut_path: str
    pred_start_frame: int
    pred_stop_frame: int
    pred_start_probability: float
    pred_stop_probability: float
    true_start_frame: Optional[int] = None
    true_stop_frame: Optional[int] = None
    start_offset: Optional[int] = None
    stop_offset: Optional[int] = None
    cut_length_frames: Optional[int] = None
    probability_plot: Optional[str] = None
    trajectory_plot: Optional[str] = None
    animation: Optional[str] = None
    side_by_side_animation: Optional[str] = None


def build_lstm(input_dim: int, window_size: int = 5) -> keras.Model:
    inputs = keras.Input(shape=(window_size, input_dim), name="input")
    x = layers.LSTM(64, return_sequences=True, dropout=0.2, name="lstm_1")(inputs)
    x = layers.LSTM(32, return_sequences=False, dropout=0.2, name="lstm_2")(x)
    x = layers.Dense(32, activation="relu", name="fc_1")(x)
    x = layers.Dropout(0.2, name="drop_1")(x)
    outputs = layers.Dense(3, activation="softmax", name="output")(x)
    return keras.Model(inputs, outputs, name="LSTM")


def build_gru(input_dim: int, window_size: int = 5) -> keras.Model:
    inputs = keras.Input(shape=(window_size, input_dim), name="input")
    x = layers.GRU(64, return_sequences=True, dropout=0.2, name="gru_1")(inputs)
    x = layers.GRU(32, return_sequences=False, dropout=0.2, name="gru_2")(x)
    x = layers.Dense(32, activation="relu", name="fc_1")(x)
    x = layers.Dropout(0.2, name="drop_1")(x)
    outputs = layers.Dense(3, activation="softmax", name="output")(x)
    return keras.Model(inputs, outputs, name="GRU")


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
    hip_mid_z = (df["left_hip_z"] + df["right_hip_z"]) / 2 if "left_hip_z" in df.columns else 0
    df["head_to_hip"] = np.sqrt(
        (df["head_x"] - hip_mid_x) ** 2 +
        (df["head_y"] - hip_mid_y) ** 2 +
        ((df["head_z"] - hip_mid_z) ** 2 if "head_z" in df.columns else 0)
    )
    return df


def distance(df: pd.DataFrame, a: str, b: str) -> np.ndarray:
    z_part = (df[f"{a}_z"] - df[f"{b}_z"]) ** 2 if f"{a}_z" in df.columns and f"{b}_z" in df.columns else 0
    return np.sqrt((df[f"{a}_x"] - df[f"{b}_x"]) ** 2 + (df[f"{a}_y"] - df[f"{b}_y"]) ** 2 + z_part)


def calculate_velocity_features(df: pd.DataFrame, fps: float = 30.0) -> pd.DataFrame:
    df = df.copy()
    for joint in ["head", "left_hand", "right_hand", "left_elbow", "right_elbow"]:
        for dim in ["x", "y", "z"]:
            col = f"{joint}_{dim}"
            if col in df.columns:
                df[f"{joint}_v{dim}"] = np.diff(df[col], prepend=df[col].iloc[0]) * fps
            else:
                df[f"{joint}_v{dim}"] = 0.0
        df[f"{joint}_speed"] = np.sqrt(df[f"{joint}_vx"] ** 2 + df[f"{joint}_vy"] ** 2 + df[f"{joint}_vz"] ** 2)
    return df


def calculate_acceleration_features(df: pd.DataFrame, fps: float = 30.0) -> pd.DataFrame:
    df = df.copy()
    for joint in ["head", "left_hand", "right_hand", "left_elbow", "right_elbow"]:
        for dim in ["x", "y", "z"]:
            vcol = f"{joint}_v{dim}"
            df[f"{joint}_a{dim}"] = np.diff(df[vcol], prepend=df[vcol].iloc[0]) * fps
        df[f"{joint}_accel"] = np.sqrt(df[f"{joint}_ax"] ** 2 + df[f"{joint}_ay"] ** 2 + df[f"{joint}_az"] ** 2)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = calculate_joint_distances(df)
    df = calculate_velocity_features(df)
    df = calculate_acceleration_features(df)
    return df


def feature_columns_for_modality(df: pd.DataFrame, modality: str) -> List[str]:
    base = KINECT_COLS if modality.upper() == "A" else POSENET_COLS
    cols = [c for c in base if c in df.columns]
    cols += [c for c in EXTRA_COLS if c in df.columns]
    return cols


def make_windows(X: np.ndarray, window_size: int = 5) -> np.ndarray:
    if len(X) < window_size:
        raise ValueError(f"Sequence has {len(X)} frames, but model needs at least {window_size} frames.")
    return np.asarray([X[i:i + window_size] for i in range(len(X) - window_size + 1)], dtype=np.float32)


def smooth_signal(values: np.ndarray, window: int = 7, method: str = "moving_average") -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    s = pd.Series(values)
    if method == "median":
        return s.rolling(window, center=True, min_periods=1).median().to_numpy()
    return s.rolling(window, center=True, min_periods=1).mean().to_numpy()


def pick_start_stop(
    start_prob: np.ndarray,
    stop_prob: np.ndarray,
    frames: np.ndarray,
    min_gap: int = 5,
    top_k: int = 20,
) -> Tuple[int, int, float, float]:
    """
    Pick one start and one stop frame.

    Strategy:
    1. Take the top-k most confident start candidates and stop candidates.
    2. Keep only pairs where start < stop and gap >= min_gap.
    3. Select pair with highest combined confidence.
    4. Fallback: global start peak and best stop after it.
    """
    start_prob = np.asarray(start_prob)
    stop_prob = np.asarray(stop_prob)
    frames = np.asarray(frames)

    start_candidates = np.argsort(start_prob)[-top_k:][::-1]
    stop_candidates = np.argsort(stop_prob)[-top_k:][::-1]

    best = None
    best_score = -np.inf
    for si in start_candidates:
        for ti in stop_candidates:
            if frames[si] + min_gap <= frames[ti]:
                score = float(start_prob[si] + stop_prob[ti])
                if score > best_score:
                    best_score = score
                    best = (si, ti)

    if best is None:
        si = int(np.argmax(start_prob))
        valid_stops = np.where(frames > frames[si])[0]
        ti = int(valid_stops[np.argmax(stop_prob[valid_stops])]) if len(valid_stops) else int(np.argmax(stop_prob))
    else:
        si, ti = best

    return int(frames[si]), int(frames[ti]), float(start_prob[si]), float(stop_prob[ti])


def load_ground_truth(labels_csv: Optional[Path]) -> Dict[str, Tuple[int, int]]:
    if labels_csv is None or not Path(labels_csv).exists():
        return {}
    labels = pd.read_csv(labels_csv)
    labels.columns = labels.columns.str.strip()
    required = {"video_id", "start_frame", "stop_frame"}
    if not required.issubset(labels.columns):
        required = {"file_id", "start_frame", "stop_frame"}
    id_col = "video_id" if "video_id" in labels.columns else "file_id"
    truth = {}
    for _, row in labels.iterrows():
        truth[str(row[id_col])] = (int(row["start_frame"]), int(row["stop_frame"]))
    return truth


def load_model_and_scaler(modality: str, window_size: int = 5):
    cfg = MODEL_CONFIGS[modality.upper()]
    scaler = joblib.load(cfg["scaler"])
    input_dim = int(getattr(scaler, "n_features_in_", len(getattr(scaler, "mean_", []))))
    if cfg["arch"] == "LSTM":
        model = build_lstm(input_dim=input_dim, window_size=window_size)
    elif cfg["arch"] == "GRU":
        model = build_gru(input_dim=input_dim, window_size=window_size)
    else:
        raise ValueError(f"Unsupported architecture: {cfg['arch']}")
    model.load_weights(str(cfg["weights"]))
    return model, scaler, cfg


def predict_probabilities(
    df_raw: pd.DataFrame,
    modality: str,
    model,
    scaler,
    window_size: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    df_feat = add_features(df_raw)
    feat_cols = feature_columns_for_modality(df_feat, modality)
    expected_dim = int(getattr(scaler, "n_features_in_", len(getattr(scaler, "mean_", []))))
    if len(feat_cols) != expected_dim:
        raise ValueError(
            f"Feature mismatch for modality {modality}: built {len(feat_cols)} columns but scaler expects {expected_dim}.\n"
            f"Built columns: {feat_cols}\n"
            "This usually means the saved scaler was trained with a different feature set."
        )

    X = df_feat[feat_cols].to_numpy(dtype=np.float32)
    X_win = make_windows(X, window_size=window_size)
    n, w, f = X_win.shape
    X_scaled = scaler.transform(X_win.reshape(-1, f)).reshape(n, w, f)
    probs_win = model.predict(X_scaled, verbose=0)

    # Align window prediction to the last frame in the window, matching A11_classifier.py.
    probs = np.zeros((len(df_feat), 3), dtype=float)
    probs[window_size - 1:] = probs_win
    frames = df_feat["FrameNo"].to_numpy() if "FrameNo" in df_feat.columns else np.arange(len(df_feat))
    return frames, probs[:, 1], probs[:, 2], df_feat


def cut_one_file(
    input_csv: Path,
    modality: str,
    labels: Dict[str, Tuple[int, int]],
    make_animation: bool = True,
    animation_format: str = "gif",
    smooth_window: int = 7,
    smooth_method: str = "moving_average",
    window_size: int = 5,
    min_gap: int = 5,
) -> CutResult:
    input_csv = Path(input_csv)
    file_id = input_csv.stem
    modality = modality.upper()
    model, scaler, cfg = load_model_and_scaler(modality, window_size=window_size)

    df_raw = pd.read_csv(input_csv)
    df_raw.columns = df_raw.columns.str.strip()
    frames, start_prob, stop_prob, df_feat = predict_probabilities(df_raw, modality, model, scaler, window_size)

    start_smooth = smooth_signal(start_prob, smooth_window, smooth_method)
    stop_smooth = smooth_signal(stop_prob, smooth_window, smooth_method)
    pred_start, pred_stop, p_start, p_stop = pick_start_stop(start_smooth, stop_smooth, frames, min_gap=min_gap)

    frame_mask = (frames >= pred_start) & (frames <= pred_stop)
    df_cut = df_raw.loc[frame_mask].copy()

    CUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    ANIM_DIR.mkdir(parents=True, exist_ok=True)
    out_cut = CUT_DIR / modality / f"{file_id}_{modality}_cut.csv"
    out_cut.parent.mkdir(parents=True, exist_ok=True)
    df_cut.to_csv(out_cut, index=False)

    true_start = true_stop = None
    start_offset = stop_offset = None
    if file_id in labels:
        true_start, true_stop = labels[file_id]
        start_offset = pred_start - true_start
        stop_offset = pred_stop - true_stop

    prob_plot = PLOT_DIR / modality / f"{file_id}_{modality}_probabilities.png"
    traj_plot = PLOT_DIR / modality / f"{file_id}_{modality}_trajectories.png"
    plot_probabilities(
        frames, start_smooth, stop_smooth, pred_start, pred_stop,
        prob_plot, true_start, true_stop,
        title=f"{file_id} | {cfg['name']} | smoothed start/stop probabilities",
    )
    plot_joint_trajectories(df_cut, traj_plot, modality=modality)

    anim_path = None
    side_path = None
    if make_animation:
        suffix = "." + animation_format.lower().lstrip(".")
        anim_path = ANIM_DIR / modality / f"{file_id}_{modality}_cut{suffix}"
        side_path = ANIM_DIR / modality / f"{file_id}_{modality}_side_by_side{suffix}"
        animate_sequence(df_cut, anim_path, modality=modality, pred_start=pred_start, pred_stop=pred_stop, title=f"{file_id} auto-cut")
        animate_side_by_side(df_raw, df_cut, side_path, modality=modality, pred_start=pred_start, pred_stop=pred_stop)

    return CutResult(
        file_id=file_id,
        modality=modality,
        input_path=str(input_csv),
        output_cut_path=str(out_cut),
        pred_start_frame=pred_start,
        pred_stop_frame=pred_stop,
        pred_start_probability=p_start,
        pred_stop_probability=p_stop,
        true_start_frame=true_start,
        true_stop_frame=true_stop,
        start_offset=start_offset,
        stop_offset=stop_offset,
        cut_length_frames=len(df_cut),
        probability_plot=str(prob_plot),
        trajectory_plot=str(traj_plot),
        animation=str(anim_path) if anim_path else None,
        side_by_side_animation=str(side_path) if side_path else None,
    )


def compute_metrics(results: List[CutResult], tolerances: Tuple[int, ...] = (3, 5, 10)) -> pd.DataFrame:
    rows = []
    by_modality = sorted(set(r.modality for r in results))
    for modality in by_modality:
        subset = [r for r in results if r.modality == modality and r.start_offset is not None and r.stop_offset is not None]
        if not subset:
            continue
        start_abs = np.asarray([abs(r.start_offset) for r in subset], dtype=float)
        stop_abs = np.asarray([abs(r.stop_offset) for r in subset], dtype=float)
        row = {
            "modality": modality,
            "n_sequences_with_ground_truth": len(subset),
            "start_offset_mean_abs": float(np.mean(start_abs)),
            "start_offset_median_abs": float(np.median(start_abs)),
            "start_offset_max_abs": float(np.max(start_abs)),
            "stop_offset_mean_abs": float(np.mean(stop_abs)),
            "stop_offset_median_abs": float(np.median(stop_abs)),
            "stop_offset_max_abs": float(np.max(stop_abs)),
        }
        for tol in tolerances:
            both_ok = [(abs(r.start_offset) <= tol and abs(r.stop_offset) <= tol) for r in subset]
            row[f"cut_accuracy_within_{tol}_frames"] = float(np.mean(both_ok) * 100.0)
        rows.append(row)
    return pd.DataFrame(rows)


def save_results(results: List[CutResult]) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame([asdict(r) for r in results])
    result_df.to_csv(METRICS_DIR / "auto_cut_predictions.csv", index=False)
    metrics_df = compute_metrics(results)
    metrics_df.to_csv(METRICS_DIR / "auto_cut_metrics.csv", index=False)
    with open(METRICS_DIR / "auto_cut_predictions.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nSaved predictions: {METRICS_DIR / 'auto_cut_predictions.csv'}")
    print(f"Saved metrics:     {METRICS_DIR / 'auto_cut_metrics.csv'}")
    if not metrics_df.empty:
        print("\nApplication-context accuracy:")
        print(metrics_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A11 trained classifier over uncut sequence(s) and auto-cut movements.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR, help="One CSV file or a directory of CSV files.")
    parser.add_argument("--modality", choices=["A", "B", "both"], default="both", help="A=Kinect 3D, B=PoseNet-style 2D x/y.")
    parser.add_argument("--labels", type=Path, default=BASE_DIR / "labels" / "start_stop_labels.csv", help="Optional CSV with video_id/file_id,start_frame,stop_frame.")
    parser.add_argument("--no-animation", action="store_true", help="Skip GIF/MP4 generation for faster batch runs.")
    parser.add_argument("--animation-format", choices=["gif", "mp4"], default="gif")
    parser.add_argument("--smooth-window", type=int, default=7)
    parser.add_argument("--smooth-method", choices=["moving_average", "median"], default="moving_average")
    parser.add_argument("--window-size", type=int, default=5, help="Must match the trained sequence model window size.")
    parser.add_argument("--min-gap", type=int, default=5, help="Minimum accepted frames between predicted start and stop.")
    return parser.parse_args()


def main():
    args = parse_args()
    labels = load_ground_truth(args.labels)
    modalities = ["A", "B"] if args.modality == "both" else [args.modality]

    if args.input.is_dir():
        files = sorted(args.input.glob("*.csv"))
    else:
        files = [args.input]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {args.input}")

    all_results: List[CutResult] = []
    for modality in modalities:
        print(f"\n=== Running auto-cut for modality {modality} on {len(files)} file(s) ===")
        for csv_path in files:
            print(f"Processing {csv_path.name} ...")
            result = cut_one_file(
                csv_path,
                modality=modality,
                labels=labels,
                make_animation=not args.no_animation,
                animation_format=args.animation_format,
                smooth_window=args.smooth_window,
                smooth_method=args.smooth_method,
                window_size=args.window_size,
                min_gap=args.min_gap,
            )
            all_results.append(result)
            print(f"  predicted start={result.pred_start_frame}, stop={result.pred_stop_frame}, cut={result.output_cut_path}")

    save_results(all_results)


if __name__ == "__main__":
    main()
