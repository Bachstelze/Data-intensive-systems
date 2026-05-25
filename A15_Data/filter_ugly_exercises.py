#!/usr/bin/env python3
"""
Connect scores.csv with Datasets_all/kinect_good_preprocessed, preprocess the
data, and filter ugly exercises using the task-14 A_CNN model.

Outputs to A15/:
  - a15_predictions.csv   — per-clip scores + model predictions
  - a15_ugly_list.csv     — only the clips classified as ugly
  - a15_good_list.csv     — only the clips classified as good
  - a15_summary.txt       — text summary
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

# --------------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent
REPO = HERE.parent

SCORES_CSV       = HERE / "scores.csv"
KINECT_DIR       = REPO / "Datasets_all" / "kinect_good_preprocessed"
POSENET_DIR      = REPO / "Datasets_all" / "posenet_data"
MODEL_PATH       = REPO / "models" / "A_CNN.keras"
OUT_DIR          = HERE

OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------- consts
FRAMES  = 10    # model expects exactly 10 time steps
JOINTS  = 13    # head + 6 upper-body + 6 lower-body
DIMS    = 3     # x, y, z


# --------------------------------------------------------------- preprocessing
def load_clip(csv_path: Path) -> np.ndarray:
    """Load a Kinect CSV and return (FRAMES, JOINTS, DIMS) float32 array.

    The CSV has columns: FrameNo, head_x,head_y,head_z, ..., right_foot_z
    (39 coordinate columns = 13 joints × 3 coordinates).
    Uses the same subsampling logic as prepare_classification_data_v2.py.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "FrameNo" not in df.columns:
        raise ValueError(f"{csv_path.name}: expected a FrameNo column")

    coords = df.drop(columns=["FrameNo"]).values.astype("float32")
    n_rows, n_cols = coords.shape

    if n_cols != JOINTS * DIMS:
        raise ValueError(
            f"{csv_path.name}: expected {JOINTS * DIMS} coord cols, got {n_cols}"
        )

    # Equidistant subsample to FRAMES; if shorter, pad with last frame.
    if n_rows >= FRAMES:
        idx = np.linspace(0, n_rows - 1, FRAMES, dtype=int)
        seq = coords[idx]
    else:
        seq = np.zeros((FRAMES, n_cols), dtype="float32")
        seq[:n_rows] = coords
        if n_rows > 0:
            seq[n_rows:] = coords[-1]

    return seq.reshape(FRAMES, JOINTS, DIMS)


def preprocess(X: np.ndarray) -> np.ndarray:
    """Preprocess the input for CNN inference: add batch dim.

    Input:  (FRAMES, JOINTS, DIMS)  shape (10, 13, 3)
    Output: (1, FRAMES, JOINTS, DIMS) shape (1, 10, 13, 3)
    """
    return X[np.newaxis, ...].astype("float32")


# --------------------------------------------------------------------- main
def main() -> None:
    print("=" * 60)
    print("A15 — Connect scores.csv + Datasets_all, filter ugly exercises")
    print("=" * 60)

    # --- 1. Load scores ----------------------------------------------------
    scores_df = pd.read_csv(SCORES_CSV)
    scores_df.columns = [c.strip() for c in scores_df.columns]
    # Rename for clarity
    scores_df = scores_df.rename(columns={"Var1": "clip", "Var2": "score"})
    # Remove .csv suffix if present
    scores_df["clip"] = scores_df["clip"].str.replace(r"\.csv$", "", regex=True)
    print(f"\n[1] Loaded scores: {len(scores_df)} clips")

    # --- 2. Load the model -------------------------------------------------
    print(f"\n[2] Loading A_CNN model from {MODEL_PATH}")
    if not MODEL_PATH.exists():
        print(f"  ERROR: model not found at {MODEL_PATH}")
        sys.exit(1)
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print(f"  Model input shape: {model.input_shape}")
    print(f"  Model output shape: {model.output_shape}")

    # --- 3. Process each clip -----------------------------------------------
    print(f"\n[3] Processing Kinect clips from {KINECT_DIR}")
    results: list[dict] = []
    missing_kinect: list[str] = []

    kinect_files = {f.stem for f in sorted(KINECT_DIR.glob("*.csv"))}
    print(f"  Found {len(kinect_files)} Kinect files in Datasets_all")

    for _, row in scores_df.iterrows():
        clip_name = row["clip"]
        kinect_path = KINECT_DIR / f"{clip_name}.csv"

        if not kinect_path.exists():
            missing_kinect.append(clip_name)
            continue

        try:
            seq = load_clip(kinect_path)
            X = preprocess(seq)

            # Predict: model outputs sigmoid probability of GOOD
            proba = float(model.predict(X, verbose=0).reshape(-1)[0])
            label = "GOOD" if proba >= 0.5 else "UGLY"

            results.append({
                "clip":            clip_name,
                "score":           row["score"],
                "good_probability": round(proba, 4),
                "predicted_label": label,
            })
        except Exception as e:
            print(f"  ERROR processing {clip_name}: {e}")

    if missing_kinect:
        print(f"  WARNING: {len(missing_kinect)} clips in scores.csv but "
              f"no Kinect file: {missing_kinect}")

    # --- 4. Assemble output ------------------------------------------------
    out_df = pd.DataFrame(results)
    if len(out_df) == 0:
        print("\n  No clips processed! Exiting.")
        sys.exit(1)

    out_df = out_df.sort_values("clip").reset_index(drop=True)

    # --- 5. Separate good vs ugly -----------------------------------------
    good_df = out_df[out_df.predicted_label == "GOOD"].copy()
    ugly_df = out_df[out_df.predicted_label == "UGLY"].copy()

    # --- 6. Write CSV outputs ---------------------------------------------
    out_path   = OUT_DIR / "a15_predictions.csv"
    good_path  = OUT_DIR / "a15_good_list.csv"
    ugly_path  = OUT_DIR / "a15_ugly_list.csv"
    summary_path = OUT_DIR / "a15_summary.txt"

    out_df.to_csv(out_path, index=False)
    good_df.to_csv(good_path, index=False)
    ugly_df.to_csv(ugly_path, index=False)

    print(f"\n[4] Results:")
    print(f"  Total clips processed: {len(out_df)}")
    print(f"  Predicted GOOD:        {len(good_df)}")
    print(f"  Predicted UGLY:        {len(ugly_df)}")

    # --- 7. Write summary -------------------------------------------------
    with open(summary_path, "w") as f:
        f.write("A15 — Connect scores.csv + Datasets_all, filter ugly exercises\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total clips in scores.csv:  {len(scores_df)}\n")
        f.write(f"Kinect files found:         {len(kinect_files)}\n")
        f.write(f"Successfully processed:     {len(out_df)}\n")
        f.write(f"Missing Kinect files:       {len(missing_kinect)}\n\n")
        f.write(f"Model predicted GOOD:       {len(good_df)}\n")
        f.write(f"Model predicted UGLY:       {len(ugly_df)}\n\n")
        f.write(f"Output files:\n")
        f.write(f"  {out_path}\n")
        f.write(f"  {good_path}\n")
        f.write(f"  {ugly_path}\n\n")

        if not ugly_df.empty:
            f.write("UGLY exercises (scored, predicted bad form):\n")
            for _, r in ugly_df.iterrows():
                f.write(f"  {r['clip']:20s}  "
                        f"score={r['score']:.4f}  "
                        f"P(good)={r['good_probability']:.4f}\n")
            f.write(f"\nMean score of UGLY exercises: {ugly_df.score.mean():.4f}\n")

        if not good_df.empty:
            f.write(f"\nMean score of GOOD exercises: {good_df.score.mean():.4f}\n")

    print(f"\n[5] Output written to {OUT_DIR}")
    print(f"  Predictions : {out_path.name}")
    print(f"  Good list   : {good_path.name}")
    print(f"  Ugly list   : {ugly_path.name}")
    print(f"  Summary     : {summary_path.name}")
    print("Done.")


if __name__ == "__main__":
    main()
