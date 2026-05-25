#!/usr/bin/env python3
"""
Dataset Augmentation Script for A15 Cut Skeleton Sequences

Applies the following augmentations to each cut sequence CSV in a15_cut/:
1. Mirror on y-axis (flip x-coordinates left ↔ right)
2. Rotate on y-axis by ±10 degrees
3. Stretch/compress a few % in x, y, z axes

Each augmented exercise retains the same quality score as the original.

The cut CSVs have columns: FrameNo, head_x, head_y, head_z, ... (13 joints × 3 coords)
Each row is one frame; the full file is one exercise sequence.

Output:
  - a15_cut_augmented/  directory with all original + augmented CSV files
  - a15_augmented_data.csv  combined CSV with clip names, scores, probabilities

Usage:
    python3 augment_cut_data.py
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import List, Tuple


# ── coordinate helpers ──────────────────────────────────────────────────────────

def get_coordinate_columns(df: pd.DataFrame) -> Tuple[List[int], List[int], List[int]]:
    """
    After FrameNo (column 0), the columns follow the pattern
      joint1_x, joint1_y, joint1_z, joint2_x, joint2_y, joint2_z, …
    so (col_index - 1) % 3 tells us the axis.

    Returns:
        (x_indices, y_indices, z_indices) — column indices (0-based) into the DataFrame.
    """
    coord_cols = list(range(1, len(df.columns)))          # skip FrameNo
    x_idx = [c for c in coord_cols if (c - 1) % 3 == 0]
    y_idx = [c for c in coord_cols if (c - 1) % 3 == 1]
    z_idx = [c for c in coord_cols if (c - 1) % 3 == 2]
    return x_idx, y_idx, z_idx


# ── augmentation primitives ────────────────────────────────────────────────────

def mirror_on_y_axis(df: pd.DataFrame, x_idx: List[int]) -> pd.DataFrame:
    """Mirror left ↔ right by negating all x-coordinates."""
    df_aug = df.copy()
    for idx in x_idx:
        df_aug.iloc[:, idx] = -df.iloc[:, idx]
    return df_aug


def rotate_on_y_axis(df: pd.DataFrame, x_idx: List[int], z_idx: List[int],
                     angle_deg: float) -> pd.DataFrame:
    """
    Rotate every joint around the y-axis.

        x' = x·cos(θ) + z·sin(θ)
        y' = y
        z' = –x·sin(θ) + z·cos(θ)
    """
    df_aug = df.copy()
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    for xi, zi in zip(x_idx, z_idx):
        x_orig = df.iloc[:, xi].values
        z_orig = df.iloc[:, zi].values
        df_aug.iloc[:, xi] = x_orig * cos_a + z_orig * sin_a
        df_aug.iloc[:, zi] = -x_orig * sin_a + z_orig * cos_a

    return df_aug


def stretch_compress(df: pd.DataFrame, x_idx: List[int], y_idx: List[int],
                     z_idx: List[int], scale_x: float, scale_y: float,
                     scale_z: float) -> pd.DataFrame:
    """Scale coordinates independently per axis."""
    df_aug = df.copy()
    for idx in x_idx:
        df_aug.iloc[:, idx] *= scale_x
    for idx in y_idx:
        df_aug.iloc[:, idx] *= scale_y
    for idx in z_idx:
        df_aug.iloc[:, idx] *= scale_z
    return df_aug


# ── I/O helpers ─────────────────────────────────────────────────────────────────

def load_score_map(score_file: str) -> dict:
    """Return {clip_name: (score_rescaled, good_probability)} from a15_good_rescaled.csv."""
    df_scores = pd.read_csv(score_file)
    return {
        row['clip']: (row['score_rescaled'], row['good_probability'])
        for _, row in df_scores.iterrows()
    }


def augment_all_sequences(input_dir: str, score_file: str,
                          output_dir: str, output_csv: str) -> None:
    # Load scores
    print(f"Loading scores from {score_file} …")
    score_map = load_score_map(score_file)
    print(f"  → {len(score_map)} clips with scores")

    # Discover cut sequence CSVs
    csv_pattern = os.path.join(input_dir, '*_kinect.csv')
    csv_files = sorted(glob.glob(csv_pattern))
    n_original = len(csv_files)
    if n_original == 0:
        print(f"ERROR: No *_kinect.csv files found in {input_dir}.")
        return
    print(f"Found {n_original} original sequence files")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Augmentation recipes → (suffix, callable)
    augmentations = [
        ('mirror',      lambda df, x, y, z: mirror_on_y_axis(df, x)),
        ('rotate_pos',  lambda df, x, y, z: rotate_on_y_axis(df, x, z,  10)),
        ('rotate_neg',  lambda df, x, y, z: rotate_on_y_axis(df, x, z, -10)),
        ('stretch',     lambda df, x, y, z: stretch_compress(df, x, y, z, 1.05, 0.95, 1.02)),
    ]

    all_entries: List[Tuple[str, float, float]] = []   # (clip, score, prob)
    total_augmented = 0

    for csv_path in csv_files:
        basename = os.path.basename(csv_path)                     # e.g. "A100_kinect.csv"
        clip_name = basename.replace('.csv', '')                  # e.g. "A100_kinect"

        # Skip if no score is known (should not happen for the good list)
        if clip_name not in score_map:
            print(f"  ⚠ WARNING: no score for {clip_name}, skipping")
            continue

        score_val, prob_val = score_map[clip_name]

        # Read the frame-by-frame skeleton data
        df = pd.read_csv(csv_path)
        x_idx, y_idx, z_idx = get_coordinate_columns(df)

        # ── 1. Keep original ──
        orig_path = os.path.join(output_dir, basename)
        df.to_csv(orig_path, index=False)
        all_entries.append((clip_name, score_val, prob_val))

        # ── 2–5. Augmented variants ──
        for suffix, aug_fn in augmentations:
            df_aug = aug_fn(df.copy(), x_idx, y_idx, z_idx)
            aug_clip_name = f"{clip_name}_{suffix}"
            aug_filename = f"{aug_clip_name}.csv"
            df_aug.to_csv(os.path.join(output_dir, aug_filename), index=False)
            all_entries.append((aug_clip_name, score_val, prob_val))
            total_augmented += 1

        if (len(csv_files) <= 10 or csv_files.index(csv_path) % 20 == 0):
            print(f"  ✓ {clip_name} → original + 4 augmented variants")

    # ── Write combined metadata CSV ──
    df_out = pd.DataFrame(all_entries, columns=['clip', 'score_rescaled', 'good_probability'])
    df_out.to_csv(output_csv, index=False)

    print(f"\n{'=' * 50}")
    print(f"  Augmentation complete")
    print(f"  Original clips          : {n_original}")
    print(f"  Augmented variants      : {total_augmented}")
    print(f"  Total (orig + augment)  : {len(df_out)}")
    print(f"  Augmented CSV directory : {output_dir}/")
    print(f"  Combined metadata CSV   : {output_csv}")
    print(f"{'=' * 50}")


# ── entry point ─────────────────────────────────────────────────────────────────

def main():
    # Paths relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_dir   = os.path.join(script_dir, 'a15_cut')
    score_file  = os.path.join(script_dir, 'a15_good_rescaled.csv')
    output_dir  = os.path.join(script_dir, 'a15_cut_augmented')
    output_csv  = os.path.join(script_dir, 'a15_augmented_data.csv')

    augment_all_sequences(
        input_dir=input_dir,
        score_file=score_file,
        output_dir=output_dir,
        output_csv=output_csv,
    )


if __name__ == '__main__':
    main()
