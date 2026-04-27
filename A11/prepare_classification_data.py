#!/usr/bin/env python3
"""
Script to prepare data for start/stop classification model.
Compares files in A11_kinect_good_preprocessed_not_cut and kinect_good_preprocessed,
then creates a unified dataset with features for classification.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

# Define paths
BASE_DIR = Path(__file__).parent
NOT_CUT_DIR = BASE_DIR / "A11_kinect_good_preprocessed_not_cut"
CUT_DIR = BASE_DIR / "../kinect_good_preprocessed"
OUTPUT_DIR = BASE_DIR / "classification_data"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_joint_distances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate distances between key joints for motion analysis.
    """
    df = df.copy()

    # Calculate distances between joints
    # Hand-to-shoulder distances
    df['left_hand_to_left_shoulder'] = np.sqrt(
        (df['left_hand_x'] - df['left_shoulder_x'])**2 +
        (df['left_hand_y'] - df['left_shoulder_y'])**2 +
        (df['left_hand_z'] - df['left_shoulder_z'])**2
    )

    df['right_hand_to_right_shoulder'] = np.sqrt(
        (df['right_hand_x'] - df['right_shoulder_x'])**2 +
        (df['right_hand_y'] - df['right_shoulder_y'])**2 +
        (df['right_hand_z'] - df['right_shoulder_z'])**2
    )

    # Hand-to-hip distances
    df['left_hand_to_left_hip'] = np.sqrt(
        (df['left_hand_x'] - df['left_hip_x'])**2 +
        (df['left_hand_y'] - df['left_hip_y'])**2 +
        (df['left_hand_z'] - df['left_hip_z'])**2
    )

    df['right_hand_to_right_hip'] = np.sqrt(
        (df['right_hand_x'] - df['right_hip_x'])**2 +
        (df['right_hand_y'] - df['right_hip_y'])**2 +
        (df['right_hand_z'] - df['right_hip_z'])**2
    )

    # Elbow-to-shoulder distances
    df['left_elbow_to_left_shoulder'] = np.sqrt(
        (df['left_elbow_x'] - df['left_shoulder_x'])**2 +
        (df['left_elbow_y'] - df['left_shoulder_y'])**2 +
        (df['left_elbow_z'] - df['left_shoulder_z'])**2
    )

    df['right_elbow_to_right_shoulder'] = np.sqrt(
        (df['right_elbow_x'] - df['right_shoulder_x'])**2 +
        (df['right_elbow_y'] - df['right_shoulder_y'])**2 +
        (df['right_elbow_z'] - df['right_shoulder_z'])**2
    )

    # Head-to-hip distance (body height proxy)
    df['head_to_hip'] = np.sqrt(
        (df['head_x'] - (df['left_hip_x'] + df['right_hip_x'])/2)**2 +
        (df['head_y'] - (df['left_hip_y'] + df['right_hip_y'])/2)**2 +
        (df['head_z'] - (df['left_hip_z'] + df['right_hip_z'])/2)**2
    )

    return df


def calculate_velocity_features(df: pd.DataFrame, fps: float = 30.0) -> pd.DataFrame:
    """
    Calculate velocity features from position data.
    """
    df = df.copy()

    # Calculate velocities (differences between frames)
    joints = ['head', 'left_hand', 'right_hand', 'left_elbow', 'right_elbow']

    for joint in joints:
        x_col = f'{joint}_x'
        y_col = f'{joint}_y'
        z_col = f'{joint}_z'

        # Velocity components
        df[f'{joint}_vx'] = np.diff(df[x_col], prepend=df[x_col].iloc[0]) * fps
        df[f'{joint}_vy'] = np.diff(df[y_col], prepend=df[y_col].iloc[0]) * fps
        df[f'{joint}_vz'] = np.diff(df[z_col], prepend=df[z_col].iloc[0]) * fps

        # Total velocity
        df[f'{joint}_speed'] = np.sqrt(df[f'{joint}_vx']**2 + df[f'{joint}_vy']**2 + df[f'{joint}_vz']**2)

    return df


def calculate_acceleration_features(df: pd.DataFrame, fps: float = 30.0) -> pd.DataFrame:
    """
    Calculate acceleration features from velocity data.
    """
    df = df.copy()

    joints = ['head', 'left_hand', 'right_hand', 'left_elbow', 'right_elbow']

    for joint in joints:
        vx_col = f'{joint}_vx'
        vy_col = f'{joint}_vy'
        vz_col = f'{joint}_vz'

        # Acceleration components
        df[f'{joint}_ax'] = np.diff(df[vx_col], prepend=df[vx_col].iloc[0]) * fps
        df[f'{joint}_ay'] = np.diff(df[vy_col], prepend=df[vy_col].iloc[0]) * fps
        df[f'{joint}_az'] = np.diff(df[vz_col], prepend=df[vz_col].iloc[0]) * fps

        # Total acceleration
        df[f'{joint}_accel'] = np.sqrt(df[f'{joint}_ax']**2 + df[f'{joint}_ay']**2 + df[f'{joint}_az']**2)

    return df


def create_sliding_window_features(df: pd.DataFrame, window_size: int = 10) -> pd.DataFrame:
    """
    Create sliding window features for temporal analysis.
    """
    df = df.copy()

    joints = ['head', 'left_hand', 'right_hand', 'left_elbow', 'right_elbow']

    for joint in joints:
        speed_col = f'{joint}_speed'

        # Rolling statistics
        df[f'{joint}_speed_mean'] = df[speed_col].rolling(window=window_size, min_periods=1).mean()
        df[f'{joint}_speed_std'] = df[speed_col].rolling(window=window_size, min_periods=1).std().fillna(0)
        df[f'{joint}_speed_max'] = df[speed_col].rolling(window=window_size, min_periods=1).max()
        df[f'{joint}_speed_min'] = df[speed_col].rolling(window=window_size, min_periods=1).min()

    return df


def determine_label(row: pd.Series, next_row: Optional[pd.Series] = None) -> str:
    """
    Determine if a frame represents 'start' or 'stop' based on motion characteristics.

    Start: Significant increase in velocity/acceleration
    Stop: Significant decrease in velocity/acceleration
    """
    # Use head and hand speeds for classification
    head_speed = row['head_speed']
    left_hand_speed = row['left_hand_speed']
    right_hand_speed = row['right_hand_speed']

    # Calculate changes
    if next_row is not None:
        delta_head = next_row['head_speed'] - head_speed
        delta_left_hand = next_row['left_hand_speed'] - left_hand_speed
        delta_right_hand = next_row['right_hand_speed'] - right_hand_speed
    else:
        delta_head = 0
        delta_left_hand = 0
        delta_right_hand = 0

    # Thresholds for classification
    start_threshold = 0.5  # Significant increase in speed
    stop_threshold = -0.5  # Significant decrease in speed

    # Determine label
    if delta_head > start_threshold or delta_left_hand > start_threshold or delta_right_hand > start_threshold:
        return 'start'
    elif delta_head < stop_threshold or delta_left_hand < stop_threshold or delta_right_hand < stop_threshold:
        return 'stop'
    else:
        return 'neutral'


def process_file(file_path: Path, is_not_cut: bool = True) -> pd.DataFrame:
    """
    Process a single CSV file and extract features.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Calculate distances
    df = calculate_joint_distances(df)

    # Calculate velocities
    df = calculate_velocity_features(df)

    # Calculate accelerations
    df = calculate_acceleration_features(df)

    # Create sliding window features
    df = create_sliding_window_features(df)

    # Add file identifier
    df['file_id'] = file_path.stem

    # Add is_not_cut flag
    df['is_not_cut'] = is_not_cut

    return df


def create_classification_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create the main classification dataset from both directories.
    """
    print("Processing files from A11_kinect_good_preprocessed_not_cut...")

    # Process not-cut files
    not_cut_files = sorted(NOT_CUT_DIR.glob("*.csv"))
    not_cut_dfs = []

    for file_path in not_cut_files:
        try:
            df = process_file(file_path, is_not_cut=True)
            not_cut_dfs.append(df)
            print(f"  Processed: {file_path.name} ({len(df)} frames)")
        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")

    # Combine not-cut data
    if not_cut_dfs:
        not_cut_data = pd.concat(not_cut_dfs, ignore_index=True)
    else:
        not_cut_data = pd.DataFrame()

    print(f"\nTotal frames from not-cut files: {len(not_cut_data)}")

    print("\nProcessing files from kinect_good_preprocessed...")

    # Process cut files
    cut_files = sorted(CUT_DIR.glob("*.csv"))
    cut_dfs = []

    for file_path in cut_files:
        try:
            df = process_file(file_path, is_not_cut=False)
            cut_dfs.append(df)
            print(f"  Processed: {file_path.name} ({len(df)} frames)")
        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")

    # Combine cut data
    if cut_dfs:
        cut_data = pd.concat(cut_dfs, ignore_index=True)
    else:
        cut_data = pd.DataFrame()

    print(f"\nTotal frames from cut files: {len(cut_data)}")

    return not_cut_data, cut_data


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add start/stop labels to the dataset.
    """
    df = df.copy()
    df['label'] = 'neutral'

    # Get unique file_ids
    file_ids = df['file_id'].unique()

    for file_id in file_ids:
        mask = df['file_id'] == file_id
        indices = df[mask].index.tolist()

        for i, idx in enumerate(indices):
            if i < len(indices) - 1:
                next_idx = indices[i + 1]
                next_row = df.loc[next_idx]
            else:
                next_row = None

            df.loc[idx, 'label'] = determine_label(df.loc[idx], next_row)

    return df


def main():
    """
    Main function to prepare classification data.
    """
    print("=" * 60)
    print("Start/Stop Classification Data Preparation")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process both datasets
    not_cut_data, cut_data = create_classification_dataset()

    # Add labels to both datasets
    print("\nAdding labels to not-cut data...")
    not_cut_data = add_labels(not_cut_data)

    print("Adding labels to cut data...")
    cut_data = add_labels(cut_data)

    # Save individual datasets
    print("\nSaving datasets...")

    not_cut_data.to_csv(OUTPUT_DIR / "not_cut_classification_data.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'not_cut_classification_data.csv'}")

    cut_data.to_csv(OUTPUT_DIR / "cut_classification_data.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'cut_classification_data.csv'}")

    # Create combined dataset
    combined_data = pd.concat([not_cut_data, cut_data], ignore_index=True)
    combined_data.to_csv(OUTPUT_DIR / "combined_classification_data.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'combined_classification_data.csv'}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    print("\nLabel distribution in not-cut data:")
    print(not_cut_data['label'].value_counts())

    print("\nLabel distribution in cut data:")
    print(cut_data['label'].value_counts())

    print("\nLabel distribution in combined data:")
    print(combined_data['label'].value_counts())

    print("\n" + "=" * 60)
    print("Feature columns available:")
    print("=" * 60)
    print(combined_data.columns.tolist())

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
