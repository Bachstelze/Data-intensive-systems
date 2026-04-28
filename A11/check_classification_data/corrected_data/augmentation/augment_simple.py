#!/usr/bin/env python3
"""
Simple Dataset Augmentation Script for Skeleton-based Classification Data

This script applies the following augmentations to the filtered classification data:
1. Mirror on y-axis (flip left/right)
2. Rotate on y-axis by a few degrees
3. Stretch/compress a few % in x, y, z axes

Usage:
    python3 augment_simple.py --input <input_csv> --output <output_csv>
"""

import argparse
import pandas as pd
import numpy as np
from typing import List, Tuple


def get_skeleton_columns(df_columns: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Get column names for different types of data based on actual dataframe columns.

    Args:
        df_columns: List of column names from the dataframe

    Returns:
        Tuple of (position_columns, velocity_columns, acceleration_columns)
    """
    # Joint names
    joints = [
        'head', 'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow',
        'left_hand', 'right_hand', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_foot', 'right_foot'
    ]

    # Position columns (x, y, z for each joint)
    position_columns = [col for col in df_columns if any(joint in col for joint in joints) and col.endswith(('_x', '_y', '_z'))]

    # Velocity columns
    velocity_columns = [col for col in df_columns if any(joint in col for joint in joints) and col.endswith(('_vx', '_vy', '_vz'))]
    velocity_columns.extend([col for col in df_columns if 'speed' in col and col not in velocity_columns])

    # Acceleration columns
    acceleration_columns = [col for col in df_columns if any(joint in col for joint in joints) and col.endswith(('_ax', '_ay', '_az'))]
    acceleration_columns.extend([col for col in df_columns if 'accel' in col and col not in acceleration_columns])

    return position_columns, velocity_columns, acceleration_columns


def mirror_on_y_axis(df: pd.DataFrame, position_cols: List[str]) -> pd.DataFrame:
    """
    Mirror the skeleton on the y-axis by flipping x-coordinates and swapping left/right.

    Args:
        df: Input dataframe
        position_cols: List of position column names

    Returns:
        Mirrored dataframe
    """
    df_augmented = df.copy()

    # Flip x-coordinates (mirror on y-axis)
    for col in position_cols:
        if col.endswith('_x'):
            df_augmented[col] = -df_augmented[col]

    # Swap left and right joint positions
    left_right_pairs = [
        ('left_shoulder', 'right_shoulder'),
        ('left_elbow', 'right_elbow'),
        ('left_hand', 'right_hand'),
        ('left_hip', 'right_hip'),
        ('left_knee', 'right_knee'),
        ('left_foot', 'right_foot')
    ]

    for left, right in left_right_pairs:
        # Swap x coordinates
        if f'{left}_x' in df.columns and f'{right}_x' in df.columns:
            df_augmented[f'{left}_x'], df_augmented[f'{right}_x'] = df[f'{right}_x'].values, df[f'{left}_x'].values
        # Swap y coordinates
        if f'{left}_y' in df.columns and f'{right}_y' in df.columns:
            df_augmented[f'{left}_y'], df_augmented[f'{right}_y'] = df[f'{right}_y'].values, df[f'{left}_y'].values
        # Swap z coordinates
        if f'{left}_z' in df.columns and f'{right}_z' in df.columns:
            df_augmented[f'{left}_z'], df_augmented[f'{right}_z'] = df[f'{right}_z'].values, df[f'{left}_z'].values

    # Swap velocity components (only for joints that have velocity data)
    left_right_velocity_pairs = [
        ('left_elbow', 'right_elbow'),
        ('left_hand', 'right_hand'),
    ]

    for left, right in left_right_velocity_pairs:
        # Swap velocity components
        if f'{left}_vx' in df.columns and f'{right}_vx' in df.columns:
            df_augmented[f'{left}_vx'], df_augmented[f'{right}_vx'] = df[f'{right}_vx'].values, df[f'{left}_vx'].values
        if f'{left}_vy' in df.columns and f'{right}_vy' in df.columns:
            df_augmented[f'{left}_vy'], df_augmented[f'{right}_vy'] = df[f'{right}_vy'].values, df[f'{left}_vy'].values
        if f'{left}_vz' in df.columns and f'{right}_vz' in df.columns:
            df_augmented[f'{left}_vz'], df_augmented[f'{right}_vz'] = df[f'{right}_vz'].values, df[f'{left}_vz'].values

    # Swap acceleration components (only for joints that have acceleration data)
    for left, right in left_right_velocity_pairs:
        # Swap acceleration components
        if f'{left}_ax' in df.columns and f'{right}_ax' in df.columns:
            df_augmented[f'{left}_ax'], df_augmented[f'{right}_ax'] = df[f'{right}_ax'].values, df[f'{left}_ax'].values
        if f'{left}_ay' in df.columns and f'{right}_ay' in df.columns:
            df_augmented[f'{left}_ay'], df_augmented[f'{right}_ay'] = df[f'{right}_ay'].values, df[f'{left}_ay'].values
        if f'{left}_az' in df.columns and f'{right}_az' in df.columns:
            df_augmented[f'{left}_az'], df_augmented[f'{right}_az'] = df[f'{right}_az'].values, df[f'{left}_az'].values

    # Swap distance features
    distance_pairs = [
        ('left_hand_to_left_shoulder', 'right_hand_to_right_shoulder'),
        ('left_hand_to_left_hip', 'right_hand_to_right_hip'),
        ('left_elbow_to_left_shoulder', 'right_elbow_to_right_shoulder')
    ]

    for left, right in distance_pairs:
        if left in df.columns and right in df.columns:
            df_augmented[left], df_augmented[right] = df[right].values, df[left].values

    return df_augmented


def rotate_on_y_axis(df: pd.DataFrame, position_cols: List[str],
                     angle_deg: float) -> pd.DataFrame:
    """
    Rotate the skeleton around the y-axis by a given angle.

    Args:
        df: Input dataframe
        position_cols: List of position column names
        angle_deg: Rotation angle in degrees (positive = counter-clockwise)

    Returns:
        Rotated dataframe
    """
    df_augmented = df.copy()
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Rotation matrix for y-axis:
    # x' = x*cos(θ) + z*sin(θ)
    # y' = y
    # z' = -x*sin(θ) + z*cos(θ)

    # Apply rotation to position coordinates
    for col in position_cols:
        if col.endswith('_x'):
            joint_name = col[:-2]  # Remove '_x' to get joint name
            x_col = f'{joint_name}_x'
            y_col = f'{joint_name}_y'
            z_col = f'{joint_name}_z'

            if x_col in df_augmented.columns and z_col in df_augmented.columns:
                # Store original values
                x_orig = df_augmented[x_col].values
                z_orig = df_augmented[z_col].values

                # Apply rotation
                df_augmented[x_col] = x_orig * cos_a + z_orig * sin_a
                df_augmented[z_col] = -x_orig * sin_a + z_orig * cos_a

    # Also rotate velocity and acceleration vectors
    for suffix in ['_vx', '_vy', '_vz', '_ax', '_ay', '_az']:
        for col in position_cols:
            if col.endswith(suffix):
                joint_name = col[:-len(suffix)]
                if suffix in ['_vx', '_ax']:
                    # Velocity/acceleration x component
                    x_col = f'{joint_name}{suffix.replace("_v", "_x").replace("_a", "_x")}'
                    if x_col in df_augmented.columns:
                        z_col = f'{joint_name}{suffix.replace("_vx", "_vz").replace("_ax", "_az")}'
                        if z_col in df_augmented.columns:
                            x_orig = df_augmented[x_col].values
                            z_orig = df_augmented[z_col].values
                            df_augmented[x_col] = x_orig * cos_a + z_orig * sin_a
                            df_augmented[z_col] = -x_orig * sin_a + z_orig * cos_a

    return df_augmented


def stretch_compress(df: pd.DataFrame, position_cols: List[str],
                     scale_x: float, scale_y: float, scale_z: float) -> pd.DataFrame:
    """
    Apply scaling/stretching to the skeleton data.

    Args:
        df: Input dataframe
        position_cols: List of position column names
        scale_x: Scale factor for x-axis (e.g., 1.05 = 5% stretch)
        scale_y: Scale factor for y-axis
        scale_z: Scale factor for z-axis

    Returns:
        Scaled dataframe
    """
    df_augmented = df.copy()

    # Apply scaling to position coordinates
    for col in position_cols:
        if col.endswith('_x'):
            df_augmented[col] *= scale_x
        elif col.endswith('_y'):
            df_augmented[col] *= scale_y
        elif col.endswith('_z'):
            df_augmented[col] *= scale_z

    # Velocity and acceleration are affected by scaling
    for suffix in ['_vx', '_vy', '_vz', '_ax', '_ay', '_az']:
        for col in position_cols:
            if col.endswith(suffix):
                joint_name = col[:-len(suffix)]
                if suffix in ['_vx', '_ax']:
                    df_augmented[col] *= scale_x
                elif suffix in ['_vy', '_ay']:
                    df_augmented[col] *= scale_y
                elif suffix in ['_vz', '_az']:
                    df_augmented[col] *= scale_z

    return df_augmented


def generate_augmented_dataset(input_file: str, output_file: str) -> None:
    """
    Generate an augmented dataset from the input file.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")

    position_cols, velocity_cols, accel_cols = get_skeleton_columns(df.columns)
    print(f"Position columns: {len(position_cols)}")
    print(f"Velocity columns: {len(velocity_cols)}")
    print(f"Acceleration columns: {len(accel_cols)}")

    # Define augmentation configurations
    # 1. Mirror on y-axis
    print("\n1. Applying mirror on y-axis...")
    df_mirror = mirror_on_y_axis(df, position_cols)
    df_mirror['file_id'] = df_mirror['file_id'].astype(str) + '_mirror'

    # 2. Rotate on y-axis by +10 degrees
    print("2. Applying y-axis rotation (+10 degrees)...")
    df_rotate_pos = rotate_on_y_axis(df, position_cols, 10)
    df_rotate_pos['file_id'] = df_rotate_pos['file_id'].astype(str) + '_rotate_pos'

    # 3. Rotate on y-axis by -10 degrees
    print("3. Applying y-axis rotation (-10 degrees)...")
    df_rotate_neg = rotate_on_y_axis(df, position_cols, -10)
    df_rotate_neg['file_id'] = df_rotate_neg['file_id'].astype(str) + '_rotate_neg'

    # 4. Stretch/compress in x, y, z axes
    print("4. Applying stretch/compress (x: +5%, y: -5%, z: +2%)...")
    df_stretch = stretch_compress(df, position_cols, 1.05, 0.95, 1.02)
    df_stretch['file_id'] = df_stretch['file_id'].astype(str) + '_stretch'

    # Combine all augmented data with original
    df_combined = pd.concat([
        df,           # Original
        df_mirror,    # Mirror
        df_rotate_pos, # Rotate +10
        df_rotate_neg, # Rotate -10
        df_stretch    # Stretch
    ], ignore_index=True)

    print(f"\n=== Summary ===")
    print(f"Original samples: {len(df)}")
    print(f"Mirror samples: {len(df_mirror)}")
    print(f"Rotate +10 samples: {len(df_rotate_pos)}")
    print(f"Rotate -10 samples: {len(df_rotate_neg)}")
    print(f"Stretch samples: {len(df_stretch)}")
    print(f"Total samples: {len(df_combined)}")

    # Save to CSV
    print(f"\nSaving to {output_file}...")
    df_combined.to_csv(output_file, index=False)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Dataset Augmentation for Skeleton Data')
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file path')

    args = parser.parse_args()

    generate_augmented_dataset(
        input_file=args.input,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
