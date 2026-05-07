#!/usr/bin/env python3
"""
Dataset Augmentation Script for Processed Skeleton-based Classification Data

This script applies the following augmentations to the processed classification data:
1. Mirror on y-axis (flip left/right)
2. Rotate on y-axis by a few degrees
3. Stretch/compress a few % in x, y, z axes

The script only augments original datapoints, not generated ones.

Usage:
    python3 augment_processed_data.py --input <input_csv> --output <output_csv>
"""

import argparse
import pandas as pd
import numpy as np
from typing import List


def get_coordinate_indices(df: pd.DataFrame) -> List[int]:
    """
    Get indices for coordinate values in the dataframe.
    The processed data has 1020 features per row (10 frames x 102 features),
    preceded by 'filename' and 'label' columns.

    Args:
        df: Input dataframe

    Returns:
        List of indices corresponding to coordinate values
    """
    # Skip the first 2 columns (filename, label) to get to the coordinate data
    start_idx = 2
    end_idx = min(len(df.columns), 1022)  # 2 (filename, label) + 1020 (features)
    return list(range(start_idx, end_idx))


def get_frame_indices() -> List[List[int]]:
    """
    Get the indices for each frame in the sequence.
    Each sequence has 10 frames with 102 features per frame.

    Returns:
        List of lists, where each inner list contains the indices for one frame
    """
    frame_indices = []
    # Start from index 2 to skip filename and label columns
    for frame_idx in range(10):  # 10 frames per sequence
        start = 2 + (frame_idx * 102)  # Skip filename and label (indices 0, 1)
        end = 2 + ((frame_idx + 1) * 102)
        frame_indices.append(list(range(start, end)))
    return frame_indices


def identify_original_samples(df: pd.DataFrame) -> pd.Series:
    """
    Identify original samples (not augmented ones) based on filename patterns.

    Args:
        df: Input dataframe with 'filename' column

    Returns:
        Boolean Series indicating which rows are original samples
    """
    # Original samples have simple names like G01, W01, A1, etc.
    # Augmented samples would have suffixes like _mirror, _rotate, etc.
    original_mask = ~df['filename'].str.contains(r'_mirror|_rotate|_stretch|_neg', na=False)
    return original_mask


def mirror_on_y_axis(df: pd.DataFrame, coord_indices: List[int]) -> pd.DataFrame:
    """
    Mirror the skeleton on the y-axis by flipping x-coordinates.
    This assumes coordinates are arranged in x, y, z groups throughout the sequence.

    Args:
        df: Input dataframe
        coord_indices: List of indices for coordinate values

    Returns:
        Mirrored dataframe
    """
    df_augmented = df.copy()

    # In skeleton data, coordinates typically follow an x, y, z pattern
    # So every third coordinate starting from the first coordinate is an x-value
    # Since we start from index 2 (after filename and label), the first coordinate is at index 2
    # Then we have x, y, z at indices 2, 3, 4; then x, y, z at indices 5, 6, 7; etc.

    # Find x-coordinate positions (every third index starting from the first coordinate position)
    for i in range(0, len(coord_indices), 3):  # Every third coordinate is x
        x_idx = coord_indices[i]
        if x_idx < df.shape[1]:
            df_augmented.iloc[:, x_idx] = -df.iloc[:, x_idx]

    return df_augmented


def rotate_on_y_axis(df: pd.DataFrame, frame_indices: List[List[int]],
                     angle_deg: float) -> pd.DataFrame:
    """
    Rotate the skeleton around the y-axis by a given angle.
    This assumes coordinates are arranged in x, y, z groups.

    Args:
        df: Input dataframe
        frame_indices: List of indices for each frame
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

    # Apply rotation to each frame
    for frame_idx_list in frame_indices:
        # Process every group of 3 coordinates (x, y, z) in this frame
        for i in range(0, len(frame_idx_list), 3):
            if i + 2 < len(frame_idx_list):  # Ensure we have x, y, z indices
                x_idx = frame_idx_list[i]
                y_idx = frame_idx_list[i + 1]
                z_idx = frame_idx_list[i + 2]

                if x_idx < df.shape[1] and y_idx < df.shape[1] and z_idx < df.shape[1]:
                    # Store original values
                    x_orig = df.iloc[:, x_idx].values
                    y_orig = df.iloc[:, y_idx].values
                    z_orig = df.iloc[:, z_idx].values

                    # Apply rotation
                    df_augmented.iloc[:, x_idx] = x_orig * cos_a + z_orig * sin_a
                    df_augmented.iloc[:, z_idx] = -x_orig * sin_a + z_orig * cos_a
                    # y remains unchanged

    return df_augmented


def stretch_compress(df: pd.DataFrame, frame_indices: List[List[int]],
                     scale_x: float, scale_y: float, scale_z: float) -> pd.DataFrame:
    """
    Apply scaling/stretching to the skeleton data.
    This assumes coordinates are arranged in x, y, z groups.

    Args:
        df: Input dataframe
        frame_indices: List of indices for each frame
        scale_x: Scale factor for x-axis (e.g., 1.05 = 5% stretch)
        scale_y: Scale factor for y-axis
        scale_z: Scale factor for z-axis

    Returns:
        Scaled dataframe
    """
    df_augmented = df.copy()

    # Apply scaling to each frame
    for frame_idx_list in frame_indices:
        # Process every group of 3 coordinates (x, y, z) in this frame
        for i in range(0, len(frame_idx_list), 3):
            if i + 2 < len(frame_idx_list):  # Ensure we have x, y, z indices
                x_idx = frame_idx_list[i]
                y_idx = frame_idx_list[i + 1]
                z_idx = frame_idx_list[i + 2]

                if x_idx < df.shape[1]:
                    df_augmented.iloc[:, x_idx] *= scale_x
                if y_idx < df.shape[1]:
                    df_augmented.iloc[:, y_idx] *= scale_y
                if z_idx < df.shape[1]:
                    df_augmented.iloc[:, z_idx] *= scale_z

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

    # Identify original samples only (not previously augmented ones)
    original_mask = identify_original_samples(df)
    df_original = df[original_mask].copy()
    print(f"Found {len(df_original)} original samples to augment")

    # Get coordinate indices and frame structure
    coord_indices = get_coordinate_indices(df_original)
    frame_indices = get_frame_indices()

    # Define augmentation configurations
    # 1. Mirror on y-axis
    print("\n1. Applying mirror on y-axis...")
    df_mirror = mirror_on_y_axis(df_original.copy(), coord_indices)
    df_mirror['filename'] = df_original['filename'].astype(str) + '_mirror'

    # 2. Rotate on y-axis by +10 degrees
    print("2. Applying y-axis rotation (+10 degrees)...")
    df_rotate_pos = rotate_on_y_axis(df_original.copy(), frame_indices, 10)
    df_rotate_pos['filename'] = df_original['filename'].astype(str) + '_rotate_pos'

    # 3. Rotate on y-axis by -10 degrees
    print("3. Applying y-axis rotation (-10 degrees)...")
    df_rotate_neg = rotate_on_y_axis(df_original.copy(), frame_indices, -10)
    df_rotate_neg['filename'] = df_original['filename'].astype(str) + '_rotate_neg'

    # 4. Stretch/compress in x, y, z axes
    print("4. Applying stretch/compress (x: +5%, y: -5%, z: +2%)...")
    df_stretch = stretch_compress(df_original.copy(), frame_indices, 1.05, 0.95, 1.02)
    df_stretch['filename'] = df_original['filename'].astype(str) + '_stretch'

    # Combine all augmented data with original
    df_combined = pd.concat([
        df_original,    # Original
        df_mirror,      # Mirror
        df_rotate_pos,  # Rotate +10
        df_rotate_neg,  # Rotate -10
        df_stretch      # Stretch
    ], ignore_index=True)

    print(f"\n=== Summary ===")
    print(f"Original samples: {len(df_original)}")
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
    parser = argparse.ArgumentParser(description='Dataset Augmentation for Processed Skeleton Data')
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
