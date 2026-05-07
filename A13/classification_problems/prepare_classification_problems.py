#!/usr/bin/env python3
"""
Script to prepare data for 2 classification problems:
- Problem A (3D): Kinect frame sequence: 13 joints x 3 dimensions = 39 features per frame
- Problem B (2D): PoseNet frame sequence: 13 joints x 2 dimensions = 26 features per frame

Each problem will have two approaches:
- Dense: Flattened features for dense neural networks
- CNN: Structured features for convolutional neural networks
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os


def load_processed_data(csv_path):
    """Load processed data from CSV file."""
    df = pd.read_csv(csv_path)
    # Extract features (skip filename and label columns)
    feature_cols = [col for col in df.columns if col not in ['filename', 'label']]
    X = df[feature_cols].values
    y = df['label'].values
    filenames = df['filename'].values

    return X, y, filenames


def reshape_for_3d_problem(X, frames_per_seq=10, joints_per_frame=13, dims=3):
    """
    Reshape data for 3D problem (Kinect: 13 joints x 3 dimensions = 39 features per frame).

    Args:
        X: Input data of shape (samples, total_features)
        frames_per_seq: Number of frames per sequence (default 10)
        joints_per_frame: Number of joints per frame (default 13)
        dims: Number of dimensions (default 3 for 3D)

    Returns:
        Reshaped data of shape (samples, frames_per_seq, joints_per_frame, dims)
    """
    total_features = frames_per_seq * joints_per_frame * dims
    samples = X.shape[0]

    # Check if the data has the expected number of features
    if X.shape[1] != total_features:
        print(f"Warning: Expected {total_features} features per sample, got {X.shape[1]}")
        print("Attempting to extract 3D features by taking first 39 per frame...")

        # If we have more features per frame, take the first 39 per frame as 3D coordinates
        features_per_frame = X.shape[1] // frames_per_seq
        if features_per_frame >= joints_per_frame * dims:
            # Extract 3D coordinates from each frame
            X_3d = np.zeros((samples, frames_per_seq, joints_per_frame, dims))
            for frame_idx in range(frames_per_seq):
                start_idx = frame_idx * features_per_frame
                end_idx = start_idx + joints_per_frame * dims
                frame_data = X[:, start_idx:end_idx]
                X_3d[:, frame_idx, :, :] = frame_data.reshape(samples, joints_per_frame, dims)
        else:
            raise ValueError(f"Insufficient features per frame for 3D interpretation: {features_per_frame}")
    else:
        X_3d = X.reshape(samples, frames_per_seq, joints_per_frame, dims)

    return X_3d


def reshape_for_2d_problem(X, frames_per_seq=10, joints_per_frame=13, dims=2):
    """
    Reshape data for 2D problem (PoseNet: 13 joints x 2 dimensions = 26 features per frame).

    Args:
        X: Input data of shape (samples, total_features)
        frames_per_seq: Number of frames per sequence (default 10)
        joints_per_frame: Number of joints per frame (default 13)
        dims: Number of dimensions (default 2 for 2D)

    Returns:
        Reshaped data of shape (samples, frames_per_seq, joints_per_frame, dims)
    """
    total_features = frames_per_seq * joints_per_frame * dims
    samples = X.shape[0]

    # Check if the data has the expected number of features
    if X.shape[1] != total_features:
        print(f"Warning: Expected {total_features} features per sample, got {X.shape[1]}")
        print("Attempting to extract 2D features by taking first 26 per frame...")

        # If we have more features per frame, take the first 26 per frame as 2D coordinates
        features_per_frame = X.shape[1] // frames_per_seq
        if features_per_frame >= joints_per_frame * dims:
            # Extract 2D coordinates from each frame
            X_2d = np.zeros((samples, frames_per_seq, joints_per_frame, dims))
            for frame_idx in range(frames_per_seq):
                start_idx = frame_idx * features_per_frame
                end_idx = start_idx + joints_per_frame * dims
                frame_data = X[:, start_idx:end_idx]
                X_2d[:, frame_idx, :, :] = frame_data.reshape(samples, joints_per_frame, dims)
        else:
            raise ValueError(f"Insufficient features per frame for 2D interpretation: {features_per_frame}")
    else:
        X_2d = X.reshape(samples, frames_per_seq, joints_per_frame, dims)

    return X_2d


def prepare_adense_data(X_3d):
    """
    Prepare data for ADense (3D Dense network).

    Args:
        X_3d: 3D data of shape (samples, frames, joints, dims)

    Returns:
        Flattened data of shape (samples, frames*joints*dims)
    """
    samples, frames, joints, dims = X_3d.shape
    X_flat = X_3d.reshape(samples, frames * joints * dims)
    return X_flat


def prepare_acnn_data(X_3d):
    """
    Prepare data for ACNN (3D Convolutional network).

    Args:
        X_3d: 3D data of shape (samples, frames, joints, dims)

    Returns:
        Data suitable for CNN: (samples, channels, frames, joints, dims) or (samples, frames, joints, dims)
    """
    # For CNN, we can keep the 4D structure or add a channel dimension
    # Standard format for 3D CNN would be (samples, channels, depth, height, width)
    # Or we can use (samples, time_steps, joints, features) for temporal CNN
    return X_3d


def prepare_bdense_data(X_2d):
    """
    Prepare data for BDense (2D Dense network).

    Args:
        X_2d: 2D data of shape (samples, frames, joints, dims)

    Returns:
        Flattened data of shape (samples, frames*joints*dims)
    """
    samples, frames, joints, dims = X_2d.shape
    X_flat = X_2d.reshape(samples, frames * joints * dims)
    return X_flat


def prepare_bcnn_data(X_2d):
    """
    Prepare data for BCNN (2D Convolutional network).

    Args:
        X_2d: 2D data of shape (samples, frames, joints, dims)

    Returns:
        Data suitable for CNN: (samples, frames, joints, dims) or with added channel dim
    """
    # For CNN, we can keep the structure as is for temporal processing
    return X_2d


def save_data(X, y, filenames, output_dir, prefix):
    """Save prepared data to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, f'{prefix}_X.npy'), X)
    np.save(os.path.join(output_dir, f'{prefix}_y.npy'), y)
    np.save(os.path.join(output_dir, f'{prefix}_filenames.npy'), filenames)

    print(f"Saved {prefix} data: X shape {X.shape}, y shape {y.shape}")


def main():
    print("Preparing data for classification problems A and B...")

    # Load the processed data
    data_dir = Path("Data-intensive-systems/A13/Processed_Data")

    # Load training data
    print("\nLoading training data...")
    X_train, y_train, fn_train = load_processed_data(
        data_dir / "processed_sequences_Good_vs_Bad_train.csv"
    )
    print(f"Training data shape: {X_train.shape}")

    # Load test data
    print("Loading test data...")
    X_test, y_test, fn_test = load_processed_data(
        data_dir / "processed_sequences_Good_vs_Bad_test.csv"
    )
    print(f"Test data shape: {X_test.shape}")

    # Load augmented training data
    print("Loading augmented training data...")
    X_train_aug, y_train_aug, fn_train_aug = load_processed_data(
        data_dir / "processed_sequences_Good_vs_Bad_train_augmented.csv"
    )
    print(f"Augmented training data shape: {X_train_aug.shape}")

    # Load augmented test data
    print("Loading augmented test data...")
    X_test_aug, y_test_aug, fn_test_aug = load_processed_data(
        data_dir / "processed_sequences_Good_vs_Bad_test_augmented.csv"
    )
    print(f"Augmented test data shape: {X_test_aug.shape}")

    # Prepare output directory
    output_dir = Path("Data-intensive-systems/A13/classification_problems/prepared_data")
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for Problem A (3D - Kinect)
    print("\n" + "="*60)
    print("PREPARING PROBLEM A (3D - Kinect: 13 joints x 3 dims)")
    print("="*60)

    try:
        # Convert to 3D format (samples, frames, joints, dimensions)
        X_train_3d = reshape_for_3d_problem(X_train, frames_per_seq=10, joints_per_frame=13, dims=3)
        X_test_3d = reshape_for_3d_problem(X_test, frames_per_seq=10, joints_per_frame=13, dims=3)
        X_train_aug_3d = reshape_for_3d_problem(X_train_aug, frames_per_seq=10, joints_per_frame=13, dims=3)
        X_test_aug_3d = reshape_for_3d_problem(X_test_aug, frames_per_seq=10, joints_per_frame=13, dims=3)

        print(f"3D training data shape: {X_train_3d.shape}")
        print(f"3D test data shape: {X_test_3d.shape}")
        print(f"3D augmented training data shape: {X_train_aug_3d.shape}")
        print(f"3D augmented test data shape: {X_test_aug_3d.shape}")

        # Prepare ADense data (flattened)
        print("\nPreparing ADense data (flattened)...")
        X_train_adense = prepare_adense_data(X_train_3d)
        X_test_adense = prepare_adense_data(X_test_3d)
        X_train_aug_adense = prepare_adense_data(X_train_aug_3d)
        X_test_aug_adense = prepare_adense_data(X_test_aug_3d)

        print(f"A-Dense training shape: {X_train_adense.shape}")
        print(f"A-Dense test shape: {X_test_adense.shape}")

        # Save ADense data
        save_data(X_train_adense, y_train, fn_train, output_dir, "A_Dense_train")
        save_data(X_test_adense, y_test, fn_test, output_dir, "A_Dense_test")
        save_data(X_train_aug_adense, y_train_aug, fn_train_aug, output_dir, "A_Dense_train_aug")
        save_data(X_test_aug_adense, y_test_aug, fn_test_aug, output_dir, "A_Dense_test_aug")

        # Prepare ACNN data (structured)
        print("\nPreparing ACNN data (structured)...")
        X_train_acnn = prepare_acnn_data(X_train_3d)
        X_test_acnn = prepare_acnn_data(X_test_3d)
        X_train_aug_acnn = prepare_acnn_data(X_train_aug_3d)
        X_test_aug_acnn = prepare_acnn_data(X_test_aug_3d)

        print(f"A-CNN training shape: {X_train_acnn.shape}")
        print(f"A-CNN test shape: {X_test_acnn.shape}")

        # Save ACNN data
        save_data(X_train_acnn, y_train, fn_train, output_dir, "A_CNN_train")
        save_data(X_test_acnn, y_test, fn_test, output_dir, "A_CNN_test")
        save_data(X_train_aug_acnn, y_train_aug, fn_train_aug, output_dir, "A_CNN_train_aug")
        save_data(X_test_aug_acnn, y_test_aug, fn_test_aug, output_dir, "A_CNN_test_aug")

        print("\nProblem A (3D) data preparation completed!")

    except Exception as e:
        print(f"Error preparing Problem A data: {e}")
        print("Skipping Problem A...")

    # Prepare data for Problem B (2D - PoseNet)
    print("\n" + "="*60)
    print("PREPARING PROBLEM B (2D - PoseNet: 13 joints x 2 dims)")
    print("="*60)

    try:
        # Convert to 2D format (samples, frames, joints, dimensions)
        X_train_2d = reshape_for_2d_problem(X_train, frames_per_seq=10, joints_per_frame=13, dims=2)
        X_test_2d = reshape_for_2d_problem(X_test, frames_per_seq=10, joints_per_frame=13, dims=2)
        X_train_aug_2d = reshape_for_2d_problem(X_train_aug, frames_per_seq=10, joints_per_frame=13, dims=2)
        X_test_aug_2d = reshape_for_2d_problem(X_test_aug, frames_per_seq=10, joints_per_frame=13, dims=2)

        print(f"2D training data shape: {X_train_2d.shape}")
        print(f"2D test data shape: {X_test_2d.shape}")
        print(f"2D augmented training data shape: {X_train_aug_2d.shape}")
        print(f"2D augmented test data shape: {X_test_aug_2d.shape}")

        # Prepare BDense data (flattened)
        print("\nPreparing BDense data (flattened)...")
        X_train_bdense = prepare_bdense_data(X_train_2d)
        X_test_bdense = prepare_bdense_data(X_test_2d)
        X_train_aug_bdense = prepare_bdense_data(X_train_aug_2d)
        X_test_aug_bdense = prepare_bdense_data(X_test_aug_2d)

        print(f"B-Dense training shape: {X_train_bdense.shape}")
        print(f"B-Dense test shape: {X_test_bdense.shape}")

        # Save BDense data
        save_data(X_train_bdense, y_train, fn_train, output_dir, "B_Dense_train")
        save_data(X_test_bdense, y_test, fn_test, output_dir, "B_Dense_test")
        save_data(X_train_aug_bdense, y_train_aug, fn_train_aug, output_dir, "B_Dense_train_aug")
        save_data(X_test_aug_bdense, y_test_aug, fn_test_aug, output_dir, "B_Dense_test_aug")

        # Prepare BCNN data (structured)
        print("\nPreparing BCNN data (structured)...")
        X_train_bcnn = prepare_bcnn_data(X_train_2d)
        X_test_bcnn = prepare_bcnn_data(X_test_2d)
        X_train_aug_bcnn = prepare_bcnn_data(X_train_aug_2d)
        X_test_aug_bcnn = prepare_bcnn_data(X_test_aug_2d)

        print(f"B-CNN training shape: {X_train_bcnn.shape}")
        print(f"B-CNN test shape: {X_test_bcnn.shape}")

        # Save BCNN data
        save_data(X_train_bcnn, y_train, fn_train, output_dir, "B_CNN_train")
        save_data(X_test_bcnn, y_test, fn_test, output_dir, "B_CNN_test")
        save_data(X_train_aug_bcnn, y_train_aug, fn_train_aug, output_dir, "B_CNN_train_aug")
        save_data(X_test_aug_bcnn, y_test_aug, fn_test_aug, output_dir, "B_CNN_test_aug")

        print("\nProblem B (2D) data preparation completed!")

    except Exception as e:
        print(f"Error preparing Problem B data: {e}")
        print("Skipping Problem B...")

    print("\n" + "="*60)
    print("CLASSIFICATION PROBLEMS DATA PREPARATION SUMMARY")
    print("="*60)
    print("Problem A (3D - Kinect): 13 joints x 3 dimensions per frame")
    print("  - ADense: Flattened features for dense networks")
    print("  - ACNN: Structured features for convolutional networks")
    print("")
    print("Problem B (2D - PoseNet): 13 joints x 2 dimensions per frame")
    print("  - BDense: Flattened features for dense networks")
    print("  - BCNN: Structured features for convolutional networks")
    print("")
    print("All prepared datasets saved to:", output_dir)
    print("Both original and augmented versions are available")


if __name__ == "__main__":
    main()
