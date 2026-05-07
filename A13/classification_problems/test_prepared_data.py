#!/usr/bin/env python3
"""
Test script to verify that the prepared data for classification problems can be loaded correctly
"""

import numpy as np
import os

def test_data_loading():
    """Test that all prepared data files can be loaded correctly."""
    base_path = "Data-intensive-systems/A13/classification_problems/prepared_data/"

    print("Testing data loading for classification problems...")

    # Define all expected files
    expected_files = [
        # ADense files
        "A_Dense_train_X.npy", "A_Dense_train_y.npy",
        "A_Dense_test_X.npy", "A_Dense_test_y.npy",
        "A_Dense_train_aug_X.npy", "A_Dense_train_aug_y.npy",
        "A_Dense_test_aug_X.npy", "A_Dense_test_aug_y.npy",

        # ACNN files
        "A_CNN_train_X.npy", "A_CNN_train_y.npy",
        "A_CNN_test_X.npy", "A_CNN_test_y.npy",
        "A_CNN_train_aug_X.npy", "A_CNN_train_aug_y.npy",
        "A_CNN_test_aug_X.npy", "A_CNN_test_aug_y.npy",

        # BDense files
        "B_Dense_train_X.npy", "B_Dense_train_y.npy",
        "B_Dense_test_X.npy", "B_Dense_test_y.npy",
        "B_Dense_train_aug_X.npy", "B_Dense_train_aug_y.npy",
        "B_Dense_test_aug_X.npy", "B_Dense_test_aug_y.npy",

        # BCNN files
        "B_CNN_train_X.npy", "B_CNN_train_y.npy",
        "B_CNN_test_X.npy", "B_CNN_test_y.npy",
        "B_CNN_train_aug_X.npy", "B_CNN_train_aug_y.npy",
        "B_CNN_test_aug_X.npy", "B_CNN_test_aug_y.npy"
    ]

    print(f"\nChecking {len(expected_files)} expected files...")

    missing_files = []
    loaded_data_info = {}

    for file in expected_files:
        file_path = os.path.join(base_path, file)
        if os.path.exists(file_path):
            try:
                data = np.load(file_path)
                loaded_data_info[file] = {
                    'shape': data.shape,
                    'dtype': data.dtype,
                    'size_mb': data.nbytes / (1024*1024)
                }
                print(f"✓ Loaded {file}: shape={data.shape}, dtype={data.dtype}, size={data.nbytes/(1024*1024):.2f}MB")
            except Exception as e:
                print(f"✗ Error loading {file}: {e}")
                missing_files.append(file)
        else:
            print(f"✗ Missing {file}")
            missing_files.append(file)

    if missing_files:
        print(f"\n❌ {len(missing_files)} files could not be loaded:")
        for f in missing_files:
            print(f"  - {f}")
        return False
    else:
        print(f"\n✅ All {len(expected_files)} files loaded successfully!")

    # Print summary statistics
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)

    # Problem A (3D) statistics
    print("\nProblem A (3D - Kinect):")
    print(f"  ADense - Train: {loaded_data_info['A_Dense_train_X.npy']['shape']}")
    print(f"  ADense - Test: {loaded_data_info['A_Dense_test_X.npy']['shape']}")
    print(f"  ADense - Train Aug: {loaded_data_info['A_Dense_train_aug_X.npy']['shape']}")
    print(f"  ADense - Test Aug: {loaded_data_info['A_Dense_test_aug_X.npy']['shape']}")
    print(f"  ACNN - Train: {loaded_data_info['A_CNN_train_X.npy']['shape']}")
    print(f"  ACNN - Test: {loaded_data_info['A_CNN_test_X.npy']['shape']}")
    print(f"  ACNN - Train Aug: {loaded_data_info['A_CNN_train_aug_X.npy']['shape']}")
    print(f"  ACNN - Test Aug: {loaded_data_info['A_CNN_test_aug_X.npy']['shape']}")

    # Problem B (2D) statistics
    print("\nProblem B (2D - PoseNet):")
    print(f"  BDense - Train: {loaded_data_info['B_Dense_train_X.npy']['shape']}")
    print(f"  BDense - Test: {loaded_data_info['B_Dense_test_X.npy']['shape']}")
    print(f"  BDense - Train Aug: {loaded_data_info['B_Dense_train_aug_X.npy']['shape']}")
    print(f"  BDense - Test Aug: {loaded_data_info['B_Dense_test_aug_X.npy']['shape']}")
    print(f"  BCNN - Train: {loaded_data_info['B_CNN_train_X.npy']['shape']}")
    print(f"  BCNN - Test: {loaded_data_info['B_CNN_test_X.npy']['shape']}")
    print(f"  BCNN - Train Aug: {loaded_data_info['B_CNN_train_aug_X.npy']['shape']}")
    print(f"  BCNN - Test Aug: {loaded_data_info['B_CNN_test_aug_X.npy']['shape']}")

    # Verify label consistency
    print("\nLabel consistency check:")
    a_train_labels = np.load(os.path.join(base_path, "A_Dense_train_y.npy"))
    b_train_labels = np.load(os.path.join(base_path, "B_Dense_train_y.npy"))
    a_test_labels = np.load(os.path.join(base_path, "A_Dense_test_y.npy"))
    b_test_labels = np.load(os.path.join(base_path, "B_Dense_test_y.npy"))

    print(f"  A train labels: {a_train_labels.shape}, unique values: {np.unique(a_train_labels)}")
    print(f"  B train labels: {b_train_labels.shape}, unique values: {np.unique(b_train_labels)}")
    print(f"  A test labels: {a_test_labels.shape}, unique values: {np.unique(a_test_labels)}")
    print(f"  B test labels: {b_test_labels.shape}, unique values: {np.unique(b_test_labels)}")

    print("\n✅ All prepared data is ready for machine learning experiments!")
    return True

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\n🎉 Classification problems data preparation verified successfully!")
    else:
        print("\n❌ Issues found with data preparation.")
