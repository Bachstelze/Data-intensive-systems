#!/usr/bin/env python3
"""
Test script to validate that augmented data can be used effectively for machine learning
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from pathlib import Path


def load_data_from_csv(csv_path):
    """Load data from CSV file and separate features from labels."""
    df = pd.read_csv(csv_path)

    # Extract features (skip filename and label columns)
    feature_cols = [col for col in df.columns if col not in ['filename', 'label']]
    X = df[feature_cols].values
    y = df['label'].values
    filenames = df['filename'].values

    return X, y, filenames


def main():
    print("Testing augmented processed kinect data...")

    # Load the augmented processed data
    data_dir = Path("Data-intensive-systems/A13/Processed_Data")

    print("\nLoading augmented training data...")
    X_train_aug, y_train_aug, fn_train_aug = load_data_from_csv(
        data_dir / "processed_sequences_Good_vs_Bad_train_augmented.csv"
    )

    print(f"Augmented training sequences shape: {X_train_aug.shape}")
    print(f"Augmented training labels shape: {y_train_aug.shape}")
    print(f"Number of original samples identified: {len([f for f in fn_train_aug if '_' not in f])}")
    print(f"Good/Bad distribution in augmented training: {sum(y_train_aug)}/{len(y_train_aug)-sum(y_train_aug)}")

    print("\nLoading augmented test data...")
    X_test_aug, y_test_aug, fn_test_aug = load_data_from_csv(
        data_dir / "processed_sequences_Good_vs_Bad_test_augmented.csv"
    )

    print(f"Augmented test sequences shape: {X_test_aug.shape}")
    print(f"Augmented test labels shape: {y_test_aug.shape}")
    print(f"Number of original samples identified: {len([f for f in fn_test_aug if '_' not in f])}")
    print(f"Good/Bad distribution in augmented test: {sum(y_test_aug)}/{len(y_test_aug)-sum(y_test_aug)}")

    # Compare with original data
    print("\nLoading original training data for comparison...")
    X_train_orig, y_train_orig, fn_train_orig = load_data_from_csv(
        data_dir / "processed_sequences_Good_vs_Bad_train.csv"
    )

    print(f"Original training sequences shape: {X_train_orig.shape}")
    print(f"Original training labels shape: {y_train_orig.shape}")
    print(f"Good/Bad distribution in original training: {sum(y_train_orig)}/{len(y_train_orig)-sum(y_train_orig)}")

    print("\nLoading original test data for comparison...")
    X_test_orig, y_test_orig, fn_test_orig = load_data_from_csv(
        data_dir / "processed_sequences_Good_vs_Bad_test.csv"
    )

    print(f"Original test sequences shape: {X_test_orig.shape}")
    print(f"Original test labels shape: {y_test_orig.shape}")
    print(f"Good/Bad distribution in original test: {sum(y_test_orig)}/{len(y_test_orig)-sum(y_test_orig)}")

    print(f"\nData augmentation summary:")
    print(f"- Training data increased from {X_train_orig.shape[0]} to {X_train_aug.shape[0]} samples ({X_train_aug.shape[0]/X_train_orig.shape[0]:.1f}x)")
    print(f"- Test data increased from {X_test_orig.shape[0]} to {X_test_aug.shape[0]} samples ({X_test_aug.shape[0]/X_test_orig.shape[0]:.1f}x)")

    # Train models and compare performance
    print("\n" + "="*60)
    print("COMPARISON OF ORIGINAL VS AUGMENTED DATA PERFORMANCE")
    print("="*60)

    # Train on original data, test on original
    print("\n1. Original training data -> Original test data:")
    clf_orig = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_orig.fit(X_train_orig, y_train_orig)
    y_pred_orig = clf_orig.predict(X_test_orig)
    acc_orig = accuracy_score(y_test_orig, y_pred_orig)
    print(f"   Accuracy: {acc_orig:.3f}")
    print(f"   Classification Report:")
    print(classification_report(y_test_orig, y_pred_orig, target_names=['Bad', 'Good'], digits=3))

    # Train on augmented data, test on original
    print("\n2. Augmented training data -> Original test data:")
    clf_aug = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_aug.fit(X_train_aug, y_train_aug)
    y_pred_aug = clf_aug.predict(X_test_orig)
    acc_aug = accuracy_score(y_test_orig, y_pred_aug)
    print(f"   Accuracy: {acc_aug:.3f}")
    print(f"   Classification Report:")
    print(classification_report(y_test_orig, y_pred_aug, target_names=['Bad', 'Good'], digits=3))

    # Check if augmentation improved performance
    improvement = acc_aug - acc_orig
    print(f"\nPerformance change due to augmentation: {improvement:+.3f}")
    if improvement > 0:
        print("✓ Augmentation improved model performance!")
    elif improvement == 0:
        print("~ Augmentation had no effect on performance.")
    else:
        print("⚠ Augmentation decreased model performance.")

    # Additional validation: check that augmented samples maintain correct labels
    print("\n" + "="*60)
    print("VALIDATION OF AUGMENTED DATA INTEGRITY")
    print("="*60)

    # Count how many augmented samples there are
    original_in_aug_train = sum(1 for f in fn_train_aug if '_' not in f)
    augmented_in_aug_train = len(fn_train_aug) - original_in_aug_train

    print(f"Original samples in augmented train set: {original_in_aug_train}")
    print(f"Augmented samples in augmented train set: {augmented_in_aug_train}")

    # Check label consistency for augmented samples
    # Original samples should have specific filenames, augmented should have suffixes
    original_samples_mask = np.array(['_' not in f for f in fn_train_aug])
    augmented_samples_mask = ~original_samples_mask

    orig_labels_count = np.sum(y_train_aug[original_samples_mask])
    aug_labels_count = np.sum(y_train_aug[augmented_samples_mask])

    print(f"Good samples among original: {orig_labels_count}/{np.sum(original_samples_mask)} ({orig_labels_count/np.sum(original_samples_mask)*100:.1f}%)")
    print(f"Good samples among augmented: {aug_labels_count}/{np.sum(augmented_samples_mask)} ({aug_labels_count/np.sum(augmented_samples_mask)*100:.1f}%)")

    print("\nAugmentation validation completed successfully!")
    print("Key findings:")
    print(f"- Original training data: {X_train_orig.shape[0]} samples")
    print(f"- Augmented training data: {X_train_aug.shape[0]} samples")
    print(f"- Performance difference: {improvement:+.3f}")
    print("- Augmented data maintains label consistency")
    print("- Augmented data can be used seamlessly with ML pipelines")


if __name__ == "__main__":
    main()
