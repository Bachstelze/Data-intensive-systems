#!/usr/bin/env python3
"""
Example script showing how to use the processed data for machine learning
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path


def main():
    print("Loading processed kinect data...")

    # Load the pre-split processed data for the first directory
    output_dir = Path("Data-intensive-systems/A13/Processed_Data")
    X_train = np.load(output_dir / "sequences_Good_vs_Bad_train.npy")  # Shape: (n_train, 10, 102)
    X_test = np.load(output_dir / "sequences_Good_vs_Bad_test.npy")    # Shape: (n_test, 10, 102)
    y_train = np.load(output_dir / "labels_Good_vs_Bad_train.npy")      # Shape: (n_train,)
    y_test = np.load(output_dir / "labels_Good_vs_Bad_test.npy")        # Shape: (n_test,)

    print(f"Training sequences shape: {X_train.shape}")
    print(f"Test sequences shape: {X_test.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test labels shape: {y_test.shape}")

    # Reshape sequences to 2D for traditional ML algorithms
    # Option 1: Flatten all frames and features into a single vector per sample
    n_train, n_frames, n_features = X_train.shape
    n_test = X_test.shape[0]
    X_train_flattened = X_train.reshape(n_train, n_frames * n_features)
    X_test_flattened = X_test.reshape(n_test, n_frames * n_features)

    print(f"Training flattened features shape: {X_train_flattened.shape}")
    print(f"Test flattened features shape: {X_test_flattened.shape}")

    print(f"Training set: {X_train_flattened.shape[0]} samples")
    print(f"Test set: {X_test_flattened.shape[0]} samples")
    print(f"Good/Bad distribution in training: {sum(y_train)}/{len(y_train)-sum(y_train)}")
    print(f"Good/Bad distribution in test: {sum(y_test)}/{len(y_test)-sum(y_test)}")

    # Train a simple classifier
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_flattened, y_train)

    # Make predictions
    y_pred = clf.predict(X_test_flattened)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.3f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Bad', 'Good']))

    # Alternative approach: Use only the mean of each feature across frames
    print("\n" + "="*50)
    print("Alternative approach: Using mean values across frames")

    # Calculate mean across frames for each feature
    X_train_mean = np.mean(X_train, axis=1)  # Take mean across the frame dimension
    X_test_mean = np.mean(X_test, axis=1)    # Take mean across the frame dimension

    print(f"Training mean features shape: {X_train_mean.shape}")
    print(f"Test mean features shape: {X_test_mean.shape}")

    # Train the classifier using mean values
    clf_mean = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_mean.fit(X_train_mean, y_train)

    y_pred_m = clf_mean.predict(X_test_mean)
    accuracy_m = accuracy_score(y_test, y_pred_m)

    print(f"Mean-based Accuracy: {accuracy_m:.3f}")
    print("\nMean-based Classification Report:")
    print(classification_report(y_test, y_pred_m, target_names=['Bad', 'Good']))

    # Show feature importance (for the mean-based model)
    feature_importance = clf_mean.feature_importances_
    top_10_indices = np.argsort(feature_importance)[-10:][::-1]

    print(f"\nTop 10 most important features (out of {X_train_mean.shape[1]}):")
    for i, idx in enumerate(top_10_indices):
        print(f"{i+1:2d}. Feature {idx}: Importance = {feature_importance[idx]:.4f}")

    print("\nData preprocessing completed successfully!")
    print("The processed data is ready for various machine learning approaches:")
    print("- Traditional ML (Random Forest, SVM, etc.) using flattened features")
    print("- Deep learning with RNN/LSTM using sequential structure")
    print("- CNN approaches treating frames as temporal 'images'")


if __name__ == "__main__":
    main()
