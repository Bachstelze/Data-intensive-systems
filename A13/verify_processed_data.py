#!/usr/bin/env python3
"""
Script to verify the processed data
"""

import numpy as np
import pandas as pd
from pathlib import Path


def main():
    output_dir = Path("Data-intensive-systems/A13/Processed_Data")

    # Load the processed data
    sequences = np.load(output_dir / "sequences.npy")
    labels = np.load(output_dir / "labels.npy")

    print("Verification of processed data:")
    print(f"- Sequences shape: {sequences.shape}")
    print(f"- Labels shape: {labels.shape}")
    print(f"- Number of samples: {len(sequences)}")
    print(f"- Number of frames per sequence: {sequences.shape[1]}")
    print(f"- Number of features per frame: {sequences.shape[2]}")
    print(f"- Number of good sequences (label=1): {np.sum(labels == 1)}")
    print(f"- Number of bad sequences (label=0): {np.sum(labels == 0)}")

    # Check if data looks reasonable
    print(f"\nSample from first sequence (first frame, first 10 features):")
    print(sequences[0, 0, :10])

    print(f"\nSample from last sequence (first frame, first 10 features):")
    print(sequences[-1, 0, :10])

    # Verify labels correspond to expected good/bad classification
    df = pd.read_csv(output_dir / "processed_sequences_with_labels.csv")
    print(f"\nCSV file shape: {df.shape}")
    print(f"First few entries:")
    print(df.head())

    # Count good vs bad from CSV
    good_count = sum(df['label'] == 1)
    bad_count = sum(df['label'] == 0)
    print(f"\nFrom CSV - Good: {good_count}, Bad: {bad_count}")

    print("\nProcessing completed successfully!")


if __name__ == "__main__":
    main()
