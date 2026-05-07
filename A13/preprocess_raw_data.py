#!/usr/bin/env python3
"""
Script to preprocess raw kinect data according to the specifications:
- Fixed size sequence with c=10 frames equidistantly distributed
- Mark as good/bad based on filenames (A1, G* = good; W* = bad)
- Average coordinates of surrounding frames (optional)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_csv_data(csv_path):
    """Load CSV data and return as numpy array. Handles both space-separated and comma-separated files with or without headers."""
    # Read the first line to check if it contains headers
    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()

    # Try to convert first line to float to check if it's data or header
    try:
        # Try converting the first element to float
        parts = first_line.split(None, 1)  # Split on first whitespace to check first value
        first_value = parts[0]
        float(first_value)
        # If successful, the file doesn't have headers, process as before
        data = []
        with open(csv_path, 'r') as f:
            for line in f:
                row = [float(x) for x in line.strip().split()]
                data.append(row)
        return np.array(data)
    except ValueError:
        # If conversion fails, it's likely a header, so skip it
        data = []
        with open(csv_path, 'r') as f:
            next(f)  # Skip header line
            for line in f:
                # Handle both space-separated and comma-separated values
                if ',' in line:
                    row = [float(x) for x in line.strip().split(',')]
                else:
                    row = [float(x) for x in line.strip().split()]
                data.append(row)
        return np.array(data)


def extract_frame_sequence(data, start_frame, end_frame):
    """Extract frames from start_frame to end_frame (inclusive)."""
    mask = (data[:, 0] >= start_frame) & (data[:, 0] <= end_frame)
    return data[mask]


def select_equidistant_frames(sequence, c=10):
    """Select c equidistant frames from the sequence."""
    if len(sequence) <= c:
        # If sequence is shorter than c, pad with zeros or repeat last frame
        selected_frames = []
        for i in range(c):
            if i < len(sequence):
                selected_frames.append(sequence[i])
            else:
                # Pad with zeros for missing frames
                padded_frame = np.zeros_like(sequence[0])
                # Keep the frame number as an indicator
                padded_frame[0] = -1  # Use -1 to indicate padding
                selected_frames.append(padded_frame)
        return np.array(selected_frames)

    # Select c equidistant indices
    indices = np.linspace(0, len(sequence) - 1, c, dtype=int)
    return sequence[indices]


def average_surrounding_frames(frame_data, window_size=3):
    """
    Average coordinates of surrounding frames (optional processing).
    This helps smooth the data by considering neighboring frames.
    """
    if len(frame_data) == 1:
        return frame_data

    averaged_data = np.copy(frame_data)

    for i in range(len(frame_data)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(frame_data), i + window_size // 2 + 1)

        # Average the coordinate columns (skip the first few columns which are metadata)
        # Assuming first 3 columns are metadata (frame_num, timestamp, ?) and rest are coordinates
        coord_start = 3  # Start averaging from column 3 onwards

        for j in range(coord_start, frame_data.shape[1]):
            avg_val = np.mean(frame_data[start_idx:end_idx, j])
            averaged_data[i, j] = avg_val

    return averaged_data


def process_video_file(csv_path, start_stop_path=None, apply_averaging=False):
    """Process a single video file and return processed data."""
    # Load the raw data
    raw_data = load_csv_data(csv_path)

    if start_stop_path and os.path.exists(start_stop_path):
        # Load start and stop frames if available
        with open(start_stop_path, 'r') as f:
            start_frame, stop_frame = map(int, f.read().strip().split())

        # Extract the relevant frame sequence
        frame_sequence = extract_frame_sequence(raw_data, start_frame, stop_frame)
    else:
        # Use the full sequence if no start/stop is provided
        frame_sequence = raw_data

    # Select equidistant frames (c=10)
    equidistant_frames = select_equidistant_frames(frame_sequence, c=10)

    # Optionally average surrounding frames
    if apply_averaging:
        equidistant_frames = average_surrounding_frames(equidistant_frames)

    return equidistant_frames


def get_label_from_filename(filename):
    """Determine if the file represents a 'good' or 'bad' sequence."""
    basename = Path(filename).stem
    if basename.startswith('G') or basename == 'A1':
        return 1  # Good
    elif basename.startswith('W'):
        return 0  # Bad
    else:
        raise ValueError(f"Unknown file type for {filename}")


def process_folder(base_dir, output_dir):
    """Process all files in a folder."""
    # Find all CSV files
    csv_files = list(base_dir.glob("*.csv"))

    processed_sequences = []
    labels = []
    filenames = []

    print(f"Processing video files in {base_dir}...")

    for csv_file in csv_files:
        # Skip if it's not a main video file (e.g., skip .new_json files)
        if '.new_json' in str(csv_file) or '.json' in str(csv_file):
            continue

        # Check if corresponding start_stop file exists
        start_stop_file = csv_file.with_suffix('.start_stop_frames')

        # Process the file
        try:
            processed_data = process_video_file(csv_file, start_stop_file if start_stop_file.exists() else None, apply_averaging=True)

            # Get the label based on filename
            label = get_label_from_filename(csv_file.name)

            processed_sequences.append(processed_data)
            labels.append(label)
            filenames.append(csv_file.stem)

            print(f"Processed {csv_file.name}: shape {processed_data.shape}, label {label}")

        except Exception as e:
            print(f"Error processing {csv_file.name}: {str(e)}")
            continue

    return processed_sequences, labels, filenames


def main():
    # Define paths
    base_dirs = [
        Path("Data-intensive-systems/A13/Raw_data/Good vs Bad"),
        Path("Data-intensive-systems/A13/Raw_data/kinect_good_vs_bad_not_preprocessed")
    ]
    output_dir = Path("Data-intensive-systems/A13/Processed_Data")
    output_dir.mkdir(exist_ok=True)

    for i, base_dir in enumerate(base_dirs):
        if base_dir.exists():
            print(f"\nProcessing directory {i+1}: {base_dir}")
            sequences, labels, filenames = process_folder(base_dir, output_dir)

            if not sequences:
                print(f"No files were processed successfully in {base_dir}.")
                continue

            # Convert to numpy arrays
            processed_sequences = np.array(sequences)
            labels = np.array(labels)

            print(f"Final dataset shape: {processed_sequences.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Number of good sequences: {np.sum(labels == 1)}")
            print(f"Number of bad sequences: {np.sum(labels == 0)}")

            # Split the data into train and test sets
            X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
                processed_sequences, labels, filenames,
                test_size=0.2, random_state=42, stratify=labels
            )

            print(f"Training set shape: {X_train.shape}")
            print(f"Test set shape: {X_test.shape}")
            print(f"Training good sequences: {np.sum(y_train == 1)}")
            print(f"Training bad sequences: {np.sum(y_train == 0)}")
            print(f"Test good sequences: {np.sum(y_test == 1)}")
            print(f"Test bad sequences: {np.sum(y_test == 0)}")

            # Determine directory name for saving files
            dir_name = base_dir.name.replace(' ', '_').replace('-', '_')

            # Save the processed data with directory-specific naming
            np.save(output_dir / f"sequences_{dir_name}_train.npy", X_train)
            np.save(output_dir / f"sequences_{dir_name}_test.npy", X_test)
            np.save(output_dir / f"labels_{dir_name}_train.npy", y_train)
            np.save(output_dir / f"labels_{dir_name}_test.npy", y_test)

            # Also save as CSV for easier inspection
            # Reshape sequences to 2D for CSV export
            n_train, n_frames, n_features = X_train.shape
            n_test = X_test.shape[0]

            reshaped_X_train = X_train.reshape(n_train, n_frames * n_features)
            reshaped_X_test = X_test.reshape(n_test, n_frames * n_features)

            # Create CSV for training data
            df_train = pd.DataFrame(reshaped_X_train)
            df_train.insert(0, 'label', y_train)
            df_train.insert(0, 'filename', filenames_train)
            df_train.to_csv(output_dir / f"processed_sequences_{dir_name}_train.csv", index=False)

            # Create CSV for test data
            df_test = pd.DataFrame(reshaped_X_test)
            df_test.insert(0, 'label', y_test)
            df_test.insert(0, 'filename', filenames_test)
            df_test.to_csv(output_dir / f"processed_sequences_{dir_name}_test.csv", index=False)

            print(f"\nProcessed data for {base_dir.name} saved to {output_dir}/")
            print(f"- sequences_{dir_name}_train.npy: Numpy array of shape (n_train, 10, n_features) for training")
            print(f"- sequences_{dir_name}_test.npy: Numpy array of shape (n_test, 10, n_features) for testing")
            print(f"- labels_{dir_name}_train.npy: Numpy array of shape (n_train,) with binary labels for training")
            print(f"- labels_{dir_name}_test.npy: Numpy array of shape (n_test,) with binary labels for testing")
            print(f"- processed_sequences_{dir_name}_train.csv: CSV file with training labels and filenames")
            print(f"- processed_sequences_{dir_name}_test.csv: CSV file with test labels and filenames")


if __name__ == "__main__":
    main()
