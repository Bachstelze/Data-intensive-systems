#!/usr/bin/env python3
"""
Script to check and correct outliers in start/stop classification data.
Identifies frames where the label may be incorrect based on velocity/acceleration changes.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import sys

# Define paths
BASE_DIR = Path(__file__).parent
CLASSIFICATION_DATA_DIR = BASE_DIR.parent / "classification_data"
OUTPUT_DIR = BASE_DIR / "corrected_data"

# Thresholds for outlier detection
VELOCITY_CHANGE_THRESHOLD = 2.0  # Significant change in velocity
ACCELERATION_THRESHOLD = 5.0     # High acceleration indicates start/stop
SPEED_THRESHOLD = 0.3            # Minimum speed for motion
# Threshold for cut frame comparison
CUT_START_STOP_RATIO_THRESHOLD = 0.3  # Max ratio of start/stop in cut vs not-cut
# Threshold for cut frame count comparison
CUT_FRAME_COUNT_THRESHOLD = 0.5  # Min ratio of cut frames to not-cut frames


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect potential outliers in start/stop labels based on motion characteristics.

    Returns a DataFrame with outlier information.
    """
    df = df.copy()

    # Calculate velocity changes between consecutive frames
    joints = ['head', 'left_hand', 'right_hand', 'left_elbow', 'right_elbow']

    for joint in joints:
        speed_col = f'{joint}_speed'
        if speed_col in df.columns:
            df[f'{joint}_speed_change'] = df[speed_col].diff().abs()

    # Calculate acceleration magnitude
    for joint in joints:
        accel_col = f'{joint}_accel'
        if accel_col in df.columns:
            df[f'{joint}_accel_abs'] = df[accel_col].abs()

    # Identify potential outliers
    outlier_mask = pd.Series([False] * len(df), index=df.index)

    for joint in joints:
        speed_change_col = f'{joint}_speed_change'
        accel_abs_col = f'{joint}_accel_abs'

        if speed_change_col in df.columns:
            outlier_mask = outlier_mask | (df[speed_change_col] > VELOCITY_CHANGE_THRESHOLD)
        if accel_abs_col in df.columns:
            outlier_mask = outlier_mask | (df[accel_abs_col] > ACCELERATION_THRESHOLD)

    # Create outlier info DataFrame
    outlier_df = df[outlier_mask].copy()

    # Add outlier reason
    outlier_df['outlier_reason'] = ''
    for joint in joints:
        speed_change_col = f'{joint}_speed_change'
        accel_abs_col = f'{joint}_accel_abs'

        if speed_change_col in outlier_df.columns:
            mask = outlier_df[speed_change_col] > VELOCITY_CHANGE_THRESHOLD
            outlier_df.loc[mask, 'outlier_reason'] += f'{joint}_speed_change,'
        if accel_abs_col in outlier_df.columns:
            mask = outlier_df[accel_abs_col] > ACCELERATION_THRESHOLD
            outlier_df.loc[mask, 'outlier_reason'] += f'{joint}_accel,'

    # Remove trailing comma
    outlier_df['outlier_reason'] = outlier_df['outlier_reason'].str.rstrip(',')

    return outlier_df


def analyze_label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze label distribution and identify potential mislabeled frames.

    A frame is considered potentially mislabeled if:
    - It's labeled 'start' but has low velocity change
    - It's labeled 'stop' but has low velocity change
    - It's labeled 'neutral' but has high velocity change
    """
    df = df.copy()

    joints = ['head', 'left_hand', 'right_hand', 'left_elbow', 'right_elbow']

    # Calculate velocity changes
    for joint in joints:
        speed_col = f'{joint}_speed'
        if speed_col in df.columns:
            df[f'{joint}_speed_change'] = df[speed_col].diff().abs()

    # Identify frames with high velocity changes
    high_change_mask = pd.Series([False] * len(df), index=df.index)
    for joint in joints:
        speed_change_col = f'{joint}_speed_change'
        if speed_change_col in df.columns:
            high_change_mask = high_change_mask | (df[speed_change_col] > VELOCITY_CHANGE_THRESHOLD)

    df['high_velocity_change'] = high_change_mask

    # Identify potentially mislabeled frames
    mislabel_mask = (
        # 'start' but no high velocity change
        ((df['label'] == 'start') & ~high_change_mask) |
        # 'stop' but no high velocity change
        ((df['label'] == 'stop') & ~high_change_mask) |
        # 'neutral' but high velocity change
        ((df['label'] == 'neutral') & high_change_mask)
    )

    mislabeled_df = df[mislabel_mask].copy()

    # Add mislabel reason
    mislabeled_df['mislabel_reason'] = ''
    mislabeled_df.loc[df['label'] == 'start', 'mislabel_reason'] = 'start_without_velocity_change'
    mislabeled_df.loc[df['label'] == 'stop', 'mislabel_reason'] = 'stop_without_velocity_change'
    mislabeled_df.loc[df['label'] == 'neutral', 'mislabel_reason'] = 'neutral_with_velocity_change'

    return mislabeled_df


def get_file_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for each file in the dataset.
    """
    summary = []

    for file_id in df['file_id'].unique():
        file_df = df[df['file_id'] == file_id]

        label_counts = file_df['label'].value_counts()

        summary.append({
            'file_id': file_id,
            'total_frames': len(file_df),
            'start_count': label_counts.get('start', 0),
            'stop_count': label_counts.get('stop', 0),
            'neutral_count': label_counts.get('neutral', 0),
            'start_ratio': label_counts.get('start', 0) / len(file_df) * 100,
            'stop_ratio': label_counts.get('stop', 0) / len(file_df) * 100,
            'neutral_ratio': label_counts.get('neutral', 0) / len(file_df) * 100,
        })

    return pd.DataFrame(summary)


def get_cut_comparison_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary of cut vs not-cut comparison for each file.
    """
    # Separate cut and not-cut data
    not_cut_df = df[df['is_not_cut'] == True]
    cut_df = df[df['is_not_cut'] == False]

    # Get file summaries for both
    not_cut_summary = {}
    for file_id in not_cut_df['file_id'].unique():
        file_df = not_cut_df[not_cut_df['file_id'] == file_id]
        not_cut_summary[file_id] = {
            'start_count': len(file_df[file_df['label'] == 'start']),
            'stop_count': len(file_df[file_df['label'] == 'stop']),
            'total_frames': len(file_df),
        }

    cut_summary = {}
    for file_id in cut_df['file_id'].unique():
        file_df = cut_df[cut_df['file_id'] == file_id]
        cut_summary[file_id] = {
            'start_count': len(file_df[file_df['label'] == 'start']),
            'stop_count': len(file_df[file_df['label'] == 'stop']),
            'total_frames': len(file_df),
        }

    # Create comparison summary
    comparison_summary = []
    all_file_ids = set(not_cut_summary.keys()) | set(cut_summary.keys())

    for file_id in sorted(all_file_ids):
        not_cut_data = not_cut_summary.get(file_id, {'start_count': 0, 'stop_count': 0, 'total_frames': 0})
        cut_data = cut_summary.get(file_id, {'start_count': 0, 'stop_count': 0, 'total_frames': 0})

        not_cut_start_stop = not_cut_data['start_count'] + not_cut_data['stop_count']
        cut_start_stop = cut_data['start_count'] + cut_data['stop_count']

        # Calculate ratio
        if not_cut_start_stop > 0:
            ratio = cut_start_stop / not_cut_start_stop
        else:
            ratio = 0

        comparison_summary.append({
            'file_id': file_id,
            'not_cut_total': not_cut_data['total_frames'],
            'not_cut_start_stop': not_cut_start_stop,
            'cut_total': cut_data['total_frames'],
            'cut_start_stop': cut_start_stop,
            'ratio': ratio,
        })

    return pd.DataFrame(comparison_summary)


def compare_cut_and_not_cut(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare start/stop frame counts between cut and not-cut data.

    Files in 'not_cut' directory should have more start/stop labels than
    the corresponding 'cut' files since cutting removes the motion boundaries.

    Returns a DataFrame with comparison information.
    """
    df = df.copy()

    # Separate cut and not-cut data
    not_cut_df = df[df['is_not_cut'] == True]
    cut_df = df[df['is_not_cut'] == False]

    # Get file summaries for both
    not_cut_summary = {}
    for file_id in not_cut_df['file_id'].unique():
        file_df = not_cut_df[not_cut_df['file_id'] == file_id]
        not_cut_summary[file_id] = {
            'start_count': len(file_df[file_df['label'] == 'start']),
            'stop_count': len(file_df[file_df['label'] == 'stop']),
            'total_frames': len(file_df),
        }

    cut_summary = {}
    for file_id in cut_df['file_id'].unique():
        file_df = cut_df[cut_df['file_id'] == file_id]
        cut_summary[file_id] = {
            'start_count': len(file_df[file_df['label'] == 'start']),
            'stop_count': len(file_df[file_df['label'] == 'stop']),
            'total_frames': len(file_df),
        }

    # Compare and find outliers
    comparison_results = []

    for file_id in not_cut_summary.keys():
        not_cut_data = not_cut_summary[file_id]
        cut_data = cut_summary.get(file_id, {'start_count': 0, 'stop_count': 0, 'total_frames': 0})

        not_cut_start_stop = not_cut_data['start_count'] + not_cut_data['stop_count']
        cut_start_stop = cut_data['start_count'] + cut_data['stop_count']

        # Calculate ratio of start/stop in cut vs not-cut
        if not_cut_start_stop > 0:
            ratio = cut_start_stop / not_cut_start_stop
        else:
            ratio = 0

        # Check if cut has significantly more start/stop than expected
        # (this would indicate the cut file was incorrectly processed)
        if ratio > CUT_START_STOP_RATIO_THRESHOLD and cut_start_stop > 0:
            comparison_results.append({
                'file_id': file_id,
                'not_cut_start': not_cut_data['start_count'],
                'not_cut_stop': not_cut_data['stop_count'],
                'not_cut_start_stop': not_cut_start_stop,
                'cut_start': cut_data['start_count'],
                'cut_stop': cut_data['stop_count'],
                'cut_start_stop': cut_start_stop,
                'ratio': ratio,
                'reason': f'cut has {ratio*100:.1f}% of not-cut start/stop frames'
            })

    return pd.DataFrame(comparison_results)


def interactive_correction(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    Interactive correction of labels.

    Args:
        df: DataFrame with classification data
        output_path: Path to save corrected data

    Returns:
        Corrected DataFrame
    """
    print("\n" + "=" * 60)
    print("Interactive Label Correction")
    print("=" * 60)
    print("\nCommands:")
    print("  's' - change label to 'start'")
    print("  't' - change label to 'stop' (for 'stop' label)")
    print("  'n' - change label to 'neutral'")
    print("  'k' - keep current label")
    print("  'q' - quit without saving")
    print("  'a' - accept all remaining changes")
    print()

    corrected_df = df.copy()

    # Get unique file_ids
    file_ids = corrected_df['file_id'].unique()

    for file_id in file_ids:
        print(f"\n{'=' * 60}")
        print(f"Processing file: {file_id}")
        print(f"{'=' * 60}")

        mask = corrected_df['file_id'] == file_id
        indices = corrected_df[mask].index.tolist()

        for i, idx in enumerate(indices):
            current_row = corrected_df.loc[idx]
            current_label = current_row['label']

            # Get previous and next rows for context
            prev_row = None
            next_row = None

            if i > 0:
                prev_idx = indices[i - 1]
                prev_row = corrected_df.loc[prev_idx]

            if i < len(indices) - 1:
                next_idx = indices[i + 1]
                next_row = corrected_df.loc[next_idx]

            # Calculate velocity changes
            head_speed_change = 0
            left_hand_speed_change = 0
            right_hand_speed_change = 0

            if prev_row is not None:
                head_speed_change = abs(current_row['head_speed'] - prev_row['head_speed'])
                left_hand_speed_change = abs(current_row['left_hand_speed'] - prev_row['left_hand_speed'])
                right_hand_speed_change = abs(current_row['right_hand_speed'] - prev_row['right_hand_speed'])

            # Print frame info
            print(f"\nFrame {current_row['FrameNo']} (index {idx}):")
            print(f"  Current label: {current_label}")
            print(f"  Head speed: {current_row['head_speed']:.3f}, "
                  f"Left hand speed: {current_row['left_hand_speed']:.3f}, "
                  f"Right hand speed: {current_row['right_hand_speed']:.3f}")
            print(f"  Head speed change: {head_speed_change:.3f}, "
                  f"Left hand speed change: {left_hand_speed_change:.3f}, "
                  f"Right hand speed change: {right_hand_speed_change:.3f}")

            # Show previous and next labels if available
            if prev_row is not None:
                print(f"  Previous label: {prev_row['label']}")
            if next_row is not None:
                print(f"  Next label: {next_row['label']}")

            # Get user input
            while True:
                user_input = input(f"  Action (s/t/n/k/q/a): ").strip().lower()

                if user_input == 's':
                    corrected_df.loc[idx, 'label'] = 'start'
                    print(f"  Changed to: start")
                    break
                elif user_input == 't':
                    if current_label == 'stop':
                        print("  Already 'stop', keeping...")
                        break
                    corrected_df.loc[idx, 'label'] = 'stop'
                    print(f"  Changed to: stop")
                    break
                elif user_input == 'n':
                    corrected_df.loc[idx, 'label'] = 'neutral'
                    print(f"  Changed to: neutral")
                    break
                elif user_input == 'k':
                    print(f"  Kept: {current_label}")
                    break
                elif user_input == 'q':
                    print("\nQuitting without saving...")
                    return df
                elif user_input == 'a':
                    print("\nAccepting all remaining changes...")
                    # Save the corrected data
                    corrected_df.to_csv(output_path, index=False)
                    print(f"Saved corrected data to: {output_path}")
                    return corrected_df
                else:
                    print("  Invalid input. Please enter s, t, n, k, q, or a.")

    # Save the corrected data
    corrected_df.to_csv(output_path, index=False)
    print(f"\nSaved corrected data to: {output_path}")

    return corrected_df


def check_label_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for unusual label transitions that might indicate errors.

    Unusual transitions:
    - start -> stop (very short duration)
    - stop -> start (very short duration)
    - Multiple consecutive start or stop labels
    """
    df = df.copy()

    # Get unique file_ids
    file_ids = df['file_id'].unique()

    unusual_transitions = []

    for file_id in file_ids:
        file_df = df[df['file_id'] == file_id].sort_values('FrameNo')
        indices = file_df.index.tolist()

        for i in range(len(indices) - 1):
            curr_idx = indices[i]
            next_idx = indices[i + 1]

            curr_label = file_df.loc[curr_idx, 'label']
            next_label = file_df.loc[next_idx, 'label']

            # Check for start -> stop or stop -> start transitions
            if (curr_label == 'start' and next_label == 'stop') or \
               (curr_label == 'stop' and next_label == 'start'):
                frame_diff = file_df.loc[next_idx, 'FrameNo'] - file_df.loc[curr_idx, 'FrameNo']
                if frame_diff < 5:  # Less than 5 frames between transitions
                    unusual_transitions.append({
                        'file_id': file_id,
                        'frame_no': file_df.loc[curr_idx, 'FrameNo'],
                        'current_label': curr_label,
                        'next_label': next_label,
                        'frame_diff': frame_diff,
                        'reason': f'{curr_label} -> {next_label} (very short duration)'
                    })

    return pd.DataFrame(unusual_transitions)


def print_summary(df: pd.DataFrame, title: str = "Summary"):
    """
    Print a summary of the dataset.
    """
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)

    print("\nLabel distribution:")
    print(df['label'].value_counts())

    print("\nLabel distribution by file:")
    for file_id in df['file_id'].unique():
        file_df = df[df['file_id'] == file_id]
        print(f"\n  {file_id}:")
        print(f"    Total frames: {len(file_df)}")
        print(f"    Start: {len(file_df[file_df['label'] == 'start'])}")
        print(f"    Stop: {len(file_df[file_df['label'] == 'stop'])}")
        print(f"    Neutral: {len(file_df[file_df['label'] == 'neutral'])}")


def main():
    """
    Main function to check and correct classification data.
    """
    print("=" * 60)
    print("Classification Data Checker")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load combined data
    combined_path = CLASSIFICATION_DATA_DIR / "combined_classification_data.csv"

    if not combined_path.exists():
        print(f"Error: Combined data file not found at {combined_path}")
        print("Please run prepare_classification_data.py first.")
        sys.exit(1)

    print(f"\nLoading data from: {combined_path}")
    df = pd.read_csv(combined_path)

    print(f"Total frames loaded: {len(df)}")

    # Print initial summary
    print_summary(df, "Initial Summary")

    # Detect outliers based on velocity/acceleration
    print("\n" + "=" * 60)
    print("Detecting Outliers")
    print("=" * 60)

    outliers = detect_outliers(df)
    print(f"\nFound {len(outliers)} frames with potential outliers")

    if len(outliers) > 0:
        print("\nTop 10 outlier frames:")
        print(outliers[['FrameNo', 'file_id', 'label', 'outlier_reason']].head(10).to_string())

    # Analyze label distribution
    print("\n" + "=" * 60)
    print("Analyzing Label Distribution")
    print("=" * 60)

    mislabeled = analyze_label_distribution(df)
    print(f"\nFound {len(mislabeled)} potentially mislabeled frames")

    if len(mislabeled) > 0:
        print("\nTop 10 potentially mislabeled frames:")
        print(mislabeled[['FrameNo', 'file_id', 'label', 'mislabel_reason']].head(10).to_string())

    # Check label transitions
    print("\n" + "=" * 60)
    print("Checking Label Transitions")
    print("=" * 60)

    unusual_transitions = check_label_transitions(df)
    print(f"\nFound {len(unusual_transitions)} unusual label transitions")

    if len(unusual_transitions) > 0:
        print("\nUnusual transitions:")
        print(unusual_transitions.to_string())

    # Compare cut and not-cut data
    print("\n" + "=" * 60)
    print("Comparing Cut and Not-Cut Data")
    print("=" * 60)

    cut_comparison = compare_cut_and_not_cut(df)
    print(f"\nFound {len(cut_comparison)} files with potential cut frame issues")

    if len(cut_comparison) > 0:
        print("\nCut frame comparison outliers:")
        print(cut_comparison.to_string())

    # Get cut comparison summary
    print("\n" + "=" * 60)
    print("Cut vs Not-Cut Summary")
    print("=" * 60)

    cut_summary = get_cut_comparison_summary(df)
    print(cut_summary.to_string())

    # Get file summary
    print("\n" + "=" * 60)
    print("File Summary")
    print("=" * 60)

    file_summary = get_file_summary(df)
    print(file_summary.to_string())

    # Identify outlier files
    print("\n" + "=" * 60)
    print("Identifying Outlier Files")
    print("=" * 60)

    outlier_files = set()

    # Add files with cut comparison issues
    for _, row in cut_comparison.iterrows():
        outlier_files.add(row['file_id'])

    print(f"\nFound {len(outlier_files)} files with potential issues")

    if len(outlier_files) > 0:
        print("\nOutlier files:")
        for file_id in sorted(outlier_files):
            print(f"  - {file_id}")

    # Ask user if they want to correct labels
    print("\n" + "=" * 60)
    print("Correction Options")
    print("=" * 60)

    while True:
        choice = input("\nDo you want to interactively correct labels? (y/n): ").strip().lower()

        if choice == 'y':
            output_path = OUTPUT_DIR / "corrected_classification_data.csv"
            corrected_df = interactive_correction(df, output_path)

            # Print corrected summary
            print_summary(corrected_df, "Corrected Summary")
            break
        elif choice == 'n':
            print("\nExiting without corrections...")
            break
        else:
            print("Please enter 'y' or 'n'.")

    # Ask user if they want to filter outlier files
    print("\n" + "=" * 60)
    print("Filter Options")
    print("=" * 60)

    while True:
        filter_choice = input("\nDo you want to filter out outlier files? (y/n): ").strip().lower()

        if filter_choice == 'y':
            # Filter out outlier files
            filtered_df = df[~df['file_id'].isin(outlier_files)].copy()
            filtered_path = OUTPUT_DIR / "filtered_classification_data.csv"
            filtered_df.to_csv(filtered_path, index=False)

            print(f"\nFiltered {len(outlier_files)} outlier files")
            print(f"Original frames: {len(df)}")
            print(f"Filtered frames: {len(filtered_df)}")
            print(f"Removed: {len(df) - len(filtered_df)} frames")
            print(f"Saved filtered data to: {filtered_path}")

            # Print filtered summary
            print_summary(filtered_df, "Filtered Summary")
            break
        elif filter_choice == 'n':
            print("\nSkipping filter...")
            break
        else:
            print("Please enter 'y' or 'n'.")

    print("\n" + "=" * 60)
    print("Check complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
