#!/usr/bin/env python3
"""
Script to check classification data for outliers and allow for data correction.
Analyzes the prepared classification data and identifies potential issues.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Define paths
BASE_DIR = Path(__file__).parent
CLASSIFICATION_DATA_DIR = BASE_DIR / "classification_data"
OUTPUT_DIR = BASE_DIR / "check_general_results"
CORRECTIONS_DIR = BASE_DIR / "general_corrections"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)


class DataChecker:
    """Main class for checking classification data for outliers."""

    def __init__(self, data_dir: Path = CLASSIFICATION_DATA_DIR):
        self.data_dir = data_dir
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.outliers: Dict[str, pd.DataFrame] = {}
        self.corrections_log: List[Dict[str, Any]] = []

    def load_data(self) -> None:
        """Load all CSV files from the classification data directory."""
        print("=" * 70)
        print("Loading Classification Data")
        print("=" * 70)

        csv_files = list(self.data_dir.glob("*.csv"))

        if not csv_files:
            print(f"No CSV files found in {self.data_dir}")
            return

        for csv_file in csv_files:
            print(f"\nLoading: {csv_file.name}")
            try:
                df = pd.read_csv(csv_file)
                self.datasets[csv_file.stem] = df
                print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"  Error loading {csv_file.name}: {e}")

    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns (excluding metadata and label)."""
        if not self.datasets:
            return []

        # Get columns from first dataset
        all_columns = list(self.datasets[list(self.datasets.keys())[0]].columns)

        # Exclude metadata columns
        metadata_cols = ['FrameNo', 'file_id', 'is_not_cut', 'label']

        feature_cols = [col for col in all_columns if col not in metadata_cols]
        return feature_cols

    def detect_position_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers in position data using z-score method.
        Returns DataFrame with outlier flags.
        """
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        outlier_info = {}

        # Position columns to check
        pos_cols = [col for col in df.columns if col.endswith('_x') or col.endswith('_y') or col.endswith('_z')]

        for col in pos_cols:
            # Calculate z-scores
            mean = df[col].mean()
            std = df[col].std()

            if std > 0:
                z_scores = np.abs((df[col] - mean) / std)
                outliers = z_scores > threshold

                if outliers.any():
                    outlier_mask = outlier_mask | outliers
                    outlier_info[col] = {
                        'count': outliers.sum(),
                        'mean': mean,
                        'std': std,
                        'min_outlier': df.loc[outliers, col].min(),
                        'max_outlier': df.loc[outliers, col].max()
                    }

        return pd.DataFrame({
            'is_outlier': outlier_mask,
            'outlier_reason': 'position_outlier'
        })

    def detect_velocity_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers in velocity data using z-score method.
        Returns DataFrame with outlier flags.
        """
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        outlier_info = {}

        # Velocity columns to check
        vel_cols = [col for col in df.columns if col.endswith('_vx') or col.endswith('_vy') or col.endswith('_vz')]

        for col in vel_cols:
            # Calculate z-scores
            mean = df[col].mean()
            std = df[col].std()

            if std > 0:
                z_scores = np.abs((df[col] - mean) / std)
                outliers = z_scores > threshold

                if outliers.any():
                    outlier_mask = outlier_mask | outliers
                    outlier_info[col] = {
                        'count': outliers.sum(),
                        'mean': mean,
                        'std': std,
                        'min_outlier': df.loc[outliers, col].min(),
                        'max_outlier': df.loc[outliers, col].max()
                    }

        return pd.DataFrame({
            'is_outlier': outlier_mask,
            'outlier_reason': 'velocity_outlier'
        })

    def detect_acceleration_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers in acceleration data using z-score method.
        Returns DataFrame with outlier flags.
        """
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        outlier_info = {}

        # Acceleration columns to check
        accel_cols = [col for col in df.columns if col.endswith('_ax') or col.endswith('_ay') or col.endswith('_az')]

        for col in accel_cols:
            # Calculate z-scores
            mean = df[col].mean()
            std = df[col].std()

            if std > 0:
                z_scores = np.abs((df[col] - mean) / std)
                outliers = z_scores > threshold

                if outliers.any():
                    outlier_mask = outlier_mask | outliers
                    outlier_info[col] = {
                        'count': outliers.sum(),
                        'mean': mean,
                        'std': std,
                        'min_outlier': df.loc[outliers, col].min(),
                        'max_outlier': df.loc[outliers, col].max()
                    }

        return pd.DataFrame({
            'is_outlier': outlier_mask,
            'outlier_reason': 'acceleration_outlier'
        })

    def detect_speed_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers in speed data using z-score method.
        Returns DataFrame with outlier flags.
        """
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        outlier_info = {}

        # Speed columns to check
        speed_cols = [col for col in df.columns if col.endswith('_speed')]

        for col in speed_cols:
            # Calculate z-scores
            mean = df[col].mean()
            std = df[col].std()

            if std > 0:
                z_scores = np.abs((df[col] - mean) / std)
                outliers = z_scores > threshold

                if outliers.any():
                    outlier_mask = outlier_mask | outliers
                    outlier_info[col] = {
                        'count': outliers.sum(),
                        'mean': mean,
                        'std': std,
                        'min_outlier': df.loc[outliers, col].min(),
                        'max_outlier': df.loc[outliers, col].max()
                    }

        return pd.DataFrame({
            'is_outlier': outlier_mask,
            'outlier_reason': 'speed_outlier'
        })

    def detect_distance_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers in distance data using z-score method.
        Returns DataFrame with outlier flags.
        """
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        outlier_info = {}

        # Distance columns to check
        dist_cols = [col for col in df.columns if 'to_' in col and col.endswith('_distance')]

        for col in dist_cols:
            # Calculate z-scores
            mean = df[col].mean()
            std = df[col].std()

            if std > 0:
                z_scores = np.abs((df[col] - mean) / std)
                outliers = z_scores > threshold

                if outliers.any():
                    outlier_mask = outlier_mask | outliers
                    outlier_info[col] = {
                        'count': outliers.sum(),
                        'mean': mean,
                        'std': std,
                        'min_outlier': df.loc[outliers, col].min(),
                        'max_outlier': df.loc[outliers, col].max()
                    }

        return pd.DataFrame({
            'is_outlier': outlier_mask,
            'outlier_reason': 'distance_outlier'
        })

    def detect_label_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect potential label issues based on motion patterns.
        Returns DataFrame with issue flags.
        """
        issue_mask = pd.Series([False] * len(df), index=df.index)
        issue_info = {}

        # Check for inconsistent labels
        # Start should have increasing speed, stop should have decreasing speed

        for file_id in df['file_id'].unique():
            file_mask = df['file_id'] == file_id
            file_df = df[file_mask].copy()

            if len(file_df) < 2:
                continue

            # Get indices
            indices = file_df.index.tolist()

            for i, idx in enumerate(indices):
                if i >= len(indices) - 1:
                    break

                current_speed = file_df.loc[idx, 'head_speed']
                next_speed = file_df.loc[indices[i + 1], 'head_speed']
                delta_speed = next_speed - current_speed
                current_label = file_df.loc[idx, 'label']

                # Check for label inconsistencies
                if current_label == 'start' and delta_speed < -0.3:
                    issue_mask.loc[idx] = True
                    if 'label_inconsistency' not in issue_info:
                        issue_info['label_inconsistency'] = {'count': 0}
                    issue_info['label_inconsistency']['count'] += 1

                elif current_label == 'stop' and delta_speed > 0.3:
                    issue_mask.loc[idx] = True
                    if 'label_inconsistency' not in issue_info:
                        issue_info['label_inconsistency'] = {'count': 0}
                    issue_info['label_inconsistency']['count'] += 1

        return pd.DataFrame({
            'is_outlier': issue_mask,
            'outlier_reason': 'label_issue'
        })

    def detect_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect rows with missing or invalid data.
        Returns DataFrame with missing data flags.
        """
        # Start with an empty boolean mask
        missing_mask = pd.Series([False] * len(df), index=df.index, dtype=bool)

        # Check for NaN values
        nan_mask = pd.isna(df).any(axis=1)
        missing_mask = missing_mask | nan_mask

        # Also check for extremely large values that might indicate data corruption
        for col in df.select_dtypes(include=[np.number]).columns:
            # Check for NaN, Inf, or extremely large values (> 1e6)
            # Use .values to get numpy array and avoid pandas dtype issues
            col_values = df[col].values
            is_nan = pd.isna(df[col]).values
            is_large = np.abs(col_values) > 1e6
            is_small = np.abs(col_values) < 1e-10
            invalid_mask = is_nan | is_large | is_small
            missing_mask = missing_mask | pd.Series(invalid_mask, index=df.index, dtype=bool)

        return pd.DataFrame({
            'is_outlier': missing_mask,
            'outlier_reason': 'missing_or_invalid_data'
        })

    def analyze_dataset(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on a dataset.
        """
        print(f"\n{'=' * 70}")
        print(f"Analyzing: {dataset_name}")
        print(f"{'=' * 70}")

        results = {
            'dataset_name': dataset_name,
            'total_rows': len(df),
            'outliers': {},
            'summary': {}
        }

        # Detect different types of outliers
        print("\n1. Checking for position outliers...")
        pos_outliers = self.detect_position_outliers(df)
        results['outliers']['position'] = pos_outliers

        print("2. Checking for velocity outliers...")
        vel_outliers = self.detect_velocity_outliers(df)
        results['outliers']['velocity'] = vel_outliers

        print("3. Checking for acceleration outliers...")
        accel_outliers = self.detect_acceleration_outliers(df)
        results['outliers']['acceleration'] = accel_outliers

        print("4. Checking for speed outliers...")
        speed_outliers = self.detect_speed_outliers(df)
        results['outliers']['speed'] = speed_outliers

        print("5. Checking for distance outliers...")
        dist_outliers = self.detect_distance_outliers(df)
        results['outliers']['distance'] = dist_outliers

        print("6. Checking for label issues...")
        label_issues = self.detect_label_issues(df)
        results['outliers']['label_issues'] = label_issues

        print("7. Checking for missing/invalid data...")
        missing_data = self.detect_missing_data(df)
        results['outliers']['missing_data'] = missing_data

        # Combine all outliers
        all_outlier_masks = pd.DataFrame({
            'position': pos_outliers['is_outlier'],
            'velocity': vel_outliers['is_outlier'],
            'acceleration': accel_outliers['is_outlier'],
            'speed': speed_outliers['is_outlier'],
            'distance': dist_outliers['is_outlier'],
            'label_issues': label_issues['is_outlier'],
            'missing_data': missing_data['is_outlier']
        })

        total_outliers = all_outlier_masks.any(axis=1).sum()
        results['summary']['total_outliers'] = int(total_outliers)
        results['summary']['outlier_percentage'] = float(total_outliers / len(df) * 100)

        # Count outliers by type
        for col in all_outlier_masks.columns:
            results['summary'][f'{col}_outliers'] = int(all_outlier_masks[col].sum())

        # Print summary
        print(f"\n{'=' * 70}")
        print("Summary")
        print(f"{'=' * 70}")
        print(f"Total rows: {len(df)}")
        print(f"Total outliers: {total_outliers} ({results['summary']['outlier_percentage']:.2f}%)")
        print(f"\nOutliers by type:")
        for col in all_outlier_masks.columns:
            count = all_outlier_masks[col].sum()
            if count > 0:
                print(f"  {col}: {count}")

        # Label distribution
        print(f"\nLabel distribution:")
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            for label, count in label_counts.items():
                print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")

        # File distribution
        print(f"\nFile distribution:")
        file_counts = df['file_id'].value_counts()
        for file_id, count in file_counts.items():
            print(f"  {file_id}: {count} frames")

        return results

    def analyze_all_datasets(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze all loaded datasets.
        """
        all_results = {}

        for dataset_name, df in self.datasets.items():
            results = self.analyze_dataset(df, dataset_name)
            all_results[dataset_name] = results
            self.outliers[dataset_name] = results

        return all_results

    def save_analysis_report(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """Save analysis results to a report file."""
        report_path = OUTPUT_DIR / "analysis_report.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("CLASSIFICATION DATA ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")

            for dataset_name, results in all_results.items():
                f.write(f"\n{'=' * 70}\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"{'=' * 70}\n\n")

                f.write(f"Total rows: {results['total_rows']}\n")
                f.write(f"Total outliers: {results['summary']['total_outliers']}\n")
                f.write(f"Outlier percentage: {results['summary']['outlier_percentage']:.2f}%\n\n")

                f.write("Outliers by type:\n")
                for key, value in results['summary'].items():
                    if key.endswith('_outliers'):
                        f.write(f"  {key}: {value}\n")

                f.write("\nLabel distribution:\n")
                if dataset_name in self.datasets:
                    df = self.datasets[dataset_name]
                    if 'label' in df.columns:
                        label_counts = df['label'].value_counts()
                        for label, count in label_counts.items():
                            f.write(f"  {label}: {count} ({count/len(df)*100:.1f}%)\n")

        print(f"\nAnalysis report saved to: {report_path}")

    def save_outlier_indices(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """Save outlier indices to JSON files for correction reference."""
        for dataset_name, results in all_results.items():
            if dataset_name not in self.datasets:
                continue

            df = self.datasets[dataset_name]

            # Combine all outlier masks
            all_outlier_masks = pd.DataFrame({
                'position': results['outliers']['position']['is_outlier'],
                'velocity': results['outliers']['velocity']['is_outlier'],
                'acceleration': results['outliers']['acceleration']['is_outlier'],
                'speed': results['outliers']['speed']['is_outlier'],
                'distance': results['outliers']['distance']['is_outlier'],
                'label_issues': results['outliers']['label_issues']['is_outlier'],
                'missing_data': results['outliers']['missing_data']['is_outlier']
            })

            # Get outlier indices
            outlier_indices = all_outlier_masks.any(axis=1)
            outlier_rows = df[outlier_indices].copy()

            # Add outlier reasons
            outlier_reasons = []
            for idx in outlier_rows.index:
                reasons = []
                for col in all_outlier_masks.columns:
                    if all_outlier_masks.loc[idx, col]:
                        reasons.append(col)
                outlier_reasons.append(','.join(reasons))

            outlier_rows['outlier_reasons'] = outlier_reasons

            # Save to CSV
            output_path = OUTPUT_DIR / f"{dataset_name}_outliers.csv"
            outlier_rows.to_csv(output_path, index=False)

            # Save to JSON for easier correction
            json_path = OUTPUT_DIR / f"{dataset_name}_outliers.json"

            # Convert to list of dicts for JSON
            outlier_data = []
            for idx, row in outlier_rows.iterrows():
                row_dict = row.to_dict()
                row_dict['original_index'] = idx
                outlier_data.append(row_dict)

            with open(json_path, 'w') as f:
                json.dump(outlier_data, f, indent=2, default=str)

            print(f"Outlier indices saved to: {output_path}")
            print(f"Outlier JSON saved to: {json_path}")


class DataCorrector:
    """Class for correcting outliers in the data."""

    def __init__(self, data_checker: DataChecker):
        self.data_checker = data_checker
        self.corrections: Dict[str, List[Dict[str, Any]]] = {}

    def load_existing_corrections(self) -> None:
        """Load any existing correction files."""
        correction_files = list(CORRECTIONS_DIR.glob("*.json"))

        for file_path in correction_files:
            try:
                with open(file_path, 'r') as f:
                    self.corrections[file_path.stem] = json.load(f)
                print(f"Loaded existing corrections from: {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    def interactive_correction(self, dataset_name: str, output_name: str = None) -> pd.DataFrame:
        """
        Interactive mode for correcting outliers.
        """
        if dataset_name not in self.data_checker.datasets:
            print(f"Dataset '{dataset_name}' not found.")
            return None

        df = self.data_checker.datasets[dataset_name].copy()

        # Load outlier information
        outlier_json_path = OUTPUT_DIR / f"{dataset_name}_outliers.json"

        if not outlier_json_path.exists():
            print(f"No outlier file found for {dataset_name}. Run analysis first.")
            return df

        with open(outlier_json_path, 'r') as f:
            outliers = json.load(f)

        print(f"\n{'=' * 70}")
        print(f"Interactive Correction Mode: {dataset_name}")
        print(f"{'=' * 70}")
        print(f"Total outliers to review: {len(outliers)}")
        print("\nCommands:")
        print("  'c' - Correct (keep as is)")
        print("  'd' - Delete row")
        print("  'i' - Interpolate (for consecutive outliers)")
        print("  's' - Skip (save progress and exit)")
        print("  'q' - Quit without saving")
        print("  'a' - Apply all remaining as 'correct'")
        print()

        corrections = []
        corrected_indices = set()

        for i, outlier in enumerate(outliers):
            idx = outlier['original_index']

            # Skip if already corrected
            if idx in corrected_indices:
                continue

            print(f"\n{'=' * 70}")
            print(f"Outlier {i+1}/{len(outliers)}")
            print(f"File: {outlier.get('file_id', 'N/A')}")
            print(f"Frame: {outlier.get('FrameNo', 'N/A')}")
            print(f"Reasons: {outlier.get('outlier_reasons', 'N/A')}")
            print(f"{'=' * 70}")

            # Show key values
            print("\nKey values:")
            for key in ['head_x', 'head_y', 'head_z', 'head_speed', 'head_accel', 'label']:
                if key in outlier:
                    print(f"  {key}: {outlier[key]}")

            # Show previous and next frame values
            prev_idx = idx - 1 if idx > 0 else None
            next_idx = idx + 1 if idx < len(df) - 1 else None

            if prev_idx is not None and prev_idx in df.index:
                prev_row = df.loc[prev_idx]
                print(f"\nPrevious frame ({prev_idx}):")
                for key in ['head_x', 'head_y', 'head_z', 'head_speed']:
                    if key in prev_row:
                        print(f"  {key}: {prev_row[key]}")

            if next_idx is not None and next_idx in df.index:
                next_row = df.loc[next_idx]
                print(f"\nNext frame ({next_idx}):")
                for key in ['head_x', 'head_y', 'head_z', 'head_speed']:
                    if key in next_row:
                        print(f"  {key}: {next_row[key]}")

            # Get user input
            while True:
                choice = input("\nAction (c/d/i/s/q/a): ").strip().lower()

                if choice == 'c':
                    print("Keeping as is.")
                    corrections.append({
                        'index': idx,
                        'action': 'keep',
                        'reason': 'reviewed'
                    })
                    corrected_indices.add(idx)
                    break

                elif choice == 'd':
                    print("Marking for deletion.")
                    corrections.append({
                        'index': idx,
                        'action': 'delete',
                        'reason': 'reviewed'
                    })
                    corrected_indices.add(idx)
                    break

                elif choice == 'i':
                    print("Marking for interpolation.")
                    corrections.append({
                        'index': idx,
                        'action': 'interpolate',
                        'reason': 'reviewed'
                    })
                    corrected_indices.add(idx)
                    break

                elif choice == 's':
                    print("Saving progress and exiting...")
                    self._save_corrections(dataset_name, corrections, output_name)
                    return df

                elif choice == 'q':
                    print("Quitting without saving...")
                    return df

                elif choice == 'a':
                    print("Applying 'keep' to all remaining outliers...")
                    for j in range(i, len(outliers)):
                        o = outliers[j]
                        if o['original_index'] not in corrected_indices:
                            corrections.append({
                                'index': o['original_index'],
                                'action': 'keep',
                                'reason': 'auto-applied'
                            })
                            corrected_indices.add(o['original_index'])
                    break

                else:
                    print("Invalid choice. Please try again.")

        # Apply corrections
        if corrections:
            self._apply_corrections(df, corrections)
            self._save_corrections(dataset_name, corrections, output_name)

        return df

    def _apply_corrections(self, df: pd.DataFrame, corrections: List[Dict[str, Any]]) -> pd.DataFrame:
        """Apply corrections to the DataFrame."""
        df = df.copy()

        for correction in corrections:
            idx = correction['index']
            action = correction['action']

            if action == 'delete':
                if idx in df.index:
                    df = df.drop(idx)

            elif action == 'interpolate':
                if idx in df.index:
                    # Find previous and next valid indices
                    prev_idx = None
                    next_idx = None

                    for i in range(idx - 1, -1, -1):
                        if i not in [c['index'] for c in corrections if c['action'] == 'delete']:
                            prev_idx = i
                            break

                    for i in range(idx + 1, len(df)):
                        if i not in [c['index'] for c in corrections if c['action'] == 'delete']:
                            next_idx = i
                            break

                    if prev_idx is not None and next_idx is not None:
                        # Linear interpolation
                        for col in df.select_dtypes(include=[np.number]).columns:
                            if col not in ['FrameNo', 'file_id', 'is_not_cut', 'label']:
                                prev_val = df.loc[prev_idx, col]
                                next_val = df.loc[next_idx, col]
                                df.loc[idx, col] = (prev_val + next_val) / 2

        return df

    def _save_corrections(self, dataset_name: str, corrections: List[Dict[str, Any]], output_name: str = None) -> None:
        """Save corrections to a file."""
        if output_name is None:
            output_name = dataset_name

        corrections_path = CORRECTIONS_DIR / f"{output_name}_corrections.json"

        with open(corrections_path, 'w') as f:
            json.dump(corrections, f, indent=2)

        print(f"\nCorrections saved to: {corrections_path}")
        print(f"Total corrections: {len(corrections)}")


def generate_statistics_report(checker: DataChecker, all_results: Dict[str, Dict[str, Any]]) -> None:
    """Generate a comprehensive statistics report."""
    stats_path = OUTPUT_DIR / "statistics_report.txt"

    with open(stats_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("COMPREHENSIVE STATISTICS REPORT\n")
        f.write("=" * 70 + "\n\n")

        for dataset_name, results in all_results.items():
            f.write(f"\n{'=' * 70}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"{'=' * 70}\n\n")

            if dataset_name not in checker.datasets:
                continue

            df = checker.datasets[dataset_name]

            # Numeric columns statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            f.write("Numeric Column Statistics:\n\n")
            f.write(f"{'Column':<40} {'Mean':>15} {'Std':>15} {'Min':>15} {'Max':>15}\n")
            f.write("-" * 100 + "\n")

            for col in numeric_cols:
                if col in ['FrameNo', 'file_id', 'is_not_cut', 'label']:
                    continue
                mean = df[col].mean()
                std = df[col].std()
                min_val = df[col].min()
                max_val = df[col].max()
                f.write(f"{col:<40} {mean:>15.6f} {std:>15.6f} {min_val:>15.6f} {max_val:>15.6f}\n")

    print(f"Statistics report saved to: {stats_path}")


def main():
    """Main function to run the data checking process."""
    print("=" * 70)
    print("CLASSIFICATION DATA CHECKER")
    print("=" * 70)

    # Initialize checker
    checker = DataChecker()

    # Load data
    checker.load_data()

    if not checker.datasets:
        print("No data loaded. Exiting.")
        return

    # Analyze all datasets
    all_results = checker.analyze_all_datasets()

    # Save analysis reports
    checker.save_analysis_report(all_results)
    checker.save_outlier_indices(all_results)

    # Generate statistics report
    generate_statistics_report(checker, all_results)

    # Interactive correction mode
    print(f"\n{'=' * 70}")
    print("Interactive Correction Mode")
    print(f"{'=' * 70}")

    corrector = DataCorrector(checker)
    corrector.load_existing_corrections()

    while True:
        print("\nAvailable datasets:")
        for i, dataset_name in enumerate(checker.datasets.keys(), 1):
            total = all_results[dataset_name]['summary']['total_outliers']
            print(f"  {i}. {dataset_name} ({total} outliers)")

        print("\nOptions:")
        print("  Enter number to correct a dataset")
        print("  'a' to correct all datasets")
        print("  'q' to quit")

        choice = input("\nChoice: ").strip().lower()

        if choice == 'q':
            print("Exiting...")
            break

        elif choice == 'a':
            for dataset_name in checker.datasets.keys():
                output_name = dataset_name.replace('_classification_data', '')
                print(f"\n{'=' * 70}")
                print(f"Correcting: {dataset_name}")
                print(f"{'=' * 70}")
                corrector.interactive_correction(dataset_name, output_name)

        else:
            try:
                idx = int(choice) - 1
                dataset_names = list(checker.datasets.keys())
                if 0 <= idx < len(dataset_names):
                    dataset_name = dataset_names[idx]
                    output_name = dataset_name.replace('_classification_data', '')
                    corrector.interactive_correction(dataset_name, output_name)
                else:
                    print("Invalid choice.")
            except ValueError:
                print("Invalid choice.")


if __name__ == "__main__":
    main()

"""
## Features

### 1. **Outlier Detection**
- **Position outliers**: Uses z-score method to detect unusual joint positions
- **Velocity outliers**: Detects unusual movement speeds
- **Acceleration outliers**: Identifies abnormal acceleration patterns
- **Speed outliers**: Flags frames with unusual speed values
- **Distance outliers**: Detects unusual joint-to-joint distances
- **Label issues**: Identifies frames where labels don't match motion patterns
- **Missing/invalid data**: Finds NaN, infinite, or extremely large values

### 2. **Analysis Reports**
- Creates detailed analysis reports with outlier counts by type
- Saves label distribution and file distribution summaries
- Generates comprehensive statistics for all numeric columns

### 3. **Interactive Correction**
- Interactive command-line mode for reviewing outliers
- Multiple correction actions:
  - **c** - Keep as is (reviewed)
  - **d** - Delete row
  - **i** - Interpolate values
  - **a** - Apply "keep" to all remaining outliers
  - **s** - Save progress and exit
  - **q** - Quit without saving

### 4. **Correction Tracking**
- Saves corrections to JSON files for reproducibility
- Maintains a corrections log
- Allows loading existing corrections for continued work

## Usage

1. **Run the script**:
   ```bash
   python check_classification_data.py
   ```

2. **Review the analysis** - The script will automatically analyze all datasets and create reports

3. **Enter interactive correction mode** - Review and correct outliers one by one

4. **Output files**:
   - `analysis_report.txt` - Summary of all analyses
   - `{dataset}_outliers.csv` - Detailed outlier information
   - `{dataset}_outliers.json` - JSON format for easier correction
   - `statistics_report.txt` - Comprehensive statistics
   - `corrections/{dataset}_corrections.json` - Applied corrections

The script is designed to help you identify data quality issues and make corrections in a controlled, reproducible way.
"""
