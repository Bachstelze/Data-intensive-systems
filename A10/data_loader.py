"""
A10 Data Loader Module
======================
Data loading and preprocessing for 2D Pose (MoveNet/PoseNet) to 3D Kinect mapping.

This module provides functionality to:
- Load MoveNet/PoseNet 2D keypoints 
- Load Kinect 3D skeleton data
- Map between MoveNet (17 COCO) and Kinect (13 joints) keypoint formats
- Create paired training datasets
- Implement data normalization

Issue #40 - A10: 2D Pose Estimation to 3D Mapping - Deep Learning Pipeline
"""

import os
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# =============================================================================
# Joint Definitions
# =============================================================================

# MoveNet COCO keypoints (17 keypoints)
MOVENET_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Kinect joints (13 joints)
KINECT_JOINTS = [
    'head', 'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow',
    'left_hand', 'right_hand', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_foot', 'right_foot'
]

# Mapping from MoveNet keypoints to Kinect joints
# MoveNet index -> Kinect joint name
MOVENET_TO_KINECT_MAP = {
    0: 'head',            # nose -> head
    5: 'left_shoulder',   # left_shoulder -> left_shoulder
    6: 'right_shoulder',  # right_shoulder -> right_shoulder
    7: 'left_elbow',      # left_elbow -> left_elbow
    8: 'right_elbow',     # right_elbow -> right_elbow
    9: 'left_hand',       # left_wrist -> left_hand
    10: 'right_hand',     # right_wrist -> right_hand
    11: 'left_hip',       # left_hip -> left_hip
    12: 'right_hip',      # right_hip -> right_hip
    13: 'left_knee',      # left_knee -> left_knee
    14: 'right_knee',     # right_knee -> right_knee
    15: 'left_foot',      # left_ankle -> left_foot
    16: 'right_foot',     # right_ankle -> right_foot
}

# Indices of MoveNet keypoints that map to Kinect (excluding eyes and ears)
MOVENET_VALID_INDICES = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

N_KINECT_JOINTS = len(KINECT_JOINTS)        # 13 joints
N_INPUT_2D = N_KINECT_JOINTS * 2            # 26 features (x, y for each joint)
N_OUTPUT_3D = N_KINECT_JOINTS               # 13 z-coordinates (or 39 for full xyz)

# Aliases for compatibility with models.py and train.py
N_INPUT = N_INPUT_2D
N_OUTPUT_Z = N_OUTPUT_3D


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_kinect_csv(filepath_or_bytes: Union[str, bytes]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a single Kinect CSV file and extract x, y, z coordinates.
    
    Args:
        filepath_or_bytes: Path to CSV file or raw bytes
        
    Returns:
        Tuple of (X_xy, Y_z, Y_xyz):
        - X_xy: Input features (N, 26) - x,y for 13 joints
        - Y_z: Z-coordinates only (N, 13)
        - Y_xyz: Full 3D coordinates (N, 39) - x,y,z for 13 joints
    """
    if isinstance(filepath_or_bytes, (str, os.PathLike)):
        df = pd.read_csv(filepath_or_bytes)
    else:
        df = pd.read_csv(io.BytesIO(filepath_or_bytes))
    
    df.columns = df.columns.str.strip()
    
    # Build column lists
    xy_cols = []
    z_cols = []
    xyz_cols = []
    
    for joint in KINECT_JOINTS:
        xy_cols.extend([f"{joint}_x", f"{joint}_y"])
        z_cols.append(f"{joint}_z")
        xyz_cols.extend([f"{joint}_x", f"{joint}_y", f"{joint}_z"])
    
    X_xy = df[xy_cols].values.astype(np.float32)      # (N, 26)
    Y_z = df[z_cols].values.astype(np.float32)         # (N, 13)
    Y_xyz = df[xyz_cols].values.astype(np.float32)     # (N, 39)
    
    return X_xy, Y_z, Y_xyz


def load_all_kinect_sequences(folder_path: str) -> Tuple[List[Tuple], List[str]]:
    """
    Load all Kinect CSV files from a folder.
    
    Args:
        folder_path: Path to folder containing CSV files
        
    Returns:
        Tuple of (sequences, file_names):
        - sequences: List of tuples (X_xy, Y_z, Y_xyz)
        - file_names: List of CSV file names
    """
    sequences = []
    file_names = []
    
    csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    print(f"Found {len(csv_files)} Kinect CSV files in {folder_path}")
    
    for name in csv_files:
        file_path = os.path.join(folder_path, name)
        X_xy, Y_z, Y_xyz = load_kinect_csv(file_path)
        sequences.append((X_xy, Y_z, Y_xyz))
        file_names.append(name)
    
    return sequences, file_names


def load_movenet_keypoints(keypoints_dict: Dict) -> np.ndarray:
    """
    Convert MoveNet keypoints dictionary to Kinect-aligned array.
    
    Args:
        keypoints_dict: Dictionary from MoveNetPoseEstimator.detect_pose()
                       Format: {'keypoints': {'nose': {'x': float, 'y': float, ...}, ...}}
    
    Returns:
        Array of shape (26,) with x,y coordinates for 13 Kinect joints
    """
    kps = keypoints_dict['keypoints']
    result = np.zeros(N_INPUT_2D, dtype=np.float32)
    
    # Map MoveNet keypoints to Kinect joint positions
    kinect_to_movenet = {
        'head': 'nose',
        'left_shoulder': 'left_shoulder',
        'right_shoulder': 'right_shoulder',
        'left_elbow': 'left_elbow',
        'right_elbow': 'right_elbow',
        'left_hand': 'left_wrist',
        'right_hand': 'right_wrist',
        'left_hip': 'left_hip',
        'right_hip': 'right_hip',
        'left_knee': 'left_knee',
        'right_knee': 'right_knee',
        'left_foot': 'left_ankle',
        'right_foot': 'right_ankle',
    }
    
    for i, kinect_joint in enumerate(KINECT_JOINTS):
        movenet_name = kinect_to_movenet[kinect_joint]
        kp = kps[movenet_name]
        result[i * 2] = kp['x']
        result[i * 2 + 1] = kp['y']
    
    return result


def load_movenet_csv(filepath: str) -> np.ndarray:
    """
    Load MoveNet keypoints from a CSV file (exported from pose_estimator).
    
    Expected CSV format: frame_id, nose_x, nose_y, nose_conf, ...
    
    Args:
        filepath: Path to MoveNet keypoints CSV
        
    Returns:
        Array of shape (N, 26) with x,y for 13 Kinect-mapped joints
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    # Map MoveNet columns to Kinect order
    kinect_to_movenet = {
        'head': 'nose',
        'left_shoulder': 'left_shoulder',
        'right_shoulder': 'right_shoulder',
        'left_elbow': 'left_elbow',
        'right_elbow': 'right_elbow',
        'left_hand': 'left_wrist',
        'right_hand': 'right_wrist',
        'left_hip': 'left_hip',
        'right_hip': 'right_hip',
        'left_knee': 'left_knee',
        'right_knee': 'right_knee',
        'left_foot': 'left_ankle',
        'right_foot': 'right_ankle',
    }
    
    xy_cols = []
    for kinect_joint in KINECT_JOINTS:
        movenet_name = kinect_to_movenet[kinect_joint]
        xy_cols.extend([f"{movenet_name}_x", f"{movenet_name}_y"])
    
    X = df[xy_cols].values.astype(np.float32)
    return X


# =============================================================================
# Data Preprocessing Functions
# =============================================================================

def flatten_sequences(sequences: List[Tuple]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten list of sequences into single arrays (for Dense models).
    
    Args:
        sequences: List of (X_xy, Y_z, Y_xyz) tuples
        
    Returns:
        Tuple of (X_xy_flat, Y_z_flat, Y_xyz_flat)
    """
    X_flat = np.concatenate([s[0] for s in sequences], axis=0)
    Y_z_flat = np.concatenate([s[1] for s in sequences], axis=0)
    Y_xyz_flat = np.concatenate([s[2] for s in sequences], axis=0)
    return X_flat, Y_z_flat, Y_xyz_flat


def make_windowed_sequences(
    sequences: List[Tuple],
    window_size: int = 30,
    stride: int = 1,
    output_type: str = 'z'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create fixed-length windows from sequences (for Conv1D, LSTM, GRU).
    
    Args:
        sequences: List of (X_xy, Y_z, Y_xyz) tuples
        window_size: Number of frames per window
        stride: Step size between windows
        output_type: 'z' for z-only output, 'xyz' for full 3D
        
    Returns:
        Tuple of (X_seq, Y_seq) arrays
    """
    X_list, Y_list = [], []
    
    for X_xy, Y_z, Y_xyz in sequences:
        Y = Y_z if output_type == 'z' else Y_xyz
        n = len(X_xy)
        
        for start in range(0, n - window_size + 1, stride):
            X_list.append(X_xy[start:start + window_size])
            Y_list.append(Y[start:start + window_size])
    
    X_seq = np.array(X_list, dtype=np.float32)
    Y_seq = np.array(Y_list, dtype=np.float32)
    
    return X_seq, Y_seq


class DataNormalizer:
    """
    Normalizer for input (2D) and output (3D) data.
    
    Supports StandardScaler (z-score) and MinMaxScaler normalization.
    """
    
    def __init__(self, method: str = 'standard'):
        """
        Args:
            method: 'standard' for z-score, 'minmax' for [0,1] scaling
        """
        self.method = method
        self.input_scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        self.output_scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        self._fitted = False
    
    def fit(self, X: np.ndarray, Y: np.ndarray):
        """Fit scalers on training data."""
        self.input_scaler.fit(X)
        self.output_scaler.fit(Y)
        self._fitted = True
        return self
    
    def transform(self, X: np.ndarray, Y: np.ndarray = None) -> Union[np.ndarray, Tuple]:
        """Transform data using fitted scalers."""
        if not self._fitted:
            raise RuntimeError("Normalizer must be fitted before transform")
        
        X_norm = self.input_scaler.transform(X)
        if Y is not None:
            Y_norm = self.output_scaler.transform(Y)
            return X_norm.astype(np.float32), Y_norm.astype(np.float32)
        return X_norm.astype(np.float32)
    
    def fit_transform(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one step."""
        self.fit(X, Y)
        return self.transform(X, Y)
    
    def inverse_transform_output(self, Y_norm: np.ndarray) -> np.ndarray:
        """Convert normalized predictions back to original scale."""
        return self.output_scaler.inverse_transform(Y_norm)


# =============================================================================
# Cross-Validation Utilities
# =============================================================================

def create_cv_splits(
    sequences: List[Tuple],
    n_folds: int = 5,
    test_sequences_per_fold: int = 10,
    random_state: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """
    Create cross-validation splits at the sequence level.
    
    Args:
        sequences: List of sequence tuples
        n_folds: Number of CV folds
        test_sequences_per_fold: Approximate test sequences per fold
        random_state: Random seed for reproducibility
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    np.random.seed(random_state)
    n_sequences = len(sequences)
    indices = np.arange(n_sequences)
    np.random.shuffle(indices)
    
    fold_size = max(1, n_sequences // n_folds)
    splits = []
    
    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else n_sequences
        
        test_idx = indices[start:end].tolist()
        train_idx = [i for i in indices if i not in test_idx]
        splits.append((train_idx, test_idx))
    
    return splits


def get_fold_data(
    sequences: List[Tuple],
    train_indices: List[int],
    test_indices: List[int],
    output_type: str = 'z'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get train/test data for a specific CV fold.
    
    Args:
        sequences: All sequences
        train_indices: Indices of training sequences
        test_indices: Indices of test sequences
        output_type: 'z' or 'xyz'
        
    Returns:
        Tuple of (X_train, Y_train, X_test, Y_test)
    """
    train_seqs = [sequences[i] for i in train_indices]
    test_seqs = [sequences[i] for i in test_indices]
    
    X_train, Y_z_train, Y_xyz_train = flatten_sequences(train_seqs)
    X_test, Y_z_test, Y_xyz_test = flatten_sequences(test_seqs)
    
    Y_train = Y_z_train if output_type == 'z' else Y_xyz_train
    Y_test = Y_z_test if output_type == 'z' else Y_xyz_test
    
    return X_train, Y_train, X_test, Y_test


# =============================================================================
# Main Data Loading Function
# =============================================================================

def load_dataset(
    kinect_folder: str,
    normalize: bool = True,
    output_type: str = 'z',
    test_split: float = 0.2,
    random_state: int = 42
) -> Dict:
    """
    Load and prepare the full dataset for training.
    
    Args:
        kinect_folder: Path to Kinect CSV files
        normalize: Whether to normalize data
        output_type: 'z' for depth only, 'xyz' for full 3D
        test_split: Fraction of sequences for testing
        random_state: Random seed
        
    Returns:
        Dictionary with dataset components
    """
    # Load all sequences
    sequences, file_names = load_all_kinect_sequences(kinect_folder)
    
    # Split sequences
    n_sequences = len(sequences)
    n_test = int(n_sequences * test_split)
    
    np.random.seed(random_state)
    indices = np.random.permutation(n_sequences)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Get flat data
    X_train, Y_train, X_test, Y_test = get_fold_data(
        sequences, train_indices.tolist(), test_indices.tolist(), output_type
    )
    
    # Normalize
    normalizer = None
    if normalize:
        normalizer = DataNormalizer(method='standard')
        X_train, Y_train = normalizer.fit_transform(X_train, Y_train)
        X_test, Y_test = normalizer.transform(X_test, Y_test)
    
    return {
        'X_train': X_train,
        'Y_train': Y_train,
        'X_test': X_test,
        'Y_test': Y_test,
        'sequences': sequences,
        'file_names': file_names,
        'train_indices': train_indices.tolist(),
        'test_indices': test_indices.tolist(),
        'normalizer': normalizer,
        'output_type': output_type,
    }


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == '__main__':
    # Test with Kinect data
    REPO_ROOT = Path(__file__).parent.parent
    KINECT_PATH = REPO_ROOT / 'kinect_good_preprocessed'
    
    if KINECT_PATH.exists():
        print("Loading Kinect data...")
        dataset = load_dataset(
            str(KINECT_PATH),
            normalize=True,
            output_type='z'
        )
        
        print(f"\nDataset loaded:")
        print(f"  X_train: {dataset['X_train'].shape}")
        print(f"  Y_train: {dataset['Y_train'].shape}")
        print(f"  X_test:  {dataset['X_test'].shape}")
        print(f"  Y_test:  {dataset['Y_test'].shape}")
        print(f"  Sequences: {len(dataset['sequences'])}")
        print(f"  Train sequences: {len(dataset['train_indices'])}")
        print(f"  Test sequences:  {len(dataset['test_indices'])}")
    else:
        print(f"Kinect data not found at: {KINECT_PATH}")
