"""
A10 Data Loader Module - Issue #40
===================================
Data loading and preprocessing for 2D PoseNet/MoveNet to 2D Kinect mapping.

Per Issue #40 specification (clarified by Rasa):
- Input:  PoseNet/MoveNet 2D keypoints (xpn, ypn) -> 26 features
- Output: Kinect 2D keypoints          (xk,  yk)  -> 26 features

NOTE: The one-step variant (xpn,ypn) -> (xk,yk,zk) is Issue #41 (separate task).

This module supports three output modes for flexibility:
- 'xy'  : Kinect (xk, yk)      -> 26 features   [Issue #40 - PRIMARY]
- 'z'   : Kinect zk only       -> 13 features   [legacy depth-only]
- 'xyz' : Kinect (xk, yk, zk)  -> 39 features   [Issue #41 one-step variant]
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

# Kinect joints (13 joints) - matches slide column order
KINECT_JOINTS = [
    'head', 'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow',
    'left_hand', 'right_hand', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_foot', 'right_foot'
]

# Mapping: Kinect joint -> MoveNet keypoint name
KINECT_TO_MOVENET = {
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

N_KINECT_JOINTS = len(KINECT_JOINTS)        # 13 joints
N_INPUT = N_KINECT_JOINTS * 2               # 26 features (PoseNet x,y)
N_OUTPUT_XY = N_KINECT_JOINTS * 2           # 26 features (Kinect x,y) - Issue #40
N_OUTPUT_Z = N_KINECT_JOINTS                # 13 features (Kinect z only)
N_OUTPUT_XYZ = N_KINECT_JOINTS * 3          # 39 features (Kinect x,y,z) - Issue #41


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_kinect_csv(filepath: Union[str, bytes]) -> Dict[str, np.ndarray]:
    """
    Load a Kinect CSV file.

    Returns:
        Dict with:
          'xy'  : (N, 26) Kinect x,y   (Issue #40 target)
          'z'   : (N, 13) Kinect z
          'xyz' : (N, 39) Kinect x,y,z (Issue #41 target)
    """
    if isinstance(filepath, (str, os.PathLike)):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_csv(io.BytesIO(filepath))
    df.columns = df.columns.str.strip()

    xy_cols, z_cols, xyz_cols = [], [], []
    for joint in KINECT_JOINTS:
        xy_cols.extend([f"{joint}_x", f"{joint}_y"])
        z_cols.append(f"{joint}_z")
        xyz_cols.extend([f"{joint}_x", f"{joint}_y", f"{joint}_z"])

    return {
        'xy':  df[xy_cols].values.astype(np.float32),
        'z':   df[z_cols].values.astype(np.float32),
        'xyz': df[xyz_cols].values.astype(np.float32),
    }


def load_posenet_csv(filepath: str) -> np.ndarray:
    """
    Load a PoseNet/MoveNet CSV already aligned to Kinect joint order.

    Expected columns (per slide spec):
        FrameNo, head_x, head_y, left_shoulder_x, left_shoulder_y, ...

    Returns:
        (N, 26) PoseNet x,y for 13 joints.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    xy_cols = []
    for joint in KINECT_JOINTS:
        xy_cols.extend([f"{joint}_x", f"{joint}_y"])
    return df[xy_cols].values.astype(np.float32)


def load_movenet_raw_csv(filepath: str) -> np.ndarray:
    """
    Load raw MoveNet CSV (17 COCO keypoints) and project to Kinect's 13 joints.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    xy_cols = []
    for kinect_joint in KINECT_JOINTS:
        movenet_name = KINECT_TO_MOVENET[kinect_joint]
        xy_cols.extend([f"{movenet_name}_x", f"{movenet_name}_y"])
    return df[xy_cols].values.astype(np.float32)


def load_paired_sequence(
    kinect_path: str,
    posenet_path: Optional[str] = None,
    simulate_posenet: bool = True,
    noise_std: float = 0.02,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load one paired sequence: (PoseNet input, Kinect targets).

    If posenet_path is None and simulate_posenet=True, PoseNet input is
    synthesised from Kinect xy by adding gaussian noise. This allows the
    pipeline to be validated before real PoseNet CSVs are generated.
    """
    kinect = load_kinect_csv(kinect_path)

    if posenet_path is not None:
        X = load_posenet_csv(posenet_path)
        n = min(len(X), len(kinect['xy']))
        X = X[:n]
        kinect = {k: v[:n] for k, v in kinect.items()}
    elif simulate_posenet:
        rng = np.random.default_rng(random_state)
        X = kinect['xy'] + rng.normal(0.0, noise_std, kinect['xy'].shape).astype(np.float32)
    else:
        raise ValueError("Provide posenet_path or set simulate_posenet=True")

    return X, kinect


def load_all_paired_sequences(
    kinect_folder: str,
    posenet_folder: Optional[str] = None,
    simulate_posenet: bool = True,
    noise_std: float = 0.02,
    random_state: int = 42,
) -> Tuple[List[Tuple[np.ndarray, Dict]], List[str]]:
    """Load all paired (PoseNet, Kinect) sequences from a folder."""
    sequences, file_names = [], []
    csv_files = sorted(f for f in os.listdir(kinect_folder) if f.endswith('.csv'))
    print(f"Found {len(csv_files)} Kinect CSVs in {kinect_folder}")

    for i, name in enumerate(csv_files):
        k_path = os.path.join(kinect_folder, name)
        p_path = None
        if posenet_folder is not None:
            cand = os.path.join(posenet_folder, name)
            if os.path.exists(cand):
                p_path = cand

        X, targets = load_paired_sequence(
            k_path,
            posenet_path=p_path,
            simulate_posenet=simulate_posenet,
            noise_std=noise_std,
            random_state=random_state + i,
        )
        sequences.append((X, targets))
        file_names.append(name)

    if posenet_folder is None and simulate_posenet:
        print(f"Note: using SIMULATED PoseNet input (Kinect xy + noise std={noise_std}).")

    return sequences, file_names


# =============================================================================
# Preprocessing
# =============================================================================

def flatten_sequences(
    sequences: List[Tuple[np.ndarray, Dict]],
    output_type: str = 'xy',
) -> Tuple[np.ndarray, np.ndarray]:
    if output_type not in ('xy', 'z', 'xyz'):
        raise ValueError(f"output_type must be 'xy', 'z', or 'xyz'; got {output_type!r}")
    X = np.concatenate([s[0] for s in sequences], axis=0)
    Y = np.concatenate([s[1][output_type] for s in sequences], axis=0)
    return X, Y


def make_windowed_sequences(
    sequences: List[Tuple[np.ndarray, Dict]],
    window_size: int = 30,
    stride: int = 1,
    output_type: str = 'xy',
) -> Tuple[np.ndarray, np.ndarray]:
    """Create fixed-length windows for Conv1D/LSTM/GRU (returns full Y windows)."""
    if output_type not in ('xy', 'z', 'xyz'):
        raise ValueError(f"Invalid output_type: {output_type}")

    X_list, Y_list = [], []
    for X_seq, targets in sequences:
        Y_seq = targets[output_type]
        n = len(X_seq)
        for start in range(0, n - window_size + 1, stride):
            X_list.append(X_seq[start:start + window_size])
            Y_list.append(Y_seq[start:start + window_size])

    return (np.array(X_list, dtype=np.float32),
            np.array(Y_list, dtype=np.float32))


class DataNormalizer:
    """StandardScaler/MinMaxScaler normalizer for input and output."""

    def __init__(self, method: str = 'standard'):
        self.method = method
        self.input_scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        self.output_scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        self._fitted = False

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.input_scaler.fit(X)
        self.output_scaler.fit(Y)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray, Y: np.ndarray = None):
        if not self._fitted:
            raise RuntimeError("Normalizer must be fitted before transform")
        X_norm = self.input_scaler.transform(X).astype(np.float32)
        if Y is None:
            return X_norm
        return X_norm, self.output_scaler.transform(Y).astype(np.float32)

    def fit_transform(self, X: np.ndarray, Y: np.ndarray):
        self.fit(X, Y)
        return self.transform(X, Y)

    def inverse_transform_output(self, Y_norm: np.ndarray) -> np.ndarray:
        return self.output_scaler.inverse_transform(Y_norm)


# =============================================================================
# CV Utilities
# =============================================================================

def create_cv_splits(
    sequences: List,
    n_folds: int = 5,
    random_state: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """Sequence-level CV splits (~10 sequences per fold per instructions)."""
    rng = np.random.default_rng(random_state)
    n = len(sequences)
    indices = np.arange(n)
    rng.shuffle(indices)

    fold_size = max(1, n // n_folds)
    splits = []
    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else n
        test_idx = indices[start:end].tolist()
        train_idx = [i for i in indices if i not in test_idx]
        splits.append((train_idx, test_idx))
    return splits


def get_fold_data(
    sequences: List[Tuple[np.ndarray, Dict]],
    train_indices: List[int],
    test_indices: List[int],
    output_type: str = 'xy',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_seqs = [sequences[i] for i in train_indices]
    test_seqs = [sequences[i] for i in test_indices]
    X_train, Y_train = flatten_sequences(train_seqs, output_type=output_type)
    X_test, Y_test = flatten_sequences(test_seqs, output_type=output_type)
    return X_train, Y_train, X_test, Y_test


# =============================================================================
# Main entry
# =============================================================================

def load_dataset(
    kinect_folder: str,
    posenet_folder: Optional[str] = None,
    simulate_posenet: bool = True,
    output_type: str = 'xy',
    normalize: bool = True,
    test_split: float = 0.2,
    random_state: int = 42,
    noise_std: float = 0.02,
) -> Dict:
    """
    Load full paired dataset and split train/test.

    Default (output_type='xy') implements Issue #40:
        Input  : PoseNet 2D (26)
        Output : Kinect  2D (26)
    """
    sequences, file_names = load_all_paired_sequences(
        kinect_folder,
        posenet_folder=posenet_folder,
        simulate_posenet=simulate_posenet,
        noise_std=noise_std,
        random_state=random_state,
    )

    rng = np.random.default_rng(random_state)
    n = len(sequences)
    indices = rng.permutation(n)
    n_test = int(n * test_split)
    test_idx = indices[:n_test].tolist()
    train_idx = indices[n_test:].tolist()

    X_train, Y_train, X_test, Y_test = get_fold_data(
        sequences, train_idx, test_idx, output_type
    )

    normalizer = None
    if normalize:
        normalizer = DataNormalizer(method='standard')
        X_train, Y_train = normalizer.fit_transform(X_train, Y_train)
        X_test, Y_test = normalizer.transform(X_test, Y_test)

    return {
        'X_train': X_train, 'Y_train': Y_train,
        'X_test':  X_test,  'Y_test':  Y_test,
        'sequences': sequences, 'file_names': file_names,
        'train_indices': train_idx, 'test_indices': test_idx,
        'normalizer': normalizer,
        'output_type': output_type,
        'input_dim': X_train.shape[1],
        'output_dim': Y_train.shape[1],
    }


# =============================================================================
# Demo
# =============================================================================

if __name__ == '__main__':
    REPO_ROOT = Path(__file__).parent.parent
    KINECT_PATH = REPO_ROOT / 'kinect_good_preprocessed'

    if not KINECT_PATH.exists():
        print(f"Kinect data not found at: {KINECT_PATH}")
    else:
        print("Loading paired dataset (Issue #40: PoseNet 2D -> Kinect 2D)...")
        data = load_dataset(
            str(KINECT_PATH),
            posenet_folder=None,
            simulate_posenet=True,
            output_type='xy',
            normalize=True,
        )
        print(f"\nInput  dim: {data['input_dim']} (PoseNet x,y for 13 joints)")
        print(f"Output dim: {data['output_dim']} (Kinect  x,y for 13 joints)")
        print(f"X_train: {data['X_train'].shape}")
        print(f"Y_train: {data['Y_train'].shape}")
        print(f"X_test : {data['X_test'].shape}")
        print(f"Y_test : {data['Y_test'].shape}")
        print(f"Sequences: total={len(data['sequences'])}, "
              f"train={len(data['train_indices'])}, test={len(data['test_indices'])}")
