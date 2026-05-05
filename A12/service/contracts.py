"""Input/output contracts for the A12 Gradio service tab.

The selected endpoint flavour is a Gradio tab inside app.py.  The endpoint
accepts a pose-feature CSV, validates that it has the feature columns used by
Rasa's A12 classifiers, and returns a structured prediction dictionary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

JOINTS: List[str] = [
    "head",
    "left_shoulder",
    "left_elbow",
    "right_shoulder",
    "right_elbow",
    "left_hand",
    "right_hand",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
]

KINECT_COLS: List[str] = [axis for joint in JOINTS for axis in (f"{joint}_x", f"{joint}_y", f"{joint}_z")]
POSENET_COLS: List[str] = [axis for joint in JOINTS for axis in (f"{joint}_x", f"{joint}_y")]

EXTRA_COLS: List[str] = [
    "left_hand_to_left_shoulder",
    "right_hand_to_right_shoulder",
    "left_hand_to_left_hip",
    "right_hand_to_right_hip",
    "left_elbow_to_left_shoulder",
    "right_elbow_to_right_shoulder",
    "head_to_hip",
    "head_vx",
    "head_vy",
    "head_vz",
    "head_speed",
    "left_hand_vx",
    "left_hand_vy",
    "left_hand_vz",
    "left_hand_speed",
    "right_hand_vx",
    "right_hand_vy",
    "right_hand_vz",
    "right_hand_speed",
    "head_ax",
    "head_ay",
    "head_az",
    "head_accel",
    "left_hand_ax",
    "left_hand_ay",
    "left_hand_az",
    "left_hand_accel",
    "right_hand_ax",
    "right_hand_ay",
    "right_hand_az",
    "right_hand_accel",
]

FEATURES_BY_PROBLEM: Dict[str, List[str]] = {
    "A": KINECT_COLS + EXTRA_COLS,
    "B": POSENET_COLS + EXTRA_COLS,
}

LABEL_NAMES = ["non-exercise", "exercise"]


def normalize_problem(problem: str) -> str:
    """Return canonical problem name A/B or raise ValueError."""
    value = str(problem).strip().upper()
    if value.startswith("A"):
        return "A"
    if value.startswith("B"):
        return "B"
    raise ValueError("Problem must be 'A' or 'B'.")


def read_pose_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read a pose feature CSV and strip whitespace from column names."""
    if not csv_path:
        raise ValueError("Please upload a pose CSV file.")

    path = Path(csv_path)
    if path.suffix.lower() != ".csv":
        raise ValueError("The uploaded file must be a .csv pose-feature file.")
    if not path.exists():
        raise ValueError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if df.empty:
        raise ValueError("The pose CSV is empty.")
    return df


def missing_columns(df: pd.DataFrame, expected: Iterable[str]) -> List[str]:
    """Return expected columns that are not present in *df*."""
    return [column for column in expected if column not in df.columns]


def validate_pose_dataframe(df: pd.DataFrame, problem: str) -> Tuple[pd.DataFrame, List[str]]:
    """Validate and return numeric feature matrix as a DataFrame.

    The current A12 saved scalers expect all base pose columns plus engineered
    columns from A12_classifier.py.  This function fails fast with a useful
    message instead of silently filling missing model features.
    """
    problem_key = normalize_problem(problem)
    expected = FEATURES_BY_PROBLEM[problem_key]
    missing = missing_columns(df, expected)
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise ValueError(
            f"CSV is missing {len(missing)} required columns for Problem {problem_key}: "
            f"{preview}{suffix}"
        )

    features = df[expected].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        bad = features.columns[features.isna().any()].tolist()[:8]
        raise ValueError(
            "Pose CSV contains non-numeric or missing values in required columns: "
            + ", ".join(bad)
        )
    return features, expected
