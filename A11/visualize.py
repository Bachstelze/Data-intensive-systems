#!/usr/bin/env python3
"""
Visualization helpers for A11 auto-cutting.

Creates:
- probability curve plots with predicted and ground-truth markers
- per-joint trajectory plots for the predicted cut segment
- 2D/3D skeleton animations, including side-by-side uncut vs cut views
"""

from pathlib import Path
from typing import Optional, Dict, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation

try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception:
    Axes3D = None

JOINTS = [
    "head", "left_shoulder", "left_elbow", "right_shoulder", "right_elbow",
    "left_hand", "right_hand", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_foot", "right_foot",
]

# Skeleton connections using the 13-joint format from A11_classifier.py
BONES = [
    ("head", "left_shoulder"), ("head", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_hand"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_hand"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_foot"),
    ("right_hip", "right_knee"), ("right_knee", "right_foot"),
]


def _safe_frame_col(df: pd.DataFrame) -> np.ndarray:
    if "FrameNo" in df.columns:
        return df["FrameNo"].to_numpy()
    return np.arange(len(df))


def plot_probabilities(
    frames: Sequence[int],
    start_prob: Sequence[float],
    stop_prob: Sequence[float],
    pred_start: int,
    pred_stop: int,
    out_path: Path,
    true_start: Optional[int] = None,
    true_stop: Optional[int] = None,
    title: str = "Start/stop probability curves",
) -> Path:
    """Plot classifier probabilities and predicted/ground-truth cut markers."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = np.asarray(frames)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frames, start_prob, label="start probability")
    ax.plot(frames, stop_prob, label="stop probability")

    ax.axvline(pred_start, linestyle="--", label=f"predicted start={pred_start}")
    ax.axvline(pred_stop, linestyle="--", label=f"predicted stop={pred_stop}")
    ax.axvspan(pred_start, pred_stop, alpha=0.12, label="predicted cut region")

    if true_start is not None:
        ax.axvline(true_start, linestyle=":", label=f"true start={true_start}")
    if true_stop is not None:
        ax.axvline(true_stop, linestyle=":", label=f"true stop={true_stop}")
    if true_start is not None and true_stop is not None:
        ax.axvspan(true_start, true_stop, alpha=0.08, label="ground-truth region")

    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_joint_trajectories(
    df_cut: pd.DataFrame,
    out_path: Path,
    modality: str = "A",
    joints: Sequence[str] = ("head", "left_hand", "right_hand", "left_foot", "right_foot"),
) -> Path:
    """Plot x/y(/z) trajectories for selected joints inside the predicted cut segment."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = _safe_frame_col(df_cut)
    dims = ["x", "y", "z"] if modality.upper() == "A" else ["x", "y"]

    fig, axes = plt.subplots(len(dims), 1, figsize=(12, 3.2 * len(dims)), sharex=True)
    if len(dims) == 1:
        axes = [axes]

    for ax, dim in zip(axes, dims):
        for joint in joints:
            col = f"{joint}_{dim}"
            if col in df_cut.columns:
                ax.plot(frames, df_cut[col], label=joint)
        ax.set_ylabel(dim)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Frame")
    fig.suptitle("Joint trajectories in predicted cut segment")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _axis_limits(df: pd.DataFrame, dims: Sequence[str]) -> Dict[str, tuple]:
    limits = {}
    for dim in dims:
        cols = [f"{j}_{dim}" for j in JOINTS if f"{j}_{dim}" in df.columns]
        if not cols:
            limits[dim] = (-1, 1)
            continue
        values = df[cols].to_numpy(dtype=float)
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        pad = max((vmax - vmin) * 0.1, 0.05)
        limits[dim] = (vmin - pad, vmax + pad)
    return limits


def _draw_skeleton_2d(ax, row: pd.Series, title: str, marker_text: str = ""):
    ax.clear()
    for a, b in BONES:
        ax.plot([row.get(f"{a}_x", np.nan), row.get(f"{b}_x", np.nan)],
                [row.get(f"{a}_y", np.nan), row.get(f"{b}_y", np.nan)],
                linewidth=2)
    xs = [row.get(f"{j}_x", np.nan) for j in JOINTS]
    ys = [row.get(f"{j}_y", np.nan) for j in JOINTS]
    ax.scatter(xs, ys, s=25)
    ax.set_title(title + (f"\n{marker_text}" if marker_text else ""), fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2)


def _draw_skeleton_3d(ax, row: pd.Series, title: str, marker_text: str = ""):
    ax.clear()
    for a, b in BONES:
        ax.plot([row.get(f"{a}_x", np.nan), row.get(f"{b}_x", np.nan)],
                [row.get(f"{a}_y", np.nan), row.get(f"{b}_y", np.nan)],
                [row.get(f"{a}_z", np.nan), row.get(f"{b}_z", np.nan)],
                linewidth=2)
    xs = [row.get(f"{j}_x", np.nan) for j in JOINTS]
    ys = [row.get(f"{j}_y", np.nan) for j in JOINTS]
    zs = [row.get(f"{j}_z", np.nan) for j in JOINTS]
    ax.scatter(xs, ys, zs, s=25)
    ax.set_title(title + (f"\n{marker_text}" if marker_text else ""), fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def animate_sequence(
    df: pd.DataFrame,
    out_path: Path,
    modality: str = "A",
    pred_start: Optional[int] = None,
    pred_stop: Optional[int] = None,
    fps: int = 15,
    max_frames: int = 250,
    title: str = "Skeleton animation",
) -> Path:
    """Animate one sequence as GIF/MP4 depending on output suffix."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_anim = df.copy()
    if len(df_anim) > max_frames:
        idx = np.linspace(0, len(df_anim) - 1, max_frames).astype(int)
        df_anim = df_anim.iloc[idx].reset_index(drop=True)

    is_3d = modality.upper() == "A" and all(f"head_{d}" in df.columns for d in ["x", "y", "z"])
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)
    limits = _axis_limits(df, ["x", "y", "z"] if is_3d else ["x", "y"])

    def update(i):
        row = df_anim.iloc[i]
        frame_no = int(row.get("FrameNo", i))
        marker = ""
        if pred_start is not None and frame_no == pred_start:
            marker = "PREDICTED START"
        elif pred_stop is not None and frame_no == pred_stop:
            marker = "PREDICTED STOP"
        frame_title = f"{title} | frame {frame_no}"
        if is_3d:
            _draw_skeleton_3d(ax, row, frame_title, marker)
            ax.set_xlim(*limits["x"]); ax.set_ylim(*limits["y"]); ax.set_zlim(*limits["z"])
        else:
            _draw_skeleton_2d(ax, row, frame_title, marker)
            ax.set_xlim(*limits["x"]); ax.set_ylim(*limits["y"])
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(df_anim), interval=1000 / fps, blit=False)
    _save_animation(ani, out_path, fps)
    plt.close(fig)
    return out_path


def animate_side_by_side(
    df_full: pd.DataFrame,
    df_cut: pd.DataFrame,
    out_path: Path,
    modality: str = "A",
    pred_start: Optional[int] = None,
    pred_stop: Optional[int] = None,
    fps: int = 15,
    max_frames: int = 250,
) -> Path:
    """Animate uncut sequence and predicted cut segment side-by-side."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    full = df_full.copy()
    cut = df_cut.copy()
    n = min(max(len(full), len(cut)), max_frames)
    full_idx = np.linspace(0, len(full) - 1, n).astype(int) if len(full) else []
    cut_idx = np.linspace(0, len(cut) - 1, n).astype(int) if len(cut) else []
    full = full.iloc[full_idx].reset_index(drop=True)
    cut = cut.iloc[cut_idx].reset_index(drop=True)

    is_3d = modality.upper() == "A" and all(f"head_{d}" in df_full.columns for d in ["x", "y", "z"])
    fig = plt.figure(figsize=(12, 6))
    if is_3d:
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
    else:
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
    limits = _axis_limits(df_full, ["x", "y", "z"] if is_3d else ["x", "y"])

    def update(i):
        full_row = full.iloc[i]
        cut_row = cut.iloc[i]
        full_frame = int(full_row.get("FrameNo", i))
        marker = ""
        if pred_start is not None and full_frame == pred_start:
            marker = "PREDICTED START"
        elif pred_stop is not None and full_frame == pred_stop:
            marker = "PREDICTED STOP"
        if is_3d:
            _draw_skeleton_3d(ax1, full_row, f"Uncut | frame {full_frame}", marker)
            _draw_skeleton_3d(ax2, cut_row, f"Auto-cut | frame {int(cut_row.get('FrameNo', i))}")
            for ax in [ax1, ax2]:
                ax.set_xlim(*limits["x"]); ax.set_ylim(*limits["y"]); ax.set_zlim(*limits["z"])
        else:
            _draw_skeleton_2d(ax1, full_row, f"Uncut | frame {full_frame}", marker)
            _draw_skeleton_2d(ax2, cut_row, f"Auto-cut | frame {int(cut_row.get('FrameNo', i))}")
            for ax in [ax1, ax2]:
                ax.set_xlim(*limits["x"]); ax.set_ylim(*limits["y"])
        return []

    ani = animation.FuncAnimation(fig, update, frames=n, interval=1000 / fps, blit=False)
    _save_animation(ani, out_path, fps)
    plt.close(fig)
    return out_path


def _save_animation(ani: animation.FuncAnimation, out_path: Path, fps: int):
    suffix = out_path.suffix.lower()
    if suffix == ".gif":
        ani.save(out_path, writer="pillow", fps=fps)
    elif suffix == ".mp4":
        ani.save(out_path, writer="ffmpeg", fps=fps)
    else:
        raise ValueError("Animation output must end with .gif or .mp4")
