from __future__ import annotations

"""
A10 visualizer.py
=================
3D skeleton visualization utilities for Issue #42 / Sprint 10.

Designed to fit the current A10 sprint codebase where:
- PoseNet/MoveNet input is 13 joints x 2 coordinates = 26 features
- Kinect / one-step output is 13 joints x 3 coordinates = 39 features
- Joint order must match data_loader.KINECT_JOINTS

This module supports:
- Static 3D skeleton plots
- Side-by-side and overlay comparison plots
- Per-joint error coloring
- Multiple camera angles (front / side / top)
- Joint trajectory trails
- GIF / MP4 export with matplotlib animations
- Interactive HTML viewer with play/pause, frame slider, and speed buttons
- Saving prediction bundles so the visualizer can be called from training code

Typical use:
    from visualizer import (
        save_prediction_bundle,
        create_evaluation_visuals,
    )

    bundle_dir = save_prediction_bundle(
        output_dir='A10/prediction_runs/demo',
        predicted_xyz=pred_xyz,
        ground_truth_xyz=true_xyz,
        sequence_name='A1_kinect.csv',
        metadata={'model': 'Dense_shallow_adam'}
    )

    create_evaluation_visuals(bundle_dir)
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.colors import Normalize
from matplotlib import cm

# Plotly is optional but very useful for the interactive viewer.
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False


# -----------------------------------------------------------------------------
# Skeleton definition
# -----------------------------------------------------------------------------

JOINTS = [
    'head', 'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow',
    'left_hand', 'right_hand', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_foot', 'right_foot'
]

JOINT_INDEX = {name: idx for idx, name in enumerate(JOINTS)}

# Bone graph for the 13 Kinect joints used in your A10 codebase.
BONES = [
    ('head', 'left_shoulder'),
    ('head', 'right_shoulder'),
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_hand'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_hand'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_foot'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_foot'),
]

VIEW_PRESETS = {
    'front': dict(elev=15, azim=-90),
    'side': dict(elev=15, azim=0),
    'top': dict(elev=90, azim=-90),
    'iso': dict(elev=20, azim=-55),
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _as_path(pathlike: Union[str, Path]) -> Path:
    return pathlike if isinstance(pathlike, Path) else Path(pathlike)


def _ensure_dir(pathlike: Union[str, Path]) -> Path:
    path = _as_path(pathlike)
    path.mkdir(parents=True, exist_ok=True)
    return path


def reshape_xyz(data: np.ndarray) -> np.ndarray:
    """
    Convert xyz data into shape (n_frames, 13, 3).

    Accepts:
    - (n_frames, 39)
    - (39,)
    - (n_frames, 13, 3)
    - (13, 3)
    """
    arr = np.asarray(data, dtype=np.float32)

    if arr.ndim == 1:
        if arr.shape[0] != 39:
            raise ValueError(f'1D xyz input must have 39 values, got {arr.shape[0]}')
        arr = arr.reshape(1, 13, 3)
    elif arr.ndim == 2:
        if arr.shape == (13, 3):
            arr = arr.reshape(1, 13, 3)
        elif arr.shape[1] == 39:
            arr = arr.reshape(arr.shape[0], 13, 3)
        else:
            raise ValueError(f'2D xyz input must be (n,39) or (13,3); got {arr.shape}')
    elif arr.ndim == 3:
        if arr.shape[1:] != (13, 3):
            raise ValueError(f'3D xyz input must be (n,13,3); got {arr.shape}')
    else:
        raise ValueError(f'Unsupported xyz input shape: {arr.shape}')

    return arr


def compute_joint_errors(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> np.ndarray:
    """Euclidean error per joint, shape (n_frames, 13)."""
    pred = reshape_xyz(pred_xyz)
    gt = reshape_xyz(gt_xyz)
    n = min(len(pred), len(gt))
    pred = pred[:n]
    gt = gt[:n]
    return np.linalg.norm(pred - gt, axis=2)


def compute_frame_errors(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> np.ndarray:
    """Mean Euclidean joint error per frame, shape (n_frames,)."""
    return compute_joint_errors(pred_xyz, gt_xyz).mean(axis=1)


def infer_axis_limits(*arrays: np.ndarray, pad_ratio: float = 0.08) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    stacked = np.concatenate([reshape_xyz(a).reshape(-1, 3) for a in arrays], axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    pads = spans * pad_ratio
    mins -= pads
    maxs += pads

    # Make a cubic box so the skeleton does not look distorted.
    center = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 1e-4)
    return (
        (center[0] - radius, center[0] + radius),
        (center[1] - radius, center[1] + radius),
        (center[2] - radius, center[2] + radius),
    )


def _bone_segments(points: np.ndarray) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    for j1, j2 in BONES:
        yield points[JOINT_INDEX[j1]], points[JOINT_INDEX[j2]]


def save_prediction_bundle(
    output_dir: Union[str, Path],
    predicted_xyz: np.ndarray,
    ground_truth_xyz: Optional[np.ndarray] = None,
    sequence_name: Optional[str] = None,
    metadata: Optional[Dict] = None,
    posenet_xy: Optional[np.ndarray] = None,
) -> Path:
    """
    Save model outputs in a simple, reusable format for the visualizer.
    """
    out_dir = _ensure_dir(output_dir)
    pred = reshape_xyz(predicted_xyz)
    np.save(out_dir / 'predicted_xyz.npy', pred)

    if ground_truth_xyz is not None:
        gt = reshape_xyz(ground_truth_xyz)
        n = min(len(pred), len(gt))
        np.save(out_dir / 'ground_truth_xyz.npy', gt[:n])
        pred = pred[:n]

    if posenet_xy is not None:
        np.save(out_dir / 'posenet_xy.npy', np.asarray(posenet_xy, dtype=np.float32))

    meta = dict(metadata or {})
    meta['sequence_name'] = sequence_name
    meta['n_frames'] = int(len(pred))
    meta['joints'] = JOINTS
    meta['bones'] = BONES
    with open(out_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    return out_dir


def load_prediction_bundle(bundle_dir: Union[str, Path]) -> Dict[str, Optional[np.ndarray]]:
    bundle = _as_path(bundle_dir)
    pred = np.load(bundle / 'predicted_xyz.npy')
    gt_path = bundle / 'ground_truth_xyz.npy'
    xy_path = bundle / 'posenet_xy.npy'
    meta_path = bundle / 'metadata.json'

    gt = np.load(gt_path) if gt_path.exists() else None
    posenet_xy = np.load(xy_path) if xy_path.exists() else None
    metadata = {}
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    return {
        'predicted_xyz': reshape_xyz(pred),
        'ground_truth_xyz': reshape_xyz(gt) if gt is not None else None,
        'posenet_xy': posenet_xy,
        'metadata': metadata,
        'bundle_dir': bundle,
    }


# -----------------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------------

def _draw_skeleton(
    ax,
    points: np.ndarray,
    title: Optional[str] = None,
    joint_errors: Optional[np.ndarray] = None,
    cmap_name: str = 'turbo',
    error_norm: Optional[Normalize] = None,
    bone_color: str = 'black',
    marker_size: int = 36,
    alpha: float = 1.0,
    show_labels: bool = False,
    trails: Optional[np.ndarray] = None,
):
    points = np.asarray(points, dtype=np.float32)
    cmap = cm.get_cmap(cmap_name)

    if joint_errors is None:
        joint_colors = ['tab:blue'] * len(points)
    else:
        if error_norm is None:
            error_norm = Normalize(vmin=float(np.min(joint_errors)), vmax=float(np.max(joint_errors)) + 1e-9)
        joint_colors = [cmap(error_norm(v)) for v in joint_errors]

    # Trails first
    if trails is not None and len(trails) > 1:
        for j in range(points.shape[0]):
            trail = trails[:, j, :]
            ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], alpha=0.25, linewidth=1.0)

    # Bones
    for p1, p2 in _bone_segments(points):
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color=bone_color,
            linewidth=2,
            alpha=alpha,
        )

    # Joints
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=joint_colors, s=marker_size, alpha=alpha)

    if show_labels:
        for idx, name in enumerate(JOINTS):
            x, y, z = points[idx]
            ax.text(x, y, z, name, fontsize=8)

    if title:
        ax.set_title(title)


def _format_axes(ax, axis_limits, view_name='iso'):
    (xlim, ylim, zlim) = axis_limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    view = VIEW_PRESETS.get(view_name, VIEW_PRESETS['iso'])
    ax.view_init(elev=view['elev'], azim=view['azim'])


def plot_frame_comparison(
    predicted_xyz: np.ndarray,
    ground_truth_xyz: Optional[np.ndarray] = None,
    frame_idx: int = 0,
    save_path: Optional[Union[str, Path]] = None,
    show_labels: bool = False,
    overlay: bool = True,
) -> plt.Figure:
    """
    Create a report-friendly static figure.
    Layout:
    - GT skeleton
    - Pred skeleton
    - Overlay
    - Error heatmap overlay
    """
    pred = reshape_xyz(predicted_xyz)
    gt = reshape_xyz(ground_truth_xyz) if ground_truth_xyz is not None else None
    frame_idx = int(np.clip(frame_idx, 0, len(pred) - 1))
    pred_f = pred[frame_idx]
    gt_f = gt[frame_idx] if gt is not None else None

    if gt_f is not None:
        joint_errors = np.linalg.norm(pred_f - gt_f, axis=1)
        error_norm = Normalize(vmin=0.0, vmax=max(float(joint_errors.max()), 1e-6))
        axis_limits = infer_axis_limits(pred_f, gt_f)
    else:
        joint_errors = None
        error_norm = None
        axis_limits = infer_axis_limits(pred_f)

    if gt_f is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        _draw_skeleton(ax, pred_f, title=f'Predicted skeleton — frame {frame_idx}', show_labels=show_labels)
        _format_axes(ax, axis_limits, 'iso')
    else:
        fig = plt.figure(figsize=(14, 10))
        ax1 = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222, projection='3d')
        ax3 = fig.add_subplot(223, projection='3d')
        ax4 = fig.add_subplot(224, projection='3d')

        _draw_skeleton(ax1, gt_f, title='Ground truth', bone_color='tab:green', show_labels=show_labels)
        _draw_skeleton(ax2, pred_f, title='Prediction', bone_color='tab:blue', show_labels=show_labels)

        if overlay:
            _draw_skeleton(ax3, gt_f, title='Overlay', bone_color='tab:green', alpha=0.65)
            _draw_skeleton(ax3, pred_f, bone_color='tab:blue', alpha=0.65)
        else:
            _draw_skeleton(ax3, pred_f, title='Prediction')

        _draw_skeleton(
            ax4,
            pred_f,
            title=f'Error heatmap (mean={joint_errors.mean():.4f})',
            joint_errors=joint_errors,
            error_norm=error_norm,
            bone_color='gray',
            show_labels=show_labels,
        )

        for ax, view in zip([ax1, ax2, ax3, ax4], ['iso', 'iso', 'front', 'iso']):
            _format_axes(ax, axis_limits, view)

    fig.suptitle(f'3D Skeleton comparison — frame {frame_idx}', fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        save_path = _as_path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches='tight')

    return fig


def plot_multiview_frame(
    predicted_xyz: np.ndarray,
    ground_truth_xyz: Optional[np.ndarray] = None,
    frame_idx: int = 0,
    save_path: Optional[Union[str, Path]] = None,
    trails: int = 0,
):
    pred = reshape_xyz(predicted_xyz)
    gt = reshape_xyz(ground_truth_xyz) if ground_truth_xyz is not None else None
    frame_idx = int(np.clip(frame_idx, 0, len(pred) - 1))
    pred_f = pred[frame_idx]
    gt_f = gt[frame_idx] if gt is not None else None

    trail_arr = None
    if trails > 0:
        s = max(0, frame_idx - trails)
        trail_arr = pred[s:frame_idx + 1]

    axis_limits = infer_axis_limits(pred, gt) if gt is not None else infer_axis_limits(pred)

    fig = plt.figure(figsize=(16, 4.5))
    axes = [fig.add_subplot(1, 4, i + 1, projection='3d') for i in range(4)]
    titles = ['Front', 'Side', 'Top', 'Overlay / iso']
    views = ['front', 'side', 'top', 'iso']

    if gt_f is not None:
        joint_errors = np.linalg.norm(pred_f - gt_f, axis=1)
        error_norm = Normalize(vmin=0.0, vmax=max(float(joint_errors.max()), 1e-6))
    else:
        joint_errors = None
        error_norm = None

    for ax, title, view in zip(axes, titles, views):
        if gt_f is not None and view == 'iso':
            _draw_skeleton(ax, gt_f, bone_color='tab:green', alpha=0.6)
            _draw_skeleton(ax, pred_f, joint_errors=joint_errors, error_norm=error_norm, bone_color='gray', alpha=0.85, trails=trail_arr)
        else:
            _draw_skeleton(ax, pred_f, joint_errors=joint_errors, error_norm=error_norm, bone_color='gray', trails=trail_arr)
        ax.set_title(title)
        _format_axes(ax, axis_limits, view)

    fig.suptitle(f'Multiview 3D skeleton — frame {frame_idx}', fontsize=14)
    fig.tight_layout()
    if save_path is not None:
        save_path = _as_path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches='tight')
    return fig


# -----------------------------------------------------------------------------
# Animation export
# -----------------------------------------------------------------------------

def animate_skeletons_matplotlib(
    predicted_xyz: np.ndarray,
    ground_truth_xyz: Optional[np.ndarray] = None,
    save_path: Union[str, Path] = 'animation.gif',
    fps: int = 15,
    dpi: int = 120,
    show_labels: bool = False,
    trail_length: int = 10,
    view_name: str = 'iso',
):
    """
    Save a GIF or MP4 animation.
    Uses a 3-panel layout: GT, prediction, overlay/error.
    """
    pred = reshape_xyz(predicted_xyz)
    gt = reshape_xyz(ground_truth_xyz) if ground_truth_xyz is not None else None
    n_frames = len(pred) if gt is None else min(len(pred), len(gt))
    pred = pred[:n_frames]
    if gt is not None:
        gt = gt[:n_frames]
        all_joint_errors = np.linalg.norm(pred - gt, axis=2)
        error_norm = Normalize(vmin=0.0, vmax=max(float(all_joint_errors.max()), 1e-6))
    else:
        all_joint_errors = None
        error_norm = None

    axis_limits = infer_axis_limits(pred, gt) if gt is not None else infer_axis_limits(pred)

    fig = plt.figure(figsize=(15, 5))
    axes = [fig.add_subplot(1, 3, i + 1, projection='3d') for i in range(3)]

    def update(frame_idx):
        for ax in axes:
            ax.cla()

        pred_f = pred[frame_idx]
        gt_f = gt[frame_idx] if gt is not None else None
        trail = pred[max(0, frame_idx - trail_length):frame_idx + 1] if trail_length > 0 else None

        if gt_f is not None:
            joint_errors = all_joint_errors[frame_idx]
            _draw_skeleton(axes[0], gt_f, title='Ground truth', bone_color='tab:green', show_labels=show_labels)
            _draw_skeleton(axes[1], pred_f, title='Prediction', joint_errors=joint_errors, error_norm=error_norm, bone_color='gray', show_labels=show_labels)
            _draw_skeleton(axes[2], gt_f, title=f'Overlay — frame {frame_idx}', bone_color='tab:green', alpha=0.5)
            _draw_skeleton(axes[2], pred_f, joint_errors=joint_errors, error_norm=error_norm, bone_color='gray', alpha=0.9, trails=trail)
        else:
            _draw_skeleton(axes[0], pred_f, title=f'Prediction — frame {frame_idx}', bone_color='tab:blue', show_labels=show_labels, trails=trail)
            axes[1].set_visible(False)
            axes[2].set_visible(False)

        for ax in axes:
            if ax.get_visible():
                _format_axes(ax, axis_limits, view_name)

        fig.suptitle(f'3D skeleton animation — frame {frame_idx + 1}/{n_frames}', fontsize=14)
        return axes

    anim = FuncAnimation(fig, update, frames=n_frames, interval=int(1000 / max(fps, 1)), blit=False)

    save_path = _as_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = save_path.suffix.lower()
    if suffix == '.gif':
        writer = PillowWriter(fps=fps)
    elif suffix in {'.mp4', '.m4v'}:
        writer = FFMpegWriter(fps=fps)
    else:
        raise ValueError('save_path must end with .gif or .mp4')

    anim.save(save_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return save_path


# -----------------------------------------------------------------------------
# Plotly interactive viewer
# -----------------------------------------------------------------------------

def _scatter3d_points(points, name, color, size=5, text=None):
    return go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers+text' if text is not None else 'markers',
        marker=dict(size=size, color=color),
        text=text,
        textposition='top center',
        name=name,
    )


def _scatter3d_bones(points, name, color, width=5):
    xs, ys, zs = [], [], []
    for p1, p2 in _bone_segments(points):
        xs.extend([p1[0], p2[0], None])
        ys.extend([p1[1], p2[1], None])
        zs.extend([p1[2], p2[2], None])
    return go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', line=dict(color=color, width=width), name=name)


def export_interactive_html(
    predicted_xyz: np.ndarray,
    ground_truth_xyz: Optional[np.ndarray] = None,
    html_path: Union[str, Path] = 'viewer.html',
    show_labels: bool = False,
):
    """
    Export an interactive Plotly viewer with:
    - play / pause
    - frame stepping via slider
    - speed buttons
    - optional GT overlay toggle via legend
    """
    if not PLOTLY_AVAILABLE:
        raise RuntimeError('Plotly is not installed. Run: pip install plotly')

    pred = reshape_xyz(predicted_xyz)
    gt = reshape_xyz(ground_truth_xyz) if ground_truth_xyz is not None else None
    n_frames = len(pred) if gt is None else min(len(pred), len(gt))
    pred = pred[:n_frames]
    if gt is not None:
        gt = gt[:n_frames]
        err = np.linalg.norm(pred - gt, axis=2)
        err_mean = err.mean(axis=1)
    else:
        err = None
        err_mean = np.zeros(n_frames)

    (xlim, ylim, zlim) = infer_axis_limits(pred, gt) if gt is not None else infer_axis_limits(pred)
    text = JOINTS if show_labels else None

    def frame_data(i, speed_label='normal'):
        pred_f = pred[i]
        traces = [
            _scatter3d_bones(pred_f, 'Prediction bones', 'royalblue'),
            _scatter3d_points(pred_f, 'Prediction joints', 'royalblue', text=text),
        ]
        if gt is not None:
            gt_f = gt[i]
            traces += [
                _scatter3d_bones(gt_f, 'Ground truth bones', 'green'),
                _scatter3d_points(gt_f, 'Ground truth joints', 'green', text=text),
            ]
        return traces

    frames = [go.Frame(data=frame_data(i), name=str(i), layout=go.Layout(title_text=f'Frame {i} | mean error={err_mean[i]:.4f}' if gt is not None else f'Frame {i}')) for i in range(n_frames)]

    fig = go.Figure(data=frame_data(0), frames=frames)
    fig.update_layout(
        title='Interactive 3D skeleton viewer',
        scene=dict(
            xaxis=dict(range=list(xlim), title='X'),
            yaxis=dict(range=list(ylim), title='Y'),
            zaxis=dict(range=list(zlim), title='Z'),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.3, y=1.3, z=0.8)),
        ),
        updatemenus=[
            dict(
                type='buttons',
                direction='left',
                x=0.0,
                y=1.15,
                buttons=[
                    dict(label='Play', method='animate', args=[None, {'frame': {'duration': 80, 'redraw': True}, 'fromcurrent': True}]),
                    dict(label='Pause', method='animate', args=[[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]),
                    dict(label='Slow', method='animate', args=[None, {'frame': {'duration': 180, 'redraw': True}, 'fromcurrent': True}]),
                    dict(label='Normal', method='animate', args=[None, {'frame': {'duration': 80, 'redraw': True}, 'fromcurrent': True}]),
                    dict(label='Fast', method='animate', args=[None, {'frame': {'duration': 30, 'redraw': True}, 'fromcurrent': True}]),
                ],
            )
        ],
        sliders=[{
            'pad': {'b': 10, 't': 35},
            'len': 0.95,
            'x': 0.03,
            'y': 0.0,
            'steps': [
                {
                    'args': [[str(i)], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                    'label': str(i),
                    'method': 'animate',
                }
                for i in range(n_frames)
            ],
        }],
        showlegend=True,
    )

    html_path = _as_path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path), include_plotlyjs='cdn')
    return html_path


# -----------------------------------------------------------------------------
# High-level workflow helpers
# -----------------------------------------------------------------------------

def create_evaluation_visuals(
    bundle_dir: Union[str, Path],
    frame_indices: Optional[Sequence[int]] = None,
    export_gif: bool = True,
    export_mp4: bool = False,
    export_html: bool = True,
    fps: int = 15,
    trail_length: int = 10,
) -> Dict[str, List[str]]:
    """
    Generate all standard outputs into:
    - bundle_dir/skeleton_plots/
    - bundle_dir/animations/
    """
    bundle = load_prediction_bundle(bundle_dir)
    pred = bundle['predicted_xyz']
    gt = bundle['ground_truth_xyz']
    n_frames = len(pred) if gt is None else min(len(pred), len(gt))

    plot_dir = _ensure_dir(_as_path(bundle_dir) / 'skeleton_plots')
    anim_dir = _ensure_dir(_as_path(bundle_dir) / 'animations')
    outputs = {'plots': [], 'animations': [], 'interactive': []}

    if frame_indices is None:
        frame_indices = sorted(set([0, max(0, n_frames // 2), max(0, n_frames - 1)]))

    for frame_idx in frame_indices:
        frame_idx = int(np.clip(frame_idx, 0, n_frames - 1))
        static_path = plot_dir / f'comparison_frame_{frame_idx:04d}.png'
        multiview_path = plot_dir / f'multiview_frame_{frame_idx:04d}.png'
        plot_frame_comparison(pred, gt, frame_idx=frame_idx, save_path=static_path)
        plt.close('all')
        plot_multiview_frame(pred, gt, frame_idx=frame_idx, save_path=multiview_path, trails=trail_length)
        plt.close('all')
        outputs['plots'] += [str(static_path), str(multiview_path)]

    if export_gif:
        gif_path = anim_dir / 'comparison_animation.gif'
        animate_skeletons_matplotlib(pred, gt, gif_path, fps=fps, trail_length=trail_length)
        outputs['animations'].append(str(gif_path))

    if export_mp4:
        mp4_path = anim_dir / 'comparison_animation.mp4'
        animate_skeletons_matplotlib(pred, gt, mp4_path, fps=fps, trail_length=trail_length)
        outputs['animations'].append(str(mp4_path))

    if export_html:
        html_path = anim_dir / 'interactive_viewer.html'
        export_interactive_html(pred, gt, html_path)
        outputs['interactive'].append(str(html_path))

    summary = {
        'bundle_dir': str(bundle_dir),
        'n_frames': int(n_frames),
        'has_ground_truth': gt is not None,
        'outputs': outputs,
    }
    with open(_as_path(bundle_dir) / 'visualization_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    return outputs


def save_prediction_bundle_from_model(
    model,
    X_input: np.ndarray,
    y_true_xyz: Optional[np.ndarray],
    output_dir: Union[str, Path],
    output_scaler=None,
    sequence_name: Optional[str] = None,
    metadata: Optional[Dict] = None,
):
    """
    Convenience helper for training code.
    - model.predict on X_input
    - optional inverse transform using output_scaler
    - save prediction bundle
    """
    pred = model.predict(X_input, verbose=0)
    if output_scaler is not None:
        pred = output_scaler.inverse_transform(pred)
        if y_true_xyz is not None:
            y_true_xyz = output_scaler.inverse_transform(y_true_xyz)

    return save_prediction_bundle(
        output_dir=output_dir,
        predicted_xyz=pred,
        ground_truth_xyz=y_true_xyz,
        sequence_name=sequence_name,
        metadata=metadata,
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='A10 3D skeleton visualizer')
    parser.add_argument('--bundle_dir', type=str, help='Folder containing predicted_xyz.npy and optional ground_truth_xyz.npy')
    parser.add_argument('--pred_npy', type=str, help='Path to predicted xyz .npy file')
    parser.add_argument('--gt_npy', type=str, default=None, help='Path to ground-truth xyz .npy file')
    parser.add_argument('--out_dir', type=str, default='visualizer_outputs', help='Output directory when using --pred_npy/--gt_npy')
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--no_html', action='store_true')
    parser.add_argument('--mp4', action='store_true')
    args = parser.parse_args()

    if args.bundle_dir:
        create_evaluation_visuals(
            bundle_dir=args.bundle_dir,
            export_gif=True,
            export_mp4=args.mp4,
            export_html=not args.no_html,
            fps=args.fps,
        )
        print(f'Visualization outputs created in {args.bundle_dir}')
    elif args.pred_npy:
        pred = np.load(args.pred_npy)
        gt = np.load(args.gt_npy) if args.gt_npy else None
        bundle = save_prediction_bundle(args.out_dir, pred, gt)
        create_evaluation_visuals(
            bundle_dir=bundle,
            export_gif=True,
            export_mp4=args.mp4,
            export_html=not args.no_html,
            fps=args.fps,
        )
        print(f'Visualization outputs created in {bundle}')
    else:
        parser.error('Provide either --bundle_dir or --pred_npy')
