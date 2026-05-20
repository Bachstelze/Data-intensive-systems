"""
Gradio 3D MediaPipe Pose Livestream
===================================

A minimal Gradio app that streams from the webcam, runs MediaPipe's
``PoseLandmarker`` (Tasks API, ``pose_landmarker_lite.task``) on each frame
and renders:

* a 2D annotated frame (pose skeleton drawn straight from the Tasks-API
  ``PoseLandmarksConnections`` topology)
* a 3D plot of the world landmarks rendered with matplotlib using the same
  MediaPipe topology

This intentionally only uses the new ``mediapipe.tasks`` API. The legacy
``mediapipe.solutions`` / ``mediapipe.framework`` packages are *not*
required (they aren't always shipped with newer ``mediapipe`` wheels).

Dependencies on top of what ``mediapipe`` itself pulls in:
    pip install gradio opencv-python numpy matplotlib

Run:
    python gradio_app.py
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import cv2
import gradio as gr
import matplotlib

# Headless matplotlib backend so we can render plots in worker threads.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after backend switch)
import numpy as np  # noqa: E402

import mediapipe as mp  # noqa: E402
from mediapipe.tasks import python as mp_python  # noqa: E402
from mediapipe.tasks.python import vision as mp_vision  # noqa: E402
from mediapipe.tasks.python.vision import PoseLandmarksConnections  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "pose_landmarker_lite.task")

# Skeleton edges straight from the Tasks API. ``POSE_LANDMARKS`` is a list of
# ``Connection(start, end)``. We materialise it into a list of (int, int)
# tuples once for fast iteration.
POSE_EDGES: list[tuple[int, int]] = [
    (c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS
]


# ---------------------------------------------------------------------------
# Landmarker (singleton, VIDEO running mode for monotonic timestamps)
# ---------------------------------------------------------------------------

def _build_landmarker() -> mp_vision.PoseLandmarker:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Run `python download_model.py` first."
        )

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


_LANDMARKER: Optional[mp_vision.PoseLandmarker] = None
_T0_NS: int = 0


def _get_landmarker() -> mp_vision.PoseLandmarker:
    global _LANDMARKER, _T0_NS
    if _LANDMARKER is None:
        _LANDMARKER = _build_landmarker()
        _T0_NS = time.monotonic_ns()
    return _LANDMARKER


def _next_timestamp_ms() -> int:
    """Monotonically increasing ms timestamp required by VIDEO mode."""
    return (time.monotonic_ns() - _T0_NS) // 1_000_000


# ---------------------------------------------------------------------------
# 2D overlay
# ---------------------------------------------------------------------------

_POINT_COLOR = (0, 255, 255)   # yellow filled circle
_POINT_RING = (0, 0, 255)      # red outline
_LINE_COLOR = (0, 255, 0)      # green skeleton lines


def _draw_2d(frame_bgr: np.ndarray, landmarks, *, vis_thresh: float = 0.3) -> np.ndarray:
    """Draw the 2D pose skeleton on the BGR frame.

    ``landmarks`` is the per-image entry from
    ``PoseLandmarkerResult.pose_landmarks`` (a list of ``NormalizedLandmark``).
    """
    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]

    # Pre-compute pixel coordinates + visibility flag once.
    pts: list[tuple[int, int, float]] = []
    for lm in landmarks:
        vis = getattr(lm, "visibility", 1.0) or 0.0
        pts.append((int(lm.x * w), int(lm.y * h), float(vis)))

    # Bones first so joints draw on top of them.
    for a, b in POSE_EDGES:
        if a >= len(pts) or b >= len(pts):
            continue
        ax, ay, av = pts[a]
        bx, by, bv = pts[b]
        if av < vis_thresh or bv < vis_thresh:
            continue
        cv2.line(annotated, (ax, ay), (bx, by), _LINE_COLOR, 2, cv2.LINE_AA)

    for x, y, v in pts:
        if v < vis_thresh:
            continue
        cv2.circle(annotated, (x, y), 5, _POINT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(annotated, (x, y), 6, _POINT_RING, 1, cv2.LINE_AA)

    return annotated


# ---------------------------------------------------------------------------
# 3D world-landmark plot
# ---------------------------------------------------------------------------

def _render_3d(world_landmarks, *, vis_thresh: float = 0.3) -> np.ndarray:
    """Render the 3D world landmarks to an RGB image using matplotlib.

    Uses MediaPipe's ``POSE_LANDMARKS`` topology so visualization stays
    consistent with the 2D overlay.
    """
    fig = plt.figure(figsize=(5, 5), dpi=110)
    ax = fig.add_subplot(111, projection="3d")

    if world_landmarks is None:
        ax.set_title("No pose detected")
    else:
        xs = np.array([lm.x for lm in world_landmarks])
        ys = np.array([lm.y for lm in world_landmarks])
        zs = np.array([lm.z for lm in world_landmarks])
        vis = np.array([getattr(lm, "visibility", 1.0) for lm in world_landmarks])

        # MediaPipe world coords: x right, y down, z forward (towards the
        # camera is negative). Flip so "up" really is up and the subject
        # faces the viewer.
        plot_x = xs
        plot_y = -zs
        plot_z = -ys

        ax.scatter(plot_x, plot_y, plot_z, c="#FF3030", s=20, depthshade=True)

        for a, b in POSE_EDGES:
            if a >= len(world_landmarks) or b >= len(world_landmarks):
                continue
            if vis[a] < vis_thresh or vis[b] < vis_thresh:
                continue
            ax.plot(
                [plot_x[a], plot_x[b]],
                [plot_y[a], plot_y[b]],
                [plot_z[a], plot_z[b]],
                color="#30D158",
                linewidth=2,
            )

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

    ax.set_xlabel("x")
    ax.set_ylabel("z (depth)")
    ax.set_zlabel("y (height)")
    ax.view_init(elev=10, azim=-70)
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# Gradio callback
# ---------------------------------------------------------------------------

def process_frame(frame_rgb: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Process a single webcam frame.

    Gradio's ``Image(sources='webcam', streaming=True)`` delivers RGB numpy
    arrays. Returns a tuple ``(annotated_rgb, plot_3d_rgb)``.
    """
    if frame_rgb is None:
        return None, None

    landmarker = _get_landmarker()

    # Tasks API wants an mp.Image in SRGB (= RGB) format.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    try:
        result = landmarker.detect_for_video(mp_image, _next_timestamp_ms())
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[gradio_app] detect_for_video failed: {exc}")
        return frame_rgb, _render_3d(None)

    if not result.pose_landmarks:
        return frame_rgb, _render_3d(None)

    pose2d = result.pose_landmarks[0]
    pose3d = result.pose_world_landmarks[0] if result.pose_world_landmarks else None

    # 2D overlay (work in BGR for OpenCV, swap back at the end).
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    annotated_bgr = _draw_2d(bgr, pose2d)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    plot_rgb = _render_3d(pose3d)
    return annotated_rgb, plot_rgb


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    with gr.Blocks(title="MediaPipe 3D Pose Livestream") as demo:
        gr.Markdown(
            "# MediaPipe 3D Pose Livestream\n"
            "Live webcam pose estimation using **MediaPipe Tasks** "
            "(`pose_landmarker_lite.task`). The left panel shows the 2D "
            "skeleton overlay; the right panel shows the 3D world landmarks."
        )

        with gr.Row():
            webcam = gr.Image(
                sources=["webcam"],
                streaming=True,
                type="numpy",
                label="Webcam (input)",
            )

        with gr.Row():
            out_2d = gr.Image(type="numpy", label="2D pose overlay", streaming=True)
            out_3d = gr.Image(type="numpy", label="3D world landmarks", streaming=True)

        webcam.stream(
            fn=process_frame,
            inputs=[webcam],
            outputs=[out_2d, out_3d],
            stream_every=0.1,  # ~10 FPS, easy on the CPU
            show_progress="hidden",
        )

    return demo


if __name__ == "__main__":
    # Eagerly load the model so the first frame isn't slow.
    _get_landmarker()
    build_demo().launch()
