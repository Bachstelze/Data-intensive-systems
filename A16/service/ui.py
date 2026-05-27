"""Gradio UI tab for the A16 final unified endpoint."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from A16.service.endpoint import (
    STATUS_OK,
    STATUS_REJECTED_UGLY,
    run_pipeline_3d,
)


# ---------------------------------------------------------------------------
# Live skeleton overlay (webcam → MediaPipe + OpenCV draw)
#
# Self-contained inline implementation: we deliberately do *not* call the
# A14 livestream helper because that one also renders a matplotlib 3D plot
# (~200-500 ms per frame), which capped the overlay at ~10 FPS. Here we run
# MediaPipe Tasks API + OpenCV drawing only, target ~30 FPS so the overlay
# is comfortable to watch.
# ---------------------------------------------------------------------------

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "A14", "pose_landmarker_lite.task"
)

# Lazy singletons — initialised on the first webcam frame.
_LIVE_LANDMARKER = None
_LIVE_LANDMARKER_LOCK = threading.Lock()
_LIVE_T0_NS = 0
_LIVE_EDGES: list = []

# Colours (BGR for OpenCV draw; converted back to RGB for Gradio output).
_LINE_BGR = (0, 255, 0)        # green skeleton
_POINT_BGR = (0, 255, 255)     # yellow filled
_POINT_RING_BGR = (0, 0, 255)  # red outline


def _ensure_live_landmarker():
    """Build the MediaPipe Tasks PoseLandmarker once (VIDEO mode)."""
    global _LIVE_LANDMARKER, _LIVE_T0_NS, _LIVE_EDGES
    if _LIVE_LANDMARKER is not None:
        return _LIVE_LANDMARKER

    with _LIVE_LANDMARKER_LOCK:
        if _LIVE_LANDMARKER is not None:
            return _LIVE_LANDMARKER

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.vision import PoseLandmarksConnections

        base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        _LIVE_LANDMARKER = mp_vision.PoseLandmarker.create_from_options(options)
        _LIVE_T0_NS = time.monotonic_ns()
        _LIVE_EDGES = [
            (c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS
        ]
    return _LIVE_LANDMARKER


def _next_live_timestamp_ms() -> int:
    return (time.monotonic_ns() - _LIVE_T0_NS) // 1_000_000


def _live_process_frame(frame_rgb: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Overlay a live 2D pose skeleton on a webcam RGB frame.

    Returns the original frame on errors / no-pose so the preview never goes
    blank. Designed to be fast (no matplotlib, no 3D plot) so the stream
    can run at ~30 FPS.
    """
    if frame_rgb is None:
        return None
    try:
        import cv2
        import mediapipe as mp
        landmarker = _ensure_live_landmarker()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect_for_video(
            mp_image, _next_live_timestamp_ms()
        )
    except Exception:  # pragma: no cover — keep the stream alive
        return frame_rgb

    if not result.pose_landmarks:
        return frame_rgb

    landmarks = result.pose_landmarks[0]
    h, w = frame_rgb.shape[:2]

    pts = []
    for lm in landmarks:
        vis = getattr(lm, "visibility", 1.0) or 0.0
        pts.append((int(lm.x * w), int(lm.y * h), float(vis)))

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    vis_thresh = 0.3
    for a, b in _LIVE_EDGES:
        if a >= len(pts) or b >= len(pts):
            continue
        ax, ay, av = pts[a]
        bx, by, bv = pts[b]
        if av < vis_thresh or bv < vis_thresh:
            continue
        cv2.line(bgr, (ax, ay), (bx, by), _LINE_BGR, 2, cv2.LINE_AA)
    for x, y, v in pts:
        if v < vis_thresh:
            continue
        cv2.circle(bgr, (x, y), 5, _POINT_BGR, -1, cv2.LINE_AA)
        cv2.circle(bgr, (x, y), 6, _POINT_RING_BGR, 1, cv2.LINE_AA)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Endpoint result rendering
# ---------------------------------------------------------------------------

def _format_summary(resp: Dict[str, Any]) -> str:
    """Render the response as a human-readable Markdown block."""
    rec = resp["recording"]
    seg = resp["segment"]
    cls = resp["classification"]
    sc = resp["score"]
    t = resp["timing_ms"]

    lines = [
        f"### A16 Final Endpoint — {resp['variant']} variant",
        f"**Status:** `{resp['status']}`",
    ]
    if resp.get("message"):
        lines.append(f"**Message:** {resp['message']}")

    lines += [
        "",
        "#### Recording quality (ugly gate)",
        f"- Label: **{rec['quality_label']}**",
        f"- Confidence: **{rec['quality_confidence']}** "
        f"(threshold {rec['threshold']})",
        "",
        "#### Exercise segment (start/stop cut)",
        f"- Frames: **{seg['start_frame']} → {seg['stop_frame']}** "
        f"of {seg['total_frames']}",
        f"- Duration: **{seg['duration_frames']} frames "
        f"≈ {seg['duration_sec']} s**",
        "",
        "#### Good / Bad classification",
        f"- Label: **{cls['label']}**",
        f"- Confidence: **{cls['confidence']}**",
        "",
        "#### Score (0–4, lower is better)",
        f"- Score: **{sc['value']}**",
        f"- Band: **{sc['band']}**",
        "",
        "#### Timing (ms)",
        f"- Upstream (pose + 3D + cut + ugly/good-bad): **{t['upstream_ms']}**",
        f"- Scorer total: **{t['scorer_total_ms']}**  "
        f"(NN only: **{t['scorer_nn_ms']}**)",
        f"- End-to-end total: **{t['total_ms']}**",
    ]
    if resp.get("warnings"):
        lines += ["", "#### Warnings"]
        for w in resp["warnings"]:
            lines.append(f"- {w.splitlines()[0] if w else ''}")
    return "\n".join(lines)


def _status_badge(resp: Dict[str, Any]) -> str:
    """Compact one-line status string for the textbox."""
    if resp["status"] == STATUS_OK:
        return f"OK — score {resp['score']['value']} ({resp['score']['band']})"
    if resp["status"] == STATUS_REJECTED_UGLY:
        return (
            f"REJECTED — ugly recording "
            f"(conf {resp['recording']['quality_confidence']})"
        )
    return f"{resp['status']} — {resp.get('message', '')}"


def run_a16_tab(
    video_path: str,
    quality_threshold: float,
) -> Tuple[str, str, Any, Dict[str, Any]]:
    """Gradio callback for the A16 tab.

    Returns ``(status_text, summary_markdown, skeleton_video, full_json)``.
    """
    resp = run_pipeline_3d(video_path, quality_threshold=quality_threshold)
    skeleton_video = resp["artefacts"].get("skeleton_mp4")
    return _status_badge(resp), _format_summary(resp), skeleton_video, resp


def build_a16_tab(gr):
    """Build the A16 Gradio tab. ``gr`` is the imported ``gradio`` module.

    Kept as a builder so [app.py](../../app.py) can import and mount the tab
    without re-implementing the layout.
    """
    with gr.TabItem("A16 Final Endpoint"):
        gr.Markdown(
            """
            ## A16 — Final unified endpoint (3D alternative)

            The webcam preview below runs a live **MediaPipe** pose overlay
            (~30 FPS target, CPU only) so you can frame yourself before
            recording. Upload a clip further down to run the full Part-II
            chain: **pose → PoseNet→Kinect 2D → 2D→3D → start/stop cut →
            ugly/good-bad → 0–4 score**. A 2D-only alternative is reserved
            on the same response schema (see
            `A16.service.endpoint.run_pipeline_2d`).
            """
        )

        # ---- Live camera with skeleton overlay (headline) ------------------
        with gr.Row():
            a16_webcam = gr.Image(
                sources=["webcam"],
                streaming=True,
                type="numpy",
                label="Webcam (input)",
                mirror_webcam=True,
            )
            a16_overlay = gr.Image(
                type="numpy",
                label="Live pose overlay",
                streaming=True,
            )
        a16_webcam.stream(
            fn=_live_process_frame,
            inputs=[a16_webcam],
            outputs=[a16_overlay],
            # ~30 FPS target; Gradio caps to actual network round-trip on
            # slower hosts, which is fine — the overlay just degrades to
            # whatever the link can do.
            stream_every=0.033,
            show_progress="hidden",
        )

        # ---- Recorded-video endpoint ---------------------------------------
        gr.Markdown("### Score a recorded exercise")
        with gr.Row():
            with gr.Column():
                a16_video = gr.Video(label="Upload exercise video")
                a16_threshold = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.6, step=0.05,
                    label="Recording quality threshold "
                          "(detection-rate based — 0.6 = pose visible "
                          "in 60% of the first 30 frames)",
                )
                a16_run = gr.Button(
                    "Run A16 endpoint", variant="primary"
                )

            with gr.Column():
                a16_status = gr.Textbox(
                    label="Status", interactive=False
                )
                a16_summary = gr.Markdown()
                a16_video_out = gr.Video(label="3D skeleton animation")
                a16_json = gr.JSON(label="Full response (A16 schema)")

        a16_run.click(
            fn=run_a16_tab,
            inputs=[a16_video, a16_threshold],
            outputs=[a16_status, a16_summary, a16_video_out, a16_json],
        )
