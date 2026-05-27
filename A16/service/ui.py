"""Gradio UI tab for the A16 final unified endpoint."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from A16.service.endpoint import (
    STATUS_OK,
    STATUS_REJECTED_UGLY,
    run_pipeline_3d,
)


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

            **Record** a clip with your webcam (or upload one), then click
            **Run A16 endpoint**. The result appears on the right: a video
            with the **skeleton overlaid** on your recording, plus the
            full Part-II chain output — **pose → PoseNet→Kinect 2D →
            2D→3D → start/stop cut → ugly/good-bad → 0–4 score**.

            > Processing is currently CPU-only and runs the full chain
            > end-to-end, so a 5-10 s clip can take roughly 20-60 s on
            > the HF Space.
            """
        )

        with gr.Row():
            with gr.Column():
                a16_video = gr.Video(
                    label="Record or upload exercise video",
                    sources=["webcam", "upload"],
                )
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
                a16_video_out = gr.Video(
                    label="Skeleton overlay (3D animation)"
                )
                a16_summary = gr.Markdown()
                a16_json = gr.JSON(label="Full response (A16 schema)")

        a16_run.click(
            fn=run_a16_tab,
            inputs=[a16_video, a16_threshold],
            outputs=[a16_status, a16_summary, a16_video_out, a16_json],
        )
