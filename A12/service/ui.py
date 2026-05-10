from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from .pipeline import run_video_pipeline, validate_video



def run_a12_video_tab(video_path: str, confidence_threshold: float, smoothing_strategy: str, smoothing_method: str):
    """Gradio callback: video -> annotated video, animation json, keypoints csv, JSON, Markdown."""
    try:
        result = run_video_pipeline(
            video_path=video_path,
            confidence_threshold=confidence_threshold,
            smoothing_strategy=smoothing_strategy,
            smoothing_method=smoothing_method,
        )
        payload = result.to_json()
        summary = f"""
### A12 pipeline completed

- **Classification:** `{payload['classification']['label']}`
- **Confidence:** `{payload['classification']['confidence']:.3f}`
- **Frames:** `{payload['metadata']['frame_count_cut']}` cut from `{payload['metadata']['frame_count_original']}` original frames
- **Cut window:** frames `{payload['metadata']['cut_start_frame']}` to `{payload['metadata']['cut_end_frame']}`
- **Classifier mode:** `{payload['metadata']['classifier_mode']}`
"""
        if payload["warnings"]:
            summary += "\n### Warnings\n" + "\n".join(f"- {w}" for w in payload["warnings"])
        return result.annotated_video_path, result.animation_data_path, result.keypoints_csv_path, payload, summary
    except Exception as exc:
        return None, None, None, {"error": str(exc)}, f"### Error\n{exc}"


# Backward-compatible function name used by the uploaded app.py A12 CSV prototype.
def run_a12_tab(csv_path: str, problem: str = "B"):
    if not csv_path:
        return {"error": "Upload a CSV file first."}, "### Error\nUpload a CSV file first."
    return {
        "message": "This CSV endpoint was superseded by the Issue #12 video pipeline tab.",
        "csv_path": csv_path,
        "problem": problem,
    }, "### CSV received\nThe Issue #12 implementation uses the video pipeline endpoint."
