from __future__ import annotations

import json
import time
from pathlib import Path

from exercise_pipeline import ExercisePipeline


def run_full_pipeline(video_path: str):
    """
    Runs the existing exercise pipeline and adapts its saved output files
    to the Gradio UI.
    """

    start_time = time.time()

    pipeline = ExercisePipeline()

    print("\n[DEBUG] Running full pipeline...")
    result = pipeline.process_video(video_path)

    if result is None:
        raise ValueError("Pipeline failed to process video.")

    print("\n[DEBUG] Pipeline result:")
    print(result)

    video_path = Path(video_path)
    stem = video_path.stem

    output_dir = Path("outputs")

    cut_csv_path = output_dir / f"{stem}_cut_3d_points.csv"
    full_csv_path = output_dir / f"{stem}_3d_points.csv"
    animation_video_path = output_dir / f"{stem}_skeleton.mp4"
    results_json_path = output_dir / f"{stem}_results.json"

    if not cut_csv_path.exists():
        raise FileNotFoundError(f"Expected cut CSV was not found: {cut_csv_path}")

    if not animation_video_path.exists():
        raise FileNotFoundError(f"Expected skeleton animation video was not found: {animation_video_path}")

    if results_json_path.exists():
        try:
            results_json = json.loads(results_json_path.read_text())
        except Exception:
            results_json = {}
    else:
        results_json = {}

    quality_label = result.get("quality_label", results_json.get("quality_label", "UNKNOWN"))

    confidence = (
        result.get("quality_confidence")
        or results_json.get("quality_confidence")
        or results_json.get("confidence")
        or 0.0
    )

    elapsed = (time.time() - start_time) * 1000

    return {
        "annotated_video": None,
        "animation_video": make_browser_playable(animation_video_path),
        "csv_path": str(cut_csv_path),
        "full_csv_path": str(full_csv_path) if full_csv_path.exists() else None,
        "results_json_path": str(results_json_path) if results_json_path.exists() else None,
        "classification": {
            "label": quality_label,
            "confidence": confidence,
        },
        "metadata": {
            "inference_time_ms": elapsed,
            "total_frames": result.get("total_frames"),
            "start_frame": result.get("start_frame"),
            "stop_frame": result.get("stop_frame"),
            "exercise_frames": result.get("exercise_frames"),
            "exercise_duration_sec": result.get("exercise_duration_sec"),
            "pipeline_version": result.get("pipeline_version"),
        },
        "raw_pipeline_result": result,
    }

import subprocess

def make_browser_playable(video_path):
    video_path = Path(video_path)
    fixed_path = video_path.with_name(video_path.stem + "_browser.mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(fixed_path),
    ]

    try:
        subprocess.run(cmd, check=True)
        return str(fixed_path)
    except Exception as e:
        print(f"[WARNING] Could not convert video for browser playback: {e}")
        return str(video_path)