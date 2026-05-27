"""A16 — Final unified server endpoint (capstone, week 22 deliverable).

Single entry point that runs the full chain in one call:

    video → MediaPipe pose → PoseNet→Kinect 2D → 2D→3D
        → start/stop cut → recording-quality (ugly) gate
        → good/bad classifier → 0–4 score (A15)

Two alternatives are scaffolded; only the **3D** variant is wired today. The
2D-only fast path is reserved as the named stretch alternative (see
``run_pipeline_2d``) so the public surface is stable when it lands.

Design notes
------------
- Heavy lifting is **delegated** to the existing :class:`ExercisePipeline`
  (``exercise_pipeline.py``). A16 only adds: (1) A15 scoring on top of the cut
  3D CSV, (2) a unified response schema, (3) per-stage timing instrumentation,
  (4) a stable JSON contract that the UI / future REST clients can rely on.
- Models that fail to load are reported in the response (``warnings``) instead
  of crashing the endpoint — the wider service must stay demo-able even when
  individual deep-learning artefacts are out of sync with the installed Keras.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"

# Stable response schema — bump on breaking changes.
A16_RESPONSE_VERSION = "1.0.0"

# Status enum kept narrow on purpose; UI / tests pattern-match on these.
STATUS_OK = "OK"
STATUS_REJECTED_UGLY = "REJECTED_UGLY_RECORDING"
STATUS_ERROR_NO_VIDEO = "ERROR_NO_VIDEO"
STATUS_ERROR_PIPELINE = "ERROR_PIPELINE"
STATUS_ERROR_SCORER = "ERROR_SCORER"
STATUS_ERROR_TOO_SHORT = "ERROR_TOO_SHORT_AFTER_CUT"


def _empty_timing() -> Dict[str, float]:
    return {
        "upstream_ms": 0.0,
        "scorer_nn_ms": 0.0,
        "scorer_total_ms": 0.0,
        "total_ms": 0.0,
    }


def _base_response(
    status: str,
    *,
    variant: str = "3D",
    message: str = "",
) -> Dict[str, Any]:
    """Skeleton response — every endpoint return goes through this."""
    return {
        "schema_version": A16_RESPONSE_VERSION,
        "endpoint": "A16",
        "variant": variant,
        "status": status,
        "message": message,
        "recording": {
            "quality_label": None,
            "quality_confidence": None,
            "threshold": None,
        },
        "segment": {
            "start_frame": None,
            "stop_frame": None,
            "duration_frames": None,
            "duration_sec": None,
            "total_frames": None,
        },
        "classification": {
            "label": None,
            "confidence": None,
        },
        "score": {
            "value": None,
            "band": None,
            "scale": "0=best, 4=worst",
        },
        "artefacts": {
            "full_3d_csv": None,
            "cut_3d_csv": None,
            "skeleton_mp4": None,
        },
        "timing_ms": _empty_timing(),
        "warnings": [],
    }


def _attach_upstream_fields(resp: Dict[str, Any], upstream: Dict[str, Any]) -> None:
    """Copy the relevant slice of the ``ExercisePipeline`` result into resp."""
    resp["recording"]["quality_label"] = upstream.get("recording_quality")
    resp["recording"]["quality_confidence"] = upstream.get("recording_confidence")
    resp["recording"]["threshold"] = upstream.get("recording_threshold")

    resp["segment"]["start_frame"] = upstream.get("start_frame")
    resp["segment"]["stop_frame"] = upstream.get("stop_frame")
    resp["segment"]["duration_frames"] = upstream.get("exercise_frames")
    resp["segment"]["duration_sec"] = upstream.get("exercise_duration_sec")
    resp["segment"]["total_frames"] = upstream.get("total_frames")

    resp["classification"]["label"] = upstream.get("quality_label")
    resp["classification"]["confidence"] = upstream.get("quality_confidence")


def _resolve_artefacts(resp: Dict[str, Any], video_path: str) -> Optional[Path]:
    """Populate artefact paths from the upstream stage. Returns cut CSV path."""
    stem = Path(video_path).stem
    full_csv = OUTPUTS_DIR / f"{stem}_3d_points.csv"
    cut_csv = OUTPUTS_DIR / f"{stem}_cut_3d_points.csv"
    skel_mp4 = OUTPUTS_DIR / f"{stem}_skeleton.mp4"

    resp["artefacts"]["full_3d_csv"] = str(full_csv) if full_csv.exists() else None
    resp["artefacts"]["cut_3d_csv"] = str(cut_csv) if cut_csv.exists() else None
    resp["artefacts"]["skeleton_mp4"] = str(skel_mp4) if skel_mp4.exists() else None
    return cut_csv if cut_csv.exists() else None


def run_pipeline_3d(
    video_path: Optional[str],
    quality_threshold: float = 0.6,
) -> Dict[str, Any]:
    """Run the full 3D A16 pipeline on one video.

    Parameters
    ----------
    video_path : str or None
        Path to the input video. ``None`` returns a structured error.
    quality_threshold : float
        Recording-quality threshold forwarded to :class:`ExercisePipeline`.

    Returns
    -------
    dict
        A response dictionary matching the schema in :data:`A16_RESPONSE_VERSION`.
        Errors are reported via ``status`` + ``message`` rather than raised.
    """
    t_total = time.perf_counter()
    resp = _base_response(STATUS_OK, variant="3D")

    if not video_path:
        resp["status"] = STATUS_ERROR_NO_VIDEO
        resp["message"] = "No video provided."
        resp["timing_ms"]["total_ms"] = (time.perf_counter() - t_total) * 1000.0
        return resp

    # ---- Stage 1-5: upstream pipeline (pose → 3D → cut → ugly/good-bad) ----
    t_up = time.perf_counter()
    try:
        # Local import keeps test-time mocking easy and avoids importing
        # TensorFlow when this module is merely inspected.
        from exercise_pipeline import ExercisePipeline

        pipeline = ExercisePipeline(quality_threshold=quality_threshold)
        try:
            upstream = pipeline.process_video(video_path)
        finally:
            pipeline.close()
    except Exception as exc:  # pragma: no cover — surfaced via response
        resp["status"] = STATUS_ERROR_PIPELINE
        resp["message"] = f"{type(exc).__name__}: {exc}"
        resp["warnings"].append(traceback.format_exc(limit=3))
        resp["timing_ms"]["upstream_ms"] = (time.perf_counter() - t_up) * 1000.0
        resp["timing_ms"]["total_ms"] = (time.perf_counter() - t_total) * 1000.0
        return resp
    resp["timing_ms"]["upstream_ms"] = (time.perf_counter() - t_up) * 1000.0

    if not upstream:
        resp["status"] = STATUS_ERROR_PIPELINE
        resp["message"] = "Upstream pipeline returned no result."
        resp["timing_ms"]["total_ms"] = (time.perf_counter() - t_total) * 1000.0
        return resp

    _attach_upstream_fields(resp, upstream)
    cut_csv = _resolve_artefacts(resp, video_path)

    # Ugly recording → early return, no scoring.
    if upstream.get("pipeline_stopped") or upstream.get("recording_quality") == "UGLY":
        resp["status"] = STATUS_REJECTED_UGLY
        resp["message"] = upstream.get(
            "reason", "Recording rejected by quality gate."
        )
        resp["timing_ms"]["total_ms"] = (time.perf_counter() - t_total) * 1000.0
        return resp

    # ---- Stage 6: A15 scoring on the cut 3D CSV ----
    if cut_csv is None:
        resp["status"] = STATUS_ERROR_PIPELINE
        resp["message"] = "Cut 3D CSV not produced by the upstream stage."
        resp["timing_ms"]["total_ms"] = (time.perf_counter() - t_total) * 1000.0
        return resp

    t_score = time.perf_counter()
    try:
        import pandas as pd
        from A15.inference import A15_C, predict_score

        df = pd.read_csv(cut_csv)
        if len(df) < A15_C:
            resp["status"] = STATUS_ERROR_TOO_SHORT
            resp["message"] = (
                f"Only {len(df)} cut frames available; scorer needs {A15_C}."
            )
        else:
            score, band, nn_ms = predict_score(df)
            resp["score"]["value"] = round(score, 4)
            resp["score"]["band"] = band
            resp["timing_ms"]["scorer_nn_ms"] = round(nn_ms, 2)
    except Exception as exc:
        resp["status"] = STATUS_ERROR_SCORER
        resp["message"] = f"Scorer failed: {type(exc).__name__}: {exc}"
        resp["warnings"].append(traceback.format_exc(limit=3))
    resp["timing_ms"]["scorer_total_ms"] = round(
        (time.perf_counter() - t_score) * 1000.0, 2
    )
    resp["timing_ms"]["upstream_ms"] = round(resp["timing_ms"]["upstream_ms"], 2)
    resp["timing_ms"]["total_ms"] = round(
        (time.perf_counter() - t_total) * 1000.0, 2
    )
    return resp


def run_pipeline_2d(
    video_path: Optional[str],
    quality_threshold: float = 0.6,
) -> Dict[str, Any]:
    """Reserved 2D-only alternative endpoint.

    Not implemented yet — kept as a named slot so the UI / clients can probe
    its existence and the schema stays stable when the 2D path lands. Returns
    a well-formed response with ``status = ERROR_PIPELINE`` and an explanatory
    message.
    """
    resp = _base_response(
        STATUS_ERROR_PIPELINE,
        variant="2D",
        message="2D alternative not implemented yet — see A16_Report.ipynb.",
    )
    return resp


def run_pipeline(
    video_path: Optional[str],
    quality_threshold: float = 0.6,
    variant: str = "3D",
) -> Dict[str, Any]:
    """Dispatcher — pick the 2D or 3D alternative."""
    variant = (variant or "3D").upper()
    if variant == "2D":
        return run_pipeline_2d(video_path, quality_threshold=quality_threshold)
    return run_pipeline_3d(video_path, quality_threshold=quality_threshold)


__all__ = [
    "A16_RESPONSE_VERSION",
    "STATUS_OK",
    "STATUS_REJECTED_UGLY",
    "STATUS_ERROR_NO_VIDEO",
    "STATUS_ERROR_PIPELINE",
    "STATUS_ERROR_SCORER",
    "STATUS_ERROR_TOO_SHORT",
    "run_pipeline",
    "run_pipeline_3d",
    "run_pipeline_2d",
]
