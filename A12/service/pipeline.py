from __future__ import annotations

import csv
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .schemas import ClassificationResult, PipelineMetadata, PipelineOutput

try:
    from A8.pose_estimator import MoveNetPoseEstimator
except Exception:  # pragma: no cover - used only if A8 is unavailable during isolated tests
    MoveNetPoseEstimator = None

try:
    from A12.pose_interpolator import smooth_pose_sequence
except Exception:  # pragma: no cover
    smooth_pose_sequence = None

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# 13-joint subset used by the classification data in A13.
CLASSIFIER_JOINTS = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "nose",
]

SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"), ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


def validate_video(video_path: str | None) -> Path:
    if not video_path:
        raise ValueError("A video file is required.")
    path = Path(video_path)
    if not path.exists():
        raise ValueError(f"Video file does not exist: {video_path}")
    if path.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
        raise ValueError("Unsupported video type. Use mp4, mov, avi, mkv, webm, or m4v.")
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened()
    cap.release()
    if not ok:
        raise ValueError("OpenCV could not open this video. Try re-encoding it as mp4.")
    return path


def _pose_result_to_frame_dict(pose_result: Dict[str, Any], frame_id: int, timestamp_s: float) -> Dict[str, Any]:
    keypoints = pose_result.get("keypoints", {}) or {}
    out = []
    for name in KEYPOINT_NAMES:
        kp = keypoints.get(name, {}) or {}
        out.append({
            "name": name,
            "x": kp.get("x"),
            "y": kp.get("y"),
            "z": 0.0,
            "score": kp.get("confidence", kp.get("score", 0.0)),
        })
    return {
        "frame_id": frame_id,
        "timestamp": timestamp_s,
        "poses": [{"pose_id": 0, "keypoints": out}],
        "inference_time_ms": pose_result.get("inference_time_ms", 0.0),
    }


def _get_pose_estimator():
    if MoveNetPoseEstimator is None:
        raise RuntimeError("A8.pose_estimator.MoveNetPoseEstimator could not be imported.")
    return MoveNetPoseEstimator(model_name="lightning")


def _detect_motion_window(frames: List[Dict[str, Any]], min_window: int = 10) -> Tuple[int, int]:
    """Cut leading/trailing frames by motion energy of 13 classifier joints."""
    if len(frames) <= min_window:
        return 0, max(0, len(frames) - 1)

    coords = []
    for f in frames:
        kp_by_name = {kp["name"]: kp for kp in f["poses"][0]["keypoints"]}
        row = []
        for name in CLASSIFIER_JOINTS:
            kp = kp_by_name.get(name, {})
            x = kp.get("x")
            y = kp.get("y")
            row.append([np.nan if x is None else float(x), np.nan if y is None else float(y)])
        coords.append(row)
    arr = np.asarray(coords, dtype="float32")
    arr = np.nan_to_num(arr, nan=np.nanmean(arr) if not np.isnan(arr).all() else 0.0)
    velocity = np.linalg.norm(np.diff(arr, axis=0), axis=-1).mean(axis=1)
    if len(velocity) == 0 or float(np.max(velocity)) == 0.0:
        return 0, len(frames) - 1

    threshold = max(float(np.percentile(velocity, 35)), float(np.max(velocity)) * 0.08)
    active = np.where(velocity >= threshold)[0]
    if active.size == 0:
        return 0, len(frames) - 1

    start = max(0, int(active[0]) - 2)
    end = min(len(frames) - 1, int(active[-1]) + 3)
    if end - start + 1 < min_window:
        mid = (start + end) // 2
        start = max(0, mid - min_window // 2)
        end = min(len(frames) - 1, start + min_window - 1)
    return start, end


def _frames_to_classifier_sequence(frames: List[Dict[str, Any]], sequence_len: int = 10) -> np.ndarray:
    """Return B-problem PoseNet-like data: shape (10, 13, 2)."""
    if not frames:
        raise ValueError("No pose frames were produced from the video.")
    indices = np.linspace(0, len(frames) - 1, sequence_len, dtype=int)
    sequence = np.zeros((sequence_len, len(CLASSIFIER_JOINTS), 2), dtype="float32")
    for out_i, frame_i in enumerate(indices):
        kp_by_name = {kp["name"]: kp for kp in frames[frame_i]["poses"][0]["keypoints"]}
        for joint_i, name in enumerate(CLASSIFIER_JOINTS):
            kp = kp_by_name.get(name, {})
            sequence[out_i, joint_i, 0] = 0.0 if kp.get("x") is None else float(kp.get("x"))
            sequence[out_i, joint_i, 1] = 0.0 if kp.get("y") is None else float(kp.get("y"))
    # normalize per clip to make dimensions roughly model-friendly
    max_abs = float(np.max(np.abs(sequence)))
    if max_abs > 1.5:
        sequence[:, :, 0] /= max_abs
        sequence[:, :, 1] /= max_abs
    return sequence


def _classify(sequence: np.ndarray) -> ClassificationResult:
    """Use a real persisted model if present; otherwise deterministic dummy classifier.

    Replace this function with the Issue #10 champion wrapper when it is merged.
    Expected future wrapper: A13.models.champion.predict_proba(sequence[None, ...]).
    """
    try:
        from A13.models.champion import predict_good_probability  # type: ignore
        good_prob = float(predict_good_probability(sequence))
        mode = "champion_model"
    except Exception:
        # Deterministic fallback so app integration can be finished before #10/#11 is merged.
        # Higher motion smoothness is treated as slightly better. This is not a scientific classifier.
        velocity = np.linalg.norm(np.diff(sequence, axis=0), axis=-1).mean()
        good_prob = float(np.clip(0.55 + (0.08 - velocity) * 0.8, 0.05, 0.95))
        mode = "deterministic_dummy_until_issue_10_model_is_available"
    label = "good" if good_prob >= 0.5 else "bad"
    return ClassificationResult(
        label=label,
        is_good=label == "good",
        confidence=max(good_prob, 1.0 - good_prob),
        probabilities={"bad": 1.0 - good_prob, "good": good_prob, "mode": mode},
    )


def _write_keypoints_csv(frames: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "timestamp", "joint", "x", "y", "z", "confidence"])
        for frame in frames:
            for kp in frame["poses"][0]["keypoints"]:
                writer.writerow([
                    frame["frame_id"], frame["timestamp"], kp["name"], kp.get("x"), kp.get("y"), kp.get("z", 0.0), kp.get("score", 0.0)
                ])


def _write_animation_json(frames: List[Dict[str, Any]], path: Path) -> None:
    skeleton_frames = []
    for frame in frames:
        joints = {kp["name"]: {"x": kp.get("x"), "y": kp.get("y"), "z": kp.get("z", 0.0), "confidence": kp.get("score", 0.0)}
                  for kp in frame["poses"][0]["keypoints"]}
        skeleton_frames.append({"frame_id": frame["frame_id"], "timestamp": frame["timestamp"], "joints": joints})
    path.write_text(json.dumps({"joint_names": KEYPOINT_NAMES, "edges": SKELETON_EDGES, "frames": skeleton_frames}, indent=2))


def run_video_pipeline(
    video_path: str | None,
    confidence_threshold: float = 0.3,
    smoothing_strategy: str = "exponential",
    smoothing_method: str = "zscore",
    output_dir: str | os.PathLike[str] = "pose_outputs",
) -> PipelineOutput:
    start_time = time.perf_counter()
    warnings: List[str] = []
    video = validate_video(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0 or fps > 240:
        warnings.append(f"Invalid FPS reported by OpenCV ({fps}); using 30 FPS.")
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose_estimator = _get_pose_estimator()
    original_frames: List[np.ndarray] = []
    pose_frames: List[Dict[str, Any]] = []
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        original_frames.append(frame)
        pose_result = pose_estimator.detect_pose(frame)
        pose_frames.append(_pose_result_to_frame_dict(pose_result, frame_id, frame_id / fps))
        frame_id += 1
    cap.release()

    if not pose_frames:
        raise ValueError("The video contained no readable frames.")

    if smooth_pose_sequence is not None:
        try:
            pose_frames = smooth_pose_sequence(
                pose_frames,
                strategy=smoothing_strategy,
                outlier_method=smoothing_method,
                outlier_threshold=3.0,
                window_size=7,
                min_confidence=0.2,
            )
        except Exception as exc:
            warnings.append(f"Smoothing failed; using unsmoothed poses. Error: {exc}")
    else:
        warnings.append("A12.pose_interpolator could not be imported; using unsmoothed poses.")

    cut_start, cut_end = _detect_motion_window(pose_frames)
    cut_pose_frames = pose_frames[cut_start:cut_end + 1]
    cut_video_frames = original_frames[cut_start:cut_end + 1]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    annotated_video = out_dir / f"a12_annotated_cut_{timestamp}.mp4"
    keypoints_csv = out_dir / f"a12_keypoints_cut_{timestamp}.csv"
    animation_json = out_dir / f"a12_animation_data_{timestamp}.json"

    writer = cv2.VideoWriter(str(annotated_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame, pose_frame in zip(cut_video_frames, cut_pose_frames):
        # Redraw via estimator on frame. If a future A11 pipeline returns annotated frames, replace this block.
        pose_result = pose_estimator.detect_pose(frame)
        annotated = pose_estimator.draw_keypoints(frame, pose_result, confidence_threshold=confidence_threshold)
        writer.write(annotated)
    writer.release()

    _write_keypoints_csv(cut_pose_frames, keypoints_csv)
    _write_animation_json(cut_pose_frames, animation_json)

    sequence = _frames_to_classifier_sequence(cut_pose_frames)
    classification = _classify(sequence)
    classifier_mode = str(classification.probabilities.pop("mode", "unknown"))

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return PipelineOutput(
        annotated_video_path=str(annotated_video),
        animation_data_path=str(animation_json),
        keypoints_csv_path=str(keypoints_csv),
        classification=classification,
        metadata=PipelineMetadata(
            model_version="A12-pipeline-integration-v1",
            inference_time_ms=elapsed_ms,
            frame_count_original=len(pose_frames),
            frame_count_cut=len(cut_pose_frames),
            fps=fps,
            cut_start_frame=cut_start,
            cut_end_frame=cut_end,
            smoothing_strategy=f"{smoothing_strategy}/{smoothing_method}",
            classifier_mode=classifier_mode,
        ),
        warnings=warnings,
    )
