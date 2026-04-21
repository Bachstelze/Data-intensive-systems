"""
Generate PoseNet/MoveNet (xpn, ypn) CSVs from video files.

For each .avi video in --videos-dir, runs MoveNet on every frame, projects the
17 COCO keypoints to the 13 Kinect-aligned joints used by Issue #40, and writes
one CSV per video to --out-dir with columns:

    FrameNo, head_x, head_y, left_shoulder_x, left_shoulder_y, ...

The FrameNo column preserves the video frame index (0-based), so temporal
alignment with the corresponding Kinect CSV (which may start at a later
FrameNo such as 68) can be done later in the data loader.

Usage:
    python3 generate_posenet_data.py \
        --videos-dir /Users/amol/Desktop/LNU/LNU_Masters/intensive/all_videos \
        --out-dir   /Users/amol/Desktop/LNU/LNU_Masters/intensive/second_github/Data-intensive-systems/posenet_preprocessed \
        --model lightning

    # limit to first N videos (useful for a quick test):
    python3 generate_posenet_data.py ... --limit 3
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

# Make the A8 pose_estimator importable (MoveNetPoseEstimator, KEYPOINT_NAMES)
A8_PATH = Path(
    "/Users/amol/Desktop/LNU/LNU_Masters/intensive/github/Data-intensive-systems/A8"
)
if str(A8_PATH) not in sys.path:
    sys.path.insert(0, str(A8_PATH))

from pose_estimator import MoveNetPoseEstimator, KEYPOINT_NAMES  # noqa: E402


def _load_movenet_local(model_name: str = 'lightning') -> MoveNetPoseEstimator:
    """
    Build a MoveNetPoseEstimator using a locally cached Kaggle-hub SavedModel.

    tfhub.dev is deprecated and currently returns 404 for MoveNet; we fetch
    the model from Kaggle Hub instead (kagglehub.model_download) and wire it
    into the existing A8 estimator class.
    """
    import kagglehub
    import tensorflow as tf

    kaggle_handle = {
        'lightning': 'google/movenet/tensorFlow2/singlepose-lightning',
        'thunder':   'google/movenet/tensorFlow2/singlepose-thunder',
    }[model_name]
    model_dir = kagglehub.model_download(kaggle_handle)

    est = MoveNetPoseEstimator.__new__(MoveNetPoseEstimator)
    est.model_name = model_name
    est.input_size = MoveNetPoseEstimator.INPUT_SIZES[model_name]
    est.model = tf.saved_model.load(model_dir)
    est.movenet = est.model.signatures['serving_default']
    print(f"Loaded MoveNet '{model_name}' from {model_dir}")
    return est

# Kinect-aligned joint order (must match data_loader.KINECT_JOINTS).
KINECT_JOINTS = [
    'head', 'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow',
    'left_hand', 'right_hand', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_foot', 'right_foot',
]

# Kinect joint -> MoveNet COCO keypoint.
KINECT_TO_MOVENET = {
    'head': 'nose',
    'left_shoulder': 'left_shoulder',
    'right_shoulder': 'right_shoulder',
    'left_elbow': 'left_elbow',
    'right_elbow': 'right_elbow',
    'left_hand': 'left_wrist',
    'right_hand': 'right_wrist',
    'left_hip': 'left_hip',
    'right_hip': 'right_hip',
    'left_knee': 'left_knee',
    'right_knee': 'right_knee',
    'left_foot': 'left_ankle',
    'right_foot': 'right_ankle',
}

# Pre-computed COCO indices in Kinect order.
KEYPOINT_INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
COCO_IDX_IN_KINECT_ORDER = [
    KEYPOINT_INDEX[KINECT_TO_MOVENET[j]] for j in KINECT_JOINTS
]

CSV_HEADER = ['FrameNo'] + [
    f"{j}_{axis}" for j in KINECT_JOINTS for axis in ('x', 'y')
]


def video_id_from_filename(video_path: Path) -> str:
    """A1.avi -> A1_kinect (matches Kinect CSV base name)."""
    return f"{video_path.stem}_kinect"


def process_video(
    estimator: MoveNetPoseEstimator,
    video_path: Path,
    out_csv: Path,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    written = 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            raw = estimator.detect_pose_raw(frame)  # (17, 3) -> [y, x, conf]
            row: List[float] = [frame_idx]
            for coco_idx in COCO_IDX_IN_KINECT_ORDER:
                y, x, _conf = raw[coco_idx]
                row.extend([float(x), float(y)])
            writer.writerow(row)
            written += 1
            frame_idx += 1

            if frame_idx % 50 == 0:
                print(f"  {video_path.name}: {frame_idx}/{total_frames}", flush=True)

    cap.release()
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--model', choices=['lightning', 'thunder'], default='lightning')
    parser.add_argument('--limit', type=int, default=0,
                        help="Process only the first N videos (0 = all).")
    parser.add_argument('--overwrite', action='store_true',
                        help="Re-generate CSVs even if they already exist.")
    parser.add_argument('--pattern', default='*.avi',
                        help="Glob pattern to find videos (default *.avi).")
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(videos_dir.glob(args.pattern))
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        print(f"No videos found in {videos_dir} matching {args.pattern}")
        return

    print(f"Found {len(videos)} video(s). Output -> {out_dir}")
    print(f"Loading MoveNet '{args.model}' ...")
    estimator = _load_movenet_local(args.model)

    t0 = time.time()
    total_frames = 0
    for i, v in enumerate(videos, 1):
        out_csv = out_dir / f"{video_id_from_filename(v)}.csv"
        if out_csv.exists() and not args.overwrite:
            print(f"[{i}/{len(videos)}] skip existing {out_csv.name}")
            continue
        print(f"[{i}/{len(videos)}] {v.name} -> {out_csv.name}")
        n = process_video(estimator, v, out_csv)
        total_frames += n

    dt = time.time() - t0
    print(f"\nDone. Processed {total_frames} frames in {dt:.1f}s "
          f"({total_frames / dt if dt else 0:.1f} fps)")


if __name__ == '__main__':
    main()
