"""
generate_posenet_data.py  —  STEP 1 of Task 1
==============================================
Generates PoseNet CSV files from Kinect videos using MoveNet.
Run this FIRST, then run one_step_model.py.

Expected folder structure:
    project_root/
        Datasets_all/
            all_videos/                  <- professor's Kinect .avi files
                A1.avi, A159.avi, B22.avi ...
            kinect_good_preprocessed/    <- already exist
                A1_kinect.csv ...
            posenet_data/                <- CREATED BY THIS SCRIPT
                A1_kinect.csv ...        <- PoseNet x,y coordinates
        A10/
            generate_posenet_data.py     <- THIS FILE
            one_step_model.py

Video to CSV name matching:
    A1.avi  ->  A1_kinect.csv   (strips extension, adds _kinect.csv)
    B22.avi ->  B22_kinect.csv

Joint mapping (MoveNet 17 -> Kinect 13):
    nose          -> head
    left_shoulder -> left_shoulder
    left_elbow    -> left_elbow
    right_shoulder-> right_shoulder
    right_elbow   -> right_elbow
    left_wrist    -> left_hand
    right_wrist   -> right_hand
    left_hip      -> left_hip
    right_hip     -> right_hip
    left_knee     -> left_knee
    right_knee    -> right_knee
    left_ankle    -> left_foot
    right_ankle   -> right_foot
    (eyes and ears skipped - no Kinect equivalent)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths — relative to this file's location (A10 folder) ────────────────────
THIS_DIR        = Path(__file__).parent          # A10/
PROJECT_ROOT    = THIS_DIR.parent                # project_root/
DATASETS_DIR    = PROJECT_ROOT / 'Datasets_all'
VIDEO_DIR       = DATASETS_DIR / 'all_videos'
KINECT_CSV_DIR  = DATASETS_DIR / 'kinect_good_preprocessed'
POSENET_OUT_DIR = DATASETS_DIR / 'posenet_data'

# ── Settings ──────────────────────────────────────────────────────────────────
MOVENET_MODEL        = 'lightning'
CONFIDENCE_THRESHOLD = 0.3
VIDEO_EXTENSIONS     = ['.avi', '.mp4', '.mov', '.mkv']

# ── Kinect joints in exact order that models.py uses ─────────────────────────
KINECT_JOINTS = [
    'head', 'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow',
    'left_hand', 'right_hand', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_foot', 'right_foot',
]

# MoveNet name for each Kinect joint
KINECT_TO_MOVENET = {
    'head': 'nose', 'left_shoulder': 'left_shoulder',
    'left_elbow': 'left_elbow', 'right_shoulder': 'right_shoulder',
    'right_elbow': 'right_elbow', 'left_hand': 'left_wrist',
    'right_hand': 'right_wrist', 'left_hip': 'left_hip',
    'right_hip': 'right_hip', 'left_knee': 'left_knee',
    'right_knee': 'right_knee', 'left_foot': 'left_ankle',
    'right_foot': 'right_ankle',
}

# Output CSV columns: FrameNo + 13 joints x 2 = 27 total
OUTPUT_COLUMNS = ['FrameNo']
for j in KINECT_JOINTS:
    OUTPUT_COLUMNS += [f'{j}_x', f'{j}_y']


def process_video(video_path, out_csv, estimator, kinect_csv=None):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f'  Cannot open: {video_path.name}')
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f'  {total} frames @ {fps:.1f}fps')

    if kinect_csv and kinect_csv.exists():
        k = sum(1 for _ in open(kinect_csv)) - 1
        if abs(total - k) > 5:
            print(f'  WARNING: video={total} frames, kinect_csv={k} frames')

    rows, idx, low = [], 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        kps = estimator.detect_pose(frame)['keypoints']
        row = {'FrameNo': idx}
        for j in KINECT_JOINTS:
            kp = kps[KINECT_TO_MOVENET[j]]
            if kp['confidence'] >= CONFIDENCE_THRESHOLD:
                row[f'{j}_x'] = round(float(kp['x']), 6)
                row[f'{j}_y'] = round(float(kp['y']), 6)
            else:
                row[f'{j}_x'] = row[f'{j}_y'] = 0.0
                low += 1
        rows.append(row)
        idx += 1
        if idx % 30 == 0:
            print(f'  {idx}/{total} frames...')
    cap.release()

    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(str(out_csv), index=False)
    print(f'  Saved {idx} frames -> {out_csv.name}')
    if low:
        print(f'  {low} low-conf keypoints set to 0.0')
    return idx


def main():
    POSENET_OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not VIDEO_DIR.exists():
        print(f'ERROR: Video folder not found:\n  {VIDEO_DIR}')
        return

    videos = sorted([f for f in VIDEO_DIR.iterdir()
                     if f.suffix.lower() in VIDEO_EXTENSIONS])
    if not videos:
        print(f'No video files found in {VIDEO_DIR}')
        return

    print(f'Found {len(videos)} videos')
    print(f'Output -> {POSENET_OUT_DIR}\n')

    # Load pose_estimator.py
    for path in [THIS_DIR, THIS_DIR.parent / 'A8',
                 THIS_DIR.parent / 'A9', THIS_DIR.parent]:
        sys.path.insert(0, str(path))

    try:
        from pose_estimator import MoveNetPoseEstimator
    except ImportError:
        print('ERROR: pose_estimator.py not found')
        print('Place pose_estimator.py in the same A10/ folder or in A8/')
        return

    print(f'Loading MoveNet {MOVENET_MODEL}...')
    est = MoveNetPoseEstimator(model_name=MOVENET_MODEL)
    print('Ready!\n')

    ok, skip, fail = [], [], []

    for i, v in enumerate(videos):
        base    = v.stem                         # 'A1' from 'A1.avi'
        csv_name = f'{base}_kinect.csv'           # 'A1_kinect.csv'
        out_csv  = POSENET_OUT_DIR / csv_name
        k_csv    = KINECT_CSV_DIR  / csv_name

        print(f'[{i+1}/{len(videos)}] {v.name} -> {csv_name}')

        if not k_csv.exists():
            print(f'  WARNING: No matching Kinect CSV: {csv_name}')

        if out_csv.exists():
            print(f'  Already done, skipping')
            skip.append(v.name)
            continue

        try:
            n = process_video(v, out_csv, est, k_csv)
            ok.append({'name': v.name, 'frames': n, 'csv': csv_name})
        except Exception as e:
            print(f'  ERROR: {e}')
            fail.append({'name': v.name, 'error': str(e)})

    print(f'\n{"="*50}')
    print(f'Done  —  OK:{len(ok)}  Skipped:{len(skip)}  Failed:{len(fail)}')
    if fail:
        for f in fail:
            print(f'  FAILED: {f["name"]} — {f["error"]}')
    if ok:
        # Verify first output
        first = POSENET_OUT_DIR / ok[0]['csv']
        df = pd.read_csv(str(first))
        xc = [c for c in df.columns if c.endswith('_x')]
        yc = [c for c in df.columns if c.endswith('_y')]
        print(f'\nVerify {first.name}: {len(df)} frames, {len(df.columns)} cols')
        print(f'  X range: [{df[xc].min().min():.3f}, {df[xc].max().max():.3f}]')
        print(f'  Y range: [{df[yc].min().min():.3f}, {df[yc].max().max():.3f}]')
        print(f'  (Values should be 0.0 to 1.0)')
    print(f'\nNow run: python one_step_model.py')


if __name__ == '__main__':
    main()
