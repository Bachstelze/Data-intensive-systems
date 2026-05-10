import os
import sys
import cv2
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

JOINTS = [
    'head', 'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow',
    'left_hand', 'right_hand', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_foot', 'right_foot'
]

KINECT_TO_MOVENET = {
    'head':           'nose',
    'left_shoulder':  'left_shoulder',
    'left_elbow':     'left_elbow',
    'right_shoulder': 'right_shoulder',
    'right_elbow':    'right_elbow',
    'left_hand':      'left_wrist',
    'right_hand':     'right_wrist',
    'left_hip':       'left_hip',
    'right_hip':      'right_hip',
    'left_knee':      'left_knee',
    'right_knee':     'right_knee',
    'left_foot':      'left_ankle',
    'right_foot':     'right_ankle',
}

# Feature columns
KINECT_3D_COLS = [f'{j}_{c}' for j in JOINTS for c in ['x', 'y', 'z']]  # 39
KINECT_2D_COLS = [f'{j}_{c}' for j in JOINTS for c in ['x', 'y']]        # 26
EXTRA_COLS = [
    'left_hand_to_left_shoulder', 'right_hand_to_right_shoulder',
    'left_hand_to_left_hip',      'right_hand_to_right_hip',
    'left_elbow_to_left_shoulder','right_elbow_to_right_shoulder',
    'head_to_hip',
    'head_vx',       'head_vy',       'head_vz',       'head_speed',
    'left_hand_vx',  'left_hand_vy',  'left_hand_vz',  'left_hand_speed',
    'right_hand_vx', 'right_hand_vy', 'right_hand_vz', 'right_hand_speed',
    'head_ax',       'head_ay',       'head_az',        'head_accel',
    'left_hand_ax',  'left_hand_ay',  'left_hand_az',   'left_hand_accel',
    'right_hand_ax', 'right_hand_ay', 'right_hand_az',  'right_hand_accel',
]

SEGMENT_FEATURE_COLS = KINECT_3D_COLS + EXTRA_COLS 
WINDOW_SIZE = 30    
CONFIDENCE_THRESHOLD = 0.75


def calculate_distances(df):
    df = df.copy()
    df['left_hand_to_left_shoulder'] = np.sqrt(
        (df['left_hand_x'] - df['left_shoulder_x'])**2 +
        (df['left_hand_y'] - df['left_shoulder_y'])**2 +
        (df['left_hand_z'] - df['left_shoulder_z'])**2)
    df['right_hand_to_right_shoulder'] = np.sqrt(
        (df['right_hand_x'] - df['right_shoulder_x'])**2 +
        (df['right_hand_y'] - df['right_shoulder_y'])**2 +
        (df['right_hand_z'] - df['right_shoulder_z'])**2)
    df['left_hand_to_left_hip'] = np.sqrt(
        (df['left_hand_x'] - df['left_hip_x'])**2 +
        (df['left_hand_y'] - df['left_hip_y'])**2 +
        (df['left_hand_z'] - df['left_hip_z'])**2)
    df['right_hand_to_right_hip'] = np.sqrt(
        (df['right_hand_x'] - df['right_hip_x'])**2 +
        (df['right_hand_y'] - df['right_hip_y'])**2 +
        (df['right_hand_z'] - df['right_hip_z'])**2)
    df['left_elbow_to_left_shoulder'] = np.sqrt(
        (df['left_elbow_x'] - df['left_shoulder_x'])**2 +
        (df['left_elbow_y'] - df['left_shoulder_y'])**2 +
        (df['left_elbow_z'] - df['left_shoulder_z'])**2)
    df['right_elbow_to_right_shoulder'] = np.sqrt(
        (df['right_elbow_x'] - df['right_shoulder_x'])**2 +
        (df['right_elbow_y'] - df['right_shoulder_y'])**2 +
        (df['right_elbow_z'] - df['right_shoulder_z'])**2)
    df['head_to_hip'] = np.sqrt(
        (df['head_x'] - (df['left_hip_x']+df['right_hip_x'])/2)**2 +
        (df['head_y'] - (df['left_hip_y']+df['right_hip_y'])/2)**2 +
        (df['head_z'] - (df['left_hip_z']+df['right_hip_z'])/2)**2)
    return df


def calculate_velocities(df, fps=30.0):
    df = df.copy()
    for joint in ['head', 'left_hand', 'right_hand']:
        for axis in ['x', 'y', 'z']:
            col = f'{joint}_{axis}'
            df[f'{joint}_v{axis}'] = np.diff(
                df[col], prepend=df[col].iloc[0]) * fps
        df[f'{joint}_speed'] = np.sqrt(
            df[f'{joint}_vx']**2 +
            df[f'{joint}_vy']**2 +
            df[f'{joint}_vz']**2)
    return df


def calculate_accelerations(df, fps=30.0):
    df = df.copy()
    for joint in ['head', 'left_hand', 'right_hand']:
        for axis in ['x', 'y', 'z']:
            vcol = f'{joint}_v{axis}'
            df[f'{joint}_a{axis}'] = np.diff(
                df[vcol], prepend=df[vcol].iloc[0]) * fps
        df[f'{joint}_accel'] = np.sqrt(
            df[f'{joint}_ax']**2 +
            df[f'{joint}_ay']**2 +
            df[f'{joint}_az']**2)
    return df


def engineer_features(df):
    df = calculate_distances(df)
    df = calculate_velocities(df)
    df = calculate_accelerations(df)
    return df


class ExercisePipeline:
    def _load_model_safe(self, path):
        """
        Load a Keras model handling version mismatches.
        Tries multiple loading strategies in order.
        """
        custom_objs = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'mae': tf.keras.losses.MeanAbsoluteError(),
            'rmse': tf.keras.metrics.RootMeanSquaredError(),
            'MeanSquaredError': tf.keras.losses.MeanSquaredError(),
            'MeanAbsoluteError': tf.keras.losses.MeanAbsoluteError()
        }
        try:
            model = tf.keras.models.load_model(path, compile=False, custom_objects=custom_objs)
            print(f"    Loaded with standard method")
            return model
        except Exception as e1:
            print(f"    Standard load failed: {type(e1).__name__}")

        try:
            from tensorflow.keras.layers import Dense as OriginalDense
            from tensorflow.keras import layers

            class PatchedDense(layers.Dense):
                def __init__(self, *args, **kwargs):
                    kwargs.pop('quantization_config', None)  
                    super().__init__(*args, **kwargs)

            model = tf.keras.models.load_model(
                path,
                compile=False,
                custom_objects={'Dense': PatchedDense}
            )
            print(f"    Loaded with patched Dense")
            return model
        except Exception as e2:
            print(f"    Patched load failed: {type(e2).__name__}")

        # Strategy 3: Use tf.saved_model loader
        try:
            model = tf.saved_model.load(path)
            print(f"    Loaded as SavedModel")
            return model
        except Exception as e3:
            print(f"    SavedModel load failed: {type(e3).__name__}")

        raise RuntimeError(
            f"Could not load model from {path}.\n"
            f"using the same Keras version as this pipeline."
        )

    def __init__(self,
                model_posenet_to_k2d='week16_result.h5',
                model_k2d_to_k3d='week15_2d_to_3d.h5',
                model_segment='week17_start_and_stop.h5',
                scaler_segment='week17_start_and_stop.pkl',
                scaler_onestep_X='week16_scaler_X.pkl',
                scaler_onestep_y='week16_scaler_y.pkl'):

        print("=" * 55)
        print("Initialising Exercise Pipeline")
        print("=" * 55)

        # Step 1: MoveNet 
        print("\n[1] Loading MoveNet pose estimator...")
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from pose_estimator import MoveNetPoseEstimator
            self.pose_estimator = MoveNetPoseEstimator(model_name='lightning')
            print("    MoveNet loaded")
        except Exception as e:
            print(f"    ERROR: {e}")
            raise

        # Step 2: PoseNet→Kinect2D  
        print(f"\n[2] Loading PoseNet→Kinect2D: {model_posenet_to_k2d}")
        self.model_p2k = self._load_model_safe(MODELS_DIR /model_posenet_to_k2d)  
        
        try:
            self.scaler_onestep_X = joblib.load(MODELS_DIR /scaler_onestep_X)
            self.scaler_onestep_y = joblib.load(MODELS_DIR /scaler_onestep_y)
            print("    One-Step Scalers (X and y) loaded")
        except Exception as e:
            print(f"    WARNING: Could not load One-Step scalers: {e}")
            self.scaler_onestep_X = None
            self.scaler_onestep_y = None

        self.p2k_output_dim = self.model_p2k.output_shape[-1]
        if self.p2k_output_dim == 39:
            print(f"    One-step model detected — outputs full (xk,yk,zk) directly")
            print(f"    model_3d will be skipped")
        elif self.p2k_output_dim == 26:
            print(f"    Two-step model detected — model_3d needed for z prediction")

        # Step 3: Kinect2D→3D  
        print(f"\n[3] Loading Kinect2D→3D: {model_k2d_to_k3d}")
        self.model_3d = self._load_model_safe(MODELS_DIR /model_k2d_to_k3d)  
        print(f"    Input: {self.model_3d.input_shape}  "
            f"Output: {self.model_3d.output_shape}")

        # Step 4: Start/Stop classifier  
        print(f"\n[4] Loading Start/Stop classifier: {model_segment}")
        self.model_segment = self._load_model_safe(MODELS_DIR /model_segment)  
        self.scaler_segment = joblib.load(MODELS_DIR / scaler_segment)
        self.segment_is_seq = len(self.model_segment.input_shape) == 3
        print(f"    Sequence model: {self.segment_is_seq}")

        self.quality_enabled = False

    def _extract_posenet_coords(self, frame):
        result = self.pose_estimator.detect_pose(frame)
        kps    = result['keypoints']

        coords = []
        for joint in JOINTS:
            movenet_name = KINECT_TO_MOVENET[joint]
            kp = kps[movenet_name]
            coords.extend([kp['x'], kp['y']])

        return np.array(coords, dtype=np.float32)   # shape (26,)


    def _predict_kinect_2d(self, posenet_coords):
        
        X = posenet_coords.reshape(1, -1)

        # 2. Apply Input Scaling (PoseNet 2D)
        if self.scaler_onestep_X is not None:
            X_scaled = self.scaler_onestep_X.transform(X)
        else:
            # Fallback if scaler missing
            X_scaled = X - 0.5 

        # 3. Run the model (Prediction)
        output_scaled = self.model_p2k.predict(X_scaled, verbose=0)

        # 4. Apply Output Un-scaling 
        if self.scaler_onestep_y is not None:
            output = self.scaler_onestep_y.inverse_transform(output_scaled)
        else:
            output = output_scaled

        return output.flatten()                 # (26,)


    def _predict_kinect_3d(self, kinect_2d_window):
        X = np.array(kinect_2d_window).reshape(1, 30, 26)
        z = self.model_3d.predict(X, verbose=0)
        if len(z.shape) == 3:
            z_last = z[0, -1, :] # Take last frame of the sequence
        else:
            z_last = z[0, :]     # Take the only frame provided
        current_xy = kinect_2d_window[-1].reshape(13, 2)   # last frame xy
        current_z  = z_last.reshape(13, 1)
        xyz = np.hstack([current_xy, current_z])
        return xyz.flatten()


    def _build_dataframe(self, frame_data_list):
        df = pd.DataFrame(frame_data_list, columns=KINECT_3D_COLS)
        df.index.name = 'FrameNo'
        return df


    def _predict_exercise_segment(self, df_3d):
        feat_df = engineer_features(df_3d)
        available_cols = [c for c in SEGMENT_FEATURE_COLS if c in feat_df.columns]
        feat_arr = feat_df[available_cols].values.astype(np.float32)

        n_frames   = len(feat_arr)
        n_features = feat_arr.shape[1]
        predictions = []

        if self.segment_is_seq:
            # Sequence model (Conv1D/LSTM/GRU) — sliding window
            for i in range(n_frames - WINDOW_SIZE + 1):
                window    = feat_arr[i : i + WINDOW_SIZE]   # (30, n_features)
                window_sc = self.scaler_segment.transform(
                    window).reshape(1, WINDOW_SIZE, n_features)
                prob = self.model_segment.predict(window_sc, verbose=0)
                pred = 1 if prob[0][0] >= CONFIDENCE_THRESHOLD else int(np.argmax(prob[0]))
                predictions.append(pred)

            predictions = [0] * (WINDOW_SIZE - 1) + predictions

        else:
            # Dense model — one frame at a time
            for i in range(n_frames):
                frame    = feat_arr[i].reshape(1, -1)            # (1, n_features)
                frame_sc = self.scaler_segment.transform(frame)  # (1, n_features)
                prob     = self.model_segment.predict(frame_sc, verbose=0)

                if prob.shape[-1] == 1:
                    pred = 1 if prob[0][0] >= CONFIDENCE_THRESHOLD else 0
                else:
                    pred = int(np.argmax(prob[0]))
                predictions.append(pred)

        predictions = np.array(predictions)
        print(f"  Predictions: {predictions.sum()} exercise frames "
            f"out of {n_frames} total")

        # Find start: first 0→1 transition
        start_frame = None
        for i in range(1, len(predictions)):
            if predictions[i-1] == 0 and predictions[i] == 1:
                start_frame = i
                break

        # Find stop: last 1→0 transition
        stop_frame = None
        for i in range(len(predictions)-1, 0, -1):
            if predictions[i-1] == 1 and predictions[i] == 0:
                stop_frame = i
                break

        # Fallback if no clean transition found
        if start_frame is None and predictions.sum() > 0:
            start_frame = int(np.where(predictions == 1)[0][0])
        if stop_frame is None and predictions.sum() > 0:
            stop_frame  = int(np.where(predictions == 1)[0][-1])

        # Safety bounds
        if start_frame is not None:
            start_frame = max(0, start_frame)
        if stop_frame is not None:
            stop_frame  = min(n_frames - 1, stop_frame)

        return start_frame, stop_frame, predictions


    # def _create_annotated_video(self, video_path, start_frame, stop_frame,
    #                          df_cut, output_path):
    #     """
    #     Create trimmed annotated video with skeleton overlay.
    #     Draws joint circles and skeleton lines on each exercise frame.
        
    #     df_cut: DataFrame with 3D skeleton for exercise frames only
    #     """
    #     cap    = cv2.VideoCapture(str(video_path))
    #     fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    #     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     out    = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        

    #     # Skeleton connections — which joints to draw lines between
    #     SKELETON_CONNECTIONS = [
    #         ('head',           'left_shoulder'),
    #         ('head',           'right_shoulder'),
    #         ('left_shoulder',  'right_shoulder'),
    #         ('left_shoulder',  'left_elbow'),
    #         ('left_elbow',     'left_hand'),
    #         ('right_shoulder', 'right_elbow'),
    #         ('right_elbow',    'right_hand'),
    #         ('left_shoulder',  'left_hip'),
    #         ('right_shoulder', 'right_hip'),
    #         ('left_hip',       'right_hip'),
    #         ('left_hip',       'left_knee'),
    #         ('left_knee',      'left_foot'),
    #         ('right_hip',      'right_knee'),
    #         ('right_knee',     'right_foot'),
    #     ]

    #     frame_idx    = 0
    #     cut_frame_idx = 0   # index into df_cut

    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             break

    #         # Only process and write exercise frames
    #         if start_frame <= frame_idx <= stop_frame:

    #             # Get 3D skeleton for this frame
    #             if cut_frame_idx < len(df_cut):
    #                 row = df_cut.iloc[cut_frame_idx]
    #                 joint_pixels = {}
    #                 for joint in JOINTS:
    #                     x_3d = row[f'{joint}_x']   # Kinect metres (-1 to 1)
    #                     y_3d = row[f'{joint}_y']   # Kinect metres (-1 to 1)

    #                     px = int((x_3d + 1.0) / 2.0 * width)
    #                     py = int((1.0 - (y_3d + 1.0) / 2.0) * height)

    #                     # Clamp to image bounds
    #                     px = max(0, min(width-1,  px))
    #                     py = max(0, min(height-1, py))

    #                     joint_pixels[joint] = (px, py)

    #                 # Draw skeleton lines
    #                 for joint_a, joint_b in SKELETON_CONNECTIONS:
    #                     if joint_a in joint_pixels and joint_b in joint_pixels:
    #                         cv2.line(frame,
    #                                 joint_pixels[joint_a],
    #                                 joint_pixels[joint_b],
    #                                 (0, 255, 0), 2)   # green lines

    #                 # Draw joint circles
    #                 for joint, (px, py) in joint_pixels.items():
    #                     cv2.circle(frame, (px, py), 5,
    #                             (0, 255, 255), -1)  # yellow filled
    #                     cv2.circle(frame, (px, py), 5,
    #                             (0, 0, 0), 1)        # black border

    #                 cut_frame_idx += 1

    #             # Draw text overlay
    #             cv2.putText(frame,
    #                         f"EXERCISE  frame {frame_idx}",
    #                         (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
    #                         1.0, (0, 255, 0), 2)
    #             cv2.putText(frame,
    #                         f"start={start_frame}  stop={stop_frame}",
    #                         (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
    #                         0.7, (0, 255, 0), 2)

    #             out.write(frame)

    #         frame_idx += 1

    #     cap.release()
    #     out.release()
    #     print(f"  Annotated video saved: {output_path.name}")


    #  MAIN ENTRY POINT 

    def process_video(self, video_path):
        window_buffer = [] 
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"ERROR: Video not found: {video_path}")
            return None

        stem    = video_path.stem
        out_dir = OUTPUT_DIR

        print(f"\nProcessing: {video_path.name}")

        print("\nStage 1-3: Extracting 3D skeleton from video...")
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  Total frames: {total_frames}")

        all_3d = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # --- 1. Get PoseNet dots from MoveNet (0.0 to 1.0 range) ---
            posenet_coords = self._extract_posenet_coords(frame)

            # --- 2. Scale the PoseNet input (StandardScaler X) ---
            X_input = posenet_coords.reshape(1, -1)
            if hasattr(self, 'scaler_onestep_X') and self.scaler_onestep_X is not None:
                X_input = self.scaler_onestep_X.transform(X_input)

            # --- 3. Run the model prediction ---
            kinect_output_scaled = self.model_p2k.predict(X_input, verbose=0)

            if self.p2k_output_dim == 39:
                # ONE-STEP MODEL PATH
                if hasattr(self, 'scaler_onestep_y') and self.scaler_onestep_y is not None:
                    kinect_3d = self.scaler_onestep_y.inverse_transform(kinect_output_scaled).flatten()
                else:
                    kinect_3d = kinect_output_scaled.flatten()

            else:
                kinect_coords = kinect_output_scaled.flatten()
                window_buffer.append(kinect_coords)
                if len(window_buffer) < 30:
                    frame_idx += 1
                    continue
                if len(window_buffer) > 30:
                    window_buffer.pop(0)
                kinect_3d = self._predict_kinect_3d(window_buffer)

            all_3d.append(kinect_3d)
            frame_idx += 1

            if frame_idx % 30 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames...")

        cap.release()
        print(f"  Done. Extracted {frame_idx} frames.")

        ## smoothing
        if len(all_3d) > 0:
            print("  Smoothing 3D skeleton with PoseInterpolator...")
            try:
                from pose_interpolator import PoseInterpolator
                arr_3d = np.array(all_3d)          # (F, 39)
                arr_3d = arr_3d.reshape(-1, 13, 3) # (F, 13, 3) — joints × xyz

                arr_xy = arr_3d[:, :, :2]          # (F, 13, 2) — x,y only
                conf   = np.ones((*arr_xy.shape[:2], 1))  # (F, 13, 1) confidence=1
                arr_xyc = np.concatenate([arr_xy, conf], axis=2)  # (F, 13, 3)

                interp = PoseInterpolator(strategy='moving_average', window_size=5)
                smoothed_xyc = interp.fit_transform(arr_xyc)  # (F, 13, 3)

                # Put smoothed x,y back, keep original z
                arr_3d[:, :, 0] = smoothed_xyc[:, :, 0]  # smoothed x
                arr_3d[:, :, 1] = smoothed_xyc[:, :, 1]  # smoothed y

                all_3d = list(arr_3d.reshape(-1, 39))  # back to list of 39-value arrays
                print("  Smoothing done (moving_average strategy)")

            except Exception as e:
                print(f"  PoseInterpolator not available: {e}")
                print("  Falling back to simple moving average...")
                smoothed_3d = []
                window_size = 5
                for i in range(len(all_3d)):
                    start_idx = max(0, i - window_size // 2)
                    end_idx   = min(len(all_3d), i + window_size // 2 + 1)
                    smoothed_frame = np.mean(all_3d[start_idx:end_idx], axis=0)
                    smoothed_3d.append(smoothed_frame)
                all_3d = smoothed_3d

        df_3d = self._build_dataframe(all_3d)

        # Save full 3D points
        full_csv = out_dir / f"{stem}_3d_points.csv"
        df_3d.to_csv(str(full_csv))
        print(f"\n  Saved full 3D points: {full_csv.name}  ({len(df_3d)} frames)")

        # STAGE 4: Find exercise segment 
        print("\nStage 4: Detecting exercise start and stop...")
        start_frame, stop_frame, predictions = self._predict_exercise_segment(df_3d)

        if start_frame is None or stop_frame is None:
            print("  WARNING: Could not detect exercise segment.")
            start_frame = 0
            stop_frame  = len(df_3d) - 1

        print(f"  Exercise detected: frame {start_frame} → {stop_frame}")
        print(f"  Duration: {stop_frame - start_frame} frames "
              f"= {(stop_frame-start_frame)/30:.1f}s at 30fps")

        # Save exercise-only 3D points
        df_cut = df_3d.iloc[start_frame : stop_frame + 1].copy()
        cut_csv = out_dir / f"{stem}_cut_3d_points.csv"
        df_cut.to_csv(str(cut_csv))
        print(f"  Saved cut 3D points: {cut_csv.name}  ({len(df_cut)} frames)")

        print("\nGenerating 3D skeleton animation...")
        try:
            from generate_skeleton_animation import render_skeleton_video
            skeleton_out = out_dir / f"{stem}_skeleton.mp4"
            render_skeleton_video(
                csv_path=str(cut_csv),
                output_path=str(skeleton_out),
                fps=30
            )
            print(f"  Skeleton animation: {skeleton_out.name}")
        except Exception as e:
            print(f"  Skeleton animation skipped: {e}")

        # Save results JSON
        results = {
            "video":         video_path.name,
            "total_frames":  frame_idx,
            "start_frame":   int(start_frame) if start_frame is not None else None,
            "stop_frame":    int(stop_frame)  if stop_frame  is not None else None,
            "exercise_frames": int(stop_frame - start_frame + 1),
            "exercise_duration_sec": round((stop_frame - start_frame + 1) / 30.0, 2),
            "quality_label": "(A13 not ready)",
            "pipeline_version": "A8-A12"
        }
        json_path = out_dir / f"{stem}_results.json"
        with open(str(json_path), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved results JSON: {json_path.name}")

        # Create annotated video
        # print("\nCreating annotated video...")
        # video_out = out_dir / f"{stem}_annotated.mp4"
        # self._create_annotated_video(video_path, start_frame, stop_frame, df_cut, video_out)

        # ── SUMMARY ───────────────────────────────────────────────────────
        print(f"\n{'='*55}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*55}")
        print(f"  Video          : {video_path.name}")
        print(f"  Total frames   : {frame_idx}")
        print(f"  Exercise start : frame {start_frame}")
        print(f"  Exercise stop  : frame {stop_frame}")
        print(f"  Exercise length: {stop_frame - start_frame + 1} frames "
              f"({(stop_frame-start_frame+1)/30:.1f}s)")
        print(f"\n  Files generated:")
        print(f"    {full_csv.name}      ← all 3D skeleton data")
        print(f"    {cut_csv.name}  ← exercise-only 3D data")
        print(f"    {json_path.name}      ← start/stop metadata")
        print(f"\n  A13 results wait")

        return results


# CLI entry point 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Exercise pipeline A8-A12')
    parser.add_argument('--video',        required=True,
                        help='Path to input video file')
    parser.add_argument('--model_p2k',    default='week16_result.h5',
                        help='PoseNet→Kinect2D model (Lab 10)')
    parser.add_argument('--model_3d',     default='week15_2d_to_3d.h5',
                        help='Kinect2D→3D model (Lab 9)')
    parser.add_argument('--model_seg',    default='week17_start_and_stop.h5',
                        help='Start/Stop classifier (Lab 11)')
    parser.add_argument('--scaler_seg',   default='week17_start_and_stop.pkl',
                        help='Scaler for Lab 11 model')
    args = parser.parse_args()

    pipeline = ExercisePipeline(
        model_posenet_to_k2d=args.model_p2k,
        model_k2d_to_k3d=args.model_3d,
        model_segment=args.model_seg,
        scaler_segment=args.scaler_seg,
    )
    pipeline.process_video(args.video)
