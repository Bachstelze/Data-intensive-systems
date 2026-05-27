from PIL import Image
import gradio as gr
from A8.pose_estimator import MoveNetPoseEstimator
from A12.pose_interpolator import smooth_pose_sequence
#http://127.0.0.1:7860from A12.service.ui import run_a12_tab
from A12.service.ui import run_a12_video_tab
from A16.service.ui import build_a16_tab
from exercise_pipeline import ExercisePipeline
import json
import csv
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import cv2
import tempfile
import time

# --- A15 scoring model (lazy-loaded) -------------------------------------
A15_JOINTS = [
    'head', 'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow',
    'left_hand', 'right_hand', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_foot', 'right_foot',
]
A15_C = 10  # frames per clip the scorer was trained on
_A15_MODEL = None
_A15_SCALER = None


def _load_a15_scorer():
    """Lazy-load the deployed regression scorer (issue #20 wiring)."""
    global _A15_MODEL, _A15_SCALER
    if _A15_MODEL is not None and _A15_SCALER is not None:
        return _A15_MODEL, _A15_SCALER
    import joblib
    from tensorflow import keras
    from tensorflow.keras import layers
    repo_root = Path(__file__).parent
    model_path = repo_root / 'models' / 'scoring_model.keras'
    scaler_path = repo_root / 'models' / 'scoring_scaler.pkl'
    try:
        _A15_MODEL = keras.models.load_model(str(model_path))
    except (TypeError, ValueError):
        # Saved with a newer Keras (e.g. extra `quantization_config` kwarg);
        # rebuild Dense_medium and load weights only. Architecture matches
        # training_summary.json's deployed champion.
        inp = keras.Input(shape=(390,))
        x = layers.Dense(64, activation='relu')(inp)
        x = layers.Dropout(0.2)(x)
        out = layers.Dense(1, activation='linear')(x)
        _A15_MODEL = keras.Model(inp, out, name='Dense')
        _A15_MODEL.load_weights(str(model_path))
    _A15_SCALER = joblib.load(str(scaler_path))
    return _A15_MODEL, _A15_SCALER


def _a15_sample_frames(df) -> np.ndarray:
    df.columns = df.columns.str.strip()
    idx = np.linspace(0, len(df) - 1, A15_C).astype(int)
    sub = df.iloc[idx]
    frames = []
    for _, row in sub.iterrows():
        frames.append([[row[f'{j}_x'], row[f'{j}_y'], row[f'{j}_z']]
                       for j in A15_JOINTS])
    return np.array(frames, dtype=np.float32)


def _a15_score_band(score: float) -> str:
    if score < 1.0:
        return "GREEN — acceptable form (0-1)"
    if score < 2.0:
        return "AMBER — borderline (1-2)"
    return "RED — poor form (2-4)"


def run_a15_scoring(video_path, quality_threshold):
    """End-to-end A15 scoring: video → cut 3D CSV → 0-4 score with timing."""
    if video_path is None:
        return "No video uploaded", "N/A", "N/A", {}

    import pandas as pd

    # 1) Upstream: pose extraction + 3D lift + A12 cut via ExercisePipeline.
    t_up_start = time.perf_counter()
    pipeline = ExercisePipeline(quality_threshold=quality_threshold)
    try:
        results = pipeline.process_video(video_path)
    finally:
        pipeline.close()
    t_upstream = (time.perf_counter() - t_up_start) * 1000.0

    if results is None or results.get("pipeline_stopped"):
        return (
            f"REJECTED — poor recording quality "
            f"(conf {results.get('recording_confidence', 0):.2f})"
            if results else "REJECTED — could not open video",
            "N/A",
            "N/A",
            results or {},
        )

    # 2) Load the cut 3D CSV produced by the pipeline.
    stem = Path(video_path).stem
    cut_csv = Path(__file__).parent / "outputs" / f"{stem}_cut_3d_points.csv"
    if not cut_csv.exists():
        return ("ERROR — cut 3D CSV not produced by pipeline", "N/A", "N/A", results)

    df = pd.read_csv(cut_csv)
    if len(df) < A15_C:
        return (
            f"REJECTED — too few frames after cut ({len(df)} < {A15_C})",
            "N/A", "N/A", results,
        )

    # 3) Adapter: sample, scale, predict (timed separately).
    model, scaler = _load_a15_scorer()
    t_sample_s = time.perf_counter()
    frames = _a15_sample_frames(df)
    flat = frames.reshape(1, -1)
    scaled = scaler.transform(flat).astype(np.float32)
    if len(model.input_shape) == 3:
        scaled = scaled.reshape(1, A15_C, len(A15_JOINTS) * 3)
    t_adapter = (time.perf_counter() - t_sample_s) * 1000.0

    t_nn_s = time.perf_counter()
    raw = float(model.predict(scaled, verbose=0).flatten()[0])
    t_nn = (time.perf_counter() - t_nn_s) * 1000.0

    score = float(np.clip(raw, 0.0, 4.0))
    band = _a15_score_band(score)

    t_total = t_upstream + t_adapter + t_nn
    timing_md = (
        f"**Score:** `{score:.2f} / 4`  \n"
        f"**Band:** {band}  \n"
        f"**Decision time (NN only):** {t_nn:.1f} ms  \n"
        f"**Adapter (sample + scale):** {t_adapter:.1f} ms  \n"
        f"**Upstream (pose + 3D lift + cut):** {t_upstream:.1f} ms  \n"
        f"**End-to-end total:** {t_total/1000:.2f} s  \n"
        f"**NN as % of total:** {(t_nn/t_total)*100:.2f} %"
    )

    results_with_score = dict(results)
    results_with_score["a15_score"] = round(score, 4)
    results_with_score["a15_band"] = band
    results_with_score["a15_timing_ms"] = {
        "nn_predict": round(t_nn, 2),
        "adapter":    round(t_adapter, 2),
        "upstream":   round(t_upstream, 2),
        "total":      round(t_total, 2),
    }
    return (band, f"{score:.2f} / 4", timing_md, results_with_score)
# --- end A15 ------------------------------------------------------------

# Initialize MoveNet pose estimator
pose_estimator = MoveNetPoseEstimator(model_name='lightning')

# COCO Keypoint definitions (17 keypoints)
KEYPOINT_NAMES = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]


def extract_joint_positions_from_movenet(pose_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract joint positions from MoveNet pose result."""
    keypoints = pose_result.get('keypoints', {})
    all_keypoints = []

    for joint_name in KEYPOINT_NAMES:
        kp = keypoints.get(joint_name, {})
        x = kp.get('x')
        y = kp.get('y')
        score = kp.get('confidence')

        all_keypoints.append({
            "x": x,
            "y": y,
            "score": score,
            "name": joint_name
        })

    return {
        "poses": [{
            "pose_id": 0,
            "total_score": 0.0,
            "total_parts": len([k for k in all_keypoints if k['x'] is not None]),
            "keypoints": all_keypoints
        }],
        "timestamp": datetime.now().isoformat(),
        "joint_names": KEYPOINT_NAMES,
        "inference_time_ms": pose_result.get('inference_time_ms', 0)
    }


def save_to_csv(joint_data: Dict[str, Any], filename: str = None) -> str:
    """Save joint positions to CSV file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pose_data_{timestamp}.csv"

    filepath = os.path.join("pose_outputs", filename)
    os.makedirs("pose_outputs", exist_ok=True)

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Pose_ID", "Joint", "X", "Y", "Confidence", "Visible"])

        poses = joint_data.get("poses", [])
        for pose in poses:
            pose_id = pose.get("pose_id", 0)
            for kp in pose.get("keypoints", []):
                x = kp.get("x")
                y = kp.get("y")
                score = kp.get("score")
                name = kp.get("name", "Unknown")

                visible = "Yes" if x is not None and y is not None else "No"

                writer.writerow([
                    pose_id,
                    name,
                    f"{x:.2f}" if x is not None else "N/A",
                    f"{y:.2f}" if y is not None else "N/A",
                    f"{score:.3f}" if score is not None else "N/A",
                    visible
                ])

        writer.writerow([])
        writer.writerow(["Timestamp", joint_data.get("timestamp", "")])
        writer.writerow(["Inference_Time_ms", joint_data.get("inference_time_ms", 0)])

    return filepath


def save_to_json(joint_data: Dict[str, Any], filename: str = None) -> str:
    """Save joint positions to JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pose_data_{timestamp}.json"

    filepath = os.path.join("pose_outputs", filename)
    os.makedirs("pose_outputs", exist_ok=True)

    with open(filepath, 'w') as jsonfile:
        json.dump(joint_data, jsonfile, indent=2)

    return filepath


def process_single_image(image: Image.Image, confidence_threshold: float = 0.3) -> tuple:
    """Process a single image and return annotated image with pose data."""
    img_array = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    pose_result = pose_estimator.detect_pose(img_bgr)
    joint_data = extract_joint_positions_from_movenet(pose_result)

    result_bgr = pose_estimator.draw_keypoints(img_bgr, pose_result, confidence_threshold=confidence_threshold)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    result_image = Image.fromarray(result_rgb)

    csv_path = save_to_csv(joint_data)
    json_path = save_to_json(joint_data)
    joint_data["csv_path"] = csv_path
    joint_data["json_path"] = json_path

    return result_image, joint_data


def process_video_frame(frame: np.ndarray, confidence_threshold: float = 0.3) -> np.ndarray:
    """Process a single video frame and return annotated frame."""
    # Handle frame format - OpenCV videos are BGR with 3 channels
    # If frame has 3 channels, assume BGR. If 4 channels, convert BGRA to BGR.
    # If grayscale (2D), convert to BGR.
    if len(frame.shape) == 3:
        if frame.shape[2] == 3:
            img_bgr = frame  # Already BGR
        elif frame.shape[2] == 4:
            img_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR
        else:
            img_bgr = frame  # Fallback
    else:
        img_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR

    pose_result = pose_estimator.detect_pose(img_bgr)
    annotated_bgr = pose_estimator.draw_keypoints(img_bgr, pose_result, confidence_threshold=confidence_threshold)

    return annotated_bgr


def format_pose_output(joint_data: Dict[str, Any]) -> str:
    """Format pose data for display in Gradio."""
    output = "### Detected Poses\n\n"
    output += f"**Timestamp:** {joint_data.get('timestamp', 'N/A')}\n"
    output += f"**Inference Time:** {joint_data.get('inference_time_ms', 0):.2f} ms\n\n"

    poses = joint_data.get("poses", [])
    if not poses:
        output += "No pose data available.\n\n"
    else:
        for pose in poses:
            output += f"#### Pose #{pose.get('pose_id', 0)}\n"
            output += f"- **Total Parts:** {pose.get('total_parts', 0)}\n\n"

            output += "| Joint | X | Y | Confidence | Visible |\n"
            output += "|-------|---|---|------------|---------|\n"

            for kp in pose.get("keypoints", []):
                name = kp.get("name", "Unknown")
                x = kp.get("x")
                y = kp.get("y")
                score = kp.get("score")

                x_str = f"{x:.1f}" if x is not None else "N/A"
                y_str = f"{y:.1f}" if y is not None else "N/A"
                score_str = f"{score:.3f}" if score is not None else "N/A"
                visible = "Yes" if x is not None and y is not None else "No"

                output += f"| {name} | {x_str} | {y_str} | {score_str} | {visible} |\n"

            output += "\n"

    output += f"**CSV File:** `{joint_data.get('csv_path', 'N/A')}`\n"
    output += f"**JSON File:** `{joint_data.get('json_path', 'N/A')}`\n"

    return output

def run_a14_pipeline(video_path, quality_threshold):
    if video_path is None:
        return None, "No video uploaded", "N/A", {}

    pipeline = ExercisePipeline(quality_threshold=quality_threshold)
    try:
        results = pipeline.process_video(video_path)
    finally:
        pipeline.close()

    # Handle UGLY case
    if results is None or results.get("pipeline_stopped"):
        return (
            None,
            f"REJECTED — Poor recording quality "
            f"(conf: {results.get('recording_confidence', 0):.2f})",
            "N/A",
            results or {}
        )

    # Handle SUCCESS case
    stem    = Path(video_path).stem

    pipeline_dir = Path(__file__).parent
    out_dir      = pipeline_dir / "outputs"
    video_3d_path = out_dir / f"{stem}_skeleton.mp4"

    video_3d = None
    if video_3d_path.exists():
        import shutil
        import tempfile
        tmp = tempfile.NamedTemporaryFile(
            suffix='.mp4', delete=False)
        shutil.copy(str(video_3d_path), tmp.name)
        video_3d = tmp.name
        print(f"  Copied to temp: {tmp.name}")

    status_text  = (f"ACCEPTED — Recording OK "
                    f"(conf: {results.get('recording_confidence', 0):.2f})")
    quality_text = (f"{results.get('quality_label', 'N/A')} "
                    f"({results.get('quality_confidence', 0):.1%})")

    return (
        video_3d,      # 1. a14_3d_output
        status_text,   # 2. a14_rec_status
        quality_text,  # 3. a14_exercise_quality
        results        # 4. a14_json_output
    )


def process_and_display(image: Image.Image, confidence_threshold: float = 0.3) -> tuple:
    """Process image and return pose output with data files."""
    result, joint_data = process_single_image(image, confidence_threshold)
    pose_info = format_pose_output(joint_data)
    return result, pose_info


def process_webcam_video(
    video_path: str,
    confidence_threshold: float = 0.3,
    smoothing_strategy: str = "exponential",
    smoothing_method: str = "zscore",
    progress=gr.Progress()
) -> tuple:
    """Process uploaded video with pose estimation."""
    if video_path is None:
        return None, "No video uploaded."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Could not open video."

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: FPS={fps}, Width={width}, Height={height}, TotalFrames={total_frames}")

    # Validate FPS - if it's extremely high or invalid, use a reasonable default
    if fps <= 0 or fps > 240:  # 240 FPS is unrealistically high for normal videos
        print(f"Invalid FPS ({fps}), using default 30 FPS")
        fps = 30
    else:
        print(f"Using FPS: {fps}")

    # Create output video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("pose_outputs", f"annotated_video_{timestamp}.mp4")
    os.makedirs("pose_outputs", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Verify video writer opened successfully
    if not out.isOpened():
        print(f"Error: Video writer failed to open. Output path: {output_path}")
        return None, "Failed to create output video. Please check the video format and try again."

    all_keypoints = []
    frame_count = 0

    progress(0, desc="Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Frame read failed at frame {frame_count}")
            break

        # Debug: Check frame properties
        print(f"Frame {frame_count}: shape={frame.shape if frame is not None else None}")

        # Process frame
        annotated_frame = process_video_frame(frame, confidence_threshold)

        # Verify frame dimensions match video writer
        if annotated_frame.shape[1] != width or annotated_frame.shape[0] != height:
            print(f"Resizing frame from {annotated_frame.shape[1]}x{annotated_frame.shape[0]} to {width}x{height}")
            annotated_frame = cv2.resize(annotated_frame, (width, height))

        out.write(annotated_frame)

        # Extract keypoints for this frame
        img_bgr = frame if frame.shape[2] == 3 else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        pose_result = pose_estimator.detect_pose(img_bgr)
        joint_data = extract_joint_positions_from_movenet(pose_result)
        joint_data['frame_id'] = frame_count
        joint_data['timestamp'] = frame_count / fps if fps > 0 else 0
        all_keypoints.append(joint_data)

        frame_count += 1

        # Update progress
        if frame_count % 30 == 0:
            progress(frame_count / total_frames if total_frames > 0 else 0, desc=f"Processing frame {frame_count}/{total_frames if total_frames > 0 else '?'}...")

    cap.release()
    out.release()

    print(f"Total frames processed: {frame_count}")

    # Apply smoothing to the keypoints
    try:
        smoothed_keypoints = smooth_pose_sequence(
            all_keypoints,
            strategy=smoothing_strategy,
            outlier_method=smoothing_method,
            outlier_threshold=3.0,
            window_size=7,
            min_confidence=0.2,
        )
    except Exception as e:
        print(f"Error applying smoothing: {e}")
        # Fallback to original keypoints if smoothing fails
        smoothed_keypoints = all_keypoints

    # Save smoothed keypoints to CSV
    csv_path = os.path.join("pose_outputs", f"video_keypoints_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Frame_ID", "Joint", "X", "Y", "Confidence", "Visible"])

        for frame_data in smoothed_keypoints:
            frame_id = frame_data.get('frame_id', 0)
            for kp in frame_data['poses'][0]['keypoints']:
                x = kp.get('x')
                y = kp.get('y')
                score = kp.get('score')
                name = kp.get('name', 'Unknown')

                visible = "Yes" if x is not None and y is not None else "No"
                writer.writerow([
                    frame_id,
                    name,
                    f"{x:.2f}" if x is not None else "N/A",
                    f"{y:.2f}" if y is not None else "N/A",
                    f"{score:.3f}" if score is not None else "N/A",
                    visible
                ])

    avg_inference = np.mean([k.get('inference_time_ms', 0) for k in all_keypoints]) if all_keypoints else 0

    result_text = f"""### Video Processing Complete

- **Frames processed:** {frame_count}
- **Average inference time:** {avg_inference:.2f} ms/frame
- **Output video:** `{output_path}`
- **Keypoints CSV:** `{csv_path}`
"""

    return output_path, result_text


# Gradio UI with Tabs
with gr.Blocks(title="MoveNet Pose Estimation") as demo:
    gr.Markdown("# 🏃 MoveNet Pose Estimation")
    gr.Markdown("Estimate human poses using Google's MoveNet model. Supports single images and video files.")

    with gr.Tabs():
        # Image Processing Tab
        with gr.TabItem("📸 Image Processing"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Upload Image")
                    image_input = gr.Image(type="pil", label="Input Image")
                    confidence_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.05,
                        label="Confidence Threshold"
                    )
                    process_btn = gr.Button("🚀 Process Image", variant="primary")

                with gr.Column():
                    gr.Markdown("### Results")
                    image_output = gr.Image(type="pil", label="Annotated Output")
                    pose_text = gr.Textbox(label="Pose Data", lines=15)

            process_btn.click(
                fn=process_and_display,
                inputs=[image_input, confidence_slider],
                outputs=[image_output, pose_text]
            )

        # Video Processing Tab
        with gr.TabItem("🎥 Video Processing"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Upload Video")
                    video_input = gr.Video(label="Input Video")
                    video_confidence = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.05,
                        label="Confidence Threshold"
                    )
                    smoothing_strategy = gr.Dropdown(
                        choices=["exponential", "moving_average", "gaussian", "median", "savitzky_golay", "kalman", "spline", "hybrid"],
                        value="exponential",
                        label="Smoothing Strategy"
                    )
                    smoothing_method = gr.Dropdown(
                        choices=["zscore", "velocity", "none"],
                        value="zscore",
                        label="Outlier Detection Method"
                    )
                    process_video_btn = gr.Button("🎬 Process Video", variant="primary")

                with gr.Column():
                    gr.Markdown("### Results")
                    video_output = gr.Video(label="Annotated Video")
                    video_result = gr.Textbox(label="Processing Results", lines=15)

            process_video_btn.click(
                fn=process_webcam_video,
                inputs=[video_input, video_confidence, smoothing_strategy, smoothing_method],
                outputs=[video_output, video_result]
            )

        # A12 Video Pipeline Tab
        with gr.TabItem("🧪 Video Pipeline"):
            gr.Markdown(
                """
                ### Issue #12: App development and pipeline integration

                Endpoint alternative chosen: **Gradio tab inside the existing app.py**.

                **Input:** one video file.
                **Output:** annotated cut 2D video, 3D skeleton animation video, keypoints CSV,
                and good/bad classification JSON.
                """
            )

            with gr.Row():
                with gr.Column():
                    a12_video_input = gr.Video(label="Input exercise video")
                    a12_confidence = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.05,
                        label="Confidence threshold"
                    )
                    a12_smoothing_strategy = gr.Dropdown(
                        choices=[
                            "exponential",
                            "moving_average",
                            "gaussian",
                            "median",
                            "savitzky_golay",
                            "kalman",
                            "spline",
                            "hybrid"
                        ],
                        value="exponential",
                        label="Smoothing strategy",
                    )
                    a12_smoothing_method = gr.Dropdown(
                        choices=["zscore", "velocity", "none"],
                        value="zscore",
                        label="Outlier detection method",
                    )
                    a12_run_btn = gr.Button("Run A12 pipeline", variant="primary")

                with gr.Column():
                    #a12_video_output = gr.Video(label="Annotated cut 2D video")
                    a12_animation_output = gr.Video(label="3D Skeleton Animation")
                    a12_keypoints_file = gr.File(label="3D joint CSV")
                    a12_json_output = gr.JSON(label="Structured output")
                    a12_summary = gr.Markdown()

            a12_run_btn.click(
                fn=run_a12_video_tab,
                inputs=[
                    a12_video_input,
                    a12_confidence,
                    a12_smoothing_strategy,
                    a12_smoothing_method
                ],
                outputs=[
                    a12_animation_output,
                    a12_keypoints_file,
                    a12_json_output,
                    a12_summary
                ],
            )

            # Exercise pipeline A14
        with gr.TabItem("Exercise Analysis (A14)"):
            gr.Markdown(
                """
                ## A14: Advanced Exercise Pipeline
                **Features:** Automated 'Ugly' recording rejection + 'Good/Bad' form classification.
                """
            )

            with gr.Row():
                with gr.Column():
                    a14_input_video = gr.Video(label="Upload Exercise Video")
                    a14_threshold = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.6, step=0.05,
                        label="Recording Quality Threshold"
                    )
                    a14_run_btn = gr.Button("Run Full Analysis", variant="primary")

                with gr.Column():
                    # High-visibility results
                    with gr.Row():
                        a14_rec_status = gr.Textbox(label="Recording Status", interactive=False)
                        a14_exercise_quality = gr.Label(label="Exercise quality")
                    
                    a14_3d_output = gr.Video(label="3D Skeleton Animation")
                    a14_json_output = gr.JSON(label="Full Metadata")

            # Link the button to the bridge function
            a14_run_btn.click(
                fn=run_a14_pipeline,
                inputs=[a14_input_video, a14_threshold],
                outputs=[
                    a14_3d_output, 
                    a14_rec_status, 
                    a14_exercise_quality, 
                    a14_json_output
                ]
            )

        # A15 Exercise Scoring tab — 0-4 regression score
        with gr.TabItem("Exercise Scoring (A15)"):
            gr.Markdown(
                """
                ## A15: Exercise Scoring (0–4 regression)

                **Score scale:** `0` = perfect form, `4` = worst kept clip.

                Bands:
                - **GREEN** `< 1` — acceptable form
                - **AMBER** `1–2` — borderline, consider another take
                - **RED**   `≥ 2` — poor form

                The same upstream pipeline as A14 is reused (pose extraction +
                3D lift + A12 start/stop cut). Decision-time of the NN and the
                overall response-time breakdown are reported alongside the score.
                """
            )

            with gr.Row():
                with gr.Column():
                    a15_input_video = gr.Video(label="Upload Exercise Video")
                    a15_threshold = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.6, step=0.05,
                        label="Recording Quality Threshold"
                    )
                    a15_run_btn = gr.Button("Run A15 scoring", variant="primary")

                with gr.Column():
                    a15_band = gr.Textbox(label="Band", interactive=False)
                    a15_score = gr.Textbox(label="Score (0–4)", interactive=False)
                    a15_timing = gr.Markdown(label="Timing breakdown")
                    a15_json = gr.JSON(label="Full results")

            a15_run_btn.click(
                fn=run_a15_scoring,
                inputs=[a15_input_video, a15_threshold],
                outputs=[a15_band, a15_score, a15_timing, a15_json],
            )

        # A16 Final unified endpoint (capstone)
        build_a16_tab(gr)


    # Example section
    with gr.Accordion("ℹ️ Information", open=False):
        gr.Markdown("""
        ### Features
        - **Single Image Processing**: Upload and process static images
        - **Video Processing**: Upload video files for pose estimation
        - **17 COCO Keypoints**: Detects nose, eyes, ears, shoulders, elbows, wrists, hips, knees, and ankles
        - **Confidence Threshold**: Adjust detection sensitivity
        - **CSV/JSON Export**: Download pose data for further analysis

        ### Model Details
        - Model: MoveNet SinglePose (Lightning)
        - Input size: 192x192 pixels
        - Fast and efficient real-time pose estimation
        """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
