from PIL import Image
import gradio as gr
from A8.pose_estimator import MoveNetPoseEstimator
import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import cv2
import tempfile
import time

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


def process_and_display(image: Image.Image, confidence_threshold: float = 0.3) -> tuple:
    """Process image and return pose output with data files."""
    result, joint_data = process_single_image(image, confidence_threshold)
    pose_info = format_pose_output(joint_data)
    return result, pose_info


def process_webcam_video(
    video_path: str,
    confidence_threshold: float = 0.3,
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

    # Save keypoints to CSV
    csv_path = os.path.join("pose_outputs", f"video_keypoints_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Frame_ID", "Joint", "X", "Y", "Confidence", "Visible"])

        for frame_data in all_keypoints:
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
                    process_video_btn = gr.Button("🎬 Process Video", variant="primary")

                with gr.Column():
                    gr.Markdown("### Results")
                    video_output = gr.Video(label="Annotated Video")
                    video_result = gr.Textbox(label="Processing Results", lines=15)

            process_video_btn.click(
                fn=process_webcam_video,
                inputs=[video_input, video_confidence],
                outputs=[video_output, video_result]
            )

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
