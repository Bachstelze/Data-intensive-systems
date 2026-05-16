"""
MediaPipe Pose Estimator Module
===============================
A Python module for human pose estimation using Google's MediaPipe Pose Landmarker model.

This module provides functionality to:
- Load and run MediaPipe pose estimation model
- Process images and videos
- Extract 33 pose landmarks
- Visualize pose detection results

Uses the lightweight model from https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Pose Landmark definitions (33 keypoints)
KEYPOINT_NAMES = [
    'nose',
    'left_eye_inner',
    'left_eye',
    'left_eye_outer',
    'right_eye_inner',
    'right_eye',
    'right_eye_outer',
    'left_ear',
    'right_ear',
    'mouth_left',
    'mouth_right',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_pinky',
    'right_pinky',
    'left_index',
    'right_index',
    'left_thumb',
    'right_thumb',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
    'left_heel',
    'right_heel',
    'left_foot_index',
    'right_foot_index'
]

# Skeleton connections for visualization (based on MediaPipe pose connections)
KEYPOINT_EDGES = [
    # Face connections
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),  # Nose to eyes and ears
    # Upper body
    (11, 12),  # Shoulders
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # Left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # Right arm
    (11, 23), (12, 24),  # Shoulder to hip
    # Lower body
    (23, 24),  # Hips
    (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
    (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
]


class MediaPipePoseEstimator:
    """
    MediaPipe-based human pose estimator using Pose Landmarker.

    Uses the lightweight model for real-time performance.

    Example usage:
        estimator = MediaPipePoseEstimator(model_asset_path='pose_landmarker_lite.task')
        keypoints = estimator.detect_pose(image)
        visualized = estimator.draw_keypoints(image, keypoints)
    """

    # MediaPipe model asset name for the lite version
    MODEL_NAME = 'pose_landmarker_lite.task'

    def __init__(self, model_asset_path: Optional[str] = None, min_detection_confidence: float = 0.5):
        """
        Initialize the MediaPipe pose estimator.

        Args:
            model_asset_path: Path to the .task model file (defaults to lite model)
            min_detection_confidence: Minimum confidence for pose detection
        """
        # Use the lite model if no path is provided
        if model_asset_path is None:
            # We'll need to download the model separately, for now use a placeholder
            self.model_asset_path = self.MODEL_NAME
        else:
            self.model_asset_path = model_asset_path

        self.min_detection_confidence = min_detection_confidence

        # Create PoseLandmarker options
        base_options = python.BaseOptions(model_asset_path=self.model_asset_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=min_detection_confidence
        )

        print(f"Loading MediaPipe Pose Landmarker model: {self.model_asset_path}...")
        try:
            self.detector = vision.PoseLandmarker.create_from_options(options)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Make sure you have the pose_landmarker_lite.task model file.")
            raise

    def detect_pose(self, image: np.ndarray) -> Dict:
        """
        Detect pose keypoints in an image.

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            Dictionary with keypoint data:
            {
                'keypoints': {
                    'nose': {'x': float, 'y': float, 'confidence': float},
                    ...
                },
                'inference_time_ms': float
            }
        """
        start_time = time.time()

        # Convert BGR to RGB as MediaPipe expects RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run inference
        detection_result = self.detector.detect(mp_image)

        inference_time = (time.time() - start_time) * 1000

        # Parse keypoints if detections found
        keypoints_dict = {}
        if detection_result.pose_landmarks:
            # Use the first detected pose (MediaPipe can detect multiple poses)
            landmarks = detection_result.pose_landmarks[0]

            for i, landmark in enumerate(landmarks):
                if i < len(KEYPOINT_NAMES):
                    keypoints_dict[KEYPOINT_NAMES[i]] = {
                        'x': float(landmark.x),
                        'y': float(landmark.y),
                        'confidence': float(landmark.visibility)  # Using visibility as confidence
                    }
                else:
                    # Handle extra landmarks if any
                    keypoints_dict[f'landmark_{i}'] = {
                        'x': float(landmark.x),
                        'y': float(landmark.y),
                        'confidence': float(landmark.visibility)
                    }
        else:
            # No pose detected, return empty keypoints
            for name in KEYPOINT_NAMES:
                keypoints_dict[name] = {
                    'x': 0.0,
                    'y': 0.0,
                    'confidence': 0.0
                }

        return {
            'keypoints': keypoints_dict,
            'inference_time_ms': inference_time
        }

    def detect_pose_raw(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect pose and return raw keypoints array.

        Args:
            image: Input image (BGR format)

        Returns:
            Array of shape (33, 3) with [x, y, confidence] for each keypoint, or None if no pose detected
        """
        # Convert BGR to RGB as MediaPipe expects RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run inference
        detection_result = self.detector.detect(mp_image)

        if detection_result.pose_landmarks:
            # Use the first detected pose
            landmarks = detection_result.pose_landmarks[0]
            keypoints_array = np.zeros((len(landmarks), 3))  # x, y, confidence

            for i, landmark in enumerate(landmarks):
                keypoints_array[i] = [landmark.x, landmark.y, landmark.visibility]

            return keypoints_array

        return None

    def draw_keypoints(
        self,
        image: np.ndarray,
        keypoints: Dict,
        confidence_threshold: float = 0.3,
        circle_radius: int = 5,
        line_thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detected keypoints and skeleton on image.

        Args:
            image: Input image (will be copied, not modified)
            keypoints: Keypoint dictionary from detect_pose()
            confidence_threshold: Minimum confidence to draw keypoint
            circle_radius: Radius of keypoint circles
            line_thickness: Thickness of skeleton lines

        Returns:
            Image with keypoints and skeleton drawn
        """
        output_image = image.copy()
        height, width = image.shape[:2]

        kps = keypoints['keypoints']

        # Draw skeleton connections
        for edge in KEYPOINT_EDGES:
            start_idx, end_idx = edge
            if start_idx < len(KEYPOINT_NAMES) and end_idx < len(KEYPOINT_NAMES):
                start_name = KEYPOINT_NAMES[start_idx]
                end_name = KEYPOINT_NAMES[end_idx]

                start_kp = kps[start_name]
                end_kp = kps[end_name]

                if start_kp['confidence'] > confidence_threshold and end_kp['confidence'] > confidence_threshold:
                    start_point = (int(start_kp['x'] * width), int(start_kp['y'] * height))
                    end_point = (int(end_kp['x'] * width), int(end_kp['y'] * height))

                    # Draw line in green
                    cv2.line(output_image, start_point, end_point, (0, 255, 0), line_thickness)

        # Draw keypoints
        for name, kp in kps.items():
            if kp['confidence'] > confidence_threshold:
                x = int(kp['x'] * width)
                y = int(kp['y'] * height)
                cv2.circle(output_image, (x, y), circle_radius, (0, 255, 255), -1)  # Yellow filled circle
                cv2.circle(output_image, (x, y), circle_radius + 1, (0, 0, 255), 1)  # Red outline

        return output_image

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show_preview: bool = False,
        confidence_threshold: float = 0.3
    ) -> List[Dict]:
        """
        Process a video file and extract keypoints from each frame.

        Args:
            video_path: Path to input video file
            output_path: Optional path to save annotated video
            show_preview: Whether to show live preview (press 'q' to quit)
            confidence_threshold: Minimum confidence for visualization

        Returns:
            List of keypoint dictionaries, one per frame
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")

        # Setup video writer if output path specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_keypoints = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect pose
            result = self.detect_pose(frame)
            result['frame_id'] = frame_idx
            result['timestamp'] = frame_idx / fps if fps > 0 else 0
            all_keypoints.append(result)

            # Draw and optionally show/save
            annotated_frame = self.draw_keypoints(frame, result, confidence_threshold)

            if writer:
                writer.write(annotated_frame)

            if show_preview:
                cv2.imshow('Pose Estimation', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames...")

        cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()

        print(f"Completed! Processed {frame_idx} frames.")
        avg_inference = np.mean([r['inference_time_ms'] for r in all_keypoints])
        print(f"Average inference time: {avg_inference:.2f} ms/frame")

        return all_keypoints

    def process_image_file(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        confidence_threshold: float = 0.3
    ) -> Dict:
        """
        Process a single image file.

        Args:
            image_path: Path to input image
            output_path: Optional path to save annotated image
            confidence_threshold: Minimum confidence for visualization

        Returns:
            Keypoint dictionary for the image
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        result = self.detect_pose(image)

        if output_path:
            annotated = self.draw_keypoints(image, result, confidence_threshold)
            cv2.imwrite(output_path, annotated)
            print(f"Saved annotated image to: {output_path}")

        return result


def main():
    """Demo: Test the pose estimator on a sample image or webcam."""
    import argparse

    parser = argparse.ArgumentParser(description='MediaPipe Pose Estimation Demo')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the .task model file (default: pose_landmarker_lite.task)')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    parser.add_argument('--output', type=str, help='Output path for annotated image/video')
    parser.add_argument('--conf_thresh', type=float, default=0.3, help='Confidence threshold (default: 0.3)')
    args = parser.parse_args()

    # Initialize estimator
    try:
        estimator = MediaPipePoseEstimator(model_asset_path=args.model_path)
    except Exception as e:
        print(f"Error initializing estimator: {e}")
        print("Make sure you have downloaded the pose_landmarker_lite.task model file")
        print("from https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker")
        return

    if args.image:
        # Process image
        print(f"\nProcessing image: {args.image}")
        result = estimator.process_image_file(
            args.image,
            output_path=args.output,
            confidence_threshold=args.conf_thresh
        )
        print(f"Inference time: {result['inference_time_ms']:.2f} ms")
        print("\nDetected keypoints:")
        for name, kp in result['keypoints'].items():
            if kp['confidence'] > args.conf_thresh:
                print(f"  {name}: ({kp['x']:.3f}, {kp['y']:.3f}) conf={kp['confidence']:.3f}")

    elif args.video:
        # Process video
        print(f"\nProcessing video: {args.video}")
        keypoints = estimator.process_video(
            args.video,
            output_path=args.output,
            show_preview=True,
            confidence_threshold=args.conf_thresh
        )
        print(f"\nExtracted keypoints from {len(keypoints)} frames")

    elif args.webcam:
        # Webcam demo
        print("\nStarting webcam demo (press 'q' to quit)...")
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = estimator.detect_pose(frame)
            annotated = estimator.draw_keypoints(frame, result, args.conf_thresh)

            # Add FPS display
            fps_text = f"Inference: {result['inference_time_ms']:.1f} ms"
            cv2.putText(annotated, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('MediaPipe Pose Estimation', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        print("Please specify --image, --video, or --webcam")
        print("Example: python mediapipe_pose_estimator.py --image test.jpg --output result.jpg")


if __name__ == '__main__':
    main()
