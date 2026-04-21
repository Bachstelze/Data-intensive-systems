"""
MoveNet Pose Estimator Module
=============================
A Python module for human pose estimation using TensorFlow's MoveNet model.

This module provides functionality to:
- Load and run MoveNet pose estimation model
- Process images and videos
- Extract 17 COCO keypoints
- Visualize pose detection results

Issue #33 - A8: PoseNet/MoveNet Python Environment Setup
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


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

# Skeleton connections for visualization
KEYPOINT_EDGES = {
    (0, 1): 'face',
    (0, 2): 'face',
    (1, 3): 'face',
    (2, 4): 'face',
    (0, 5): 'torso',
    (0, 6): 'torso',
    (5, 7): 'left_arm',
    (7, 9): 'left_arm',
    (6, 8): 'right_arm',
    (8, 10): 'right_arm',
    (5, 6): 'torso',
    (5, 11): 'torso',
    (6, 12): 'torso',
    (11, 12): 'torso',
    (11, 13): 'left_leg',
    (13, 15): 'left_leg',
    (12, 14): 'right_leg',
    (14, 16): 'right_leg',
}

# Colors for different body parts (BGR format for OpenCV)
EDGE_COLORS = {
    'face': (255, 255, 0),      # Cyan
    'torso': (0, 255, 0),       # Green
    'left_arm': (255, 0, 0),    # Blue
    'right_arm': (0, 0, 255),   # Red
    'left_leg': (255, 165, 0),  # Orange
    'right_leg': (128, 0, 128), # Purple
}


class MoveNetPoseEstimator:
    """
    MoveNet-based human pose estimator.
    
    Supports two model variants:
    - 'lightning': Faster, lower accuracy (default)
    - 'thunder': Slower, higher accuracy
    
    Example usage:
        estimator = MoveNetPoseEstimator(model_name='lightning')
        keypoints = estimator.detect_pose(image)
        visualized = estimator.draw_keypoints(image, keypoints)
    """
    
    # TensorFlow Hub model URLs
    MODEL_URLS = {
        'lightning': 'https://tfhub.dev/google/movenet/singlepose/lightning/4',
        'thunder': 'https://tfhub.dev/google/movenet/singlepose/thunder/4',
    }
    
    # Input sizes for each model
    INPUT_SIZES = {
        'lightning': 192,
        'thunder': 256,
    }
    
    def __init__(self, model_name: str = 'lightning'):
        """
        Initialize the MoveNet pose estimator.
        
        Args:
            model_name: Model variant ('lightning' or 'thunder')
        """
        if model_name not in self.MODEL_URLS:
            raise ValueError(f"Model must be one of: {list(self.MODEL_URLS.keys())}")
        
        self.model_name = model_name
        self.input_size = self.INPUT_SIZES[model_name]
        
        print(f"Loading MoveNet {model_name} model...")
        self.model = hub.load(self.MODEL_URLS[model_name])
        self.movenet = self.model.signatures['serving_default']
        print(f"Model loaded successfully. Input size: {self.input_size}x{self.input_size}")
    
    def preprocess_image(self, image: np.ndarray) -> tf.Tensor:
        """
        Preprocess image for MoveNet inference.
        
        Args:
            image: Input image (BGR or RGB format, any size)
            
        Returns:
            Preprocessed tensor ready for inference
        """
        # Convert BGR to RGB if needed (OpenCV loads as BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize to model input size
        input_image = tf.image.resize_with_pad(
            tf.expand_dims(image_rgb, axis=0),
            self.input_size,
            self.input_size
        )
        
        # Convert to int32 as required by MoveNet
        input_image = tf.cast(input_image, dtype=tf.int32)
        
        return input_image
    
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
        
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        outputs = self.movenet(input_tensor)
        keypoints_with_scores = outputs['output_0'].numpy()[0, 0, :, :]
        
        inference_time = (time.time() - start_time) * 1000
        
        # Parse keypoints
        keypoints_dict = {}
        for i, name in enumerate(KEYPOINT_NAMES):
            y, x, confidence = keypoints_with_scores[i]
            keypoints_dict[name] = {
                'x': float(x),
                'y': float(y),
                'confidence': float(confidence)
            }
        
        return {
            'keypoints': keypoints_dict,
            'inference_time_ms': inference_time
        }
    
    def detect_pose_raw(self, image: np.ndarray) -> np.ndarray:
        """
        Detect pose and return raw keypoints array.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Array of shape (17, 3) with [y, x, confidence] for each keypoint
        """
        input_tensor = self.preprocess_image(image)
        outputs = self.movenet(input_tensor)
        return outputs['output_0'].numpy()[0, 0, :, :]
    
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
        
        # Draw skeleton edges first (so keypoints appear on top)
        for (start_idx, end_idx), body_part in KEYPOINT_EDGES.items():
            start_name = KEYPOINT_NAMES[start_idx]
            end_name = KEYPOINT_NAMES[end_idx]
            
            start_kp = kps[start_name]
            end_kp = kps[end_name]
            
            if start_kp['confidence'] > confidence_threshold and end_kp['confidence'] > confidence_threshold:
                start_point = (int(start_kp['x'] * width), int(start_kp['y'] * height))
                end_point = (int(end_kp['x'] * width), int(end_kp['y'] * height))
                color = EDGE_COLORS[body_part]
                cv2.line(output_image, start_point, end_point, color, line_thickness)
        
        # Draw keypoints
        for name, kp in kps.items():
            if kp['confidence'] > confidence_threshold:
                x = int(kp['x'] * width)
                y = int(kp['y'] * height)
                cv2.circle(output_image, (x, y), circle_radius, (0, 255, 255), -1)
                cv2.circle(output_image, (x, y), circle_radius, (0, 0, 0), 1)
        
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
    
    parser = argparse.ArgumentParser(description='MoveNet Pose Estimation Demo')
    parser.add_argument('--model', choices=['lightning', 'thunder'], default='lightning',
                        help='Model variant (default: lightning)')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    parser.add_argument('--output', type=str, help='Output path for annotated image/video')
    args = parser.parse_args()
    
    # Initialize estimator
    estimator = MoveNetPoseEstimator(model_name=args.model)
    
    if args.image:
        # Process image
        print(f"\nProcessing image: {args.image}")
        result = estimator.process_image_file(
            args.image,
            output_path=args.output
        )
        print(f"Inference time: {result['inference_time_ms']:.2f} ms")
        print("\nDetected keypoints:")
        for name, kp in result['keypoints'].items():
            if kp['confidence'] > 0.3:
                print(f"  {name}: ({kp['x']:.3f}, {kp['y']:.3f}) conf={kp['confidence']:.3f}")
    
    elif args.video:
        # Process video
        print(f"\nProcessing video: {args.video}")
        keypoints = estimator.process_video(
            args.video,
            output_path=args.output,
            show_preview=True
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
            annotated = estimator.draw_keypoints(frame, result)
            
            # Add FPS display
            fps_text = f"Inference: {result['inference_time_ms']:.1f} ms"
            cv2.putText(annotated, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('MoveNet Pose Estimation', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    else:
        print("Please specify --image, --video, or --webcam")
        print("Example: python pose_estimator.py --image test.jpg --output result.jpg")


if __name__ == '__main__':
    main()
