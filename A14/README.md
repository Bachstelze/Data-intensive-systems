# MediaPipe Pose Estimator

This module implements a human pose estimator using Google's MediaPipe Pose Landmarker model. It provides functionality similar to the MoveNetPoseEstimator but uses MediaPipe's lightweight model for pose estimation.

## Features

- Extracts 33 pose landmarks from images and videos
- Real-time processing capabilities
- Visualization of detected poses
- Support for image, video, and webcam input
- Confidence values based on landmark visibility

## Installation

Before using this module, install the required dependencies:

```bash
pip install opencv-python numpy mediapipe
```

## Model Download

The MediaPipe Pose Landmarker model needs to be downloaded separately. Run the download script to get the lightweight model:

```bash
python download_model.py
```

Alternatively, you can manually download the `pose_landmarker_lite.task` model file from [Google's MediaPipe Pose Landmarker page](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker).

## Usage

### Basic Usage

```python
from mediapipe_pose_estimator import MediaPipePoseEstimator

# Initialize the estimator with the model file
estimator = MediaPipePoseEstimator(model_asset_path='pose_landmarker_lite.task')

# Process an image
image = cv2.imread('path_to_image.jpg')
result = estimator.detect_pose(image)

# Access the keypoints
keypoints = result['keypoints']
for name, kp in keypoints.items():
    print(f"{name}: ({kp['x']:.3f}, {kp['y']:.3f}) conf={kp['confidence']:.3f}")
```

### Processing Images

```python
# Process a single image file
result = estimator.process_image_file(
    image_path='input.jpg',
    output_path='output.jpg',
    confidence_threshold=0.3
)
```

### Processing Videos

```python
# Process a video file
keypoints_list = estimator.process_video(
    video_path='input.mp4',
    output_path='output.mp4',
    show_preview=True,
    confidence_threshold=0.3
)
```

### Command Line Usage

```bash
# Process an image
python mediapipe_pose_estimator.py --image input.jpg --output output.jpg

# Process a video
python mediapipe_pose_estimator.py --video input.mp4 --output output.mp4

# Use webcam
python mediapipe_pose_estimator.py --webcam
```

## Key Differences from MoveNetPoseEstimator

- Uses MediaPipe's Pose Landmarker model instead of MoveNet
- Extracts 33 landmarks instead of 17 COCO keypoints
- Confidence values are based on landmark visibility from MediaPipe
- Different skeleton connection patterns
- Potentially different performance characteristics

## Key Points

The MediaPipe Pose Landmarker detects the following 33 pose landmarks:

- nose
- left_eye_inner, left_eye, left_eye_outer
- right_eye_inner, right_eye, right_eye_outer
- left_ear, right_ear
- mouth_left, mouth_right
- left_shoulder, right_shoulder
- left_elbow, right_elbow
- left_wrist, right_wrist
- left_pinky, right_pinky
- left_index, right_index
- left_thumb, right_thumb
- left_hip, right_hip
- left_knee, right_knee
- left_ankle, right_ankle
- left_heel, right_heel
- left_foot_index, right_foot_index