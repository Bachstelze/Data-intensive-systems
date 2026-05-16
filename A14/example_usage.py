"""
Example usage of MediaPipePoseEstimator.

This script demonstrates how to use the MediaPipePoseEstimator to detect
pose landmarks in an image and visualize the results.
"""

import cv2
import numpy as np
from mediapipe_pose_estimator import MediaPipePoseEstimator

def main():
    # Initialize the estimator with the model file
    # Note: You need to have the pose_landmarker_lite.task file in the current directory
    try:
        estimator = MediaPipePoseEstimator(model_asset_path='pose_landmarker_lite.task')
    except Exception as e:
        print(f"Error initializing estimator: {e}")
        print("Make sure you have downloaded the pose_landmarker_lite.task model file")
        print("Run: python download_model.py")
        return

    # Create a sample image or load an existing one
    # For this example, we'll create a blank image
    sample_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add a simple stick figure for testing (optional)
    # Head
    cv2.circle(sample_image, (320, 100), 30, (255, 255, 255), -1)
    # Body
    cv2.line(sample_image, (320, 130), (320, 250), (255, 255, 255), 2)
    # Arms
    cv2.line(sample_image, (270, 180), (370, 180), (255, 255, 255), 2)
    # Legs
    cv2.line(sample_image, (320, 250), (280, 320), (255, 255, 255), 2)
    cv2.line(sample_image, (320, 250), (360, 320), (255, 255, 255), 2)

    print("Detecting pose in sample image...")

    # Detect pose
    result = estimator.detect_pose(sample_image)

    print(f"Inference time: {result['inference_time_ms']:.2f} ms")
    print(f"Number of keypoints: {len(result['keypoints'])}")

    # Print detected keypoints with confidence > 0.1
    print("\nDetected keypoints (confidence > 0.1):")
    for name, kp in result['keypoints'].items():
        if kp['confidence'] > 0.1:
            print(f"  {name}: ({kp['x']:.3f}, {kp['y']:.3f}) conf={kp['confidence']:.3f}")

    # Draw keypoints on the image
    annotated_image = estimator.draw_keypoints(sample_image, result)

    # Display the result
    cv2.imshow('Original Image', sample_image)
    cv2.imshow('Annotated Image', annotated_image)

    print("\nPress any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the annotated image
    cv2.imwrite('annotated_sample.jpg', annotated_image)
    print("Annotated image saved as 'annotated_sample.jpg'")

if __name__ == '__main__':
    main()
