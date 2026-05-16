"""
Script to download the MediaPipe Pose Landmarker model file.

Downloads the lightweight model from Google's AI Edge solutions.
"""

import urllib.request
import os

def download_model():
    """Download the MediaPipe pose landmarker lite model."""
    model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    model_filename = "pose_landmarker_lite.task"

    print(f"Downloading {model_url}...")

    try:
        urllib.request.urlretrieve(model_url, model_filename)
        print(f"Successfully downloaded {model_filename}")
        print(f"File size: {os.path.getsize(model_filename)} bytes")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please visit https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker to download the model manually")

if __name__ == "__main__":
    download_model()
