"""
A14 Package - MediaPipe Pose Estimator

This package provides a MediaPipe-based human pose estimator that detects
2D landmarks with visibility as confidence, similar to the MoveNetPoseEstimator.
"""

from .mediapipe_pose_estimator import MediaPipePoseEstimator

__all__ = ['MediaPipePoseEstimator']
__version__ = '1.0.0'
