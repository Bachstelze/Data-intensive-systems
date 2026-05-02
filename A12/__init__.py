#!/usr/bin/env python3
"""
A12 – Pose Interpolation & Smoothing
=====================================
Outlier-robust smoothing strategies for pose-estimation keypoint sequences.

Provides
--------
- :class:`PoseInterpolator` – low‑level pipeline with multiple strategies
- :func:`smooth_pose_sequence` – high‑level convenience for *app.py* data
- :class:`SmoothingStrategy` – enumeration of available methods
- :class:`KalmanFilter1D` – reusable 1‑D constant‑velocity Kalman filter
- Detection helpers: :func:`detect_outliers_velocity`,
  :func:`detect_outliers_zscore`

Supported strategies
--------------------
=========================== ===================================================
Strategy                    Description
=========================== ===================================================
``moving_average``          Sliding-window box-car average
``gaussian``                Gaussian-weighted convolution
``exponential``             Exponential moving average (EMA)
``median``                  Median filter – kills isolated spikes
``savitzky_golay``          Savitzky-Golay – preserves signal shape
``kalman``                  1‑D constant-velocity Kalman filter
``spline``                  Cubic-spline interpolation through
                            high-confidence points
``hybrid``                  (Default) outlier → interpolate → Savitzky‑Golay
=========================== ===================================================

Usage examples
--------------

.. code:: python

    from A12 import smooth_pose_sequence, PoseInterpolator

    # Quick hybrid smoothing (recommended for animation)
    smoothed = smooth_pose_sequence(all_keypoints)

    # Fine-grained control
    interp = PoseInterpolator(strategy="kalman", process_noise=0.001,
                              measurement_noise=0.05)
    arr = interp.keypoints_to_array(all_keypoints)
    smoothed_arr = interp.fit_transform(arr)
    smoothed_frames = interp.array_to_keypoints(smoothed_arr, all_keypoints)
"""

from A12.pose_interpolator import (  # noqa: F401
    # High-level API
    smooth_pose_sequence,
    # Low-level API
    PoseInterpolator,
    SmoothingStrategy,
    KalmanFilter1D,
    # Outlier detection utilities
    detect_outliers_velocity,
    detect_outliers_zscore,
    # Constants
    COCO_KEYPOINTS,
    A11_JOINTS,
)
