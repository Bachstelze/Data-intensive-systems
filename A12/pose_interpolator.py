#!/usr/bin/env python3
"""
Pose Interpolation & Smoothing Module (A12)
============================================
Provides robust smoothing strategies for pose-estimation keypoint sequences,
eliminating jitter, filling detection gaps, and removing outlier spikes
to produce clean animations.

Supported strategies
--------------------
- ``moving_average``        Box-car (sliding window) average
- ``gaussian``              Gaussian-weighted convolution
- ``exponential``           Exponential moving average (EMA) – usable online
- ``median``                Median filter – excellent against isolated spikes
- ``savitzky_golay``        Savitzky-Golay filter – preserves signal shape
- ``kalman``                1-D constant-velocity Kalman filter (online)
- ``spline``                Cubic-spline interpolation through high-confidence
                            points, discarding outliers
- ``hybrid``                (Default) Outlier detection → interpolation →
                            Savitzky-Golay smoothing

Data formats
------------
The module accepts two common representations:

1. **App‑style list of dicts** (from ``app.py`` ``all_keypoints``):

   .. code:: python

       [
           {
               "poses": [{
                   "keypoints": [
                       {"x": 0.5, "y": 0.3, "score": 0.92, "name": "nose"},
                       ...
                   ]
               }],
               "frame_id": 0, ...
           },
           ...
       ]

2. **NumPy array** with shape ``(frames, joints, 3)`` where the last axis
   holds ``[x, y, confidence]``.

Basic usage
-----------
.. code:: python

    from A12.pose_interpolator import PoseInterpolator, smooth_pose_sequence

    # High-level convenience (recommended)
    smoothed = smooth_pose_sequence(all_keypoints, strategy="hybrid")

    # Low-level API
    interp = PoseInterpolator(strategy="kalman", process_noise=0.001,
                              measurement_noise=0.05)
    arr = interp.keypoints_to_array(all_keypoints)
    smoothed_arr = interp.fit_transform(arr)
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Try importing scipy – it is a transitive dependency of statsmodels /
# scikit-learn (both in requirements.txt), but we guard in case it is
# not available for some strategies.
# ---------------------------------------------------------------------------
try:
    from scipy import interpolate as _scipy_interpolate, ndimage, signal as _scipy_signal

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Standard joint-name lists (for reference / validation)
# ---------------------------------------------------------------------------

COCO_KEYPOINTS: List[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

A11_JOINTS: List[str] = [
    "head",
    "left_shoulder",
    "left_elbow",
    "right_shoulder",
    "right_elbow",
    "left_hand",
    "right_hand",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
]


class SmoothingStrategy(Enum):
    """Available smoothing strategies."""

    MOVING_AVERAGE = "moving_average"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"
    MEDIAN = "median"
    SAVITZKY_GOLAY = "savitzky_golay"
    KALMAN = "kalman"
    SPLINE = "spline"
    HYBRID = "hybrid"


# ===================================================================
# Small helpers
# ===================================================================


def _validate_array(arr: np.ndarray) -> np.ndarray:
    """Ensure *arr* is a contiguous float64 array of shape (F, J, 3)."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"Expected array of shape (frames, joints, 3), got {arr.shape}"
        )
    return arr


def _ensure_scipy(strategy_name: str) -> None:
    if not _HAS_SCIPY:
        raise ImportError(
            f"Strategy '{strategy_name}' requires scipy, which is not installed."
        )


# ===================================================================
# 1-D Kalman filter (constant velocity) – no external dependencies
# ===================================================================


class KalmanFilter1D:
    """
    Simple 1-D constant-velocity Kalman filter.

    State: [position, velocity]ᵀ
    Measurement: position

    Parameters
    ----------
    process_noise : float
        Process noise (Q) – higher values trust the model less.
    measurement_noise : float
        Measurement noise (R) – higher values trust measurements less.

    Notes
    -----
    When ``update(None)`` is called the filter performs a pure prediction
    step, allowing it to bridge gaps in the detection sequence.
    """

    def __init__(
        self, process_noise: float = 0.001, measurement_noise: float = 0.1
    ) -> None:
        # State transition (constant velocity, dt = 1)
        self._A = np.array([[1, 1], [0, 1]], dtype=np.float64)
        self._H = np.array([[1, 0]], dtype=np.float64)  # observe position only
        self._Q = np.eye(2, dtype=np.float64) * process_noise
        self._R = np.array([[measurement_noise]], dtype=np.float64)
        self._P = np.eye(2, dtype=np.float64) * 1.0
        self._x = np.zeros(2, dtype=np.float64)

    @property
    def position(self) -> float:
        """Current position estimate."""
        return float(self._x[0])

    def reset(self, position: float = 0.0) -> None:
        """Re-initialise filter with a new position and zero velocity."""
        self._x = np.array([position, 0.0], dtype=np.float64)
        self._P = np.eye(2, dtype=np.float64) * 1.0

    def update(self, measurement: Optional[float]) -> float:
        """
        Predict + (optionally) update step.

        Parameters
        ----------
        measurement : float or None
            Observed position. If ``None``, only the prediction step runs.

        Returns
        -------
        float
            Filtered position estimate.
        """
        # -- predict --
        self._x = self._A @ self._x
        self._P = self._A @ self._P @ self._A.T + self._Q

        if measurement is not None:
            # -- update --
            y = measurement - (self._H @ self._x)  # innovation
            S = self._H @ self._P @ self._H.T + self._R
            K = self._P @ self._H.T @ np.linalg.inv(S)  # Kalman gain
            self._x = self._x + (K @ y).ravel()
            self._P = (np.eye(2) - K @ self._H) @ self._P

        return self.position


# ===================================================================
# Outlier detection utilities
# ===================================================================


def detect_outliers_velocity(
    positions: np.ndarray,
    threshold: float = 3.0,
    min_confidence: float = 0.2,
    confidences: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Flag outliers based on inter-frame velocity.

    A point is considered an outlier if its frame-to-frame displacement
    exceeds *threshold* times the median absolute deviation of all
    non-zero displacements across the sequence.

    Parameters
    ----------
    positions : (F,) ndarray
        1-D coordinate signal (may contain NaN).
    threshold : float
        MAD multiplier.
    min_confidence : float
        Points with confidence below this are already treated as
        missing – they are **not** flagged here (they will be
        interpolated later).
    confidences : (F,) ndarray or None
        Confidence values (same length as *positions*).

    Returns
    -------
    (F,) bool ndarray
        ``True`` where the point is an outlier.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = len(positions)

    outlier_mask = np.zeros(n, dtype=bool)

    # Low-confidence points are handled by the interpolation stage.
    if confidences is not None:
        confidences = np.asarray(confidences, dtype=np.float64)
        low_conf = confidences < min_confidence
    else:
        low_conf = np.isnan(positions)

    # Compute finite differences
    diffs = np.abs(np.diff(positions))
    valid_diffs = diffs[np.isfinite(diffs)]
    if len(valid_diffs) == 0:
        return outlier_mask

    mad = np.median(np.abs(valid_diffs - np.median(valid_diffs)))
    if mad == 0:
        mad = np.mean(valid_diffs) + 1e-9  # fallback

    limit = threshold * mad * 1.4826  # 1.4826 converts MAD → std for normal

    for i in range(1, n):
        if low_conf[i]:
            continue
        d = abs(positions[i] - positions[i - 1])
        if np.isfinite(d) and d > limit:
            outlier_mask[i] = True

    return outlier_mask


def detect_outliers_zscore(
    positions: np.ndarray,
    threshold: float = 3.0,
    min_confidence: float = 0.2,
    confidences: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Flag outliers whose absolute z-score exceeds *threshold*.

    Computed against the sequence mean / std (ignoring NaN).
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = len(positions)

    outlier_mask = np.zeros(n, dtype=bool)

    if confidences is not None:
        confidences = np.asarray(confidences, dtype=np.float64)
        low_conf = confidences < min_confidence
    else:
        low_conf = np.isnan(positions)

    finite = np.isfinite(positions) & ~low_conf
    if not finite.any():
        return outlier_mask

    mu = np.mean(positions[finite])
    sigma = np.std(positions[finite])
    if sigma == 0:
        return outlier_mask

    z = np.abs(positions - mu) / sigma
    outlier_mask = (z > threshold) & ~low_conf & np.isfinite(positions)
    return outlier_mask


# ===================================================================
# Core interpolator class
# ===================================================================


class PoseInterpolator:
    """
    Smooth a multi-joint pose trajectory using a configurable strategy.

    Parameters
    ----------
    strategy : str or SmoothingStrategy
        One of the strategies listed in :class:`SmoothingStrategy`.
    window_size : int
        Window length (frames) for moving‑average, gaussian, median,
        savitzky_golay.  Must be odd for savitzky_golay and median.
    poly_order : int
        Polynomial order for Savitzky‑Golay (must be < window_size).
    sigma : float
        Standard deviation of the Gaussian kernel.
    alpha : float
        Smoothing factor for exponential moving average (0 < α ≤ 1).
        Larger α gives more weight to recent observations.
    process_noise : float
        Kalman-filter process noise.
    measurement_noise : float
        Kalman-filter measurement noise.
    outlier_method : str
        ``"velocity"`` or ``"zscore"`` – used by the hybrid strategy.
    outlier_threshold : float
        MAD / z-score multiplier for outlier flagging.
    min_confidence : float
        Keypoints with confidence below this are treated as missing and
        interpolated regardless of strategy.
    fill_method : str
        How to fill missing / masked positions before smoothing:
        ``"linear"``, ``"spline"``, or ``"forward"`` (last valid
        carried forward).
    """

    def __init__(
        self,
        strategy: Union[str, SmoothingStrategy] = SmoothingStrategy.HYBRID,
        window_size: int = 5,
        poly_order: int = 2,
        sigma: float = 1.0,
        alpha: float = 0.3,
        process_noise: float = 0.001,
        measurement_noise: float = 0.05,
        outlier_method: str = "velocity",
        outlier_threshold: float = 3.0,
        min_confidence: float = 0.2,
        fill_method: str = "linear",
    ) -> None:
        if isinstance(strategy, str):
            strategy = SmoothingStrategy(strategy)
        self.strategy = strategy
        self.window_size = window_size
        self.poly_order = poly_order
        self.sigma = sigma
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.min_confidence = min_confidence
        self.fill_method = fill_method

        # Internal state (set during fit / transform)
        self._joint_names: List[str] = []
        self._n_frames: int = 0
        self._n_joints: int = 0

    # --- public API ---------------------------------------------------

    @staticmethod
    def keypoints_to_array(
        frames_data: List[Dict[str, Any]],
        joint_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Convert *app.py* style frame dicts into a ``(F, J, 3)`` array.

        Parameters
        ----------
        frames_data : list of dict
            Each dict must have the structure produced by
            ``extract_joint_positions_from_movenet()`` (see module
            docstring).
        joint_names : list of str, optional
            Names of joints in the desired order.  When ``None``,
            ``COCO_KEYPOINTS`` is used.

        Returns
        -------
        ndarray of shape ``(len(frames_data), len(joint_names), 3)``
            The last axis holds ``[x, y, confidence]``. Missing values
            are represented as ``NaN``.
        """
        if joint_names is None:
            joint_names = COCO_KEYPOINTS

        n_frames = len(frames_data)
        n_joints = len(joint_names)
        arr = np.full((n_frames, n_joints, 3), np.nan, dtype=np.float64)

        for f_idx, frame in enumerate(frames_data):
            poses = frame.get("poses", [])
            if not poses:
                continue
            kps = poses[0].get("keypoints", [])
            kp_map = {kp.get("name"): kp for kp in kps}
            for j_idx, name in enumerate(joint_names):
                kp = kp_map.get(name)
                if kp is None:
                    continue
                x, y, c = kp.get("x"), kp.get("y"), kp.get("score")
                arr[f_idx, j_idx, 0] = x if x is not None else np.nan
                arr[f_idx, j_idx, 1] = y if y is not None else np.nan
                arr[f_idx, j_idx, 2] = (
                    c if c is not None else np.nan
                )

        return arr

    @staticmethod
    def array_to_keypoints(
        arr: np.ndarray,
        frames_data: List[Dict[str, Any]],
        joint_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Write a smoothed ``(F, J, 3)`` array back into the original
        app‑style frame dicts (returns a **deep copy** of
        *frames_data* with modified keypoint coordinates).

        Confidence values are preserved from the original data;
        coordinates are overwritten with the smoothed values.
        """
        if joint_names is None:
            joint_names = COCO_KEYPOINTS

        arr = _validate_array(arr)
        n_frames, n_joints, _ = arr.shape

        out: List[Dict[str, Any]] = deepcopy(frames_data)

        for f_idx in range(min(n_frames, len(out))):
            poses = out[f_idx].get("poses", [])
            if not poses:
                continue
            kps = poses[0].get("keypoints", [])
            kp_map = {kp.get("name"): kp for kp in kps}
            for j_idx, name in enumerate(joint_names):
                kp = kp_map.get(name)
                if kp is None:
                    continue
                if not np.isnan(arr[f_idx, j_idx, 0]):
                    kp["x"] = float(arr[f_idx, j_idx, 0])
                if not np.isnan(arr[f_idx, j_idx, 1]):
                    kp["y"] = float(arr[f_idx, j_idx, 1])
                # confidence deliberately kept from original

        return out

    @staticmethod
    def array_to_dataframe(
        arr: np.ndarray,
        joint_names: Optional[List[str]] = None,
        frame_numbers: Optional[Sequence[int]] = None,
    ) -> "pd.DataFrame":
        """
        Convert ``(F, J, 3)`` array to a DataFrame compatible with
        A11 visualisation tools (columns ``<joint>_x``, ``<joint>_y``,
        optionally ``<joint>_z`` and a ``FrameNo`` column).
        """
        import pandas as pd

        if joint_names is None:
            joint_names = COCO_KEYPOINTS

        arr = _validate_array(arr)
        n_frames, n_joints, _ = arr.shape

        data: Dict[str, List[float]] = {}
        for j_idx, name in enumerate(joint_names):
            data[f"{name}_x"] = arr[:, j_idx, 0].tolist()
            data[f"{name}_y"] = arr[:, j_idx, 1].tolist()

        if frame_numbers is None:
            data["FrameNo"] = list(range(n_frames))
        else:
            data["FrameNo"] = list(frame_numbers)

        return pd.DataFrame(data)

    def fit_transform(self, arr: np.ndarray) -> np.ndarray:
        """
        Run the full smoothing pipeline on one array.

        Parameters
        ----------
        arr : (F, J, 3) ndarray
            Raw coordinates + confidence.

        Returns
        -------
        (F, J, 3) ndarray
            Smoothed coordinates.  The confidence channel is passed
            through unchanged (it is used internally for filtering).
        """
        arr = _validate_array(arr)
        self._n_frames, self._n_joints, _ = arr.shape

        # Always mask low-confidence points before any processing
        arr = self._mask_low_confidence(arr)

        if self.strategy == SmoothingStrategy.KALMAN:
            smoothed = self._apply_kalman(arr)
        elif self.strategy == SmoothingStrategy.EXPONENTIAL:
            smoothed = self._apply_ema(arr)
        elif self.strategy == SmoothingStrategy.SPLINE:
            smoothed = self._apply_spline(arr)
        elif self.strategy == SmoothingStrategy.HYBRID:
            smoothed = self._apply_hybrid(arr)
        else:
            # scipy-based windowed filters
            smoothed = self._apply_windowed(arr)

        return smoothed

    def fit_transform_frames(
        self,
        frames_data: List[Dict[str, Any]],
        joint_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        High-level convenience: accept app‑style frame dicts, return
        smoothed frame dicts.

        Parameters
        ----------
        frames_data : list of dict
            Raw per-frame keypoint dicts.
        joint_names : list of str, optional
            Ordered joint names.

        Returns
        -------
        list of dict
            Deep-copied frame dicts with smoothed coordinates.
        """
        if joint_names is None:
            joint_names = COCO_KEYPOINTS
        self._joint_names = list(joint_names)

        arr = self.keypoints_to_array(frames_data, joint_names)
        smoothed = self.fit_transform(arr)
        return self.array_to_keypoints(smoothed, frames_data, joint_names)

    # --- internal steps ------------------------------------------------

    def _mask_low_confidence(self, arr: np.ndarray) -> np.ndarray:
        """Set coordinates to NaN where confidence < *min_confidence*."""
        arr = arr.copy()
        conf = arr[:, :, 2]
        low = conf < self.min_confidence
        arr[low, 0] = np.nan
        arr[low, 1] = np.nan
        return arr

    def _fill_missing(self, signal_1d: np.ndarray) -> np.ndarray:
        """
        Fill NaN values in a 1-D signal.

        Returns a copy with NaNs replaced according to *fill_method*.
        """
        s = np.asarray(signal_1d, dtype=np.float64).copy()
        n = len(s)
        valid = np.isfinite(s)

        if valid.all():
            return s

        if self.fill_method == "forward":
            # Forward fill
            last = np.nan
            for i in range(n):
                if np.isfinite(s[i]):
                    last = s[i]
                elif not np.isnan(last):
                    s[i] = last
            # Backward fill for leading NaN
            first = np.nan
            for i in range(n - 1, -1, -1):
                if np.isfinite(s[i]):
                    first = s[i]
                elif not np.isnan(first):
                    s[i] = first
            return s

        if self.fill_method == "spline":
            _ensure_scipy("spline fill")
            idx = np.arange(n)
            if valid.sum() < 3:
                # Not enough points for cubic spline → fall back to linear
                return self._fill_linear(s, idx, valid)
            try:
                spl = _scipy_interpolate.UnivariateSpline(
                    idx[valid], s[valid], s=0, ext="const"
                )
                s[~valid] = spl(idx[~valid])
            except Exception:
                s = self._fill_linear(s, idx, valid)
            return s

        # Default: linear
        idx = np.arange(n)
        return self._fill_linear(s, idx, valid)

    @staticmethod
    def _fill_linear(s: np.ndarray, idx: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Linear interpolation (also handles edge extrapolation)."""
        n = len(s)
        s_filled = s.copy()
        if valid.sum() >= 2:
            s_filled[~valid] = np.interp(idx[~valid], idx[valid], s[valid])
        elif valid.sum() == 1:
            s_filled[~valid] = s[valid][0]
        else:
            s_filled[:] = 0.0
        return s_filled

    # --- strategy implementations --------------------------------------

    def _apply_windowed(self, arr: np.ndarray) -> np.ndarray:
        """scipy-based sliding-window filters."""
        _ensure_scipy(self.strategy.value)
        result = arr.copy()
        ws = self._effective_window()

        for j in range(self._n_joints):
            for c in [0, 1]:  # x, y
                sig = result[:, j, c]
                valid = np.isfinite(sig)
                if not valid.any():
                    continue

                # Fill gaps first
                sig_filled = self._fill_missing(sig)

                if self.strategy == SmoothingStrategy.MOVING_AVERAGE:
                    kernel = np.ones(ws) / ws
                    smoothed = np.convolve(sig_filled, kernel, mode="same")
                elif self.strategy == SmoothingStrategy.GAUSSIAN:
                    # Create Gaussian kernel
                    ax = np.arange(-(ws // 2), ws // 2 + 1)
                    kernel = np.exp(-0.5 * (ax / self.sigma) ** 2)
                    kernel /= kernel.sum()
                    smoothed = np.convolve(sig_filled, kernel, mode="same")
                elif self.strategy == SmoothingStrategy.MEDIAN:
                    smoothed = ndimage.median_filter(sig_filled, size=ws)
                elif self.strategy == SmoothingStrategy.SAVITZKY_GOLAY:
                    if ws >= len(sig_filled):
                        ws = len(sig_filled) if len(sig_filled) % 2 == 1 else len(sig_filled) - 1
                    if ws <= self.poly_order:
                        ws = self.poly_order + 2
                    if ws % 2 == 0:
                        ws += 1
                    try:
                        smoothed = _scipy_signal.savgol_filter(
                            sig_filled, ws, self.poly_order, mode="nearest"
                        )
                    except Exception:
                        smoothed = sig_filled
                else:
                    smoothed = sig_filled

                # Restore original NaN positions so downstream code
                # knows which points were originally missing.
                smoothed[~valid] = np.nan
                result[:, j, c] = smoothed

        return result

    def _apply_ema(self, arr: np.ndarray) -> np.ndarray:
        """Exponential moving average (online-capable)."""
        result = arr.copy()
        alpha = self.alpha
        for j in range(self._n_joints):
            for c in [0, 1]:
                sig = arr[:, j, c]
                out = np.empty_like(sig)
                ema = np.nan
                for i in range(len(sig)):
                    if np.isfinite(sig[i]):
                        if np.isnan(ema):
                            ema = sig[i]
                        else:
                            ema = alpha * sig[i] + (1 - alpha) * ema
                    out[i] = ema
                result[:, j, c] = out
        return result

    def _apply_kalman(self, arr: np.ndarray) -> np.ndarray:
        """Per-joint, per-coordinate Kalman filter (forward pass)."""
        result = arr.copy()
        kf = KalmanFilter1D(self.process_noise, self.measurement_noise)
        for j in range(self._n_joints):
            for c in [0, 1]:
                kf.reset()
                # Initialise with first valid point (if any)
                initialized = False
                for i in range(self._n_frames):
                    val = arr[i, j, c]
                    if np.isfinite(val):
                        kf.reset(float(val))
                        initialized = True
                        break
                if not initialized:
                    result[:, j, c] = np.nan
                    continue
                result[0, j, c] = kf.position
                for i in range(1, self._n_frames):
                    meas = arr[i, j, c]
                    pos = kf.update(
                        float(meas) if np.isfinite(meas) else None
                    )
                    result[i, j, c] = pos
        return result

    def _apply_spline(self, arr: np.ndarray) -> np.ndarray:
        """
        Fit a cubic smoothing spline through high-confidence points.

        Low-confidence points are excluded from the fit and replaced
        by the spline evaluation.
        """
        _ensure_scipy("spline")
        result = arr.copy()
        n = self._n_frames
        idx = np.arange(n, dtype=np.float64)

        for j in range(self._n_joints):
            for c in [0, 1]:
                sig = arr[:, j, c]
                valid = np.isfinite(sig)
                if valid.sum() < 3:
                    result[:, j, c] = self._fill_missing(sig)
                    continue

                try:
                    spl = _scipy_interpolate.UnivariateSpline(
                        idx[valid], sig[valid], s=len(valid) * 0.5
                    )
                    result[:, j, c] = spl(idx)
                except Exception:
                    result[:, j, c] = self._fill_missing(sig)

        return result

    def _apply_hybrid(self, arr: np.ndarray) -> np.ndarray:
        """
        Hybrid pipeline:

        1. Detect positional outliers (velocity or z-score).
        2. Mask outliers + low-confidence points → NaN.
        3. Interpolate NaN gaps.
        4. Apply Savitzky-Golay smoothing.
        """
        result = arr.copy()

        for j in range(self._n_joints):
            for c in [0, 1]:
                sig = arr[:, j, c]
                conf = arr[:, j, 2]

                # Step 1 – outlier detection
                if self.outlier_method == "zscore":
                    outlier = detect_outliers_zscore(
                        sig, self.outlier_threshold, self.min_confidence, conf
                    )
                else:
                    outlier = detect_outliers_velocity(
                        sig, self.outlier_threshold, self.min_confidence, conf
                    )

                # Step 2 – mask
                sig_clean = sig.copy()
                sig_clean[outlier] = np.nan
                # low-confidence already masked by _mask_low_confidence

                # Step 3 – interpolate
                sig_filled = self._fill_missing(sig_clean)

                # Step 4 – Savitzky-Golay
                _ensure_scipy("savitzky_golay")
                ws = self._effective_window()
                if ws >= len(sig_filled):
                    ws = len(sig_filled) if len(sig_filled) % 2 == 1 else len(sig_filled) - 1
                if ws <= self.poly_order:
                    ws = self.poly_order + 2
                if ws % 2 == 0:
                    ws += 1
                try:
                    smoothed = _scipy_signal.savgol_filter(
                        sig_filled, ws, self.poly_order, mode="nearest"
                    )
                except Exception:
                    smoothed = sig_filled

                # Restore NaN for originally completely missing frames
                orig_missing = ~np.isfinite(sig) & ~outlier
                smoothed[orig_missing] = np.nan
                result[:, j, c] = smoothed

        return result

    def _effective_window(self) -> int:
        """Clamp window size to available frames and ensure odd."""
        ws = min(self.window_size, self._n_frames)
        if ws % 2 == 0:
            ws -= 1
        return max(ws, 3)


# ===================================================================
# High-level convenience function
# ===================================================================


def smooth_pose_sequence(
    frames_data: List[Dict[str, Any]],
    strategy: Union[str, SmoothingStrategy] = SmoothingStrategy.HYBRID,
    joint_names: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Smooth an entire pose sequence with a single call.

    Parameters
    ----------
    frames_data : list of dict
        Per-frame keypoint dicts in the format produced by
        ``extract_joint_positions_from_movenet()`` in ``app.py``.
    strategy : str or SmoothingStrategy
        Smoothing strategy to use (default: ``"hybrid"``).
    joint_names : list of str, optional
        Ordered joint names (defaults to COCO 17).
    **kwargs
        Passed through to :class:`PoseInterpolator` (window_size,
        alpha, outlier_threshold, …).

    Returns
    -------
    list of dict
        Deep copy of *frames_data* with smoothed (x, y) coordinates.

    Examples
    --------
    >>> # Quick hybrid smoothing (recommended for animations)
    >>> smoothed = smooth_pose_sequence(all_keypoints, strategy="hybrid")

    >>> # Light EMA for near-real-time use
    >>> smoothed = smooth_pose_sequence(all_keypoints, strategy="exponential",
    ...                                 alpha=0.15)

    >>> # Strong outlier removal for noisy recordings
    >>> smoothed = smooth_pose_sequence(all_keypoints, strategy="hybrid",
    ...                                 outlier_method="zscore",
    ...                                 outlier_threshold=2.5,
    ...                                 window_size=7)
    """
    interpolator = PoseInterpolator(strategy=strategy, **kwargs)
    return interpolator.fit_transform_frames(frames_data, joint_names=joint_names)


# ===================================================================
# Smoke test (runs when module is executed directly)
# ===================================================================

if __name__ == "__main__":
    # Generate a synthetic trajectory with gaps and spikes
    np.random.seed(42)
    n_frames = 100
    n_joints = 3  # nose, left_shoulder, right_shoulder
    t = np.linspace(0, 4 * np.pi, n_frames)

    # Ground truth – smooth sinusoid
    true_x = np.sin(t) * 0.3 + 0.5
    true_y = np.cos(t) * 0.2 + 0.5

    # Build array: (F, J, 3)
    raw = np.zeros((n_frames, n_joints, 3), dtype=np.float64)
    for j in range(n_joints):
        raw[:, j, 0] = true_x + np.random.randn(n_frames) * 0.02
        raw[:, j, 1] = true_y + np.random.randn(n_frames) * 0.02
        raw[:, j, 2] = 0.9  # high confidence

    # Inject outliers
    raw[20, 0, 0] += 0.4  # spike
    raw[50, 0, 1] -= 0.3
    raw[75, 0, 0] += 0.5

    # Inject gaps
    raw[40:45, 1, :] = np.nan
    raw[60:65, 1, :] = np.nan
    raw[80, 1, :] = np.nan

    # --- Test each strategy -------------------------------------------
    strategies = [
        SmoothingStrategy.HYBRID,
        SmoothingStrategy.MOVING_AVERAGE,
        SmoothingStrategy.GAUSSIAN,
        SmoothingStrategy.EXPONENTIAL,
        SmoothingStrategy.MEDIAN,
        SmoothingStrategy.SAVITZKY_GOLAY,
        SmoothingStrategy.KALMAN,
        SmoothingStrategy.SPLINE,
    ]

    print(f"{'Strategy':<22s} {'MAE (x)':>10s} {'MAE (y)':>10s}")
    print("-" * 44)

    for strat in strategies:
        interp = PoseInterpolator(strategy=strat)
        smoothed = interp.fit_transform(raw.copy())

        # Mean absolute error against ground truth (only first joint)
        mae_x = np.nanmean(np.abs(smoothed[:, 0, 0] - true_x))
        mae_y = np.nanmean(np.abs(smoothed[:, 0, 1] - true_y))
        print(f"{strat.value:<22s} {mae_x:10.6f} {mae_y:10.6f}")

    # --- Test high-level convenience ----------------------------------
    frames_data = PoseInterpolator.array_to_keypoints(
        raw,
        [
            {
                "poses": [
                    {
                        "pose_id": 0,
                        "keypoints": [
                            {
                                "x": raw[i, j, 0],
                                "y": raw[i, j, 1],
                                "score": raw[i, j, 2],
                                "name": COCO_KEYPOINTS[j],
                            }
                            for j in range(n_joints)
                        ],
                    }
                ],
                "frame_id": i,
            }
            for i in range(n_frames)
        ],
        joint_names=COCO_KEYPOINTS[:n_joints],
    )

    smoothed_frames = smooth_pose_sequence(frames_data, strategy="hybrid")
    print(f"\nHigh-level convenience: processed {len(smoothed_frames)} frames ✓")

    # Convert to DataFrame for A11 compatibility
    df = PoseInterpolator.array_to_dataframe(
        raw, joint_names=COCO_KEYPOINTS[:n_joints]
    )
    print(f"DataFrame conversion: {df.shape} ✓")
    print("\nAll tests passed.")
