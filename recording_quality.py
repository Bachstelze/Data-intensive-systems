"""Pure helpers for the recording-quality (UGLY) gate.

Extracted from :mod:`exercise_pipeline` so the logic can be unit-tested in
CI without pulling in OpenCV / TensorFlow / MediaPipe. The pipeline
delegates to :func:`assess_recording_quality` so there is a single source
of truth.

The metric is a **per-frame detection rate** rather than an average of
MediaPipe's ``landmark.visibility``, because the modern Tasks API
(``pose_landmarker_lite.task``) returns visibility values on a tiny scale
(~0.05 even for a clearly visible pose) that made the legacy
average-visibility metric reject every real recording.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


# Joints used to decide whether a frame contains a usable pose. Names match
# the MoveNet / MediaPipe key-naming convention used throughout the repo.
QUALITY_JOINTS: List[str] = [
    'nose', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
]

# A frame counts as "detected" when at least this fraction of QUALITY_JOINTS
# have non-zero coordinates with a non-trivial visibility.
JOINT_PRESENCE_RATIO = 0.6

# Visibility below this is treated as "no detection" — guards against the
# fully-zeroed fallback frames emitted by MediaPipe when no pose was found.
MIN_JOINT_VISIBILITY = 1e-3


def assess_recording_quality(
    pose_results: Iterable[Dict[str, Any]],
    threshold: float = 0.6,
    n_frames: int = 30,
) -> Tuple[str, float]:
    """Decide whether the early frames of a recording are usable.

    Parameters
    ----------
    pose_results : iterable of dict
        Per-frame pose results in the shape produced by
        :class:`A14.mediapipe_pose_estimator.MediaPipePoseEstimator` —
        each item is ``{"keypoints": {joint: {"x", "y", "confidence"}}}``.
    threshold : float, default 0.6
        Minimum detection-rate required to mark the recording as ``GOOD``.
    n_frames : int, default 30
        Number of leading frames inspected.

    Returns
    -------
    (label, confidence) : tuple of (str, float)
        ``label`` ∈ {"GOOD", "UGLY"} and ``confidence`` ∈ [0, 1] is the
        fraction of detected frames in the early window.
    """
    early_frames = list(pose_results)[:n_frames]
    if not early_frames:
        return 'UGLY', 0.0

    joints_needed_per_frame = max(
        1, int(JOINT_PRESENCE_RATIO * len(QUALITY_JOINTS))
    )
    detected_frames = 0

    for result in early_frames:
        kps = (result or {}).get('keypoints', {})
        joints_present = 0
        for j in QUALITY_JOINTS:
            kp = kps.get(j)
            if not kp:
                continue
            x = kp.get('x', 0.0) or 0.0
            y = kp.get('y', 0.0) or 0.0
            vis = kp.get('confidence', 0.0) or 0.0
            if (x != 0.0 or y != 0.0) and vis > MIN_JOINT_VISIBILITY:
                joints_present += 1
        if joints_present >= joints_needed_per_frame:
            detected_frames += 1

    detection_rate = detected_frames / float(len(early_frames))
    label = 'GOOD' if detection_rate >= threshold else 'UGLY'
    return label, float(detection_rate)


__all__ = [
    'QUALITY_JOINTS',
    'JOINT_PRESENCE_RATIO',
    'MIN_JOINT_VISIBILITY',
    'assess_recording_quality',
]
