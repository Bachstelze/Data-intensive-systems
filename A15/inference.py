"""A15 scoring — reusable inference helpers.

Extracted from ``app.py`` so that other endpoints (notably A16) can reuse the
deployed 0–4 regression scorer without importing the Gradio app module.

The deployed champion architecture (Dense_medium) and scaling are described in
``A15_results/training_summary.json``. Models live in ``<repo_root>/models/``:

    - ``scoring_model.keras``  — Dense regressor (input: 390 = 10×13×3)
    - ``scoring_scaler.pkl``   — StandardScaler fitted on flattened frames
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

# 13-joint Kinect ordering used during A15 training.
A15_JOINTS = [
    'head', 'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow',
    'left_hand', 'right_hand', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_foot', 'right_foot',
]
A15_C = 10  # frames per clip the scorer was trained on

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH = _REPO_ROOT / 'models' / 'scoring_model.keras'
_SCALER_PATH = _REPO_ROOT / 'models' / 'scoring_scaler.pkl'

_MODEL = None
_SCALER = None


def load_a15_scorer():
    """Lazy-load the deployed A15 regression scorer.

    Returns
    -------
    (model, scaler) : tuple
        Keras model and fitted StandardScaler. Cached after first call.
    """
    global _MODEL, _SCALER
    if _MODEL is not None and _SCALER is not None:
        return _MODEL, _SCALER

    import joblib  # local import keeps app startup light
    from tensorflow import keras
    from tensorflow.keras import layers

    try:
        _MODEL = keras.models.load_model(str(_MODEL_PATH))
    except (TypeError, ValueError):
        # Saved with a newer Keras (extra ``quantization_config`` kwarg);
        # rebuild Dense_medium and load weights only. Architecture matches
        # ``A15_results/training_summary.json``'s deployed champion.
        inp = keras.Input(shape=(390,))
        x = layers.Dense(64, activation='relu')(inp)
        x = layers.Dropout(0.2)(x)
        out = layers.Dense(1, activation='linear')(x)
        _MODEL = keras.Model(inp, out, name='Dense')
        _MODEL.load_weights(str(_MODEL_PATH))

    _SCALER = joblib.load(str(_SCALER_PATH))
    return _MODEL, _SCALER


def sample_frames(df) -> np.ndarray:
    """Sample ``A15_C`` equally-spaced frames from a cut 3D dataframe.

    Returns array of shape ``(A15_C, len(A15_JOINTS), 3)``.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    idx = np.linspace(0, len(df) - 1, A15_C).astype(int)
    sub = df.iloc[idx]
    frames = []
    for _, row in sub.iterrows():
        frames.append([
            [row[f'{j}_x'], row[f'{j}_y'], row[f'{j}_z']]
            for j in A15_JOINTS
        ])
    return np.array(frames, dtype=np.float32)


def score_band(score: float) -> str:
    """Map a 0–4 score onto a traffic-light band."""
    if score < 1.0:
        return "GREEN — acceptable form (0-1)"
    if score < 2.0:
        return "AMBER — borderline (1-2)"
    return "RED — poor form (2-4)"


def predict_score(df) -> Tuple[float, str, float]:
    """End-to-end scoring: cut 3D dataframe → (clipped score, band, NN ms).

    Parameters
    ----------
    df : pandas.DataFrame
        Cut 3D dataframe with columns ``<joint>_x/y/z`` for the 13 A15 joints.

    Returns
    -------
    (score, band, nn_ms) : tuple
        ``score`` is clipped to ``[0, 4]``. ``nn_ms`` is the wall-clock time
        spent inside ``model.predict`` only (excludes sampling / scaling).
    """
    import time

    model, scaler = load_a15_scorer()
    frames = sample_frames(df)
    flat = frames.reshape(1, -1)
    scaled = scaler.transform(flat).astype(np.float32)
    if len(model.input_shape) == 3:
        scaled = scaled.reshape(1, A15_C, len(A15_JOINTS) * 3)

    t0 = time.perf_counter()
    raw = float(model.predict(scaled, verbose=0).flatten()[0])
    nn_ms = (time.perf_counter() - t0) * 1000.0

    score = float(np.clip(raw, 0.0, 4.0))
    return score, score_band(score), nn_ms
