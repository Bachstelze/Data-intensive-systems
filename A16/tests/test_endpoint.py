"""Unit tests for the A16 unified endpoint.

These tests are intentionally **model-free**: the upstream
``ExercisePipeline`` and the A15 scorer are monkey-patched so CI does not
need GPU / large model artefacts. They lock in the public response schema
and the error/ugly branches.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from A16.service import endpoint as ep  # noqa: E402


# ---------- helpers ----------------------------------------------------------

class _FakePipeline:
    """Stand-in for :class:`ExercisePipeline` — returns a canned result."""

    def __init__(self, result, *args, **kwargs):
        self._result = result

    def process_video(self, video_path):  # noqa: D401 — signature match
        return self._result

    def close(self):
        pass


def _install_fake_pipeline(monkeypatch, result):
    """Replace the upstream pipeline with a fake returning ``result``."""
    import types

    fake_module = types.ModuleType("exercise_pipeline")

    class _Factory(_FakePipeline):
        def __init__(self, *args, **kwargs):
            super().__init__(result)

    fake_module.ExercisePipeline = _Factory
    monkeypatch.setitem(sys.modules, "exercise_pipeline", fake_module)


# ---------- schema -----------------------------------------------------------

REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version",
    "endpoint",
    "variant",
    "status",
    "message",
    "recording",
    "segment",
    "classification",
    "score",
    "artefacts",
    "timing_ms",
    "warnings",
}

REQUIRED_TIMING_KEYS = {
    "upstream_ms", "scorer_nn_ms", "scorer_total_ms", "total_ms",
}


def _assert_schema(resp):
    assert isinstance(resp, dict)
    missing = REQUIRED_TOP_LEVEL_KEYS - set(resp.keys())
    assert not missing, f"missing top-level keys: {missing}"
    assert resp["endpoint"] == "A16"
    assert resp["schema_version"] == ep.A16_RESPONSE_VERSION
    assert resp["variant"] in {"2D", "3D"}
    assert REQUIRED_TIMING_KEYS <= set(resp["timing_ms"].keys())
    for section in ("recording", "segment", "classification", "score", "artefacts"):
        assert isinstance(resp[section], dict)
    assert isinstance(resp["warnings"], list)


# ---------- tests ------------------------------------------------------------

class TestSchema:

    def test_no_video_returns_well_formed_error(self):
        resp = ep.run_pipeline_3d(None)
        _assert_schema(resp)
        assert resp["status"] == ep.STATUS_ERROR_NO_VIDEO

    def test_2d_alternative_returns_well_formed_placeholder(self):
        resp = ep.run_pipeline_2d("dummy.mp4")
        _assert_schema(resp)
        assert resp["variant"] == "2D"
        # 2D not implemented yet — must surface as structured error, not raise.
        assert resp["status"] == ep.STATUS_ERROR_PIPELINE

    def test_dispatcher_routes_variant(self):
        assert ep.run_pipeline(None, variant="2D")["variant"] == "2D"
        assert ep.run_pipeline(None, variant="3D")["variant"] == "3D"


class TestUglyPath:

    def test_ugly_recording_short_circuits(self, monkeypatch):
        ugly_upstream = {
            "video": "x.mp4",
            "total_frames": 30,
            "recording_quality": "UGLY",
            "recording_confidence": 0.31,
            "recording_threshold": 0.6,
            "pipeline_stopped": True,
            "reason": "Poor recording quality.",
        }
        _install_fake_pipeline(monkeypatch, ugly_upstream)
        resp = ep.run_pipeline_3d("does-not-need-to-exist.mp4")
        _assert_schema(resp)
        assert resp["status"] == ep.STATUS_REJECTED_UGLY
        assert resp["recording"]["quality_label"] == "UGLY"
        assert resp["recording"]["quality_confidence"] == 0.31
        # No score should have been computed on the ugly branch.
        assert resp["score"]["value"] is None
        assert resp["timing_ms"]["scorer_nn_ms"] == 0.0


class TestHappyPath:

    def test_full_pipeline_with_mocked_scorer(self, monkeypatch, tmp_path):
        # Fake cut CSV with 10 frames and the expected 13-joint xyz columns.
        import pandas as pd
        from A15.inference import A15_JOINTS, A15_C

        cols = [f"{j}_{ax}" for j in A15_JOINTS for ax in ("x", "y", "z")]
        df = pd.DataFrame(
            [[0.0] * len(cols) for _ in range(A15_C)],
            columns=cols,
        )
        cut_csv = tmp_path / "demo_cut_3d_points.csv"
        df.to_csv(cut_csv, index=False)

        # Point the endpoint's artefact resolution at our tmp dir.
        monkeypatch.setattr(ep, "OUTPUTS_DIR", tmp_path)

        good_upstream = {
            "video": "demo.mp4",
            "total_frames": 90,
            "start_frame": 10,
            "stop_frame": 70,
            "exercise_frames": 61,
            "exercise_duration_sec": 2.03,
            "quality_label": "GOOD",
            "quality_confidence": 0.87,
            "recording_quality": "GOOD",
            "recording_confidence": 0.78,
            "recording_threshold": 0.6,
        }
        _install_fake_pipeline(monkeypatch, good_upstream)

        # Mock the A15 scorer so we don't load Keras / joblib in CI.
        import A15.inference as inf
        monkeypatch.setattr(
            inf, "predict_score", lambda d: (0.42, "GREEN — acceptable form (0-1)", 1.5)
        )

        # `_resolve_artefacts` uses the video stem; mirror it in tmp.
        video_path = str(tmp_path / "demo.mp4")
        # Need the upstream stem-prefixed CSV to exist where _resolve_artefacts looks.
        (tmp_path / "demo_cut_3d_points.csv").write_text(cut_csv.read_text())

        resp = ep.run_pipeline_3d(video_path)
        _assert_schema(resp)
        assert resp["status"] == ep.STATUS_OK
        assert resp["classification"]["label"] == "GOOD"
        assert resp["segment"]["start_frame"] == 10
        assert resp["segment"]["stop_frame"] == 70
        assert resp["score"]["value"] == pytest.approx(0.42)
        assert "GREEN" in resp["score"]["band"]
        assert resp["timing_ms"]["scorer_nn_ms"] == 1.5
        assert resp["timing_ms"]["total_ms"] >= 0


class TestBandMapping:

    @pytest.mark.parametrize("score,prefix", [
        (0.0, "GREEN"),
        (0.99, "GREEN"),
        (1.0, "AMBER"),
        (1.99, "AMBER"),
        (2.0, "RED"),
        (4.0, "RED"),
    ])
    def test_score_band_boundaries(self, score, prefix):
        from A15.inference import score_band
        assert score_band(score).startswith(prefix)


class TestRecordingQualityGate:
    """Regression tests for the detection-rate based UGLY gate.

    The previous implementation averaged MediaPipe ``visibility`` values
    which collapse to ~0.05 on the modern Tasks API → every real recording
    was rejected. The new metric is the fraction of early frames in which
    enough joints were detected. These tests pin that behaviour.
    """

    JOINTS = [
        'nose', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
    ]

    def _frame(self, *, detected: bool):
        if detected:
            kps = {j: {"x": 0.5, "y": 0.5, "confidence": 0.5} for j in self.JOINTS}
        else:
            kps = {j: {"x": 0.0, "y": 0.0, "confidence": 0.0} for j in self.JOINTS}
        return {"keypoints": kps}

    def _gate(self):
        # Bind the unbound method so we don't need to instantiate the full
        # pipeline (which would load every Keras model).
        from exercise_pipeline import ExercisePipeline
        return ExercisePipeline.assess_recording_quality

    def test_all_frames_detected_returns_good(self):
        frames = [self._frame(detected=True) for _ in range(30)]
        label, conf = self._gate()(None, frames, threshold=0.6)
        assert label == "GOOD"
        assert conf == pytest.approx(1.0)

    def test_no_frames_detected_returns_ugly(self):
        frames = [self._frame(detected=False) for _ in range(30)]
        label, conf = self._gate()(None, frames, threshold=0.6)
        assert label == "UGLY"
        assert conf == pytest.approx(0.0)

    def test_mediapipe_tasks_low_visibility_regression(self):
        """A pose detected on every frame but with the Tasks-API typical
        low visibility (~0.05) must NOT be rejected — this is the exact
        scenario that bit the OK-squat upload in production."""
        frames = []
        for _ in range(30):
            kps = {
                j: {"x": 0.5, "y": 0.5, "confidence": 0.05} for j in self.JOINTS
            }
            frames.append({"keypoints": kps})
        label, conf = self._gate()(None, frames, threshold=0.6)
        assert label == "GOOD", \
            "Tasks-API visibility ~0.05 with full pose detected must pass"
        assert conf >= 0.6

    def test_empty_results_returns_ugly(self):
        label, conf = self._gate()(None, [], threshold=0.6)
        assert label == "UGLY"
        assert conf == 0.0
