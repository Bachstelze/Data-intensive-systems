import pandas as pd
import pytest

from A12.service.contracts import FEATURES_BY_PROBLEM, normalize_problem, validate_pose_dataframe


def test_normalize_problem_accepts_a_and_b():
    assert normalize_problem("A") == "A"
    assert normalize_problem("b") == "B"
    assert normalize_problem("B - PoseNet") == "B"


def test_normalize_problem_rejects_unknown_value():
    with pytest.raises(ValueError):
        normalize_problem("C")


def test_validate_pose_dataframe_reports_missing_columns():
    df = pd.DataFrame({"head_x": [1.0]})
    with pytest.raises(ValueError, match="missing"):
        validate_pose_dataframe(df, "B")


def test_validate_pose_dataframe_accepts_problem_b_feature_schema():
    df = pd.DataFrame({col: [0.5] for col in FEATURES_BY_PROBLEM["B"]})
    features, names = validate_pose_dataframe(df, "B")
    assert list(features.columns) == FEATURES_BY_PROBLEM["B"]
    assert names == FEATURES_BY_PROBLEM["B"]
    assert features.shape == (1, len(FEATURES_BY_PROBLEM["B"]))
