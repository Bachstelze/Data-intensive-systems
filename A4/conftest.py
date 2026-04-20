# Provides reusable model paths sample data and loaded model fixtures for testing regression and classification models that we have so far.

import pytest
import os
import pickle
import pandas as pd

# path fixtures

@pytest.fixture
def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def models_dir(repo_root):
    return os.path.join(repo_root, "A3", "models")


@pytest.fixture
def a4_models_dir(repo_root):
    return os.path.join(repo_root, "A4", "models")


@pytest.fixture
def regression_model_path(models_dir):
    return os.path.join(models_dir, "champion_model_final_2.pkl")


@pytest.fixture
def classification_model_path(a4_models_dir):
    return os.path.join(a4_models_dir, "weaklink_classifier_rf.pkl")


@pytest.fixture
def datasets_dir(repo_root):
    return os.path.join(repo_root, "Datasets_all")


# Model Fixtures

@pytest.fixture
def regression_artifact(regression_model_path):
    
    # return the regression model dict
    if not os.path.exists(regression_model_path):
        pytest.skip(f"Model not found: {regression_model_path}")
    
    with open(regression_model_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture
def classification_artifact(classification_model_path):
    
    # return the classification model dict
    if not os.path.exists(classification_model_path):
        pytest.skip(f"Model not found: {classification_model_path}")
    
    with open(classification_model_path, "rb") as f:
        return pickle.load(f)


# sample data

@pytest.fixture
def sample_regression_features(regression_artifact):
    
    # sample feature and data for testing
    feature_columns = regression_artifact["feature_columns"]
    
    sample_data = {col: [0.5] for col in feature_columns}
    return pd.DataFrame(sample_data)


@pytest.fixture
def sample_classification_features(classification_artifact):

    feature_columns = classification_artifact["feature_columns"]
    
    sample_data = {col: [0.5] for col in feature_columns}
    return pd.DataFrame(sample_data)

# expected values
@pytest.fixture
def expected_classification_classes():
    # all 14 weaklink categories
    return [
        'ExcessiveForwardLean', 'ForwardHead', 'LeftArmFallForward',
        'LeftAsymmetricalWeightShift', 'LeftHeelRises', 'LeftKneeMovesInward',
        'LeftKneeMovesOutward', 'LeftShoulderElevation', 'RightArmFallForward',
        'RightAsymmetricalWeightShift', 'RightHeelRises', 'RightKneeMovesInward',
        'RightKneeMovesOutward', 'RightShoulderElevation'
    ]
