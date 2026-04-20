import pytest
import os
import pickle
import numpy as np

# regression model tests
class TestRegressionModelLoading:

    def test_regression_model_file_exists(self, regression_model_path):
        if not os.path.exists(regression_model_path):
            pytest.skip(f"Model not found (LFS not pulled?): {regression_model_path}")
        assert os.path.exists(regression_model_path)

    def test_regression_artifact_is_dict(self, regression_artifact):
        assert isinstance(regression_artifact, dict)

    def test_regression_artifact_has_model_key(self, regression_artifact):
        assert "model" in regression_artifact

    def test_regression_artifact_has_feature_columns(self, regression_artifact):
        assert "feature_columns" in regression_artifact

    def test_regression_feature_columns_not_empty(self, regression_artifact):
        assert len(regression_artifact["feature_columns"]) > 0

    def test_regression_model_has_predict_method(self, regression_artifact):
        model = regression_artifact["model"]
        assert hasattr(model, "predict")


class TestRegressionModelPrediction:

    def test_regression_prediction_returns_array(
        self, regression_artifact, sample_regression_features
    ):
        # regression model should return numpy
        model = regression_artifact["model"]
        prediction = model.predict(sample_regression_features)
        assert isinstance(prediction, np.ndarray)

    def test_regression_prediction_shape(
        self, regression_artifact, sample_regression_features
    ):
        # one value for sample
        model = regression_artifact["model"]
        prediction = model.predict(sample_regression_features)
        assert prediction.shape[0] == len(sample_regression_features)

    def test_regression_prediction_is_numeric(
        self, regression_artifact, sample_regression_features
    ):
        # should be a number
        model = regression_artifact["model"]
        prediction = model.predict(sample_regression_features)
        assert np.issubdtype(prediction.dtype, np.number)

    def test_regression_prediction_in_reasonable_range(
        self, regression_artifact, sample_regression_features
    ):
        model = regression_artifact["model"]
        prediction = model.predict(sample_regression_features)[0]
        # Allow some tolerance outside 0-1 for edge cases
        assert -0.5 <= prediction <= 1.5


class TestClassificationModelLoading:

    def test_classification_model_file_exists(self, classification_model_path):
        if not os.path.exists(classification_model_path):
            pytest.skip(f"Model not found (LFS not pulled?): {classification_model_path}")
        assert os.path.exists(classification_model_path)

    def test_classification_artifact_is_dict(self, classification_artifact):
        assert isinstance(classification_artifact, dict)

    def test_classification_artifact_has_model_key(self, classification_artifact):
        assert "model" in classification_artifact

    def test_classification_artifact_has_feature_columns(self, classification_artifact):
        assert "feature_columns" in classification_artifact

    
    def test_classification_model_has_classes(self, classification_artifact):
        model = classification_artifact["model"]
        assert hasattr(model, "classes_")

    def test_classification_model_has_predict_method(self, classification_artifact):
        model = classification_artifact["model"]
        assert hasattr(model, "predict")

    def test_classification_classes_match_expected(
        self, classification_artifact, expected_classification_classes
    ):
        
        classes = list(classification_artifact["model"].classes_)
        assert sorted(classes) == sorted(expected_classification_classes)
        


class TestClassificationModelPrediction:

    def test_classification_prediction_returns_array(
        self, classification_artifact, sample_classification_features
    ):
        model = classification_artifact["model"]
        scaler = classification_artifact.get("scaler")
        features = sample_classification_features
        if scaler is not None:
            features = scaler.transform(features)
        prediction = model.predict(features)
        assert isinstance(prediction, np.ndarray)

    def test_classification_prediction_shape(
        self, classification_artifact, sample_classification_features
    ):
        # one class per sample
        model = classification_artifact["model"]
        scaler = classification_artifact.get("scaler")
        features = sample_classification_features
        if scaler is not None:
            features = scaler.transform(features)
        prediction = model.predict(features)
        assert prediction.shape[0] == len(sample_classification_features)

    def test_classification_prediction_is_valid_class(
        self, classification_artifact, sample_classification_features,
        expected_classification_classes
    ):
        # should be a valid class
        model = classification_artifact["model"]
        scaler = classification_artifact.get("scaler")
        features = sample_classification_features
        if scaler is not None:
            features = scaler.transform(features)
        prediction = model.predict(features)[0]
        assert prediction in expected_classification_classes


class TestModelArtifactStructure:

    def test_regression_artifact_has_metrics(self, regression_artifact):
        assert "test_metrics" in regression_artifact

    
    #def test_classification_artifact_has_metrics(self, classification_artifact):
        #assert "test_metrics" in classification_artifact
        
    def test_classification_artifact_has_metrics(self, classification_artifact):
        assert "test_performance" in classification_artifact  # match actual key

    def test_regression_metrics_has_r2(self, regression_artifact):
        metrics = regression_artifact.get("test_metrics", {})
        assert "r2" in metrics

    def test_regression_r2_is_positive(self, regression_artifact):
        metrics = regression_artifact.get("test_metrics", {})
        r2 = metrics.get("r2", 0)
        assert r2 > 0

class TestErrorHandling:

    def test_load_nonexistent_model_raises_error(self, repo_root):
        fake_path = os.path.join(repo_root, "nonexistent_model.pkl")
        with pytest.raises(FileNotFoundError):
            with open(fake_path, "rb") as f:
                pickle.load(f)

    def test_regression_model_with_wrong_features_raises(
        self, regression_artifact
    ):
        import pandas as pd
        model = regression_artifact["model"]
        wrong_features = pd.DataFrame({"wrong_feature": [0.5]})
        
        with pytest.raises((ValueError, KeyError)):
            model.predict(wrong_features)
