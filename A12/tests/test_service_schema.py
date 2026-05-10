from A12.service.schemas import ClassificationResult, PipelineMetadata, PipelineOutput


def test_pipeline_output_schema():
    payload = PipelineOutput(
        annotated_video_path="out.mp4",
        animation_data_path="anim.json",
        keypoints_csv_path="points.csv",
        classification=ClassificationResult(
            label="good",
            is_good=True,
            confidence=0.8,
            probabilities={"good": 0.8, "bad": 0.2},
        ),
        metadata=PipelineMetadata(
            model_version="test",
            inference_time_ms=1.0,
            frame_count_original=20,
            frame_count_cut=10,
            fps=30.0,
            cut_start_frame=2,
            cut_end_frame=11,
            smoothing_strategy="exponential/zscore",
            classifier_mode="dummy",
        ),
        warnings=[],
    ).to_json()
    assert payload["classification"]["label"] in {"good", "bad"}
    assert set(payload["classification"]["probabilities"]) == {"good", "bad"}
    assert payload["metadata"]["frame_count_cut"] == 10
