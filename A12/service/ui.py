"""Small UI helpers for the A12 Gradio tab."""

from __future__ import annotations

from typing import Any, Dict

from A12.service.model_service import safe_predict_pose_csv


def run_a12_tab(csv_file: str | None, problem: str) -> tuple[Dict[str, Any], str]:
    """Run prediction and return JSON plus a concise Markdown summary."""
    result = safe_predict_pose_csv(csv_file, problem)
    if result.get("status") != "ok":
        return result, f"### Prediction failed\n\n{result.get('message', 'Unknown error')}"

    prediction = result["prediction"]
    metadata = result["metadata"]
    summary = f"""### A12 prediction

- **Problem:** {result['problem']}
- **Model:** {result['model_name']}
- **Overall label:** {prediction['label']}
- **Confidence:** {prediction['confidence']:.3f}
- **Exercise frame ratio:** {prediction['exercise_frame_ratio']:.3f}
- **Rows processed:** {metadata['rows']}
- **Inference time:** {metadata['inference_time_ms']} ms
"""
    return result, summary
