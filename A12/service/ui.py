from A12.service.pipeline import run_full_pipeline


def run_a12_video_tab(video_path, confidence, smoothing_strategy, smoothing_method):
    """
    Gradio UI callback
    """

    try:
        result = run_full_pipeline(video_path)

        annotated_video = result["annotated_video"]
        animation_video = result["animation_video"]
        csv_path = result["csv_path"]

        classification = result.get("classification", {})
        label = classification.get("label", "unknown")
        conf = classification.get("confidence", 0.0)

        summary = f"""
### Pipeline Results

- Classification: **{label}**
- Confidence: **{conf:.2f}**

Outputs:
- Annotated 2D video
- 3D skeleton animation video
- CSV joint data
"""

        return (
            animation_video,
            csv_path,
            result,
            summary
        )

    except Exception as e:
        return None, None, None, {"error": str(e)}, f"### Error\n{e}"