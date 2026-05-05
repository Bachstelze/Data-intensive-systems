## A12 service endpoint

Endpoint alternative chosen: **Gradio interface tab inside the existing `app.py`**.

This was chosen because the project is already deployed as a Gradio HuggingFace Space. A Gradio tab keeps deployment simple, avoids maintaining a separate FastAPI/Flask service, and is easy for team members to test manually in the existing UI.

### Input contract

The A12 tab accepts a pose-feature `.csv` file.

- Problem A expects Kinect 3D pose columns from `A12_classifier.py`.
- Problem B expects PoseNet 2D pose columns from `A12_classifier.py`.
- Both problems also expect the engineered distance, velocity, and acceleration columns used during training.

The endpoint intentionally fails fast if required columns are missing, because silently filling missing ML features would make predictions misleading.

### Output contract

The tab returns structured JSON:

```json
{
  "status": "ok",
  "endpoint": "Gradio tab inside app.py",
  "problem": "B",
  "model_name": "B_PoseNet_Dense_relu_adam_bs64",
  "model_version": "A12 Dense classifiers from A12_results",
  "metadata": {
    "rows": 100,
    "features": 57,
    "inference_time_ms": 25.1
  },
  "prediction": {
    "label": "exercise",
    "confidence": 0.91,
    "exercise_frame_ratio": 0.78
  },
  "frame_preview": []
}
```

On validation or model-loading errors, the same component returns:

```json
{
  "status": "error",
  "endpoint": "Gradio tab inside app.py",
  "problem": "B",
  "message": "CSV is missing required columns..."
}
```

### Local run

```bash
cd /Users/reemothman/Downloads/DIS/Data-intensive-systems
source .venv/bin/activate
python3 app.py
```

Then open the terminal link and choose the **A12 Classifier** tab.

### Tests

```bash
pytest A12/tests -v
```

### Rollback

If the endpoint breaks the HuggingFace Space, revert the commit that introduced the A12 tab and service files:

```bash
git revert <commit-sha>
git push origin main
```

If only the trained model is problematic, restore the previous known-good files in `A12/A12_results/` and push again.
