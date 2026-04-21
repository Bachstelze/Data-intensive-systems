"""
Integration examples for A10/visualizer.py

This file is intentionally small and copy-paste friendly.
It shows exactly what to add after model prediction / evaluation.
"""

from pathlib import Path

from visualizer import save_prediction_bundle, create_evaluation_visuals


def integrate_after_evaluate_model(model, X_test, Y_test, normalizer, results_dir, run_name='demo_run'):
    """
    Example for teammate #2 one-step pipeline.

    Assumes:
    - model predicts xyz output with 39 values per frame
    - X_test and Y_test are already the test tensors
    - normalizer.output_scaler exists and was fit on the training targets
    """
    pred_norm = model.predict(X_test, verbose=0)
    pred_xyz = normalizer.inverse_transform_output(pred_norm)
    gt_xyz = normalizer.inverse_transform_output(Y_test)

    bundle_dir = Path(results_dir) / run_name
    save_prediction_bundle(
        output_dir=bundle_dir,
        predicted_xyz=pred_xyz,
        ground_truth_xyz=gt_xyz,
        metadata={'run_name': run_name, 'source': 'one_step_model'},
    )
    create_evaluation_visuals(bundle_dir)
    return bundle_dir


TRAIN_PATCH_SNIPPET = r'''
# Add this import near the top of train.py
from visualizer import save_prediction_bundle, create_evaluation_visuals

# Add this near the end of train_final_model(...) after metrics are computed
Y_pred = model.predict(X_test, verbose=0)
if normalizer is not None:
    Y_pred_vis = normalizer.inverse_transform_output(Y_pred)
    Y_test_vis = normalizer.inverse_transform_output(Y_test)
else:
    Y_pred_vis = Y_pred
    Y_test_vis = Y_test

vis_dir = Path(__file__).parent / 'visualization_runs' / f'{model_type}_{output_type}'
save_prediction_bundle(
    output_dir=vis_dir,
    predicted_xyz=Y_pred_vis,
    ground_truth_xyz=Y_test_vis,
    metadata={
        'model_type': model_type,
        'output_type': output_type,
        'optimizer': optimizer,
        'learning_rate': learning_rate,
    },
)

# Only call this when output_type == 'xyz', otherwise there is no 3D skeleton to draw
if output_type == 'xyz':
    create_evaluation_visuals(vis_dir)
'''


if __name__ == '__main__':
    print(TRAIN_PATCH_SNIPPET)
