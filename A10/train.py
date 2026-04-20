"""
A10 Training Module
===================
Training pipeline for 2D Pose to 3D Kinect mapping models.

This module provides:
- K-Fold cross-validation training
- Hyperparameter search
- Early stopping
- Training logging and metrics
- Per-joint evaluation

Issue #40 - A10: 2D Pose Estimation to 3D Mapping - Deep Learning Pipeline
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data_loader import (
    load_all_kinect_sequences,
    flatten_sequences,
    make_windowed_sequences,
    create_cv_splits,
    get_fold_data,
    DataNormalizer,
    KINECT_JOINTS,
    N_INPUT,
    N_OUTPUT_Z,
)
from models import (
    build_dense_model,
    build_conv1d_model,
    build_lstm_model,
    build_gru_model,
    compile_model,
    build_model_from_params,
)


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    joints: List[str] = None
) -> Dict:
    """
    Compute evaluation metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        joints: Joint names for per-joint analysis
        
    Returns:
        Dictionary of metrics
    """
    joints = joints or KINECT_JOINTS
    
    metrics = {
        'mse': float(mean_squared_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
    }
    
    # Per-joint metrics (for z-only output with 13 joints)
    if y_true.shape[1] == len(joints):
        per_joint = {}
        for i, joint in enumerate(joints):
            per_joint[joint] = {
                'mse': float(mean_squared_error(y_true[:, i], y_pred[:, i])),
                'mae': float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
                'r2': float(r2_score(y_true[:, i], y_pred[:, i])),
            }
        metrics['per_joint'] = per_joint
    
    return metrics


def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    normalizer: DataNormalizer = None
) -> Dict:
    """
    Evaluate a trained model.
    
    Args:
        model: Trained Keras model
        X_test: Test input data
        Y_test: Test target data
        normalizer: Optional normalizer for inverse transform
        
    Returns:
        Evaluation metrics dictionary
    """
    Y_pred = model.predict(X_test, verbose=0)
    
    # Inverse transform if normalizer provided
    if normalizer is not None:
        Y_test = normalizer.inverse_transform_output(Y_test)
        Y_pred = normalizer.inverse_transform_output(Y_pred)
    
    return compute_metrics(Y_test, Y_pred)


# =============================================================================
# Training Callbacks
# =============================================================================

def get_callbacks(
    model_path: str = None,
    early_stopping_patience: int = 10,
    reduce_lr_patience: int = 5,
    min_delta: float = 1e-4
) -> List:
    """
    Get training callbacks.
    
    Args:
        model_path: Path to save best model
        early_stopping_patience: Epochs to wait before early stopping
        reduce_lr_patience: Epochs to wait before reducing LR
        min_delta: Minimum change to qualify as improvement
        
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            min_delta=min_delta,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        ),
    ]
    
    if model_path:
        callbacks.append(
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        )
    
    return callbacks


# =============================================================================
# Cross-Validation Training
# =============================================================================

def train_with_cv(
    sequences: List[Tuple],
    model_type: str,
    model_config: Dict,
    optimizer: str = 'adam',
    learning_rate: float = 0.001,
    loss: str = 'mse',
    epochs: int = 100,
    batch_size: int = 32,
    n_folds: int = 5,
    output_type: str = 'z',
    normalize: bool = True,
    early_stopping_patience: int = 10,
    verbose: int = 1,
    results_dir: str = None
) -> Dict:
    """
    Train model with K-Fold cross-validation.
    
    Args:
        sequences: List of data sequences
        model_type: 'dense', 'conv1d', 'lstm', 'gru'
        model_config: Model hyperparameters
        optimizer: Optimizer name
        learning_rate: Learning rate
        loss: Loss function
        epochs: Maximum epochs
        batch_size: Batch size
        n_folds: Number of CV folds
        output_type: 'z' or 'xyz'
        normalize: Whether to normalize data
        early_stopping_patience: Early stopping patience
        verbose: Verbosity level
        results_dir: Directory to save results
        
    Returns:
        Dictionary with CV results
    """
    output_dim = N_OUTPUT_Z if output_type == 'z' else N_OUTPUT_Z * 3
    splits = create_cv_splits(sequences, n_folds=n_folds)
    
    fold_results = []
    histories = []
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*50}")
        
        # Get fold data
        X_train, Y_train, X_test, Y_test = get_fold_data(
            sequences, train_idx, test_idx, output_type
        )
        
        # Normalize
        normalizer = None
        if normalize:
            normalizer = DataNormalizer(method='standard')
            X_train, Y_train = normalizer.fit_transform(X_train, Y_train)
            X_test, Y_test = normalizer.transform(X_test, Y_test)
        
        # For sequence models, create windowed data
        if model_type in ['conv1d', 'lstm', 'gru']:
            window_size = model_config.get('window_size', 30)
            train_seqs = [sequences[i] for i in train_idx]
            test_seqs = [sequences[i] for i in test_idx]
            X_train, Y_train = make_windowed_sequences(
                train_seqs, window_size=window_size, output_type=output_type
            )
            X_test, Y_test = make_windowed_sequences(
                test_seqs, window_size=window_size, output_type=output_type
            )
            
            # For sequence models, we need the middle frame output
            if len(Y_train.shape) == 3:
                Y_train = Y_train[:, window_size // 2, :]
                Y_test = Y_test[:, window_size // 2, :]
        
        # Build model
        model = build_model_from_params(
            model_type=model_type,
            n_layers=model_config.get('n_layers', 2),
            n_units=model_config.get('n_units', 64),
            dropout_rate=model_config.get('dropout_rate', 0.2),
            window_size=model_config.get('window_size', 30),
            output_dim=output_dim
        )
        compile_model(model, optimizer=optimizer, learning_rate=learning_rate, loss=loss)
        
        # Callbacks
        callbacks = get_callbacks(
            early_stopping_patience=early_stopping_patience
        )
        
        # Train
        history = model.fit(
            X_train, Y_train,
            validation_data=(X_test, Y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        histories.append(history.history)
        
        # Evaluate
        metrics = evaluate_model(model, X_test, Y_test, normalizer)
        metrics['fold'] = fold + 1
        metrics['epochs_trained'] = len(history.history['loss'])
        fold_results.append(metrics)
        
        print(f"Fold {fold + 1} - MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")
        
        # Clean up
        keras.backend.clear_session()
    
    # Aggregate results
    cv_results = {
        'model_type': model_type,
        'model_config': model_config,
        'optimizer': optimizer,
        'learning_rate': learning_rate,
        'loss': loss,
        'epochs': epochs,
        'batch_size': batch_size,
        'n_folds': n_folds,
        'fold_results': fold_results,
        'mean_mse': float(np.mean([r['mse'] for r in fold_results])),
        'std_mse': float(np.std([r['mse'] for r in fold_results])),
        'mean_mae': float(np.mean([r['mae'] for r in fold_results])),
        'std_mae': float(np.std([r['mae'] for r in fold_results])),
        'mean_r2': float(np.mean([r['r2'] for r in fold_results])),
        'std_r2': float(np.std([r['r2'] for r in fold_results])),
    }
    
    print(f"\nCV Results: MSE={cv_results['mean_mse']:.4f}±{cv_results['std_mse']:.4f}, "
          f"MAE={cv_results['mean_mae']:.4f}±{cv_results['std_mae']:.4f}, "
          f"R2={cv_results['mean_r2']:.4f}±{cv_results['std_r2']:.4f}")
    
    # Save results
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(results_dir, f"cv_{model_type}_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(cv_results, f, indent=2)
        print(f"Results saved to: {results_path}")
    
    return cv_results


# =============================================================================
# Hyperparameter Search
# =============================================================================

def run_hyperparameter_search(
    sequences: List[Tuple],
    model_types: List[str] = None,
    optimizers: List[str] = None,
    learning_rates: List[float] = None,
    n_layers_options: List[int] = None,
    n_units_options: List[int] = None,
    n_folds: int = 3,
    epochs: int = 50,
    batch_size: int = 32,
    results_dir: str = 'cv_results',
    verbose: int = 0
) -> pd.DataFrame:
    """
    Run systematic hyperparameter search.
    
    Args:
        sequences: Training sequences
        model_types: List of model types to try
        optimizers: List of optimizers
        learning_rates: List of learning rates
        n_layers_options: Number of layers options
        n_units_options: Number of units options
        n_folds: CV folds
        epochs: Max epochs per experiment
        batch_size: Batch size
        results_dir: Directory to save results
        verbose: Verbosity
        
    Returns:
        DataFrame with all results
    """
    # Defaults
    model_types = model_types or ['dense', 'conv1d', 'lstm', 'gru']
    optimizers = optimizers or ['adam', 'sgd', 'rmsprop']
    learning_rates = learning_rates or [0.001, 0.0001]
    n_layers_options = n_layers_options or [1, 2, 3]
    n_units_options = n_units_options or [32, 64, 128]
    
    os.makedirs(results_dir, exist_ok=True)
    all_results = []
    total_experiments = (len(model_types) * len(optimizers) * len(learning_rates) *
                        len(n_layers_options) * len(n_units_options))
    
    print(f"\nStarting hyperparameter search: {total_experiments} experiments")
    print(f"Model types: {model_types}")
    print(f"Optimizers: {optimizers}")
    print(f"Learning rates: {learning_rates}")
    print(f"Layers: {n_layers_options}")
    print(f"Units: {n_units_options}")
    print("=" * 60)
    
    experiment_idx = 0
    for model_type in model_types:
        for optimizer in optimizers:
            for lr in learning_rates:
                for n_layers in n_layers_options:
                    for n_units in n_units_options:
                        experiment_idx += 1
                        print(f"\n[{experiment_idx}/{total_experiments}] "
                              f"{model_type}, {optimizer}, lr={lr}, "
                              f"layers={n_layers}, units={n_units}")
                        
                        try:
                            result = train_with_cv(
                                sequences=sequences,
                                model_type=model_type,
                                model_config={
                                    'n_layers': n_layers,
                                    'n_units': n_units,
                                    'dropout_rate': 0.2,
                                    'window_size': 30,
                                },
                                optimizer=optimizer,
                                learning_rate=lr,
                                epochs=epochs,
                                batch_size=batch_size,
                                n_folds=n_folds,
                                early_stopping_patience=5,
                                verbose=verbose,
                                results_dir=None
                            )
                            
                            all_results.append({
                                'model_type': model_type,
                                'optimizer': optimizer,
                                'learning_rate': lr,
                                'n_layers': n_layers,
                                'n_units': n_units,
                                'mean_mse': result['mean_mse'],
                                'std_mse': result['std_mse'],
                                'mean_mae': result['mean_mae'],
                                'std_mae': result['std_mae'],
                                'mean_r2': result['mean_r2'],
                                'std_r2': result['std_r2'],
                            })
                            
                        except Exception as e:
                            print(f"Error: {e}")
                            all_results.append({
                                'model_type': model_type,
                                'optimizer': optimizer,
                                'learning_rate': lr,
                                'n_layers': n_layers,
                                'n_units': n_units,
                                'mean_mse': np.nan,
                                'std_mse': np.nan,
                                'mean_mae': np.nan,
                                'std_mae': np.nan,
                                'mean_r2': np.nan,
                                'std_r2': np.nan,
                                'error': str(e),
                            })
    
    # Create DataFrame and save
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('mean_mse', ascending=True)
    
    results_path = os.path.join(results_dir, 'hyperparam_search_results.csv')
    df_results.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Print top 5
    print("\nTop 5 configurations:")
    print(df_results.head().to_string(index=False))
    
    return df_results


# =============================================================================
# Final Model Training
# =============================================================================

def train_final_model(
    sequences: List[Tuple],
    model_type: str,
    model_config: Dict,
    optimizer: str = 'adam',
    learning_rate: float = 0.001,
    epochs: int = 200,
    batch_size: int = 32,
    test_split: float = 0.2,
    output_type: str = 'z',
    model_save_path: str = None,
    random_state: int = 42
) -> Tuple[keras.Model, Dict]:
    """
    Train final model on full training data with test holdout.
    
    Args:
        sequences: All data sequences
        model_type: Model type
        model_config: Model configuration
        optimizer: Optimizer
        learning_rate: Learning rate
        epochs: Training epochs
        batch_size: Batch size
        test_split: Fraction for test set
        output_type: 'z' or 'xyz'
        model_save_path: Path to save trained model
        random_state: Random seed
        
    Returns:
        Tuple of (trained model, metrics dictionary)
    """
    output_dim = N_OUTPUT_Z if output_type == 'z' else N_OUTPUT_Z * 3
    
    # Split sequences
    np.random.seed(random_state)
    n_seqs = len(sequences)
    indices = np.random.permutation(n_seqs)
    n_test = int(n_seqs * test_split)
    test_idx = indices[:n_test].tolist()
    train_idx = indices[n_test:].tolist()
    
    print(f"Training sequences: {len(train_idx)}")
    print(f"Test sequences: {len(test_idx)}")
    
    # Get data
    X_train, Y_train, X_test, Y_test = get_fold_data(
        sequences, train_idx, test_idx, output_type
    )
    
    # Normalize
    normalizer = DataNormalizer(method='standard')
    X_train, Y_train = normalizer.fit_transform(X_train, Y_train)
    X_test, Y_test = normalizer.transform(X_test, Y_test)
    
    # Handle sequence models
    if model_type in ['conv1d', 'lstm', 'gru']:
        window_size = model_config.get('window_size', 30)
        train_seqs = [sequences[i] for i in train_idx]
        test_seqs = [sequences[i] for i in test_idx]
        X_train, Y_train = make_windowed_sequences(
            train_seqs, window_size=window_size, output_type=output_type
        )
        X_test, Y_test = make_windowed_sequences(
            test_seqs, window_size=window_size, output_type=output_type
        )
        if len(Y_train.shape) == 3:
            Y_train = Y_train[:, window_size // 2, :]
            Y_test = Y_test[:, window_size // 2, :]
    
    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {Y_test.shape}")
    
    # Build model
    model = build_model_from_params(
        model_type=model_type,
        n_layers=model_config.get('n_layers', 2),
        n_units=model_config.get('n_units', 64),
        dropout_rate=model_config.get('dropout_rate', 0.2),
        window_size=model_config.get('window_size', 30),
        output_dim=output_dim
    )
    compile_model(model, optimizer=optimizer, learning_rate=learning_rate)
    model.summary()
    
    # Callbacks
    callbacks = get_callbacks(
        model_path=model_save_path,
        early_stopping_patience=15
    )
    
    # Train
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    metrics = evaluate_model(model, X_test, Y_test, normalizer)
    metrics['epochs_trained'] = len(history.history['loss'])
    metrics['final_train_loss'] = float(history.history['loss'][-1])
    metrics['final_val_loss'] = float(history.history['val_loss'][-1])
    
    print(f"\nFinal Results:")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R2:  {metrics['r2']:.4f}")
    
    # Save model
    if model_save_path:
        model.save(model_save_path)
        print(f"\nModel saved to: {model_save_path}")
    
    return model, metrics


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point for training pipeline."""
    print("A10 Training Pipeline")
    print("=" * 60)
    
    # Paths
    REPO_ROOT = Path(__file__).parent.parent
    KINECT_PATH = REPO_ROOT / 'kinect_good_preprocessed'
    RESULTS_DIR = Path(__file__).parent / 'cv_results'
    MODELS_DIR = Path(__file__).parent / 'models'
    
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Check for data
    if not KINECT_PATH.exists():
        print(f"ERROR: Kinect data not found at: {KINECT_PATH}")
        print("Please ensure the kinect_good_preprocessed folder exists.")
        return
    
    # Load data
    print(f"\nLoading data from: {KINECT_PATH}")
    sequences, file_names = load_all_kinect_sequences(str(KINECT_PATH))
    print(f"Loaded {len(sequences)} sequences")
    
    # Quick test with Dense model
    print("\n" + "=" * 60)
    print("Quick test with Dense model (3-fold CV)")
    print("=" * 60)
    
    result = train_with_cv(
        sequences=sequences,
        model_type='dense',
        model_config={
            'n_layers': 2,
            'n_units': 64,
            'dropout_rate': 0.2,
        },
        optimizer='adam',
        learning_rate=0.001,
        epochs=50,
        batch_size=32,
        n_folds=3,
        early_stopping_patience=5,
        verbose=0,
        results_dir=str(RESULTS_DIR)
    )
    
    # Full hyperparameter search (optional - takes time)
    # Uncomment to run full search
    """
    print("\n" + "=" * 60)
    print("Running full hyperparameter search")
    print("=" * 60)
    
    search_results = run_hyperparameter_search(
        sequences=sequences,
        model_types=['dense', 'conv1d'],
        optimizers=['adam', 'rmsprop'],
        learning_rates=[0.001, 0.0001],
        n_layers_options=[1, 2],
        n_units_options=[64, 128],
        n_folds=3,
        epochs=50,
        results_dir=str(RESULTS_DIR),
        verbose=0
    )
    """
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
