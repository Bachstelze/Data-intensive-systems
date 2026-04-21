"""
A10 Models Module
=================
Deep Learning model architectures for 2D Pose to 3D Kinect mapping.

This module provides:
- Dense (MLP) networks for frame-level prediction
- Conv1D networks for sequence data
- LSTM/GRU recurrent networks for temporal modeling
- Model factory functions for hyperparameter search

Issue #40 - A10: 2D Pose Estimation to 3D Mapping - Deep Learning Pipeline
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers


# =============================================================================
# Model Constants
# =============================================================================

N_JOINTS = 13                    # Number of Kinect joints
N_INPUT = N_JOINTS * 2           # Input: 26 features (PoseNet x,y per joint)
N_OUTPUT_XY = N_JOINTS * 2       # Output: 26 features (Kinect x,y) -- Issue #40 PRIMARY
N_OUTPUT_Z = N_JOINTS            # Output: 13 z-coordinates (legacy)
N_OUTPUT_XYZ = N_JOINTS * 3      # Output: 39 coordinates (Issue #41 one-step variant)

# Default output dimension follows Issue #40: 2D -> 2D (26 features)
DEFAULT_OUTPUT_DIM = N_OUTPUT_XY

DEFAULT_WINDOW_SIZE = 30         # Default sequence length for Conv1D/RNN


# =============================================================================
# Dense (MLP) Model
# =============================================================================

def build_dense_model(
    hidden_units: Tuple[int, ...] = (128, 64),
    activation: str = 'relu',
    dropout_rate: float = 0.2,
    l2_reg: float = 1e-4,
    output_dim: int = DEFAULT_OUTPUT_DIM,
    name: str = 'DenseModel'
) -> keras.Model:
    """
    Build a Dense (fully connected) neural network for frame-level prediction.
    
    Maps 2D keypoints (26 features) to 3D coordinates per frame.
    
    Args:
        hidden_units: Tuple of hidden layer sizes (e.g., (128, 64))
        activation: Activation function ('relu', 'tanh', 'elu')
        dropout_rate: Dropout rate between layers
        l2_reg: L2 regularization coefficient
        output_dim: Output dimension (13 for z-only, 39 for xyz)
        name: Model name
        
    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=(N_INPUT,), name='xy_input')
    x = inputs
    
    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None,
            name=f'dense_{i+1}'
        )(x)
        
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
    
    outputs = layers.Dense(output_dim, activation='linear', name='output')(x)
    
    model = keras.Model(inputs, outputs, name=name)
    return model


# =============================================================================
# Conv1D Model
# =============================================================================

def build_conv1d_model(
    window_size: int = DEFAULT_WINDOW_SIZE,
    filters: Tuple[int, ...] = (64, 128),
    kernel_size: int = 3,
    pool_size: int = 2,
    dense_units: Tuple[int, ...] = (64,),
    activation: str = 'relu',
    dropout_rate: float = 0.2,
    output_dim: int = DEFAULT_OUTPUT_DIM,
    name: str = 'Conv1DModel'
) -> keras.Model:
    """
    Build a Conv1D network for sequence-based prediction.
    
    Processes windows of frames using 1D convolutions.
    
    Args:
        window_size: Number of frames in input sequence
        filters: Tuple of filter counts per Conv1D layer
        kernel_size: Convolution kernel size
        pool_size: MaxPooling pool size
        dense_units: Dense layer sizes after convolution
        activation: Activation function
        dropout_rate: Dropout rate
        output_dim: Output dimension
        name: Model name
        
    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=(window_size, N_INPUT), name='xy_seq_input')
    x = inputs
    
    # Convolutional layers
    for i, f in enumerate(filters):
        x = layers.Conv1D(
            f, kernel_size, activation=activation, padding='same',
            name=f'conv_{i+1}'
        )(x)
        x = layers.MaxPooling1D(pool_size, padding='same', name=f'pool_{i+1}')(x)
        
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f'drop_conv_{i+1}')(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    
    # Dense layers
    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation=activation, name=f'fc_{i+1}')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f'drop_fc_{i+1}')(x)
    
    outputs = layers.Dense(output_dim, activation='linear', name='output')(x)
    
    model = keras.Model(inputs, outputs, name=name)
    return model


# =============================================================================
# LSTM Model
# =============================================================================

def build_lstm_model(
    window_size: int = DEFAULT_WINDOW_SIZE,
    lstm_units: Tuple[int, ...] = (64, 32),
    dense_units: Tuple[int, ...] = (32,),
    activation: str = 'tanh',
    dropout_rate: float = 0.2,
    recurrent_dropout: float = 0.0,
    output_dim: int = DEFAULT_OUTPUT_DIM,
    name: str = 'LSTMModel'
) -> keras.Model:
    """
    Build an LSTM network for sequence prediction.
    
    Uses recurrent layers to capture temporal dependencies.
    
    Args:
        window_size: Number of frames in input sequence
        lstm_units: Tuple of LSTM layer sizes
        dense_units: Dense layer sizes after LSTM
        activation: LSTM activation function
        dropout_rate: Dropout rate
        recurrent_dropout: LSTM recurrent dropout
        output_dim: Output dimension
        name: Model name
        
    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=(window_size, N_INPUT), name='xy_seq_input')
    x = inputs
    
    # LSTM layers
    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)  # Only last LSTM returns single output
        x = layers.LSTM(
            units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            name=f'lstm_{i+1}'
        )(x)
    
    # Dense layers
    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation='relu', name=f'fc_{i+1}')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f'drop_fc_{i+1}')(x)
    
    outputs = layers.Dense(output_dim, activation='linear', name='output')(x)
    
    model = keras.Model(inputs, outputs, name=name)
    return model


# =============================================================================
# GRU Model
# =============================================================================

def build_gru_model(
    window_size: int = DEFAULT_WINDOW_SIZE,
    gru_units: Tuple[int, ...] = (64, 32),
    dense_units: Tuple[int, ...] = (32,),
    dropout_rate: float = 0.2,
    recurrent_dropout: float = 0.0,
    output_dim: int = DEFAULT_OUTPUT_DIM,
    name: str = 'GRUModel'
) -> keras.Model:
    """
    Build a GRU network for sequence prediction.
    
    Similar to LSTM but with simpler gating mechanism.
    
    Args:
        window_size: Number of frames in input sequence
        gru_units: Tuple of GRU layer sizes
        dense_units: Dense layer sizes after GRU
        dropout_rate: Dropout rate
        recurrent_dropout: GRU recurrent dropout
        output_dim: Output dimension
        name: Model name
        
    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=(window_size, N_INPUT), name='xy_seq_input')
    x = inputs
    
    # GRU layers
    for i, units in enumerate(gru_units):
        return_sequences = (i < len(gru_units) - 1)
        x = layers.GRU(
            units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            name=f'gru_{i+1}'
        )(x)
    
    # Dense layers
    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation='relu', name=f'fc_{i+1}')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f'drop_fc_{i+1}')(x)
    
    outputs = layers.Dense(output_dim, activation='linear', name='output')(x)
    
    model = keras.Model(inputs, outputs, name=name)
    return model


# =============================================================================
# Model Factory
# =============================================================================

MODEL_BUILDERS = {
    'dense': build_dense_model,
    'conv1d': build_conv1d_model,
    'lstm': build_lstm_model,
    'gru': build_gru_model,
}


def create_model(
    model_type: str,
    config: Dict,
    output_dim: int = DEFAULT_OUTPUT_DIM
) -> keras.Model:
    """
    Factory function to create models by type and configuration.
    
    Args:
        model_type: One of 'dense', 'conv1d', 'lstm', 'gru'
        config: Dictionary of model hyperparameters
        output_dim: Output dimension
        
    Returns:
        Keras model
    """
    if model_type not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from: {list(MODEL_BUILDERS.keys())}")
    
    builder = MODEL_BUILDERS[model_type]
    return builder(**config, output_dim=output_dim)


def compile_model(
    model: keras.Model,
    optimizer: str = 'adam',
    learning_rate: float = 0.001,
    loss: str = 'mse'
) -> keras.Model:
    """
    Compile a model with specified optimizer and loss.
    
    Args:
        model: Keras model to compile
        optimizer: Optimizer name ('adam', 'sgd', 'rmsprop')
        learning_rate: Learning rate
        loss: Loss function ('mse', 'mae', 'huber')
        
    Returns:
        Compiled model
    """
    # Create optimizer
    if optimizer.lower() == 'adam':
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'sgd':
        opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer.lower() == 'rmsprop':
        opt = optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    # Compile
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=['mae']
    )
    
    return model


# =============================================================================
# Model Configurations for Hyperparameter Search
# =============================================================================

# Default configurations for each model type
DEFAULT_CONFIGS = {
    'dense': {
        'hidden_units': (128, 64),
        'activation': 'relu',
        'dropout_rate': 0.2,
        'l2_reg': 1e-4,
    },
    'conv1d': {
        'window_size': 30,
        'filters': (64, 128),
        'kernel_size': 3,
        'pool_size': 2,
        'dense_units': (64,),
        'activation': 'relu',
        'dropout_rate': 0.2,
    },
    'lstm': {
        'window_size': 30,
        'lstm_units': (64, 32),
        'dense_units': (32,),
        'activation': 'tanh',
        'dropout_rate': 0.2,
        'recurrent_dropout': 0.0,
    },
    'gru': {
        'window_size': 30,
        'gru_units': (64, 32),
        'dense_units': (32,),
        'dropout_rate': 0.2,
        'recurrent_dropout': 0.0,
    },
}


def get_hyperparameter_grid() -> Dict:
    """
    Get the hyperparameter grid for systematic search.
    
    Returns:
        Dictionary of hyperparameter options
    """
    return {
        'model_type': ['dense', 'conv1d', 'lstm', 'gru'],
        'optimizer': ['adam', 'sgd', 'rmsprop'],
        'learning_rate': [0.001, 0.0001],
        'hidden_layers': [1, 2, 3],
        'hidden_units': [32, 64, 128],
        'epochs': [50, 100, 200],
        'batch_size': [32, 64],
        'dropout_rate': [0.1, 0.2, 0.3],
    }


def build_model_from_params(
    model_type: str,
    n_layers: int,
    n_units: int,
    dropout_rate: float = 0.2,
    window_size: int = 30,
    output_dim: int = DEFAULT_OUTPUT_DIM
) -> keras.Model:
    """
    Build a model from simplified hyperparameters.
    
    Args:
        model_type: 'dense', 'conv1d', 'lstm', 'gru'
        n_layers: Number of hidden layers
        n_units: Units per layer
        dropout_rate: Dropout rate
        window_size: Sequence length (for sequence models)
        output_dim: Output dimension
        
    Returns:
        Keras model
    """
    units_tuple = tuple([n_units] * n_layers)
    
    if model_type == 'dense':
        return build_dense_model(
            hidden_units=units_tuple,
            dropout_rate=dropout_rate,
            output_dim=output_dim
        )
    elif model_type == 'conv1d':
        return build_conv1d_model(
            window_size=window_size,
            filters=units_tuple,
            dense_units=(n_units // 2,),
            dropout_rate=dropout_rate,
            output_dim=output_dim
        )
    elif model_type == 'lstm':
        return build_lstm_model(
            window_size=window_size,
            lstm_units=units_tuple,
            dense_units=(n_units // 2,),
            dropout_rate=dropout_rate,
            output_dim=output_dim
        )
    elif model_type == 'gru':
        return build_gru_model(
            window_size=window_size,
            gru_units=units_tuple,
            dense_units=(n_units // 2,),
            dropout_rate=dropout_rate,
            output_dim=output_dim
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# Demo
# =============================================================================

if __name__ == '__main__':
    print("Testing model architectures...\n")
    
    # Test Dense model
    print("=" * 50)
    print("Dense Model")
    print("=" * 50)
    dense_model = build_dense_model(hidden_units=(128, 64))
    compile_model(dense_model, optimizer='adam')
    dense_model.summary()
    
    # Test Conv1D model
    print("\n" + "=" * 50)
    print("Conv1D Model")
    print("=" * 50)
    conv_model = build_conv1d_model(window_size=30, filters=(64, 128))
    compile_model(conv_model, optimizer='adam')
    conv_model.summary()
    
    # Test LSTM model
    print("\n" + "=" * 50)
    print("LSTM Model")
    print("=" * 50)
    lstm_model = build_lstm_model(window_size=30, lstm_units=(64, 32))
    compile_model(lstm_model, optimizer='adam')
    lstm_model.summary()
    
    # Test GRU model
    print("\n" + "=" * 50)
    print("GRU Model")
    print("=" * 50)
    gru_model = build_gru_model(window_size=30, gru_units=(64, 32))
    compile_model(gru_model, optimizer='adam')
    gru_model.summary()
    
    print("\nAll models created successfully!")
