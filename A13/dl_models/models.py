"""Model factories for Issue #10.

Two architectures are provided per problem:

* :func:`build_dense` -- multi-layer perceptron over the flattened sequence.
* :func:`build_cnn`   -- small Conv2D-over-(time, joint) network. The default
  hyper-parameters were chosen so that the CNN has at most ~20 % of the
  parameters of the Dense baseline (verified by :func:`assert_param_budget`).

All models output a single sigmoid logit (good=1 / bad=0) and are compiled
with ``binary_crossentropy`` plus the metrics required by issue #10:
True/False Positives & Negatives, AUC, BinaryAccuracy, Precision, Recall.
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


# --------------------------------------------------------------------------- #
# Metrics & compile helper                                                    #
# --------------------------------------------------------------------------- #
def make_metrics() -> list[tf.keras.metrics.Metric]:
    return [
        tf.keras.metrics.TruePositives(name="tp"),
        tf.keras.metrics.FalsePositives(name="fp"),
        tf.keras.metrics.TrueNegatives(name="tn"),
        tf.keras.metrics.FalseNegatives(name="fn"),
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=make_metrics(),
    )
    return model


# --------------------------------------------------------------------------- #
# Architectures                                                               #
# --------------------------------------------------------------------------- #
def build_dense(
    input_dim: int,
    hidden_units: Sequence[int] = (128, 64, 32),
    dropout: float = 0.3,
    l2: float = 1e-4,
    learning_rate: float = 1e-3,
    name: str = "dense",
) -> tf.keras.Model:
    """MLP for flattened sequences (Dense approach)."""

    reg = regularizers.l2(l2) if l2 else None
    inputs = layers.Input(shape=(input_dim,), name="features")
    x = layers.BatchNormalization()(inputs)
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation="relu", kernel_regularizer=reg, name=f"fc{i+1}")(x)
        if dropout:
            x = layers.Dropout(dropout)(x)
    output = layers.Dense(1, activation="sigmoid", name="prob")(x)
    return compile_model(models.Model(inputs, output, name=name), learning_rate)


def build_cnn(
    input_shape: tuple[int, int, int],
    filters: Sequence[int] = (8, 16),
    kernel_size: tuple[int, int] = (3, 3),
    dense_units: int = 16,
    dropout: float = 0.3,
    l2: float = 1e-4,
    learning_rate: float = 1e-3,
    name: str = "cnn",
) -> tf.keras.Model:
    """Compact 2D CNN over (time, joint, coordinate) tensors.

    The default ``filters`` and ``dense_units`` produce <20 % of the Dense
    baseline's parameters for both problem A and problem B.
    """

    reg = regularizers.l2(l2) if l2 else None
    inputs = layers.Input(shape=input_shape, name="sequence")
    x = layers.BatchNormalization()(inputs)
    for i, f in enumerate(filters):
        x = layers.Conv2D(
            f, kernel_size=kernel_size, padding="same", activation="relu",
            kernel_regularizer=reg, name=f"conv{i+1}",
        )(x)
        # only pool on the time axis; joint axis is small (13).
        x = layers.MaxPool2D(pool_size=(2, 1), name=f"pool{i+1}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    if dense_units:
        x = layers.Dense(dense_units, activation="relu", kernel_regularizer=reg, name="fc")(x)
        if dropout:
            x = layers.Dropout(dropout)(x)
    output = layers.Dense(1, activation="sigmoid", name="prob")(x)
    return compile_model(models.Model(inputs, output, name=name), learning_rate)


# --------------------------------------------------------------------------- #
# Parameter budget                                                            #
# --------------------------------------------------------------------------- #
def count_params(model: tf.keras.Model) -> int:
    return int(model.count_params())


def assert_param_budget(dense: tf.keras.Model, cnn: tf.keras.Model, ratio: float = 0.20) -> None:
    """Raise if the CNN exceeds ``ratio`` × Dense parameter count."""
    d, c = count_params(dense), count_params(cnn)
    if c > ratio * d:
        raise AssertionError(
            f"CNN has {c} parameters which exceeds {ratio:.0%} of Dense's {d} "
            f"({c / d:.1%}). Reduce CNN filters/dense_units."
        )
