"""Inference-model helpers used when exporting trained models."""

import numpy as np
import tensorflow as tf


def _float32_clone_layer(layer):
    """Clone a layer (including nested models) with float32 computation."""
    if isinstance(layer, tf.keras.Model):
        return _clone_float32_model(layer)

    config = layer.get_config()
    if "dtype" in config:
        config["dtype"] = "float32"
    return layer.__class__.from_config(config)


def _clone_float32_model(model):
    previous_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy("float32")
    try:
        clone = tf.keras.models.clone_model(
            model, clone_function=_float32_clone_layer
        )
        # A nested Model can autocast its inputs using its own policy even when
        # all its constituent layers have been cloned with float32 policies.
        clone._dtype_policy = tf.keras.mixed_precision.Policy("float32")
        clone.set_weights(model.get_weights())
    finally:
        tf.keras.mixed_precision.set_global_policy(previous_policy)
    return clone


def make_float32_inference_model(model, rtol=2e-2, atol=2e-3):
    """Return a float32 clone with the trained weights and matching predictions.

    Mixed-float16 training models can retain float16 operations in their graph,
    which the built-in TFLite converter cannot lower. This clone preserves the
    model topology and weights while rebuilding every layer for float32.
    """
    if len(model.inputs) != 1:
        raise ValueError("TFLite export currently supports single-input models only.")

    clone = _clone_float32_model(model)
    input_shape = tuple(model.inputs[0].shape)
    if any(dimension is None for dimension in input_shape[1:]):
        raise ValueError(
            "The model must have a fixed non-batch input shape for TFLite export."
        )
    sample_shape = (1,) + tuple(int(dimension) for dimension in input_shape[1:])
    sample = tf.random.uniform(sample_shape, dtype=tf.float32)

    expected = model(sample, training=False)
    actual = clone(sample, training=False)
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=rtol,
        atol=atol,
        err_msg="Float32 export clone predictions differ from the trained model.",
    )
    return clone
