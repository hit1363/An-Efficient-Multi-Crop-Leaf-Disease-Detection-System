"""Inference helpers for reliable float32 TensorFlow Lite conversion."""

import numpy as np
import tensorflow as tf


def _clone_layer_float32(layer):
    if isinstance(layer, tf.keras.Model):
        return _clone_model_float32(layer)
    config = layer.get_config()
    if "dtype" in config:
        config["dtype"] = "float32"
    return layer.__class__.from_config(config)


def _clone_model_float32(model):
    previous_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy("float32")
    try:
        clone = tf.keras.models.clone_model(model, clone_function=_clone_layer_float32)
        clone._dtype_policy = tf.keras.mixed_precision.Policy("float32")
        clone.set_weights(model.get_weights())
    finally:
        tf.keras.mixed_precision.set_global_policy(previous_policy)
    return clone


def make_float32_inference_model(model, rtol=2e-2, atol=2e-3):
    """Clone mixed-precision models and verify prediction parity."""
    if len(model.inputs) != 1:
        raise ValueError("TFLite export currently supports single-input models only")
    clone = _clone_model_float32(model)
    input_shape = tuple(model.inputs[0].shape)
    if any(size is None for size in input_shape[1:]):
        raise ValueError("TFLite export requires a fixed image input shape")
    sample = tf.random.uniform((1,) + tuple(int(size) for size in input_shape[1:]))
    expected = model(sample, training=False)
    actual = clone(sample, training=False)
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol,
        err_msg="Float32 export clone predictions differ from the trained model",
    )
    return clone
