"""Regression tests for the EfficientNet-Lite0 TF Hub dtype contract.

The Lite0 backbone is a TF1-format SavedModel whose input signature is fixed at
float32 at save time. Under the mixed_float16 policy used for GPU training a
plain hub.KerasLayer inherits the global policy, autocasts its input to float16
and fails inside the module with:

    expected to be a float tensor but is a half tensor

These tests build the real model factory against a fake hub module so they run
offline while still exercising the dtype plumbing.
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from training import model as model_module  # noqa: E402


_FLOAT16_ERROR = (
    "cannot compute __inference_pruned_21353 as input #0(zero-based) was "
    "expected to be a float tensor but is a half tensor "
    "[Op:__inference_pruned_21353]"
)


class _FixedFloat32SignatureLayer(tf.keras.layers.Layer):
    """Stand-in for a TF1 Hub module whose input dtype is frozen at float32."""

    def __init__(
        self, handle, input_shape=None, trainable=False, dtype=None, **kwargs
    ):
        super().__init__(trainable=trainable, dtype=dtype, **kwargs)
        self.handle = handle
        self.signature_input_shape = tuple(input_shape or ())
        self.observed_input_dtypes = []

    def build(self, input_shape):
        flat = int(np.prod(self.signature_input_shape))
        self._kernel = self.add_weight(
            name="kernel",
            shape=(flat, 8),
            initializer="ones",
            trainable=self.trainable,
        )

    def call(self, inputs):
        self.observed_input_dtypes.append(inputs.dtype)
        if inputs.dtype != tf.float32:
            # Mirrors the real failure: the SavedModel signature is float32.
            raise tf.errors.InvalidArgumentError(None, None, _FLOAT16_ERROR)
        features = int(np.prod(self.signature_input_shape))
        flat = tf.reshape(inputs, (-1, features))
        return tf.matmul(flat, self._kernel)


class _FakeHubModule:
    """Minimal tensorflow_hub stand-in exposing only KerasLayer."""

    def __init__(self, layer_class):
        self.KerasLayer = layer_class


def _install_fake_hub(monkeypatch, layer_class):
    created = []

    class _RecordingLayer(layer_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(
        model_module,
        "_get_tfhub_module",
        lambda: _FakeHubModule(_RecordingLayer),
    )
    # Never touch the network: pretend the module URL resolves directly.
    monkeypatch.setattr(
        model_module,
        "_resolve_module_handle",
        lambda hub_url, cache: (hub_url, True),
    )
    return created


@pytest.fixture()
def mixed_precision():
    previous = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        yield
    finally:
        tf.keras.mixed_precision.set_global_policy(previous)
        tf.keras.backend.clear_session()


def test_lite0_backbone_is_pinned_to_float32_under_mixed_precision(
    monkeypatch, mixed_precision
):
    created = _install_fake_hub(monkeypatch, _FixedFloat32SignatureLayer)

    model, backbone = model_module.create_efficientnet_model(
        input_shape=(32, 32, 3), num_classes=72, dropout_rate=0.2
    )

    # The hub layer must opt out of the global mixed_float16 policy, otherwise
    # Keras autocasts its input to float16 before calling the saved model.
    assert created, "the factory never created a hub layer"
    assert backbone is created[0]
    assert backbone.dtype_policy.name == "float32"
    assert backbone.compute_dtype == "float32"
    assert backbone.observed_input_dtypes
    assert all(dtype == tf.float32 for dtype in backbone.observed_input_dtypes)

    assert model.inputs[0].dtype == tf.float32
    assert model.output.dtype == tf.float32
    logits = model(tf.zeros((2, 32, 32, 3), dtype=tf.float32), training=False)
    assert tuple(logits.shape) == (2, 72)


def test_unpinned_hub_layer_reproduces_the_float16_failure(mixed_precision):
    """Guard the fixture: an unpinned layer must reproduce the failure."""
    layer = _FixedFloat32SignatureLayer(
        "fake://module", input_shape=(8, 8, 3), trainable=False
    )
    assert layer.compute_dtype == "float16"
    with pytest.raises(tf.errors.InvalidArgumentError, match="half tensor"):
        layer(np.zeros((1, 8, 8, 3), dtype=np.float32))


def test_dtype_error_is_not_treated_as_a_cache_error():
    dtype_exc = RuntimeError(_FLOAT16_ERROR)
    assert model_module._is_tfhub_dtype_error(dtype_exc)
    assert not model_module._is_tfhub_cache_error(dtype_exc)

    network_exc = RuntimeError(
        "URL fetch failure on https://tfhub.dev/module: 404 -- Not Found"
    )
    assert not model_module._is_tfhub_dtype_error(network_exc)
    assert model_module._is_tfhub_cache_error(network_exc)


def test_dtype_error_fails_fast_without_wiping_the_cache(monkeypatch):
    cleared = []
    monkeypatch.setattr(
        model_module,
        "_clear_tfhub_cache",
        lambda cache_dir: cleared.append(cache_dir),
    )

    class _BrokenLayer(_FixedFloat32SignatureLayer):
        def call(self, inputs):
            raise tf.errors.InvalidArgumentError(None, None, _FLOAT16_ERROR)

    _install_fake_hub(monkeypatch, _BrokenLayer)

    with pytest.raises(RuntimeError, match="float32"):
        model_module.create_efficientnet_model(
            input_shape=(16, 16, 3),
            num_classes=5,
            hub_download_retries=3,
            hub_download_delay_sec=0,
        )

    assert cleared == []
