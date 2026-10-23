"""TensorFlow smoke tests for the offline-capable model factories.

The EfficientNet-Lite0 model is intentionally excluded because it requires a
TensorFlow Hub download. Run that model's notebook in Colab/Kaggle separately.
"""

import pytest


tf = pytest.importorskip("tensorflow")

from training.model import get_model, unfreeze_base_model  # noqa: E402


@pytest.mark.parametrize(
    ("architecture", "kwargs"),
    [
        ("mobilenetv2", {"weights": None}),
        ("efficientnet_b0", {"weights": None}),
        (
            "mobilenetv3_small",
            {"weights": None, "include_preprocessing": False},
        ),
        ("shufflenetv2_05", {"weights": None}),
    ],
)
def test_factory_builds_and_runs_forward_pass(architecture, kwargs):
    model, base_model = get_model(
        architecture=architecture,
        input_shape=(64, 64, 3),
        num_classes=5,
        dropout_rate=0.1,
        **kwargs,
    )

    assert model.output_shape == (None, 5)
    assert base_model.trainable is False

    output = model(tf.zeros((2, 64, 64, 3)), training=False)
    assert tuple(output.shape) == (2, 5)
    assert output.dtype == tf.float32

    unfreeze_base_model(base_model, unfreeze_from_layer=0)
    assert any(weight.trainable for weight in base_model.weights)


def test_efficientnet_b0_builds_with_mixed_precision():
    previous_policy = tf.keras.mixed_precision.global_policy()
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        model, base_model = get_model(
            architecture="efficientnet_b0",
            input_shape=(64, 64, 3),
            num_classes=5,
            dropout_rate=0.1,
            weights=None,
        )

        assert tf.keras.mixed_precision.global_policy().name == "mixed_float16"
        output = model(tf.zeros((1, 64, 64, 3)), training=False)

        assert base_model.compute_dtype == "float32"
        assert tuple(output.shape) == (1, 5)
        assert output.dtype == tf.float32
        assert bool(tf.reduce_all(tf.math.is_finite(output)))
    finally:
        tf.keras.mixed_precision.set_global_policy(previous_policy)
