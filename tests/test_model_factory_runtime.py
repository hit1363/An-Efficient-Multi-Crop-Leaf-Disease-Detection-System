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

