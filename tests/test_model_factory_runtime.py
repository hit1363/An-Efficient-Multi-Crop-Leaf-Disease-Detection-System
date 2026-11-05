"""TensorFlow smoke tests for the offline-capable model factories.

The EfficientNet-Lite0 model is intentionally excluded because it requires a
TensorFlow Hub download. Run that model's notebook in Colab/Kaggle separately.
"""

import pytest


tf = pytest.importorskip("tensorflow")

from pathlib import Path

from training.model import get_model, unfreeze_base_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


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
        input_shape=(192, 192, 3),
        num_classes=5,
        dropout_rate=0.1,
        **kwargs,
    )

    assert model.output_shape == (None, 5)
    assert base_model.trainable is False

    output = model(tf.zeros((1, 192, 192, 3)), training=False)
    assert tuple(output.shape) == (1, 5)
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


def test_config_unfreeze_from_layer_is_within_backbone_depth():
    """Every two-phase config must unfreeze a real slice of its backbone.

    ``unfreeze_base_model`` slices ``layers[unfreeze_from_layer:]``, so an
    index at or beyond the backbone depth freezes every layer and phase 2
    silently fine-tunes nothing (mobilenet_v1 shipped ``100`` against an
    86-layer backbone).
    """
    import os
    import yaml

    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    import tf_keras as keras_apps  # noqa: F401  (legacy applications namespace)
    from tf_keras.applications import (
        EfficientNetB0,
        EfficientNetV2B0,
        MobileNet,
        MobileNetV2,
        MobileNetV3Large,
        MobileNetV3Small,
        NASNetMobile,
    )

    backbones = {
        "mobilenet_v1": lambda: MobileNet(
            input_shape=(192, 192, 3), include_top=False, weights=None
        ),
        "mobilenetv2": lambda: MobileNetV2(
            input_shape=(192, 192, 3), include_top=False, weights=None
        ),
        "mobilenetv3_small": lambda: MobileNetV3Small(
            input_shape=(192, 192, 3), include_top=False, weights=None
        ),
        "mobilenet_v3_large": lambda: MobileNetV3Large(
            input_shape=(192, 192, 3), include_top=False, weights=None
        ),
        "nasnet_mobile": lambda: NASNetMobile(
            input_shape=(192, 192, 3), include_top=False, weights=None
        ),
        "efficientnet_b0": lambda: EfficientNetB0(
            input_shape=(192, 192, 3), include_top=False, weights=None
        ),
        "efficientnet_v2_b0": lambda: EfficientNetV2B0(
            input_shape=(192, 192, 3), include_top=False, weights=None
        ),
    }

    for architecture, build in backbones.items():
        config = yaml.safe_load(
            (ROOT / "training" / f"config_{architecture}.yaml").read_text(
                encoding="utf-8"
            )
        )
        unfreeze_from = int(config["training"]["unfreeze_from_layer"])
        base_model = build()
        total_layers = len(base_model.layers)
        assert unfreeze_from < total_layers, (
            f"{architecture}: unfreeze_from_layer={unfreeze_from} is at or "
            f"beyond the {total_layers}-layer backbone; phase 2 would "
            "fine-tune nothing"
        )
        unfreeze_base_model(base_model, unfreeze_from_layer=unfreeze_from)
        trainable = sum(layer.trainable for layer in base_model.layers)
        assert trainable > 0, (
            f"{architecture}: phase 2 left zero trainable backbone layers"
        )
