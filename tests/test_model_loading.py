"""Regression tests for loading models produced by the training pipeline."""

import pytest

tf = pytest.importorskip("tensorflow")

from quantization import post_training_quant
from quantization import qat
from training import evaluate
from training.serialization import get_custom_objects


@pytest.mark.parametrize(
    ("loader", "load_model"),
    [
        (evaluate, lambda module, path: module._load_model(path, (224, 224, 3))),
        (
            post_training_quant,
            lambda module, path: module._load_keras_model(path),
        ),
        (qat, lambda module, path: module._load_float_model(path, {})),
    ],
)
def test_h5_loaders_pass_registered_training_objects(monkeypatch, loader, load_model):
    captured = {}

    def fake_load_model(model_path, **kwargs):
        captured["path"] = model_path
        captured["custom_objects"] = kwargs.get("custom_objects")
        return "loaded-model"

    monkeypatch.setattr(loader.keras.models, "load_model", fake_load_model)

    result = load_model(loader, "trained_model.h5")

    assert result == "loaded-model"
    assert captured["path"] == "trained_model.h5"
    assert captured["custom_objects"] == get_custom_objects()


def test_custom_object_mapping_has_legacy_and_registered_names():
    custom_objects = get_custom_objects()

    assert custom_objects["FocalLoss"] is custom_objects["LeafDisease>FocalLoss"]
    assert custom_objects["MacroPrecisionRecall"] is custom_objects[
        "LeafDisease>MacroPrecisionRecall"
    ]
