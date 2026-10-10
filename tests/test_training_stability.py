from types import SimpleNamespace

import pytest

tf = pytest.importorskip("tensorflow")

from training.model import unfreeze_base_model
from training.train import (
    compile_model,
    create_fine_tune_optimizer,
    select_final_weights,
)
from training.utils import compute_class_weights


def _config():
    return {
        "model": {"num_classes": 2},
        "optimizer": {
            "name": "adam",
            "learning_rate": 3e-4,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-7,
            "clipnorm": 1.0,
        },
        "training": {"fine_tune_learning_rate": 5e-6},
        "loss": {"name": "categorical_crossentropy"},
        "metrics": ["accuracy"],
    }


def test_keeps_phase_one_weights_when_fine_tuning_is_worse():
    phase1 = SimpleNamespace(best=0.42)
    phase2 = SimpleNamespace(best=0.51, weights=["phase-two"])

    weights, phase, loss = select_final_weights(phase1, ["phase-one"], phase2)

    assert weights == ["phase-one"]
    assert phase == "phase_1"
    assert loss == pytest.approx(0.42)


def test_accepts_phase_two_weights_only_on_strict_improvement():
    phase1 = SimpleNamespace(best=0.42)
    phase2 = SimpleNamespace(best=0.41, weights=["phase-two"])

    weights, phase, loss = select_final_weights(phase1, ["phase-one"], phase2)

    assert weights == ["phase-two"]
    assert phase == "phase_2"
    assert loss == pytest.approx(0.41)


def test_unfreezing_keeps_batch_normalization_frozen():
    base_model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer((8, 8, 3)),
            tf.keras.layers.Conv2D(4, 1, name="early_conv"),
            tf.keras.layers.BatchNormalization(name="batch_norm"),
            tf.keras.layers.Conv2D(4, 1, name="late_conv"),
        ]
    )

    unfreeze_base_model(base_model, unfreeze_from_layer=1)

    assert base_model.get_layer("early_conv").trainable is False
    assert base_model.get_layer("batch_norm").trainable is False
    assert base_model.get_layer("late_conv").trainable is True


def test_gradient_clipping_is_applied_in_both_training_phases():
    config = _config()
    model = tf.keras.Sequential(
        [tf.keras.layers.InputLayer((4,)), tf.keras.layers.Dense(2, activation="softmax")]
    )

    compile_model(model, config)
    fine_tune_optimizer = create_fine_tune_optimizer(config)

    assert model.optimizer.clipnorm == pytest.approx(1.0)
    assert fine_tune_optimizer.clipnorm == pytest.approx(1.0)


def test_capped_class_weights_are_bounded_and_sample_normalized(tmp_path):
    counts = {"common": 100, "rare": 2}
    for class_name, count in counts.items():
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for index in range(count):
            (class_dir / f"{index}.jpg").touch()

    weights = compute_class_weights(tmp_path, max_weight=3.0, normalize=True)
    sample_weighted_mean = sum(
        counts[class_name] * weights[index]
        for index, class_name in enumerate(sorted(counts))
    ) / sum(counts.values())

    assert max(weights.values()) <= 3.0
    assert sample_weighted_mean == pytest.approx(1.0)
