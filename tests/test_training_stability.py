from types import SimpleNamespace

import pytest

tf = pytest.importorskip("tensorflow")

from training.model import unfreeze_base_model
from training.train import (
    build_metrics,
    compile_model,
    create_fine_tune_optimizer,
    select_final_weights,
)
from training.utils import (
    compute_class_weights,
    create_augmentation_layer,
    get_checkpoint_path,
    validate_dataset_splits,
)


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


def test_training_precision_recall_are_skipped_as_duplicate_of_accuracy():
    with pytest.warns(UserWarning, match="duplicates accuracy"):
        metrics = build_metrics(["accuracy", "precision", "recall"], 3)

    assert [metric.name for metric in metrics] == ["accuracy"]


def test_split_validation_requires_matching_classes_and_train_support(tmp_path):
    for split in ("train", "val", "test"):
        for class_name in ("healthy", "disease"):
            class_dir = tmp_path / split / class_name
            class_dir.mkdir(parents=True)
            (class_dir / "sample.jpg").write_bytes(b"image")

    counts = validate_dataset_splits(
        tmp_path / "train", tmp_path / "val", tmp_path / "test"
    )
    assert counts["train"]["total"] == 2
    assert counts["test"]["classes"] == {"disease": 1, "healthy": 1}

    (tmp_path / "val" / "disease" / "sample.jpg").unlink()
    counts = validate_dataset_splits(
        tmp_path / "train", tmp_path / "val", tmp_path / "test"
    )
    assert counts["val"]["classes"]["disease"] == 0

    (tmp_path / "train" / "disease" / "sample.jpg").unlink()
    with pytest.raises(ValueError, match="no supported images"):
        validate_dataset_splits(
            tmp_path / "train", tmp_path / "val", tmp_path / "test"
        )


def test_augmentation_respects_configured_flips_and_ranges():
    config = {
        "augmentation": {
            "horizontal_flip": True,
            "vertical_flip": False,
            "rotation_range": 0,
            "zoom_range": [1.0, 1.0],
            "contrast_range": 0,
        }
    }
    augmentation = create_augmentation_layer(config)

    assert [layer.__class__.__name__ for layer in augmentation.layers] == [
        "RandomFlip"
    ]
    assert augmentation.layers[0].mode == "horizontal"


def test_checkpoint_can_be_scoped_to_a_run_directory(tmp_path):
    config = {
        "model": {"architecture": "mobilenetv2"},
        "export": {"save_dir": str(tmp_path)},
        "callbacks": {
            "checkpoint": {
                "directory": str(tmp_path / "runs" / "run-1" / "checkpoints")
            }
        },
    }

    checkpoint_path, weights_only = get_checkpoint_path(config)

    assert checkpoint_path == str(
        tmp_path / "runs" / "run-1" / "checkpoints" / "best_model.h5"
    )
    assert weights_only is False
