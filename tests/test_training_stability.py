from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from training.model import unfreeze_base_model
from training.train import (
    build_metrics,
    compile_model,
    compile_fine_tune_model,
    create_fine_tune_optimizer,
    select_final_weights,
)
from training.export import make_float32_inference_model
from training.utils import (
    _dataset_class_names,
    compute_class_weights,
    create_augmentation_layer,
    get_checkpoint_path,
    validate_dataset_splits,
)
from dataset.taxonomy import CANONICAL_CLASSES, LABEL_ALIASES


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


def test_fine_tune_compile_creates_metrics_in_distribution_scope():
    strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0"])
    with strategy.scope():
        model = tf.keras.Sequential(
            [tf.keras.layers.InputLayer((4,)), tf.keras.layers.Dense(2, activation="softmax")]
        )

    compile_fine_tune_model(model, _config(), strategy)

    assert model.optimizer is not None
    # TensorFlow 2.19's legacy Keras defers populating ``model.metrics`` until
    # the first train step. The configured user metric is available immediately.
    assert [metric.name for metric in model.compiled_metrics._user_metrics] == [
        "accuracy"
    ]


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
            (class_dir / "sample.jpg").write_bytes(
                f"{split}-{class_name}".encode("utf-8")
            )

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


def test_alias_split_validation_hashes_actual_source_folders_without_manifest(tmp_path):
    source_by_canonical = {
        canonical: alias for alias, canonical in LABEL_ALIASES.items()
    }
    expected_source_order = [
        source_by_canonical.get(name, name) for name in CANONICAL_CLASSES
    ]

    for split in ("train", "val", "test"):
        for index, class_name in enumerate(CANONICAL_CLASSES):
            source_name = expected_source_order[index]
            class_dir = tmp_path / split / source_name
            class_dir.mkdir(parents=True)
            (class_dir / "sample.jpg").write_bytes(
                f"{split}-{class_name}".encode("utf-8")
            )

    assert _dataset_class_names(tmp_path / "train") == expected_source_order
    counts = validate_dataset_splits(
        tmp_path / "train", tmp_path / "val", tmp_path / "test"
    )
    assert list(counts["train"]["classes"]) == list(CANONICAL_CLASSES)
    assert counts["train"]["total"] == len(CANONICAL_CLASSES)

    train_image = tmp_path / "train" / expected_source_order[0] / "sample.jpg"
    val_image = tmp_path / "val" / expected_source_order[0] / "sample.jpg"
    val_image.write_bytes(train_image.read_bytes())
    with pytest.raises(ValueError, match="exact duplicate image hashes"):
        validate_dataset_splits(
            tmp_path / "train", tmp_path / "val", tmp_path / "test"
        )


def test_class_folder_alias_resolution_rejects_invalid_taxonomy(tmp_path):
    unknown_dir = tmp_path / "unknown"
    (unknown_dir / "Unmapped___Disease").mkdir(parents=True)
    with pytest.raises(ValueError, match="Unknown dataset class folder"):
        _dataset_class_names(unknown_dir, CANONICAL_CLASSES)

    missing_dir = tmp_path / "missing"
    (missing_dir / CANONICAL_CLASSES[0]).mkdir(parents=True)
    with pytest.raises(ValueError, match="missing="):
        _dataset_class_names(missing_dir, CANONICAL_CLASSES)

    duplicate_dir = tmp_path / "duplicate"
    (duplicate_dir / "Mango___Anthracnose").mkdir(parents=True)
    (duplicate_dir / "Mango__Anthracnose").mkdir(parents=True)
    with pytest.raises(ValueError, match="Duplicate dataset class folders"):
        _dataset_class_names(duplicate_dir, CANONICAL_CLASSES)


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


def test_mixed_precision_model_clones_to_builtin_tflite_and_preserves_predictions():
    previous_policy = tf.keras.mixed_precision.global_policy()
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        inputs = tf.keras.Input((8, 8, 3))
        backbone = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu"),
                tf.keras.layers.BatchNormalization(),
            ],
            name="backbone",
        )
        features = backbone(inputs)
        features = tf.keras.layers.GlobalAveragePooling2D()(features)
        outputs = tf.keras.layers.Dense(3, activation="softmax", dtype="float32")(
            features
        )
        mixed_model = tf.keras.Model(inputs, outputs)
        sample = tf.random.uniform((2, 8, 8, 3))
        expected = mixed_model(sample, training=False).numpy()

        float32_model = make_float32_inference_model(mixed_model)
        assert float32_model.compute_dtype == "float32"
        assert float32_model.get_layer("backbone").compute_dtype == "float32"
        actual = float32_model(sample, training=False).numpy()
        np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-3)

        converter = tf.lite.TFLiteConverter.from_keras_model(float32_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.allow_custom_ops = False
        tflite_model = converter.convert()
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]["index"], sample[:1].numpy())
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]["index"])
        np.testing.assert_allclose(
            tflite_output, expected[:1], rtol=3e-2, atol=3e-3
        )
    finally:
        tf.keras.mixed_precision.set_global_policy(previous_policy)


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
