import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from training.model import unfreeze_base_model
from training.train import (
    _estimated_split_counts,
    _finite_dataset_steps,
    build_metrics,
    compile_model,
    compile_fine_tune_model,
    configure_runtime,
    create_fine_tune_optimizer,
    phase_one_heading,
    save_run_metadata,
    select_final_weights,
)
from training.export import make_float32_inference_model
from training.utils import (
    _dataset_class_names,
    compute_class_weights,
    compute_class_weights_from_counts,
    create_augmentation_layer,
    get_minority_class_indices,
    get_training_class_counts,
    get_checkpoint_path,
    load_dataset,
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


@pytest.mark.parametrize(
    ("freeze_base", "expected"),
    [
        (True, "Phase 1: Training classifier head (base frozen)"),
        (False, "Phase 1: Training full model (backbone trainable)"),
    ],
)
def test_phase_one_heading_matches_backbone_trainability(freeze_base, expected):
    assert phase_one_heading(freeze_base) == expected


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


def test_weighted_fit_uses_categorical_crossentropy():
    config = _config()
    model = tf.keras.Sequential(
        [tf.keras.layers.InputLayer((4,)), tf.keras.layers.Dense(2, activation="softmax")]
    )

    compile_model(model, config)

    assert model.loss == "categorical_crossentropy"


def test_fit_time_class_weights_complete_a_small_fit():
    model = tf.keras.Sequential(
        [tf.keras.layers.InputLayer((4,)), tf.keras.layers.Dense(2, activation="softmax")]
    )
    compile_model(model, _config())
    features = tf.constant([[0.0] * 4, [1.0] * 4, [0.5] * 4, [0.25] * 4])
    labels = tf.constant([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(2)

    history = model.fit(
        dataset, epochs=1, class_weight={0: 1.0, 1: 3.0}, verbose=0
    )

    assert np.isfinite(history.history["loss"][0])


def test_synthetic_weighted_fit_runs_both_warmup_and_fine_tune_phases():
    strategy = tf.distribute.get_strategy()
    base_model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer((4,)),
            tf.keras.layers.Dense(3, activation="relu", name="features"),
        ],
        name="synthetic_base",
    )
    base_model.trainable = False
    inputs = tf.keras.Input((4,))
    outputs = tf.keras.layers.Dense(2, activation="softmax")(
        base_model(inputs)
    )
    model = tf.keras.Model(inputs, outputs)
    config = _config()
    config["performance"] = {"jit_compile": False}
    features = tf.constant(
        [[0.0, 0.1, 0.2, 0.3], [1.0, 0.9, 0.8, 0.7],
         [0.2, 0.3, 0.4, 0.5], [0.8, 0.7, 0.6, 0.5]],
        dtype=tf.float32,
    )
    labels = tf.constant(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=tf.float32,
    )
    train_dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(2)
    val_dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(2)
    class_weights = {0: 1.0, 1: 3.0}

    compile_model(model, config)
    warmup_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, mode="min"
        )
    ]
    warmup_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=1,
        class_weight=class_weights,
        callbacks=warmup_callbacks,
        verbose=0,
    )
    assert np.isfinite(warmup_history.history["loss"][0])
    assert not base_model.trainable_weights

    unfreeze_base_model(base_model, unfreeze_from_layer=0)
    compile_fine_tune_model(model, config, strategy)
    fine_tune_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, mode="min"
        )
    ]
    fine_tune_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        initial_epoch=1,
        epochs=2,
        class_weight=class_weights,
        callbacks=fine_tune_callbacks,
        verbose=0,
    )

    assert base_model.trainable_weights
    assert np.isfinite(fine_tune_history.history["loss"][0])


def test_runtime_warns_when_multiple_gpus_are_visible_but_strategy_is_single(
    monkeypatch,
):
    warnings = []
    logger = SimpleNamespace(
        info=lambda *args: None,
        warning=lambda *args: warnings.append(args),
    )
    monkeypatch.setattr(
        tf.config, "list_physical_devices", lambda device_type: [object(), object()]
    )
    monkeypatch.setattr(tf.keras.mixed_precision, "set_global_policy", lambda _: None)

    strategy = configure_runtime(
        {
            "performance": {
                "mixed_precision": False,
                "distribution": "single",
                "multi_gpu": False,
            }
        },
        logger,
    )

    assert strategy.num_replicas_in_sync == 1
    assert any("Multiple GPUs are visible" in entry[0] for entry in warnings)


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


def test_class_weights_from_counts_match_directory_scan(tmp_path):
    counts = {"common": 100, "rare": 2, "empty": 0}
    for class_name, count in counts.items():
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for index in range(count):
            (class_dir / f"{index}.jpg").touch()

    scanned = compute_class_weights(tmp_path, max_weight=3.0, normalize=True)
    from_counts = compute_class_weights_from_counts(
        [counts[name] for name in sorted(counts)], max_weight=3.0, normalize=True
    )

    assert from_counts == pytest.approx(scanned)


def test_training_counts_use_manifest_and_select_only_positive_minority_classes(tmp_path):
    class_names = ["common", "rare", "empty"]
    for class_name in class_names:
        (tmp_path / "train" / class_name).mkdir(parents=True)
    with (tmp_path / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "canonical_label"])
        writer.writeheader()
        writer.writerows(
            [{"split": "train", "canonical_label": "common"}] * 600
            + [{"split": "train", "canonical_label": "rare"}] * 350
        )

    counts = get_training_class_counts(tmp_path / "train", class_names=class_names)

    assert counts == {"common": 600, "rare": 350, "empty": 0}
    assert get_minority_class_indices(counts, max_train_samples=500) == [1]


def test_class_aware_pipeline_replaces_only_minority_training_images(tmp_path):
    for class_name in ("common", "rare"):
        (tmp_path / "train" / class_name).mkdir(parents=True)
        (tmp_path / "val" / class_name).mkdir(parents=True)
    image = tf.zeros((4, 4, 3), dtype=tf.uint8)
    for split in ("train", "val"):
        for class_name in ("common", "rare"):
            tf.io.write_file(
                str(tmp_path / split / class_name / "sample.png"),
                tf.io.encode_png(image),
            )

    baseline = tf.keras.layers.Lambda(lambda x: x + 1.0)
    minority = tf.keras.layers.Lambda(lambda x: x + 10.0)
    train_ds, val_ds, class_names = load_dataset(
        tmp_path / "train",
        tmp_path / "val",
        batch_size=2,
        image_size=(4, 4),
        augmentation=baseline,
        minority_augmentation=minority,
        minority_class_indices=[1],
        shuffle_buffer=0,
        cache_mode="none",
    )

    train_images, train_labels = next(iter(train_ds))
    values_by_label = {
        int(label): float(tf.reduce_mean(train_images[index]).numpy())
        for index, label in enumerate(tf.argmax(train_labels, axis=-1).numpy())
    }
    val_images, _ = next(iter(val_ds))

    assert class_names == ["common", "rare"]
    assert values_by_label == {0: 1.0, 1: 10.0}
    assert float(tf.reduce_max(val_images).numpy()) == 0.0


def test_run_metadata_reuses_supplied_split_counts(tmp_path):
    class Model:
        name = "test_model"
        trainable_weights = [SimpleNamespace(shape=(2,))]
        non_trainable_weights = []

        @staticmethod
        def count_params():
            return 2

        @staticmethod
        def summary(print_fn):
            print_fn("test model")

    split_counts = {
        "train": {"total": 3, "classes": {"healthy": 2, "disease": 1}},
        "val": {"total": 2, "classes": {"healthy": 1, "disease": 1}},
        "test": {"total": 1, "classes": {"healthy": 1, "disease": 0}},
    }
    config = {
        "model": {"architecture": "test_model"},
        "dataset": {
            "train_dir": "/data/train",
            "val_dir": "/data/val",
            "test_dir": "/data/test",
        },
        "class_weights": {"enabled": True, "max_weight": 3.0, "normalize": True},
        "export": {"save_dir": str(tmp_path)},
    }

    run_dir = save_run_metadata(
        config=config,
        config_path="config.yaml",
        model=Model(),
        raw_class_weights={0: 0.75, 1: 1.5},
        applied_class_weights={0: 0.75, 1: 1.5},
        split_counts=split_counts,
        run_timestamp="test-run",
    )

    with open(
        tmp_path / "test_model" / "runs" / "test-run" / "run_metadata.json",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)

    assert run_dir == str(tmp_path / "test_model" / "runs" / "test-run")
    assert metadata["datasets"]["train"] == {
        "path": "/data/train",
        **split_counts["train"],
    }
    assert metadata["class_weights"]["applied_range"] == {
        "min": pytest.approx(0.75),
        "max": pytest.approx(1.5),
    }


def test_run_metadata_records_null_weight_ranges_when_weighting_is_disabled(tmp_path):
    class Model:
        name = "test_model"
        trainable_weights = []
        non_trainable_weights = []

        @staticmethod
        def count_params():
            return 0

        @staticmethod
        def summary(print_fn):
            print_fn("test model")

    config = {
        "model": {"architecture": "test_model"},
        "dataset": {},
        "class_weights": {"enabled": False},
        "export": {"save_dir": str(tmp_path)},
    }
    save_run_metadata(
        config=config,
        config_path="config.yaml",
        model=Model(),
        raw_class_weights=None,
        applied_class_weights=None,
        split_counts={},
        run_timestamp="unweighted-run",
    )

    with open(
        tmp_path / "test_model" / "runs" / "unweighted-run" / "run_metadata.json",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)

    assert metadata["class_weights"]["enabled"] is False
    assert metadata["class_weights"]["raw_range"] is None
    assert metadata["class_weights"]["applied_range"] is None


def test_training_precision_recall_are_skipped_as_duplicate_of_accuracy():
    with pytest.warns(UserWarning, match="duplicates accuracy"):
        metrics = build_metrics(["accuracy", "precision", "recall"], 3)

    assert [metric.name for metric in metrics] == ["accuracy"]


def test_explicit_steps_and_estimated_counts_use_finite_dataset_cardinality():
    train_dataset = tf.data.Dataset.range(3).batch(2)
    val_dataset = tf.data.Dataset.range(2).batch(2)

    train_steps = _finite_dataset_steps(train_dataset, "training")
    val_steps = _finite_dataset_steps(val_dataset, "validation")
    counts = _estimated_split_counts(train_steps, val_steps, batch_size=2)

    assert train_steps == 2
    assert val_steps == 1
    assert counts == {
        "train": {"total": 4, "classes": {}},
        "val": {"total": 2, "classes": {}},
        "test": {"total": 0, "classes": {}},
    }


def test_explicit_steps_reject_unknown_or_empty_dataset_cardinality():
    with pytest.raises(ValueError, match="known positive finite cardinality"):
        _finite_dataset_steps(tf.data.Dataset.range(0).batch(2), "training")

    unknown_dataset = tf.data.Dataset.from_generator(
        lambda: iter([1, 2]), output_signature=tf.TensorSpec((), tf.int64)
    ).batch(2)
    with pytest.raises(ValueError, match="known positive finite cardinality"):
        _finite_dataset_steps(unknown_dataset, "training")


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


def test_split_validation_uses_manifest_counts_without_directory_image_scans(tmp_path):
    class_names = ["healthy", "disease"]
    for split in ("train", "val", "test"):
        for class_name in class_names:
            (tmp_path / split / class_name).mkdir(parents=True)

    manifest_rows = [
        ("train", "healthy", "train-healthy"),
        ("train", "disease", "train-disease"),
        ("train", "disease", "train-disease-2"),
        ("val", "healthy", "val-healthy"),
        ("test", "disease", "test-disease"),
    ]
    with (tmp_path / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["split", "canonical_label", "source_sha256"]
        )
        writer.writeheader()
        for split, label, digest in manifest_rows:
            writer.writerow(
                {"split": split, "canonical_label": label, "source_sha256": digest}
            )

    counts = validate_dataset_splits(
        tmp_path / "train",
        tmp_path / "val",
        tmp_path / "test",
        class_names=class_names,
    )

    assert counts["train"] == {
        "total": 3,
        "classes": {"healthy": 1, "disease": 2},
    }
    assert counts["val"]["classes"] == {"healthy": 1, "disease": 0}


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
