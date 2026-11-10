import csv
import io
import json
import logging
import os
from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from training.model import (
    _attach_classifier_head,
    unfreeze_backbone_for_scratch_training,
    unfreeze_base_model,
)
from training.train import (
    _estimated_split_counts,
    _finite_dataset_steps,
    _fit_verbosity,
    _fit_sample_count,
    _add_keras_progress_logger,
    _configure_progress_logging,
    _prefetch_to_device_for_single_replica,
    _progress_log_interval,
    _progress_style,
    _resolve_fit_step_counts,
    _save_native_keras_model,
    _should_skip_final_export,
    _stdout_is_interactive,
    build_metrics,
    compile_model,
    compile_fine_tune_model,
    configure_runtime,
    _configure_phase2_early_stopping,
    create_fine_tune_optimizer,
    BatchProgressLogger,
    EpochThroughputLogger,
    DiagnosticBatchProgress,
    InlineProgressBar,
    phase_one_heading,
    save_run_metadata,
    select_final_weights,
)
from training.export import make_float32_inference_model
from training.utils import (
    _dataset_class_names,
    _estimate_dataset_cache_gb,
    compute_class_weights,
    compute_class_weights_from_counts,
    create_augmentation_layer,
    get_minority_class_indices,
    get_training_class_counts,
    get_checkpoint_path,
    get_epoch_log_mode,
    load_benchmark_class_names,
    load_dataset,
    setup_callbacks,
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


def test_phase2_lr_override_restarts_decay_at_unfreeze_epoch(monkeypatch):
    """Fine-tuning must start a fresh decay curve, not inherit Phase 1's.

    Keras hands the scheduler global epoch indices that continue from the
    Phase 1 ``initial_epoch``; without ``epoch_offset`` the restart began
    two-thirds decayed and floored at ``alpha * fine_tune_lr`` within a few
    epochs (val_loss 0.1856 -> 0.1844 across 9 epochs in run 20261104_053311).
    """
    import training.train as train

    captured = {}

    def fake_setup_callbacks(config, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(train, "setup_callbacks", fake_setup_callbacks)

    train._setup_phase2_callbacks(
        {
            "training": {
                "epochs": 30,
                "unfreeze_epoch": 20,
                "fine_tune_learning_rate": 2e-5,
            },
        }
    )

    assert captured["lr_schedule_override"] == {
        "initial_learning_rate": 2e-5,
        "decay_epochs": 10.0,
        "epoch_offset": 20.0,
    }


def test_cosine_decay_honors_epoch_offset_for_fine_tuning_restart(tmp_path):
    config = {
        "model": {"num_classes": 2},
        "optimizer": {"learning_rate": 3e-4},
        "training": {"log_mode": "summary"},
        "dataset": {"batch_size": 2},
        "performance": {"terminate_on_nan": False},
        "lr_schedule": {"type": "cosine_decay", "decay_epochs": 10, "alpha": 0.01},
        "callbacks": {
            "checkpoint": {"enabled": False},
            "early_stopping": {"enabled": False},
            "tensorboard": {"enabled": False},
            "csv_logger": {"enabled": False},
        },
    }

    callbacks = setup_callbacks(
        config,
        include_throughput=False,
        lr_schedule_override={
            "initial_learning_rate": 2e-5,
            "decay_epochs": 10,
            "epoch_offset": 20,
        },
    )

    scheduler = next(
        callback
        for callback in callbacks
        if isinstance(callback, tf.keras.callbacks.LearningRateScheduler)
    )

    # First fine-tune epoch (global index 20) starts the fresh curve peak.
    assert scheduler.schedule(20, None) == pytest.approx(2e-5)
    # Global epoch 25 sits mid-curve instead of five-tenths past decayed start.
    mid_curve = 2e-5 * (0.01 + 0.99 * 0.5 * (1.0 + np.cos(np.pi * 0.5)))
    assert scheduler.schedule(25, None) == pytest.approx(mid_curve)
    # The curve reaches its alpha floor when phase 2 completes.
    floor = 2e-5 * 0.01
    assert scheduler.schedule(30, None) == pytest.approx(floor)
    assert scheduler.schedule(31, None) == pytest.approx(floor)


def test_cosine_warmup_ramps_then_decays():
    config = {
        "model": {"num_classes": 2},
        "optimizer": {"learning_rate": 1e-3},
        "training": {"log_mode": "summary"},
        "dataset": {"batch_size": 2},
        "performance": {"terminate_on_nan": False},
        "lr_schedule": {
            "type": "cosine_decay",
            "decay_epochs": 10,
            "alpha": 0.01,
            "warmup_epochs": 3,
            "warmup_start_lr": 1e-4,
        },
        "callbacks": {
            "checkpoint": {"enabled": False},
            "early_stopping": {"enabled": False},
            "tensorboard": {"enabled": False},
            "csv_logger": {"enabled": False},
        },
    }

    callbacks = setup_callbacks(config, include_throughput=False)
    scheduler = next(
        callback
        for callback in callbacks
        if isinstance(callback, tf.keras.callbacks.LearningRateScheduler)
    )

    # Warmup: linear ramp from 1e-4 to 1e-3 over epochs 0, 1, 2.
    assert scheduler.schedule(0, None) == pytest.approx(1e-4)
    assert scheduler.schedule(1, None) == pytest.approx(4e-4)
    assert scheduler.schedule(2, None) == pytest.approx(7e-4)
    # Warmup end: epoch 3 reaches initial_lr.
    assert scheduler.schedule(3, None) == pytest.approx(1e-3)
    # Cosine mid-point: epoch 6 (shifted 3 past warmup, progress 3/7).
    progress = 3.0 / 7.0
    mid = 1e-3 * (0.01 + 0.99 * 0.5 * (1.0 + np.cos(np.pi * progress)))
    assert scheduler.schedule(6, None) == pytest.approx(mid)
    # Cosine floor: epoch 10 (shifted 7 past warmup, progress 1.0).
    floor = 1e-3 * 0.01
    assert scheduler.schedule(10, None) == pytest.approx(floor)


def test_cosine_without_warmup_keys_is_unchanged():
    config = {
        "model": {"num_classes": 2},
        "optimizer": {"learning_rate": 1e-3},
        "training": {"log_mode": "summary"},
        "dataset": {"batch_size": 2},
        "performance": {"terminate_on_nan": False},
        "lr_schedule": {"type": "cosine_decay", "decay_epochs": 10, "alpha": 0.01},
        "callbacks": {
            "checkpoint": {"enabled": False},
            "early_stopping": {"enabled": False},
            "tensorboard": {"enabled": False},
            "csv_logger": {"enabled": False},
        },
    }

    callbacks = setup_callbacks(config, include_throughput=False)
    scheduler = next(
        callback
        for callback in callbacks
        if isinstance(callback, tf.keras.callbacks.LearningRateScheduler)
    )

    # No warmup: epoch 0 starts at initial_lr.
    assert scheduler.schedule(0, None) == pytest.approx(1e-3)
    # Mid-curve: epoch 5 (progress 0.5).
    mid = 1e-3 * (0.01 + 0.99 * 0.5 * (1.0 + np.cos(np.pi * 0.5)))
    assert scheduler.schedule(5, None) == pytest.approx(mid)
    # Floor: epoch 10 (progress 1.0).
    floor = 1e-3 * 0.01
    assert scheduler.schedule(10, None) == pytest.approx(floor)


def test_invalid_warmup_raises():
    config = {
        "model": {"num_classes": 2},
        "optimizer": {"learning_rate": 1e-3},
        "training": {"log_mode": "summary"},
        "dataset": {"batch_size": 2},
        "performance": {"terminate_on_nan": False},
        "lr_schedule": {
            "type": "cosine_decay",
            "decay_epochs": 10,
            "alpha": 0.01,
            "warmup_epochs": 10,  # >= decay_epochs
        },
        "callbacks": {
            "checkpoint": {"enabled": False},
            "early_stopping": {"enabled": False},
            "tensorboard": {"enabled": False},
            "csv_logger": {"enabled": False},
        },
    }

    with pytest.raises(ValueError, match="warmup_epochs must be smaller"):
        setup_callbacks(config, include_throughput=False)


@pytest.mark.parametrize(
    ("freeze_base", "expected"),
    [
        (True, "Phase 1: Training classifier head (base frozen)"),
        (False, "Phase 1: Training full model (backbone trainable)"),
    ],
)
def test_phase_one_heading_matches_backbone_trainability(freeze_base, expected):
    assert phase_one_heading(freeze_base) == expected


def test_phase_two_early_stopping_honors_disabled_setting():
    retained_callback = object()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4),
        retained_callback,
    ]

    configured = _configure_phase2_early_stopping(
        callbacks,
        {"enabled": False, "patience": 2},
        "summary",
    )

    assert configured == [retained_callback]


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


def test_scratch_training_unfreezes_batch_normalization():
    """A from-scratch backbone must learn BN moving statistics, not freeze them."""
    base_model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer((8, 8, 3)),
            tf.keras.layers.Conv2D(4, 1, name="early_conv"),
            tf.keras.layers.BatchNormalization(name="batch_norm"),
            tf.keras.layers.Conv2D(4, 1, name="late_conv"),
        ]
    )
    base_model.trainable = False

    unfreeze_backbone_for_scratch_training(base_model)

    assert base_model.trainable is True
    for layer in base_model.layers:
        assert layer.trainable is True, layer.name


def test_classifier_head_propagates_backbone_training_mode():
    """Pretrained backbones infer; scratch backbones must train their BN layers."""
    base_model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer((8, 8, 3)),
            tf.keras.layers.Conv2D(4, 1, name="conv"),
            tf.keras.layers.BatchNormalization(name="batch_norm"),
        ]
    )
    moving_mean = base_model.get_layer("batch_norm").moving_mean
    initial_mean = tf.identity(moving_mean)

    frozen_model = _attach_classifier_head(
        base_model, (8, 8, 3), num_classes=3, backbone_training=False
    )
    frozen_model(tf.ones((4, 8, 8, 3)), training=True)
    np.testing.assert_allclose(moving_mean.numpy(), initial_mean.numpy())

    scratch_model = _attach_classifier_head(
        base_model, (8, 8, 3), num_classes=3, backbone_training=None
    )
    scratch_model(tf.ones((4, 8, 8, 3)) * 3.0, training=True)

    assert not np.allclose(moving_mean.numpy(), initial_mean.numpy())
    # Propagation means inference mode no longer updates the moving statistics.
    ref = tf.identity(moving_mean)
    scratch_model(tf.ones((4, 8, 8, 3)) * 3.0, training=False)
    np.testing.assert_allclose(moving_mean.numpy(), ref.numpy())


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


def test_device_prefetch_is_skipped_for_mirrored_strategy(caplog):
    class Dataset:
        def __init__(self):
            self.applied = []

        def apply(self, transformation):
            self.applied.append(transformation)
            return self

    train_ds = Dataset()
    val_ds = Dataset()
    caplog.set_level(logging.INFO)

    result = _prefetch_to_device_for_single_replica(
        train_ds,
        val_ds,
        SimpleNamespace(num_replicas_in_sync=2),
        logging.getLogger("test_prefetch"),
    )

    assert result == (train_ds, val_ds)
    assert train_ds.applied == []
    assert val_ds.applied == []
    assert "Skipping GPU prefetch for 2-replica strategy" in caplog.text


def test_device_prefetch_remains_enabled_for_single_replica(monkeypatch):
    class Dataset:
        def __init__(self):
            self.applied = []

        def apply(self, transformation):
            self.applied.append(transformation)
            return self

    train_ds = Dataset()
    val_ds = Dataset()
    monkeypatch.setattr(
        tf.data.experimental,
        "prefetch_to_device",
        lambda device: device,
    )
    monkeypatch.setattr(
        tf.config, "list_physical_devices", lambda device_type: [object()]
    )

    result = _prefetch_to_device_for_single_replica(
        train_ds,
        val_ds,
        SimpleNamespace(num_replicas_in_sync=1),
        logging.getLogger("test_prefetch"),
    )

    assert result == (train_ds, val_ds)
    assert train_ds.applied == ["/gpu:0"]
    assert val_ds.applied == ["/gpu:0"]


def test_device_prefetch_stays_on_host_without_a_gpu(monkeypatch, caplog):
    """prefetch_to_device('/gpu:0') applies on CPU-only hosts and pins batches in RAM."""

    class Dataset:
        def __init__(self):
            self.applied = []

        def apply(self, transformation):
            self.applied.append(transformation)
            return self

    train_ds = Dataset()
    val_ds = Dataset()
    monkeypatch.setattr(
        tf.data.experimental,
        "prefetch_to_device",
        lambda device: device,
    )
    monkeypatch.setattr(tf.config, "list_physical_devices", lambda device_type: [])
    caplog.set_level(logging.INFO)

    result = _prefetch_to_device_for_single_replica(
        train_ds,
        val_ds,
        SimpleNamespace(num_replicas_in_sync=1),
        logging.getLogger("test_prefetch"),
    )

    assert result == (train_ds, val_ds)
    assert train_ds.applied == []
    assert val_ds.applied == []
    assert "No GPU detected" in caplog.text


def test_diagnostic_step_caps_do_not_change_full_split_counts():
    full_train_steps = 2524
    full_validation_steps = 542

    fit_steps = _resolve_fit_step_counts(
        full_train_steps,
        full_validation_steps,
        {"max_train_steps": 8, "max_validation_steps": 2},
    )

    assert fit_steps == (8, 2)
    assert (full_train_steps, full_validation_steps) == (2524, 542)
    full_counts = {"total": 80746, "classes": {"rare": 350, "common": 80396}}
    assert _fit_sample_count(
        full_counts["total"], fit_steps[0], batch_size=64, diagnostic=True
    ) == 512
    assert full_counts["total"] == 80746
    assert _fit_sample_count(
        full_counts["total"], full_train_steps, batch_size=64
    ) == 80746
    assert _resolve_fit_step_counts(2524, 542, {}) == (2524, 542)
    with pytest.raises(ValueError, match="positive integer"):
        _resolve_fit_step_counts(2524, 542, {"max_train_steps": 0})


def test_diagnostic_fit_reports_batch_boundaries_and_skips_final_export(
    tmp_path, capsys
):
    model = tf.keras.Sequential(
        [tf.keras.layers.InputLayer((4,)), tf.keras.layers.Dense(2, activation="softmax")]
    )
    config = _config()
    config["performance"] = {"steps_per_execution": 1, "jit_compile": False}
    compile_model(model, config)

    features = tf.reshape(tf.range(32, dtype=tf.float32) / 32.0, (8, 4))
    labels = tf.constant(
        [[1.0, 0.0], [0.0, 1.0]] * 4,
        dtype=tf.float32,
    )
    train_dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(1)
    val_dataset = tf.data.Dataset.from_tensor_slices((features[:2], labels[:2])).batch(1)
    checkpoint_path = tmp_path / "diagnostic.weights.h5"
    callbacks = [
        DiagnosticBatchProgress(train_steps=8, validation_steps=2),
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), save_weights_only=True, monitor="val_loss"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, mode="min", restore_best_weights=True
        ),
        EpochThroughputLogger(samples=8, log_mode="progress"),
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=1,
        steps_per_epoch=8,
        validation_steps=2,
        class_weight={0: 1.0, 1: 2.0},
        callbacks=callbacks,
        verbose=0,
    )

    output_lines = capsys.readouterr().out.splitlines()
    assert sum("Diagnostic train batch " in line and " started" in line for line in output_lines) == 8
    assert sum("Diagnostic train batch " in line and " completed" in line for line in output_lines) == 8
    assert sum("Diagnostic validation batch " in line and " started" in line for line in output_lines) == 2
    assert sum("Diagnostic validation batch " in line and " completed" in line for line in output_lines) == 2
    assert checkpoint_path.exists()
    assert np.isfinite(history.history["loss"][0])
    assert _should_skip_final_export({"runtime": {"diagnostic": True}})
    assert not _should_skip_final_export({"runtime": {"diagnostic": False}})


def test_batch_progress_is_sparse_flushed_and_works_with_grouped_steps(capsys):
    model = tf.keras.Sequential(
        [tf.keras.layers.InputLayer((4,)), tf.keras.layers.Dense(2, activation="softmax")]
    )
    config = _config()
    config["performance"] = {"steps_per_execution": 8, "jit_compile": False}
    compile_model(model, config)

    features = tf.reshape(tf.range(72 * 4, dtype=tf.float32) / 100.0, (72, 4))
    labels = tf.one_hot(tf.range(72) % 2, depth=2)
    train_dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(1)
    val_dataset = tf.data.Dataset.from_tensor_slices((features[:16], labels[:16])).batch(1)
    progress = BatchProgressLogger(
        train_steps=72,
        validation_steps=16,
        interval=64,
        steps_per_execution=8,
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=1,
        steps_per_epoch=72,
        validation_steps=16,
        callbacks=[progress],
        verbose=0,
    )

    output = capsys.readouterr().out
    output_lines = output.splitlines()
    assert "Train batch 1/72 started" in output_lines
    assert "Train progress: 8/72 batches completed" in output_lines
    assert "Train progress: 64/72 batches completed" in output_lines
    assert "Train progress: 72/72 batches completed" in output_lines
    assert "Validation batch 1/16 started" in output_lines
    assert "Validation progress: 8/16 batches completed" in output_lines
    assert "Validation progress: 16/16 batches completed" in output_lines
    assert "\r" not in output
    assert np.isfinite(history.history["loss"][0])


def test_synthetic_weighted_fit_runs_both_warmup_and_fine_tune_phases(capsys):
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
    config["training"]["log_mode"] = "summary"
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
        ),
        EpochThroughputLogger(samples=4, log_mode="summary"),
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
        ),
        EpochThroughputLogger(samples=4, log_mode="summary"),
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
    output_lines = capsys.readouterr().out.splitlines()
    start_lines = [line for line in output_lines if line.endswith(" started")]
    summary_lines = [line for line in output_lines if " - loss:" in line]
    assert len(start_lines) == 2
    assert start_lines[0] == "Epoch 1/1 started"
    assert start_lines[1] == "Epoch 2/2 started"
    assert len(summary_lines) == 2
    assert summary_lines[0].startswith("Epoch 1/1 - loss:")
    assert "val_loss:" in summary_lines[0]
    assert "val_accuracy:" in summary_lines[0]
    assert "lr:" in summary_lines[0]
    assert "examples_per_second:" in summary_lines[0]
    assert summary_lines[1].startswith("Epoch 2/2 - loss:")


def test_cache_size_estimate_uses_cache_and_label_dtypes():
    float32_gb = _estimate_dataset_cache_gb(
        100,
        50,
        (224, 224),
        cache_dtype="float32",
        label_count=72,
    )
    float16_gb = _estimate_dataset_cache_gb(
        100,
        50,
        (224, 224),
        cache_dtype="float16",
        label_count=72,
    )
    image_only_float32_gb = _estimate_dataset_cache_gb(
        100, 50, (224, 224), cache_dtype="float32"
    )
    image_only_float16_gb = _estimate_dataset_cache_gb(
        100, 50, (224, 224), cache_dtype="float16"
    )

    assert image_only_float16_gb == pytest.approx(image_only_float32_gb / 2)
    assert float16_gb < float32_gb
    assert float16_gb > float32_gb / 2  # labels remain float32 in the cache


def test_summary_mode_silences_keras_callback_chatter(tmp_path):
    config = {
        "training": {"log_mode": "summary"},
        "dataset": {"batch_size": 2},
        "performance": {"terminate_on_nan": False},
        "optimizer": {"learning_rate": 1e-3},
        "lr_schedule": {"type": "cosine_decay", "decay_epochs": 2},
        "callbacks": {
            "checkpoint": {
                "enabled": True,
                "directory": str(tmp_path),
                "save_weights_only": True,
            },
            "early_stopping": {"enabled": True},
            "csv_logger": {"enabled": False},
        },
    }

    callbacks = setup_callbacks(config)

    assert callbacks
    assert all(getattr(callback, "verbose", 0) == 0 for callback in callbacks)
    assert "ThroughputCallback" not in {
        type(callback).__name__ for callback in callbacks
    }
    assert get_epoch_log_mode(config) == "summary"
    assert _fit_verbosity(get_epoch_log_mode(config)) == 0


@pytest.mark.parametrize(
    ("mode", "expected_verbosity"), [("summary", 0), ("progress", 1)]
)
def test_training_log_mode_controls_keras_batch_progress(mode, expected_verbosity):
    assert _fit_verbosity(mode) == expected_verbosity


def test_progress_logging_matches_keras_epoch_layout(tmp_path, capsys):
    model = tf.keras.Sequential(
        [tf.keras.layers.InputLayer((4,)), tf.keras.layers.Dense(2, activation="softmax")]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    features = tf.constant(
        [[0.0, 0.1, 0.2, 0.3], [1.0, 0.9, 0.8, 0.7]], dtype=tf.float32
    )
    labels = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(2)
    checkpoint_path = tmp_path / "best.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        EpochThroughputLogger(samples=2, log_mode="progress"),
    ]
    _add_keras_progress_logger(callbacks, "progress")

    model.fit(
        dataset,
        validation_data=dataset,
        epochs=2,
        callbacks=callbacks,
        verbose=1,
    )

    output = capsys.readouterr().out
    assert "Epoch 1/2" in output
    assert "Epoch 2/2" in output
    assert "1/1 [" in output
    assert "Epoch 1: val_loss" in output
    assert " - examples_per_second:" in output
    assert "accuracy:" in output
    assert "val_loss:" in output
    assert "val_accuracy:" in output
    assert "lr:" in output
    assert "examples_per_second:" in output
    assert checkpoint_path.exists()


def test_training_log_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="training.log_mode"):
        get_epoch_log_mode({"training": {"log_mode": "compact"}})


class _FakeStdout:
    def __init__(self, interactive):
        self._interactive = interactive

    def isatty(self):
        return self._interactive


class _BrokenStdout:
    def isatty(self):
        raise ValueError("I/O operation on closed file")


def test_stdout_is_interactive_tolerates_unusual_streams():
    assert _stdout_is_interactive(_FakeStdout(True)) is True
    assert _stdout_is_interactive(_FakeStdout(False)) is False
    # Streams without isatty() (some notebook/kernel replacements) and closed
    # streams must both degrade to the safe newline-terminated path.
    assert _stdout_is_interactive(object()) is False
    assert _stdout_is_interactive(_BrokenStdout()) is False


@pytest.mark.parametrize(
    ("training_section", "expected"),
    [
        ({"progress_log_interval": "summary-only"}, 0),
        ({"progress_log_interval": 32}, 32),
        ({"progress_log_interval": "64"}, 64),
        ({"progress_log_interval": None}, 0),
        ({}, 0),
    ],
)
def test_progress_log_interval_ignores_legacy_string_values(
    training_section, expected
):
    assert _progress_log_interval({"training": training_section}) == expected


def test_piped_stdout_uses_newline_safe_progress_logging(monkeypatch):
    import training.train as train_module

    monkeypatch.setattr(train_module, "_stdout_is_interactive", lambda *_: False)
    existing = train_module.keras.callbacks.EarlyStopping()
    messages = []
    logger = SimpleNamespace(info=lambda *args: messages.append(args))

    callbacks, verbosity = _configure_progress_logging(
        [existing],
        {"training": {}, "performance": {"steps_per_execution": 8}},
        "progress",
        700,
        151,
        logger=logger,
    )

    assert verbosity == 0
    assert not any(
        isinstance(callback, train_module.keras.callbacks.ProgbarLogger)
        for callback in callbacks
    )
    progress = callbacks[0]
    assert isinstance(progress, BatchProgressLogger)
    assert progress.interval == 64
    assert progress.steps_per_execution == 8
    assert progress.train_steps == 700
    assert progress.validation_steps == 151
    assert existing in callbacks
    assert messages and "newline-safe progress logging" in messages[0][0]


def _inline_config(style="inline"):
    return {
        "training": {"progress_style": style},
        "performance": {"steps_per_execution": 8},
    }


def test_inline_progress_style_replaces_newline_per_batch_logging(monkeypatch):
    import training.train as train_module

    monkeypatch.setattr(train_module, "_stdout_is_interactive", lambda *_: False)
    existing = train_module.keras.callbacks.EarlyStopping()

    callbacks, verbosity = _configure_progress_logging(
        [existing],
        _inline_config(),
        "progress",
        700,
        151,
    )

    assert verbosity == 0
    progress = callbacks[0]
    assert isinstance(progress, InlineProgressBar)
    assert not any(isinstance(cb, BatchProgressLogger) for cb in callbacks)
    assert progress.train_steps == 700
    assert progress.validation_steps == 151
    assert existing in callbacks


def test_inline_progress_style_is_opt_in_and_validated():
    # Plain non-notebook runs keep the newline-safe output, because carriage
    # returns collapse into unreadable re-renders in ordinary log files.
    assert _progress_style({}) == "lines"
    assert _progress_style({"training": {}}) == "lines"
    assert _progress_style(_inline_config()) == "inline"
    assert _progress_style({"training": {"progress_style": "  INLINE "}}) == "inline"
    assert _progress_style({"training": {"progress_style": "lines"}}) == "lines"
    with pytest.raises(ValueError, match="training.progress_style"):
        _progress_style({"training": {"progress_style": "compact"}})


def test_inline_progress_repaints_one_line_per_epoch(capsys):
    model = tf.keras.Sequential(
        [tf.keras.layers.InputLayer((4,)), tf.keras.layers.Dense(2, activation="softmax")]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    features = tf.reshape(tf.range(32 * 4, dtype=tf.float32) / 100.0, (32, 4))
    labels = tf.one_hot(tf.range(32) % 2, depth=2)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(8)
    progress = InlineProgressBar(
        train_steps=4, validation_steps=2, min_interval=0.0
    )

    model.fit(
        dataset,
        validation_data=dataset,
        epochs=2,
        steps_per_epoch=4,
        validation_steps=2,
        callbacks=[progress],
        verbose=0,
    )

    output = capsys.readouterr().out
    # One newline per epoch: the live status line is repainted with "\r".
    assert output.count("\n") == 2
    assert output.count("\r") >= 4
    assert "Train progress:" not in output
    assert "Epoch 1/2" in output
    assert "Epoch 2/2" in output
    # The rendered line reports a fixed-width bar, ETA, and live metrics.
    assert "[" in output and "]" in output
    assert "loss:" in output
    assert "accuracy:" in output
    assert "val_loss:" in output
    assert "val_accuracy:" in output

    # The epoch's last repaint is the completed-epoch line: full bar plus both
    # train and validation metrics, with no leftover step counter.
    epoch_lines = [line for line in output.split("\n") if line.strip()]
    assert len(epoch_lines) == 2
    for index, line in enumerate(epoch_lines):
        final = line.split("\r")[-1]
        assert final.startswith(f"Epoch {index + 1}/2 [====================] -")
        assert "loss:" in final
        assert "accuracy:" in final
        assert "val_loss:" in final
        assert "val_accuracy:" in final


def test_inline_progress_pads_to_clear_stale_characters(capsys):
    progress = InlineProgressBar(train_steps=10, validation_steps=0, min_interval=0.0)
    progress.params = {"epochs": 1}
    progress.on_epoch_begin(0)

    # Force a long first render, then a short one that must overwrite it.
    progress._last_length = 0
    progress._at_line_start = True
    progress._write("X" * 40)
    progress._last_length = 40
    progress._at_line_start = False
    capsys.readouterr()

    progress._completed = 1
    progress._phase = "train"
    progress.status_text = lambda logs=None: "short"
    progress._render({}, force=True)

    line = capsys.readouterr().out
    assert line.startswith("\r")
    assert line == "\r" + "short" + " " * (40 - len("short"))


def test_inline_progress_status_reports_extra_metrics():
    progress = InlineProgressBar(train_steps=10, validation_steps=5, min_interval=0.0)
    progress.params = {"epochs": 3}
    progress.on_epoch_begin(0)
    progress._completed = 4

    status = progress.status_text(
        {"loss": 4.7198, "accuracy": 0.1074, "top_k_categorical_accuracy": 0.42}
    )
    assert "loss: 4.7198" in status
    assert "accuracy: 0.1074" in status
    assert "top5_acc: 0.4200" in status

    progress._phase = "validation"
    validation_status = progress.status_text(
        {"loss": 4.1, "accuracy": 0.2, "top_k_categorical_accuracy": 0.6}
    )
    assert "val_loss: 4.1000" in validation_status
    assert "val_accuracy: 0.2000" in validation_status
    assert "val_top5_acc: 0.6000" in validation_status


def test_inline_progress_final_line_uses_validation_values():
    # The completed-epoch line must pull each val metric from the val_-prefixed
    # epoch log, not re-render the train value under a val_ label, and it must
    # list train metrics first, then validation metrics.
    progress = InlineProgressBar(train_steps=175, validation_steps=38)
    progress.params = {"epochs": 20}
    progress.on_epoch_begin(1)
    text = progress._final_text(
        {
            "loss": 3.6636,
            "accuracy": 0.1861,
            "val_loss": 3.2269,
            "val_accuracy": 0.2831,
        }
    )
    assert text == (
        "Epoch 2/20 [====================] - loss: 3.6636 - accuracy: 0.1861"
        " - val_loss: 3.2269 - val_accuracy: 0.2831"
    )


def test_inline_progress_drops_the_redundant_epoch_heading(capsys):
    # The bar already names the epoch, so "Epoch 1/1 started" would be a
    # duplicate line above the very line it describes.
    inline_logger = EpochThroughputLogger(
        samples=4, log_mode="summary", progress_style="inline"
    )
    inline_logger.params = {"epochs": 1}
    inline_logger.model = SimpleNamespace(optimizer=SimpleNamespace(learning_rate=1e-3))
    inline_logger.on_epoch_begin(0)
    assert capsys.readouterr().out == ""

    # Plain log-file runs keep the heading, which is all they get.
    lines_logger = EpochThroughputLogger(
        samples=4, log_mode="summary", progress_style="lines"
    )
    lines_logger.params = {"epochs": 1}
    lines_logger.model = SimpleNamespace(optimizer=SimpleNamespace(learning_rate=1e-3))
    lines_logger.on_epoch_begin(0)
    assert "Epoch 1/1 started" in capsys.readouterr().out


def test_inline_progress_silences_the_learning_rate_scheduler(capsys):
    base_config = {
        "model": {"num_classes": 2},
        "optimizer": {"learning_rate": 1e-3},
        "dataset": {"batch_size": 2},
        "performance": {"terminate_on_nan": False},
        "lr_schedule": {"type": "cosine_decay", "decay_epochs": 10},
        "callbacks": {
            "checkpoint": {"enabled": False},
            "early_stopping": {"enabled": False},
            "tensorboard": {"enabled": False},
            "csv_logger": {"enabled": False},
        },
    }

    def scheduler_verbose(progress_style):
        config = {
            **base_config,
            "training": {"log_mode": "progress", "progress_style": progress_style},
        }
        callbacks = setup_callbacks(config, include_throughput=False)
        scheduler = next(
            callback
            for callback in callbacks
            if isinstance(callback, tf.keras.callbacks.LearningRateScheduler)
        )
        # Keras prints the "Epoch N: LearningRateScheduler setting learning
        # rate to ..." line only when verbose is set.
        return getattr(scheduler, "verbose", 0)

    assert scheduler_verbose("inline") == 0
    assert scheduler_verbose("lines") == 1


def test_piped_stdout_respects_configured_progress_interval(monkeypatch):
    import training.train as train_module

    monkeypatch.setattr(train_module, "_stdout_is_interactive", lambda *_: False)

    callbacks, verbosity = _configure_progress_logging(
        [],
        {"training": {"progress_log_interval": 128}},
        "progress",
        700,
        151,
    )

    assert verbosity == 0
    assert callbacks[0].interval == 128


def test_interactive_stdout_keeps_the_live_keras_progress_bar(monkeypatch):
    import training.train as train_module

    monkeypatch.setattr(train_module, "_stdout_is_interactive", lambda *_: True)

    callbacks, verbosity = _configure_progress_logging(
        [], {"training": {}}, "progress", 700, 151
    )

    assert verbosity == 1
    assert isinstance(callbacks[-1], train_module.keras.callbacks.ProgbarLogger)
    assert not any(isinstance(cb, BatchProgressLogger) for cb in callbacks)


def test_interactive_summary_mode_adds_sparse_progress_when_configured(monkeypatch):
    import training.train as train_module

    monkeypatch.setattr(train_module, "_stdout_is_interactive", lambda *_: True)

    callbacks, verbosity = _configure_progress_logging(
        [], {"training": {"progress_log_interval": 32}}, "summary", 700, 151
    )

    assert verbosity == 0
    assert isinstance(callbacks[0], BatchProgressLogger)
    assert callbacks[0].interval == 32
    assert not any(
        isinstance(callback, train_module.keras.callbacks.ProgbarLogger)
        for callback in callbacks
    )


def test_diagnostic_runs_keep_their_own_progress_logging(monkeypatch):
    import training.train as train_module

    monkeypatch.setattr(train_module, "_stdout_is_interactive", lambda *_: False)
    existing = train_module.keras.callbacks.EarlyStopping()

    callbacks, verbosity = _configure_progress_logging(
        [existing],
        {"training": {}},
        "summary",
        10,
        5,
        diagnostic_run=True,
    )

    assert callbacks == [existing]
    assert verbosity == 0


def test_native_keras_export_roundtrips(tmp_path):
    model = tf.keras.Sequential(
        [tf.keras.layers.InputLayer((4,)), tf.keras.layers.Dense(2, activation="softmax")]
    )

    path = _save_native_keras_model(model, str(tmp_path), "tiny", "20260101_000000")

    assert path == str(tmp_path / "tiny" / "tiny_20260101_000000.keras")
    assert os.path.exists(path)
    restored = tf.keras.models.load_model(path)
    assert restored.output_shape == (None, 2)


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


def test_degenerate_cutoff_is_reported(caplog):
    """A cutoff above the largest class silently disables baseline augmentation."""
    # Shape of the cap-500 benchmark: most classes sit at/near the cap and a
    # handful of Guava classes are the only real minorities.
    counts = {f"class_{index}": 350 for index in range(67)}
    counts.update({f"guava_{index}": value for index, value in enumerate([2, 2, 6, 3, 3])})

    with caplog.at_level(logging.WARNING, logger="training.utils"):
        degenerate = get_minority_class_indices(
            counts, max_train_samples=500, logger=logging.getLogger("training.utils")
        )
    assert len(degenerate) == len(counts) == 72
    assert "baseline augmentation is effectively disabled" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="training.utils"):
        selected = get_minority_class_indices(
            counts, max_train_samples=100, logger=logging.getLogger("training.utils")
        )
    # Only the five Guava classes (2, 2, 6, 3, 3) fall below the cutoff.
    assert len(selected) == 5
    assert "baseline augmentation is effectively disabled" not in caplog.text


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
        cache_mode="memory",
        cache_dtype="float16",
    )

    train_batches = list(train_ds.as_numpy_iterator())
    val_batches = list(val_ds.as_numpy_iterator())
    train_images, train_labels = train_batches[0]
    values_by_label = {
        int(label): float(tf.reduce_mean(train_images[index]).numpy())
        for index, label in enumerate(tf.argmax(train_labels, axis=-1).numpy())
    }
    val_images, val_labels = val_batches[0]

    assert class_names == ["common", "rare"]
    assert values_by_label == {0: 1.0, 1: 10.0}
    assert train_images.dtype == np.float32
    assert val_images.dtype == np.float32
    assert float(np.max(val_images)) == 0.0
    assert np.array_equal(train_labels, val_labels)


def test_class_aware_pipeline_supports_mixed_float16(tmp_path):
    previous_policy = tf.keras.mixed_precision.global_policy()
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
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
        train_ds, _, class_names = load_dataset(
            tmp_path / "train",
            tmp_path / "val",
            batch_size=2,
            image_size=(4, 4),
            augmentation=baseline,
            minority_augmentation=minority,
            minority_class_indices=[1],
            shuffle_buffer=0,
            cache_mode="memory",
            cache_dtype="float16",
        )

        train_images, train_labels = next(iter(train_ds))
        values_by_label = {
            int(label): float(tf.reduce_mean(train_images[index]).numpy())
            for index, label in enumerate(tf.argmax(train_labels, axis=-1).numpy())
        }

        assert class_names == ["common", "rare"]
        assert values_by_label == pytest.approx({0: 1.0, 1: 10.0})
        assert bool(tf.reduce_all(tf.math.is_finite(train_images)).numpy())
    finally:
        tf.keras.mixed_precision.set_global_policy(previous_policy)


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


def test_training_precision_recall_are_macro_averaged():
    """precision/recall must be macro, not the micro average that equals accuracy.

    tf_keras's Precision/Recall have no "average" argument, so the defaults
    collapse to accuracy on a single-label multiclass fit. The custom metric
    averages over classes that were actually predicted or supported, which is
    what evaluate.py reports.
    """
    metrics = build_metrics(["accuracy", "precision", "recall"], 3)

    assert [metric.name for metric in metrics] == ["accuracy", "precision", "recall"]

    precision, recall = metrics[1], metrics[2]
    # Three samples: 0->0 (right), 1->0 (wrong), 2->2 (right).
    # Precision: class 0 is predicted twice and right once -> 0.5; class 1 is
    # never predicted so it drops out of the mean; class 2 is right once -> 1.0.
    # Macro precision is (0.5 + 1.0) / 2 = 0.75, where the micro average is the
    # 2/3 accuracy of this batch.
    precision.update_state(
        tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        tf.constant([[0.9, 0.1, 0.0], [0.9, 0.1, 0.0], [0.0, 0.1, 0.9]]),
    )
    assert float(precision.result()) == pytest.approx(0.75)
    assert float(precision.result()) != pytest.approx(2 / 3)

    # Recall counts by support: 1/1, 0/1, 1/1 -> 2/3 for all three classes.
    recall.update_state(
        tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        tf.constant([[0.9, 0.1, 0.0], [0.9, 0.1, 0.0], [0.0, 0.1, 0.9]]),
    )
    assert float(recall.result()) == pytest.approx(2 / 3)

    precision.reset_state()
    assert float(precision.result()) == 0.0

    # The metric is saved inside the .keras/.h5 artifacts, so it must round-trip
    # through get_config; an unregistered class makes load_model fail afterwards.
    clone = precision.__class__.from_config(precision.get_config())
    assert (clone.num_classes, clone.kind, clone.name) == (3, "precision", "precision")


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


def test_estimated_counts_mark_unscanned_split_classes_as_unknown():
    counts = _estimated_split_counts(
        train_steps=2,
        val_steps=1,
        batch_size=2,
        class_names=["healthy", "disease"],
    )

    assert counts["val"]["total"] == 2
    assert counts["val"]["classes"] == {"healthy": None, "disease": None}
    assert counts["val"]["estimated"] is True
    assert counts["test"]["total"] is None


def test_benchmark_class_names_prefer_dataset_metadata(tmp_path):
    class_names = ["Crop___Disease", "Crop___Healthy"]
    (tmp_path / "dataset_info.json").write_text(
        json.dumps({"class_names": class_names}),
        encoding="utf-8",
    )

    assert load_benchmark_class_names(str(tmp_path)) == class_names


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


def test_split_validation_tolerates_registered_degenerate_reuse(tmp_path):
    """Benchmark builders disclose 1-2 unique-image classes in
    dataset_info.json; their one shared copy per split must not abort an
    otherwise manifest-backed strict validation run."""
    class_names = ["healthy", "Guava___Anthracnose"]
    for split in ("train", "val", "test"):
        for class_name in class_names:
            (tmp_path / split / class_name).mkdir(parents=True)

    rows = [
        ("train", "healthy", "h-train"),
        ("val", "healthy", "h-val"),
        ("test", "healthy", "h-test"),
        ("train", "Guava___Anthracnose", "g-shared"),
        ("val", "Guava___Anthracnose", "g-shared"),
        ("test", "Guava___Anthracnose", "g-shared"),
    ]
    with (tmp_path / "split_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["split", "canonical_label", "source_sha256"]
        )
        writer.writeheader()
        for split, label, digest in rows:
            writer.writerow(
                {"split": split, "canonical_label": label, "source_sha256": digest}
            )

    with pytest.raises(ValueError, match="exact duplicate image hashes"):
        validate_dataset_splits(
            tmp_path / "train", tmp_path / "val", tmp_path / "test",
            class_names=class_names,
        )

    (tmp_path / "dataset_info.json").write_text(
        json.dumps({"degenerate_classes": {"Guava___Anthracnose": 1}}),
        encoding="utf-8",
    )
    counts = validate_dataset_splits(
        tmp_path / "train", tmp_path / "val", tmp_path / "test",
        class_names=class_names,
    )
    assert counts["train"]["classes"] == {"healthy": 1, "Guava___Anthracnose": 1}
    assert counts["test"]["classes"]["Guava___Anthracnose"] == 1


def test_split_validation_logs_hashing_progress_without_manifest(tmp_path, caplog, monkeypatch):
    """Hosted runs without split_manifest.csv hash every image in silence;
    the scan must emit a visible heartbeat so the wait is diagnosable."""
    import training.utils as utils_module

    monkeypatch.setattr(utils_module, "_HASH_PROGRESS_INTERVAL", 1)
    for split in ("train", "val", "test"):
        for class_name in ("healthy", "disease"):
            class_dir = tmp_path / split / class_name
            class_dir.mkdir(parents=True)
            (class_dir / "sample.jpg").write_bytes(
                f"{split}-{class_name}".encode("utf-8")
            )

    with caplog.at_level(logging.INFO, logger="training.utils"):
        counts = validate_dataset_splits(
            tmp_path / "train", tmp_path / "val", tmp_path / "test"
        )

    assert counts["train"]["total"] == 2
    assert "Hashed 1 images" in caplog.text
    assert "current split:" in caplog.text
    assert "Finished hashing 6 images" in caplog.text


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


def test_split_validation_allows_cross_split_duplicates_when_opted_in(tmp_path, caplog):
    """Hosted datasets (e.g. Kaggle inputs) can contain genuine cross-split
    duplicates. The strict default must still raise; the opt-in must warn and
    return counts so training can proceed."""
    for split in ("train", "val"):
        for class_name in CANONICAL_CLASSES:
            (tmp_path / split / class_name).mkdir(parents=True)
            (tmp_path / split / class_name / "sample.jpg").write_bytes(
                f"{split}-{class_name}".encode("utf-8")
            )
    (tmp_path / "train" / CANONICAL_CLASSES[0] / "sample.jpg").write_bytes(b"dup")
    (tmp_path / "val" / CANONICAL_CLASSES[0] / "sample.jpg").write_bytes(b"dup")

    with caplog.at_level(logging.WARNING, logger="training.utils"):
        counts = validate_dataset_splits(
            tmp_path / "train",
            tmp_path / "val",
            allow_cross_split_duplicates=True,
        )
    assert counts["train"]["total"] == len(CANONICAL_CLASSES)
    assert counts["val"]["total"] == len(CANONICAL_CLASSES)
    assert "leakage allowed" in caplog.text

    with pytest.raises(ValueError, match="exact duplicate image hashes"):
        validate_dataset_splits(tmp_path / "train", tmp_path / "val")


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


def test_summary_lines_are_consolidated_instead_of_raw_dumps():
    from training.train import _summarize_split_counts

    detailed = _summarize_split_counts(
        {
            "train": {"total": 22382, "classes": {"a": 350, "b": 2}},
            "val": {"total": 4801, "classes": {"a": 75}},
            "test": {"total": 4797, "classes": {}},
        }
    )
    assert detailed.startswith("Split totals: train=22382, val=4801, test=4797")
    assert "(per-class 2-350;" in detailed
    assert "{" not in detailed

    estimated = _summarize_split_counts(
        {
            "train": {"total": 22382, "classes": {}},
            "val": {"total": 4801, "classes": {"a": None}},
            "test": {"total": None, "classes": {}},
        },
        estimated=True,
    )
    assert estimated.startswith("Split totals: train=22382, val=4801, test=?")
    assert "estimated" in estimated


def test_run_header_and_summary_report_the_operational_facts():
    from training.train import _log_run_header, _log_run_summary

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer((4,)),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
    config = {
        "model": {
            "architecture": "probe_net",
            "input_shape": [192, 192, 3],
            "dropout_rate": 0.3,
            "head_units": [256, 128],
        },
        "training": {"fine_tune_learning_rate": 2e-5},
        "performance": {"steps_per_execution": 32, "jit_compile": False},
        "optimizer": {"learning_rate": 3e-4},
        "loss": {"name": "categorical_crossentropy"},
        "dataset": {"batch_size": 128},
        "metrics": ["accuracy", "auc"],
    }
    records = []

    class _Recorder:
        def info(self, message, *args, **_kwargs):
            records.append(message % args if args else message)

    _log_run_header(
        _Recorder(),
        config,
        model,
        "/models/probe_net/runs/20261107_141805",
        {"num_replicas": 2},
        175,
        38,
        22382,
        20,
        30,
        class_weights={"a": 0.95, "b": 5.0},
    )
    header = "\n".join(records)
    assert "Run: probe_net @ 20261107_141805" in header
    assert "input 192x192x3" in header
    assert "replicas=2" in header
    assert "20 frozen + 10 fine-tune epochs" in header
    assert "train=175, val=38" in header
    assert "22,382 samples/epoch" in header
    assert "applied range [0.9500, 5.0000]" in header

    records.clear()
    _log_run_summary(
        _Recorder(),
        {
            "loss": [4.3, 3.65, 3.01],
            "val_loss": [3.95, 3.28, 2.85],
            "val_accuracy": [0.19, 0.27, 0.35],
            "examples_per_second": [50.3, 166.5, 162.6],
        },
        "phase_2",
        2.84999,
        3725.0,
        {"saved_model": "/models/probe_net/saved_model_x"},
    )
    summary = "\n".join(records)
    assert "weights=phase_2" in summary
    assert "best_epoch=3" in summary
    assert "steady_throughput=164.6 img/s" in summary
    # Epoch 1 is reported separately because it includes cache warm-up.
    assert "first_epoch=50.3 img/s" in summary
    assert "Artifact saved_model: /models/probe_net/saved_model_x" in summary

def test_repeated_log_lines_are_collapsed_within_the_window():
    from training.utils import _RepeatedLineFilter

    record = logging.LogRecord(
        "tensorflow", logging.INFO, __file__, 1, "Collective all_reduce tensors", None, None
    )
    repeat_filter = _RepeatedLineFilter(window_seconds=30.0)

    assert repeat_filter.filter(record) is True
    assert repeat_filter.filter(record) is False

    other = logging.LogRecord(
        "tensorflow", logging.INFO, __file__, 1, "Reduce to CPU then broadcast", None, None
    )
    assert repeat_filter.filter(other) is True

    # A disabled window must never drop anything.
    passthrough = _RepeatedLineFilter(window_seconds=0)
    assert passthrough.filter(record) is True
    assert passthrough.filter(record) is True


def test_transient_load_warnings_are_unaffected_by_the_repeat_filter():
    from training.utils import _RepeatedLineFilter

    repeat_filter = _RepeatedLineFilter(window_seconds=30.0)
    record = logging.LogRecord(
        "training.utils", logging.WARNING, __file__, 1, "GPU prefetch unavailable", None, None
    )
    assert repeat_filter.filter(record) is True


def test_inline_progress_reports_single_line_with_lr_and_throughput(tmp_path):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer((4,)),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    dataset = tf.data.Dataset.from_tensors(
        (np.zeros((4, 4), dtype=np.float32), np.array([[1.0, 0.0]] * 4, dtype=np.float32))
    ).batch(4)
    stream = io.StringIO()
    bar = InlineProgressBar(
        train_steps=1, validation_steps=1, min_interval=0.0, stream=stream, samples=4
    )
    logger = EpochThroughputLogger(samples=4, log_mode="summary", progress_style="inline")

    model.fit(
        dataset,
        validation_data=dataset,
        epochs=1,
        callbacks=[bar, logger],
        verbose=0,
    )

    output = stream.getvalue()
    # The bar repaints in place, so only the newline-terminated records are
    # separate log lines; exactly one must be produced for the epoch.
    assert output.count("\n") == 1
    assert output.endswith("\n")
    final_line = output.rstrip("\n").split("\r")[-1]
    assert "[====================]" in final_line
    assert "lr: 3.000e-04" in final_line
    assert "examples_per_second:" in final_line

def test_minority_augmentation_runs_only_when_the_batch_contains_a_minority_label(
    tmp_path,
):
    """A NaN-emitting minority stack makes the tf.cond gate directly observable.

    If the minority stack ran on a pure-majority batch, the zero mask would
    still multiply its NaN output into the blend (NaN * 0 == NaN), so a clean
    batch proves the branch was skipped rather than merely masked out.
    """
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

    # Minority label 0 is present in the batch, so the minority stack must run
    # and replace only the minority image of the blend.
    train_ds, _val_ds, class_names = load_dataset(
        tmp_path / "train",
        tmp_path / "val",
        batch_size=2,
        image_size=(4, 4),
        augmentation=baseline,
        minority_augmentation=minority,
        minority_class_indices=[0],
        shuffle_buffer=0,
        cache_mode="memory",
        cache_dtype="float32",
    )
    batch_images, batch_labels = next(iter(train_ds))
    labels = [int(label) for label in tf.argmax(batch_labels, axis=-1).numpy()]
    values_by_label = {
        label: float(tf.reduce_mean(batch_images[index]).numpy())
        for index, label in enumerate(labels)
    }
    assert class_names == ["common", "rare"]
    assert values_by_label == {0: 10.0, 1: 1.0}


def test_minority_augmentation_is_skipped_for_pure_majority_batches(tmp_path):
    for class_name in ("common", "rare"):
        (tmp_path / "train" / class_name).mkdir(parents=True)
        (tmp_path / "val" / class_name).mkdir(parents=True)
    # image_dataset_from_directory yields classes in name order, so a 4/1 split
    # puts four "common" images first and the whole first batch is majority-only.
    for index in range(4):
        tf.io.write_file(
            str(tmp_path / "train" / "common" / f"sample{index}.png"),
            tf.io.encode_png(tf.zeros((4, 4, 3), dtype=tf.uint8)),
        )
    tf.io.write_file(
        str(tmp_path / "train" / "rare" / "sample.png"),
        tf.io.encode_png(tf.zeros((4, 4, 3), dtype=tf.uint8)),
    )
    for class_name in ("common", "rare"):
        tf.io.write_file(
            str(tmp_path / "val" / class_name / "sample.png"),
            tf.io.encode_png(tf.zeros((4, 4, 3), dtype=tf.uint8)),
        )

    baseline = tf.keras.layers.Lambda(lambda x: x + 1.0)
    minority = tf.keras.layers.Lambda(lambda x: x * np.nan)

    train_ds, _val_ds, _class_names = load_dataset(
        tmp_path / "train",
        tmp_path / "val",
        batch_size=2,
        image_size=(4, 4),
        augmentation=baseline,
        minority_augmentation=minority,
        minority_class_indices=[1],
        shuffle_buffer=0,
        cache_mode="memory",
        cache_dtype="float32",
    )

    batch_images, batch_labels = next(iter(train_ds))
    labels = [int(label) for label in tf.argmax(batch_labels, axis=-1).numpy()]

    # No minority image is in this batch, so no NaN may leak into the blend.
    assert not np.isnan(batch_images.numpy()).any()
    assert labels == [0, 0]
    np.testing.assert_allclose(batch_images.numpy(), 1.0, rtol=1e-6, atol=1e-6)


def test_cache_record_counts_examples_for_unbatched_datasets(monkeypatch):
    """The cache is resolved before batching, so cardinality already counts examples.

    ``load_dataset`` calls ``_resolve_cache_record`` while its datasets are still
    unbatched (``image_dataset_from_directory(batch_size=None)``). Multiplying
    that cardinality by ``batch_size`` again inflated a 7 GB estimate into
    897 GB and silently disabled the RAM cache for a run that fit the limit.
    """
    from training.utils import _resolve_cache_record

    unbatched = tf.data.Dataset.from_tensor_slices(
        (tf.zeros((8, 4, 4, 3)), tf.zeros((8, 3)))
    )

    monkeypatch.setattr("training.utils._available_host_memory_gb", lambda: 64.0)
    monkeypatch.setattr("training.utils._available_disk_gb", lambda _path: 200.0)

    record = _resolve_cache_record(
        "memory",
        unbatched,
        unbatched,
        (4, 4),
        128,
        memory_limit_gb=8.0,
        cache_dtype="float16",
    )

    # 16 examples * (4*4*3*2 + 3*4) bytes * 1.25 headroom is far below 8 MB.
    assert record["estimate_gb"] <= 0.01
    assert record["mode"] == "memory", record["reason"]


def test_cache_record_still_multiplies_for_batched_datasets(monkeypatch):
    """A batched dataset reports batch cardinality, so it must be multiplied."""
    from training.utils import _example_count

    batched = tf.data.Dataset.from_tensor_slices(
        (tf.zeros((8, 4, 4, 3)), tf.zeros((8, 3)))
    ).batch(4)
    unbatched = tf.data.Dataset.from_tensor_slices(
        (tf.zeros((8, 4, 4, 3)), tf.zeros((8, 3)))
    )

    batched_cardinality = tf.data.experimental.cardinality(batched)
    unbatched_cardinality = tf.data.experimental.cardinality(unbatched)

    assert _example_count(batched, batched_cardinality, 4) == 8
    assert _example_count(unbatched, unbatched_cardinality, 4) == 8


def test_cache_record_explains_the_decision(monkeypatch):
    from training.utils import _resolve_cache_record

    dataset = tf.data.Dataset.from_tensors(
        (tf.zeros((2, 4, 4, 3)), tf.zeros((2, 3)))
    ).batch(2)
    monkeypatch.setattr(
        "training.utils._available_host_memory_gb", lambda: 64.0
    )
    monkeypatch.setattr("training.utils._available_disk_gb", lambda _path: 200.0)

    record = _resolve_cache_record(
        "auto", dataset, dataset, (4, 4), 2, memory_limit_gb=8.0, cache_dtype="float16"
    )
    assert record["mode"] == "memory"
    assert record["requested"] == "auto"
    assert record["limit_gb"] == 8.0
    assert record["estimate_gb"] >= 0
    assert record["reason"]

    # An explicit memory request that cannot fit must degrade instead of
    # silently claiming a cache mode the run cannot afford.
    oversized = _resolve_cache_record(
        "memory", dataset, dataset, (4, 4), 2, memory_limit_gb=1e-9, cache_dtype="float32"
    )
    assert oversized["mode"] != "memory"
    assert "exceeds" in oversized["reason"]

    disabled = _resolve_cache_record(
        "none", dataset, dataset, (4, 4), 2, memory_limit_gb=8.0
    )
    assert disabled["mode"] == "none"
    assert disabled["reason"] == "explicitly disabled"
