"""
Training Script for Multi-Crop Leaf Disease Detection
Uses transfer learning with MobileNetV2 or EfficientNet-Lite0
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
# Silence TensorFlow's C++ absl logging before the first import: otherwise every
# INFO notice ("Reduce to ... then broadcast", NCCL collectives, ...) is printed
# once by absl and once again by TensorFlow's Python logger.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import numpy as np
import subprocess
import sys
import time
import tensorflow as tf
import yaml
from tensorflow import keras
from datetime import datetime

try:
    # Support module execution: python -m training.train
    from .model import (
        get_model,
        unfreeze_base_model,
        unfreeze_backbone_for_scratch_training,
        print_model_summary,
    )
    from .export import make_float32_inference_model
    from .utils import (
        load_dataset,
        compute_class_weights_from_counts,
        create_class_aware_augmentation_layers,
        get_minority_class_indices,
        get_training_class_counts,
        get_preprocess_fn,
        save_model_checkpoint,
        setup_callbacks,
        setup_logging,
        save_class_names,
        validate_dataset_splits,
        choose_batch_size,
        validate_dataset_batch,
        get_epoch_log_mode,
        get_progress_style,
    )
except ImportError:
    # Fallback for script execution: python training/train.py
    from model import (
        get_model,
        unfreeze_base_model,
        unfreeze_backbone_for_scratch_training,
        print_model_summary,
    )
    from export import make_float32_inference_model
    from utils import (
        load_dataset,
        compute_class_weights_from_counts,
        create_class_aware_augmentation_layers,
        get_minority_class_indices,
        get_training_class_counts,
        get_preprocess_fn,
        save_model_checkpoint,
        setup_callbacks,
        setup_logging,
        save_class_names,
        validate_dataset_splits,
        choose_batch_size,
        validate_dataset_batch,
        get_epoch_log_mode,
        get_progress_style,
    )


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def _resolve_path(path_value, base_dir):
    """Resolve relative paths against base_dir while preserving absolute paths."""
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return os.path.normpath(path_value)
    return os.path.normpath(os.path.join(base_dir, path_value))


def resolve_config_paths(config, config_path):
    """Resolve path-like config values relative to the config file directory."""
    config_dir = os.path.dirname(os.path.abspath(config_path))

    # Dataset directories
    dataset_cfg = config.get("dataset", {})
    for key in ["data_dir", "train_dir", "val_dir", "test_dir"]:
        if key in dataset_cfg:
            dataset_cfg[key] = _resolve_path(dataset_cfg[key], config_dir)

    # Export directory
    export_cfg = config.get("export", {})
    if "save_dir" in export_cfg:
        export_cfg["save_dir"] = _resolve_path(export_cfg["save_dir"], config_dir)

    # Logging file paths
    logging_cfg = config.get("logging", {})
    if "log_file" in logging_cfg:
        logging_cfg["log_file"] = _resolve_path(logging_cfg["log_file"], config_dir)

    callbacks_cfg = config.get("callbacks", {})
    tensorboard_cfg = callbacks_cfg.get("tensorboard", {})
    if "log_dir" in tensorboard_cfg:
        tensorboard_cfg["log_dir"] = _resolve_path(
            tensorboard_cfg["log_dir"], config_dir
        )
    checkpoint_cfg = callbacks_cfg.get("checkpoint", {})
    if checkpoint_cfg.get("directory"):
        checkpoint_cfg["directory"] = _resolve_path(
            checkpoint_cfg["directory"], config_dir
        )

    csv_logger_cfg = callbacks_cfg.get("csv_logger", {})
    if "filename" in csv_logger_cfg:
        csv_logger_cfg["filename"] = _resolve_path(
            csv_logger_cfg["filename"], config_dir
        )

    return config


def build_metrics(metric_names, num_classes):
    """Build Keras metrics for multiclass classification."""
    built_metrics = []

    for metric_name in metric_names:
        if not isinstance(metric_name, str):
            built_metrics.append(metric_name)
            continue

        metric_key = metric_name.lower()
        if metric_key == "accuracy":
            built_metrics.append(keras.metrics.CategoricalAccuracy(name="accuracy"))
        elif metric_key in {"precision", "recall"}:
            # For a single-label multiclass problem, top-k precision/recall
            # collapse to accuracy. Per-class metrics are computed by evaluate.py.
            import warnings

            warnings.warn(
                f"Ignoring training metric '{metric_key}': top-1 multiclass "
                "precision/recall duplicates accuracy. Use the evaluator for "
                "macro and per-class metrics.",
                UserWarning,
                stacklevel=2,
            )
        elif metric_key == "auc":
            # Single-label softmax classifier: macro AUC over the 1-hot labels.
            # multi_label=True would misreport AUC for this task.
            built_metrics.append(keras.metrics.AUC(name="auc"))
        else:
            # Keep custom/unknown metric names to avoid breaking user-provided settings.
            built_metrics.append(metric_name)

    return built_metrics


class FocalLoss(keras.losses.Loss):
    """Focal loss for class-imbalanced single-label classification.

    ``FL(p_t) = -alpha_t * (1 - p_t) ** gamma * log(p_t)``

    ``alpha_t`` is the per-sample weight of the *true* class. For one-hot labels
    that reduces to the configured ``alpha`` scalar; for soft labels (e.g. from
    MixUp/CutMix) it interpolates between ``alpha`` and ``1 - alpha`` weighted by
    the soft targets, which keeps the loss on the same scale as a plain
    cross-entropy.

    Configurable via ``loss.gamma`` (default 2.0) and ``loss.alpha`` (default 0.25)
    in the YAML. Set ``loss.name: focal_loss`` to activate.

    Note: for balanced datasets prefer ``categorical_crossentropy`` with label
    smoothing — focal loss mainly helps when some classes are genuinely rare.
    """

    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Per-sample cross-entropy (handles label smoothing and log/eps clamping).
        # Returns shape (batch,) so Keras can apply its own reduction.
        cross_entropy = keras.losses.categorical_crossentropy(
            y_true, y_pred, label_smoothing=self.label_smoothing
        )

        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            targets = (
                y_true * (1.0 - self.label_smoothing)
                + self.label_smoothing / num_classes
            )
        else:
            targets = y_true

        # p_t is the probability mass placed on the true class(es). For one-hot
        # labels this is the softmax probability of the correct class.
        p_t = tf.reduce_sum(targets * y_pred, axis=-1)
        p_t = tf.clip_by_value(
            p_t, keras.backend.epsilon(), 1.0 - keras.backend.epsilon()
        )

        # alpha_t weights the true class. With a scalar alpha this is exactly
        # ``alpha`` for one-hot labels (NOT a sum over the other classes, which
        # would inflate the loss by a factor of ~C).
        focal = self.alpha * tf.pow(1.0 - p_t, self.gamma) * cross_entropy

        # Return per-sample losses; the base Loss applies sum_over_batch_size.
        return focal

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha,
                        "label_smoothing": self.label_smoothing})
        return config


class BestValidationWeights(keras.callbacks.Callback):
    """Keep an in-memory copy of the lowest finite validation-loss weights."""

    def __init__(self):
        super().__init__()
        self.best = float("inf")
        self.weights = None

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get("val_macro_f1", (logs or {}).get("val_loss"))
        if value is not None and np.isfinite(value) and value < self.best:
            self.best = float(value)
            self.weights = self.model.get_weights()


def select_final_weights(phase1_best, phase1_weights, phase2_best=None):
    """Return the best weights across phases, never degrading Phase 1.

    Fine-tuning is accepted only when its best finite validation score strictly
    improves on the best frozen-backbone validation loss.
    """
    if (
        phase2_best is not None
        and phase2_best.weights is not None
        and np.isfinite(phase2_best.best)
        and phase2_best.best < phase1_best.best
    ):
        return phase2_best.weights, "phase_2", phase2_best.best
    return phase1_weights, "phase_1", phase1_best.best


def phase_one_heading(freeze_base):
    """Return an accurate Phase 1 status line for the configured backbone."""
    if freeze_base:
        return "Phase 1: Training classifier head (base frozen)"
    return "Phase 1: Training full model (backbone trainable)"


def _fit_verbosity(log_mode):
    """Select Keras output verbosity for compact or live-progress logging."""
    return 0 if log_mode == "summary" else 1


def _add_keras_progress_logger(callbacks, log_mode):
    """Put Keras' progress bar last so end-of-epoch metrics include custom logs."""
    if log_mode == "progress":
        callbacks.append(
            keras.callbacks.ProgbarLogger(
                count_mode="steps",
                stateful_metrics={"lr", "examples_per_second"},
            )
        )
    return callbacks


def _stdout_is_interactive(stream=None):
    """Return True when stdout can repaint a progress bar in place.

    Hosted notebooks run training as a subprocess, so stdout is a pipe and
    Keras' carriage-return progress bar degenerates into hundreds of partial
    lines. Non-interactive streams therefore get newline-safe logging.
    """
    stream = sys.stdout if stream is None else stream
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def _progress_log_interval(config):
    """Return the sparse progress interval, ignoring legacy string values."""
    raw = config.get("training", {}).get("progress_log_interval", 0)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _progress_style(config):
    """Return how non-interactive progress is rendered.

    Delegates to :func:`training.utils.get_progress_style` so the validation
    rules live in exactly one place. ``inline`` repaints a single
    carriage-return line, which is what hosted notebook streams render as one
    continuously updating status line. ``lines`` keeps the newline-per-update
    output, which stays the default because carriage returns collapse into
    unreadable re-renders in ordinary log files.
    """
    return get_progress_style(config)


def _configure_progress_logging(
    callbacks,
    config,
    log_mode,
    train_steps,
    validation_steps,
    diagnostic_run=False,
    logger=None,
    sample_count=None,
):
    """Attach progress callbacks that survive a piped (non-TTY) stdout.

    Returns ``(callbacks, fit_verbose)``. Interactive terminals keep Keras'
    live progress bar; piped runs drop the carriage-return bar in favor of
    sparse, newline-terminated progress plus the compact per-epoch summary.
    """
    if diagnostic_run:
        return callbacks, _fit_verbosity(log_mode)

    steps_per_execution = config.get("performance", {}).get(
        "steps_per_execution", 1
    )
    if _stdout_is_interactive():
        progress_interval = _progress_log_interval(config)
        if progress_interval > 0 and log_mode != "progress":
            callbacks.insert(
                0,
                BatchProgressLogger(
                    train_steps=train_steps,
                    validation_steps=validation_steps,
                    interval=progress_interval,
                    steps_per_execution=steps_per_execution,
                ),
            )
        _add_keras_progress_logger(callbacks, log_mode)
        return callbacks, _fit_verbosity(log_mode)

    interval = _progress_log_interval(config) or 64
    if _progress_style(config) == "inline":
        callbacks.insert(
            0,
            InlineProgressBar(
                train_steps=train_steps,
                validation_steps=validation_steps,
                steps_per_execution=steps_per_execution,
                samples=sample_count,
            ),
        )
        if logger is not None:
            logger.info(
                "Non-interactive stdout: using a single in-place progress line "
                "(every %.2fs) instead of one line per batch",
                callbacks[0].min_interval,
            )
        return callbacks, _fit_verbosity("summary")

    callbacks.insert(
        0,
        BatchProgressLogger(
            train_steps=train_steps,
            validation_steps=validation_steps,
            interval=interval,
            steps_per_execution=steps_per_execution,
        ),
    )
    if logger is not None:
        logger.info(
            "Non-interactive stdout: using newline-safe progress logging "
            "(every %d batches) instead of the carriage-return progress bar",
            interval,
        )
    return callbacks, _fit_verbosity("summary")


def _resolve_fit_step_counts(train_steps, validation_steps, runtime_config):
    """Apply optional diagnostic caps without changing dataset cardinalities."""
    def resolve(value, name):
        if value is None:
            return train_steps if name == "training" else validation_steps
        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} step cap must be a positive integer") from exc
        if value <= 0:
            raise ValueError(f"{name} step cap must be a positive integer")
        return value

    return (
        resolve(runtime_config.get("max_train_steps"), "training"),
        resolve(runtime_config.get("max_validation_steps"), "validation"),
    )


def _fit_sample_count(
    full_train_count, fit_train_steps, batch_size, diagnostic=False
):
    """Return the examples processed by fit, without changing split metadata."""
    full_train_count = max(0, int(full_train_count))
    if not diagnostic:
        return full_train_count
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    fit_examples = max(0, int(fit_train_steps)) * batch_size
    return min(full_train_count, fit_examples)


def _should_skip_final_export(config):
    """Return whether a diagnostic run should stop after validating training."""
    return bool(config.get("runtime", {}).get("diagnostic", False))


def _save_native_keras_model(model, save_dir, model_name, timestamp, logger=None):
    """Save the native ``.keras`` artifact and return its path."""
    keras_path = os.path.join(
        save_dir, model_name, f"{model_name}_{timestamp}.keras"
    )
    os.makedirs(os.path.dirname(keras_path), exist_ok=True)
    model.save(keras_path)
    if logger is not None:
        logger.info(f"Saved Keras model: {keras_path}")
    return keras_path


def _git_revision(repo_dir):
    """Return the checked-out Git revision when the runtime has Git metadata."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            check=False,
            text=True,
        )
    except (FileNotFoundError, OSError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _model_summary_text(model):
    """Capture the Keras summary in a file-friendly form."""
    lines = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines) + "\n"


def save_run_metadata(
    config,
    config_path,
    model,
    raw_class_weights,
    applied_class_weights,
    split_counts,
    run_timestamp,
    minority_class_names=None,
):
    """Persist the exact pre-fit configuration and data/model context."""
    model_config = config.get("model", {})
    dataset_config = config.get("dataset", {})
    weight_config = config.get("class_weights", {})
    save_dir = config.get("export", {}).get("save_dir", "../models")
    model_name = model_config.get("architecture", "model")
    run_dir = os.path.join(save_dir, model_name, "runs", run_timestamp)
    os.makedirs(run_dir, exist_ok=True)

    with open(
        os.path.join(run_dir, "resolved_config.yaml"), "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    with open(
        os.path.join(run_dir, "model_summary.txt"), "w", encoding="utf-8"
    ) as handle:
        handle.write(_model_summary_text(model))

    raw_values = list((raw_class_weights or {}).values())
    applied_values = list((applied_class_weights or {}).values())
    repo_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    metadata = {
        "run_timestamp": run_timestamp,
        "source_config_path": os.path.abspath(config_path),
        "git_revision": _git_revision(repo_dir),
        "tensorflow_version": tf.__version__,
        "model": {
            "name": model.name,
            "total_parameters": int(model.count_params()),
            "trainable_parameters": int(
                sum(np.prod(variable.shape) for variable in model.trainable_weights)
            ),
            "non_trainable_parameters": int(
                sum(
                    np.prod(variable.shape) for variable in model.non_trainable_weights
                )
            ),
        },
        "datasets": {
            split: {
                "path": dataset_config.get(f"{split}_dir"),
                **split_counts.get(split, {"total": 0, "classes": {}}),
            }
            for split in ("train", "val", "test")
        },
        "class_weights": {
            "enabled": bool(weight_config.get("enabled", False)),
            "configured_max_weight": weight_config.get("max_weight"),
            "normalize": bool(weight_config.get("normalize", True)),
            "raw_range": (
                {"min": min(raw_values), "max": max(raw_values)} if raw_values else None
            ),
            "applied_range": (
                {"min": min(applied_values), "max": max(applied_values)}
                if applied_values
                else None
            ),
        },
        "class_aware_augmentation": {
            "enabled": bool(
                config.get("augmentation", {}).get("class_aware", {}).get("enabled", False)
            ),
            "max_train_samples": config.get("augmentation", {}).get(
                "class_aware", {}
            ).get("max_train_samples"),
            "class_names": list(minority_class_names or []),
        },
    }
    with open(
        os.path.join(run_dir, "run_metadata.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    return run_dir


def _get_loss(config):
    """Resolve the loss function from config.

    Supports:
      - ``categorical_crossentropy`` (weighted per example at ``model.fit``)
      - ``focal_loss`` with optional ``loss.gamma`` and ``loss.alpha``

    Both losses honour ``loss.label_smoothing``.
    """
    loss_cfg = config.get("loss", {})
    loss_name = loss_cfg.get("name", "categorical_crossentropy")
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0) or 0.0)

    if loss_name == "focal_loss":
        return FocalLoss(
            gamma=loss_cfg.get("gamma", 2.0),
            alpha=loss_cfg.get("alpha", 0.25),
            label_smoothing=label_smoothing,
        )

    if label_smoothing > 0:
        return keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing
        )

    return loss_name


def compile_model(model, config):
    """Compile model with optimizer and loss function"""

    # Setup optimizer
    optimizer_config = config["optimizer"]
    optimizer_name = optimizer_config.get("name", "adam").lower()
    lr = optimizer_config.get("learning_rate", 0.001)
    decay = optimizer_config.get("decay", 0.0)
    weight_decay = decay if decay and decay > 0 else None
    clipnorm = optimizer_config.get("clipnorm")
    clipvalue = optimizer_config.get("clipvalue")

    if optimizer_name == "adam":
        adam_kwargs = {"learning_rate": lr}
        if clipnorm:
            adam_kwargs["clipnorm"] = float(clipnorm)
        if clipvalue:
            adam_kwargs["clipvalue"] = float(clipvalue)
        for key in ["beta_1", "beta_2", "epsilon"]:
            if key in optimizer_config and optimizer_config[key] is not None:
                adam_kwargs[key] = optimizer_config[key]
        if weight_decay is not None:
            adam_kwargs["weight_decay"] = weight_decay
        optimizer = keras.optimizers.Adam(**adam_kwargs)
    elif optimizer_name == "sgd":
        sgd_kwargs = {
            "learning_rate": lr,
            "momentum": optimizer_config.get("momentum", 0.9),
        }
        if clipnorm:
            sgd_kwargs["clipnorm"] = float(clipnorm)
        if clipvalue:
            sgd_kwargs["clipvalue"] = float(clipvalue)
        if weight_decay is not None:
            sgd_kwargs["weight_decay"] = weight_decay
        optimizer = keras.optimizers.SGD(**sgd_kwargs)
    elif optimizer_name == "rmsprop":
        rmsprop_kwargs = {
            "learning_rate": lr,
            "momentum": optimizer_config.get("momentum", 0.0),
        }
        if clipnorm:
            rmsprop_kwargs["clipnorm"] = float(clipnorm)
        if clipvalue:
            rmsprop_kwargs["clipvalue"] = float(clipvalue)
        if weight_decay is not None:
            rmsprop_kwargs["weight_decay"] = weight_decay
        optimizer = keras.optimizers.RMSprop(**rmsprop_kwargs)
    else:
        raise ValueError(
            f"Unsupported optimizer '{optimizer_name}'. Supported: adam, sgd, rmsprop"
        )

    # Setup loss
    loss = _get_loss(config)

    # Setup metrics
    metric_names = config.get("metrics", ["accuracy"])
    metrics = build_metrics(metric_names, config["model"]["num_classes"])

    optimizer = _maybe_loss_scale_optimizer(optimizer)
    performance_config = config.get("performance", {})
    compile_options = {}
    if performance_config.get("steps_per_execution"):
        compile_options["steps_per_execution"] = int(
            performance_config["steps_per_execution"]
        )
    if "jit_compile" in performance_config:
        compile_options["jit_compile"] = bool(performance_config["jit_compile"])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **compile_options)

    return model


def compile_fine_tune_model(model, config, strategy):
    """Recompile a model for fine-tuning inside its distribution scope."""
    with strategy.scope():
        metrics = build_metrics(
            config.get("metrics", ["accuracy"]),
            config["model"]["num_classes"],
        )
        optimizer = _maybe_loss_scale_optimizer(create_fine_tune_optimizer(config))
        performance_config = config.get("performance", {})
        compile_options = {}
        if performance_config.get("steps_per_execution"):
            compile_options["steps_per_execution"] = int(
                performance_config["steps_per_execution"]
            )
        if "jit_compile" in performance_config:
            compile_options["jit_compile"] = bool(performance_config["jit_compile"])
        model.compile(
            optimizer=optimizer,
            loss=_get_loss(config),
            metrics=metrics,
            **compile_options,
        )
    return model


def create_fine_tune_optimizer(config):
    """Create a fresh optimizer for backbone fine-tuning."""
    optimizer_config = config.get("optimizer", {})
    fine_tune_config = config.get("training", {}).get("fine_tune_optimizer", {})
    optimizer_name = fine_tune_config.get("name", "adam").lower()
    kwargs = {"learning_rate": config.get("training", {}).get("fine_tune_learning_rate", 0.0001)}
    clipnorm = fine_tune_config.get("clipnorm", optimizer_config.get("clipnorm"))
    clipvalue = fine_tune_config.get("clipvalue", optimizer_config.get("clipvalue"))
    if clipnorm:
        kwargs["clipnorm"] = float(clipnorm)
    if clipvalue:
        kwargs["clipvalue"] = float(clipvalue)
    weight_decay = fine_tune_config.get("decay", optimizer_config.get("decay", 0.0))
    if weight_decay and weight_decay > 0:
        kwargs["weight_decay"] = weight_decay
    if optimizer_name == "adam":
        for key in ["beta_1", "beta_2", "epsilon"]:
            value = fine_tune_config.get(key, optimizer_config.get(key))
            if value is not None:
                kwargs[key] = value
        return keras.optimizers.Adam(**kwargs)
    if optimizer_name == "sgd":
        kwargs["momentum"] = fine_tune_config.get("momentum", 0.9)
        return keras.optimizers.SGD(**kwargs)
    if optimizer_name == "rmsprop":
        kwargs["momentum"] = fine_tune_config.get("momentum", 0.0)
        epsilon = fine_tune_config.get("epsilon", optimizer_config.get("epsilon"))
        if epsilon is not None:
            kwargs["epsilon"] = epsilon
        return keras.optimizers.RMSprop(**kwargs)
    raise ValueError(
        f"Unsupported fine-tuning optimizer '{optimizer_name}'. Supported: adam, sgd, rmsprop"
    )


class EpochThroughputLogger(keras.callbacks.Callback):
    """Record throughput and optionally emit a compact per-epoch summary."""

    def __init__(self, samples, log_mode="progress", progress_style="lines"):
        super().__init__()
        self.samples = max(0, int(samples))
        self.log_mode = log_mode
        # The inline bar already names the running epoch, so the extra
        # "Epoch n/N started" heading would be a duplicate line.
        self.progress_style = progress_style
        self.epoch_started = None
        self.train_elapsed = None
        self.epoch_learning_rate = None

    def on_epoch_begin(self, epoch, logs=None):
        del logs
        if self.log_mode == "summary" and self.progress_style != "inline":
            total_epochs = self.params.get("epochs", epoch + 1)
            print(f"Epoch {epoch + 1}/{total_epochs} started", flush=True)
        self.epoch_started = time.perf_counter()
        self.train_elapsed = None
        try:
            self.epoch_learning_rate = float(
                keras.backend.get_value(self.model.optimizer.learning_rate)
            )
        except (AttributeError, TypeError, ValueError):
            self.epoch_learning_rate = None

    def on_test_begin(self, logs=None):
        del logs
        if self.epoch_started is not None:
            self.train_elapsed = max(time.perf_counter() - self.epoch_started, 1e-9)

    def on_epoch_end(self, epoch, logs=None):
        elapsed = self.train_elapsed or max(
            time.perf_counter() - self.epoch_started, 1e-9
        )
        throughput = self.samples / elapsed
        if logs is not None:
            logs["examples_per_second"] = throughput
            if self.epoch_learning_rate is not None:
                logs["lr"] = self.epoch_learning_rate
        if self.log_mode == "summary":
            if self.progress_style == "inline":
                # The inline bar owns the single per-epoch line and already
                # reports lr and throughput; a second summary line here would
                # duplicate the epoch twice in the log.
                return
            metrics = logs or {}
            total_epochs = self.params.get("epochs", epoch + 1)
            fields = [f"Epoch {epoch + 1}/{total_epochs}"]
            for label, candidates in (
                ("loss", ("loss",)),
                (
                    "accuracy",
                    ("accuracy", "categorical_accuracy", "sparse_categorical_accuracy"),
                ),
                ("val_loss", ("val_loss",)),
                (
                    "val_accuracy",
                    (
                        "val_accuracy",
                        "val_categorical_accuracy",
                        "val_sparse_categorical_accuracy",
                    ),
                ),
            ):
                value = next(
                    (metrics[key] for key in candidates if key in metrics), None
                )
                if value is not None:
                    try:
                        fields.append(f"{label}: {float(value):.4f}")
                    except (TypeError, ValueError):
                        continue
            if self.epoch_learning_rate is not None:
                fields.append(f"lr: {self.epoch_learning_rate:.3e}")
            fields.append(f"examples_per_second: {throughput:.1f}")
            print(" - ".join(fields), flush=True)
        else:
            print(f" - examples_per_second: {throughput:.1f}", flush=True)


class BoundedProfilerCallback(keras.callbacks.Callback):
    """Capture a bounded number of training execution groups in TensorFlow Profiler."""

    def __init__(self, log_dir, start_epoch=2, execution_groups=1, logger=None):
        super().__init__()
        self.log_dir = os.path.abspath(log_dir)
        self.start_epoch = int(start_epoch)
        self.execution_groups = int(execution_groups)
        self.logger = logger
        self._remaining_groups = 0
        self._active = False

    def _stop(self):
        if not self._active:
            return
        try:
            tf.profiler.experimental.stop()
            if self.logger:
                self.logger.info("TensorFlow profiler trace saved to: %s", self.log_dir)
        except Exception as exc:  # Profiling failures must not interrupt training.
            if self.logger:
                self.logger.warning("Could not stop TensorFlow profiler: %s", exc)
        finally:
            self._active = False
            self._remaining_groups = 0

    def on_epoch_begin(self, epoch, logs=None):
        del logs
        if self._active or epoch + 1 != self.start_epoch:
            return
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            tf.profiler.experimental.start(self.log_dir)
            self._active = True
            self._remaining_groups = self.execution_groups
            if self.logger:
                self.logger.info(
                    "TensorFlow profiler started for epoch %d (%d execution group(s)); "
                    "steps_per_execution determines batches per group",
                    self.start_epoch,
                    self.execution_groups,
                )
        except Exception as exc:  # Profiling failures must not interrupt training.
            if self.logger:
                self.logger.warning("Could not start TensorFlow profiler: %s", exc)

    def on_train_batch_end(self, batch, logs=None):
        del batch, logs
        if not self._active:
            return
        self._remaining_groups -= 1
        if self._remaining_groups <= 0:
            self._stop()

    def on_epoch_end(self, epoch, logs=None):
        del epoch, logs
        self._stop()

    def on_train_end(self, logs=None):
        del logs
        self._stop()


def _make_profiler_callback(config, config_path, run_timestamp, logger):
    """Create an opt-in profiler for a cache-warmed epoch execution group."""
    profiler_config = config.get("performance", {}).get("profiler", {})
    if isinstance(profiler_config, bool):
        profiler_config = {"enabled": profiler_config}
    elif not isinstance(profiler_config, dict):
        raise ValueError("performance.profiler must be a boolean or a mapping")
    env_enabled = os.environ.get("LEAF_TRAIN_PROFILE", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if not profiler_config.get("enabled", False) and not env_enabled:
        return None

    start_epoch = int(profiler_config.get("start_epoch", 2))
    execution_groups = int(profiler_config.get("execution_groups", 1))
    if start_epoch < 1:
        raise ValueError("performance.profiler.start_epoch must be >= 1")
    if execution_groups < 1:
        raise ValueError("performance.profiler.execution_groups must be >= 1")

    configured_dir = profiler_config.get("log_dir")
    if configured_dir:
        profile_root = _resolve_path(configured_dir, os.path.dirname(config_path))
    else:
        profile_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(config_path))),
            "logs",
            "tensorflow_profiler",
        )
    profile_dir = os.path.join(profile_root, run_timestamp)
    return BoundedProfilerCallback(
        profile_dir,
        start_epoch=start_epoch,
        execution_groups=execution_groups,
        logger=logger,
    )


class DiagnosticBatchProgress(keras.callbacks.Callback):
    """Emit deterministic batch boundaries for short diagnostic runs."""

    def __init__(self, train_steps, validation_steps):
        super().__init__()
        self.train_steps = int(train_steps)
        self.validation_steps = int(validation_steps)

    def on_train_batch_begin(self, batch, logs=None):
        del logs
        print(f"Diagnostic train batch {batch + 1}/{self.train_steps} started", flush=True)

    def on_train_batch_end(self, batch, logs=None):
        del logs
        print(f"Diagnostic train batch {batch + 1}/{self.train_steps} completed", flush=True)

    def on_test_batch_begin(self, batch, logs=None):
        del logs
        print(
            f"Diagnostic validation batch {batch + 1}/{self.validation_steps} started",
            flush=True,
        )

    def on_test_batch_end(self, batch, logs=None):
        del logs
        print(
            f"Diagnostic validation batch {batch + 1}/{self.validation_steps} completed",
            flush=True,
        )


class BatchProgressLogger(keras.callbacks.Callback):
    """Print sparse, flushed progress at Keras execution-group boundaries."""

    def __init__(
        self,
        train_steps,
        validation_steps,
        interval=64,
        steps_per_execution=1,
    ):
        super().__init__()
        self.train_steps = int(train_steps)
        self.validation_steps = int(validation_steps)
        self.interval = int(interval)
        self.steps_per_execution = int(steps_per_execution)
        if self.interval < 1:
            raise ValueError("progress interval must be a positive integer")
        if self.steps_per_execution < 1:
            raise ValueError("steps_per_execution must be a positive integer")
        self._train_execution_reported = False
        self._validation_execution_reported = False

    @staticmethod
    def _completed_batches(batch, total_steps):
        # With steps_per_execution > 1, Keras reports the final step index of
        # each execution group from on_*_batch_end.
        return min(int(batch) + 1, int(total_steps))

    def on_epoch_begin(self, epoch, logs=None):
        del epoch, logs
        self._train_execution_reported = False
        self._validation_execution_reported = False

    def on_train_batch_begin(self, batch, logs=None):
        del logs
        if int(batch) == 0:
            print(f"Train batch 1/{self.train_steps} started", flush=True)

    def on_train_batch_end(self, batch, logs=None):
        del logs
        completed = self._completed_batches(batch, self.train_steps)
        first_execution = min(self.steps_per_execution, self.train_steps)
        if (
            not self._train_execution_reported
            or completed % self.interval == 0
            or completed >= self.train_steps
        ):
            print(
                f"Train progress: {completed}/{self.train_steps} batches completed",
                flush=True,
            )
        self._train_execution_reported = completed >= first_execution

    def on_test_batch_begin(self, batch, logs=None):
        del logs
        if int(batch) == 0:
            print(
                f"Validation batch 1/{self.validation_steps} started", flush=True
            )

    def on_test_batch_end(self, batch, logs=None):
        del logs
        completed = self._completed_batches(batch, self.validation_steps)
        first_execution = min(self.steps_per_execution, self.validation_steps)
        if (
            not self._validation_execution_reported
            or completed % self.interval == 0
            or completed >= self.validation_steps
        ):
            print(
                "Validation progress: "
                f"{completed}/{self.validation_steps} batches completed",
                flush=True,
            )
        self._validation_execution_reported = completed >= first_execution


def _format_duration(seconds):
    """Format a duration as ``H:MM:SS`` or ``M:SS``."""
    total = int(max(float(seconds), 0.0))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class InlineProgressBar(keras.callbacks.Callback):
    """Repaint one carriage-return status line instead of printing per batch.

    Keras' own ``Progbar`` only repaints in place when ``sys.stdout`` is a
    TTY, so a hosted notebook that streams training through a pipe gets one
    line per update (``Train progress: 64/700 batches completed`` and so on).
    Colab and Jupyter do render ``\\r`` as an in-place overwrite, so this
    callback keeps Keras' familiar layout while emitting a single updating
    line::

        Epoch 1/5 64/700 [==>.........] - ETA 21:32 - loss: 4.7198 - accuracy: 0.1074

    The line is finalized with a newline at the end of each epoch so the
    per-epoch summary and callback messages stay on their own lines.
    """

    def __init__(
        self,
        train_steps,
        validation_steps,
        steps_per_execution=1,
        min_interval=0.25,
        width=None,
        stream=None,
        samples=None,
    ):
        super().__init__()
        self.train_steps = max(1, int(train_steps))
        self.validation_steps = max(0, int(validation_steps))
        self.steps_per_execution = max(1, int(steps_per_execution))
        self.min_interval = max(0.0, float(min_interval))
        self.width = int(width) if width else self._detect_width()
        self.stream = stream
        # Finished-epoch throughput requires the epoch's sample count. When it
        # is unknown (tests, ad-hoc use) the final line omits lr/throughput.
        self.samples = max(0, int(samples)) if samples is not None else None
        self._epoch = 0
        self._total_epochs = 0
        self._phase = "train"
        self._phase_started = 0.0
        self._completed = 0
        self._last_render = 0.0
        self._last_length = 0
        self._at_line_start = True
        self._train_elapsed = None
        self._learning_rate = None

    @staticmethod
    def _detect_width():
        """Return a printable width, honouring an explicit COLUMNS override."""
        try:
            return max(40, int(os.environ.get("COLUMNS", "")))
        except (TypeError, ValueError):
            return 100

    @staticmethod
    def _bar(completed, total, bar_width=20):
        """Render a fixed-width progress bar."""
        if total <= 0:
            return "[" + "-" * bar_width + "]"
        fraction = min(max(float(completed) / float(total), 0.0), 1.0)
        filled = int(round(fraction * bar_width))
        if filled >= bar_width:
            return "[" + "=" * bar_width + "]"
        return "[" + "=" * filled + ">" + "." * (bar_width - filled - 1) + "]"

    def _get_stream(self):
        return self.stream if self.stream is not None else sys.stdout

    def _write(self, text):
        try:
            self._get_stream().write(text)
            self._get_stream().flush()
        except (AttributeError, ValueError, OSError):
            # A closed or exotic stdout must never abort training.
            pass

    # Keras reports custom metrics under their decorated names, and the same
    # metric can arrive more than once across replays. Fold them onto the short
    # labels a human reads, and keep the first occurrence of each.
    _METRIC_LABELS = (
        ("loss", ("loss",)),
        ("accuracy", ("accuracy", "categorical_accuracy", "sparse_categorical_accuracy")),
        ("top5_acc", ("top_k_categorical_accuracy", "top_k_categorical_accuracy_1")),
        ("precision", ("precision",)),
        ("recall", ("recall",)),
        ("auc", ("auc",)),
    )

    def _phase_fields(self, logs):
        """Return loss/metric pairs, prefixed with ``val_`` during validation."""
        logs = logs or {}
        prefix = "val_" if self._phase == "validation" else ""
        fields = []
        seen = set()
        for label, candidates in self._METRIC_LABELS:
            value = next((logs[key] for key in candidates if key in logs), None)
            if value is None:
                continue
            try:
                rendered = f"{prefix}{label}: {float(value):.4f}"
            except (TypeError, ValueError):
                continue
            if rendered in seen:
                continue
            seen.add(rendered)
            fields.append(rendered)
        return fields

    def status_text(self, logs=None):
        """Build the current status line body without carriage-return framing."""
        total = self.train_steps if self._phase == "train" else self.validation_steps
        completed = min(self._completed, total) if total else 0
        epoch_label = (
            f"Epoch {self._epoch}/{self._total_epochs}"
            if self._total_epochs
            else f"Epoch {self._epoch}"
        )
        parts = [f"{epoch_label} {completed}/{total} {self._bar(completed, total)}"]
        elapsed = time.perf_counter() - self._phase_started
        if completed > 0 and total > completed:
            eta = elapsed / completed * (total - completed)
            parts.append(f"ETA {_format_duration(eta)}")
        parts.extend(self._phase_fields(logs))
        return " - ".join(parts)

    def _render(self, logs=None, force=False):
        now = time.perf_counter()
        throttled = (
            not force
            and self.min_interval
            and (now - self._last_render) < self.min_interval
        )
        if throttled:
            return
        self._last_render = now
        text = self.status_text(logs)[: self.width]
        # Pad to the previous length so a shorter line cannot leave stale
        # characters from the previous repaint visible.
        tail = " " * max(0, self._last_length - len(text))
        self._last_length = len(text)
        # A fresh line needs no carriage return; the first repaint must return
        # to column zero before overwriting.
        prefix = "" if self._at_line_start else "\r"
        self._at_line_start = False
        self._write(prefix + text + tail)

    def _finish_line(self):
        """Terminate the current line, unless one was just terminated."""
        if self._at_line_start:
            return
        self._write("\n")
        self._last_length = 0
        self._at_line_start = True

    def on_epoch_begin(self, epoch, logs=None):
        del logs
        self._epoch = int(epoch) + 1
        try:
            self._total_epochs = int(self.params.get("epochs", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            self._total_epochs = 0
        self._phase = "train"
        self._phase_started = time.perf_counter()
        self._completed = 0
        self._last_length = 0
        self._train_elapsed = None
        self._learning_rate = None
        try:
            self._learning_rate = float(
                keras.backend.get_value(self.model.optimizer.learning_rate)
            )
        except (AttributeError, TypeError, ValueError):
            self._learning_rate = None

    def on_train_batch_end(self, batch, logs=None):
        self._phase = "train"
        self._completed = min(int(batch) + 1, self.train_steps)
        self._render(logs, force=self._completed >= self.train_steps)

    def on_test_begin(self, logs=None):
        del logs
        # Freeze the training-only duration before the validation timer resets,
        # so throughput excludes validation exactly as the CSV column does.
        self._train_elapsed = max(time.perf_counter() - self._phase_started, 1e-9)
        self._phase = "validation"
        self._phase_started = time.perf_counter()
        self._completed = 0
        # Keep _last_length: the validation line overwrites the still-unfinished
        # train line, so the tail padding must still cover the longer text.

    def on_test_batch_end(self, batch, logs=None):
        self._completed = min(int(batch) + 1, self.validation_steps)
        self._render(
            logs,
            force=bool(
                self.validation_steps
                and self._completed >= self.validation_steps
            ),
        )

    def _final_text(self, logs):
        """Build the completed-epoch line: full bar, train metrics then val metrics."""
        logs = logs or {}
        total = self.train_steps
        epoch_label = (
            f"Epoch {self._epoch}/{self._total_epochs}"
            if self._total_epochs
            else f"Epoch {self._epoch}"
        )
        parts = [f"{epoch_label} {self._bar(total, total)}"]
        seen = set()
        for prefix in ("", "val_"):
            for label, candidates in self._METRIC_LABELS:
                value = next(
                    (
                        logs[prefix + key]
                        for key in candidates
                        if prefix + key in logs
                    ),
                    None,
                )
                if value is None:
                    continue
                try:
                    rendered = f"{prefix}{label}: {float(value):.4f}"
                except (TypeError, ValueError):
                    continue
                if rendered in seen:
                    continue
                seen.add(rendered)
                parts.append(rendered)
        if self.samples is not None:
            # One line per epoch: fold the learning rate and the steady-state
            # throughput into the bar's final line instead of printing a second
            # summary line from EpochThroughputLogger.
            if self._learning_rate is not None:
                parts.append(f"lr: {self._learning_rate:.3e}")
            elapsed = self._train_elapsed
            if elapsed:
                parts.append(f"examples_per_second: {self.samples / elapsed:.1f}")
        return " - ".join(parts)

    def on_epoch_end(self, epoch, logs=None):
        del epoch
        # Without a validation pass on_test_begin never fired; measure the
        # training-only duration here instead.
        if self._train_elapsed is None and self._phase == "train":
            self._train_elapsed = max(time.perf_counter() - self._phase_started, 1e-9)
        # Replace the in-flight validation line with the completed-epoch line
        # (Keras' familiar "Epoch N/M [====] - loss: ..." shape), then end it so
        # the checkpoint and throughput summaries land on their own lines. The
        # line is finalized with "\n", so unlike a repaint it is not clipped to
        # the terminal width; a wrapped completed line stays readable.
        text = self._final_text(logs)
        tail = " " * max(0, self._last_length - len(text))
        prefix = "" if self._at_line_start else "\r"
        self._write(prefix + text + tail + "\n")
        self._last_length = 0
        self._at_line_start = True

    def on_train_end(self, logs=None):
        del logs
        self._finish_line()


def _add_throughput_callback(
    callbacks, sample_count, log_mode="progress", progress_style="lines"
):
    """Place throughput before CSVLogger so the rate is persisted in history."""
    throughput_callback = EpochThroughputLogger(
        sample_count, log_mode=log_mode, progress_style=progress_style
    )
    csv_logger_index = next(
        (
            index
            for index, callback in enumerate(callbacks)
            if isinstance(callback, keras.callbacks.CSVLogger)
        ),
        len(callbacks),
    )
    callbacks.insert(csv_logger_index, throughput_callback)
    return callbacks


def _finite_dataset_steps(dataset, split_name):
    """Return a positive finite batch count required for explicit fit steps."""
    try:
        steps = int(tf.data.experimental.cardinality(dataset).numpy())
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not determine {split_name} dataset cardinality") from exc
    if steps <= 0:
        raise ValueError(
            f"{split_name} dataset must have a known positive finite cardinality; "
            f"got {steps}"
        )
    return steps


def _log_run_header(
    logger,
    config,
    model,
    run_dir,
    runtime_info,
    train_steps,
    validation_steps,
    sample_count,
    initial_epochs,
    total_epochs,
    class_weights=None,
    class_names=None,
):
    """Print the compact, auditable summary of what this run will actually do."""
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    performance = config.get("performance", {})
    optimizer_config = config.get("optimizer", {})
    loss_config = config.get("loss", {})
    dataset_config = config.get("dataset", {})

    trainable = sum(np.prod(weight.shape) for weight in model.trainable_weights)
    total = model.count_params()
    input_shape = tuple(model_config.get("input_shape", ()))

    logger.info("-" * 50)
    logger.info(
        "Run: %s @ %s | dir: %s",
        model_config.get("architecture", "unknown"),
        os.path.basename(run_dir),
        run_dir,
    )
    logger.info(
        "Model: %s params (%s trainable), input %s, dropout %s, head %s",
        f"{int(total):,}",
        f"{int(trainable):,}",
        "x".join(str(int(value)) for value in input_shape) if input_shape else "?",
        model_config.get("dropout_rate"),
        model_config.get("head_units", "-"),
    )
    logger.info(
        "Runtime: policy=%s, replicas=%d, GPUs=%d, batch=%s, "
        "steps_per_execution=%s, jit_compile=%s",
        keras.mixed_precision.global_policy().name,
        runtime_info.get("num_replicas", 1),
        len(tf.config.list_physical_devices("GPU")),
        dataset_config.get("batch_size"),
        performance.get("steps_per_execution"),
        performance.get("jit_compile"),
    )
    fine_tune_lr = training_config.get("fine_tune_learning_rate")
    logger.info(
        "Schedule: %d frozen + %d fine-tune epochs, lr %s -> %s, loss %s, metrics %s",
        initial_epochs,
        max(int(total_epochs) - int(initial_epochs), 0),
        optimizer_config.get("learning_rate"),
        fine_tune_lr if fine_tune_lr is not None else optimizer_config.get("learning_rate"),
        loss_config.get("name", "categorical_crossentropy"),
        config.get("metrics", ["accuracy"]),
    )
    logger.info(
        "Steps: train=%d, val=%d (%s samples/epoch)",
        train_steps,
        validation_steps,
        f"{int(sample_count):,}" if sample_count else "?",
    )
    if class_weights:
        weights = list(class_weights.values())
        logger.info(
            "Class weights: n=%d, applied range [%.4f, %.4f]",
            len(weights),
            min(weights),
            max(weights),
        )
    elif class_names is not None:
        logger.info("Class weights: none applied to this fit")
    logger.info("-" * 50)


def _log_run_summary(logger, history, selected_phase, selected_loss, elapsed_seconds, artifacts):
    """Print the end-of-run block: winner weights, timing, throughput, artifacts."""
    history = history or {}
    epochs = len(history.get("loss", []))

    def _best(key):
        values = history.get(key) or []
        pairs = [(float(value), index + 1) for index, value in enumerate(values)]
        pairs = [pair for pair in pairs if np.isfinite(pair[0])]
        if not pairs:
            return None, None
        value, epoch = min(pairs) if "loss" in key else max(pairs)
        return value, epoch

    best_val_loss, best_epoch = _best("val_loss")
    best_val_accuracy, _ = _best("val_accuracy")
    throughputs = [
        float(value)
        for value in (history.get("examples_per_second") or [])
        if np.isfinite(float(value))
    ]
    # Epoch 1 includes dataset cache fill and graph tracing, so it is excluded
    # from the steady-state figure and reported separately.
    steady = throughputs[1:] if len(throughputs) > 1 else throughputs

    logger.info("-" * 50)
    details = [f"weights={selected_phase}"]
    if best_val_loss is not None:
        details.append(f"best_epoch={best_epoch}")
        details.append(f"val_loss={best_val_loss:.4f}")
    if best_val_accuracy is not None:
        details.append(f"val_accuracy={best_val_accuracy:.4f}")
    details.append(f"epochs={epochs}")
    details.append(f"duration={_format_duration(elapsed_seconds)}")
    if steady:
        details.append(f"steady_throughput={float(np.median(steady)):.1f} img/s")
    if throughputs:
        details.append(f"first_epoch={throughputs[0]:.1f} img/s (includes cache warm-up)")
    logger.info("Best: %s", ", ".join(details))
    if selected_loss is not None and np.isfinite(float(selected_loss)):
        logger.info("Selected val_loss: %.6f", float(selected_loss))
    for name, path in (artifacts or {}).items():
        logger.info("Artifact %s: %s", name, path)
    logger.info("-" * 50)


def _summarize_split_counts(split_counts, estimated=False):
    """Render split totals as one readable line instead of a raw dict dump."""
    parts = []
    for split_name in ("train", "val", "test"):
        counts = split_counts.get(split_name) or {}
        total = counts.get("total")
        if total is None:
            parts.append(f"{split_name}=?")
        else:
            parts.append(f"{split_name}={int(total)}")
    class_totals = []
    for split_name in ("train", "val", "test"):
        classes = (split_counts.get(split_name) or {}).get("classes") or {}
        values = [value for value in classes.values() if value is not None]
        if values:
            class_totals.append((min(values), max(values)))
    summary = "Split totals: " + ", ".join(parts)
    if class_totals:
        summary += (
            f" (per-class {min(low for low, _ in class_totals)}-"
            f"{max(high for _, high in class_totals)}; details in run_metadata.json)"
        )
    if estimated:
        summary += " [estimated: validation/test per-class counts are unknown]"
    return summary


def _estimated_split_counts(train_steps, val_steps, batch_size, class_names=None):
    """Return lightweight sample totals without filesystem scans.

    Per-class counts are intentionally marked unknown when the split was not
    scanned.  The default shape remains compatible with callers that only need
    totals.
    """
    def estimated_total(steps):
        return int(steps) * int(batch_size)

    counts = {
        "train": {"total": estimated_total(train_steps), "classes": {}},
        "val": {"total": estimated_total(val_steps), "classes": {}},
        "test": {"total": 0, "classes": {}},
    }
    if class_names is not None:
        unknown_classes = {name: None for name in class_names}
        for split in ("val", "test"):
            counts[split]["classes"] = unknown_classes.copy()
            counts[split]["estimated"] = True
        counts["test"]["total"] = None
    return counts


def _setup_phase2_callbacks(config):
    """Build fresh fine-tuning callbacks and append to the Phase 1 CSV log.

    Phase 2 must not inherit Phase 1's decayed schedule: the frozen-backbone
    phase deliberately anneals its LR toward zero, so reusing that curve would
    start fine-tuning at a much smaller LR than ``fine_tune_learning_rate`` and
    leave the backbone effectively frozen.
    """
    training_cfg = config.get("training", {})
    fine_tune_lr = float(
        training_cfg.get("fine_tune_learning_rate")
        or config.get("optimizer", {}).get("learning_rate", 1e-4)
    )
    total_epochs = int(training_cfg.get("epochs", 0) or 0)
    unfreeze_epoch = int(training_cfg.get("unfreeze_epoch", 0) or 0)
    phase2_epochs = max(total_epochs - unfreeze_epoch, 1)

    # Restart the same schedule *shape* from the fine-tuning LR and horizon.
    # epoch_offset realigns the curve: Keras calls the scheduler with global
    # epoch indices that continue from Phase 1 (initial_epoch), so without the
    # offset the fine-tuning phase would start already 2/3 decayed and land on
    # the alpha floor almost immediately.
    lr_override = {
        "initial_learning_rate": fine_tune_lr,
        "decay_epochs": float(phase2_epochs),
        "epoch_offset": float(unfreeze_epoch),
    }
    callback_kwargs = {
        "csv_append": True,
        "checkpoint_name": "best_phase2_model",
        "include_throughput": False,
    }
    if training_cfg or config.get("optimizer"):
        callback_kwargs["lr_schedule_override"] = lr_override
    return setup_callbacks(config, **callback_kwargs)


def _configure_phase2_early_stopping(callbacks, config, log_mode):
    """Apply the optional phase-two early-stopping settings."""
    if not config:
        return callbacks

    callbacks = [
        callback
        for callback in callbacks
        if not isinstance(callback, keras.callbacks.EarlyStopping)
    ]
    if config.get("enabled", True):
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=config.get("monitor", "val_loss"),
                patience=config.get("patience", 6),
                restore_best_weights=config.get("restore_best_weights", True),
                mode=config.get("mode", "min"),
                verbose=_fit_verbosity(log_mode),
            )
        )
    return callbacks


def configure_runtime(config, logger):
    """Enable optional mixed precision and mirrored multi-GPU training."""
    runtime_config = dict(config.get("performance", {}))
    runtime_config.update(config.get("runtime", {}))
    physical_gpus = tf.config.list_physical_devices("GPU")
    use_mixed_precision = bool(
        runtime_config.get("mixed_precision", False) and physical_gpus
    )
    if runtime_config.get("mixed_precision", False) and not physical_gpus:
        logger.info("Mixed precision requested but no GPU is available; using float32")
    policy_name = "mixed_float16" if use_mixed_precision else "float32"
    keras.mixed_precision.set_global_policy(policy_name)

    distribution = str(runtime_config.get("distribution", "auto")).lower()
    multi_gpu = runtime_config.get(
        "multi_gpu", distribution in {"auto", "mirrored", "multi_gpu"}
    )
    if multi_gpu and len(physical_gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()
    logger.info(
        "Runtime: policy=%s, replicas=%d, GPUs=%d",
        policy_name,
        strategy.num_replicas_in_sync,
        len(physical_gpus),
    )
    if len(physical_gpus) > 1 and strategy.num_replicas_in_sync < len(physical_gpus):
        logger.warning(
            "Multiple GPUs are visible (%d), but the active strategy has only %d "
            "replica(s); this run will not use all visible GPUs. Check the Kaggle "
            "runtime profile (performance.distribution and performance.multi_gpu).",
            len(physical_gpus),
            strategy.num_replicas_in_sync,
        )
    return strategy


def _maybe_loss_scale_optimizer(optimizer):
    """Use dynamic loss scaling for optional mixed_float16 training."""
    policy = keras.mixed_precision.global_policy().name
    if policy == "mixed_float16":
        loss_scale_optimizer = keras.mixed_precision.LossScaleOptimizer
        if not isinstance(optimizer, loss_scale_optimizer):
            optimizer = loss_scale_optimizer(optimizer)
    return optimizer


def _prefetch_to_device_for_single_replica(train_ds, val_ds, strategy, logger):
    """Use experimental GPU prefetch only outside mirrored, GPU-backed training."""
    replica_count = int(strategy.num_replicas_in_sync)
    if replica_count > 1:
        logger.info(
            "Skipping GPU prefetch for %d-replica strategy; "
            "using regular dataset prefetch",
            replica_count,
        )
        return train_ds, val_ds

    # prefetch_to_device("/gpu:0") succeeds on a CPU-only host but pins buffered
    # batches into host memory, so only apply it when a GPU actually exists.
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logger.info("No GPU detected; keeping host prefetch for the dataset pipeline")
        return train_ds, val_ds

    try:
        train_ds = train_ds.apply(tf.data.experimental.prefetch_to_device("/gpu:0"))
        val_ds = val_ds.apply(tf.data.experimental.prefetch_to_device("/gpu:0"))
        logger.info("Enabled GPU prefetch for dataset pipeline")
    except (RuntimeError, ValueError) as exc:
        logger.warning("GPU prefetch unavailable: %s", exc)
    return train_ds, val_ds


def train_model(config_path="config.yaml"):
    """Main training function"""

    # Load configuration
    config = load_config(config_path)
    config = resolve_config_paths(config, config_path)
    config.setdefault("training", {}).setdefault("log_mode", "summary")
    log_mode = get_epoch_log_mode(config)
    if config.get("runtime", {}).get("diagnostic", False):
        progress_setting = "diagnostic-every-batch"
    else:
        progress_setting = config.get("training", {}).get(
            "progress_log_interval", "summary-only"
        )

    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting training...")
    arch = config.get("model", {}).get("architecture", "unknown")
    logger.info(f"Configuration: {arch}")
    logger.info(
        "Training throughput settings: cache_mode=%s, cache_dtype=%s, "
        "steps_per_execution=%s, progress_log_interval=%s",
        config.get("dataset", {}).get("cache_mode", "none"),
        config.get("dataset", {}).get("cache_dtype", "float32"),
        config.get("performance", {}).get("steps_per_execution", 1),
        progress_setting,
    )

    runtime_strategy = configure_runtime(config, logger)
    runtime_info = {
        "num_replicas": runtime_strategy.num_replicas_in_sync,
        "has_gpu": bool(tf.config.list_physical_devices("GPU")),
    }
    choose_batch_size(config, runtime_info, logger)
    freeze_base = config.get("training", {}).get("freeze_base", True)
    fine_tuning_enabled = config.get("training", {}).get("fine_tuning", {}).get(
        "enabled", True
    )
    total_epochs = int(config.get("training", {}).get("epochs", 50))
    unfreeze_epoch = int(config.get("training", {}).get("unfreeze_epoch", 10))
    if total_epochs < 1:
        raise ValueError("training.epochs must be >= 1")

    if freeze_base and fine_tuning_enabled:
        initial_epochs = min(unfreeze_epoch, total_epochs)
        if unfreeze_epoch > total_epochs:
            logger.warning(
                "unfreeze_epoch (%s) is greater than epochs (%s); fine-tuning phase will be skipped.",
                unfreeze_epoch,
                total_epochs,
            )
    else:
        initial_epochs = total_epochs

    # Set random seeds for reproducibility
    seed = config.get("seed", 42)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Count the training split once up front. This supports class weighting and
    # class-aware augmentation without enabling expensive cross-split hashing.
    training_class_counts = get_training_class_counts(config["dataset"]["train_dir"])
    expected_class_names = list(training_class_counts)
    class_aware_config = config.get("augmentation", {}).get("class_aware", {})
    minority_indices = get_minority_class_indices(
        training_class_counts,
        class_aware_config.get("max_train_samples", 500),
        logger=logger,
    ) if class_aware_config.get("enabled", False) else []
    minority_class_names = [expected_class_names[index] for index in minority_indices]
    logger.info(
        "Class-aware augmentation: enabled=%s, cutoff=%s, selected=%d/%d, "
        "minority_classes=%s",
        bool(class_aware_config.get("enabled", False)),
        class_aware_config.get("max_train_samples"),
        len(minority_indices),
        len(expected_class_names),
        minority_class_names,
    )

    # Load datasets INSIDE strategy scope for proper GPU distribution
    logger.info("Loading datasets...")
    preprocess_fn = get_preprocess_fn(config["model"]["architecture"])
    augmentation, minority_augmentation = create_class_aware_augmentation_layers(config)

    with runtime_strategy.scope():
        train_ds, val_ds, class_names = load_dataset(
            config["dataset"]["train_dir"],
            config["dataset"]["val_dir"],
            batch_size=config["dataset"]["batch_size"],
            image_size=tuple(config["model"]["input_shape"][:2]),
            preprocess_fn=preprocess_fn,
            augmentation=augmentation,
            minority_augmentation=minority_augmentation,
            minority_class_indices=minority_indices,
            shuffle_buffer=config.get("dataset", {}).get("shuffle_buffer", 1000),
            cache_mode=config.get("dataset", {}).get("cache_mode", "none"),
            cache_memory_limit_gb=config.get("dataset", {}).get("cache_memory_limit_gb", 8.0),
            cache_dtype=config.get("dataset", {}).get("cache_dtype", "float32"),
            deterministic=config.get("performance", {}).get("deterministic", False),
            class_names=expected_class_names,
        )

    train_steps = _finite_dataset_steps(train_ds, "training")
    val_steps = _finite_dataset_steps(val_ds, "validation")
    fit_train_steps, fit_val_steps = _resolve_fit_step_counts(
        train_steps,
        val_steps,
        config.get("runtime", {}),
    )
    diagnostic_run = bool(config.get("runtime", {}).get("diagnostic", False))
    full_train_count = sum(training_class_counts.values())
    fit_sample_count = _fit_sample_count(
        full_train_count,
        fit_train_steps,
        config["dataset"]["batch_size"],
        diagnostic=diagnostic_run,
    )
    if config.get("performance", {}).get("finite_batch_check", False):
        validate_dataset_batch(train_ds, logger)
    # repeat() is applied to the finite dataset (steps were measured above), so
    # prefetch here survives the epoch boundary instead of being torn down and
    # rebuilt on every epoch boundary.
    train_ds = train_ds.repeat().prefetch(tf.data.AUTOTUNE)

    # Device prefetch is not compatible with the serialized dataset graph used
    # by MirroredStrategy. load_dataset already adds ordinary tf.data prefetch.
    train_ds, val_ds = _prefetch_to_device_for_single_replica(
        train_ds, val_ds, runtime_strategy, logger
    )

    logger.info(
        "Using repeated training dataset with explicit steps_per_epoch=%d and "
        "validation_steps=%d",
        fit_train_steps,
        fit_val_steps,
    )

    logger.info(f"Found {len(class_names)} classes")

    if len(class_names) != config["model"]["num_classes"]:
        raise ValueError(
            f"Class count mismatch: dataset has {len(class_names)} classes, "
            f"but config model.num_classes={config['model']['num_classes']}"
        )
    if class_names != expected_class_names:
        raise ValueError("Training-count label order does not match dataset label order")

    weight_config = config.get("class_weights", {})
    deep_validation = bool(
        config.get("performance", {}).get("validate_dataset_splits", True)
    )
    if deep_validation:
        # Validate that each split has the same classes and that every class has
        # examples before constructing a model or starting a potentially long run.
        logger.info("Counting and validating dataset splits...")
        split_counts = validate_dataset_splits(
            config["dataset"]["train_dir"],
            config["dataset"]["val_dir"],
            config["dataset"].get("test_dir"),
            class_names=class_names,
            allow_cross_split_duplicates=config.get("performance", {}).get(
                "allow_cross_split_duplicates", False
            ),
        )
        logger.info("%s", _summarize_split_counts(split_counts))
        for split_name in ("val", "test"):
            unsupported = sorted(
                class_name
                for class_name, count in split_counts[split_name]["classes"].items()
                if count == 0
            )
            if unsupported:
                logger.warning(
                    "%s has no independent exact-deduplicated examples for classes: %s",
                    split_name,
                    unsupported,
                )
    else:
        logger.warning(
            "Deep split validation is disabled; skipping cross-split scans and duplicate hashing."
        )
        split_counts = _estimated_split_counts(
            train_steps,
            val_steps,
            config["dataset"]["batch_size"],
            class_names=class_names,
        )
        split_counts["train"] = {
            "total": sum(training_class_counts.values()),
            "classes": training_class_counts,
        }
        logger.info("%s", _summarize_split_counts(split_counts, estimated=True))

    # Compute weights only for explicitly enabled imbalance-focused runs.
    # Fast baseline runs do not construct class-count lists or weight metadata.
    class_weights = None
    raw_class_weights = None
    if weight_config.get("enabled", False):
        train_class_counts = [training_class_counts[class_name] for class_name in class_names]
        raw_class_weights = compute_class_weights_from_counts(train_class_counts)
        class_weights = compute_class_weights_from_counts(
            train_class_counts,
            max_weight=weight_config.get("max_weight"),
            normalize=weight_config.get("normalize", True),
        )
        raw_values = list(raw_class_weights.values()) if raw_class_weights else []
        if class_weights:
            applied_values = list(class_weights.values())
            imbalance_ratio = (
                float(max(train_class_counts)) / float(min(train_class_counts))
                if train_class_counts and min(train_class_counts) > 0
                else float("inf")
            )
            min_ratio = float(weight_config.get("min_imbalance_ratio", 1.25) or 0.0)
            if imbalance_ratio < min_ratio:
                # Near-uniform class counts make inverse-frequency weights pure
                # noise (e.g. 0.9988..1.0924) that only perturbs the gradient.
                logger.warning(
                    "Class weights skipped: training counts are effectively balanced "
                    "(max/min ratio=%.3f < min_imbalance_ratio=%.2f). "
                    "Set class_weights.min_imbalance_ratio=0 to force weighting.",
                    imbalance_ratio,
                    min_ratio,
                )
                class_weights = None
            if class_weights:
                logger.info(
                    "Class weights: n=%d, raw=[%.4f, %.4f], applied=[%.4f, %.4f] "
                    "(max_weight=%s, normalize=%s)",
                    len(applied_values),
                    min(raw_values) if raw_values else float("nan"),
                    max(raw_values) if raw_values else float("nan"),
                    min(applied_values),
                    max(applied_values),
                    weight_config.get("max_weight"),
                    weight_config.get("normalize", True),
                )
    else:
        logger.info("Class weighting is disabled; skipping class-weight computation.")

    # Create model inside the distribution scope so variables are mirrored.
    logger.info(f"Creating {config['model']['architecture']} model...")
    with runtime_strategy.scope():
        model_kwargs = {
            "architecture": config["model"]["architecture"],
            "input_shape": tuple(config["model"]["input_shape"]),
            "num_classes": config["model"]["num_classes"],
            "dropout_rate": config["model"]["dropout_rate"],
            "weights": config["model"]["weights"],
            "hub_url": config["model"].get("hub_url"),
            "hub_cache_dir": config["model"].get("hub_cache_dir"),
            "hub_download_retries": config["model"].get("hub_download_retries", 1),
            "hub_download_delay_sec": config["model"].get("hub_download_delay_sec", 5),
        }
        architecture = config["model"]["architecture"].lower()
        if architecture in {"mobilenetv2", "efficientnet_b0"}:
            model_kwargs["head_units"] = tuple(
                config["model"].get("head_units", [256, 128])
            )
            model_kwargs["l2_regularization"] = config["model"].get(
                "l2_regularization", 0.0
            )
        if "classifier_units" in config["model"]:
            model_kwargs["classifier_units"] = config["model"]["classifier_units"]
        if "include_preprocessing" in config["model"]:
            model_kwargs["include_preprocessing"] = config["model"][
                "include_preprocessing"
            ]
        model, base_model = get_model(**model_kwargs)

    if not freeze_base:
        logger.info("freeze_base is False: training backbone from the first epoch")
        # A randomly initialized backbone has BatchNorm statistics at their
        # initialization (mean 0 / variance 1), so freezing those layers leaves
        # the network effectively unnormalized and it cannot converge. Keep the
        # frozen-BN policy only for backbones that start from pretrained weights.
        if config["model"].get("weights") is None and hasattr(base_model, "layers"):
            logger.info(
                "Randomly initialized backbone: unfreezing BatchNormalization "
                "so moving statistics are learned during training"
            )
            unfreeze_backbone_for_scratch_training(base_model)
        else:
            unfreeze_base_model(base_model, unfreeze_from_layer=0)

    print_model_summary(model)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_config = config.setdefault("callbacks", {}).setdefault(
        "checkpoint", {}
    )
    if checkpoint_config.get("run_scoped", True) and not checkpoint_config.get(
        "directory"
    ):
        checkpoint_config["directory"] = os.path.join(
            config["export"]["save_dir"],
            config["model"]["architecture"],
            "runs",
            run_timestamp,
            "checkpoints",
        )
    run_dir = save_run_metadata(
        config=config,
        config_path=config_path,
        model=model,
        raw_class_weights=raw_class_weights,
        applied_class_weights=class_weights,
        split_counts=split_counts,
        run_timestamp=run_timestamp,
        minority_class_names=minority_class_names,
    )
    logger.info("Saved reproducibility metadata to: %s", run_dir)

    # Compile model
    logger.info("Compiling model...")
    with runtime_strategy.scope():
        model = compile_model(model, config)

    # Setup callbacks
    logger.info("Setting up callbacks...")
    callbacks = setup_callbacks(config, include_throughput=False)
    # Piped stdout cannot render Keras' carriage-return bar; fall back to the
    # compact per-epoch summary line while keeping sparse batch progress.
    effective_log_mode = log_mode if _stdout_is_interactive() else "summary"
    _add_throughput_callback(
        callbacks,
        fit_sample_count,
        log_mode=effective_log_mode,
        progress_style=_progress_style(config),
    )
    profiler_callback = _make_profiler_callback(
        config, config_path, run_timestamp, logger
    )
    if profiler_callback is not None:
        callbacks.append(profiler_callback)
    phase1_best = BestValidationWeights()
    callbacks.append(phase1_best)
    if diagnostic_run:
        callbacks.insert(
            0,
            DiagnosticBatchProgress(
                train_steps=fit_train_steps,
                validation_steps=fit_val_steps,
            ),
        )
    callbacks, phase1_verbose = _configure_progress_logging(
        callbacks,
        config,
        log_mode,
        fit_train_steps,
        fit_val_steps,
        diagnostic_run=diagnostic_run,
        logger=logger,
        sample_count=fit_sample_count,
    )

    _log_run_header(
        logger,
        config,
        model,
        run_dir,
        runtime_info,
        fit_train_steps,
        fit_val_steps,
        fit_sample_count,
        initial_epochs,
        total_epochs,
        class_weights=class_weights,
        class_names=class_names,
    )
    training_started_at = time.perf_counter()

    # Phase 1 either warms up the classifier head or trains a randomly
    # initialized backbone end-to-end, depending on freeze_base.
    logger.info("\n" + "=" * 50)
    logger.info(phase_one_heading(freeze_base))
    logger.info("=" * 50 + "\n")

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        steps_per_epoch=fit_train_steps,
        validation_steps=fit_val_steps,
        verbose=phase1_verbose,
    )

    # The last Phase 1 epoch is not necessarily the best validation point.
    # Keep its best restored weights as the stable starting point for Phase 2.
    phase1_best_weights = phase1_best.weights or model.get_weights()

    # Phase 2: Fine-tune with unfrozen layers
    history2 = None
    # Defaults for the single-phase path so the end-of-run summary below always
    # has a defined selection to report.
    selected_phase = "phase_1"
    selected_loss = getattr(phase1_best, "best", None)
    selected_checkpoint = None
    if freeze_base and fine_tuning_enabled and initial_epochs < total_epochs:
        logger.info("\n" + "=" * 50)
        logger.info("Phase 2: Fine-tuning (unfreezing base layers)")
        logger.info("=" * 50 + "\n")

        # Unfreeze base model
        # Support both old config key "freeze_until_layer" and new "unfreeze_from_layer" for backwards compatibility
        unfreeze_from = config.get("training", {}).get("unfreeze_from_layer")
        if unfreeze_from is None:
            unfreeze_from = config.get("training", {}).get("freeze_until_layer", 100)
        unfreeze_base_model(base_model, unfreeze_from)
        model.set_weights(phase1_best_weights)

        # Recompile with lower learning rate
        compile_fine_tune_model(model, config, runtime_strategy)

        # Callbacks retain monitor/patience state. A fresh set isolates the
        # second optimizer phase from Phase 1's plateau state.
        phase2_callbacks = _setup_phase2_callbacks(config)
        callbacks_config = config.get("callbacks", {})
        phase2_early_stopping = callbacks_config.get("fine_tuning_early_stopping")
        phase2_callbacks = _configure_phase2_early_stopping(
            phase2_callbacks, phase2_early_stopping, log_mode
        )
        _add_throughput_callback(
            phase2_callbacks,
            fit_sample_count,
            log_mode=effective_log_mode,
            progress_style=_progress_style(config),
        )
        phase2_best = BestValidationWeights()
        phase2_callbacks.append(phase2_best)
        phase2_callbacks, phase2_verbose = _configure_progress_logging(
            phase2_callbacks,
            config,
            log_mode,
            fit_train_steps,
            fit_val_steps,
            diagnostic_run=diagnostic_run,
            logger=logger,
            sample_count=fit_sample_count,
        )

        # Continue training
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=initial_epochs,
            epochs=total_epochs,
            callbacks=phase2_callbacks,
            class_weight=class_weights,
            steps_per_epoch=fit_train_steps,
            validation_steps=fit_val_steps,
            verbose=phase2_verbose,
        )
        final_weights, selected_phase, selected_loss = select_final_weights(
            phase1_best, phase1_best_weights, phase2_best
        )
        model.set_weights(final_weights)
        logger.info(
            "Selected %s weights for export (best val_loss: %.6f)",
            selected_phase,
            selected_loss,
        )
        selected_checkpoint = save_model_checkpoint(model, config)
        if selected_checkpoint:
            logger.info("Saved cross-phase winner checkpoint to: %s", selected_checkpoint)

    # Merge phase histories so downstream consumers get one continuous record
    merged_history = {k: list(v) for k, v in history1.history.items()}
    if history2 is not None:
        for key, values in history2.history.items():
            merged_history.setdefault(key, []).extend(values)
    history1.history = merged_history

    if _should_skip_final_export(config):
        _log_run_summary(
            logger,
            merged_history,
            selected_phase,
            selected_loss,
            time.perf_counter() - training_started_at,
            artifacts={},
        )
        logger.info("Diagnostic run complete; skipping final model exports.")
        return model, history1

    # Save final model
    logger.info("Saving final model...")
    save_dir = config.get("export", {}).get("save_dir", "../models")
    model_name = config.get("model", {}).get("architecture", "model")
    timestamp = run_timestamp
    artifacts = {}

    # Save in multiple formats
    is_lite0_arch = model_name.lower() in ["efficientnet", "efficientnet_lite0"]

    if "keras" in config["export"]["formats"]:
        # Native Keras format: no HDF5 legacy warning and no config loss.
        keras_path = _save_native_keras_model(
            model, save_dir, model_name, timestamp, logger
        )
        if keras_path:
            artifacts["keras"] = keras_path

    if "h5" in config["export"]["formats"]:
        if is_lite0_arch:
            logger.warning(
                "Skipping H5 export for %s because TF Hub-backed models are best saved as SavedModel/TFLite.",
                model_name,
            )
        else:
            h5_path = os.path.join(save_dir, model_name, f"{model_name}_{timestamp}.h5")
            os.makedirs(os.path.dirname(h5_path), exist_ok=True)
            model.save(h5_path)
            logger.info(f"Saved H5 model: {h5_path}")
            artifacts["h5"] = h5_path

    if "saved_model" in config["export"]["formats"]:
        sm_path = os.path.join(save_dir, model_name, f"saved_model_{timestamp}")
        os.makedirs(os.path.dirname(sm_path), exist_ok=True)
        if hasattr(model, "export"):
            # Keras 3 requires export() for SavedModel directories.
            model.export(sm_path)
        else:
            # Legacy Keras 2 behavior.
            model.save(sm_path)
        logger.info(f"Saved SavedModel: {sm_path}")
        artifacts["saved_model"] = sm_path

    if "tflite" in config["export"]["formats"]:
        tflite_path = os.path.join(
            save_dir, model_name, f"{model_name}_{timestamp}.tflite"
        )
        os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
        tflite_export_model = make_float32_inference_model(model)
        converter = tf.lite.TFLiteConverter.from_keras_model(tflite_export_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.allow_custom_ops = False
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        logger.info(f"Saved TFLite model: {tflite_path}")
        artifacts["tflite"] = tflite_path

    class_names_path = os.path.join(save_dir, model_name, "class_names.txt")
    save_class_names(class_names, class_names_path)
    logger.info(f"Saved class names: {class_names_path}")
    artifacts["class_names"] = class_names_path

    if config.get("export", {}).get("sync_flutter_labels", False):
        flutter_labels_path = os.path.normpath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "flutter_app",
                "assets",
                "labels",
                "labels.txt",
            )
        )
        if os.path.exists(os.path.dirname(flutter_labels_path)):
            save_class_names(class_names, flutter_labels_path)
            logger.info(f"Synchronized Flutter labels: {flutter_labels_path}")
            artifacts["flutter_labels"] = flutter_labels_path

    if selected_checkpoint:
        artifacts["winner_checkpoint"] = selected_checkpoint

    _log_run_summary(
        logger,
        merged_history,
        selected_phase,
        selected_loss,
        time.perf_counter() - training_started_at,
        artifacts=artifacts,
    )
    logger.info("Training completed successfully!")

    return model, history1


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="Train leaf disease detection model")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration file"
    )

    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.config:
        if os.path.isabs(args.config):
            config_path = args.config
        else:
            cwd_candidate = os.path.abspath(args.config)
            script_candidate = os.path.join(script_dir, args.config)
            config_path = (
                cwd_candidate if os.path.exists(cwd_candidate) else script_candidate
            )
    else:
        config_path = os.path.join(script_dir, "config.yaml")

    config_path = os.path.normpath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Check GPU availability
    devices = tf.config.list_physical_devices("GPU")
    print(
        f"Device: {'GPU' if devices else 'CPU'} "
        f"({len(devices)} visible), config: {config_path}",
        flush=True,
    )

    # Train model
    train_model(config_path)


if __name__ == "__main__":
    main()
