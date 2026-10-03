"""
Training Script for Multi-Crop Leaf Disease Detection
Uses transfer learning with MobileNetV2 or EfficientNet-Lite0
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse
import json
import numpy as np
import subprocess
import tensorflow as tf
import yaml
from tensorflow import keras
from datetime import datetime

try:
    # Support module execution: python -m training.train
    from .model import get_model, unfreeze_base_model, print_model_summary
    from .dataset_audit import audit_dataset_splits
    from .utils import (
        load_dataset,
        compute_class_weights,
        create_augmentation_layer,
        get_preprocess_fn,
        save_model_checkpoint,
        setup_callbacks,
        setup_logging,
        save_class_names,
        dataset_batch_count,
        per_replica_batch_size,
    )
except ImportError:
    # Fallback for script execution: python training/train.py
    from model import get_model, unfreeze_base_model, print_model_summary
    from dataset_audit import audit_dataset_splits
    from utils import (
        load_dataset,
        compute_class_weights,
        create_augmentation_layer,
        get_preprocess_fn,
        save_model_checkpoint,
        setup_callbacks,
        setup_logging,
        save_class_names,
        dataset_batch_count,
        per_replica_batch_size,
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

    csv_logger_cfg = callbacks_cfg.get("csv_logger", {})
    if "filename" in csv_logger_cfg:
        csv_logger_cfg["filename"] = _resolve_path(
            csv_logger_cfg["filename"], config_dir
        )

    evaluation_cfg = config.get("evaluation", {})
    if "results_dir" in evaluation_cfg:
        evaluation_cfg["results_dir"] = _resolve_path(
            evaluation_cfg["results_dir"], config_dir
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
        elif metric_key == "precision":
            built_metrics.append(keras.metrics.Precision(name="precision", top_k=1))
        elif metric_key == "recall":
            built_metrics.append(keras.metrics.Recall(name="recall", top_k=1))
        elif metric_key == "auc":
            # Single-label softmax classifier: macro AUC over the 1-hot labels.
            # multi_label=True would misreport AUC for this task.
            built_metrics.append(keras.metrics.AUC(name="auc"))
        else:
            # Keep custom/unknown metric names to avoid breaking user-provided settings.
            built_metrics.append(metric_name)

    return built_metrics


class FocalLoss(keras.losses.Loss):
    """Focal Loss for addressing class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Configurable via ``loss.gamma`` (default 2.0) and ``loss.alpha`` (default 0.25)
    in the YAML. Set ``loss.name: focal_loss`` to activate.
    """

    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Label smoothing
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes

        # Clip predictions for numerical stability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Compute focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        focal_loss = modulating_factor * tf.reduce_sum(cross_entropy, axis=-1)

        # Apply alpha weighting
        alpha_weight = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        alpha_weight = tf.reduce_sum(alpha_weight, axis=-1)
        focal_loss = alpha_weight * focal_loss

        return tf.reduce_mean(focal_loss)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha,
                        "label_smoothing": self.label_smoothing})
        return config


class BestValidationWeights(keras.callbacks.Callback):
    """Keep an in-memory copy of the lowest finite validation-loss weights.

    ``initial_best`` and ``initial_weights`` let a later training phase use the
    preceding phase's best validation point as its acceptance baseline. This
    guarantees that a harmful fine-tuning phase cannot replace a stronger
    frozen-backbone model merely because it completed an epoch.
    """

    def __init__(self, initial_best=float("inf"), initial_weights=None):
        super().__init__()
        self.best = float(initial_best)
        self.weights = initial_weights

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get("val_loss")
        if value is not None and np.isfinite(value) and value < self.best:
            self.best = float(value)
            self.weights = self.model.get_weights()


def select_final_weights(phase1_best, phase1_weights, phase2_best=None):
    """Return the best weights across phases, never degrading Phase 1.

    Fine-tuning is accepted only when its best finite validation loss strictly
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


def _count_images_by_class(data_dir):
    """Return deterministic image counts for reproducible run metadata."""
    if not data_dir or not os.path.isdir(data_dir):
        return {"total": 0, "classes": {}}

    valid_extensions = {".jpg", ".jpeg", ".png"}
    class_counts = {}
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        class_counts[class_name] = sum(
            1
            for filename in os.listdir(class_dir)
            if os.path.splitext(filename)[1].lower() in valid_extensions
        )
    return {"total": sum(class_counts.values()), "classes": class_counts}


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
    run_timestamp,
    dataset_audit=None,
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

    raw_values = list(raw_class_weights.values())
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
                **_count_images_by_class(dataset_config.get(f"{split}_dir")),
            }
            for split in ("train", "val", "test")
        },
        "dataset_audit": dataset_audit,
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
    }
    with open(
        os.path.join(run_dir, "run_metadata.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    return run_dir


def _get_loss(config):
    """Resolve the loss function from config.

    Supports:
      - ``categorical_crossentropy`` (default, passed as string)
      - ``focal_loss`` with optional ``loss.gamma`` and ``loss.alpha``
    """
    loss_cfg = config.get("loss", {})
    loss_name = loss_cfg.get("name", "categorical_crossentropy")

    if loss_name == "focal_loss":
        return FocalLoss(
            gamma=loss_cfg.get("gamma", 2.0),
            alpha=loss_cfg.get("alpha", 0.25),
            label_smoothing=loss_cfg.get("label_smoothing", 0.0),
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

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def create_fine_tune_optimizer(config):
    """Create a fresh, clipped Adam optimizer for backbone fine-tuning."""
    optimizer_config = config.get("optimizer", {})
    adam_kwargs = {
        "learning_rate": config.get("training", {}).get(
            "fine_tune_learning_rate", 0.0001
        )
    }
    clipnorm = optimizer_config.get("clipnorm")
    clipvalue = optimizer_config.get("clipvalue")
    if clipnorm:
        adam_kwargs["clipnorm"] = float(clipnorm)
    if clipvalue:
        adam_kwargs["clipvalue"] = float(clipvalue)
    for key in ["beta_1", "beta_2", "epsilon"]:
        if key in optimizer_config and optimizer_config[key] is not None:
            adam_kwargs[key] = optimizer_config[key]
    weight_decay = optimizer_config.get("decay", 0.0)
    if weight_decay and weight_decay > 0:
        adam_kwargs["weight_decay"] = weight_decay
    return keras.optimizers.Adam(**adam_kwargs)


def train_model(config_path="config.yaml"):
    """Main training function"""

    # Load configuration
    config = load_config(config_path)
    config = resolve_config_paths(config, config_path)

    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting training...")
    arch = config.get("model", {}).get("architecture", "unknown")
    logger.info(f"Configuration: {arch}")

    physical_gpus = tf.config.list_physical_devices("GPU")
    strategy = (
        tf.distribute.MirroredStrategy()
        if len(physical_gpus) > 1
        else tf.distribute.get_strategy()
    )
    logger.info(
        "Distribution strategy: %s (%d replica(s))",
        type(strategy).__name__,
        strategy.num_replicas_in_sync,
    )

    dataset_config = config["dataset"]
    audit_config = dataset_config.get("audit", {})
    dataset_audit = None
    if audit_config.get("enabled", True):
        logger.info("Auditing dataset class alignment and cross-split duplicates...")
        dataset_audit = audit_dataset_splits(
            {
                "train": dataset_config["train_dir"],
                "val": dataset_config["val_dir"],
                "test": dataset_config["test_dir"],
            }
        )
        for split, summary in dataset_audit["splits"].items():
            logger.info(
                "Dataset audit %s: %d images across %d classes (%d within-split duplicate files)",
                split,
                summary["total"],
                len(summary["classes"]),
                summary["within_split_duplicate_files"],
            )

    freeze_base = config.get("training", {}).get("freeze_base", True)
    total_epochs = int(config.get("training", {}).get("epochs", 50))
    unfreeze_epoch = int(config.get("training", {}).get("unfreeze_epoch", 10))
    if total_epochs < 1:
        raise ValueError("training.epochs must be >= 1")

    if freeze_base:
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

    # Load datasets
    logger.info("Loading datasets...")
    preprocess_fn = get_preprocess_fn(config["model"]["architecture"])
    augmentation = (
        create_augmentation_layer(config) if config.get("augmentation") else None
    )
    train_ds, val_ds, class_names = load_dataset(
        config["dataset"]["train_dir"],
        config["dataset"]["val_dir"],
        batch_size=config["dataset"]["batch_size"],
        image_size=tuple(config["model"]["input_shape"][:2]),
        preprocess_fn=preprocess_fn,
        augmentation=augmentation,
        shuffle_buffer=config.get("dataset", {}).get("shuffle_buffer", 1000),
        cache_mode=config.get("dataset", {}).get("cache_mode", "none"),
    )
    global_batch_size = int(config["dataset"]["batch_size"])
    replica_batch_size = per_replica_batch_size(
        global_batch_size, strategy.num_replicas_in_sync
    )
    train_steps = dataset_batch_count(train_ds, "training")
    val_steps = dataset_batch_count(val_ds, "validation")
    logger.info(
        "Dataset batches: train=%d, validation=%d, global_batch_size=%d, "
        "per_replica_batch_size=%d",
        train_steps,
        val_steps,
        global_batch_size,
        replica_batch_size,
    )

    logger.info(f"Found {len(class_names)} classes")
    logger.info(f"Classes: {class_names}")

    if len(class_names) != config["model"]["num_classes"]:
        raise ValueError(
            f"Class count mismatch: dataset has {len(class_names)} classes, "
            f"but config model.num_classes={config['model']['num_classes']}"
        )

    # Compute the raw range for run metadata even when the stability-first
    # baseline leaves class weighting disabled.
    weight_config = config.get("class_weights", {})
    raw_class_weights = compute_class_weights(config["dataset"]["train_dir"])

    # Compute class weights only when explicitly enabled for a later
    # imbalance-focused experiment.
    class_weights = None
    if weight_config.get("enabled", False):
        logger.info("Computing class weights...")
        class_weights = compute_class_weights(
            config["dataset"]["train_dir"],
            max_weight=weight_config.get("max_weight"),
            normalize=weight_config.get("normalize", True),
        )

    # Create model
    logger.info(f"Creating {config['model']['architecture']} model...")
    with strategy.scope():
        model, base_model = get_model(
            architecture=config["model"]["architecture"],
            input_shape=tuple(config["model"]["input_shape"]),
            num_classes=config["model"]["num_classes"],
            dropout_rate=config["model"]["dropout_rate"],
            weights=config["model"]["weights"],
            hub_url=config["model"].get("hub_url"),
            hub_cache_dir=config["model"].get("hub_cache_dir"),
            hub_download_retries=config["model"].get("hub_download_retries", 1),
            hub_download_delay_sec=config["model"].get("hub_download_delay_sec", 5),
            head_units=tuple(config["model"].get("head_units", [256, 128])),
            l2_regularization=config["model"].get("l2_regularization", 0.0),
        )

    if not freeze_base:
        logger.info("freeze_base is False: training backbone from the first epoch")
        unfreeze_base_model(base_model, unfreeze_from_layer=0)

    print_model_summary(model)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = save_run_metadata(
        config=config,
        config_path=config_path,
        model=model,
        raw_class_weights=raw_class_weights,
        applied_class_weights=class_weights,
        run_timestamp=run_timestamp,
        dataset_audit=dataset_audit,
    )
    logger.info("Saved reproducibility metadata to: %s", run_dir)

    # Compile model
    logger.info("Compiling model...")
    with strategy.scope():
        model = compile_model(model, config)

    # Repeat only the training stream and set its exact finite epoch length.
    # This prevents a distributed iterator from exhausting across consecutive
    # epochs while keeping validation bounded and repeatable.
    train_ds = train_ds.repeat()

    # Setup callbacks
    logger.info("Setting up callbacks...")
    callbacks = setup_callbacks(config, csv_append=False)
    phase1_best = BestValidationWeights()
    callbacks.append(phase1_best)

    # Phase 1: Train classifier head with frozen base
    logger.info("\n" + "=" * 50)
    logger.info("Phase 1: Training classifier head (base frozen)")
    logger.info("=" * 50 + "\n")

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        epochs=initial_epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    # Never export a model that has not completed at least one finite
    # validation measurement. TerminateOnNaN stops the fit loop, but without
    # this guard its last (possibly corrupted) weights could otherwise be
    # exported as a fallback.
    if phase1_best.weights is None:
        raise RuntimeError(
            "Phase 1 produced no finite validation loss; refusing to fine-tune "
            "or export potentially corrupted weights. Check data, learning rate, "
            "and accelerator numerical precision."
        )

    # The last Phase 1 epoch is not necessarily the best validation point.
    # Keep its best restored weights as the stable starting point for Phase 2.
    phase1_best_weights = phase1_best.weights
    # This restoration is required even when Phase 2 is skipped (for example,
    # a head-only baseline where epochs == unfreeze_epoch). Otherwise the
    # subsequent export would silently use the final, potentially overfit
    # Phase 1 epoch instead of the validated winner.
    model.set_weights(phase1_best_weights)

    # Phase 2: Fine-tune with unfrozen layers
    history2 = None
    selected_checkpoint = None
    if freeze_base and initial_epochs < total_epochs:
        logger.info("\n" + "=" * 50)
        logger.info("Phase 2: Fine-tuning (unfreezing base layers)")
        logger.info("=" * 50 + "\n")

        # Unfreeze base model
        # Support both old config key "freeze_until_layer" and new "unfreeze_from_layer" for backwards compatibility
        unfreeze_from = config.get("training", {}).get("unfreeze_from_layer") or \
                       config.get("training", {}).get("freeze_until_layer", 100)
        unfreeze_base_model(base_model, unfreeze_from)
        model.set_weights(phase1_best_weights)

        # Recompile with lower learning rate
        fine_tune_metrics = build_metrics(
            config.get("metrics", ["accuracy"]),
            config["model"]["num_classes"],
        )
        with strategy.scope():
            model.compile(
                optimizer=create_fine_tune_optimizer(config),
                loss=_get_loss(config),
                metrics=fine_tune_metrics,
            )

        # Callbacks retain monitor/patience state. A fresh set isolates the
        # second optimizer phase from Phase 1's plateau state.
        phase2_callbacks = setup_callbacks(
            config, checkpoint_name="best_phase2_model", csv_append=True
        )
        # Seed Phase 2 with the best Phase 1 validation point. If fine-tuning
        # never beats it, final export intentionally remains the Phase 1 model.
        phase2_best = BestValidationWeights(
            initial_best=phase1_best.best,
            initial_weights=phase1_best_weights,
        )
        phase2_callbacks.append(phase2_best)

        # Continue training
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            initial_epoch=initial_epochs,
            epochs=total_epochs,
            callbacks=phase2_callbacks,
            class_weight=class_weights,
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
    # Merge phase histories so downstream consumers get one continuous record
    merged_history = {k: list(v) for k, v in history1.history.items()}
    if history2 is not None:
        for key, values in history2.history.items():
            merged_history.setdefault(key, []).extend(values)
    history1.history = merged_history

    # Persist the actual selected in-memory winner in both the head-only and
    # two-phase paths. This is also the artifact evaluated on the untouched test
    # split, so test metrics always match the model chosen by validation loss.
    selected_checkpoint = save_model_checkpoint(model, config)
    if selected_checkpoint:
        logger.info("Saved selected validation winner checkpoint to: %s", selected_checkpoint)

    # Save final model
    logger.info("Saving final model...")
    save_dir = config.get("export", {}).get("save_dir", "../models")
    model_name = config.get("model", {}).get("architecture", "model")
    timestamp = run_timestamp

    # Save in multiple formats
    is_lite0_arch = model_name.lower() in ["efficientnet", "efficientnet_lite0"]

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

    if "tflite" in config["export"]["formats"]:
        tflite_path = os.path.join(
            save_dir, model_name, f"{model_name}_{timestamp}.tflite"
        )
        os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        logger.info(f"Saved TFLite model: {tflite_path}")

    class_names_path = os.path.join(save_dir, model_name, "class_names.txt")
    save_class_names(class_names, class_names_path)
    logger.info(f"Saved class names: {class_names_path}")

    evaluation_config = config.get("evaluation", {})
    if evaluation_config.get("run_after_training", False):
        if not selected_checkpoint:
            raise RuntimeError(
                "Post-training evaluation requires an enabled full-model checkpoint."
            )
        logger.info("Evaluating selected checkpoint on the untouched test split...")
        try:
            from .evaluate import evaluate_model
        except ImportError:
            from evaluate import evaluate_model
        evaluate_model(
            selected_checkpoint,
            config_path,
            results_dir=evaluation_config.get("results_dir"),
        )

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
    print("GPU Available:", tf.config.list_physical_devices("GPU"))
    print("Using config:", config_path)

    # Train model
    train_model(config_path)


if __name__ == "__main__":
    main()
