"""
Training Script for Multi-Crop Leaf Disease Detection
Uses transfer learning with the supported MobileNet, EfficientNet, and ShuffleNet backbones.
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import yaml
import argparse
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

try:
    # Support module execution: python -m training.train
    from .model import get_model, unfreeze_base_model, print_model_summary
    from .utils import (
        load_dataset,
        compute_class_weights,
        create_augmentation_layer,
        get_preprocess_fn,
        setup_callbacks,
        setup_logging,
        save_class_names,
        configure_runtime,
        choose_batch_size,
        validate_dataset_batch,
    )
except ImportError:
    # Fallback for script execution: python training/train.py
    from model import get_model, unfreeze_base_model, print_model_summary
    from utils import (
        load_dataset,
        compute_class_weights,
        create_augmentation_layer,
        get_preprocess_fn,
        setup_callbacks,
        setup_logging,
        save_class_names,
        configure_runtime,
        choose_batch_size,
        validate_dataset_batch,
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


def compile_model(model, config, phase="head"):
    """Compile a model for either classifier-head or fine-tuning training."""

    # Setup optimizer
    optimizer_config = dict(config.get("optimizer", {}))
    if phase == "fine_tune":
        fine_tune_config = config.get("training", {}).get("fine_tune_optimizer", {})
        if isinstance(fine_tune_config, str):
            fine_tune_config = {"name": fine_tune_config}
        if not isinstance(fine_tune_config, dict):
            fine_tune_config = {}
        optimizer_config.update(fine_tune_config)
        optimizer_config["learning_rate"] = config["training"].get(
            "fine_tune_learning_rate", optimizer_config.get("learning_rate", 1e-4)
        )
    optimizer_name = optimizer_config.get("name", "adam").lower()
    lr = optimizer_config.get("learning_rate", 0.001)
    decay = optimizer_config.get("decay", 0.0)
    weight_decay = decay if decay and decay > 0 else None

    if optimizer_name in {"adam", "adamw"}:
        adam_kwargs = {"learning_rate": lr}
        for key in ["beta_1", "beta_2", "epsilon"]:
            if key in optimizer_config and optimizer_config[key] is not None:
                adam_kwargs[key] = optimizer_config[key]
        for key in ["clipnorm", "clipvalue", "global_clipnorm"]:
            if key in optimizer_config and optimizer_config[key] is not None:
                adam_kwargs[key] = optimizer_config[key]
        if weight_decay is not None and optimizer_name == "adamw":
            adam_kwargs["weight_decay"] = weight_decay
        optimizer = keras.optimizers.Adam(**adam_kwargs)
        if optimizer_name == "adamw" and weight_decay is not None:
            try:
                optimizer = keras.optimizers.AdamW(
                    weight_decay=weight_decay, **{k: v for k, v in adam_kwargs.items() if k != "weight_decay"}
                )
            except AttributeError:
                # Older TensorFlow builds do not expose AdamW; Adam remains valid.
                optimizer = keras.optimizers.Adam(**adam_kwargs)
    elif optimizer_name == "sgd":
        sgd_kwargs = {
            "learning_rate": lr,
            "momentum": optimizer_config.get("momentum", 0.9),
        }
        for key in ["clipnorm", "clipvalue", "global_clipnorm"]:
            if key in optimizer_config and optimizer_config[key] is not None:
                sgd_kwargs[key] = optimizer_config[key]
        if weight_decay is not None:
            sgd_kwargs["weight_decay"] = weight_decay
        optimizer = keras.optimizers.SGD(**sgd_kwargs)
    elif optimizer_name == "rmsprop":
        rmsprop_kwargs = {
            "learning_rate": lr,
            "momentum": optimizer_config.get("momentum", 0.0),
        }
        for key in ["clipnorm", "clipvalue", "global_clipnorm"]:
            if key in optimizer_config and optimizer_config[key] is not None:
                rmsprop_kwargs[key] = optimizer_config[key]
        if weight_decay is not None:
            rmsprop_kwargs["weight_decay"] = weight_decay
        optimizer = keras.optimizers.RMSprop(**rmsprop_kwargs)
    else:
        raise ValueError(
            f"Unsupported optimizer '{optimizer_name}'. Supported: adam, adamw, sgd, rmsprop"
        )

    # Setup loss
    loss = _get_loss(config)

    # Setup metrics
    metric_names = config.get("metrics", ["accuracy"])
    metrics = build_metrics(metric_names, config["model"]["num_classes"])

    performance = config.get("performance", {})
    compile_kwargs = {
        "optimizer": optimizer,
        "loss": loss,
        "metrics": metrics,
    }
    steps_per_execution = int(performance.get("steps_per_execution", 1))
    if steps_per_execution > 1:
        compile_kwargs["steps_per_execution"] = steps_per_execution
    if performance.get("jit_compile", False):
        compile_kwargs["jit_compile"] = True

    model.compile(**compile_kwargs)

    return model


def train_model(config_path="config.yaml"):
    """Main training function"""

    # Load configuration
    config = load_config(config_path)
    config = resolve_config_paths(config, config_path)

    # Configure runtime before loading data or creating the model. The returned
    # strategy is reused for model creation, compilation, and fine-tuning.
    runtime = configure_runtime(config)
    strategy = runtime["strategy"]

    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting training...")
    arch = config.get("model", {}).get("architecture", "unknown")
    logger.info(f"Configuration: {arch}")
    batch_size = choose_batch_size(config, runtime, logger=logger)
    logger.info(
        "Runtime: GPUs=%d, strategy=%s, replicas=%d, global_batch_size=%d, "
        "per_replica_batch_size=%d, device_mode=%s",
        runtime["gpu_count"],
        runtime["strategy_name"],
        runtime["num_replicas"],
        batch_size,
        max(1, batch_size // max(1, runtime["num_replicas"])),
        "multi-GPU" if runtime["gpu_count"] > 1 else ("GPU" if runtime["has_gpu"] else "CPU"),
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

    # Load datasets
    logger.info("Loading datasets...")
    preprocess_fn = get_preprocess_fn(config["model"]["architecture"])
    augmentation = (
        create_augmentation_layer(config) if config.get("augmentation") else None
    )
    train_ds, val_ds, class_names = load_dataset(
        config["dataset"]["train_dir"],
        config["dataset"]["val_dir"],
        batch_size=batch_size,
        image_size=tuple(config["model"]["input_shape"][:2]),
        preprocess_fn=preprocess_fn,
        augmentation=augmentation,
        shuffle_buffer=config.get("dataset", {}).get("shuffle_buffer", 1000),
        cache_mode=config.get("dataset", {}).get("cache_mode", "none"),
        cache_memory_limit_gb=config.get("dataset", {}).get(
            "cache_memory_limit_gb", 8.0
        ),
        deterministic=config.get("performance", {}).get("deterministic", False),
    )

    logger.info(f"Found {len(class_names)} classes")
    logger.info(f"Classes: {class_names}")

    train_cardinality = tf.data.experimental.cardinality(train_ds)
    val_cardinality = tf.data.experimental.cardinality(val_ds)
    try:
        train_steps = int(train_cardinality.numpy())
        val_steps = int(val_cardinality.numpy())
    except (AttributeError, TypeError, ValueError):
        train_steps = val_steps = -1
    logger.info(
        "Training workload: %s train steps + %s validation steps per epoch",
        train_steps if train_steps >= 0 else "unknown",
        val_steps if val_steps >= 0 else "unknown",
    )

    if config.get("performance", {}).get("finite_batch_check", True):
        validate_dataset_batch(train_ds, logger=logger)
        validate_dataset_batch(val_ds, logger=logger)

    if len(class_names) != config["model"]["num_classes"]:
        raise ValueError(
            f"Class count mismatch: dataset has {len(class_names)} classes, "
            f"but config model.num_classes={config['model']['num_classes']}"
        )

    # Compute class weights if enabled
    class_weights = None
    if config.get("class_weights", {}).get("enabled", False):
        logger.info("Computing class weights...")
        max_weight = config.get("class_weights", {}).get("max_weight", 5.0)
        class_weights = compute_class_weights(
            config["dataset"]["train_dir"], max_weight=max_weight
        )
        logger.info(
            "Class weights enabled: min=%.3f max=%.3f",
            min(class_weights.values()) if class_weights else 0.0,
            max(class_weights.values()) if class_weights else 0.0,
        )

    # Create and compile the model inside the distribution scope. This is
    # required for MirroredStrategy to place variables on every replica.
    logger.info(f"Creating {config['model']['architecture']} model...")
    with strategy.scope():
        model_kwargs = dict(
            architecture=config["model"]["architecture"],
            input_shape=tuple(config["model"]["input_shape"]),
            num_classes=config["model"]["num_classes"],
            dropout_rate=config["model"]["dropout_rate"],
            weights=config["model"]["weights"],
            hub_url=config["model"].get("hub_url"),
            hub_cache_dir=config["model"].get("hub_cache_dir"),
            hub_download_retries=config["model"].get("hub_download_retries", 1),
            hub_download_delay_sec=config["model"].get("hub_download_delay_sec", 5),
        )
        if config["model"]["architecture"].lower() in {
            "mobilenetv2",
            "mobilenetv3_small",
            "mobilenetv3small",
            "mobilenet_v3_small",
            "shufflenetv2_05",
            "shufflenet_v2_05",
            "shufflenetv2_0.5x",
            "shufflenet_v2_0.5x",
        }:
            model_kwargs["classifier_units"] = config["model"].get(
                "classifier_units", 256
            )
        if config["model"]["architecture"].lower() in {
            "mobilenetv3_small",
            "mobilenetv3small",
            "mobilenet_v3_small",
        }:
            model_kwargs["include_preprocessing"] = config["model"].get(
                "include_preprocessing", False
            )
        model, base_model = get_model(**model_kwargs)

        if not freeze_base:
            logger.info("freeze_base is False: training backbone from the first epoch")
            unfreeze_base_model(base_model, unfreeze_from_layer=0)

        # Compile model
        logger.info("Compiling model...")
        model = compile_model(model, config)

    print_model_summary(model)

    # Setup callbacks
    logger.info("Setting up callbacks...")
    # Phase 1: Train classifier head with frozen base
    logger.info("\n" + "=" * 50)
    logger.info("Phase 1: Training classifier head (base frozen)")
    logger.info("=" * 50 + "\n")

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        callbacks=setup_callbacks(config, csv_append=False),
        class_weight=class_weights,
    )

    # Phase 2: Fine-tune with unfrozen layers
    history2 = None
    if freeze_base and initial_epochs < total_epochs:
        logger.info("\n" + "=" * 50)
        logger.info("Phase 2: Fine-tuning (unfreezing base layers)")
        logger.info("=" * 50 + "\n")

        # Unfreeze base model
        # Support both old config key "freeze_until_layer" and new "unfreeze_from_layer" for backwards compatibility
        unfreeze_from = config.get("training", {}).get("unfreeze_from_layer")
        if unfreeze_from is None:
            unfreeze_from = config.get("training", {}).get("freeze_until_layer", 100)
        with strategy.scope():
            unfreeze_base_model(base_model, unfreeze_from)

            # Recompile with the configured fine-tuning optimizer and lower LR.
            compile_model(model, config, phase="fine_tune")

        # Continue training
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=initial_epochs,
            epochs=total_epochs,
            callbacks=setup_callbacks(config, csv_append=True),
            class_weight=class_weights,
        )

    # Merge phase histories so downstream consumers get one continuous record
    merged_history = {k: list(v) for k, v in history1.history.items()}
    if history2 is not None:
        for key, values in history2.history.items():
            merged_history.setdefault(key, []).extend(values)
    history1.history = merged_history

    # Save final model
    logger.info("Saving final model...")
    save_dir = config.get("export", {}).get("save_dir", "../models")
    model_name = config.get("model", {}).get("architecture", "model")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
