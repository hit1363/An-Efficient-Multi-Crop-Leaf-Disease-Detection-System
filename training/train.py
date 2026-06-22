"""
Training Script for Multi-Crop Leaf Disease Detection
Uses transfer learning with MobileNetV2 or EfficientNet-Lite0
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import yaml
import argparse
import numpy as np
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
            built_metrics.append(
                keras.metrics.AUC(
                    name="auc",
                    multi_label=True,
                    num_labels=num_classes,
                )
            )
        else:
            # Keep custom/unknown metric names to avoid breaking user-provided settings.
            built_metrics.append(metric_name)

    return built_metrics


def compile_model(model, config):
    """Compile model with optimizer and loss function"""

    # Setup optimizer
    optimizer_config = config["optimizer"]
    optimizer_name = optimizer_config.get("name", "adam").lower()
    lr = optimizer_config.get("learning_rate", 0.001)
    decay = optimizer_config.get("decay", 0.0)
    weight_decay = decay if decay and decay > 0 else None

    if optimizer_name == "adam":
        adam_kwargs = {"learning_rate": lr}
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
        if weight_decay is not None:
            sgd_kwargs["weight_decay"] = weight_decay
        optimizer = keras.optimizers.SGD(**sgd_kwargs)
    elif optimizer_name == "rmsprop":
        rmsprop_kwargs = {
            "learning_rate": lr,
            "momentum": optimizer_config.get("momentum", 0.0),
        }
        if weight_decay is not None:
            rmsprop_kwargs["weight_decay"] = weight_decay
        optimizer = keras.optimizers.RMSprop(**rmsprop_kwargs)
    else:
        raise ValueError(
            f"Unsupported optimizer '{optimizer_name}'. Supported: adam, sgd, rmsprop"
        )

    # Setup loss
    loss = config["loss"]["name"]

    # Setup metrics
    metric_names = config.get("metrics", ["accuracy"])
    metrics = build_metrics(metric_names, config["model"]["num_classes"])

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def train_model(config_path="config.yaml"):
    """Main training function"""

    # Load configuration
    config = load_config(config_path)
    config = resolve_config_paths(config, config_path)

    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting training...")
    logger.info(f"Configuration: {config['model']['architecture']}")

    freeze_base = config.get("training", {}).get("freeze_base", True)
    total_epochs = int(config["training"]["epochs"])
    unfreeze_epoch = int(config["training"].get("unfreeze_epoch", 10))
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
    tf.random.set_seed(config["seed"])
    np.random.seed(config["seed"])

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
    )

    logger.info(f"Found {len(class_names)} classes")
    logger.info(f"Classes: {class_names}")

    if len(class_names) != config["model"]["num_classes"]:
        raise ValueError(
            f"Class count mismatch: dataset has {len(class_names)} classes, "
            f"but config model.num_classes={config['model']['num_classes']}"
        )

    # Compute class weights if enabled
    class_weights = None
    if config.get("class_weights", {}).get("enabled", False):
        logger.info("Computing class weights...")
        class_weights = compute_class_weights(config["dataset"]["train_dir"])

    # Create model
    logger.info(f"Creating {config['model']['architecture']} model...")
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
    )

    if not freeze_base:
        logger.info("freeze_base is False: training backbone from the first epoch")
        unfreeze_base_model(base_model, unfreeze_from_layer=0)

    print_model_summary(model)

    # Compile model
    logger.info("Compiling model...")
    model = compile_model(model, config)

    # Setup callbacks
    logger.info("Setting up callbacks...")
    callbacks = setup_callbacks(config)

    # Phase 1: Train classifier head with frozen base
    logger.info("\n" + "=" * 50)
    logger.info("Phase 1: Training classifier head (base frozen)")
    logger.info("=" * 50 + "\n")

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    # Phase 2: Fine-tune with unfrozen layers
    history2 = None
    if freeze_base and initial_epochs < total_epochs:
        logger.info("\n" + "=" * 50)
        logger.info("Phase 2: Fine-tuning (unfreezing base layers)")
        logger.info("=" * 50 + "\n")

        # Unfreeze base model
        unfreeze_from = config["training"].get("freeze_until_layer", 100)
        unfreeze_base_model(base_model, unfreeze_from)

        # Recompile with lower learning rate
        fine_tune_lr = config["training"].get("fine_tune_learning_rate", 0.0001)
        fine_tune_metrics = build_metrics(
            config.get("metrics", ["accuracy"]),
            config["model"]["num_classes"],
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=fine_tune_lr),
            loss=config["loss"]["name"],
            metrics=fine_tune_metrics,
        )

        # Continue training
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=initial_epochs,
            epochs=total_epochs,
            callbacks=callbacks,
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
    save_dir = config["export"]["save_dir"]
    model_name = config["model"]["architecture"]
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
