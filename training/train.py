"""
Training Script for Multi-Crop Leaf Disease Detection
Uses transfer learning with MobileNetV2 or EfficientNet-Lite0
"""

import os
import yaml
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

from model import get_model, unfreeze_base_model, print_model_summary
from utils import (
    load_dataset,
    create_augmentation_layer,
    compute_class_weights,
    setup_callbacks,
    setup_logging,
)


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def compile_model(model, config):
    """Compile model with optimizer and loss function"""

    # Setup optimizer
    optimizer_config = config["optimizer"]
    lr = optimizer_config["learning_rate"]

    if optimizer_config["name"].lower() == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_config["name"].lower() == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=lr)

    # Setup loss
    loss = config["loss"]["name"]

    # Setup metrics
    metrics = ["accuracy"]
    if "metrics" in config:
        metrics = config["metrics"]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def train_model(config_path="config.yaml"):
    """Main training function"""

    # Load configuration
    config = load_config(config_path)

    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting training...")
    logger.info(f"Configuration: {config['model']['architecture']}")

    # Set random seeds for reproducibility
    tf.random.set_seed(config["seed"])
    np.random.seed(config["seed"])

    # Load datasets
    logger.info("Loading datasets...")
    train_ds, val_ds, class_names = load_dataset(
        config["dataset"]["train_dir"],
        config["dataset"]["val_dir"],
        batch_size=config["dataset"]["batch_size"],
        image_size=tuple(config["model"]["input_shape"][:2]),
    )

    logger.info(f"Found {len(class_names)} classes")
    logger.info(f"Classes: {class_names}")

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
    )

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

    initial_epochs = config["training"].get("unfreeze_epoch", 10)

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    # Phase 2: Fine-tune with unfrozen layers
    if config["training"].get("freeze_base", True):
        logger.info("\n" + "=" * 50)
        logger.info("Phase 2: Fine-tuning (unfreezing base layers)")
        logger.info("=" * 50 + "\n")

        # Unfreeze base model
        unfreeze_from = config["training"].get("freeze_until_layer", 100)
        unfreeze_base_model(base_model, unfreeze_from)

        # Recompile with lower learning rate
        fine_tune_lr = config["training"].get("fine_tune_learning_rate", 0.0001)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=fine_tune_lr),
            loss=config["loss"]["name"],
            metrics=config.get("metrics", ["accuracy"]),
        )

        # Continue training
        total_epochs = config["training"]["epochs"]
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=initial_epochs,
            epochs=total_epochs,
            callbacks=callbacks,
            class_weight=class_weights,
        )

    # Save final model
    logger.info("Saving final model...")
    save_dir = config["export"]["save_dir"]
    model_name = config["model"]["architecture"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save in multiple formats
    if "h5" in config["export"]["formats"]:
        h5_path = os.path.join(save_dir, model_name, f"{model_name}_{timestamp}.h5")
        os.makedirs(os.path.dirname(h5_path), exist_ok=True)
        model.save(h5_path)
        logger.info(f"Saved H5 model: {h5_path}")

    if "saved_model" in config["export"]["formats"]:
        sm_path = os.path.join(save_dir, model_name, f"saved_model_{timestamp}")
        model.save(sm_path)
        logger.info(f"Saved SavedModel: {sm_path}")

    logger.info("Training completed successfully!")

    return model, history1


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="Train leaf disease detection model")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )

    args = parser.parse_args()

    # Check GPU availability
    print("GPU Available:", tf.config.list_physical_devices("GPU"))

    # Train model
    train_model(args.config)


if __name__ == "__main__":
    main()
