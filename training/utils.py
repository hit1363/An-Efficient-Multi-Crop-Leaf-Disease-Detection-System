"""
Utility Functions for Training Pipeline
"""

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight


def setup_logging(config):
    """Setup logging configuration"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))

    logging.basicConfig(
        level=level,
        format=log_config.get("format", "%(asctime)s - %(levelname)s - %(message)s"),
    )

    logger = logging.getLogger(__name__)

    # Save to file if enabled
    if log_config.get("save_to_file", False):
        log_file = log_config.get("log_file", "training.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_config["format"]))
        logger.addHandler(file_handler)

    return logger


def load_dataset(train_dir, val_dir, batch_size=32, image_size=(224, 224)):
    """
    Load training and validation datasets

    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        batch_size: Batch size for training
        image_size: Target image size (height, width)

    Returns:
        train_ds, val_ds, class_names
    """
    # Load training dataset
    train_ds = keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42,
    )

    # Load validation dataset
    val_ds = keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names

    # Optimize dataset performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def create_augmentation_layer(config):
    """
    Create data augmentation layer

    Args:
        config: Augmentation configuration

    Returns:
        Keras Sequential model with augmentation layers
    """
    aug_config = config.get("augmentation", {})

    augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(aug_config.get("rotation_range", 45) / 360),
            keras.layers.RandomZoom(aug_config.get("zoom_range", [0.8, 1.2])[0] - 1),
            keras.layers.RandomContrast(0.2),
        ]
    )

    return augmentation


def compute_class_weights(data_dir):
    """
    Compute class weights for imbalanced dataset

    Args:
        data_dir: Directory containing class subdirectories

    Returns:
        Dictionary of class weights
    """
    class_counts = {}
    class_names = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )

    for i, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        count = len(
            [
                f
                for f in os.listdir(class_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        class_counts[i] = count

    total = sum(class_counts.values())
    class_weights = {
        i: total / (len(class_counts) * count) for i, count in class_counts.items()
    }

    return class_weights


def setup_callbacks(config):
    """
    Setup training callbacks

    Args:
        config: Configuration dictionary

    Returns:
        List of Keras callbacks
    """
    callbacks = []
    callback_config = config.get("callbacks", {})

    # Model Checkpoint
    if callback_config.get("checkpoint", {}).get("enabled", True):
        checkpoint_config = callback_config["checkpoint"]
        checkpoint_path = os.path.join(
            config["export"]["save_dir"],
            config["model"]["architecture"],
            "checkpoints",
            "best_model.h5",
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor=checkpoint_config.get("monitor", "val_accuracy"),
                save_best_only=checkpoint_config.get("save_best_only", True),
                mode=checkpoint_config.get("mode", "max"),
                verbose=1,
            )
        )

    # Early Stopping
    if callback_config.get("early_stopping", {}).get("enabled", True):
        es_config = callback_config["early_stopping"]
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=es_config.get("monitor", "val_loss"),
                patience=es_config.get("patience", 10),
                restore_best_weights=es_config.get("restore_best_weights", True),
                mode=es_config.get("mode", "min"),
                verbose=1,
            )
        )

    # Learning Rate Scheduler
    if config.get("lr_schedule", {}).get("type") == "reduce_on_plateau":
        lr_config = config["lr_schedule"]
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=lr_config.get("monitor", "val_loss"),
                factor=lr_config.get("factor", 0.5),
                patience=lr_config.get("patience", 5),
                min_lr=lr_config.get("min_lr", 1e-7),
                verbose=1,
            )
        )

    # TensorBoard
    if callback_config.get("tensorboard", {}).get("enabled", False):
        tb_config = callback_config["tensorboard"]
        log_dir = tb_config.get("log_dir", "../logs")
        os.makedirs(log_dir, exist_ok=True)
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=tb_config.get("histogram_freq", 1),
                write_graph=tb_config.get("write_graph", True),
            )
        )

    # CSV Logger
    if callback_config.get("csv_logger", {}).get("enabled", True):
        csv_config = callback_config["csv_logger"]
        csv_path = csv_config.get("filename", "../results/training_log.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        callbacks.append(keras.callbacks.CSVLogger(csv_path))

    return callbacks


def preprocess_image(image_path, image_size=(224, 224)):
    """
    Preprocess single image for inference

    Args:
        image_path: Path to image file
        image_size: Target size

    Returns:
        Preprocessed image tensor
    """
    img = keras.preprocessing.image.load_img(image_path, target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    return img_array


def save_class_names(class_names, save_path="../models/class_names.txt"):
    """Save class names to text file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        for name in class_names:
            f.write(f"{name}\n")


def load_class_names(file_path="../models/class_names.txt"):
    """Load class names from text file"""
    with open(file_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")

    # Test class weight computation
    # class_weights = compute_class_weights('../dataset/processed/train')
    # print("Class weights:", class_weights)
