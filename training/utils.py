"""
Utility Functions for Training Pipeline
"""

import os
import logging
import shutil
import tempfile
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras


def configure_runtime(config):
    """Configure reproducibility and optional GPU performance features."""
    seed = int(config.get("seed", 42))
    keras.utils.set_random_seed(seed)

    performance = config.get("performance", {})
    deterministic = bool(performance.get("deterministic", False))
    if deterministic:
        tf.config.experimental.enable_op_determinism()

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # Memory growth must be set before the device is initialized.
            pass

    mixed_precision_requested = bool(performance.get("mixed_precision", False))
    if mixed_precision_requested and gpus:
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled for GPU training.")
    elif mixed_precision_requested:
        print("Mixed precision requested, but no GPU was found; using float32.")

    if bool(performance.get("xla", False)):
        tf.config.optimizer.set_jit(True)
        print("XLA JIT enabled.")

    distribution = performance.get("distribution", "auto")
    distribution_key = (
        distribution.strip().lower() if isinstance(distribution, str) else distribution
    )
    distribution_enabled = distribution_key not in {
        False,
        None,
        "false",
        "off",
        "none",
    }
    if distribution_enabled and len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        strategy_name = "MirroredStrategy"
    else:
        strategy = tf.distribute.get_strategy()
        strategy_name = "DefaultStrategy"

    print(
        "Distribution strategy: "
        f"{strategy_name} ({strategy.num_replicas_in_sync} replica(s))"
    )

    return {
        "seed": seed,
        "deterministic": deterministic,
        "has_gpu": bool(gpus),
        "gpu_count": len(gpus),
        "strategy": strategy,
        "strategy_name": strategy_name,
        "num_replicas": strategy.num_replicas_in_sync,
    }


def _normalize_cache_mode(cache_mode):
    if cache_mode is None:
        return "none"

    mode = str(cache_mode).strip().lower()
    if mode in {"none", "disabled", "off"}:
        return "none"
    if mode in {"auto", "automatic"}:
        return "auto"
    if mode in {"memory", "in_memory", "ram"}:
        return "memory"
    if mode in {"disk", "file", "on_disk"}:
        return "disk"

    raise ValueError("dataset.cache_mode must be one of: none, memory, disk, auto")


def setup_logging(config):
    """Setup logging configuration"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    format_str = log_config.get("format", "%(asctime)s - %(levelname)s - %(message)s")

    logging.basicConfig(
        level=level,
        format=format_str,
    )

    logger = logging.getLogger(__name__)

    # Save to file if enabled
    if log_config.get("save_to_file", False):
        log_file = log_config.get("log_file", "training.log")
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_str))
        logger.addHandler(file_handler)

    return logger


def get_preprocess_fn(architecture):
    """Return the appropriate preprocess_input function for a given architecture."""
    if not architecture:
        return None

    arch = architecture.lower()
    if arch == "mobilenetv2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    if arch in {"mobilenetv3_small", "mobilenetv3small", "mobilenet_v3_small"}:
        # MobileNetV3's Keras ``preprocess_input`` is a compatibility
        # placeholder. Because this project disables the backbone's built-in
        # Rescaling layer, perform the documented [-1, 1] conversion here.
        return lambda x: tf.cast(x, tf.float32) / 127.5 - 1.0
    if arch in {
        "shufflenetv2_05",
        "shufflenet_v2_05",
        "shufflenetv2_0.5x",
        "shufflenet_v2_0.5x",
    }:
        # Native ShuffleNetV2 uses the same [-1, 1] input contract as MobileNetV2.
        return tf.keras.applications.mobilenet_v2.preprocess_input
    if arch == "efficientnet_b0":
        return tf.keras.applications.efficientnet.preprocess_input
    if arch in {"efficientnet", "efficientnet_lite0"}:
        return lambda x: tf.cast(x, tf.float32) / 255.0
    return None


def _dataset_cache_path(name, cache_key=None):
    """Return a writable cache file path for the current notebook/runtime."""
    candidates = []
    if os.path.isdir("/kaggle/working"):
        candidates.append(os.path.join("/kaggle/working", "dataset_cache"))
    if os.path.isdir("/content"):
        candidates.append(os.path.join("/content", "dataset_cache"))
    candidates.append(os.path.join(tempfile.gettempdir(), "leaf_dataset_cache"))

    for cache_dir in candidates:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            test_file = os.path.join(cache_dir, ".write_test")
            with open(test_file, "w", encoding="utf-8") as handle:
                handle.write("ok")
            os.remove(test_file)
            suffix = f"_{cache_key}" if cache_key else ""
            return os.path.join(cache_dir, name + suffix)
        except OSError:
            continue

    return None


def _available_host_memory_gb():
    """Return available host RAM when psutil is available."""
    try:
        import psutil

        return float(psutil.virtual_memory().available) / (1024**3)
    except (ImportError, AttributeError, OSError):
        return None


def _available_disk_gb(path):
    """Return free disk space for ``path`` when it can be inspected."""
    try:
        return float(shutil.disk_usage(path).free) / (1024**3)
    except (OSError, TypeError):
        return None


def _resolve_cache_mode(
    cache_mode,
    train_ds,
    val_ds,
    image_size,
    batch_size,
    memory_limit_gb=8.0,
):
    """Resolve ``auto`` caching without risking an uncontrolled RAM cache."""
    requested = _normalize_cache_mode(cache_mode)
    if requested not in {"auto", "disk"}:
        return requested

    train_cardinality = tf.data.experimental.cardinality(train_ds)
    val_cardinality = tf.data.experimental.cardinality(val_ds)
    try:
        # Cardinality is measured in batches because the source datasets are
        # batched. This is intentionally an upper-bound estimate for the last
        # partial batch, which keeps the memory decision conservative.
        train_count = int(train_cardinality.numpy()) * int(batch_size)
        val_count = int(val_cardinality.numpy()) * int(batch_size)
    except (AttributeError, TypeError, ValueError):
        train_count = val_count = -1

    # The cache is created before augmentation and preprocessing, so estimate
    # the raw uint8 image footprint. Keep a generous headroom factor for TF
    # bookkeeping and the rest of the notebook process.
    required_gb = 0.0
    if train_count >= 0 and val_count >= 0:
        height, width = image_size[:2]
        required_gb = (train_count + val_count) * height * width * 3 / (1024**3)
        required_gb *= 1.25

    available_gb = _available_host_memory_gb()
    # Available RAM is not the same as safe cache capacity: TensorFlow,
    # multiprocessing workers, model weights, and augmentation need the rest.
    # Keep large hosted datasets on disk even when the VM reports abundant RAM.
    if requested == "auto" and (
        available_gb is not None
        and required_gb > 0
        and required_gb <= float(memory_limit_gb)
        and available_gb >= required_gb * 1.5
    ):
        return "memory"

    # Disk caching is slower than RAM but avoids killing hosted runtimes when
    # the dataset is too large for a safe in-memory cache. Do not select it if
    # the target filesystem cannot hold the estimated cache.
    for candidate in ("/kaggle/working", "/content", tempfile.gettempdir()):
        if os.path.isdir(candidate):
            try:
                free_gb = _available_disk_gb(candidate)
                if required_gb <= 0 or (
                    free_gb is not None and free_gb >= required_gb * 1.25
                ):
                    return "disk"
            except OSError:
                continue

    return "none"


def load_dataset(
    train_dir,
    val_dir,
    batch_size=32,
    image_size=(224, 224),
    preprocess_fn=None,
    augmentation=None,
    shuffle_buffer=1000,
    cache_mode="none",
    cache_memory_limit_gb=8.0,
    deterministic=False,
):
    """
    Load training and validation datasets

    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        batch_size: Batch size for training
        image_size: Target image size (height, width)
        preprocess_fn: Deterministic preprocessing applied to both splits
        augmentation: Random augmentation layer applied to training split only
        shuffle_buffer: Shuffle buffer size for the training split. Shuffling is
            applied AFTER caching so the order reshuffles every epoch instead of
            being frozen in the cache.
        cache_mode: Dataset cache strategy. Use "none" to disable caching,
            "memory" to keep cached batches in RAM, "disk" to cache to a
            writable filesystem path, or "auto" to select RAM only when the
            estimated footprint is safe and otherwise use disk/no cache.

    Returns:
        train_ds, val_ds, class_names
    """
    # Batch at the source. The previous unbatched -> shuffle -> batch path
    # added substantial tf.data overhead for large hosted datasets.
    train_ds = keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
        seed=42,
    )

    val_ds = keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    options = tf.data.Options()
    options.experimental_deterministic = bool(deterministic)

    cache_mode = _resolve_cache_mode(
        cache_mode,
        train_ds,
        val_ds,
        image_size,
        batch_size,
        memory_limit_gb=cache_memory_limit_gb,
    )
    logger = logging.getLogger(__name__)

    # Cache decoded/resized batches before augmentation so augmentation remains
    # stochastic on every epoch while image decoding happens only once.
    if cache_mode == "disk":
        height, width = image_size[:2]
        cache_key = f"v2_{height}x{width}_b{int(batch_size)}"
        train_cache = _dataset_cache_path("train_cache", cache_key)
        val_cache = _dataset_cache_path("val_cache", cache_key)
        if train_cache and val_cache:
            train_ds = train_ds.cache(train_cache)
            val_ds = val_ds.cache(val_cache)
        else:
            logger.warning("Disk cache is unavailable; continuing without caching.")
            cache_mode = "none"
    elif cache_mode == "memory":
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    # Shuffle batches after caching so cached datasets still reshuffle each
    # epoch. The source dataset is already batched, avoiding the expensive
    # individual-example shuffle and re-batch path.
    if shuffle_buffer and shuffle_buffer > 0:
        shuffle_batches = max(1, int(np.ceil(shuffle_buffer / batch_size)))
        train_ds = train_ds.shuffle(
            buffer_size=shuffle_batches, seed=42, reshuffle_each_iteration=True
        )

    # Augmentation only on training data. Runs each epoch because it sits
    # after the cache.
    if augmentation is not None:
        train_ds = train_ds.map(
            lambda x, y: (augmentation(tf.cast(x, tf.float32), training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    # Deterministic preprocessing on both splits.
    if preprocess_fn is not None:

        def _apply_preprocess(x, y):
            x = tf.cast(x, tf.float32)
            x = preprocess_fn(x)
            return x, y

        train_ds = train_ds.map(_apply_preprocess, num_parallel_calls=AUTOTUNE)
        val_ds = val_ds.map(_apply_preprocess, num_parallel_calls=AUTOTUNE)

    # Allow non-deterministic interleave/map scheduling when explicitly enabled.
    # This improves GPU utilization without changing labels or validation order.
    train_ds = train_ds.with_options(options).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.with_options(options).prefetch(buffer_size=AUTOTUNE)

    logger.info(
        "Dataset pipeline: batch_size=%d, train_steps=%s, val_steps=%s, "
        "cache_mode=%s, shuffle_batch_buffer=%d",
        batch_size,
        train_ds.cardinality(),
        val_ds.cardinality(),
        cache_mode,
        max(1, int(np.ceil(shuffle_buffer / batch_size))) if shuffle_buffer else 0,
    )

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

    zoom_range = aug_config.get("zoom_range", [0.8, 1.2])
    if isinstance(zoom_range, (list, tuple)) and len(zoom_range) == 2:
        zoom_factor = (zoom_range[0] - 1.0, zoom_range[1] - 1.0)
    else:
        zoom_factor = zoom_range

    layers = []
    horizontal_flip = aug_config.get("horizontal_flip", True)
    vertical_flip = aug_config.get("vertical_flip", False)
    if horizontal_flip and vertical_flip:
        flip_mode = "horizontal_and_vertical"
    elif horizontal_flip:
        flip_mode = "horizontal"
    elif vertical_flip:
        flip_mode = "vertical"
    else:
        flip_mode = None
    if flip_mode:
        layers.append(keras.layers.RandomFlip(flip_mode))

    rotation_range = float(aug_config.get("rotation_range", 45))
    if rotation_range > 0:
        layers.append(keras.layers.RandomRotation(rotation_range / 360))
    layers.append(keras.layers.RandomZoom(zoom_factor))
    layers.append(keras.layers.RandomContrast(0.2))

    augmentation = keras.Sequential(layers)

    brightness_range = aug_config.get("brightness_range")
    random_brightness = getattr(keras.layers, "RandomBrightness", None)
    if brightness_range and random_brightness is not None:
        if isinstance(brightness_range, (list, tuple)) and len(brightness_range) == 2:
            brightness_factor = (
                brightness_range[0] - 1.0,
                brightness_range[1] - 1.0,
            )
        else:
            brightness_factor = brightness_range
        augmentation.add(
            random_brightness(brightness_factor, value_range=(0, 255))
        )

    return augmentation


class ThroughputCallback(keras.callbacks.Callback):
    """Log epoch duration and effective image throughput."""

    def __init__(self, batch_size, cache_mode="none"):
        super().__init__()
        self.batch_size = int(batch_size)
        self.cache_mode = str(cache_mode)
        self._epoch_started = None

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_started = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        if self._epoch_started is None:
            return
        elapsed = max(time.perf_counter() - self._epoch_started, 1e-6)
        steps = self.params.get("steps") or 0
        images = int(steps) * self.batch_size
        images_per_second = images / elapsed if images else 0.0
        logging.getLogger(__name__).info(
            "Epoch %d throughput: %.1f images/sec (%.2f min, %d steps, cache=%s)",
            epoch + 1,
            images_per_second,
            elapsed / 60.0,
            int(steps),
            self.cache_mode,
        )
        if epoch == 0 and self.cache_mode in {"memory", "disk", "auto"}:
            logging.getLogger(__name__).info(
                "The first epoch may be slower while the dataset cache is populated."
            )


def compute_class_weights(data_dir, max_weight=5.0):
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
    nonzero = {i: count for i, count in class_counts.items() if count > 0}
    if not nonzero:
        return {}

    class_weights = {}
    max_weight = float(max_weight) if max_weight is not None else float("inf")
    if max_weight <= 0 or (
        not np.isfinite(max_weight) and max_weight != float("inf")
    ):
        raise ValueError("max_weight must be positive or None")
    for i, count in class_counts.items():
        if count > 0:
            weight = total / (len(nonzero) * count)
            class_weights[i] = min(float(weight), max_weight)
        else:
            class_weights[i] = 0.0

    if not all(np.isfinite(weight) for weight in class_weights.values()):
        raise ValueError("Class-weight computation produced a non-finite value")

    return class_weights


def validate_dataset_batch(dataset, logger=None):
    """Validate one preprocessed batch before allocating a training graph."""
    logger = logger or logging.getLogger(__name__)
    try:
        images, labels = next(iter(dataset.take(1)))
    except (StopIteration, tf.errors.InvalidArgumentError) as exc:
        raise ValueError("The training dataset produced no readable batches") from exc

    tf.debugging.check_numerics(images, "Non-finite image values in dataset")
    tf.debugging.check_numerics(labels, "Non-finite labels in dataset")
    if images.shape.rank != 4 or labels.shape.rank != 2:
        raise ValueError(
            f"Unexpected batch shapes: images={images.shape}, labels={labels.shape}"
        )
    logger.info(
        "Finite batch check passed: images=%s dtype=%s range=[%.4f, %.4f], labels=%s",
        tuple(images.shape),
        images.dtype.name,
        float(tf.reduce_min(images).numpy()),
        float(tf.reduce_max(images).numpy()),
        tuple(labels.shape),
    )


def choose_batch_size(config, runtime, logger=None):
    """Choose a conservative global batch size before building the dataset.

    A process killed by the host cannot be recovered by catching an exception.
    Therefore the fallback is selected before ``model.fit`` based on the
    available replica profile. Dual-GPU runs keep the tuned global batch size;
    single-GPU/CPU runs use the first configured fallback when appropriate.
    """
    logger = logger or logging.getLogger(__name__)
    dataset_cfg = config.setdefault("dataset", {})
    requested = int(dataset_cfg.get("batch_size", 32))
    candidates = dataset_cfg.get("batch_size_fallbacks", [])
    if not isinstance(candidates, (list, tuple)):
        candidates = []
    candidates = [int(value) for value in candidates if int(value) > 0]
    if requested not in candidates:
        candidates.insert(0, requested)

    auto_fallback = bool(dataset_cfg.get("auto_batch_fallback", True))
    replicas = int(runtime.get("num_replicas", 1))
    has_gpu = bool(runtime.get("has_gpu", False))
    selected = requested
    if auto_fallback and replicas <= 1 and candidates:
        # The first fallback is the safe single-device profile. Do not reduce
        # dual-T4 throughput, which is why global batch 128 remains default.
        selected = candidates[min(1, len(candidates) - 1)]
        if not has_gpu:
            selected = candidates[-1]

    dataset_cfg["batch_size"] = selected
    if selected != requested:
        logger.warning(
            "Using conservative batch-size fallback: requested=%d, selected=%d, "
            "replicas=%d, gpu=%s",
            requested,
            selected,
            replicas,
            has_gpu,
        )
    return selected


def setup_callbacks(config, csv_append=None):
    """
    Setup training callbacks

    Args:
        config: Configuration dictionary

    Returns:
        List of Keras callbacks
    """
    callbacks = []
    callback_config = config.get("callbacks", {})
    performance_config = config.get("performance", {})

    if performance_config.get("terminate_on_nan", True):
        callbacks.append(keras.callbacks.TerminateOnNaN())

    if performance_config.get("log_throughput", True):
        callbacks.append(
            ThroughputCallback(
                batch_size=config.get("dataset", {}).get("batch_size", 32),
                cache_mode=config.get("dataset", {}).get("cache_mode", "none"),
            )
        )

    # Model Checkpoint
    if callback_config.get("checkpoint", {}).get("enabled", True):
        checkpoint_config = callback_config["checkpoint"]
        architecture = config.get("model", {}).get("architecture", "").lower()
        is_lite0_arch = architecture in ["efficientnet", "efficientnet_lite0"]

        save_weights_only = checkpoint_config.get("save_weights_only", False)
        if is_lite0_arch and not save_weights_only:
            # TF Hub-backed models are more reliable with weights-only checkpoints.
            save_weights_only = True

        checkpoint_filename = (
            "best_model.weights.h5" if save_weights_only else "best_model.h5"
        )
        checkpoint_path = os.path.join(
            config["export"]["save_dir"],
            config["model"]["architecture"],
            "checkpoints",
            checkpoint_filename,
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor=checkpoint_config.get("monitor", "val_accuracy"),
                save_best_only=checkpoint_config.get("save_best_only", True),
                save_weights_only=save_weights_only,
                mode=checkpoint_config.get("mode", "max"),
                save_freq=checkpoint_config.get("save_freq", "epoch"),
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
    lr_config = config.get("lr_schedule", {})
    lr_type = lr_config.get("type", "reduce_on_plateau")
    initial_lr = lr_config.get("initial_learning_rate",
                              config.get("optimizer", {}).get("learning_rate", 0.001))

    if lr_type == "reduce_on_plateau":
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=lr_config.get("monitor", "val_loss"),
                factor=lr_config.get("factor", 0.5),
                patience=lr_config.get("patience", 5),
                min_lr=lr_config.get("min_lr", 1e-7),
                verbose=1,
            )
        )
    elif lr_type == "exponential_decay":
        decay_epochs = float(
            lr_config.get("decay_epochs", config.get("training", {}).get("epochs", 50))
        )
        decay_rate = lr_config.get("decay_rate", 0.96)

        def schedule(epoch, _current_lr):
            exponent = int(epoch) if lr_config.get("staircase", False) else float(epoch)
            return initial_lr * (decay_rate ** (exponent / max(decay_epochs, 1.0)))

        callbacks.append(
            keras.callbacks.LearningRateScheduler(schedule, verbose=1)
        )
    elif lr_type == "cosine_decay":
        decay_epochs = float(
            lr_config.get("decay_epochs", config.get("training", {}).get("epochs", 50))
        )
        alpha = lr_config.get("alpha", 0.0)  # minimum LR fraction

        def schedule(epoch, _current_lr):
            progress = min(float(epoch) / max(decay_epochs, 1.0), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return initial_lr * (alpha + (1.0 - alpha) * cosine)

        callbacks.append(
            keras.callbacks.LearningRateScheduler(schedule, verbose=1)
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
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        append = csv_config.get("append", False) if csv_append is None else csv_append
        callbacks.append(keras.callbacks.CSVLogger(csv_path, append=append))

    return callbacks


def preprocess_image(image_path, image_size=(224, 224), preprocess_fn=None):
    """
    Preprocess single image for inference

    Args:
        image_path: Path to image file
        image_size: Target size

    Returns:
        Preprocessed image tensor
    """
    img = tf.keras.utils.load_img(image_path, target_size=image_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0).astype("float32")
    if preprocess_fn is not None:
        img_array = preprocess_fn(img_array)
    else:
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
