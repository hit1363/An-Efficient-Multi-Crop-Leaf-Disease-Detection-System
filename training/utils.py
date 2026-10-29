"""
Utility Functions for Training Pipeline
"""

import os
import sys
import logging
import shutil
import tempfile
import time
import csv
import hashlib
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _dataset_class_names(directory, class_names=None):
    """Return source folder names in canonical order when aliases are used."""
    folders = sorted(
        name for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    )
    if class_names is not None:
        expected = list(class_names)
    else:
        expected = None

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from dataset.taxonomy import CANONICAL_CLASSES, canonicalize_label
    except ImportError:
        if expected is not None and set(folders) != set(expected):
            raise ValueError(
                f"Dataset class folders do not match the configured classes; "
                f"missing={sorted(set(expected) - set(folders))}, "
                f"extra={sorted(set(folders) - set(expected))}"
            )
        return expected

    canonical_expected = list(CANONICAL_CLASSES)
    use_canonical_taxonomy = expected is None or set(expected) == set(canonical_expected)
    if not use_canonical_taxonomy:
        if set(folders) != set(expected):
            raise ValueError(
                f"Dataset class folders do not match the configured classes; "
                f"missing={sorted(set(expected) - set(folders))}, "
                f"extra={sorted(set(folders) - set(expected))}"
            )
        return expected

    if expected is None:
        recognized = False
        for source_name in folders:
            try:
                canonicalize_label(source_name)
                recognized = True
                break
            except ValueError:
                continue
        if not recognized:
            return None

    resolved = {}
    for source_name in folders:
        try:
            canonical_name = canonicalize_label(source_name)
        except ValueError as exc:
            raise ValueError(
                f"Unknown dataset class folder {source_name!r} in {directory}: {exc}"
            ) from exc
        if canonical_name in resolved:
            raise ValueError(
                f"Duplicate dataset class folders in {directory} resolve to "
                f"{canonical_name!r}: {resolved[canonical_name]!r} and {source_name!r}"
            )
        resolved[canonical_name] = source_name

    missing = sorted(set(canonical_expected) - set(resolved))
    extra = sorted(set(resolved) - set(canonical_expected))
    if missing or extra:
        raise ValueError(
            f"Dataset class folders do not match the canonical taxonomy in {directory}; "
            f"missing={missing}, extra={extra}"
        )
    return [resolved[name] for name in canonical_expected]


def get_training_class_counts(train_dir, class_names=None):
    """Return canonical training-image counts without validating every split.

    A harmonized benchmark's manifest is authoritative and substantially faster
    than walking hosted image directories.  For ordinary directory datasets,
    count supported files only in the training split.  The returned insertion
    order is the model's categorical-label order.
    """
    source_class_names = _dataset_class_names(train_dir, class_names)
    if source_class_names is None:
        canonical_names = sorted(
            name for name in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, name))
        )
        source_class_names = canonical_names
    elif class_names is not None:
        canonical_names = list(class_names)
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        try:
            from dataset.taxonomy import CANONICAL_CLASSES

            canonical_names = list(CANONICAL_CLASSES)
        except ImportError:
            canonical_names = list(source_class_names)

    manifest_path = os.path.join(os.path.dirname(os.path.abspath(train_dir)), "split_manifest.csv")
    if os.path.isfile(manifest_path):
        manifest_counts = {name: 0 for name in canonical_names}
        with open(manifest_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("split") == "train":
                    label = row.get("canonical_label")
                    if label not in manifest_counts:
                        raise ValueError(
                            f"Training manifest has an unknown canonical label: {label!r}"
                        )
                    manifest_counts[label] += 1
        return manifest_counts

    counts = {}
    for canonical_name, source_name in zip(canonical_names, source_class_names):
        class_dir = os.path.join(train_dir, source_name)
        counts[canonical_name] = sum(
            os.path.isfile(os.path.join(class_dir, filename))
            and os.path.splitext(filename)[1].lower() in _IMAGE_EXTENSIONS
            for filename in os.listdir(class_dir)
        )
    return counts


def get_minority_class_indices(class_counts, max_train_samples=500):
    """Return label indices with positive counts below the configured cutoff."""
    if max_train_samples is None:
        return []
    cutoff = int(max_train_samples)
    if cutoff < 1:
        raise ValueError("augmentation.class_aware.max_train_samples must be >= 1")
    values = class_counts.values() if isinstance(class_counts, dict) else class_counts
    return [index for index, count in enumerate(values) if 0 < int(count) < cutoff]


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


def _normalize_cache_dtype(cache_dtype):
    """Normalize supported floating-point cache storage types."""
    try:
        dtype = tf.as_dtype(cache_dtype)
    except TypeError as exc:
        raise ValueError(
            "dataset.cache_dtype must be either 'float16' or 'float32'"
        ) from exc
    if dtype not in (tf.float16, tf.float32):
        raise ValueError("dataset.cache_dtype must be either 'float16' or 'float32'")
    return dtype.name


def _estimate_dataset_cache_gb(
    train_count,
    val_count,
    image_size,
    cache_dtype="float32",
    label_count=0,
    label_dtype="float32",
    headroom=1.25,
):
    """Estimate cache footprint, including image and categorical-label tensors."""
    image_dtype = tf.as_dtype(_normalize_cache_dtype(cache_dtype))
    label_dtype = tf.as_dtype(label_dtype)
    height, width = image_size[:2]
    per_example_bytes = (
        int(height) * int(width) * 3 * int(image_dtype.size)
        + int(label_count) * int(label_dtype.size)
    )
    return (
        (int(train_count) + int(val_count))
        * per_example_bytes
        / (1024**3)
        * float(headroom)
    )


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
    if arch in {
        "mobilenet", "mobilenet_v1", "mobilenetv1", "mobilenet_v2", "mobilenetv2",
        "mobilenet_v3_small", "mobilenetv3_small", "mobilenetv3small",
        "mobilenetv3_large", "mobilenet_v3_large", "mobilenetv3large",
        "nasnet_mobile", "nasnetmobile",
    }:
        if arch in {
            "mobilenet_v3_small", "mobilenetv3_small", "mobilenetv3small",
        }:
            # The MobileNetV3-Small config disables Keras' built-in preprocessing.
            return lambda x: tf.cast(x, tf.float32) / 127.5 - 1.0
        if arch in {
            "mobilenetv3_large", "mobilenet_v3_large", "mobilenetv3large",
        }:
            # Other MobileNetV3 configs may use Keras' built-in rescaling layer.
            return None
        if arch in {"nasnet_mobile", "nasnetmobile"}:
            return tf.keras.applications.nasnet.preprocess_input
        if arch in {"mobilenet", "mobilenet_v1", "mobilenetv1"}:
            return tf.keras.applications.mobilenet.preprocess_input
        return tf.keras.applications.mobilenet_v2.preprocess_input
    if arch == "mobilenetv2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
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
    cache_dtype="float32",
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

    # Cache creation is before augmentation and preprocessing. Account for the
    # configured image storage dtype, categorical labels, and tf.data overhead.
    required_gb = 0.0
    if train_count >= 0 and val_count >= 0:
        label_spec = train_ds.element_spec[1]
        label_count = label_spec.shape[-1] if label_spec.shape.rank else 1
        required_gb = _estimate_dataset_cache_gb(
            train_count,
            val_count,
            image_size,
            cache_dtype=cache_dtype,
            label_count=label_count or 1,
            label_dtype=label_spec.dtype,
        )

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
    minority_augmentation=None,
    minority_class_indices=None,
    shuffle_buffer=1000,
    cache_mode="none",
    cache_memory_limit_gb=8.0,
    cache_dtype="float32",
    deterministic=False,
    class_names=None,
):
    """
    Load training and validation datasets

    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        batch_size: Batch size for training
        image_size: Target image size (height, width)
        preprocess_fn: Deterministic preprocessing applied to both splits
        augmentation: Baseline random augmentation applied to training only.
        minority_augmentation: Replacement augmentation applied only to examples
            whose labels appear in minority_class_indices.
        minority_class_indices: Categorical-label indices eligible for the
            class-aware augmentation policy.
        shuffle_buffer: Shuffle buffer size for the training split. Shuffling is
            applied AFTER caching so the order reshuffles every epoch instead of
            being frozen in the cache.
        cache_mode: Dataset cache strategy. Use "none" to disable caching,
            "memory" to keep cached batches in RAM, "disk" to cache to a
            writable filesystem path, or "auto" to select RAM only when the
            estimated footprint is safe and otherwise use disk/no cache.
        cache_dtype: Floating-point dtype used to store cached images. Float16
            halves cache I/O and disk use; yielded images are restored to float32.

    Returns:
        train_ds, val_ds, class_names
    """
    # Batch at the source. The previous unbatched -> shuffle -> batch path
    # added substantial tf.data overhead for large hosted datasets.
    source_class_names = _dataset_class_names(train_dir, class_names)
    val_source_class_names = _dataset_class_names(val_dir, class_names)
    canonical_class_names = None
    if source_class_names is not None:
        if val_source_class_names != source_class_names:
            raise ValueError(
                "Train and validation class folders resolve to different canonical orders"
            )
        class_names = list(class_names) if class_names is not None else None
        if class_names is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from dataset.taxonomy import CANONICAL_CLASSES
            class_names = list(CANONICAL_CLASSES)
        canonical_class_names = class_names
    train_ds = keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
        seed=42,
        class_names=source_class_names,
    )

    val_ds = keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
        class_names=val_source_class_names,
    )

    class_names = canonical_class_names or train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    options = tf.data.Options()
    options.experimental_deterministic = bool(deterministic)

    cache_dtype = _normalize_cache_dtype(cache_dtype)
    cache_mode = _resolve_cache_mode(
        cache_mode,
        train_ds,
        val_ds,
        image_size,
        batch_size,
        memory_limit_gb=cache_memory_limit_gb,
        cache_dtype=cache_dtype,
    )
    logger = logging.getLogger(__name__)

    # Cache decoded/resized batches before augmentation so augmentation remains
    # stochastic on every epoch while image decoding happens only once. Cast
    # before caching and restore float32 immediately after the cache boundary.
    if cache_mode == "disk":
        height, width = image_size[:2]
        cache_key = f"v3_{height}x{width}_b{int(batch_size)}_{cache_dtype}"
        if cache_dtype != "float32":
            train_ds = train_ds.map(
                lambda x, y: (tf.cast(x, cache_dtype), y),
                num_parallel_calls=AUTOTUNE,
            )
            val_ds = val_ds.map(
                lambda x, y: (tf.cast(x, cache_dtype), y),
                num_parallel_calls=AUTOTUNE,
            )
        train_cache = _dataset_cache_path("train_cache", cache_key)
        val_cache = _dataset_cache_path("val_cache", cache_key)
        if train_cache and val_cache:
            train_ds = train_ds.cache(train_cache)
            val_ds = val_ds.cache(val_cache)
        else:
            logger.warning("Disk cache is unavailable; continuing without caching.")
            cache_mode = "none"
            if cache_dtype != "float32":
                train_ds = train_ds.map(
                    lambda x, y: (tf.cast(x, tf.float32), y),
                    num_parallel_calls=AUTOTUNE,
                )
                val_ds = val_ds.map(
                    lambda x, y: (tf.cast(x, tf.float32), y),
                    num_parallel_calls=AUTOTUNE,
                )
    elif cache_mode == "memory":
        if cache_dtype != "float32":
            train_ds = train_ds.map(
                lambda x, y: (tf.cast(x, cache_dtype), y),
                num_parallel_calls=AUTOTUNE,
            )
            val_ds = val_ds.map(
                lambda x, y: (tf.cast(x, cache_dtype), y),
                num_parallel_calls=AUTOTUNE,
            )
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    if cache_mode in {"disk", "memory"} and cache_dtype != "float32":
        train_ds = train_ds.map(
            lambda x, y: (tf.cast(x, tf.float32), y),
            num_parallel_calls=AUTOTUNE,
        )
        val_ds = val_ds.map(
            lambda x, y: (tf.cast(x, tf.float32), y),
            num_parallel_calls=AUTOTUNE,
        )

    # Shuffle batches after caching so cached datasets still reshuffle each
    # epoch. The source dataset is already batched, avoiding the expensive
    # individual-example shuffle and re-batch path.
    if shuffle_buffer and shuffle_buffer > 0:
        shuffle_batches = max(1, int(np.ceil(shuffle_buffer / batch_size)))
        train_ds = train_ds.shuffle(
            buffer_size=shuffle_batches, seed=42, reshuffle_each_iteration=True
        )

    # Augmentation only on training data. It runs after the cache each epoch.
    # The stronger minority branch is evaluated only for the minority examples
    # in a batch, avoiding oversampling and duplicate epoch exposure.
    if augmentation is not None:
        minority_indices = tuple(int(index) for index in (minority_class_indices or ()))

        if minority_augmentation is not None and minority_indices:
            minority_ids = tf.constant(minority_indices, dtype=tf.int32)

            def _apply_class_aware_augmentation(x, y):
                images = tf.cast(x, tf.float32)
                # Fast path: apply both augmentations, blend with mask.
                # Avoids expensive tf.gather / tf.tensor_scatter_nd_update.
                baseline_images = augmentation(images, training=True)
                label_indices = tf.argmax(y, axis=-1, output_type=tf.int32)
                is_minority = tf.reduce_any(
                    tf.equal(tf.expand_dims(label_indices, axis=-1), minority_ids),
                    axis=-1,
                )
                # Apply minority augmentation to all images (lightweight layers)
                minority_images = minority_augmentation(images, training=True)
                # Blend: where minority, use stronger augmentation, else baseline
                mask = tf.reshape(
                    tf.cast(is_minority, baseline_images.dtype), [-1, 1, 1, 1]
                )
                result = baseline_images * (1.0 - mask) + minority_images * mask
                return result, y

            train_ds = train_ds.map(
                _apply_class_aware_augmentation, num_parallel_calls=AUTOTUNE
            )
        else:
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
        "cache_mode=%s, cache_dtype=%s, shuffle_batch_buffer=%d",
        batch_size,
        train_ds.cardinality(),
        val_ds.cardinality(),
        cache_mode,
        cache_dtype if cache_mode in {"disk", "memory"} else "float32 (cache inactive)",
        max(1, int(np.ceil(shuffle_buffer / batch_size))) if shuffle_buffer else 0,
    )

    return train_ds, val_ds, class_names


def load_benchmark_class_names(data_dir):
    """Load class order from the manifest or the repository's fixed taxonomy."""
    manifest_path = os.path.join(data_dir, "class_names.json")
    if not os.path.isfile(manifest_path):
        # Kaggle input mounts are read-only. Use the repository taxonomy in memory
        # when a mounted 72-class dataset does not ship its own manifest.
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        try:
            from dataset.taxonomy import CANONICAL_CLASSES
        except ImportError as exc:
            raise FileNotFoundError(
                f"Canonical class manifest is missing: {manifest_path}, and the "
                "repository taxonomy could not be loaded."
            ) from exc
        logging.getLogger(__name__).warning(
            "Canonical class manifest is missing at %s; using the repository taxonomy. "
            "Dataset splits must contain exactly the canonical 72 classes.",
            manifest_path,
        )
        return list(CANONICAL_CLASSES)

    with open(manifest_path, "r", encoding="utf-8") as handle:
        class_names = json.load(handle)
    if (
        not isinstance(class_names, list)
        or not class_names
        or any(not isinstance(name, str) for name in class_names)
        or len(set(class_names)) != len(class_names)
    ):
        raise ValueError(f"Invalid canonical class manifest: {manifest_path}")
    return class_names


def validate_dataset_splits(train_dir, val_dir, test_dir=None, class_names=None):
    """Validate class parity, image support, and source-hash split isolation."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    split_paths = {"train": train_dir, "val": val_dir}
    if test_dir:
        split_paths["test"] = test_dir
    # A benchmark manifest is authoritative for its split assignment and
    # source hashes. Reading it once is substantially faster on hosted
    # filesystems than listing every image directory before reading it anyway.
    data_root = os.path.dirname(os.path.abspath(train_dir))
    manifest_path = os.path.join(data_root, "split_manifest.csv")
    manifest_counts = None
    hash_splits = {}
    if os.path.isfile(manifest_path):
        logging.getLogger(__name__).info(
            "Validating split counts from manifest: %s", manifest_path
        )
        manifest_counts = {split: {} for split in split_paths}
        with open(manifest_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                digest = row.get("source_sha256")
                split = row.get("split")
                label = row.get("canonical_label")
                if digest and split:
                    hash_splits.setdefault(digest, set()).add(split)
                if split in manifest_counts:
                    if not label:
                        raise ValueError(
                            f"Split manifest has no canonical label for {split!r}"
                        )
                    manifest_counts[split][label] = (
                        manifest_counts[split].get(label, 0) + 1
                    )
    counts = {}
    for split_name, split_path in split_paths.items():
        if not os.path.isdir(split_path):
            raise FileNotFoundError(f"Dataset {split_name} directory does not exist: {split_path}")
        source_folders = sorted(
            name for name in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, name))
        )
        ordered_folders = _dataset_class_names(split_path, class_names)
        if ordered_folders is None:
            folders = source_folders
            canonical_to_source = {name: name for name in folders}
        else:
            folders = list(class_names) if class_names is not None else None
            if folders is None:
                repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if repo_root not in sys.path:
                    sys.path.insert(0, repo_root)
                from dataset.taxonomy import CANONICAL_CLASSES
                folders = list(CANONICAL_CLASSES)
            canonical_to_source = dict(zip(folders, ordered_folders))
        expected = sorted(class_names) if class_names is not None else None
        if expected is not None and sorted(folders) != expected:
            raise ValueError(
                f"Dataset {split_name} class folders do not match the canonical manifest; "
                f"missing={sorted(set(expected) - set(folders))}, "
                f"extra={sorted(set(folders) - set(expected))}"
            )
        if manifest_counts is not None:
            unknown_manifest_classes = set(manifest_counts[split_name]) - set(folders)
            if unknown_manifest_classes:
                raise ValueError(
                    f"Split manifest has unknown classes for {split_name}: "
                    f"{sorted(unknown_manifest_classes)}"
                )
            split_counts = {
                class_name: manifest_counts[split_name].get(class_name, 0)
                for class_name in folders
            }
        else:
            split_counts = {}
            for class_name in folders:
                class_dir = os.path.join(split_path, canonical_to_source[class_name])
                images = [
                    os.path.join(class_dir, filename)
                    for filename in os.listdir(class_dir)
                    if os.path.splitext(filename)[1].lower() in valid_extensions
                ]
                split_counts[class_name] = len(images)
        if split_name == "train":
            empty_training_classes = [
                class_name for class_name, count in split_counts.items() if count == 0
            ]
            if empty_training_classes:
                raise ValueError(
                    f"Training class '{empty_training_classes[0]}' has no supported images"
                )
        counts[split_name] = {"total": sum(split_counts.values()), "classes": split_counts}
        if counts[split_name]["total"] == 0:
            raise ValueError(f"Dataset {split_name} split has no supported images")

    all_classes = [set(value["classes"]) for value in counts.values()]
    if any(classes != all_classes[0] for classes in all_classes[1:]):
        raise ValueError("Train, validation, and test class folders differ")

    # Without a manifest, hash every image to preserve split-isolation checks.
    if manifest_counts is None:
        logging.getLogger(__name__).warning(
            "Split manifest is unavailable; validating by scanning image files."
        )
        for split_name, split_path in split_paths.items():
            ordered_source_folders = _dataset_class_names(split_path, class_names)
            if ordered_source_folders is None:
                source_by_class = {
                    name: name for name in counts[split_name]["classes"]
                }
            else:
                repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if repo_root not in sys.path:
                    sys.path.insert(0, repo_root)
                from dataset.taxonomy import CANONICAL_CLASSES

                canonical_names = list(CANONICAL_CLASSES)
                if class_names is not None and set(class_names) != set(CANONICAL_CLASSES):
                    canonical_names = list(class_names)
                source_by_class = dict(zip(canonical_names, ordered_source_folders))
            for class_name in sorted(counts[split_name]["classes"]):
                class_dir = os.path.join(split_path, source_by_class[class_name])
                for filename in sorted(os.listdir(class_dir)):
                    path = os.path.join(class_dir, filename)
                    if os.path.isfile(path) and os.path.splitext(filename)[1].lower() in valid_extensions:
                        digest = hashlib.sha256()
                        with open(path, "rb") as image_file:
                            for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                                digest.update(chunk)
                        hash_splits.setdefault(digest.hexdigest(), set()).add(split_name)
    leaking_hashes = [digest for digest, splits in hash_splits.items() if len(splits) > 1]
    if leaking_hashes:
        raise ValueError(
            f"Found {len(leaking_hashes)} exact duplicate image hashes across dataset splits"
        )
    return counts


def _create_augmentation_layer(aug_config, name):
    """Build one stochastic image augmentation layer from a flat config."""
    zoom_range = aug_config.get("zoom_range", [0.8, 1.2])
    if isinstance(zoom_range, (list, tuple)) and len(zoom_range) == 2:
        zoom_factor = (zoom_range[0] - 1.0, zoom_range[1] - 1.0)
    else:
        zoom_factor = zoom_range

    flip_mode = {
        (True, True): "horizontal_and_vertical",
        (True, False): "horizontal",
        (False, True): "vertical",
        (False, False): "none",
    }[(bool(aug_config.get("horizontal_flip", True)),
       bool(aug_config.get("vertical_flip", False)))]

    layers = []
    if flip_mode != "none":
        layers.append(keras.layers.RandomFlip(flip_mode))
    rotation_range = float(aug_config.get("rotation_range", 15))
    if rotation_range > 0:
        layers.append(keras.layers.RandomRotation(rotation_range / 360))
    if zoom_factor != (0.0, 0.0):
        layers.append(keras.layers.RandomZoom(zoom_factor))
    contrast_range = aug_config.get("contrast_range", 0.1)
    if contrast_range and contrast_range > 0:
        layers.append(keras.layers.RandomContrast(contrast_range))

    if not layers:
        layers.append(keras.layers.Identity())
    augmentation = keras.Sequential(layers, name=name)

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


def create_augmentation_layer(config):
    """
    Create data augmentation layer

    Args:
        config: Augmentation configuration

    Returns:
        Keras Sequential model with augmentation layers
    """
    return _create_augmentation_layer(
        config.get("augmentation", {}), name="training_augmentation"
    )


def create_class_aware_augmentation_layers(config):
    """Return baseline and optional minority augmentation layers."""
    base_augmentation = create_augmentation_layer(config)
    class_aware = config.get("augmentation", {}).get("class_aware", {})
    if not class_aware.get("enabled", False):
        return base_augmentation, None

    minority_config = {
        key: value
        for key, value in class_aware.items()
        if key not in {"enabled", "max_train_samples"}
    }
    if not minority_config:
        raise ValueError("augmentation.class_aware needs at least one transform setting")
    return base_augmentation, _create_augmentation_layer(
        minority_config, name="minority_training_augmentation"
    )


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


def compute_class_weights_from_counts(class_counts, max_weight=None, normalize=True):
    """Compute class weights from counts in deterministic class order.

    Args:
        class_counts: A sequence of per-class image counts. Its position is the
            class index used by Keras labels.
        max_weight: Optional upper bound for any class weight.
        normalize: Preserve a sample-weighted mean of one when clipping.

    Returns:
        Dictionary mapping class indices to weights.
    """
    class_counts = {index: int(count) for index, count in enumerate(class_counts)}
    if any(count < 0 for count in class_counts.values()):
        raise ValueError("class counts must be non-negative")

    total = sum(class_counts.values())
    nonzero = {i: count for i, count in class_counts.items() if count > 0}
    if not nonzero:
        return {}

    class_weights = {}
    for i, count in class_counts.items():
        if count > 0:
            class_weights[i] = total / (len(nonzero) * count)
        else:
            class_weights[i] = 0.0

    if max_weight is None:
        return class_weights

    max_weight = float(max_weight)
    if max_weight <= 0:
        raise ValueError("class_weights.max_weight must be greater than zero")
    if not normalize:
        return {
            index: min(weight, max_weight)
            for index, weight in class_weights.items()
        }
    if max_weight < 1.0:
        raise ValueError(
            "class_weights.max_weight must be >= 1.0 when normalize is enabled"
        )

    # Raw inverse-frequency weights have a sample-weighted mean of one. A
    # direct clip lowers that mean, so solve for a scale that restores it
    # without allowing any individual class weight to exceed the configured
    # ceiling.
    def weighted_mean(scale):
        return sum(
            class_counts[index] * min(weight * scale, max_weight)
            for index, weight in class_weights.items()
        ) / total

    lower, upper = 0.0, 1.0
    while weighted_mean(upper) < 1.0:
        upper *= 2.0

    for _ in range(60):
        midpoint = (lower + upper) / 2.0
        if weighted_mean(midpoint) < 1.0:
            lower = midpoint
        else:
            upper = midpoint

    return {
        index: min(weight * upper, max_weight)
        for index, weight in class_weights.items()
    }


def compute_class_weights(data_dir, max_weight=None, normalize=True):
    """Compute class weights for an imbalanced directory dataset.

    Args:
        data_dir: Directory containing class subdirectories.
        max_weight: Optional upper bound for any class weight. When supplied
            with ``normalize=True``, weights are scaled before clipping so the
            sample-weighted mean remains one.
        normalize: Preserve a sample-weighted mean of one when clipping.

    Returns:
        Dictionary of class weights.
    """
    class_names = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )
    class_counts = []
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        count = len(
            [
                f
                for f in os.listdir(class_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        class_counts.append(count)

    return compute_class_weights_from_counts(
        class_counts, max_weight=max_weight, normalize=normalize
    )


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


def get_checkpoint_path(config, checkpoint_name="best_model"):
    """Return the canonical checkpoint path and its serialization mode."""
    checkpoint_config = config.get("callbacks", {}).get("checkpoint", {})
    architecture = config.get("model", {}).get("architecture", "").lower()
    is_lite0_arch = architecture in ["efficientnet", "efficientnet_lite0"]

    save_weights_only = checkpoint_config.get("save_weights_only", False)
    if is_lite0_arch and not save_weights_only:
        # TF Hub-backed models are more reliable with weights-only checkpoints.
        save_weights_only = True

    extension = ".weights.h5" if save_weights_only else ".h5"
    checkpoint_dir = checkpoint_config.get("directory") or os.path.join(
        config["export"]["save_dir"],
        config["model"]["architecture"],
        "checkpoints",
    )
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name + extension)
    return checkpoint_path, save_weights_only

def save_model_checkpoint(model, config, checkpoint_name="best_model"):
    """Save the cross-phase winner to the configured canonical checkpoint."""
    checkpoint_config = config.get("callbacks", {}).get("checkpoint", {})
    if not checkpoint_config.get("enabled", True):
        return None

    checkpoint_path, save_weights_only = get_checkpoint_path(config, checkpoint_name)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    if save_weights_only:
        model.save_weights(checkpoint_path)
    else:
        model.save(checkpoint_path)
    return checkpoint_path


def get_epoch_log_mode(config):
    """Return and validate the configured Keras training output mode."""
    mode = str(
        config.get("training", {}).get("log_mode", "summary")
    ).strip().lower()
    if mode not in {"summary", "progress"}:
        raise ValueError(
            "training.log_mode must be either 'summary' or 'progress'; "
            f"got {mode!r}"
        )
    return mode


def setup_callbacks(
    config,
    csv_append=None,
    checkpoint_name="best_model",
    include_throughput=True,
):
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
    callback_verbose = 1 if get_epoch_log_mode(config) == "progress" else 0

    if performance_config.get("terminate_on_nan", True):
        callbacks.append(keras.callbacks.TerminateOnNaN())

    if (
        include_throughput
        and performance_config.get("log_throughput", True)
        and get_epoch_log_mode(config) == "progress"
    ):
        callbacks.append(
            ThroughputCallback(
                batch_size=config.get("dataset", {}).get("batch_size", 32),
                cache_mode=config.get("dataset", {}).get("cache_mode", "none"),
            )
        )

    # Model Checkpoint
    if callback_config.get("checkpoint", {}).get("enabled", True):
        checkpoint_config = callback_config["checkpoint"]
        checkpoint_path, save_weights_only = get_checkpoint_path(
            config, checkpoint_name=checkpoint_name
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
                verbose=callback_verbose,
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
                verbose=callback_verbose,
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
                verbose=callback_verbose,
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
            keras.callbacks.LearningRateScheduler(schedule, verbose=callback_verbose)
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
            keras.callbacks.LearningRateScheduler(schedule, verbose=callback_verbose)
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
    # class_weights = compute_class_weights('../dataset/processed_harmonized/train')
    # print("Class weights:", class_weights)
