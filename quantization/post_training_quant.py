"""
Post-Training Quantization Script
Converts trained TensorFlow model to TensorFlow Lite with INT8 quantization
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import sys
import argparse
import copy
import random
import re
import traceback
from pathlib import Path

import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import shared utilities from the training package, with sys.path fallback.
try:
    from training.utils import (
        concrete_model_image_size,
        get_preprocess_fn,
        saved_model_image_size,
    )
except ImportError:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from training.utils import (
            concrete_model_image_size,
            get_preprocess_fn,
            saved_model_image_size,
        )
    except ImportError as exc:  # pragma: no cover - hard configuration error
        raise ImportError(
            "Could not import training.utils.get_preprocess_fn. Run this script "
            f"from the repository root (tried {repo_root!r}). A local fallback "
            "copy is deliberately not kept: a drifted copy silently preprocesses "
            "calibration data differently from training and destroys INT8 accuracy."
        ) from exc


def infer_architecture_from_path(model_path):
    lower = model_path.lower()
    if "efficientnet_v2_b0" in lower or "efficientnetv2_b0" in lower:
        return "efficientnet_v2_b0"
    if "mobilenet_v3_large" in lower or "mobilenetv3_large" in lower or "mobilenetv3large" in lower:
        return "mobilenet_v3_large"
    if "mobilenet_v1" in lower or "mobilenetv1" in lower:
        return "mobilenet_v1"
    if "nasnet_mobile" in lower or "nasnetmobile" in lower:
        return "nasnet_mobile"
    if "shufflenetv2_05" in lower or "shufflenet_v2_05" in lower:
        return "shufflenetv2_05"
    if "mobilenetv3_small" in lower or "mobilenet_v3_small" in lower:
        return "mobilenetv3_small"
    if "efficientnet_b0" in lower:
        return "efficientnet_b0"
    if "mobilenetv2" in lower or "mobilenet_v2" in lower:
        return "mobilenetv2"
    if "efficientnet" in lower:
        return "efficientnet_lite0"
    if "mobilenet" in lower:
        return "mobilenetv2"
    return "mobilenetv2"


def _is_saved_model_dir(model_path):
    return os.path.isdir(model_path) and (
        os.path.exists(os.path.join(model_path, "saved_model.pb"))
        or os.path.exists(os.path.join(model_path, "saved_model.pbtxt"))
    )


def _is_weights_only(model_path):
    return model_path.lower().endswith(".weights.h5")


def _load_config(config_path):
    if not config_path:
        return None
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _get_model_factory():
    try:
        from training.model import get_model

        return get_model
    except Exception:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        try:
            from training.model import get_model

            return get_model
        except Exception as exc:
            raise ImportError(
                "Unable to import training.model.get_model. "
                "Run this script from the repository root or provide a valid PYTHONPATH."
            ) from exc


# Architectures whose heads are built from an explicit head_units stack.
_HEAD_UNITS_ARCHITECTURES = {"mobilenetv2", "efficientnet_b0"}


_UNSET = object()


def _build_model_from_config(
    config, architecture_override=None, weights_override=_UNSET
):
    if not config or "model" not in config:
        raise ValueError("Model config is missing or invalid.")

    model_cfg = config.get("model", {})
    architecture = architecture_override or model_cfg.get("architecture", "mobilenetv2")
    input_shape = tuple(model_cfg.get("input_shape", [224, 224, 3]))
    num_classes = int(model_cfg.get("num_classes", 2))
    dropout_rate = float(model_cfg.get("dropout_rate", 0.5))
    if weights_override is _UNSET:
        weights = model_cfg.get("weights", "imagenet")
    else:
        # Explicit value (including None) - a checkpoint rebuild must not
        # re-download ImageNet weights it is about to overwrite anyway.
        weights = weights_override

    model_kwargs = {
        "architecture": architecture,
        "input_shape": input_shape,
        "num_classes": num_classes,
        "dropout_rate": dropout_rate,
        "weights": weights,
        "hub_url": model_cfg.get("hub_url"),
        "hub_cache_dir": model_cfg.get("hub_cache_dir"),
        "hub_download_retries": model_cfg.get("hub_download_retries", 1),
        "hub_download_delay_sec": model_cfg.get("hub_download_delay_sec", 5),
    }
    # train.py passes these for the two architectures that own a configurable
    # dense head; omitting them rebuilds a differently shaped model whose
    # checkpoint cannot be loaded.
    if str(architecture).lower() in _HEAD_UNITS_ARCHITECTURES:
        model_kwargs["head_units"] = tuple(model_cfg.get("head_units", [256, 128]))
        model_kwargs["l2_regularization"] = model_cfg.get("l2_regularization", 0.0)
    if "classifier_units" in model_cfg:
        model_kwargs["classifier_units"] = model_cfg["classifier_units"]
    if "include_preprocessing" in model_cfg:
        model_kwargs["include_preprocessing"] = model_cfg["include_preprocessing"]

    get_model = _get_model_factory()
    model, _ = get_model(**model_kwargs)

    return model


def _load_keras_model(model_path, config_path=None, architecture=None):
    if _is_weights_only(model_path):
        if not config_path:
            raise ValueError(
                "Weights-only checkpoint provided. Please pass --config to rebuild the model."
            )
        config = _load_config(config_path)
        model = _build_model_from_config(config, architecture_override=architecture)
        model.load_weights(model_path)
        return model

    return keras.models.load_model(model_path)


# Run-scoped hosted artifacts are laid out as
#   <save_dir>/<arch>/saved_model_<ts>/
#   <save_dir>/<arch>/runs/<ts>/checkpoints/best_model.weights.h5
#   <save_dir>/<arch>/runs/<ts>/resolved_config.yaml
_RUN_TIMESTAMP_RE = re.compile(r"saved_model_(\d{8}_\d{6})")


def _run_timestamp_from_saved_model(model_path):
    """Return the training run timestamp encoded in a SavedModel dir name."""
    match = _RUN_TIMESTAMP_RE.search(os.path.basename(os.path.normpath(model_path)))
    return match.group(1) if match else None


def _find_run_checkpoint(model_path):
    """Locate the checkpoint and run metadata that produced a SavedModel export.

    Returns a ``(checkpoint_path, run_config_path)`` tuple; either element may
    be ``None`` when the export predates run-scoped checkpointing.
    """
    arch_dir = os.path.dirname(os.path.abspath(os.path.normpath(model_path)))
    run_timestamp = _run_timestamp_from_saved_model(model_path)

    run_dir = None
    if run_timestamp:
        candidate = os.path.join(arch_dir, "runs", run_timestamp)
        if os.path.isdir(candidate):
            run_dir = candidate

    checkpoint_path = None
    search_dirs = []
    if run_dir:
        search_dirs.append(os.path.join(run_dir, "checkpoints"))
    search_dirs.append(os.path.join(arch_dir, "checkpoints"))
    for directory in search_dirs:
        for filename in ("best_model.weights.h5", "best_model.h5"):
            candidate = os.path.join(directory, filename)
            if os.path.isfile(candidate):
                checkpoint_path = candidate
                break
        if checkpoint_path:
            break

    run_config_path = None
    if run_dir:
        for filename in ("resolved_config.yaml", "config.yaml"):
            candidate = os.path.join(run_dir, filename)
            if os.path.isfile(candidate):
                run_config_path = candidate
                break

    return checkpoint_path, run_config_path


def _saved_model_output_classes(model_path):
    """Infer the classifier width from a SavedModel serving signature."""
    try:
        loaded = tf.saved_model.load(model_path)
        signatures = getattr(loaded, "signatures", None) or {}
        signature = signatures.get("serving_default")
        if signature is None and signatures:
            signature = next(iter(signatures.values()))
        outputs = getattr(signature, "structured_outputs", None) if signature else None
        if not outputs:
            return None
        for tensor_spec in outputs.values():
            shape = getattr(tensor_spec, "shape", None)
            if shape is None or shape.rank is None or shape.rank < 2:
                continue
            last_dim = shape[-1]
            if isinstance(last_dim, int) and last_dim > 0:
                return int(last_dim)
    except Exception:
        return None
    return None


def _rebuild_float32_model_from_checkpoint(
    model_path, config_path=None, architecture=None
):
    """Rebuild the trained model as float32 from its run-scoped checkpoint.

    A SavedModel exported straight from a mixed_float16 training model keeps
    float16 operations in its graph, which the built-in TFLite converter cannot
    lower; training/export.py works around this by converting a float32 clone of
    the same weights. This mirrors that proven path for post-hoc quantization,
    where only the SavedModel and the run checkpoint are available.

    Returns the rebuilt model, or ``None`` when no checkpoint exists (the caller
    then keeps the plain SavedModel conversion path).
    """
    checkpoint_path, run_config_path = _find_run_checkpoint(model_path)
    if not checkpoint_path:
        return None

    # The run config records what training actually applied (runtime profiles
    # rewrite the resolution, e.g. 192x192 on Kaggle), so it wins over --config.
    config = None
    for candidate in (run_config_path, config_path):
        if not candidate:
            continue
        try:
            loaded = _load_config(candidate)
        except Exception as exc:  # pragma: no cover - unreadable config
            print(f"Warning: could not read config {candidate}: {exc}")
            continue
        if loaded and loaded.get("model"):
            config = copy.deepcopy(loaded)
            break
    if config is None:
        print(
            "Warning: found a run checkpoint but no usable config; "
            "cannot rebuild the model for conversion.",
        )
        return None

    model_cfg = config.setdefault("model", {})
    if architecture:
        model_cfg["architecture"] = architecture
    image_size = saved_model_image_size(model_path)
    if image_size:
        existing = tuple(model_cfg.get("input_shape", ()))
        matches = len(existing) >= 2 and (
            existing[0], existing[1]
        ) == tuple(image_size)
        if not matches:
            model_cfg["input_shape"] = [int(image_size[0]), int(image_size[1]), 3]
    if not model_cfg.get("num_classes"):
        inferred = _saved_model_output_classes(model_path)
        if inferred:
            model_cfg["num_classes"] = inferred
    if not model_cfg.get("num_classes"):
        print(
            "Warning: could not determine num_classes; "
            "falling back to SavedModel conversion.",
        )
        return None

    print(
        f"Rebuilding float32 model from checkpoint: {checkpoint_path}"
    )
    print(
        f"Rebuild architecture={model_cfg.get('architecture')} "
        f"input_shape={tuple(model_cfg.get('input_shape'))} "
        f"num_classes={model_cfg.get('num_classes')}",
    )

    previous_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy("float32")
    try:
        model = _build_model_from_config(
            config,
            architecture_override=model_cfg.get("architecture"),
            weights_override=None,
        )
        if checkpoint_path.lower().endswith(".weights.h5"):
            model.load_weights(checkpoint_path)
        else:
            loaded_model = keras.models.load_model(checkpoint_path)
            model.set_weights(loaded_model.get_weights())
    except Exception as exc:
        print(f"Warning: could not rebuild from checkpoint: {exc}")
        print(traceback.format_exc())
        return None
    finally:
        tf.keras.mixed_precision.set_global_policy(previous_policy)

    print("Rebuilt float32 model from checkpoint weights.")
    return model


def _get_dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for filename in files:
            total += os.path.getsize(os.path.join(root, filename))
    return total


_CALIBRATION_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _calibration_image_files(directory):
    return sorted(
        path
        for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in _CALIBRATION_EXTENSIONS
    )


def stratified_calibration_paths(data_dir, num_samples=1000, seed=42):
    """Round-robin, class-balanced sample of calibration image paths.

    The previous generator consumed the first ``num_samples`` images in
    alphabetical folder order, which on the 72-class benchmark covered only
    the first ~5 class directories and biased every INT8 activation scale.
    Round-robin sampling guarantees every class contributes to calibration.
    """
    data_dir = Path(data_dir)
    queues = {}
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        files = _calibration_image_files(class_dir)
        if files:
            queues[class_dir.name] = files
    rng = random.Random(seed)
    if not queues:
        # Flat directory of images: fall back to a deterministic shuffle.
        paths = _calibration_image_files(data_dir)
        rng.shuffle(paths)
        return paths[:num_samples]
    order = sorted(queues)
    for files in queues.values():
        rng.shuffle(files)
    rng.shuffle(order)
    sampled = []
    while len(sampled) < num_samples and order:
        for name in list(order):
            if not queues[name]:
                order.remove(name)
                continue
            sampled.append(queues[name].pop())
            if len(sampled) >= num_samples:
                break
    return sampled[:num_samples]


def representative_dataset_generator(
    data_dir,
    num_samples=1000,
    image_size=(224, 224),
    preprocess_fn=None,
    seed=42,
):
    """
    Generator for representative dataset used in quantization calibration

    Args:
        data_dir: Directory containing class folders of sample images
        num_samples: Number of calibration samples
        image_size: Input image size
        preprocess_fn: Optional preprocessing applied the same way as during
            training, so calibration stays in the model's input domain
        seed: Deterministic sampling seed

    Yields:
        Preprocessed image batches
    """
    paths = stratified_calibration_paths(
        data_dir, num_samples=num_samples, seed=seed
    )
    print(
        f"Calibrating INT8 on {len(paths)} images sampled round-robin "
        f"across classes of {data_dir}",
        flush=True,
    )
    for path in paths:
        image = keras.utils.load_img(
            path, target_size=tuple(image_size), color_mode="rgb"
        )
        array = np.expand_dims(keras.utils.img_to_array(image), axis=0)
        images = tf.cast(array, tf.float32)
        if preprocess_fn is not None:
            images = preprocess_fn(images)
        yield [images]


def convert_to_tflite(
    model_path,
    output_path,
    quantize=True,
    representative_data_dir=None,
    architecture=None,
    config_path=None,
):
    """
    Convert Keras model to TensorFlow Lite format

    Args:
        model_path: Path to saved Keras model
        output_path: Path to save .tflite model
        quantize: Whether to apply INT8 quantization
        representative_data_dir: Directory with calibration images

    Returns:
        Path to saved .tflite model
    """
    print(f"Loading model from {model_path}...")

    config = _load_config(config_path) if config_path else None

    if _is_saved_model_dir(model_path):
        # A SavedModel exported from a mixed_float16 training model keeps
        # float16 operations that the TFLite converter cannot lower. Rebuild
        # the same weights as a float32 Keras model from the run checkpoint
        # (the path training/export.py already uses) and fall back to the raw
        # SavedModel only when no checkpoint accompanies the export.
        model = _rebuild_float32_model_from_checkpoint(
            model_path, config_path=config_path, architecture=architecture
        )
        if model is not None:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
        else:
            converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    else:
        model = _load_keras_model(
            model_path, config_path=config_path, architecture=architecture
        )
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

    arch = (
        architecture
        or (config.get("model", {}).get("architecture") if config else None)
        or infer_architecture_from_path(model_path)
    )
    preprocess_fn = get_preprocess_fn(arch)

    # Calibration images must match the model's concrete input resolution. A
    # SavedModel export from a hosted runtime profile (e.g. Kaggle T4 trains
    # faster-resolution architectures at 192x192) would otherwise be fed
    # 224x224 calibration batches and conversion would fail.
    calibration_image_size = None
    if model is not None:
        calibration_image_size = concrete_model_image_size(
            getattr(model, "input_shape", None)
        )
        if calibration_image_size:
            print(
                f"Calibrating at model input size: "
                f"{calibration_image_size[0]}x{calibration_image_size[1]}"
            )
    elif _is_saved_model_dir(model_path):
        calibration_image_size = saved_model_image_size(model_path)
        if calibration_image_size:
            print(
                f"Calibrating at SavedModel input size: "
                f"{calibration_image_size[0]}x{calibration_image_size[1]}"
            )
    if quantize:
        print("Applying INT8 quantization...")

        # Set optimization mode
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Set representative dataset for full integer quantization
        if representative_data_dir:
            print(f"Using representative dataset from {representative_data_dir}")
            converter.representative_dataset = lambda: representative_dataset_generator(
                representative_data_dir,
                num_samples=1000,
                preprocess_fn=preprocess_fn,
                image_size=calibration_image_size or (224, 224),
            )

            # Enforce full integer quantization
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        else:
            print(
                "Warning: No representative dataset provided. Using dynamic range quantization."
            )

    # Convert model
    print("Converting model...")
    try:
        tflite_model = converter.convert()
    except Exception:
        # The converter raises a generic message while the actionable detail
        # lives in the captured TensorFlow log. Print the full traceback so a
        # caller running this through subprocess(check=True) surfaces it.
        print("\nTFLite conversion failed. Full traceback:", file=sys.stderr)
        traceback.print_exc()
        raise

    # Save model
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    # Print model size
    if os.path.isfile(model_path):
        original_size = os.path.getsize(model_path)
    elif _is_saved_model_dir(model_path):
        original_size = _get_dir_size(model_path)
    else:
        original_size = 0
    tflite_size = os.path.getsize(output_path)

    print("\nConversion complete!")
    print(f"Original model size: {original_size / (1024 * 1024):.2f} MB")
    print(f"TFLite model size: {tflite_size / (1024 * 1024):.2f} MB")

    if original_size > 0:
        compression_ratio = (1 - tflite_size / original_size) * 100
        print(f"Size reduction: {compression_ratio:.1f}%")

    print(f"Saved to: {output_path}")

    return output_path


def evaluate_tflite_model(
    tflite_path,
    test_data_dir,
    num_samples=100,
    architecture=None,
):
    """
    Evaluate TFLite model accuracy

    Args:
        tflite_path: Path to .tflite model
        test_data_dir: Directory with test images
        num_samples: Number of test samples

    Returns:
        Accuracy score
    """
    print(f"\nEvaluating TFLite model: {tflite_path}")

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input type: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")

    # Load test dataset
    test_ds = keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        image_size=(224, 224),
        batch_size=1,
        label_mode="int",
        shuffle=False,
    )

    correct = 0
    total = 0

    arch = architecture or infer_architecture_from_path(tflite_path)
    preprocess_fn = get_preprocess_fn(arch)

    for images, labels in test_ds:
        if total >= num_samples:
            break

        images = tf.cast(images, tf.float32)
        if preprocess_fn is not None:
            images = preprocess_fn(images)

        input_data = images.numpy()

        # Check if input should be uint8
        if input_details[0]["dtype"] == np.uint8:
            input_scale, input_zero_point = input_details[0]["quantization"]
            if input_scale == 0:
                raise ValueError(
                    f"Quantization scale cannot be zero. Model may be corrupted. "
                    f"Got scale={input_scale}, zero_point={input_zero_point}"
                )
            input_data = input_data / input_scale + input_zero_point
            input_data = np.clip(input_data, 0, 255).astype(np.uint8)
        else:
            input_data = input_data.astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])

        # Dequantize output if needed
        if output_details[0]["dtype"] == np.uint8:
            output_scale, output_zero_point = output_details[0]["quantization"]
            if output_scale == 0:
                raise ValueError(
                    f"Output quantization scale cannot be zero. Model may be corrupted. "
                    f"Got scale={output_scale}, zero_point={output_zero_point}"
                )
            output_data = (
                output_data.astype(np.float32) - float(output_zero_point)
            ) * float(output_scale)

        # Get prediction
        predicted_class = np.argmax(output_data)
        true_class = int(labels.numpy().reshape(-1)[0])

        if predicted_class == true_class:
            correct += 1
        total += 1

    if total == 0:
        raise ValueError(
            "No evaluation samples were processed. Check test_data_dir and num_samples."
        )

    accuracy = correct / total
    print(f"\nAccuracy on {total} samples: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    return accuracy


def benchmark_inference_time(tflite_path, num_runs=100, architecture=None):
    """
    Benchmark inference time of TFLite model

    Args:
        tflite_path: Path to .tflite model
        num_runs: Number of inference runs

    Returns:
        Average inference time in milliseconds
    """
    import time

    print("\nBenchmarking inference time...")

    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    # Create dummy input
    arch = architecture or infer_architecture_from_path(tflite_path)
    if input_dtype == np.uint8:
        dummy_input = np.random.randint(0, 255, input_shape, dtype=np.uint8)
    else:
        if arch == "mobilenetv2":
            dummy_input = np.random.uniform(-1.0, 1.0, input_shape).astype(np.float32)
        elif "efficientnet" in arch:
            dummy_input = np.random.uniform(0.0, 1.0, input_shape).astype(np.float32)
        else:
            dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warm-up runs
    for _ in range(10):
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"Min: {np.min(times):.2f} ms, Max: {np.max(times):.2f} ms")

    return avg_time


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Convert model to TFLite with quantization"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained Keras model"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save .tflite model"
    )
    parser.add_argument(
        "--quantize",
        dest="quantize",
        action="store_true",
        help="Apply quantization (default: enabled)",
    )
    parser.add_argument(
        "--no-quantize",
        dest="quantize",
        action="store_false",
        help="Disable quantization",
    )
    parser.add_argument(
        "--representative_data",
        type=str,
        default=None,
        help="Directory with representative dataset for calibration",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Training config (required for weights-only checkpoints)",
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate TFLite model accuracy"
    )
    parser.add_argument(
        "--test_data", type=str, default=None, help="Test data directory for evaluation"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Benchmark inference time"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Model architecture for preprocessing (mobilenetv2, efficientnet_b0, or efficientnet_lite0)",
    )
    parser.set_defaults(quantize=True)

    args = parser.parse_args()

    if args.evaluate and not args.test_data:
        parser.error("--test_data is required when --evaluate is set.")

    # Convert model
    tflite_path = convert_to_tflite(
        args.model_path,
        args.output_path,
        quantize=args.quantize,
        representative_data_dir=args.representative_data,
        architecture=args.arch,
        config_path=args.config,
    )

    # Evaluate if requested
    if args.evaluate and args.test_data:
        evaluate_tflite_model(tflite_path, args.test_data, architecture=args.arch)

    # Benchmark if requested
    if args.benchmark:
        benchmark_inference_time(tflite_path, architecture=args.arch)


if __name__ == "__main__":
    main()
