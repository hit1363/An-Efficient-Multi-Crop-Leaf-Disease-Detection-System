"""
Post-Training Quantization Script
Converts trained TensorFlow model to TensorFlow Lite with INT8 quantization
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import sys
import argparse
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras


def get_preprocess_fn(architecture):
    """Return the appropriate preprocess_input function for a given architecture."""
    if not architecture:
        return None

    arch = architecture.lower()
    if arch == "mobilenetv2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    if arch in {"efficientnet", "efficientnet_lite0"}:
        return lambda x: tf.cast(x, tf.float32) / 255.0
    return None


def infer_architecture_from_path(model_path):
    lower = model_path.lower()
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


def _build_model_from_config(config, architecture_override=None):
    if not config or "model" not in config:
        raise ValueError("Model config is missing or invalid.")

    model_cfg = config.get("model", {})
    architecture = architecture_override or model_cfg.get("architecture", "mobilenetv2")
    input_shape = tuple(model_cfg.get("input_shape", [224, 224, 3]))
    num_classes = int(model_cfg.get("num_classes", 2))
    dropout_rate = float(model_cfg.get("dropout_rate", 0.5))
    weights = model_cfg.get("weights", "imagenet")

    get_model = _get_model_factory()
    model, _ = get_model(
        architecture=architecture,
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        weights=weights,
        hub_url=model_cfg.get("hub_url"),
        hub_cache_dir=model_cfg.get("hub_cache_dir"),
        hub_download_retries=model_cfg.get("hub_download_retries", 1),
        hub_download_delay_sec=model_cfg.get("hub_download_delay_sec", 5),
    )

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


def _get_dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for filename in files:
            total += os.path.getsize(os.path.join(root, filename))
    return total


def representative_dataset_generator(
    data_dir,
    num_samples=1000,
    image_size=(224, 224),
    preprocess_fn=None,
):
    """
    Generator for representative dataset used in quantization calibration

    Args:
        data_dir: Directory containing sample images
        num_samples: Number of calibration samples
        image_size: Input image size

    Yields:
        Preprocessed image batches
    """
    # Load dataset
    dataset = keras.preprocessing.image_dataset_from_directory(
        data_dir, image_size=image_size, batch_size=1, shuffle=True
    )

    count = 0
    for images, _ in dataset:
        if count >= num_samples:
            break

        if preprocess_fn is not None:
            images = tf.cast(images, tf.float32)
            images = preprocess_fn(images)

        yield [images]
        count += 1


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
    tflite_model = converter.convert()

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
        test_data_dir, image_size=(224, 224), batch_size=1, shuffle=False
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
            output_data = (
                output_data.astype(np.float32) - output_zero_point
            ) * output_scale

        # Get prediction
        predicted_class = np.argmax(output_data)
        true_class = np.argmax(labels.numpy())

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
        help="Model architecture for preprocessing (mobilenetv2 or efficientnet_lite0)",
    )
    parser.set_defaults(quantize=True)

    args = parser.parse_args()

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
