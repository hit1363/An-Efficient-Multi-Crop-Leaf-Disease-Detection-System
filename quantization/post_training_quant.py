"""
Post-Training Quantization Script
Converts trained TensorFlow model to TensorFlow Lite with INT8 quantization
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path


def representative_dataset_generator(data_dir, num_samples=1000, image_size=(224, 224)):
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

        # Normalize to [0, 1]
        images = images / 255.0
        yield [images]
        count += 1


def convert_to_tflite(
    model_path, output_path, quantize=True, representative_data_dir=None
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

    # Load the model
    if model_path.endswith(".h5"):
        model = keras.models.load_model(model_path)
    else:
        model = keras.models.load_model(model_path)

    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        print("Applying INT8 quantization...")

        # Set optimization mode
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Set representative dataset for full integer quantization
        if representative_data_dir:
            print(f"Using representative dataset from {representative_data_dir}")
            converter.representative_dataset = lambda: representative_dataset_generator(
                representative_data_dir, num_samples=1000
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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    # Print model size
    original_size = os.path.getsize(model_path) if os.path.isfile(model_path) else 0
    tflite_size = os.path.getsize(output_path)

    print(f"\nConversion complete!")
    print(f"Original model size: {original_size / (1024 * 1024):.2f} MB")
    print(f"TFLite model size: {tflite_size / (1024 * 1024):.2f} MB")

    if original_size > 0:
        compression_ratio = (1 - tflite_size / original_size) * 100
        print(f"Size reduction: {compression_ratio:.1f}%")

    print(f"Saved to: {output_path}")

    return output_path


def evaluate_tflite_model(tflite_path, test_data_dir, num_samples=100):
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

    for images, labels in test_ds:
        if total >= num_samples:
            break

        # Preprocess input
        input_data = images.numpy() / 255.0

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

    accuracy = correct / total
    print(f"\nAccuracy on {total} samples: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    return accuracy


def benchmark_inference_time(tflite_path, num_runs=100):
    """
    Benchmark inference time of TFLite model

    Args:
        tflite_path: Path to .tflite model
        num_runs: Number of inference runs

    Returns:
        Average inference time in milliseconds
    """
    import time

    print(f"\nBenchmarking inference time...")

    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    # Create dummy input
    if input_dtype == np.uint8:
        dummy_input = np.random.randint(0, 255, input_shape, dtype=np.uint8)
    else:
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warm-up runs
    for _ in range(10):
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()
        end = time.time()
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
        "--quantize", action="store_true", default=True, help="Apply INT8 quantization"
    )
    parser.add_argument(
        "--representative_data",
        type=str,
        default=None,
        help="Directory with representative dataset for calibration",
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

    args = parser.parse_args()

    # Convert model
    tflite_path = convert_to_tflite(
        args.model_path,
        args.output_path,
        quantize=args.quantize,
        representative_data_dir=args.representative_data,
    )

    # Evaluate if requested
    if args.evaluate and args.test_data:
        evaluate_tflite_model(tflite_path, args.test_data)

    # Benchmark if requested
    if args.benchmark:
        benchmark_inference_time(tflite_path)


if __name__ == "__main__":
    main()
