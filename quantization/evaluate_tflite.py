"""Evaluate a TensorFlow Lite classifier on an image-directory test split."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse
import json
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from tensorflow import keras


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.utils import get_preprocess_fn


def infer_architecture_from_path(model_path):
    lower_path = model_path.lower()
    if "efficientnet" in lower_path:
        return "efficientnet_lite0"
    if "mobilenet" in lower_path:
        return "mobilenetv2"
    raise ValueError("Pass --arch when the model filename does not identify its architecture.")


def _prepare_input(images, input_details):
    input_data = images.numpy().astype(np.float32)
    input_dtype = input_details["dtype"]
    if np.issubdtype(input_dtype, np.floating):
        return input_data.astype(input_dtype)

    scale, zero_point = input_details["quantization"]
    if scale <= 0:
        raise ValueError("The TFLite input quantization scale must be greater than zero.")

    quantized = np.rint(input_data / scale + zero_point)
    dtype_limits = np.iinfo(input_dtype)
    return np.clip(quantized, dtype_limits.min, dtype_limits.max).astype(input_dtype)


def _dequantize_output(output_data, output_details):
    output_dtype = output_details["dtype"]
    if np.issubdtype(output_dtype, np.floating):
        return output_data.astype(np.float32)

    scale, zero_point = output_details["quantization"]
    if scale <= 0:
        raise ValueError("The TFLite output quantization scale must be greater than zero.")
    return (output_data.astype(np.float32) - zero_point) * scale


def _save_confusion_matrix(matrix, class_names, output_path):
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix.astype(float),
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )

    plt.figure(figsize=(20, 18))
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Count"},
    )
    plt.title("TFLite Confusion Matrix (Normalized)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_tflite_model(
    model_path,
    test_data_dir,
    output_dir,
    architecture=None,
    max_samples=None,
    warmup_runs=10,
):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_shape = input_details["shape"]
    if len(input_shape) != 4 or input_shape[1] <= 0 or input_shape[2] <= 0:
        raise ValueError(f"Unsupported TFLite input shape: {input_shape}")

    image_size = (int(input_shape[1]), int(input_shape[2]))
    test_ds = keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        image_size=image_size,
        batch_size=1,
        label_mode="int",
        shuffle=False,
    )
    class_names = test_ds.class_names
    preprocess_fn = get_preprocess_fn(
        architecture or infer_architecture_from_path(model_path)
    )

    for images, _ in test_ds.take(warmup_runs):
        images = tf.cast(images, tf.float32)
        if preprocess_fn is not None:
            images = preprocess_fn(images)
        interpreter.set_tensor(input_details["index"], _prepare_input(images, input_details))
        interpreter.invoke()

    predictions = []
    labels = []
    inference_times_ms = []
    for images, batch_labels in test_ds:
        if max_samples is not None and len(labels) >= max_samples:
            break

        images = tf.cast(images, tf.float32)
        if preprocess_fn is not None:
            images = preprocess_fn(images)
        input_data = _prepare_input(images, input_details)

        interpreter.set_tensor(input_details["index"], input_data)
        start = time.perf_counter()
        interpreter.invoke()
        inference_times_ms.append((time.perf_counter() - start) * 1000)
        output_data = interpreter.get_tensor(output_details["index"])
        output_data = _dequantize_output(output_data, output_details)

        predictions.append(int(np.argmax(output_data[0])))
        labels.append(int(batch_labels.numpy().reshape(-1)[0]))

    if not labels:
        raise ValueError("No test samples were evaluated.")

    all_labels = range(len(class_names))
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=all_labels,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=all_labels)

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(
        {
            "class": class_names,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
        }
    ).to_csv(os.path.join(output_dir, "class_metrics.csv"), index=False)
    _save_confusion_matrix(
        matrix, class_names, os.path.join(output_dir, "confusion_matrix.png")
    )

    summary = {
        "model_path": os.path.abspath(model_path),
        "architecture": architecture or infer_architecture_from_path(model_path),
        "model_size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 4),
        "samples": len(labels),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "mean_inference_ms": float(np.mean(inference_times_ms)),
        "std_inference_ms": float(np.std(inference_times_ms)),
        "input_dtype": np.dtype(input_details["dtype"]).name,
        "output_dtype": np.dtype(output_details["dtype"]).name,
    }
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved evaluation reports to {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate a TFLite leaf disease model")
    parser.add_argument("--model", required=True, help="Path to a .tflite model")
    parser.add_argument("--test-data", required=True, help="Directory containing class folders")
    parser.add_argument("--output-dir", required=True, help="Directory for evaluation reports")
    parser.add_argument("--arch", default=None, help="mobilenetv2 or efficientnet_lite0")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Optional limit for faster checks"
    )
    parser.add_argument("--warmup-runs", type=int, default=10)
    args = parser.parse_args()

    evaluate_tflite_model(
        args.model,
        args.test_data,
        args.output_dir,
        architecture=args.arch,
        max_samples=args.max_samples,
        warmup_runs=args.warmup_runs,
    )


if __name__ == "__main__":
    main()
