"""
Model Evaluation Script
Evaluates trained model on test dataset and generates performance metrics
"""

import os
import yaml
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

try:
    # Support module execution: python -m training.evaluate
    from .utils import get_preprocess_fn
except ImportError:
    # Fallback for script execution: python training/evaluate.py
    from utils import get_preprocess_fn


def _is_saved_model_dir(model_path):
    return os.path.isdir(model_path) and (
        os.path.exists(os.path.join(model_path, "saved_model.pb"))
        or os.path.exists(os.path.join(model_path, "saved_model.pbtxt"))
    )


def _load_saved_model_as_keras(model_path, input_shape):
    if not hasattr(keras.layers, "TFSMLayer"):
        raise ValueError(
            "TFSMLayer is not available in this Keras version. "
            "Please upgrade to Keras 3 or export a .keras/.h5 model."
        )

    tfsm_layer = keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")
    inputs = keras.Input(shape=input_shape, name="input")
    outputs = tfsm_layer(inputs)
    if isinstance(outputs, dict):
        outputs = outputs[next(iter(outputs))]
    return keras.Model(inputs, outputs, name="saved_model_inference")


def _load_model(model_path, input_shape):
    try:
        return keras.models.load_model(model_path)
    except (ValueError, OSError):
        if _is_saved_model_dir(model_path):
            print("Detected SavedModel directory; using TFSMLayer for inference.")
            return _load_saved_model_as_keras(model_path, input_shape)
        raise


def load_config(config_path="config.yaml"):
    """Load configuration file"""
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

    dataset_cfg = config.get("dataset", {})
    for key in ["data_dir", "train_dir", "val_dir", "test_dir"]:
        if key in dataset_cfg:
            dataset_cfg[key] = _resolve_path(dataset_cfg[key], config_dir)

    evaluation_cfg = config.setdefault("evaluation", {})
    if "results_dir" in evaluation_cfg:
        evaluation_cfg["results_dir"] = _resolve_path(
            evaluation_cfg["results_dir"], config_dir
        )
    else:
        evaluation_cfg["results_dir"] = os.path.normpath(
            os.path.join(config_dir, "..", "results")
        )

    return config


def load_test_dataset(
    test_dir, batch_size=32, image_size=(224, 224), preprocess_fn=None
):
    """Load test dataset"""
    test_ds = keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )
    AUTOTUNE = tf.data.AUTOTUNE

    if preprocess_fn is not None:

        def _apply_preprocess(x, y):
            x = tf.cast(x, tf.float32)
            x = preprocess_fn(x)
            return x, y

        test_ds = test_ds.map(_apply_preprocess, num_parallel_calls=AUTOTUNE)

    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return test_ds, test_ds.class_names


def evaluate_model(model_path, config_path="config.yaml"):
    """
    Evaluate trained model on test set

    Args:
        model_path: Path to trained model (.h5 or SavedModel)
        config_path: Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)
    config = resolve_config_paths(config, config_path)

    # Load test dataset
    print("Loading test dataset...")
    test_dir = config["dataset"]["test_dir"]
    batch_size = config["dataset"]["batch_size"]
    input_shape = tuple(config["model"]["input_shape"])
    image_size = input_shape[:2]

    # Load model
    print(f"Loading model from {model_path}...")
    model = _load_model(model_path, input_shape)

    preprocess_fn = get_preprocess_fn(config["model"]["architecture"])
    test_ds, class_names = load_test_dataset(
        test_dir, batch_size, image_size, preprocess_fn=preprocess_fn
    )
    print(f"Found {len(class_names)} classes")

    # Get predictions
    print("Generating predictions...")
    y_pred = []
    y_true = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Compute metrics
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50 + "\n")

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )

    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(y_true, y_pred, average="weighted")
    )

    print("\nMacro Average:")
    print(f"  Precision: {precision_macro:.4f}")
    print(f"  Recall: {recall_macro:.4f}")
    print(f"  F1-Score: {f1_macro:.4f}")

    print("\nWeighted Average:")
    print(f"  Precision: {precision_weighted:.4f}")
    print(f"  Recall: {recall_weighted:.4f}")
    print(f"  F1-Score: {f1_weighted:.4f}")

    # Classification report
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50 + "\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Save metrics to CSV
    results_dir = config.get("evaluation", {}).get("results_dir")
    os.makedirs(results_dir, exist_ok=True)

    metrics_df = pd.DataFrame(
        {
            "Class": class_names,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Support": support,
        }
    )
    metrics_df.to_csv(f"{results_dir}/f1_scores.csv", index=False)
    print(f"\nSaved metrics to {results_dir}/f1_scores.csv")

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(
        cm, class_names, save_path=f"{results_dir}/confusion_matrix.png"
    )

    # Generate performance analysis report
    generate_performance_report(
        accuracy, precision, recall, f1, class_names, results_dir
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    """
    Plot and save confusion matrix

    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save figure
    """
    plt.figure(figsize=(20, 18))

    # Normalize confusion matrix safely to avoid divide-by-zero for empty classes
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm.astype("float"),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    # Plot
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Count"},
    )

    plt.title("Confusion Matrix (Normalized)", fontsize=16, pad=20)
    plt.ylabel("True Label", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved confusion matrix to {save_path}")
    plt.close()


def generate_performance_report(
    accuracy, precision, recall, f1, class_names, results_dir="../results"
):
    """Generate markdown performance analysis report"""
    report_path = f"{results_dir}/performance_analysis.md"

    with open(report_path, "w") as f:
        f.write("# Model Performance Analysis\n\n")
        f.write(f"**Overall Accuracy**: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")

        f.write("## Per-Class Performance\n\n")
        f.write("| Class | Precision | Recall | F1-Score |\n")
        f.write("|-------|-----------|--------|----------|\n")

        for i, name in enumerate(class_names):
            f.write(
                f"| {name} | {precision[i]:.4f} | {recall[i]:.4f} | {f1[i]:.4f} |\n"
            )

        f.write("\n## Best Performing Classes (Top 5 by F1-Score)\n\n")
        top_indices = np.argsort(f1)[-5:][::-1]
        for idx in top_indices:
            f.write(f"- **{class_names[idx]}**: F1={f1[idx]:.4f}\n")

        f.write("\n## Worst Performing Classes (Bottom 5 by F1-Score)\n\n")
        bottom_indices = np.argsort(f1)[:5]
        for idx in bottom_indices:
            f.write(f"- **{class_names[idx]}**: F1={f1[idx]:.4f}\n")

        f.write("\n## Recommendations\n\n")
        f.write("1. Classes with F1-score < 0.8 may need more training data\n")
        f.write("2. Review misclassifications in confusion matrix\n")
        f.write("3. Consider additional data augmentation for low-performing classes\n")

    print(f"Saved performance analysis to {report_path}")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Evaluate leaf disease detection model"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model"
    )
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

    # Evaluate model
    evaluate_model(args.model, config_path)


if __name__ == "__main__":
    main()
