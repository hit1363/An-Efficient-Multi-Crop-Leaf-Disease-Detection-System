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

from utils import load_class_names, preprocess_image


def load_config(config_path="config.yaml"):
    """Load configuration file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_test_dataset(test_dir, batch_size=32, image_size=(224, 224)):
    """Load test dataset"""
    test_ds = keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

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

    # Load model
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)

    # Load test dataset
    print("Loading test dataset...")
    test_dir = config["dataset"]["test_dir"]
    batch_size = config["dataset"]["batch_size"]
    image_size = tuple(config["model"]["input_shape"][:2])

    test_ds, class_names = load_test_dataset(test_dir, batch_size, image_size)
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

    print(f"\nMacro Average:")
    print(f"  Precision: {precision_macro:.4f}")
    print(f"  Recall: {recall_macro:.4f}")
    print(f"  F1-Score: {f1_macro:.4f}")

    print(f"\nWeighted Average:")
    print(f"  Precision: {precision_weighted:.4f}")
    print(f"  Recall: {recall_weighted:.4f}")
    print(f"  F1-Score: {f1_weighted:.4f}")

    # Classification report
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50 + "\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Save metrics to CSV
    results_dir = "../results"
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

    # Normalize confusion matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

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
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )

    args = parser.parse_args()

    # Evaluate model
    evaluate_model(args.model, args.config)


if __name__ == "__main__":
    main()
