"""Shared evaluation and ranking helpers for the Colab/Kaggle comparison notebooks."""

import json
import math
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from quantization.evaluate_tflite import (
    _prepare_input,
    evaluate_tflite_model,
)
from training.evaluate import _load_model, load_test_dataset
from training.utils import get_preprocess_fn


ARCHITECTURES = (
    "mobilenetv2",
    "efficientnet_b0",
    "efficientnet_lite0",
    "mobilenetv3_small",
    "shufflenetv2_05",
)
PREFERRED_VARIANTS = ("int8", "dynamic", "float")


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _artifact_size_mb(path):
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / (
        1024 * 1024
    )


def _newest(paths):
    paths = [Path(path) for path in paths if Path(path).exists()]
    return str(max(paths, key=lambda path: path.stat().st_mtime)) if paths else None


def discover_artifacts(model_root, architecture):
    """Find the newest float, dynamic-range, and INT8 artifacts."""
    model_dir = Path(model_root) / architecture
    saved_models = [
        path
        for path in model_dir.glob("saved_model_*")
        if path.is_dir() and (path / "saved_model.pb").exists()
    ]
    h5_models = [
        path
        for path in model_dir.glob("*.h5")
        if not path.name.endswith(".weights.h5")
    ]
    float_path = _newest(saved_models) or _newest(h5_models)

    quant_dir = model_dir / "quantized"
    dynamic_paths = list(quant_dir.glob("*dynamic*.tflite"))
    dynamic_paths += list(model_dir.glob("*dynamic*.tflite"))
    int8_paths = list(quant_dir.glob("*int8*.tflite"))
    int8_paths += list(model_dir.glob("*int8*.tflite"))

    return {
        "float": float_path,
        "dynamic": _newest(dynamic_paths),
        "int8": _newest(int8_paths),
    }


def _metric_summary(y_true, y_pred, class_names):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=range(len(class_names)),
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "class_metrics": pd.DataFrame(
            {
                "class": class_names,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": support,
            }
        ),
    }


def _save_confusion_matrix(y_true, y_pred, class_names, output_path, title):
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
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
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized count"},
    )
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def evaluate_float_model(
    model_path,
    architecture,
    test_dir,
    input_shape,
    class_names,
    output_dir,
    max_samples=None,
    batch_size=32,
):
    """Evaluate a float SavedModel/H5 artifact and save detailed reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(model_path, tuple(input_shape))
    test_ds, detected_classes = load_test_dataset(
        test_dir,
        batch_size=batch_size,
        image_size=tuple(input_shape[:2]),
        preprocess_fn=get_preprocess_fn(architecture),
    )
    if list(detected_classes) != list(class_names):
        raise ValueError(f"Class ordering mismatch while evaluating {architecture}.")

    y_true = []
    y_pred = []
    for images, labels in test_ds:
        if max_samples is not None and len(y_true) >= max_samples:
            break
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1).tolist())
        y_true.extend(np.argmax(labels.numpy(), axis=1).tolist())
        if max_samples is not None and len(y_true) >= max_samples:
            y_true = y_true[:max_samples]
            y_pred = y_pred[:max_samples]

    metrics = _metric_summary(np.array(y_true), np.array(y_pred), class_names)
    metrics["class_metrics"].to_csv(output_dir / "class_metrics.csv", index=False)
    _save_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        output_dir / "confusion_matrix.png",
        f"{architecture} float confusion matrix",
    )
    summary = {
        "architecture": architecture,
        "variant": "float",
        "status": "ok",
        "model_path": os.path.abspath(model_path),
        "model_size_mb": round(_artifact_size_mb(model_path), 4),
        "samples": len(y_true),
        **{key: value for key, value in metrics.items() if key != "class_metrics"},
    }
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2)
    return summary


def _tflite_latency(model_path, architecture, input_shape, warmup_runs=10, runs=50):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    shape = [int(value) for value in input_details["shape"]]
    if any(value <= 0 for value in shape):
        shape = [1, int(input_shape[0]), int(input_shape[1]), 3]
        interpreter.resize_tensor_input(input_details["index"], shape)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]

    raw = tf.zeros(shape, dtype=tf.float32)
    preprocess_fn = get_preprocess_fn(architecture)
    if preprocess_fn is not None:
        raw = preprocess_fn(raw)
    input_data = _prepare_input(raw, input_details)

    warmup_times = []
    for _ in range(warmup_runs):
        start = tf.timestamp()
        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke()
        warmup_times.append(float((tf.timestamp() - start).numpy()) * 1000.0)

    inference_times = []
    for _ in range(runs):
        start = tf.timestamp()
        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke()
        inference_times.append(float((tf.timestamp() - start).numpy()) * 1000.0)

    return {
        "warmup_mean_ms": float(np.mean(warmup_times)),
        "mean_inference_ms": float(np.mean(inference_times)),
        "std_inference_ms": float(np.std(inference_times)),
        "input_dtype": np.dtype(input_details["dtype"]).name,
        "output_dtype": np.dtype(output_details["dtype"]).name,
    }


def evaluate_tflite_artifact(
    model_path,
    architecture,
    test_dir,
    input_shape,
    output_dir,
    max_samples=None,
):
    """Evaluate TFLite accuracy, quantization metadata, and CPU latency."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = evaluate_tflite_model(
        model_path,
        test_dir,
        str(output_dir),
        architecture=architecture,
        max_samples=max_samples,
        warmup_runs=10,
    )
    summary.update(_tflite_latency(model_path, architecture, input_shape))
    summary.update(
        {
            "variant": "int8" if "int8" in Path(model_path).name.lower() else "dynamic",
            "quantization": "full_int8"
            if "int8" in Path(model_path).name.lower()
            else "dynamic_range",
            "model_size_mb": round(_artifact_size_mb(model_path), 4),
            "status": "ok",
        }
    )
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2)
    return summary


def _missing_record(architecture, variant):
    return {
        "architecture": architecture,
        "variant": variant,
        "status": "missing",
        "model_path": "",
    }


def _error_record(architecture, variant, error):
    return {
        "architecture": architecture,
        "variant": variant,
        "status": "error",
        "model_path": "",
        "error": str(error),
    }


def _minmax(values, higher_is_better=True):
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(0.0, index=values.index)
    low, high = valid.min(), valid.max()
    if high == low:
        result = pd.Series(1.0, index=values.index)
    else:
        result = (numeric - low) / (high - low)
    result = result.fillna(0.0)
    return result if higher_is_better else 1.0 - result


def _select_deployment_candidates(summary_df):
    candidates = []
    for architecture in ARCHITECTURES:
        group = summary_df[
            (summary_df["architecture"] == architecture)
            & (summary_df["status"] == "ok")
        ]
        selected = None
        for variant in PREFERRED_VARIANTS:
            matches = group[group["variant"] == variant]
            if not matches.empty:
                selected = matches.iloc[0].to_dict()
                break
        if selected:
            candidates.append(selected)
    return pd.DataFrame(candidates)


def _rank_candidates(summary_df):
    candidates = _select_deployment_candidates(summary_df)
    if candidates.empty:
        return candidates
    candidates = candidates.copy()
    for column in ("mean_inference_ms", "std_inference_ms", "warmup_mean_ms"):
        if column not in candidates.columns:
            candidates[column] = np.nan
    candidates["accuracy_score"] = _minmax(candidates["accuracy"])
    candidates["macro_f1_score"] = _minmax(candidates["macro_f1"])
    candidates["latency_score"] = _minmax(
        candidates.get("mean_inference_ms", pd.Series(index=candidates.index)),
        higher_is_better=False,
    )
    candidates["size_score"] = _minmax(
        candidates["model_size_mb"], higher_is_better=False
    )
    candidates["balanced_score"] = (
        0.45 * candidates["accuracy_score"]
        + 0.35 * candidates["macro_f1_score"]
        + 0.10 * candidates["latency_score"]
        + 0.10 * candidates["size_score"]
    )
    candidates["edge_score"] = (
        0.60 * candidates["latency_score"] + 0.40 * candidates["size_score"]
    )
    return candidates.sort_values("balanced_score", ascending=False).reset_index(drop=True)


def _make_plots(summary_df, ranking_df, float_df, output_dir):
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if ranking_df.empty:
        return

    chart = ranking_df.set_index("architecture")[["accuracy", "macro_f1"]]
    chart.plot(kind="bar", figsize=(12, 6), ylim=(0, 1), title="Best available variant")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(plots_dir / "accuracy_f1_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.scatter(
        ranking_df["model_size_mb"],
        ranking_df["mean_inference_ms"],
        s=100,
    )
    for _, row in ranking_df.iterrows():
        plt.annotate(row["architecture"], (row["model_size_mb"], row["mean_inference_ms"]))
    plt.xlabel("TFLite/model size (MB)")
    plt.ylabel("Mean CPU latency (ms)")
    plt.title("Edge size versus latency")
    plt.tight_layout()
    plt.savefig(plots_dir / "size_vs_latency.png", dpi=160)
    plt.close()

    int8_df = summary_df[
        (summary_df["variant"] == "int8") & (summary_df["status"] == "ok")
    ]
    if not float_df.empty and not int8_df.empty:
        merged = float_df.merge(
            int8_df[["architecture", "accuracy", "macro_f1"]],
            on="architecture",
            suffixes=("_float", "_int8"),
        )
        if not merged.empty:
            merged["accuracy_drop"] = merged["accuracy_float"] - merged["accuracy_int8"]
            merged["macro_f1_drop"] = merged["macro_f1_float"] - merged["macro_f1_int8"]
            merged.set_index("architecture")[["accuracy_drop", "macro_f1_drop"]].plot(
                kind="bar", figsize=(12, 6), title="Float to INT8 metric drop"
            )
            plt.ylabel("Drop (float - INT8)")
            plt.tight_layout()
            plt.savefig(plots_dir / "float_to_int8_drop.png", dpi=160)
            plt.close()

    pareto = ranking_df.copy()
    dominated = []
    for index, row in pareto.iterrows():
        is_dominated = any(
            other["accuracy"] >= row["accuracy"]
            and other["model_size_mb"] <= row["model_size_mb"]
            and other["mean_inference_ms"] <= row["mean_inference_ms"]
            and (
                other["accuracy"] > row["accuracy"]
                or other["model_size_mb"] < row["model_size_mb"]
                or other["mean_inference_ms"] < row["mean_inference_ms"]
            )
            for _, other in pareto.iterrows()
        )
        dominated.append(is_dominated)
    pareto["pareto_optimal"] = ~np.array(dominated)
    plt.figure(figsize=(10, 7))
    plt.scatter(
        pareto["model_size_mb"],
        pareto["mean_inference_ms"],
        c=pareto["pareto_optimal"].map({True: "tab:green", False: "tab:gray"}),
        s=120,
    )
    for _, row in pareto.iterrows():
        plt.annotate(row["architecture"], (row["model_size_mb"], row["mean_inference_ms"]))
    plt.xlabel("Size (MB)")
    plt.ylabel("Latency (ms)")
    plt.title("Edge deployment Pareto candidates")
    plt.tight_layout()
    plt.savefig(plots_dir / "pareto_frontier.png", dpi=160)
    plt.close()


def run_comparison(
    repo_dir,
    dataset_base,
    model_root,
    output_dir,
    max_test_samples=None,
    eval_batch_size=32,
):
    """Run the complete five-model comparison and write all deliverables."""
    repo_dir = Path(repo_dir)
    dataset_base = Path(dataset_base)
    model_root = Path(model_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "float_results").mkdir(exist_ok=True)
    (output_dir / "tflite_results").mkdir(exist_ok=True)
    (output_dir / "confusion_matrices").mkdir(exist_ok=True)

    test_dir = dataset_base / "test"
    class_names = sorted(
        path.name for path in test_dir.iterdir() if path.is_dir()
    )
    if not class_names:
        raise ValueError(f"No class folders found in {test_dir}")

    records = []
    manifests = []
    config_cache = {}
    for architecture in ARCHITECTURES:
        config_path = repo_dir / "training" / f"config_{architecture}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config: {config_path}")
        import yaml

        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config["dataset"]["test_dir"] = str(test_dir)
        config["model"]["num_classes"] = len(class_names)
        config_cache[architecture] = config

        artifacts = discover_artifacts(model_root, architecture)
        manifests.append({"architecture": architecture, **artifacts})
        for variant in ("float", "dynamic", "int8"):
            artifact = artifacts[variant]
            if not artifact:
                records.append(_missing_record(architecture, variant))
                continue
            try:
                if variant == "float":
                    result = evaluate_float_model(
                        artifact,
                        architecture,
                        str(test_dir),
                        tuple(config["model"]["input_shape"]),
                        class_names,
                        output_dir / "float_results" / architecture,
                        max_samples=max_test_samples,
                        batch_size=eval_batch_size,
                    )
                    shutil.copyfile(
                        output_dir / "float_results" / architecture / "confusion_matrix.png",
                        output_dir / "confusion_matrices" / f"{architecture}_float.png",
                    )
                else:
                    result = evaluate_tflite_artifact(
                        artifact,
                        architecture,
                        str(test_dir),
                        tuple(config["model"]["input_shape"]),
                        output_dir / "tflite_results" / f"{architecture}_{variant}",
                        max_samples=max_test_samples,
                    )
                    shutil.copyfile(
                        output_dir
                        / "tflite_results"
                        / f"{architecture}_{variant}"
                        / "confusion_matrix.png",
                        output_dir / "confusion_matrices" / f"{architecture}_{variant}.png",
                    )
                records.append(result)
            except Exception as error:
                print(f"{architecture}/{variant} failed: {error}")
                records.append(_error_record(architecture, variant, error))
            finally:
                tf.keras.backend.clear_session()

    summary_df = pd.DataFrame(records)
    float_df = summary_df[
        (summary_df["variant"] == "float") & (summary_df["status"] == "ok")
    ].copy()
    ranking_df = _rank_candidates(summary_df)
    if not ranking_df.empty:
        float_lookup = float_df.set_index("architecture") if not float_df.empty else None
        for index, row in ranking_df.iterrows():
            if float_lookup is not None and row["architecture"] in float_lookup.index:
                ranking_df.loc[index, "accuracy_delta_vs_float"] = (
                    row["accuracy"] - float_lookup.loc[row["architecture"], "accuracy"]
                )
                ranking_df.loc[index, "macro_f1_delta_vs_float"] = (
                    row["macro_f1"] - float_lookup.loc[row["architecture"], "macro_f1"]
                )

    summary_df.to_csv(output_dir / "comparison_summary.csv", index=False)
    summary_df.to_json(output_dir / "comparison_summary.json", orient="records", indent=2)
    ranking_df.to_csv(output_dir / "ranking.csv", index=False)
    ranking_df.to_json(output_dir / "ranking.json", orient="records", indent=2)
    with open(output_dir / "artifact_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(_json_safe(manifests), handle, indent=2)

    _make_plots(summary_df, ranking_df, float_df, output_dir)

    best_float = (
        float_df.sort_values(["accuracy", "macro_f1"], ascending=False).iloc[0].to_dict()
        if not float_df.empty
        else None
    )
    best_edge = (
        ranking_df.sort_values("edge_score", ascending=False).iloc[0].to_dict()
        if not ranking_df.empty
        else None
    )
    best_balanced = (
        ranking_df.sort_values("balanced_score", ascending=False).iloc[0].to_dict()
        if not ranking_df.empty
        else None
    )
    recommendation = {
        "best_float_accuracy": best_float,
        "best_edge_deployment": best_edge,
        "best_balanced": best_balanced,
        "score_weights": {
            "accuracy": 0.45,
            "macro_f1": 0.35,
            "latency": 0.10,
            "size": 0.10,
        },
        "max_test_samples": max_test_samples,
    }
    with open(output_dir / "best_model_report.json", "w", encoding="utf-8") as handle:
        json.dump(_json_safe(recommendation), handle, indent=2)

    report_lines = ["# Best Model Recommendation", ""]
    for label, result in (
        ("Best float accuracy", best_float),
        ("Best edge deployment", best_edge),
        ("Best balanced model", best_balanced),
    ):
        report_lines.append(f"## {label}")
        if result is None:
            report_lines.append("No complete artifact was available.")
        else:
            report_lines.extend(
                [
                    f"- Architecture: `{result['architecture']}`",
                    f"- Variant: `{result['variant']}`",
                    f"- Accuracy: `{result.get('accuracy', 0):.4f}`",
                    f"- Macro F1: `{result.get('macro_f1', 0):.4f}`",
                    f"- Size MB: `{result.get('model_size_mb', 0):.2f}`",
                    f"- Latency ms: `{result.get('mean_inference_ms', float('nan'))}`",
                    f"- Artifact: `{result.get('model_path', '')}`",
                ]
            )
        report_lines.append("")
    (output_dir / "best_model_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    archive_base = str(output_dir)
    archive_path = shutil.make_archive(archive_base, "zip", root_dir=output_dir)
    print("Comparison complete.")
    print(json.dumps(_json_safe(recommendation), indent=2))
    print("Archive:", archive_path)
    return recommendation, summary_df, ranking_df
