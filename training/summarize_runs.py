"""Aggregate per-architecture float and TFLite artifacts into the thesis table.

Scans a model root (for example ``/kaggle/working/leaf_models`` on Kaggle or
``models/`` locally). For every architecture directory it discovers the
``metrics_summary.json`` files written by ``training/evaluate.py`` (float)
and ``quantization/evaluate_tflite.py`` (dynamic-range and full INT8),
then reports accuracy/macro-F1 for each variant, the quantization accuracy
drop, artifact sizes, the compression ratio against the float32 baseline,
and PASS/FAIL flags for the project targets:

- test accuracy >= 0.90 and macro F1 >= 0.80 (float and INT8)
- INT8 accuracy drop < 2 percentage points
- INT8 size ~= 1/4 of the float32 baseline (>= 3.5x by default)

Usage:
    python training/summarize_runs.py --root /kaggle/working/leaf_models
"""

import argparse
import csv
import json
import os
from pathlib import Path

FLOAT_BUCKET = "float"
DYNAMIC_BUCKET = "dynamic"
INT8_BUCKET = "int8"

COLUMNS = [
    "architecture",
    "float_accuracy",
    "float_macro_f1",
    "float_saved_model_mb",
    "float_tflite_mb",
    "dynamic_accuracy",
    "dynamic_size_mb",
    "int8_accuracy",
    "int8_macro_f1",
    "int8_size_mb",
    "int8_mean_inference_ms",
    "accuracy_drop_pct",
    "macro_f1_drop_points",
    "compression_baseline",
    "compression_ratio_x",
    "classes_without_test_support",
    "meets_float_accuracy",
    "meets_float_macro_f1",
    "meets_int8_accuracy",
    "meets_int8_macro_f1",
    "int8_within_drop_budget",
    "meets_compression",
]


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _dir_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return round(total / (1024.0 * 1024.0), 4)


def _classify_summary(path):
    text = str(path).replace("\\", "/").lower()
    if "tflite" in text:
        if "int8" in text:
            return INT8_BUCKET
        if "dynamic" in text:
            return DYNAMIC_BUCKET
        return None
    return FLOAT_BUCKET


def _latest_summaries(arch_dir):
    buckets = {FLOAT_BUCKET: [], DYNAMIC_BUCKET: [], INT8_BUCKET: []}
    for candidate in sorted(Path(arch_dir).rglob("metrics_summary.json")):
        bucket = _classify_summary(candidate)
        if bucket is None:
            continue
        data = _read_json(candidate)
        if data is not None:
            buckets[bucket].append(candidate)
    return {
        bucket: (max(paths, key=os.path.getmtime) if paths else None)
        for bucket, paths in buckets.items()
    }


def _latest_file(paths):
    paths = list(paths)
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def _float_tflite_path(arch_dir):
    candidates = []
    for path in Path(arch_dir).glob("*.tflite"):
        name = path.name.lower()
        if "int8" in name or "dynamic" in name:
            continue
        candidates.append(path)
    return _latest_file(candidates)


def _quantized_path(arch_dir, variant):
    candidates = [
        path
        for path in Path(arch_dir).joinpath("quantized").glob("*.tflite")
        if variant in path.name.lower()
    ] if (Path(arch_dir) / "quantized").is_dir() else []
    return _latest_file(candidates)


def _saved_model_size_mb(arch_dir):
    directories = [
        path for path in Path(arch_dir).iterdir()
        if path.is_dir() and path.name.lower().startswith("saved_model")
    ]
    if not directories:
        return None
    newest = max(directories, key=os.path.getmtime)
    return _dir_size_mb(newest)


def _round(value, digits=4):
    return None if value is None else round(float(value), digits)


def _flag(value, threshold, minimum=True):
    if value is None:
        return None
    return value >= threshold if minimum else value <= threshold


def collect_architecture(arch_dir, targets):
    arch_dir = Path(arch_dir)
    row = {"architecture": arch_dir.name}
    summaries = _latest_summaries(arch_dir)

    float_data = _read_json(summaries[FLOAT_BUCKET]) if summaries[FLOAT_BUCKET] else None
    dynamic_data = (
        _read_json(summaries[DYNAMIC_BUCKET]) if summaries[DYNAMIC_BUCKET] else None
    )
    int8_data = _read_json(summaries[INT8_BUCKET]) if summaries[INT8_BUCKET] else None

    float_tflite = _float_tflite_path(arch_dir)
    int8_file = _quantized_path(arch_dir, "int8")
    dynamic_file = _quantized_path(arch_dir, "dynamic")
    saved_model_mb = _saved_model_size_mb(arch_dir)

    def _file_mb(path):
        if path is None:
            return None
        try:
            return round(path.stat().st_size / (1024.0 * 1024.0), 4)
        except OSError:
            return None

    float_accuracy = float_data.get("accuracy") if float_data else None
    float_macro_f1 = float_data.get("macro_f1") if float_data else None
    int8_accuracy = int8_data.get("accuracy") if int8_data else None
    int8_macro_f1 = int8_data.get("macro_f1") if int8_data else None

    int8_size_mb = None
    if int8_data and int8_data.get("model_size_mb") is not None:
        int8_size_mb = _round(int8_data["model_size_mb"])
    if int8_size_mb is None:
        int8_size_mb = _file_mb(int8_file)
    dynamic_size_mb = None
    if dynamic_data and dynamic_data.get("model_size_mb") is not None:
        dynamic_size_mb = _round(dynamic_data["model_size_mb"])
    if dynamic_size_mb is None:
        dynamic_size_mb = _file_mb(dynamic_file)

    float_tflite_mb = _file_mb(float_tflite)
    baseline_mb = float_tflite_mb
    baseline_label = "float32_tflite"
    if baseline_mb is None:
        baseline_mb = saved_model_mb
        baseline_label = "saved_model"
    compression_ratio = None
    if baseline_mb and int8_size_mb:
        compression_ratio = round(baseline_mb / int8_size_mb, 2)

    accuracy_drop_pct = macro_drop = None
    if float_accuracy is not None and int8_accuracy is not None:
        accuracy_drop_pct = round((float_accuracy - int8_accuracy) * 100.0, 2)
    if float_macro_f1 is not None and int8_macro_f1 is not None:
        macro_drop = round((float_macro_f1 - int8_macro_f1) * 100.0, 2)

    unsupported = ""
    if float_data and float_data.get("classes_without_test_support"):
        unsupported = ";".join(float_data["classes_without_test_support"])

    row.update(
        {
            "float_accuracy": _round(float_accuracy),
            "float_macro_f1": _round(float_macro_f1),
            "float_saved_model_mb": saved_model_mb,
            "float_tflite_mb": float_tflite_mb,
            "dynamic_accuracy": (
                _round(dynamic_data.get("accuracy")) if dynamic_data else None
            ),
            "dynamic_size_mb": dynamic_size_mb,
            "int8_accuracy": _round(int8_accuracy),
            "int8_macro_f1": _round(int8_macro_f1),
            "int8_size_mb": int8_size_mb,
            "int8_mean_inference_ms": (
                _round(int8_data.get("mean_inference_ms")) if int8_data else None
            ),
            "accuracy_drop_pct": accuracy_drop_pct,
            "macro_f1_drop_points": macro_drop,
            "compression_baseline": baseline_label if int8_size_mb else "",
            "compression_ratio_x": compression_ratio,
            "classes_without_test_support": unsupported,
            "meets_float_accuracy": _flag(
                float_accuracy, targets["min_accuracy"]
            ),
            "meets_float_macro_f1": _flag(float_macro_f1, targets["min_macro_f1"]),
            "meets_int8_accuracy": _flag(int8_accuracy, targets["min_accuracy"]),
            "meets_int8_macro_f1": _flag(int8_macro_f1, targets["min_macro_f1"]),
            "int8_within_drop_budget": (
                None
                if accuracy_drop_pct is None
                else _flag(accuracy_drop_pct, targets["max_accuracy_drop_pct"], minimum=False)
            ),
            "meets_compression": (
                None
                if compression_ratio is None
                else _flag(compression_ratio, targets["min_compression_ratio"])
            ),
        }
    )
    return row


def summarize(root, targets):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Model root does not exist: {root}")
    rows = []
    for arch_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        has_artifacts = any(
            (arch_dir / name).exists()
            for name in ("evaluation_float", "quantized", "tflite_metrics_int8",
                         "tflite_metrics_dynamic")
        ) or any(arch_dir.glob("saved_model*")) or any(arch_dir.rglob("metrics_summary.json"))
        if not has_artifacts:
            continue
        rows.append(collect_architecture(arch_dir, targets))
    return rows


def format_table(rows):
    numeric = {
        "float_accuracy", "float_macro_f1", "float_saved_model_mb", "float_tflite_mb",
        "dynamic_accuracy", "dynamic_size_mb", "int8_accuracy", "int8_macro_f1",
        "int8_size_mb", "int8_mean_inference_ms", "accuracy_drop_pct",
        "macro_f1_drop_points", "compression_ratio_x",
    }
    boolean = {
        "meets_float_accuracy", "meets_float_macro_f1", "meets_int8_accuracy",
        "meets_int8_macro_f1", "int8_within_drop_budget", "meets_compression",
    }

    def display(column, value):
        if column in boolean:
            if value is None:
                return "-"
            return "PASS" if value else "FAIL"
        if value is None or value == "":
            return "-"
        if column in numeric:
            digits = 2 if column.endswith("_mb") else 4
            return f"{float(value):.{digits}f}"
        return str(value)

    widths = {
        column: max(len(column), *(len(display(column, row.get(column))) for row in rows))
        if rows else len(column)
        for column in COLUMNS
    }
    header = " | ".join(column.ljust(widths[column]) for column in COLUMNS)
    lines = [header, "-+-".join("-" * widths[column] for column in COLUMNS)]
    for row in rows:
        lines.append(
            " | ".join(
                display(column, row.get(column)).ljust(widths[column])
                for column in COLUMNS
            )
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", required=True,
        help="Model root containing one directory per architecture "
             "(e.g. /kaggle/working/leaf_models)",
    )
    parser.add_argument("--output", default=None, help="CSV path (default: <root>/model_summary.csv)")
    parser.add_argument("--min-accuracy", type=float, default=0.90)
    parser.add_argument("--min-macro-f1", type=float, default=0.80)
    parser.add_argument("--max-accuracy-drop-pct", type=float, default=2.0)
    parser.add_argument("--min-compression-ratio", type=float, default=3.5)
    args = parser.parse_args(argv)

    targets = {
        "min_accuracy": args.min_accuracy,
        "min_macro_f1": args.min_macro_f1,
        "max_accuracy_drop_pct": args.max_accuracy_drop_pct,
        "min_compression_ratio": args.min_compression_ratio,
    }
    rows = summarize(args.root, targets)
    if not rows:
        print(f"No architecture runs found under {args.root}")
        return 1

    output = args.output or os.path.join(args.root, "model_summary.csv")
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")
    print()
    print(format_table(rows))
    print()
    print(
        "Targets: accuracy >= {0}, macro F1 >= {1}, INT8 accuracy drop <= {2} pct, "
        "compression >= {3}x vs float32 baseline".format(
            args.min_accuracy, args.min_macro_f1, args.max_accuracy_drop_pct,
            args.min_compression_ratio,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
