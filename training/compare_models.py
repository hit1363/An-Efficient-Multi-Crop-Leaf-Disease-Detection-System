"""Train, quantize, and compare the thesis mobile classifier candidates."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


ARCHITECTURES = (
    "mobilenet_v1",
    "mobilenetv2",
    "mobilenetv3_small",
    "mobilenet_v3_large",
    "nasnet_mobile",
    "efficientnet_b0",
    "efficientnet_v2_b0",
    "efficientnet_lite0",
)


def _latest_saved_model(model_dir: Path):
    candidates = sorted(
        (path for path in model_dir.glob("saved_model_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _run(command, log_path: Path):
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(
            f"Command returned {result.returncode}; see {log_path}\n"
            f"Last output:\n{result.stdout[-4000:]}"
        )


def _is_pareto_frontier(rows):
    eligible = [row for row in rows if row.get("int8_status") == "ok"]
    frontier = []
    for candidate in eligible:
        dominated = any(
            other is not candidate
            and other["int8_macro_f1"] >= candidate["int8_macro_f1"]
            and other["int8_model_size_mb"] <= candidate["int8_model_size_mb"]
            and other["int8_mean_inference_ms"] <= candidate["int8_mean_inference_ms"]
            and (
                other["int8_macro_f1"] > candidate["int8_macro_f1"]
                or other["int8_model_size_mb"] < candidate["int8_model_size_mb"]
                or other["int8_mean_inference_ms"] < candidate["int8_mean_inference_ms"]
            )
            for other in eligible
        )
        candidate["pareto_optimal"] = not dominated
        if not dominated:
            frontier.append(candidate)
    return frontier


def compare_models(data_dir, output_dir, architectures=ARCHITECTURES, epochs=None):
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    template_path = Path(__file__).with_name("config_benchmark.yaml")
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    class_names_path = data_dir / "class_names.json"
    class_names = json.loads(class_names_path.read_text(encoding="utf-8"))
    if len(class_names) != 72:
        raise ValueError(f"Expected 72 canonical classes, found {len(class_names)}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Comparison output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "configs").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "float_metrics").mkdir(exist_ok=True)
    (output_dir / "tflite").mkdir(exist_ok=True)
    (output_dir / "tflite_metrics").mkdir(exist_ok=True)

    rows = []
    for architecture in architectures:
        row = {
            "architecture": architecture,
            "float_status": "not_run",
            "int8_status": "not_run",
            "pareto_optimal": False,
        }
        config = json.loads(json.dumps(template))
        config["model"]["architecture"] = architecture
        config["model"]["num_classes"] = len(class_names)
        config["dataset"].update(
            {
                "data_dir": str(data_dir),
                "train_dir": str(data_dir / "train"),
                "val_dir": str(data_dir / "val"),
                "test_dir": str(data_dir / "test"),
            }
        )
        config["export"]["save_dir"] = str(output_dir / "models")
        config["callbacks"]["csv_logger"]["filename"] = str(
            output_dir / "logs" / f"{architecture}_training.csv"
        )
        config["logging"]["log_file"] = str(
            output_dir / "logs" / f"{architecture}.log"
        )
        config["callbacks"]["checkpoint"]["directory"] = str(
            output_dir / "checkpoints" / architecture
        )
        if epochs is not None:
            config["training"]["epochs"] = int(epochs)
            config["training"]["unfreeze_epoch"] = min(
                int(config["training"]["unfreeze_epoch"]), int(epochs)
            )
        config_path = output_dir / "configs" / f"{architecture}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        row["config_path"] = str(config_path)

        try:
            _run(
                [sys.executable, str(repo_root / "training" / "train.py"), "--config", str(config_path)],
                output_dir / "logs" / f"{architecture}_train.txt",
            )
            model_dir = output_dir / "models" / architecture
            saved_model = _latest_saved_model(model_dir)
            if saved_model is None:
                raise FileNotFoundError(f"No SavedModel exported for {architecture}")
            row["saved_model_path"] = str(saved_model)
            float_dir = output_dir / "float_metrics" / architecture
            _run(
                [
                    sys.executable,
                    str(repo_root / "training" / "evaluate.py"),
                    "--model",
                    str(saved_model),
                    "--config",
                    str(config_path),
                    "--results-dir",
                    str(float_dir),
                ],
                output_dir / "logs" / f"{architecture}_evaluate.txt",
            )
            float_summary = json.loads(
                (float_dir / "metrics_summary.json").read_text(encoding="utf-8")
            )
            row.update(
                {
                    "float_status": "ok",
                    "float_accuracy": float_summary["accuracy"],
                    "float_macro_f1": float_summary["macro_f1"],
                    "float_model_size_mb": float_summary["model_size_mb"],
                }
            )

            int8_path = output_dir / "tflite" / f"{architecture}_int8.tflite"
            try:
                _run(
                    [
                        sys.executable,
                        str(repo_root / "quantization" / "post_training_quant.py"),
                        "--model_path",
                        str(saved_model),
                        "--output_path",
                        str(int8_path),
                        "--representative_data",
                        str(data_dir / "train"),
                        "--arch",
                        architecture,
                    ],
                    output_dir / "logs" / f"{architecture}_quantize.txt",
                )
                tflite_dir = output_dir / "tflite_metrics" / architecture
                _run(
                    [
                        sys.executable,
                        str(repo_root / "quantization" / "evaluate_tflite.py"),
                        "--model",
                        str(int8_path),
                        "--test-data",
                        str(data_dir / "test"),
                        "--output-dir",
                        str(tflite_dir),
                        "--arch",
                        architecture,
                    ],
                    output_dir / "logs" / f"{architecture}_tflite_eval.txt",
                )
                int8_summary = json.loads(
                    (tflite_dir / "metrics_summary.json").read_text(encoding="utf-8")
                )
                row.update(
                    {
                        "int8_status": "ok",
                        "int8_model_path": str(int8_path),
                        "int8_accuracy": int8_summary["accuracy"],
                        "int8_macro_f1": int8_summary["macro_f1"],
                        "int8_model_size_mb": int8_summary["model_size_mb"],
                        "int8_mean_inference_ms": int8_summary["mean_inference_ms"],
                        "int8_std_inference_ms": int8_summary["std_inference_ms"],
                        "latency_device": int8_summary["latency_device"],
                    }
                )
            except Exception as exc:
                row["int8_status"] = "failed"
                row["int8_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            row["float_status"] = "failed"
            row["float_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        (output_dir / "comparison_progress.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )

    frontier = _is_pareto_frontier(rows)
    comparison_path = output_dir / "model_comparison.csv"
    fields = sorted({key for row in rows for key in row})
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset_dir": str(data_dir),
        "class_count": len(class_names),
        "seed": 42,
        "latency_note": "single-thread TFLite CPU timing is a proxy; validate finalists on target Android hardware",
        "models": rows,
        "pareto_frontier": [row["architecture"] for row in frontier],
        "comparison_csv": str(comparison_path),
    }
    (output_dir / "model_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Harmonized benchmark root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--architectures",
        default=",".join(ARCHITECTURES),
        help="Comma-separated architecture names; defaults to the thesis comparison set",
    )
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    architectures = tuple(name.strip() for name in args.architectures.split(",") if name.strip())
    unknown = sorted(set(architectures) - set(ARCHITECTURES))
    if unknown:
        parser.error(f"Unsupported comparison architectures: {unknown}")
    print(json.dumps(compare_models(args.data_dir, args.output_dir, architectures, args.epochs), indent=2))


if __name__ == "__main__":
    main()
