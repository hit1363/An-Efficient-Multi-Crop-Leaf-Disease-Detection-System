import json

from training.summarize_runs import summarize


TARGETS = {
    "min_accuracy": 0.9,
    "min_macro_f1": 0.8,
    "max_accuracy_drop_pct": 2.0,
    "min_compression_ratio": 3.5,
}


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_summarize_computes_deltas_and_flags(tmp_path):
    arch = tmp_path / "mobilenetv2"
    _write(
        arch / "evaluation_float" / "metrics_summary.json",
        {
            "accuracy": 0.93,
            "macro_f1": 0.85,
            "model_size_mb": 9.8,
            "classes_without_test_support": ["Guava___Scorch"],
        },
    )
    _write(
        arch / "tflite_metrics_int8" / "metrics_summary.json",
        {
            "accuracy": 0.922,
            "macro_f1": 0.841,
            "model_size_mb": 2.5,
            "mean_inference_ms": 18.0,
        },
    )
    _write(
        arch / "tflite_metrics_dynamic" / "metrics_summary.json",
        {"accuracy": 0.928, "macro_f1": 0.847, "model_size_mb": 2.7},
    )
    (arch / "quantized").mkdir(parents=True, exist_ok=True)
    (arch / "mobilenetv2_20250101.tflite").write_bytes(b"x" * (10 * 1024 * 1024))
    (arch / "quantized" / "mobilenetv2_int8.tflite").write_bytes(b"y")

    rows = summarize(tmp_path, TARGETS)

    assert len(rows) == 1
    row = rows[0]
    assert row["architecture"] == "mobilenetv2"
    assert row["float_accuracy"] == 0.93
    assert row["float_tflite_mb"] == 10.0
    assert row["int8_size_mb"] == 2.5
    assert row["dynamic_size_mb"] == 2.7
    assert row["accuracy_drop_pct"] == 0.8
    assert row["macro_f1_drop_points"] == 0.9
    assert row["compression_baseline"] == "float32_tflite"
    assert row["compression_ratio_x"] == 4.0
    assert row["classes_without_test_support"] == "Guava___Scorch"
    assert row["meets_float_accuracy"] is True
    assert row["meets_int8_accuracy"] is True
    assert row["int8_within_drop_budget"] is True
    assert row["meets_compression"] is True


def test_summarize_falls_back_to_saved_model_baseline(tmp_path):
    arch = tmp_path / "nasnet_mobile"
    _write(
        arch / "evaluation_float" / "metrics_summary.json",
        {"accuracy": 0.88, "macro_f1": 0.75},
    )
    _write(
        arch / "tflite_metrics_int8" / "metrics_summary.json",
        {"accuracy": 0.8, "macro_f1": 0.7, "model_size_mb": 2.0},
    )
    weights = arch / "saved_model_20250101"
    weights.mkdir(parents=True)
    (weights / "assets.txt").write_bytes(b"z" * (24 * 1024 * 1024))

    row = summarize(tmp_path, TARGETS)[0]

    assert row["compression_baseline"] == "saved_model"
    assert row["compression_ratio_x"] == 12.0
    assert row["meets_float_accuracy"] is False
    assert row["meets_float_macro_f1"] is False
    assert row["int8_within_drop_budget"] is False  # 8 percentage points


def test_summarize_skips_directories_without_metrics(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "run.txt").write_text("nothing here", encoding="utf-8")

    assert summarize(tmp_path, TARGETS) == []
