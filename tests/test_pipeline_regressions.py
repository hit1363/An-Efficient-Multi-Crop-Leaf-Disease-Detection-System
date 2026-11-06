import json

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from quantization import evaluate_tflite, post_training_quant
from training import evaluate, train
from training.utils import setup_callbacks


def test_load_test_dataset_preserves_explicit_class_order(monkeypatch):
    captured = {}

    class Dataset:
        class_names = ["late", "early"]

        def prefetch(self, buffer_size):
            return self

    def fake_loader(test_dir, **kwargs):
        captured["test_dir"] = test_dir
        captured.update(kwargs)
        return Dataset()

    monkeypatch.setattr(
        evaluate.keras.preprocessing,
        "image_dataset_from_directory",
        fake_loader,
    )

    dataset, class_names = evaluate.load_test_dataset(
        "test-split",
        class_names=["late", "early"],
    )

    assert dataset.class_names == ["late", "early"]
    assert class_names == ["late", "early"]
    assert captured["class_names"] == ["late", "early"]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("models/mobilenetv3_small/model.tflite", "mobilenetv3_small"),
        ("models/shufflenetv2_05/model.tflite", "shufflenetv2_05"),
        ("models/mobilenet_v1/model.tflite", "mobilenet_v1"),
        ("models/mobilenet_v3_large/model.tflite", "mobilenet_v3_large"),
        ("models/nasnet_mobile/model.tflite", "nasnet_mobile"),
        ("models/efficientnet_v2_b0/model.tflite", "efficientnet_v2_b0"),
    ],
)
def test_quantization_architecture_inference_preserves_model_families(path, expected):
    assert evaluate_tflite.infer_architecture_from_path(path) == expected
    assert post_training_quant.infer_architecture_from_path(path) == expected


def test_tflite_evaluation_is_single_threaded_and_scores_absent_classes(
    monkeypatch, tmp_path
):
    model_path = tmp_path / "model.tflite"
    model_path.write_bytes(b"tflite model")
    interpreter_options = {}

    class Interpreter:
        def __init__(self, **kwargs):
            interpreter_options.update(kwargs)

        def allocate_tensors(self):
            pass

        def get_input_details(self):
            return [
                {
                    "shape": np.asarray([1, 2, 2, 3]),
                    "dtype": np.float32,
                    "index": 0,
                    "quantization": (0.0, 0),
                }
            ]

        def get_output_details(self):
            return [
                {
                    "shape": np.asarray([1, 2]),
                    "dtype": np.float32,
                    "index": 1,
                    "quantization": (0.0, 0),
                }
            ]

        def set_tensor(self, index, value):
            pass

        def invoke(self):
            pass

        def get_tensor(self, index):
            return np.asarray([[0.8, 0.2]], dtype=np.float32)

    class Dataset:
        class_names = ["class_a", "class_b", "class_c"]

        def __iter__(self):
            for label in (0, 1):
                yield tf.zeros((1, 2, 2, 3)), tf.constant([label])

        def take(self, count):
            return iter(()) if count <= 0 else iter(self)

    monkeypatch.setattr(evaluate_tflite.tf.lite, "Interpreter", Interpreter)
    monkeypatch.setattr(
        evaluate_tflite.keras.preprocessing,
        "image_dataset_from_directory",
        lambda *args, **kwargs: Dataset(),
    )
    monkeypatch.setattr(evaluate_tflite, "get_preprocess_fn", lambda architecture: None)
    monkeypatch.setattr(evaluate_tflite, "_save_confusion_matrix", lambda *args: None)

    summary = evaluate_tflite.evaluate_tflite_model(
        str(model_path),
        str(tmp_path / "test"),
        str(tmp_path / "reports"),
        architecture="mobilenetv2",
        warmup_runs=0,
    )

    assert interpreter_options["num_threads"] == 1
    assert summary["latency_device"] == "single-thread CPU interpreter proxy"
    # class_c has no test support and must still count as a zero-score class.
    assert summary["macro_f1"] == pytest.approx(2 / 9)
    saved_summary = json.loads(
        (tmp_path / "reports" / "metrics_summary.json").read_text(encoding="utf-8")
    )
    assert saved_summary["latency_device"] == summary["latency_device"]


def test_tflite_evaluation_uses_parent_manifest_class_order(monkeypatch, tmp_path):
    model_path = tmp_path / "model.tflite"
    model_path.write_bytes(b"tflite model")
    test_dir = tmp_path / "test"
    (test_dir / "a").mkdir(parents=True)
    (test_dir / "b").mkdir()
    (tmp_path / "class_names.json").write_text(json.dumps(["b", "a"]))
    captured = {}

    class Interpreter:
        def __init__(self, **kwargs):
            pass

        def allocate_tensors(self):
            pass

        def get_input_details(self):
            return [{
                "shape": np.asarray([1, 2, 2, 3]),
                "dtype": np.float32,
                "index": 0,
                "quantization": (0.0, 0),
            }]

        def get_output_details(self):
            return [{
                "shape": np.asarray([1, 2]),
                "dtype": np.float32,
                "index": 1,
                "quantization": (0.0, 0),
            }]

        def set_tensor(self, index, value):
            pass

        def invoke(self):
            pass

        def get_tensor(self, index):
            return np.asarray([[0.8, 0.2]], dtype=np.float32)

    class Dataset:
        class_names = ["b", "a"]

        def __iter__(self):
            yield tf.zeros((1, 2, 2, 3)), tf.constant([0])

        def take(self, count):
            return iter(()) if count <= 0 else iter(self)

    def fake_loader(*args, **kwargs):
        captured.update(kwargs)
        return Dataset()

    monkeypatch.setattr(evaluate_tflite.tf.lite, "Interpreter", Interpreter)
    monkeypatch.setattr(
        evaluate_tflite.keras.preprocessing,
        "image_dataset_from_directory",
        fake_loader,
    )
    monkeypatch.setattr(evaluate_tflite, "get_preprocess_fn", lambda architecture: None)
    monkeypatch.setattr(evaluate_tflite, "_save_confusion_matrix", lambda *args: None)

    summary = evaluate_tflite.evaluate_tflite_model(
        str(model_path),
        str(test_dir),
        str(tmp_path / "reports"),
        architecture="mobilenetv2",
        warmup_runs=0,
    )

    assert captured["class_names"] == ["b", "a"]
    assert summary["class_names"] == ["b", "a"]


def test_phase2_callback_setup_appends_to_phase1_csv(tmp_path):
    config = {
        "model": {"architecture": "mobilenetv2"},
        "export": {"save_dir": str(tmp_path / "models")},
        "dataset": {"batch_size": 8},
        "callbacks": {
            "checkpoint": {"enabled": False},
            "early_stopping": {"enabled": False},
            "tensorboard": {"enabled": False},
            "csv_logger": {
                "enabled": True,
                "filename": str(tmp_path / "training.csv"),
                "append": False,
            },
        },
        "performance": {"terminate_on_nan": False, "log_throughput": False},
        "optimizer": {"learning_rate": 0.001},
        "training": {"epochs": 2},
    }

    callbacks = train._setup_phase2_callbacks(config)
    csv_logger = next(
        callback
        for callback in callbacks
        if isinstance(callback, tf.keras.callbacks.CSVLogger)
    )

    assert csv_logger.append is True
    assert csv_logger.filename == str(tmp_path / "training.csv")


def test_phase2_callback_factory_requests_phase2_checkpoint(monkeypatch):
    captured = {}

    def fake_setup_callbacks(config, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(train, "setup_callbacks", fake_setup_callbacks)

    train._setup_phase2_callbacks({})

    assert captured == {
        "csv_append": True,
        "checkpoint_name": "best_phase2_model",
        "include_throughput": False,
    }


def test_load_model_handles_exported_savedmodel_without_keras_metadata(tmp_path):
    """``model.export()`` writes a SavedModel with no ``keras_metadata.pb``.

    On this Keras version ``keras.models.load_model`` does NOT raise on such a
    directory — it returns a bare ``_UserObject`` with no ``predict`` — so the
    loader must validate the loaded object before falling back to the
    SavedModel signature. Without this guard the whole evaluation crashes at
    the first ``model.predict`` call.
    """
    import tensorflow as tf
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.Input(shape=(224, 224, 3)),
            keras.layers.Flatten(),
            keras.layers.Dense(72, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    saved_dir = tmp_path / "saved_model_20261105_045056"
    model.export(str(saved_dir))
    assert not (saved_dir / "keras_metadata.pb").exists()

    loaded = evaluate._load_model(str(saved_dir), (224, 224, 3))
    assert hasattr(loaded, "predict")
    outputs = loaded.predict(tf.zeros((1, 224, 224, 3), dtype=tf.float32), verbose=0)
    assert np.asarray(outputs).shape == (1, 72)


def test_find_run_checkpoint_maps_savedmodel_to_run_scoped_artifacts(tmp_path):
    """The SavedModel name carries the timestamp that locates its run checkpoint."""
    arch_dir = tmp_path / "efficientnet_b0"
    saved_dir = arch_dir / "saved_model_20261105_151511"
    saved_dir.mkdir(parents=True)
    (saved_dir / "saved_model.pb").write_bytes(b"")
    run_dir = arch_dir / "runs" / "20261105_151511"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "best_model.weights.h5").write_bytes(b"")
    (run_dir / "resolved_config.yaml").write_text(
        "model:\n  architecture: efficientnet_b0\n"
    )

    checkpoint, run_config = post_training_quant._find_run_checkpoint(str(saved_dir))

    assert checkpoint == str(run_dir / "checkpoints" / "best_model.weights.h5")
    assert run_config == str(run_dir / "resolved_config.yaml")


def test_find_run_checkpoint_without_checkpoint_returns_none(tmp_path):
    """Exports predating run-scoped checkpointing must keep the legacy path."""
    saved_dir = tmp_path / "efficientnet_b0" / "saved_model_20261105_151511"
    saved_dir.mkdir(parents=True)
    (saved_dir / "saved_model.pb").write_bytes(b"")

    checkpoint, run_config = post_training_quant._find_run_checkpoint(str(saved_dir))

    assert checkpoint is None
    assert run_config is None


def test_convert_savedmodel_uses_float32_rebuild_from_run_checkpoint(
    monkeypatch, tmp_path, capsys
):
    """A SavedModel backed by a run checkpoint converts through a float32 rebuild.

    Converting a SavedModel exported from a mixed_float16 training model fails
    inside the TFLite converter. The run checkpoint holds the identical final
    weights, so the script rebuilds a float32 model and converts that instead —
    the same path training/export.py already proves on hosted runtimes.
    """
    import tensorflow as tf

    from tensorflow import keras

    num_classes = 4
    input_shape = (32, 32, 3)

    def build_model(num_classes=num_classes, input_shape=input_shape, **kwargs):
        layers = [
            keras.Input(shape=input_shape),
            keras.layers.Conv2D(4, 3, activation="relu"),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
        model = keras.Sequential(layers)
        model.compile(optimizer="adam", loss="categorical_crossentropy")
        return model, None

    arch_dir = tmp_path / "efficientnet_b0"
    run_dir = arch_dir / "runs" / "20261105_151511"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "resolved_config.yaml").write_text(
        "model:\n"
        "  architecture: efficientnet_b0\n"
        f"  input_shape: [{input_shape[0]}, {input_shape[1]}, 3]\n"
        f"  num_classes: {num_classes}\n"
        "  dropout_rate: 0.2\n"
        "  weights: null\n"
    )

    trained = build_model()[0]
    checkpoint_path = str(run_dir / "checkpoints" / "best_model.weights.h5")
    trained.save_weights(checkpoint_path)

    saved_dir = arch_dir / "saved_model_20261105_151511"
    trained.export(str(saved_dir))

    monkeypatch.setattr(post_training_quant, "_get_model_factory", lambda: build_model)

    def fail_from_saved_model(*args, **kwargs):
        raise AssertionError("from_saved_model must not be used when a checkpoint exists")

    monkeypatch.setattr(
        tf.lite.TFLiteConverter, "from_saved_model", fail_from_saved_model
    )

    output_path = tmp_path / "out" / "model.tflite"
    result = post_training_quant.convert_to_tflite(
        str(saved_dir), str(output_path), quantize=False,
        architecture="efficientnet_b0",
    )

    stdout = capsys.readouterr().out
    assert "Rebuilding float32 model from checkpoint" in stdout
    assert "Calibrating at model input size: 32x32" in stdout
    assert result == str(output_path)
    assert output_path.is_file() and output_path.stat().st_size > 0

    interpreter = tf.lite.Interpreter(model_path=str(output_path))
    interpreter.allocate_tensors()
    details = interpreter.get_input_details()[0]
    assert tuple(details["shape"]) == (1, input_shape[0], input_shape[1], 3)


def test_convert_failure_prints_traceback(monkeypatch, tmp_path, capsys):
    """A converter failure must print its traceback for the parent process."""
    import tensorflow as tf

    from tensorflow import keras

    model_path = tmp_path / "model.h5"
    keras.Sequential(
        [keras.Input(shape=(8, 8, 3)), keras.layers.Flatten(), keras.layers.Dense(2)]
    ).save(str(model_path))

    class BrokenConverter:
        def __init__(self):
            self.optimizations = []
            self.representative_dataset = None
            self.target_spec = type("Spec", (), {})()

        def convert(self):
            raise RuntimeError("converter exploded")

    monkeypatch.setattr(
        tf.lite.TFLiteConverter, "from_keras_model", lambda model: BrokenConverter()
    )

    with pytest.raises(RuntimeError, match="converter exploded"):
        post_training_quant.convert_to_tflite(
            str(model_path), str(tmp_path / "out.tflite"), quantize=False
        )

    stderr = capsys.readouterr().err
    assert "TFLite conversion failed" in stderr
    assert "converter exploded" in stderr
