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
