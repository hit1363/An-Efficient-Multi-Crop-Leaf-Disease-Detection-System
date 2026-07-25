"""Quantization-aware fine-tuning and full-int8 TFLite export."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse
import json
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.model import get_model
from training.train import _get_loss, build_metrics, resolve_config_paths
from training.utils import (
    compute_class_weights,
    create_augmentation_layer,
    get_preprocess_fn,
    load_dataset,
)


SUPPORTED_ARCHITECTURES = {"mobilenetv2", "efficientnet_b0"}


def _load_tfmot():
    try:
        import tensorflow_model_optimization as tfmot
    except Exception as exc:
        raise RuntimeError(
            "QAT requires tensorflow-model-optimization. Install it in Kaggle "
            "with `%pip install -q tensorflow-model-optimization`, then restart "
            "the kernel if the import still fails."
        ) from exc

    if not hasattr(tfmot, "quantization") or not hasattr(
        tfmot.quantization, "keras"
    ):
        raise RuntimeError("The installed tensorflow-model-optimization package has no Keras QAT API.")
    return tfmot


def _load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_model(config):
    model_config = config["model"]
    return get_model(
        architecture=model_config["architecture"],
        input_shape=tuple(model_config["input_shape"]),
        num_classes=int(model_config["num_classes"]),
        dropout_rate=float(model_config.get("dropout_rate", 0.5)),
        weights=model_config.get("weights", "imagenet"),
        hub_url=model_config.get("hub_url"),
        hub_cache_dir=model_config.get("hub_cache_dir"),
        hub_download_retries=model_config.get("hub_download_retries", 1),
        hub_download_delay_sec=model_config.get("hub_download_delay_sec", 5),
    )[0]


def _load_float_model(model_path, config):
    if model_path.lower().endswith(".weights.h5"):
        model = _build_model(config)
        model.load_weights(model_path)
        return model

    try:
        return keras.models.load_model(model_path)
    except Exception as exc:
        if os.path.isdir(model_path):
            raise RuntimeError(
                "The SavedModel could not be loaded as a trainable Keras model. "
                "Pass the best_model.weights.h5 checkpoint to QAT instead."
            ) from exc
        raise


def _iter_layers(model):
    for layer in model.layers:
        yield layer
        if isinstance(layer, keras.Model):
            yield from _iter_layers(layer)


def _configure_qat_trainability(model, config):
    training_config = config.get("training", {})
    unfreeze_from = training_config.get("unfreeze_from_layer")
    if unfreeze_from is None:
        unfreeze_from = training_config.get("freeze_until_layer", 100)
    unfreeze_from = int(unfreeze_from)

    nested_models = [layer for layer in model.layers if isinstance(layer, keras.Model)]
    for base_model in nested_models:
        base_model.trainable = True
        for index, layer in enumerate(base_model.layers):
            layer.trainable = index >= unfreeze_from

    for layer in _iter_layers(model):
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False


def _selective_quantize(model, tfmot):
    quantize_layer = tfmot.quantization.keras.quantize_annotate_layer
    quantize_apply = tfmot.quantization.keras.quantize_apply
    quantizable_types = (
        keras.layers.Conv2D,
        keras.layers.DepthwiseConv2D,
        keras.layers.Dense,
    )

    def annotate(layer):
        if isinstance(layer, quantizable_types):
            return quantize_layer(layer)
        return layer

    annotated_model = keras.models.clone_model(model, clone_function=annotate)
    annotated_model.set_weights(model.get_weights())
    return quantize_apply(annotated_model)


def build_qat_model(model, tfmot, mode="auto"):
    """Build a QAT model, falling back to selective layer annotation."""
    errors = []
    if mode in {"auto", "full"}:
        try:
            return tfmot.quantization.keras.quantize_model(model), "full"
        except Exception as exc:
            errors.append(f"full-model QAT failed: {exc}")
            if mode == "full":
                raise RuntimeError("\n".join(errors)) from exc

    try:
        return _selective_quantize(model, tfmot), "selective"
    except Exception as exc:
        errors.append(f"selective QAT failed: {exc}")
        raise RuntimeError("Unable to construct a QAT model.\n" + "\n".join(errors)) from exc


def _representative_dataset(data_dir, image_size, preprocess_fn, max_samples):
    dataset = keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=image_size,
        batch_size=1,
        shuffle=False,
        label_mode="int",
    )

    for index, (images, _) in enumerate(dataset):
        if index >= max_samples:
            break
        images = tf.cast(images, tf.float32)
        if preprocess_fn is not None:
            images = preprocess_fn(images)
        yield [images]


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _export_full_int8(model, config, representative_data, output_path, max_samples):
    model_config = config["model"]
    preprocess_fn = get_preprocess_fn(model_config["architecture"])
    image_size = tuple(model_config["input_shape"][:2])

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative_dataset(
        representative_data,
        image_size,
        preprocess_fn,
        max_samples,
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as handle:
        handle.write(tflite_model)

    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    if input_details["dtype"] != np.uint8 or output_details["dtype"] != np.uint8:
        raise RuntimeError(
            "QAT export did not produce uint8 input/output tensors: "
            f"{input_details['dtype']} -> {output_details['dtype']}"
        )

    return {
        "path": os.path.abspath(output_path),
        "size_mb": os.path.getsize(output_path) / (1024 * 1024),
        "input_dtype": np.dtype(input_details["dtype"]).name,
        "output_dtype": np.dtype(output_details["dtype"]).name,
        "input_quantization": list(input_details["quantization"]),
        "output_quantization": list(output_details["quantization"]),
    }


def run_qat(
    model_path,
    config_path,
    output_dir,
    representative_data,
    epochs=8,
    learning_rate=1e-5,
    qat_mode="auto",
    representative_samples=1000,
):
    config = resolve_config_paths(_load_config(config_path), config_path)
    architecture = config["model"]["architecture"].lower()
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"QAT supports only {sorted(SUPPORTED_ARCHITECTURES)}; received {architecture!r}."
        )
    if not os.path.isdir(representative_data):
        raise FileNotFoundError(f"Representative dataset not found: {representative_data}")

    tfmot = _load_tfmot()
    model = _load_float_model(model_path, config)
    qat_model, applied_mode = build_qat_model(model, tfmot, mode=qat_mode)
    _configure_qat_trainability(qat_model, config)

    preprocess_fn = get_preprocess_fn(architecture)
    augmentation = create_augmentation_layer(config) if config.get("augmentation") else None
    train_ds, val_ds, class_names = load_dataset(
        config["dataset"]["train_dir"],
        config["dataset"]["val_dir"],
        batch_size=config["dataset"].get("batch_size", 32),
        image_size=tuple(config["model"]["input_shape"][:2]),
        preprocess_fn=preprocess_fn,
        augmentation=augmentation,
        shuffle_buffer=config["dataset"].get("shuffle_buffer", 1000),
        cache_mode=config["dataset"].get("cache_mode", "none"),
    )
    if len(class_names) != int(config["model"]["num_classes"]):
        raise ValueError("Dataset class count does not match the QAT configuration.")

    class_weights = None
    if config.get("class_weights", {}).get("enabled", False):
        class_weights = compute_class_weights(config["dataset"]["train_dir"])

    qat_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=_get_loss(config),
        metrics=build_metrics(
            config.get("metrics", ["accuracy"]),
            int(config["model"]["num_classes"]),
        ),
    )

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "qat_best.weights.h5")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=max(2, min(5, epochs // 2)),
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=learning_rate / 100,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(os.path.join(output_dir, "qat_training_log.csv")),
    ]

    history = qat_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(epochs),
        callbacks=callbacks,
        class_weight=class_weights,
    )

    qat_model.load_weights(checkpoint_path)
    stripped_model = tfmot.quantization.keras.strip_quantization(qat_model)
    stripped_path = os.path.join(output_dir, "qat_stripped.h5")
    stripped_model.save(stripped_path)

    tflite_path = os.path.join(output_dir, "qat_int8.tflite")
    export_info = _export_full_int8(
        stripped_model,
        config,
        representative_data,
        tflite_path,
        representative_samples,
    )
    summary = {
        "architecture": architecture,
        "qat_mode": applied_mode,
        "source_model": os.path.abspath(model_path),
        "qat_model": os.path.abspath(stripped_path),
        "epochs_requested": int(epochs),
        "epochs_completed": len(history.history.get("loss", [])),
        "learning_rate": float(learning_rate),
        "class_weighting_enabled": class_weights is not None,
        "final_val_accuracy": float(history.history.get("val_accuracy", [0])[-1]),
        "best_val_accuracy": float(max(history.history.get("val_accuracy", [0]))),
        "export": export_info,
        "created_at": datetime.now().isoformat(),
    }
    _write_json(os.path.join(output_dir, "qat_summary.json"), summary)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model with quantization-aware training")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--representative_data", required=True)
    parser.add_argument("--arch", default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--qat_mode", choices=["auto", "full", "selective"], default="auto")
    parser.add_argument("--representative_samples", type=int, default=1000)
    args = parser.parse_args()

    config = _load_config(args.config)
    configured_architecture = config.get("model", {}).get("architecture")
    if args.arch and args.arch != configured_architecture:
        raise ValueError("--arch must match model.architecture in --config.")

    run_qat(
        args.model_path,
        args.config,
        args.output_dir,
        args.representative_data,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        qat_mode=args.qat_mode,
        representative_samples=args.representative_samples,
    )


if __name__ == "__main__":
    main()
