"""
Model Architecture Definition
Defines MobileNetV2 and EfficientNet-Lite0 models for leaf disease detection
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import shutil
import tempfile
import time

import tensorflow as tf


_HUB_MODULE = None
_HUB_IMPORT_ERROR = None


def _get_tfhub_module():
    """Lazily import tensorflow_hub only when EfficientNet-Lite0 is requested."""
    global _HUB_MODULE, _HUB_IMPORT_ERROR

    if _HUB_MODULE is not None:
        return _HUB_MODULE

    if _HUB_IMPORT_ERROR is not None:
        raise ImportError(
            "tensorflow_hub is required for EfficientNet-Lite0. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from _HUB_IMPORT_ERROR

    try:
        import tensorflow_hub as hub  # type: ignore

        _HUB_MODULE = hub
        return _HUB_MODULE
    except Exception as exc:
        _HUB_IMPORT_ERROR = exc
        raise ImportError(
            "tensorflow_hub is required for EfficientNet-Lite0. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc


def _is_tfhub_cache_error(exc):
    """Heuristics to detect corrupted TF Hub downloads."""
    message = str(exc)
    return any(
        token in message
        for token in [
            "does not appear to be a valid module",
            "invalid header",
            "ReadError",
        ]
    )


def _clear_tfhub_cache(cache_dir):
    """Best-effort cleanup for TF Hub cache directory."""
    if not cache_dir:
        return
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)


def _ensure_tfhub_cache_dir(cache_dir):
    """Ensure TF Hub cache directory exists and is writable."""
    if not cache_dir:
        return None
    try:
        os.makedirs(cache_dir, exist_ok=True)
        test_file = os.path.join(cache_dir, ".tfhub_write_test")
        with open(test_file, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(test_file)
        return cache_dir
    except OSError:
        return None


def _resolve_tfhub_cache_dir(hub_cache_dir):
    """Pick a valid TF Hub cache directory across common environments."""
    candidates = []

    if hub_cache_dir:
        candidates.append(hub_cache_dir)

    env_dir = os.environ.get("TFHUB_CACHE_DIR")
    if env_dir and env_dir not in candidates:
        candidates.append(env_dir)

    if os.path.isdir("/content"):
        candidates.append(os.path.join("/content", "tfhub_modules"))

    if os.path.isdir("/kaggle/working"):
        candidates.append(os.path.join("/kaggle/working", "tfhub_modules"))

    candidates.append(os.path.join(tempfile.gettempdir(), "tfhub_modules"))
    candidates.append(os.path.join(os.path.expanduser("~"), ".tfhub_modules"))

    for candidate in candidates:
        resolved = _ensure_tfhub_cache_dir(candidate)
        if resolved:
            return resolved

    return None


def create_mobilenetv2_model(
    input_shape=(224, 224, 3), num_classes=45, dropout_rate=0.5, weights="imagenet"
):
    """
    Create MobileNetV2-based model for leaf disease classification

    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of disease classes (default 45 for multi-crop dataset)
        dropout_rate: Dropout rate for regularization
        weights: Pre-trained weights ('imagenet' for ImageNet pretrained, None for random init)

    Returns:
        Tuple of (model, base_model):
            - model: Full classifier with frozen MobileNetV2 backbone + Dense head
            - base_model: MobileNetV2 Keras model (used for unfreezing during fine-tuning)
    """
    # Load base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights=weights
    )

    # Freeze base model layers initially
    base_model.trainable = False

    # Build classification head
    # MobileNetV2 outputs (batch, 7, 7, 1280), so we use GlobalAveragePooling2D
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(
        x
    )  # Converts (batch, H, W, C) -> (batch, C)
    x = tf.keras.layers.Dense(512, activation="relu", name="fc1")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="fc2")(x)
    x = tf.keras.layers.Dropout(dropout_rate * 0.6)(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="predictions"
    )(x)

    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_disease_detector")

    return model, base_model  # base_model is a standard Keras Model


def create_efficientnet_model(
    input_shape=(224, 224, 3),
    num_classes=45,
    dropout_rate=0.5,
    weights="imagenet",
    hub_url=None,
    hub_cache_dir=None,
    hub_download_retries=1,
    hub_download_delay_sec=5,
):
    """
    Create EfficientNet-Lite0 based model for leaf disease classification

    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of disease classes (default 45 for multi-crop dataset)
        dropout_rate: Dropout rate for regularization
        weights: Pre-trained weights (note: Lite0 always uses TensorFlow Hub pretrained;
             parameter provided for API compatibility but is ignored)
        hub_url: Optional TF Hub module URL override
        hub_cache_dir: Optional TF Hub cache directory override
        hub_download_retries: Total attempts to download/load the Hub module
        hub_download_delay_sec: Delay between retries in seconds

    Returns:
        Tuple of (model, base_model):
            - model: Full classifier with frozen EfficientNet-Lite0 backbone + Dense head
            - base_model: hub.KerasLayer wrapping Lite0 feature-vector endpoint
    """
    # Load official EfficientNetLite0 from TensorFlow Hub
    # Pre-trained on ImageNet, optimized for mobile devices
    # Note: weights parameter is mostly ignored; Hub always provides pretrained ImageNet weights
    hub = _get_tfhub_module()
    hub_url = hub_url or "https://tfhub.dev/google/efficientnet/lite0/feature-vector/2"

    cache_dir = _resolve_tfhub_cache_dir(hub_cache_dir)
    if cache_dir:
        os.environ["TFHUB_CACHE_DIR"] = cache_dir

    attempts = max(1, int(hub_download_retries or 1))
    base_model = None
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            base_model = hub.KerasLayer(
                hub_url, input_shape=input_shape, trainable=False
            )
            break
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                raise
            if _is_tfhub_cache_error(exc):
                if not cache_dir:
                    cache_dir = _resolve_tfhub_cache_dir(None)
                    if cache_dir:
                        os.environ["TFHUB_CACHE_DIR"] = cache_dir
                if cache_dir:
                    _clear_tfhub_cache(cache_dir)
                    _ensure_tfhub_cache_dir(cache_dir)
            if hub_download_delay_sec and hub_download_delay_sec > 0:
                time.sleep(hub_download_delay_sec)

    if base_model is None:
        raise last_exc or RuntimeError("Failed to load TF Hub module")

    # Classification head
    # Note: Hub feature-vector endpoint outputs (batch, 1280), already aggregated
    # So we do NOT use GlobalAveragePooling2D (expects 4D input, we have 1D)
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)  # Output: (batch, 1280)
    x = tf.keras.layers.Dense(512, activation="relu")(
        x
    )  # Feed 1D output directly to Dense
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate * 0.6)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="efficientnet_lite0_disease_detector")

    return model, base_model  # base_model is a hub.KerasLayer wrapper


def unfreeze_base_model(base_model, unfreeze_from_layer=100):
    """
    Unfreeze layers of base model for fine-tuning

    Args:
        base_model: Base model to unfreeze. Can be:
            - MobileNetV2: Standard Keras Model (layers will be selectively unfrozen)
            - EfficientNet-Lite0: hub.KerasLayer (all-or-nothing; entire layer unfrozen)
        unfreeze_from_layer: Layer index to start unfreezing from (only for Keras Models)
    """
    # Check if this is a TensorFlow Hub KerasLayer
    if base_model.__class__.__module__.startswith("tensorflow_hub"):
        print("TensorFlow Hub layer detected. Enabling fine-tuning...")
        base_model.trainable = True
        print("Hub layer trainable: True")
        return

    # Standard Keras Model: freeze initial layers, unfreeze later layers
    base_model.trainable = True

    for layer in base_model.layers[:unfreeze_from_layer]:
        layer.trainable = False

    for layer in base_model.layers[unfreeze_from_layer:]:
        layer.trainable = True

    print(f"Unfroze {len(base_model.layers[unfreeze_from_layer:])} layers")
    print(
        f"Total trainable layers: {sum([layer.trainable for layer in base_model.layers])}"
    )


def print_model_summary(model):
    """Print model architecture summary"""
    model.summary()

    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")


def get_model(architecture="mobilenetv2", **kwargs):
    """
    Factory function to get model by architecture name

    Args:
        architecture: Model architecture name. Options:
            - 'mobilenetv2': MobileNetV2 backbone (full layer-level control)
            - 'efficientnet' or 'efficientnet_lite0': EfficientNet-Lite0 from TF Hub
        **kwargs: Additional arguments passed to model creation functions:
            - input_shape: tuple (default (224, 224, 3))
            - num_classes: int (default 45)
            - dropout_rate: float (default 0.5)
            - weights: str (default 'imagenet', ignored for Lite0)
            - hub_url: str (optional TF Hub URL override)
            - hub_cache_dir: str (optional TF Hub cache directory override)
            - hub_download_retries: int (total attempts)
            - hub_download_delay_sec: int or float (delay between attempts)

    Returns:
        Tuple of (model, base_model) where both are unfrozen for initial training

    Raises:
        ValueError: If architecture is not recognized
    """
    arch_lower = architecture.lower()
    if arch_lower == "mobilenetv2":
        kwargs = dict(kwargs)
        for key in [
            "hub_url",
            "hub_cache_dir",
            "hub_download_retries",
            "hub_download_delay_sec",
        ]:
            kwargs.pop(key, None)
        return create_mobilenetv2_model(**kwargs)
    elif arch_lower in ["efficientnet", "efficientnet_lite0"]:
        return create_efficientnet_model(**kwargs)
    else:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Supported: 'mobilenetv2', 'efficientnet', 'efficientnet_lite0'"
        )


if __name__ == "__main__":
    # Test model creation
    print("Creating MobileNetV2 model...")
    model, base = create_mobilenetv2_model(num_classes=45)
    print_model_summary(model)

    print("\n" + "=" * 50 + "\n")

    print("Creating EfficientNet-Lite0 model...")
    model2, base2 = create_efficientnet_model(num_classes=45)
    print_model_summary(model2)
