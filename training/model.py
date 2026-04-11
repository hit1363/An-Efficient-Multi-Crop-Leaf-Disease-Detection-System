"""
Model Architecture Definition
Defines MobileNetV2 and EfficientNet-Lite0 models for leaf disease detection
"""

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
    input_shape=(224, 224, 3), num_classes=45, dropout_rate=0.5, weights="imagenet"
):
    """
    Create EfficientNet-Lite0 based model for leaf disease classification

    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of disease classes (default 45 for multi-crop dataset)
        dropout_rate: Dropout rate for regularization
        weights: Pre-trained weights (note: Lite0 always uses TensorFlow Hub pretrained;
                 parameter provided for API compatibility but is ignored)

    Returns:
        Tuple of (model, base_model):
            - model: Full classifier with frozen EfficientNet-Lite0 backbone + Dense head
            - base_model: hub.KerasLayer wrapping Lite0 feature-vector endpoint
    """
    # Load official EfficientNetLite0 from TensorFlow Hub
    # Pre-trained on ImageNet, optimized for mobile devices
    # Note: weights parameter is mostly ignored; Hub always provides pretrained ImageNet weights
    hub = _get_tfhub_module()
    hub_url = "https://tfhub.dev/google/efficientnet/lite0/feature-vector/2"

    base_model = hub.KerasLayer(hub_url, input_shape=input_shape, trainable=False)

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

    Returns:
        Tuple of (model, base_model) where both are unfrozen for initial training

    Raises:
        ValueError: If architecture is not recognized
    """
    arch_lower = architecture.lower()
    if arch_lower == "mobilenetv2":
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
