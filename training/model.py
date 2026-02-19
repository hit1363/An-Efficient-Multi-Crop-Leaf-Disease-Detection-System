"""
Model Architecture Definition
Defines MobileNetV2 and EfficientNet-Lite0 models for leaf disease detection
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0


def create_mobilenetv2_model(
    input_shape=(224, 224, 3), num_classes=30, dropout_rate=0.5, weights="imagenet"
):
    """
    Create MobileNetV2-based model for leaf disease classification

    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of disease classes
        dropout_rate: Dropout rate for regularization
        weights: Pre-trained weights ('imagenet' or None)

    Returns:
        Keras Model instance
    """
    # Load base model
    base_model = MobileNetV2(
        input_shape=input_shape, include_top=False, weights=weights
    )

    # Freeze base model layers initially
    base_model.trainable = False

    # Build classification head
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation="relu", name="fc2")(x)
    x = layers.Dropout(dropout_rate * 0.6)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name="mobilenetv2_disease_detector")

    return model, base_model


def create_efficientnet_model(
    input_shape=(224, 224, 3), num_classes=30, dropout_rate=0.5, weights="imagenet"
):
    """
    Create EfficientNet-Lite0 based model for leaf disease classification

    Args:
        input_shape: Input image shape
        num_classes: Number of disease classes
        dropout_rate: Dropout rate
        weights: Pre-trained weights

    Returns:
        Keras Model instance
    """
    # Load base model (using EfficientNetB0 as proxy for Lite0)
    base_model = EfficientNetB0(
        input_shape=input_shape, include_top=False, weights=weights
    )

    base_model.trainable = False

    # Classification head
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate * 0.6)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="efficientnet_disease_detector")

    return model, base_model


def unfreeze_base_model(base_model, unfreeze_from_layer=100):
    """
    Unfreeze layers of base model for fine-tuning

    Args:
        base_model: Base model to unfreeze
        unfreeze_from_layer: Layer index to start unfreezing from
    """
    base_model.trainable = True

    # Freeze initial layers, unfreeze later layers
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
    Factory function to get model by name

    Args:
        architecture: Model architecture name
        **kwargs: Additional arguments for model creation

    Returns:
        Model and base_model instances
    """
    if architecture.lower() == "mobilenetv2":
        return create_mobilenetv2_model(**kwargs)
    elif architecture.lower() in ["efficientnet", "efficientnet_lite0"]:
        return create_efficientnet_model(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


if __name__ == "__main__":
    # Test model creation
    print("Creating MobileNetV2 model...")
    model, base = create_mobilenetv2_model(num_classes=30)
    print_model_summary(model)

    print("\n" + "=" * 50 + "\n")

    print("Creating EfficientNet model...")
    model2, base2 = create_efficientnet_model(num_classes=30)
    print_model_summary(model2)
