"""
Model Architecture Definition
Defines the supported mobile backbones for leaf disease detection.
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
    """Heuristics to detect corrupted TF Hub downloads and transient load failures.

    A broad match is intentional: anything that looks like a corrupt cache,
    a network/resolver issue, or a server-side error is worth clearing the
    cache and retrying once before giving up.
    """
    message = str(exc).lower()
    tokens = [
        # Corrupt / invalid module
        "does not appear to be a valid module",
        "invalid header",
        "readerror",
        "bad signature",
        "not a valid tar",
        # Network / resolver
        "404",
        "http",
        "https",
        "url",
        "failed",
        "ssl",
        "timeout",
        "timed out",
        "connection",
        "resolver",
        "urLError".lower(),
        "no such file",  # partially downloaded cache
    ]
    return any(token in message for token in tokens)


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


def _tfhub_gcs_url(hub_url):
    """Map a tfhub.dev / kaggle.com URL to its canonical GCS mirror.

    TF Hub serves every published module as a tarball on
    ``https://storage.googleapis.com/tfhub-modules/<publisher>/<name>/<version>.tar.gz``.
    Downloading from GCS directly avoids the tfhub.dev redirect/resolver which
    is the common point of failure for TF1-format modules (like Lite0).
    Returns ``None`` if the URL cannot be mapped.
    """
    if not hub_url:
        return None
    url = hub_url.split("?")[0].rstrip("/")
    # Canonical tfhub.dev URL: https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2
    for prefix in ("https://tfhub.dev/", "http://tfhub.dev/"):
        if url.lower().startswith(prefix):
            path = url[len(prefix):].strip("/")
            return f"https://storage.googleapis.com/tfhub-modules/{path}"
    # Already a GCS / kaggle URL we can't reliably map.
    return None


def _resolve_module_handle(hub_url, hub_cache_dir):
    """Return a local-or-remote handle that ``hub.KerasLayer`` can consume.

    Resolution order (first usable wins):
      1. ``hub_url`` already points at an existing local path -> use as-is.
      2. Direct GCS download via ``tf.keras.utils.get_file`` into the TF Hub
         cache dir -> return the extracted local path.
      3. Fall back to the original ``hub_url`` (let hub resolve it online).

    Returns ``(handle, is_local_path)``.
    """
    import urllib.parse as _up

    # 1. Local path provided directly (downloaded/vendored module).
    if hub_url and os.path.exists(hub_url):
        return hub_url, True

    # 2. Try a direct GCS download; bypasses the tfhub.dev resolver entirely.
    cache_dir = _resolve_tfhub_cache_dir(hub_cache_dir)
    if cache_dir:
        os.environ["TFHUB_CACHE_DIR"] = cache_dir
        gcs_url = _tfhub_gcs_url(hub_url)
        if gcs_url:
            # Stable sub-dir per module so re-runs reuse the download.
            parsed = _up.urlparse(gcs_url)
            module_subdir = parsed.path.strip("/").replace("/", "_")
            local_dir = os.path.join(cache_dir, module_subdir)
            # The GCS mirror is an archive endpoint. Keep ``gcs_url`` without
            # the suffix for stable cache naming, but request the actual tarball.
            archive_url = (
                gcs_url if gcs_url.lower().endswith(".tar.gz") else f"{gcs_url}.tar.gz"
            )
            try:
                downloaded = tf.keras.utils.get_file(
                    fname=module_subdir + ".tar.gz",
                    origin=archive_url,
                    extract=True,
                    cache_dir=cache_dir,
                    cache_subdir="downloads",
                )
                # ``get_file(extract=True)`` returns the archive path on some
                # Keras versions and the extracted directory on others. Find
                # the actual SavedModel marker instead of guessing the folder
                # name (``os.path.splitext`` only strips ``.gz`` from .tar.gz).
                search_root = (
                    downloaded if os.path.isdir(downloaded) else os.path.dirname(downloaded)
                )
                if os.path.isdir(search_root):
                    for root, _dirs, files in os.walk(search_root):
                        if "saved_model.pb" in files or "saved_model.pbtxt" in files:
                            return root, True
            except Exception as exc:
                print(
                    f"[model] Direct GCS download failed ({exc}); "
                    "falling back to TF Hub online resolver."
                )

    # 3. Fall back to the original URL (hub resolves it online w/ cache).
    return hub_url, False


def _create_tfhub_layer(hub_url, hub_cache_dir, input_shape):
    """Create a hub.KerasLayer, trying local/GCS resolution first.

    Tries the handle directly first (works if hub can resolve it online),
    then a locally-resolved handle. Returns the ``hub.KerasLayer``.
    """
    hub = _get_tfhub_module()

    # Preferred path: resolve to a local/GCS-downloaded module handle.
    handle, is_local = _resolve_module_handle(hub_url, hub_cache_dir)
    try:
        layer = hub.KerasLayer(handle, input_shape=input_shape, trainable=False)
        return layer
    except Exception:
        # If local resolution already happened and still failed, nothing more
        # to try. Otherwise fall back to the raw hub URL as a last resort.
        if is_local:
            raise
        return hub.KerasLayer(hub_url, input_shape=input_shape, trainable=False)


def _sanity_check_layer(layer, input_shape):
    """Run one forward pass on random input to catch a corrupt module early.

    Raises only if the layer cannot produce output for a dummy batch, so we
    fail in seconds rather than partway through ``model.fit()``.
    """
    try:
        dummy = tf.random.normal((1,) + tuple(input_shape))
        _ = layer(dummy, training=False)
    except Exception as exc:
        raise RuntimeError(
            "EfficientNet-Lite0 backbone failed its forward-pass sanity check. "
            "The downloaded module may be corrupt. Clear the TF Hub cache "
            f"(TFHUB_CACHE_DIR={os.environ.get('TFHUB_CACHE_DIR')}) and retry. "
            f"Underlying error: {exc}"
        ) from exc


def create_efficientnet_b0_model(
    input_shape,
    num_classes,
    dropout_rate,
    weights="imagenet",
):
    """Build a standard Keras EfficientNetB0 classifier."""
    print(
        "[model] Using tf.keras.applications.EfficientNetB0 "
        "with ImageNet weights."
    )
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate * 0.6)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="efficientnet_b0_disease_detector")
    return model, base_model


def create_mobilenetv2_model(
    input_shape=(224, 224, 3),
    num_classes=45,
    dropout_rate=0.5,
    weights="imagenet",
    classifier_units=256,
):
    """
    Create MobileNetV2-based model for leaf disease classification

    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of disease classes (default 45 for multi-crop dataset)
        dropout_rate: Dropout rate for regularization
        classifier_units: Hidden units in the lightweight classifier head. Set to
            0 to use a linear classifier on pooled backbone features.
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
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    # Normalize pooled ImageNet features to make the classifier head easier to
    # optimize during both frozen-backbone training and fine-tuning.
    x = tf.keras.layers.BatchNormalization(name="classifier_bn")(x)
    if classifier_units and int(classifier_units) > 0:
        x = tf.keras.layers.Dense(
            int(classifier_units), activation="relu", name="classifier_dense"
        )(x)
        x = tf.keras.layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    else:
        x = tf.keras.layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", dtype="float32", name="predictions"
    )(x)

    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_disease_detector")

    return model, base_model  # base_model is a standard Keras Model


def _attach_classifier_head(
    base_model,
    input_shape,
    num_classes,
    dropout_rate=0.35,
    classifier_units=256,
    model_name="classifier",
):
    """Attach the shared lightweight classifier head to a feature backbone."""
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = base_model(inputs, training=False)
    if len(x.shape) == 4:
        x = tf.keras.layers.GlobalAveragePooling2D(
            name="global_average_pooling"
        )(x)
    x = tf.keras.layers.BatchNormalization(name="classifier_bn")(x)
    if classifier_units and int(classifier_units) > 0:
        x = tf.keras.layers.Dense(
            int(classifier_units), activation="relu", name="classifier_dense"
        )(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    outputs = tf.keras.layers.Dense(
        int(num_classes),
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)
    return tf.keras.Model(inputs, outputs, name=model_name)


def create_mobilenetv3_small_model(
    input_shape=(224, 224, 3),
    num_classes=45,
    dropout_rate=0.35,
    weights="imagenet",
    classifier_units=256,
    include_preprocessing=False,
):
    """Create a MobileNetV3-Small classifier with optional ImageNet weights."""
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
        include_preprocessing=include_preprocessing,
    )
    base_model.trainable = False
    model = _attach_classifier_head(
        base_model,
        input_shape,
        num_classes,
        dropout_rate=dropout_rate,
        classifier_units=classifier_units,
        model_name="mobilenetv3_small_disease_detector",
    )
    return model, base_model


@tf.keras.utils.register_keras_serializable(package="LeafDisease")
class ChannelShuffle(tf.keras.layers.Layer):
    """Channel shuffle operation used by ShuffleNetV2."""

    def __init__(self, groups=2, **kwargs):
        super().__init__(**kwargs)
        self.groups = int(groups)

    def call(self, inputs):
        shape = tf.shape(inputs)
        channels = inputs.shape[-1]
        if channels is None or channels % self.groups != 0:
            raise ValueError("Channel count must be divisible by groups.")
        x = tf.reshape(
            inputs,
            [shape[0], shape[1], shape[2], self.groups, channels // self.groups],
        )
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        return tf.reshape(x, [shape[0], shape[1], shape[2], channels])

    def get_config(self):
        config = super().get_config()
        config.update({"groups": self.groups})
        return config


@tf.keras.utils.register_keras_serializable(package="LeafDisease")
class ChannelSplit(tf.keras.layers.Layer):
    """Serializable two-way channel split used by stride-1 units."""

    def call(self, inputs):
        return tf.split(inputs, 2, axis=-1)


def _conv_bn_relu(x, filters, kernel_size, strides=1, name="conv"):
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"{name}_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    return tf.keras.layers.ReLU(name=f"{name}_relu")(x)


def _shufflenet_v2_unit(x, output_channels, stride, name):
    """Build one ShuffleNetV2 basic unit."""
    branch_channels = output_channels // 2
    if stride == 1:
        input_channels = x.shape[-1]
        if input_channels is None or input_channels % 2 != 0:
            raise ValueError("Stride-1 ShuffleNetV2 units require even channels.")
        branch_left, branch_right = ChannelSplit(name=f"{name}_split")(x)
        projection = branch_left
    else:
        projection = tf.keras.layers.DepthwiseConv2D(
            3, strides=2, padding="same", use_bias=False, name=f"{name}_proj_dw"
        )(x)
        projection = tf.keras.layers.BatchNormalization(name=f"{name}_proj_dw_bn")(
            projection
        )
        projection = _conv_bn_relu(
            projection, branch_channels, 1, name=f"{name}_proj_pw"
        )
        branch_right = x

    branch_right = _conv_bn_relu(
        branch_right, branch_channels, 1, name=f"{name}_main_pw1"
    )
    branch_right = tf.keras.layers.DepthwiseConv2D(
        3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_main_dw",
    )(branch_right)
    branch_right = tf.keras.layers.BatchNormalization(name=f"{name}_main_dw_bn")(
        branch_right
    )
    branch_right = _conv_bn_relu(
        branch_right, branch_channels, 1, name=f"{name}_main_pw2"
    )
    x = tf.keras.layers.Concatenate(axis=-1, name=f"{name}_concat")(
        [projection, branch_right]
    )
    return ChannelShuffle(name=f"{name}_shuffle")(x)


def create_shufflenetv2_05_model(
    input_shape=(224, 224, 3),
    num_classes=45,
    dropout_rate=0.35,
    weights=None,
    classifier_units=256,
):
    """Create a native ShuffleNetV2 0.5x classifier.

    This implementation intentionally trains from scratch because the project
    does not add a third-party or PyTorch weight-conversion dependency.
    """
    if weights not in (None, "none"):
        raise ValueError(
            "ShuffleNetV2 0.5x supports weights=None only in the native Keras implementation."
        )

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = _conv_bn_relu(inputs, 24, 3, strides=2, name="stem")
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same", name="stem_pool")(x)

    for stage_index, (repeats, channels) in enumerate(
        zip((4, 8, 4), (48, 96, 192)), start=2
    ):
        x = _shufflenet_v2_unit(
            x, channels, stride=2, name=f"stage{stage_index}_unit0"
        )
        for unit_index in range(1, repeats):
            x = _shufflenet_v2_unit(
                x, channels, stride=1, name=f"stage{stage_index}_unit{unit_index}"
            )
    x = _conv_bn_relu(x, 1024, 1, name="final_conv")
    base_model = tf.keras.Model(
        inputs, x, name="shufflenetv2_05_backbone"
    )
    base_model.trainable = False
    model = _attach_classifier_head(
        base_model,
        input_shape,
        num_classes,
        dropout_rate=dropout_rate,
        classifier_units=classifier_units,
        model_name="shufflenetv2_05_disease_detector",
    )
    return model, base_model


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
    hub_url = hub_url or "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"

    cache_dir = _resolve_tfhub_cache_dir(hub_cache_dir)
    if cache_dir:
        os.environ["TFHUB_CACHE_DIR"] = cache_dir

    attempts = max(1, int(hub_download_retries or 1))
    base_model = None
    for attempt in range(1, attempts + 1):
        try:
            base_model = _create_tfhub_layer(
                hub_url=hub_url,
                hub_cache_dir=hub_cache_dir,
                input_shape=input_shape,
            )
            # Catch corrupt modules now instead of partway through fit().
            _sanity_check_layer(base_model, input_shape)
            break
        except Exception as exc:
            if attempt >= attempts:
                print(
                    f"[model] TF Hub Lite0 module failed after {attempts} attempt(s): {exc}"
                )
                raise RuntimeError(
                    "The TF Hub EfficientNet-Lite0 module could not be loaded from "
                    f"{hub_url!r}. Cache: {cache_dir!r}. "
                    "On Kaggle/Colab, enable Internet for the first download or "
                    "provide a local Hub SavedModel through model.hub_url. "
                    "If you intended the Keras implementation, use architecture "
                    "'efficientnet_b0' instead."
                ) from exc
            if _is_tfhub_cache_error(exc):
                # Clear the cache so the next attempt re-downloads cleanly.
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
        raise RuntimeError("EfficientNet-Lite0 backbone was not created.")

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
            - MobileNetV2/MobileNetV3-Small: Standard Keras models (layers selectively unfrozen)
            - ShuffleNetV2 0.5x: Native Keras model (layers selectively unfrozen)
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
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)

    print(f"Unfroze {len(base_model.layers[unfreeze_from_layer:])} layers")
    print(
        f"Total trainable layers: {sum([layer.trainable for layer in base_model.layers])}"
    )


def print_model_summary(model):
    """Print model architecture summary"""
    model.summary()

    trainable_params = sum([w.numpy().size if hasattr(w, 'numpy') else int(tf.size(w)) 
                           for w in model.trainable_weights])
    non_trainable_params = sum([w.numpy().size if hasattr(w, 'numpy') else int(tf.size(w)) 
                               for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")


def get_model(architecture="mobilenetv2", **kwargs):
    """
    Factory function to get model by architecture name

    Args:
        architecture: Model architecture name. Options:
            - 'mobilenetv2': MobileNetV2 backbone (full layer-level control)
            - 'efficientnet_b0': Standard Keras EfficientNetB0 backbone
            - 'efficientnet' or 'efficientnet_lite0': EfficientNet-Lite0 from TF Hub
            - 'mobilenetv3_small': Standard Keras MobileNetV3-Small backbone
            - 'shufflenetv2_05': Native ShuffleNetV2 0.5x backbone
        **kwargs: Additional arguments passed to model creation functions:
            - input_shape: tuple (default (224, 224, 3))
            - num_classes: int (default 45)
            - dropout_rate: float (default 0.5)
            - weights: str (default 'imagenet', ignored for Lite0)
            - classifier_units: optional hidden units for lightweight heads
            - include_preprocessing: MobileNetV3 preprocessing flag
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
    elif arch_lower in {"mobilenetv3_small", "mobilenetv3small", "mobilenet_v3_small"}:
        kwargs = dict(kwargs)
        for key in [
            "hub_url",
            "hub_cache_dir",
            "hub_download_retries",
            "hub_download_delay_sec",
        ]:
            kwargs.pop(key, None)
        return create_mobilenetv3_small_model(**kwargs)
    elif arch_lower in {
        "shufflenetv2_05",
        "shufflenet_v2_05",
        "shufflenetv2_0.5x",
        "shufflenet_v2_0.5x",
    }:
        kwargs = dict(kwargs)
        for key in [
            "hub_url",
            "hub_cache_dir",
            "hub_download_retries",
            "hub_download_delay_sec",
            "include_preprocessing",
        ]:
            kwargs.pop(key, None)
        return create_shufflenetv2_05_model(**kwargs)
    elif arch_lower == "efficientnet_b0":
        kwargs = dict(kwargs)
        for key in [
            "hub_url",
            "hub_cache_dir",
            "hub_download_retries",
            "hub_download_delay_sec",
        ]:
            kwargs.pop(key, None)
        return create_efficientnet_b0_model(**kwargs)
    elif arch_lower in ["efficientnet", "efficientnet_lite0"]:
        return create_efficientnet_model(**kwargs)
    else:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Supported: 'mobilenetv2', 'efficientnet_b0', 'efficientnet_lite0', "
            f"'mobilenetv3_small', 'shufflenetv2_05'"
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
