"""Serialization helpers for custom objects used by trained models."""

try:
    from .losses import FocalLoss
    from .metrics import MacroPrecisionRecall
except ImportError:
    # Support direct script execution from inside the training directory.
    from losses import FocalLoss
    from metrics import MacroPrecisionRecall


def get_custom_objects():
    """Return custom objects needed to load training-produced Keras models.

    Both plain and registered names are included so older and current H5
    checkpoints can be loaded by evaluation and quantization tools.
    """
    return {
        "FocalLoss": FocalLoss,
        "LeafDisease>FocalLoss": FocalLoss,
        "MacroPrecisionRecall": MacroPrecisionRecall,
        "LeafDisease>MacroPrecisionRecall": MacroPrecisionRecall,
    }
