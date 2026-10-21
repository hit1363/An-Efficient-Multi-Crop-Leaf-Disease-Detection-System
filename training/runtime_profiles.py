"""Runtime profiles shared by hosted training notebooks."""

from copy import deepcopy


SUPPORTED_MODES = frozenset({"smoke", "full"})


def apply_runtime_profile(config, mode="smoke", platform=None):
    """Apply a hosted-training profile without mutating the source config.

    ``smoke`` prioritizes fast startup and a quick end-to-end validation.
    ``full`` preserves the model configuration's accuracy-oriented settings.
    """
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported training mode {mode!r}; expected one of "
            f"{sorted(SUPPORTED_MODES)}."
        )

    runtime_config = deepcopy(config)
    runtime_config.setdefault("dataset", {})
    runtime_config.setdefault("augmentation", {}).setdefault("class_aware", {})
    runtime_config.setdefault("performance", {})
    runtime_config.setdefault("training", {})
    runtime_config.setdefault("callbacks", {}).setdefault("tensorboard", {})
    runtime_config["training"].setdefault("log_mode", "summary")

    if normalized_mode == "smoke":
        dataset = runtime_config["dataset"]
        requested_batch = int(dataset.get("batch_size", 32))
        smoke_batch = min(requested_batch, 64)
        dataset["batch_size"] = smoke_batch
        dataset["batch_size_fallbacks"] = [smoke_batch, 32, 16]
        dataset["auto_batch_fallback"] = True
        dataset["cache_mode"] = "none"

        runtime_config["augmentation"]["class_aware"]["enabled"] = False
        runtime_config["performance"].update(
            {
                "steps_per_execution": 8,
                "jit_compile": False,
                "finite_batch_check": False,
                "distribution": "single",
                "multi_gpu": False,
            }
        )
        runtime_config["training"]["epochs"] = min(
            int(runtime_config["training"].get("epochs", 5)), 5
        )
        runtime_config["callbacks"]["tensorboard"]["enabled"] = False
        runtime_config["metrics"] = ["accuracy"]

    # Kaggle's two-T4 runs previously crashed with mirrored execution plus XLA.
    # Keep both GPUs visible and use MirroredStrategy when available, while
    # leaving XLA off. A global batch of 64 gives 32 examples per T4; the batch
    # selector falls back to 32 for a single-GPU session and 16 for CPU-only.
    # Apply this override to both smoke and full runs; shared configs retain
    # their XLA setting for compatible environments.
    if str(platform or "").strip().lower() == "kaggle":
        dataset = runtime_config["dataset"]
        dataset["batch_size"] = 64
        dataset["batch_size_fallbacks"] = [64, 32, 16]
        dataset["auto_batch_fallback"] = True
        runtime_config["performance"].update(
            {
                "steps_per_execution": 1,
                "jit_compile": False,
                "distribution": "auto",
                "multi_gpu": True,
            }
        )

    runtime_config.setdefault("runtime", {}).update(
        {"mode": normalized_mode, "platform": platform}
    )
    return runtime_config
