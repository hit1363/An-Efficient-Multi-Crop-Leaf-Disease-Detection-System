"""Runtime profiles shared by hosted training notebooks."""

from copy import deepcopy


SUPPORTED_MODES = frozenset({"smoke", "full"})


def apply_runtime_profile(
    config, mode="smoke", platform=None, steps_per_execution=None,
    diagnostic=False,
):
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

    normalized_platform = str(platform or "").strip().lower()
    if diagnostic and (normalized_mode != "full" or normalized_platform != "kaggle"):
        raise ValueError(
            "The training diagnostic is supported only for full Kaggle runs."
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
    if normalized_platform == "kaggle":
        dataset = runtime_config["dataset"]
        dataset["batch_size"] = 64
        dataset["batch_size_fallbacks"] = [64, 32, 16]
        dataset["auto_batch_fallback"] = True
        configured_execution = runtime_config["performance"].get(
            "kaggle_steps_per_execution", 32
        )
        execution_steps = (
            configured_execution
            if steps_per_execution is None
            else steps_per_execution
        )
        if int(execution_steps) < 1:
            raise ValueError("steps_per_execution must be a positive integer")
        dataset["cache_dtype"] = "float16"
        runtime_config["performance"].update(
            {
                "steps_per_execution": int(execution_steps),
                "jit_compile": False,
                "distribution": "auto",
                "multi_gpu": True,
            }
        )
        # Enable memory cache for faster I/O on Kaggle GPUs (80K images ~ 6GB uncompressed).
        dataset["cache_mode"] = "memory"
        if normalized_mode == "full":
            runtime_config["training"].setdefault("progress_log_interval", 64)

    # Keep diagnosis opt-in and isolated from ordinary full runs. These caps
    # affect fit() only; dataset counts and class weights still use the full
    # training split.
    runtime = runtime_config.setdefault("runtime", {})
    for key in ("diagnostic", "max_train_steps", "max_validation_steps"):
        runtime.pop(key, None)
    if diagnostic:
        dataset = runtime_config["dataset"]
        dataset.update(
            {
                "cache_mode": "none",
                "cache_dtype": "float32",
                "shuffle_buffer": int(dataset.get("batch_size", 64)),
            }
        )
        runtime_config["performance"].update(
            {"steps_per_execution": 1, "finite_batch_check": False}
        )
        # The custom callback prints every diagnostic batch; keep Keras's
        # carriage-return progress bar and callback chatter disabled.
        runtime_config["training"].update({"epochs": 1, "log_mode": "summary"})
        runtime.update(
            {
                "diagnostic": True,
                "max_train_steps": 8,
                "max_validation_steps": 2,
            }
        )

    runtime.update({"mode": normalized_mode, "platform": platform})
    return runtime_config
