"""Runtime profiles shared by hosted training notebooks."""

from copy import deepcopy


SUPPORTED_MODES = frozenset({"smoke", "full"})


def apply_runtime_profile(
    config, mode="smoke", platform=None, steps_per_execution=None,
    diagnostic=False,
):
    """Apply a hosted-training profile without mutating the source config.

    ``smoke`` prioritizes fast startup and a quick end-to-end validation.
    ``full`` applies the shared training-time caps and architecture-specific
    input size while preserving the source configuration.
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
    runtime_config.setdefault("model", {})
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

    # Kaggle's two-T4 runs now use XLA with mirrored execution.
    # A global batch of 128 gives 64 examples per T4; the
    # selector falls back to 64 for a single-GPU session and 16 for CPU-only.
    # Apply this override to both smoke and full runs; shared configs retain
    # their XLA setting for compatible environments.
    if normalized_platform == "kaggle":
        dataset = runtime_config["dataset"]
        dataset["batch_size"] = 128
        dataset["batch_size_fallbacks"] = [128, 64, 32, 16]
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

    if normalized_mode == "full":
        # Respect config epochs and unfreeze_epoch; no capping.
        if not diagnostic:
            training = runtime_config["training"]
            fine_tuning_enabled = bool(
                training.get("freeze_base", True)
                and training.get("fine_tuning", {}).get("enabled", True)
            )
            total_epochs = int(training.get("epochs", 10))
            phase1_epochs = (
                min(int(training.get("unfreeze_epoch", 6)), total_epochs)
                if fine_tuning_enabled
                else total_epochs
            )
            phase2_epochs = total_epochs - phase1_epochs
            callbacks = runtime_config.get("callbacks", {})
            phase1_early_stopping = callbacks.get("early_stopping")
            if phase1_early_stopping and phase1_early_stopping.get(
                "enabled", True
            ):
                phase1_early_stopping["patience"] = min(
                    int(phase1_early_stopping.get("patience", 10)),
                    max(0, phase1_epochs - 1),
                )

            phase2_early_stopping = callbacks.get(
                "fine_tuning_early_stopping"
            )
            if (
                phase2_epochs > 0
                and phase2_early_stopping
                and phase2_early_stopping.get("enabled", True)
            ):
                phase2_early_stopping["patience"] = min(
                    int(phase2_early_stopping.get("patience", 6)),
                    max(0, phase2_epochs - 1),
                )

        architecture = str(
            runtime_config["model"].get("architecture", "")
        ).strip().lower().replace("-", "_")
        faster_resolution_architectures = {
            "mobilenetv2",
            "mobilenetv3_small",
            "mobilenetv3small",
            "mobilenet_v3_small",
            "efficientnet_b0",
            "shufflenetv2_05",
            "shufflenet_v2_05",
            "shufflenetv2_0.5x",
            "shufflenet_v2_0.5x",
        }
        if architecture in faster_resolution_architectures:
            input_shape = list(runtime_config["model"].get(
                "input_shape", (224, 224, 3)
            ))
            input_shape[0:2] = [192, 192]
            runtime_config["model"]["input_shape"] = input_shape

    runtime.update({"mode": normalized_mode, "platform": platform})
    return runtime_config
