import pytest

from training.runtime_profiles import apply_runtime_profile
from training.utils import choose_batch_size


def test_smoke_profile_is_fast_and_does_not_mutate_source():
    config = {
        "dataset": {"batch_size": 128, "cache_mode": "auto"},
        "augmentation": {"class_aware": {"enabled": True}},
        "performance": {"steps_per_execution": 32, "jit_compile": True},
        "training": {"epochs": 35},
        "callbacks": {"tensorboard": {"enabled": True}},
        "metrics": ["accuracy", "auc"],
    }

    smoke = apply_runtime_profile(config, "smoke", platform="colab")

    assert smoke["dataset"]["batch_size"] == 64
    assert smoke["dataset"]["cache_mode"] == "none"
    assert smoke["augmentation"]["class_aware"]["enabled"] is False
    assert smoke["performance"]["steps_per_execution"] == 8
    assert smoke["performance"]["jit_compile"] is False
    assert smoke["training"]["epochs"] == 5
    assert smoke["metrics"] == ["accuracy"]
    assert config["dataset"]["batch_size"] == 128
    assert config["performance"]["jit_compile"] is True


def test_full_profile_preserves_model_settings():
    config = {
        "dataset": {"batch_size": 128},
        "performance": {"steps_per_execution": 32, "jit_compile": True},
        "training": {"epochs": 35},
    }

    full = apply_runtime_profile(config, "full", platform="colab")

    assert full["dataset"]["batch_size"] == 128
    assert full["performance"]["steps_per_execution"] == 32
    assert full["performance"]["jit_compile"] is True
    assert full["training"]["epochs"] == 35
    assert full["runtime"] == {"mode": "full", "platform": "colab"}
    assert full["training"]["log_mode"] == "summary"


def test_runtime_profile_preserves_explicit_progress_logging():
    runtime = apply_runtime_profile(
        {"training": {"epochs": 10, "log_mode": "progress"}},
        "full",
        platform="kaggle",
    )

    assert runtime["training"]["log_mode"] == "progress"
    assert runtime["training"]["progress_log_interval"] == 64


@pytest.mark.parametrize("mode", ["smoke", "full"])
def test_kaggle_profile_uses_all_visible_gpus_without_xla(mode):
    config = {
        "dataset": {
            "batch_size": 128,
            "batch_size_fallbacks": [128, 64, 32],
            "cache_mode": "auto",
        },
        "augmentation": {"class_aware": {"enabled": True}},
        "performance": {
            "mixed_precision": True,
            "steps_per_execution": 32,
            "jit_compile": True,
            "distribution": "auto",
        },
        "training": {"epochs": 20},
    }

    runtime = apply_runtime_profile(config, mode, platform="kaggle")

    assert runtime["dataset"]["batch_size"] == 64
    assert runtime["dataset"]["batch_size_fallbacks"] == [64, 32, 16]
    assert runtime["dataset"]["auto_batch_fallback"] is True
    assert runtime["performance"]["steps_per_execution"] == 8
    assert runtime["performance"]["jit_compile"] is False
    assert runtime["performance"]["distribution"] == "auto"
    assert runtime["performance"]["multi_gpu"] is True
    assert runtime["performance"]["mixed_precision"] is True
    assert runtime["dataset"]["cache_dtype"] == "float16"
    assert config["dataset"]["batch_size"] == 128
    assert runtime["augmentation"]["class_aware"]["enabled"] is (
        mode == "full"
    )
    if mode == "full":
        assert runtime["dataset"]["cache_mode"] == "auto"


def test_kaggle_profile_preserves_explicit_steps_per_execution_override():
    runtime = apply_runtime_profile(
        {"dataset": {"batch_size": 128}},
        "full",
        platform="kaggle",
        steps_per_execution=4,
    )

    assert runtime["performance"]["steps_per_execution"] == 4
    assert runtime["dataset"]["batch_size"] == 64
    assert runtime["performance"]["jit_compile"] is False


def test_kaggle_profile_uses_configured_execution_override():
    runtime = apply_runtime_profile(
        {"performance": {"kaggle_steps_per_execution": 3}},
        "full",
        platform="kaggle",
    )

    assert runtime["performance"]["steps_per_execution"] == 3


def test_kaggle_diagnostic_profile_isolated_and_keeps_training_recipe():
    config = {
        "dataset": {
            "batch_size": 64,
            "cache_mode": "auto",
            "shuffle_buffer": 1000,
        },
        "augmentation": {"class_aware": {"enabled": True, "max_train_samples": 500}},
        "class_weights": {"enabled": True, "max_weight": 8.0, "normalize": True},
        "performance": {"mixed_precision": True, "jit_compile": True},
        "training": {"epochs": 50, "log_mode": "summary"},
    }

    diagnostic = apply_runtime_profile(
        config, "full", platform="kaggle", diagnostic=True
    )
    normal = apply_runtime_profile(config, "full", platform="kaggle")

    assert diagnostic["dataset"]["batch_size"] == 64
    assert diagnostic["dataset"]["cache_mode"] == "none"
    assert diagnostic["dataset"]["cache_dtype"] == "float32"
    assert diagnostic["dataset"]["shuffle_buffer"] == 64
    assert diagnostic["augmentation"]["class_aware"]["enabled"] is True
    assert diagnostic["class_weights"] == config["class_weights"]
    assert diagnostic["performance"]["mixed_precision"] is True
    assert diagnostic["performance"]["distribution"] == "auto"
    assert diagnostic["performance"]["multi_gpu"] is True
    assert diagnostic["performance"]["jit_compile"] is False
    assert diagnostic["performance"]["steps_per_execution"] == 1
    assert diagnostic["performance"]["finite_batch_check"] is False
    assert diagnostic["training"]["epochs"] == 1
    assert diagnostic["training"]["log_mode"] == "summary"
    assert diagnostic["training"]["progress_log_interval"] == 64
    assert diagnostic["runtime"]["diagnostic"] is True
    assert diagnostic["runtime"]["max_train_steps"] == 8
    assert diagnostic["runtime"]["max_validation_steps"] == 2
    assert normal["dataset"]["cache_mode"] == "auto"
    assert normal["performance"]["steps_per_execution"] == 8
    assert normal["training"]["progress_log_interval"] == 64
    assert "diagnostic" not in normal["runtime"]
    assert config["dataset"]["shuffle_buffer"] == 1000


@pytest.mark.parametrize(
    ("mode", "platform"), [("smoke", "kaggle"), ("full", "colab")]
)
def test_diagnostic_profile_rejects_unsupported_runtime(mode, platform):
    with pytest.raises(ValueError, match="only for full Kaggle"):
        apply_runtime_profile({}, mode, platform=platform, diagnostic=True)


@pytest.mark.parametrize(
    ("runtime", "expected_batch"),
    [
        ({"num_replicas": 2, "has_gpu": True}, 64),
        ({"num_replicas": 1, "has_gpu": True}, 32),
        ({"num_replicas": 1, "has_gpu": False}, 16),
    ],
)
def test_kaggle_batch_size_adapts_to_available_devices(runtime, expected_batch):
    config = apply_runtime_profile(
        {"dataset": {"batch_size": 128}}, "full", platform="kaggle"
    )

    assert choose_batch_size(config, runtime) == expected_batch


def test_profile_rejects_unknown_mode():
    try:
        apply_runtime_profile({}, "benchmark")
    except ValueError as error:
        assert "Unsupported training mode" in str(error)
    else:
        raise AssertionError("Unknown training mode should fail explicitly")
