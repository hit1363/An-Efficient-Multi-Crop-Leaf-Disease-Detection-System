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
    assert runtime["performance"]["steps_per_execution"] == 1
    assert runtime["performance"]["jit_compile"] is False
    assert runtime["performance"]["distribution"] == "auto"
    assert runtime["performance"]["multi_gpu"] is True
    assert runtime["performance"]["mixed_precision"] is True
    assert config["dataset"]["batch_size"] == 128
    assert runtime["augmentation"]["class_aware"]["enabled"] is (
        mode == "full"
    )
    if mode == "full":
        assert runtime["dataset"]["cache_mode"] == "auto"


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
