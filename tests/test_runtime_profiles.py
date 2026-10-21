from training.runtime_profiles import apply_runtime_profile


def test_smoke_profile_is_fast_and_does_not_mutate_source():
    config = {
        "dataset": {"batch_size": 128, "cache_mode": "auto"},
        "augmentation": {"class_aware": {"enabled": True}},
        "performance": {"steps_per_execution": 32, "jit_compile": True},
        "training": {"epochs": 35},
        "callbacks": {"tensorboard": {"enabled": True}},
        "metrics": ["accuracy", "auc"],
    }

    smoke = apply_runtime_profile(config, "smoke", platform="kaggle")

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


def test_profile_rejects_unknown_mode():
    try:
        apply_runtime_profile({}, "benchmark")
    except ValueError as error:
        assert "Unsupported training mode" in str(error)
    else:
        raise AssertionError("Unknown training mode should fail explicitly")
