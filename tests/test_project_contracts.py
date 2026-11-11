"""Dependency-light contract tests for the model and notebook expansion."""

import ast
import json
from pathlib import Path

import yaml
from IPython.core.interactiveshell import InteractiveShell


ROOT = Path(__file__).resolve().parents[1]


def _benchmark_like_class_counts():
    """Per-class training counts shaped like the shipped cap-500 benchmark.

    Mirrors the observed distribution: most classes sit near the 350 cap and a
    handful of Guava classes are true minorities. This keeps the contract test
    dependency-light (``training.utils`` imports TensorFlow) while still
    exercising the degenerate-cutoff condition.
    """
    counts = [350] * 67
    counts[0] = 193
    counts[1] = 252
    counts[2] = 261
    counts[3] = 274
    counts[4] = 280
    counts[5] = 296
    counts[6] = 319
    counts.append(2)
    counts.append(2)
    counts.append(6)
    counts.append(3)
    counts.append(3)
    return counts


def _minority_indices(counts, cutoff):
    """Local mirror of ``training.utils.get_minority_class_indices``."""
    return [index for index, count in enumerate(counts) if 0 < int(count) < int(cutoff)]


def test_training_and_quantization_sources_parse():
    source_files = [
        *sorted((ROOT / "training").glob("*.py")),
        *sorted((ROOT / "quantization").glob("*.py")),
    ]
    for path in source_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_all_model_configs_exist():
    architectures = [
        "mobilenet_v1",
        "mobilenetv2",
        "mobilenetv3_small",
        "mobilenet_v3_large",
        "nasnet_mobile",
        "efficientnet_b0",
        "efficientnet_v2_b0",
        "shufflenetv2_05",
    ]
    for architecture in architectures:
        config_path = ROOT / "training" / f"config_{architecture}.yaml"
        assert config_path.exists(), config_path
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["model"]["architecture"] == architecture


def test_new_architecture_configs_share_the_benchmark_recipe():
    benchmark = yaml.safe_load(
        (ROOT / "training" / "config_benchmark.yaml").read_text(encoding="utf-8")
    )
    for architecture in (
        "mobilenet_v1",
        "mobilenet_v3_large",
        "nasnet_mobile",
        "efficientnet_v2_b0",
    ):
        config = yaml.safe_load(
            (ROOT / "training" / f"config_{architecture}.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert config["model"]["architecture"] == architecture
        assert config["model"]["num_classes"] == 72
        # unfreeze_from_layer is a layer index into a specific backbone, so it
        # is deliberately architecture-specific: mobilenet_v1's 86-layer
        # backbone cannot share mobilenetv2's index 100 (which would freeze
        # every layer and leave phase 2 with zero trainable layers).
        benchmark_training = {
            key: value
            for key, value in benchmark["training"].items()
            if key != "unfreeze_from_layer"
        }
        config_training = {
            key: value
            for key, value in config["training"].items()
            if key != "unfreeze_from_layer"
        }
        assert config_training == benchmark_training
        for section in (
            "augmentation",
            "optimizer",
            "loss",
            "lr_schedule",
            "class_weights",
            "metrics",
            "seed",
        ):
            assert config[section] == benchmark[section]
        assert config["dataset"]["data_dir"] == benchmark["dataset"]["data_dir"]
        # Recipe v4 (2026-11): the four laggards moved from batch 32 with no
        # cache to the shared in-memory cache, so every pretrained config now
        # matches the template's dataset recipe. nasnet_mobile is the sole
        # exception: its 769-layer graph is the deepest, so it runs at 64 as an OOM precaution.
        if architecture == "nasnet_mobile":
            assert config["dataset"]["batch_size"] == 64
            assert config["dataset"]["batch_size_fallbacks"] == [64, 32, 16]
        else:
            assert config["dataset"]["batch_size"] == benchmark["dataset"]["batch_size"]
            assert (
                config["dataset"]["batch_size_fallbacks"]
                == benchmark["dataset"]["batch_size_fallbacks"]
            )
        assert config["dataset"]["cache_mode"] == benchmark["dataset"]["cache_mode"]
        assert config["export"]["save_dir"] == "../models"
        assert config["callbacks"]["csv_logger"]["filename"].endswith(
            f"{architecture}.csv"
        )


def test_pretrained_configs_share_the_recipe_v4_core():
    """Every ImageNet config trains under one recipe, differing only where the
    backbone forces it (unfreeze index, class-weight cap, NASNet batch size).

    This is the guard for the v4 unification: before it, the four laggard
    configs silently kept batch 32, no decoding cache, a plateau LR schedule,
    no label smoothing, and a 10-epoch fine-tune tail while the four
    "modernized" ones used the recipe that produced the 87% baseline.
    """
    pretrained = (
        "mobilenet_v1",
        "mobilenetv2",
        "mobilenetv3_small",
        "mobilenet_v3_large",
        "nasnet_mobile",
        "efficientnet_b0",
        "efficientnet_v2_b0",
    )
    for architecture in pretrained:
        config = yaml.safe_load(
            (ROOT / "training" / f"config_{architecture}.yaml").read_text(
                encoding="utf-8"
            )
        )
        training = config["training"]
        assert training["freeze_base"] is True
        assert training["epochs"] == 36
        assert training["unfreeze_epoch"] == 20
        assert float(training["fine_tune_learning_rate"]) == 2e-5
        assert training["fine_tuning"]["enabled"] is True
        optimizer = config["optimizer"]
        assert optimizer["name"] == "adam"
        assert float(optimizer["learning_rate"]) == 3e-4
        assert float(optimizer["clipnorm"]) == 1.0
        schedule = config["lr_schedule"]
        assert schedule["type"] == "cosine_decay"
        assert schedule["decay_epochs"] == 20
        assert float(schedule["alpha"]) == 0.01
        assert config["loss"]["name"] == "categorical_crossentropy"
        assert float(config["loss"]["label_smoothing"]) == 0.1
        assert config["augmentation"]["vertical_flip"] is True
        assert config["augmentation"]["rotation_range"] == 25
        assert config["dataset"]["cache_mode"] == "memory"
        assert config["dataset"]["cache_memory_limit_gb"] == 8.0
        assert config["performance"]["deterministic"] is False
        metrics = config["metrics"]
        assert "accuracy" in metrics and "auc" in metrics
        # Precision/recall are macro-averaged from a confusion matrix, so they
        # carry information the micro default (== accuracy) would not.
        assert "precision" in metrics and "recall" in metrics
        assert yaml.safe_load(
            (ROOT / "training" / "config_benchmark.yaml").read_text(encoding="utf-8")
        )["metrics"] == metrics
        assert config["evaluation"]["tta"] == "flip"
        if architecture == "nasnet_mobile":
            assert config["dataset"]["batch_size"] == 64
        else:
            assert config["dataset"]["batch_size"] == 128


def test_comparison_overlay_keeps_architecture_specific_settings():
    """compare_models must not hand an architecture another's unfreeze index.

    The comparison run clones config_benchmark.yaml. Its unfreeze placeholder
    is MobileNetV2's 100, which is a valid slice for MobileNetV2's 154-layer
    backbone but an empty one for MobileNet V1's 86 layers. The overlay pulls
    the calibrated index and batch size from the matching per-architecture
    config so the generated config trains what the pinned configs train.
    """
    from training.compare_models import (
        ARCHITECTURES,
        _apply_architecture_overrides,
    )

    template = yaml.safe_load(
        (ROOT / "training" / "config_benchmark.yaml").read_text(encoding="utf-8")
    )
    for architecture in ARCHITECTURES:
        source = yaml.safe_load(
            (ROOT / "training" / f"config_{architecture}.yaml").read_text(
                encoding="utf-8"
            )
        )
        config = _apply_architecture_overrides(
            json.loads(json.dumps(template)), architecture, ROOT
        )
        assert config["training"]["unfreeze_from_layer"] == (
            source["training"]["unfreeze_from_layer"]
        ), architecture
        assert config["dataset"]["batch_size"] == source["dataset"]["batch_size"]
        assert config["dataset"]["batch_size_fallbacks"] == (
            source["dataset"]["batch_size_fallbacks"]
        )

    # The template still carries its documented placeholder for unknown names.
    untouched = _apply_architecture_overrides(
        json.loads(json.dumps(template)), "not_a_real_architecture", ROOT
    )
    assert untouched["training"]["unfreeze_from_layer"] == (
        template["training"]["unfreeze_from_layer"]
    )


def test_pillow_constraints_match_project_metadata():
    requirements_text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "pillow>=12.3.0" in requirements_text
    assert '"pillow>=12.3.0"' in pyproject_text


def test_optimized_runtime_defaults_are_present():
    expected_defaults = {
        "mobilenetv2": (3.0, 128, [128, 64, 32, 16], "memory"),
        "mobilenetv3_small": (5.0, 128, [128, 64, 32, 16], "memory"),
        "shufflenetv2_05": (5.0, 128, [128, 64, 32, 16], "memory"),
        "efficientnet_b0": (5.0, 128, [128, 64, 32, 16], "memory"),
        # Recipe v4 (2026-11): the four laggards join the shared batch/cache
        # recipe. nasnet_mobile stays at 64 as an OOM precaution for its
        # 769-layer graph (runtime_profiles.py caps it there too).
        "mobilenet_v1": (5.0, 128, [128, 64, 32, 16], "memory"),
        "mobilenet_v3_large": (5.0, 128, [128, 64, 32, 16], "memory"),
        "nasnet_mobile": (5.0, 64, [64, 32, 16], "memory"),
        "efficientnet_v2_b0": (5.0, 128, [128, 64, 32, 16], "memory"),
    }
    for architecture, (max_weight, batch_size, fallbacks, cache_mode) in expected_defaults.items():
        config = yaml.safe_load(
            (ROOT / "training" / f"config_{architecture}.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert config["dataset"]["batch_size"] == batch_size
        assert config["dataset"]["batch_size_fallbacks"] == fallbacks
        assert config["dataset"]["cache_mode"] == cache_mode
        assert config["dataset"]["cache_memory_limit_gb"] == 8.0
        assert config["class_weights"]["max_weight"] == max_weight
        assert config["performance"]["finite_batch_check"] is True
        assert config["performance"]["terminate_on_nan"] is True
        assert config["performance"]["validate_dataset_splits"] is True
        assert config["performance"]["distribution"] == "auto"

    benchmark = yaml.safe_load(
        (ROOT / "training" / "config_benchmark.yaml").read_text(encoding="utf-8")
    )
    assert benchmark["dataset"]["batch_size"] == 128
    assert benchmark["dataset"]["batch_size_fallbacks"] == [128, 64, 32, 16]
    assert benchmark["dataset"]["cache_mode"] == "memory"
    assert benchmark["dataset"]["cache_memory_limit_gb"] == 8.0
    assert benchmark["performance"]["validate_dataset_splits"] is True

    utils_text = (ROOT / "training" / "utils.py").read_text(encoding="utf-8")
    train_text = (ROOT / "training" / "train.py").read_text(encoding="utf-8")
    assert "MirroredStrategy" in utils_text
    assert "batch_size=batch_size" in utils_text
    assert "ThroughputCallback" in utils_text
    assert "validate_dataset_batch" in utils_text
    assert "choose_batch_size" in utils_text
    assert "TerminateOnNaN" in utils_text
    assert "with strategy.scope()" in train_text


def test_imbalance_aware_defaults_are_enabled_in_all_training_configs():
    architectures = (
        "benchmark",
        "mobilenetv2",
        "mobilenetv3_small",
        "shufflenetv2_05",
        "efficientnet_b0",
        "mobilenet_v1",
        "mobilenet_v3_large",
        "nasnet_mobile",
        "efficientnet_v2_b0",
    )
    # Recipe v3 (2026-11): the minority cutoff is 100 for every architecture.
    # The benchmark caps classes at 350 training images, so the old cutoff of
    # 500 selected all 72 classes and disabled the baseline augmentation.
    # shufflenetv2_05 additionally trains from scratch, so its monitoring
    # differs from the pretrained baselines.
    expected_monitoring = {
        arch: {
            "max_train_samples": 100,
            "checkpoint_monitor": "val_loss",
            "checkpoint_mode": "min",
            "early_stopping_monitor": "val_loss",
            "early_stopping_mode": "min",
            "early_stopping_patience": 6,
            "early_stopping_min_delta": 0.0,
            "fine_tuning_monitor": "val_loss",
            "fine_tuning_mode": "min",
            "fine_tuning_patience": 6,
        }
        for arch in architectures
    }
    expected_monitoring["shufflenetv2_05"] = {
        "max_train_samples": 100,
        "checkpoint_monitor": "val_accuracy",
        "checkpoint_mode": "max",
        "early_stopping_monitor": "val_accuracy",
        "early_stopping_mode": "max",
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.001,
        "fine_tuning_monitor": "val_loss",
        "fine_tuning_mode": "min",
        "fine_tuning_patience": 6,
    }
    for architecture in architectures:
        config = yaml.safe_load(
            (ROOT / "training" / f"config_{architecture}.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert config["loss"]["name"] in {
            "categorical_crossentropy",
            "focal_loss",
        }
        # Recipe guard: no config may drift back to the 1e-3 peak LR that
        # destabilized the from-scratch shufflenetv2_05 run (Recipe v4).
        assert float(config["optimizer"]["learning_rate"]) <= 5e-4
        assert config["class_weights"]["enabled"] is True
        class_aware = config["augmentation"]["class_aware"]
        assert class_aware["enabled"] is True
        assert class_aware["rotation_range"] == 20
        assert class_aware["zoom_range"] == [0.9, 1.1]
        assert class_aware["horizontal_flip"] is True
        assert class_aware["vertical_flip"] is True
        assert class_aware["contrast_range"] == 0.15
        assert class_aware["brightness_range"] == [0.85, 1.15]
        monitoring = expected_monitoring[architecture]
        assert class_aware["max_train_samples"] == monitoring["max_train_samples"]
        assert config["callbacks"]["checkpoint"]["monitor"] == monitoring["checkpoint_monitor"]
        assert config["callbacks"]["checkpoint"]["mode"] == monitoring["checkpoint_mode"]
        assert config["callbacks"]["checkpoint"]["save_best_only"] is True
        assert config["callbacks"]["early_stopping"]["patience"] == monitoring["early_stopping_patience"]
        assert config["callbacks"]["early_stopping"]["monitor"] == monitoring["early_stopping_monitor"]
        assert config["callbacks"]["early_stopping"]["mode"] == monitoring["early_stopping_mode"]
        assert config["callbacks"]["early_stopping"]["restore_best_weights"] is True
        assert float(config["callbacks"]["early_stopping"].get("min_delta", 0.0) or 0.0) == monitoring["early_stopping_min_delta"]
        assert config["callbacks"]["fine_tuning_early_stopping"]["patience"] == monitoring["fine_tuning_patience"]
        assert config["callbacks"]["fine_tuning_early_stopping"]["monitor"] == monitoring["fine_tuning_monitor"]
        assert config["callbacks"]["fine_tuning_early_stopping"]["mode"] == monitoring["fine_tuning_mode"]
        assert config["callbacks"]["fine_tuning_early_stopping"]["restore_best_weights"] is True
        expected_max_weight = 3.0 if architecture == "mobilenetv2" else 5.0
        assert config["class_weights"]["max_weight"] == expected_max_weight
        assert config["class_weights"]["normalize"] is True
        assert config["performance"]["mixed_precision"] is True
        assert config["performance"]["jit_compile"] is True
        # Recipe v4 (2026-11): every pretrained config caches in memory now
        # that the four laggards left their batch-32/no-cache regime behind.
        assert config["dataset"]["cache_mode"] == "memory"
        assert config["dataset"]["cache_memory_limit_gb"] == 8.0

        # The cutoff must leave the baseline policy in charge of the bulk of
        # the benchmark. Without a shipped dataset we use the canonical
        # cap-500 benchmark distribution: most classes sit near the cap and
        # only a handful of true minorities fall below it.
        benchmark_counts = _benchmark_like_class_counts()
        selected = _minority_indices(
            benchmark_counts, class_aware["max_train_samples"]
        )
        assert len(selected) * 2 < len(benchmark_counts), (
            f"{architecture}: cutoff {class_aware['max_train_samples']} selects "
            f"{len(selected)}/{len(benchmark_counts)} classes and disables the "
            "baseline augmentation policy"
        )

    # Recipe v3 (2026-11): shufflenetv2_05 runs 40 epochs with a 3-epoch warmup
    # and reports top-5 accuracy; the pretrained baselines keep Recipe v2.
    shufflenet_config = yaml.safe_load(
        (ROOT / "training" / "config_shufflenetv2_05.yaml").read_text(encoding="utf-8")
    )
    assert shufflenet_config["training"]["epochs"] == 40
    assert shufflenet_config["lr_schedule"]["warmup_epochs"] == 3
    assert shufflenet_config["lr_schedule"]["warmup_start_lr"] == 0.0001
    assert shufflenet_config["lr_schedule"]["decay_epochs"] == 40
    # Recipe v6 (2026-11): the from-scratch run reports the same macro
    # precision/recall/AUC pair as every ImageNet config, and keeps top-5 for
    # the near-prior early epochs.
    shufflenet_metrics = shufflenet_config["metrics"]
    assert "top_k_categorical_accuracy" in shufflenet_metrics
    for metric_name in ("accuracy", "precision", "recall", "auc"):
        assert metric_name in shufflenet_metrics

    # Recipe v4 (2026-11): the v3 peak LR of 1e-3 destabilized the from-scratch
    # backbone (train accuracy peaked ~14% at epoch 9 then declined while
    # val_accuracy hovered at random). Guard the stabilizers so the recipe
    # cannot silently regress to the diverging configuration.
    assert shufflenet_config["optimizer"]["learning_rate"] <= 5e-4
    assert float(shufflenet_config["optimizer"].get("decay", 0.0) or 0.0) > 0
    assert (
        float(
            shufflenet_config["callbacks"]["early_stopping"].get("min_delta", 0.0)
            or 0.0
        )
        <= 0.001
    )

    # Recipe v5 (2026-11): the smoke run reproduced the v4 divergence (val loss
    # 4.27 at epoch 1 climbing to 6.02 by epoch 5 while train loss fell), so the
    # peak halves to 2e-4, dropout rises to 0.35, and the baseline policy stops
    # flipping leaves vertically. Monitoring stays on val_accuracy: v3 showed
    # val_loss is a fluke-prone signal for a from-scratch backbone.
    assert shufflenet_config["optimizer"]["learning_rate"] == 0.0002
    assert shufflenet_config["model"]["dropout_rate"] >= 0.3
    assert shufflenet_config["augmentation"]["vertical_flip"] is False
    assert "keras" in shufflenet_config["export"]["formats"]

    # Recipe v4 (2026-11): every two-phase config freezes the head for 20
    # epochs and then fine-tunes for 16 (36 total), matching the EfficientNetB0
    # run whose accuracy kept climbing through the long fine-tune tail.
    expected_unfreeze_epochs = {
        "benchmark": 20,
        "mobilenetv2": 20,
        "mobilenetv3_small": 20,
        "efficientnet_b0": 20,
        "mobilenet_v1": 20,
        "mobilenet_v3_large": 20,
        "nasnet_mobile": 20,
        "efficientnet_v2_b0": 20,
    }
    for architecture, unfreeze_epoch in expected_unfreeze_epochs.items():
        config = yaml.safe_load(
            (ROOT / "training" / f"config_{architecture}.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert config["training"]["freeze_base"] is True
        assert config["training"]["unfreeze_epoch"] == unfreeze_epoch
        assert config["training"]["epochs"] == 36
        # The fine-tune phase must be at least as long as the frozen phase was
        # in v2 (10 epochs); v4 doubles the tail rather than shortening it.
        assert config["training"]["epochs"] - unfreeze_epoch == 16


def test_all_model_configs_use_live_progress_logging():
    architectures = (
        "benchmark",
        "mobilenet_v1",
        "mobilenetv2",
        "mobilenetv3_small",
        "mobilenet_v3_large",
        "nasnet_mobile",
        "efficientnet_v2_b0",
        "shufflenetv2_05",
        "efficientnet_b0",
    )
    for architecture in architectures:
        config = yaml.safe_load(
            (ROOT / "training" / f"config_{architecture}.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert config["training"]["log_mode"] == "progress"


def test_colab_training_notebooks_stream_live_keras_progress():
    architectures = (
        "mobilenet_v1",
        "mobilenetv2",
        "efficientnet_b0",
        "mobilenetv3_small",
        "mobilenet_v3_large",
        "nasnet_mobile",
        "efficientnet_v2_b0",
        "shufflenetv2_05",
    )
    for architecture in architectures:
        path = ROOT / "notebooks" / f"colab_{architecture}_training.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code_cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        all_code = "\n".join(code_cells)
        training_cell = next(
            source
            for source in code_cells
            if '"train.py"' in source
            and '"--config"' in source
            and "subprocess.Popen" in source
        )

        assert 'cfg["training"]["log_mode"] = "progress"' in all_code
        assert '"-u"' in training_cell
        assert 'PYTHONUNBUFFERED' in training_cell
        assert "capture_output=True" not in training_cell
        # Colab only renders output written through sys.stdout, so the child's
        # stdio must be piped back and re-printed as it arrives. The reads are
        # chunked rather than line-based because train.py repaints a single
        # in-place progress line with "\r", which a line iterator would buffer
        # until the end-of-epoch newline.
        assert "stdout=subprocess.PIPE" in training_cell
        assert "stderr=subprocess.STDOUT" in training_cell
        assert "bufsize=0" in training_cell
        assert '.read(4096)' in training_cell
        assert 'decode("utf-8", errors="replace")' in training_cell
        assert 'print(output_text, end="", flush=True)' in training_cell
        # The single updating line is what the notebook UI shows; the parent
        # must therefore opt into it explicitly on the resolved config.
        assert 'cfg["training"]["progress_style"] = "inline"' in all_code

    generic_path = ROOT / "notebooks" / "colab_training_notebook.ipynb"
    generic_notebook = json.loads(generic_path.read_text(encoding="utf-8"))
    generic_code = "\n".join(
        "".join(cell.get("source", []))
        for cell in generic_notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert generic_code.count('cfg["training"]["log_mode"] = "progress"') == 1
    assert "!python -u train.py --config config_mobilenetv2.yaml" in generic_code


def test_kaggle_training_notebooks_stream_live_keras_progress():
    architectures = (
        "mobilenet_v1",
        "mobilenetv2",
        "efficientnet_b0",
        "mobilenetv3_small",
        "mobilenet_v3_large",
        "nasnet_mobile",
        "efficientnet_v2_b0",
        "shufflenetv2_05",
    )
    for architecture in architectures:
        path = ROOT / "notebooks" / f"kaggle_{architecture}_training.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code_cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        training_cell = next(
            source
            for source in code_cells
            if "'train.py'" in source
            and "'--config'" in source
            and "subprocess.Popen" in source
        )

        # Kaggle's Jupyter front-end renders "\r" in place, so the child must
        # opt into the single repainted status line and its output must be
        # streamed in chunks; a line iterator would buffer every repaint
        # until the end-of-epoch newline.
        assert "training_config['training']['progress_style'] = 'inline'" in training_cell
        assert "runtime_environment['TF_CPP_MIN_LOG_LEVEL'] = '2'" in training_cell
        assert "stdout=subprocess.PIPE" in training_cell
        assert "stderr=subprocess.STDOUT" in training_cell
        assert "bufsize=0" in training_cell
        assert ".read(4096)" in training_cell
        assert "decode('utf-8', errors='replace')" in training_cell
        assert "print(output_text, end='', flush=True)" in training_cell
        assert "capture_output=True" not in training_cell
        assert "text=True" not in training_cell
        assert "for output_line in training_process.stdout" not in training_cell


def test_colab_efficientnet_b0_notebook_pins_b0_architecture():
    path = ROOT / "notebooks" / "colab_efficientnet_b0_training.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    config_cell = next(
        source for source in code_cells if "config_efficientnet_b0.yaml" in source
    )
    training_cell = next(
        source for source in code_cells if '"train.py"' in source and '"--config"' in source
    )

    compile(config_cell, f"{path}:config", "exec")
    compile(training_cell, f"{path}:training", "exec")
    assert 'ARCH = "efficientnet_b0"' in config_cell
    assert 'cfg["model"]["architecture"] = ARCH' in config_cell
    assert "cfg_path" in training_cell


def test_mobilenetv3_uses_explicit_finite_input_preprocessing():
    utils_text = (ROOT / "training" / "utils.py").read_text(encoding="utf-8")
    config_text = (
        ROOT / "training" / "config_mobilenetv3_small.yaml"
    ).read_text(encoding="utf-8")
    assert "tf.cast(x, tf.float32) / 127.5 - 1.0" in utils_text
    assert "include_preprocessing: false" in config_text
    assert "clipnorm: 1.0" in config_text
    assert "clipnorm" in (ROOT / "training" / "train.py").read_text(
        encoding="utf-8"
    )


def test_standalone_notebooks_have_valid_code_and_platform_paths():
    architectures = [
        "mobilenet_v1",
        "mobilenetv2",
        "efficientnet_b0",
        "mobilenetv3_small",
        "mobilenet_v3_large",
        "nasnet_mobile",
        "efficientnet_v2_b0",
        "shufflenetv2_05",
    ]
    shell = InteractiveShell()
    for platform in ("colab", "kaggle"):
        for architecture in architectures:
            path = ROOT / "notebooks" / f"{platform}_{architecture}_training.ipynb"
            notebook = json.loads(path.read_text(encoding="utf-8"))
            code_cells = [
                cell for cell in notebook["cells"] if cell["cell_type"] == "code"
            ]
            for index, cell in enumerate(code_cells):
                source = "".join(cell.get("source", []))
                transformed_source = shell.transform_cell(source)
                compile(transformed_source, f"{path}:cell{index}", "exec")
            code = "\n".join(
                "".join(cell.get("source", [])) for cell in code_cells
            )
            assert f'ARCH = "{architecture}"' in code
            assert "--representative_data" in code
            assert "--arch" in code
            assert "train.py" in code and "--config" in code
            if platform == "kaggle":
                assert "apply_runtime_profile" in code
                assert "platform='kaggle'" in code
                assert "CUDA_VISIBLE_DEVICES" not in code
                assert "os.environ.get('TRAINING_RUN_MODE', 'full')" in code
                assert "multi_gpu=" in code
                assert "cache_mode=" in code

            config = yaml.safe_load(
                (ROOT / "training" / f"config_{architecture}.yaml").read_text(
                    encoding="utf-8"
                )
            )
            assert config["dataset"]["cache_mode"] in {"none", "memory"}
            assert config["dataset"]["cache_memory_limit_gb"] == 8.0
            assert config["performance"]["distribution"] == "auto"

    generic_kaggle = json.loads(
        (ROOT / "notebooks" / "kaggle_training_notebook.ipynb").read_text(
            encoding="utf-8"
        )
    )
    shell = InteractiveShell()
    for notebook_path, notebook in (
        ("kaggle_training_notebook.ipynb", generic_kaggle),
        (
            "kaggle_harmonized_comparison.ipynb",
            json.loads(
                (ROOT / "notebooks" / "kaggle_harmonized_comparison.ipynb").read_text(
                    encoding="utf-8"
                )
            ),
        ),
    ):
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell.get("source", []))
            transformed_source = shell.transform_cell(source)
            compile(transformed_source, f"{notebook_path}:cell{index}", "exec")

    generic_kaggle_code = "\n".join(
        "".join(cell.get("source", []))
        for cell in generic_kaggle["cells"]
        if cell["cell_type"] == "code"
    )
    assert "apply_runtime_profile(cfg, mode='full', platform='kaggle')" in generic_kaggle_code
    assert "config_mobilenetv2_kaggle_safe.yaml" in generic_kaggle_code
    assert "config_efficientnet_b0_kaggle_safe.yaml" in generic_kaggle_code
    assert "with open(cfg_path, 'w')" not in generic_kaggle_code
    assert "CUDA_VISIBLE_DEVICES" not in generic_kaggle_code
    assert "multi_gpu=" in generic_kaggle_code
    assert "cache_mode=" in generic_kaggle_code
    assert "TF_FORCE_GPU_ALLOW_GROWTH" in generic_kaggle_code
    assert "-X faulthandler -u train.py" in generic_kaggle_code

    harmonized_kaggle = (ROOT / "notebooks" / "kaggle_harmonized_comparison.ipynb").read_text(
        encoding="utf-8"
    )
    assert "'--platform', 'kaggle'" in harmonized_kaggle
    assert "TF_FORCE_GPU_ALLOW_GROWTH" in harmonized_kaggle
    assert "CUDA_VISIBLE_DEVICES" not in harmonized_kaggle
    compare_models_text = (ROOT / "training" / "compare_models.py").read_text(
        encoding="utf-8"
    )
    assert "Kaggle runtime settings for" in compare_models_text
    assert "multi_gpu={config['performance'].get('multi_gpu', False)}" in compare_models_text
    assert "cache_mode={config['dataset'].get('cache_mode', 'none')}" in compare_models_text


def test_standalone_notebooks_separate_clone_and_dependency_install_cells():
    for platform in ("colab", "kaggle"):
        for architecture in (
            "mobilenet_v1",
            "mobilenetv2",
            "efficientnet_b0",
                "mobilenetv3_small",
            "mobilenet_v3_large",
            "nasnet_mobile",
            "efficientnet_v2_b0",
            "shufflenetv2_05",
        ):
            path = ROOT / "notebooks" / f"{platform}_{architecture}_training.ipynb"
            notebook = json.loads(path.read_text(encoding="utf-8"))
            code_cells = [
                "".join(cell.get("source", []))
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            ]
            clone_cells = [
                cell for cell in code_cells if "git" in cell and "clone" in cell
            ]
            install_cells = [
                cell
                for cell in code_cells
                if "pip" in cell and "install" in cell and "requirements.txt" in cell
            ]
            assert len(clone_cells) == 1, path
            assert len(install_cells) == 1, path
            assert "pip" not in clone_cells[0]
            assert not ("git" in install_cells[0] and "clone" in install_cells[0])
            assert "requirements.txt" in install_cells[0]


def test_all_colab_notebooks_include_kaggle_dataset_setup():
    architectures = (
        "mobilenet_v1",
        "mobilenetv2",
        "efficientnet_b0",
        "mobilenetv3_small",
        "mobilenet_v3_large",
        "nasnet_mobile",
        "efficientnet_v2_b0",
        "shufflenetv2_05",
    )
    for architecture in architectures:
        path = ROOT / "notebooks" / f"colab_{architecture}_training.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        assert "files.upload()" in code, path
        assert "mdhasibultamim/multi-crop-leaf-disease" in code, path
        assert "zipfile.ZipFile" in code, path
        assert "pip\", \"install\", \"-q\", \"kaggle\"" in code, path
        assert "\"kaggle\", \"datasets\", \"download\"" in code, path
        assert "DATASET_PATH = DATA_ROOT / \"processed\"" in code, path
        assert "DATASET_BASE = str(DATASET_PATH)" in code, path
        for split in ("train", "val", "test"):
            assert f'"{split}"' in code, path


def test_comparison_notebooks_and_common_runner_contract():
    architectures = (
        "mobilenet_v1",
        "mobilenetv2",
        "mobilenetv3_small",
        "mobilenet_v3_large",
        "nasnet_mobile",
        "efficientnet_b0",
        "efficientnet_v2_b0",
    )
    common_path = ROOT / "notebooks" / "model_comparison_common.py"
    common_tree = ast.parse(
        common_path.read_text(encoding="utf-8"), filename=str(common_path)
    )
    common_code = common_path.read_text(encoding="utf-8")
    architecture_assignment = next(
        node
        for node in common_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "ARCHITECTURES"
            for target in node.targets
        )
    )
    assert ast.literal_eval(architecture_assignment.value) == architectures
    architecture_declaration = common_code.split("PREFERRED_VARIANTS", 1)[0]
    assert '"shufflenetv2_05"' not in architecture_declaration
    for required in (
        "def discover_artifacts",
        "def evaluate_float_model",
        "def evaluate_tflite_artifact",
        "def run_comparison",
        "balanced_score",
        "best_float_accuracy",
        "best_edge_deployment",
    ):
        assert required in common_code

    for platform in ("colab", "kaggle"):
        path = ROOT / "notebooks" / f"{platform}_model_comparison.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        notebook_text = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        for architecture in architectures:
            assert architecture in common_code
            assert architecture in notebook_text
        assert "shufflenetv2_05" not in notebook_text
        assert "MAX_TEST_SAMPLES" in code
        assert "run_comparison" in code
        assert "MODEL_ROOT" in code
        assert "OUTPUT_DIR" in code


def test_kaggle_notebooks_auto_select_dedup_benchmark_and_quantize_metrics():
    for architecture in (
        "mobilenet_v1",
        "mobilenetv2",
        "mobilenetv3_small",
        "mobilenet_v3_large",
        "nasnet_mobile",
        "efficientnet_b0",
        "efficientnet_v2_b0",
        "shufflenetv2_05",
    ):
        path = ROOT / "notebooks" / f"kaggle_{architecture}_training.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        assert (
            "DEDUP_BENCHMARK_DIR = '/kaggle/working/leaf_dataset/benchmark_dedup'"
            in code
        )
        assert "'validate_dataset_splits'] = True" in code
        assert "'validate_dataset_splits'] = False" in code
        assert "allow_cross_split_duplicates'] = True" in code
        assert "from dataset.prepare_benchmark import prepare_benchmark" in code
        assert "tflite_metrics_{variant}" in code
        assert "evaluate_tflite.py" in code
        assert "runtime_cfg_path = cfg_path" in code
        assert "copytree" not in code


def test_colab_notebooks_guard_manifest_less_split_validation():
    """Colab notebooks download the Kaggle archive at runtime, so they cannot
    assume split_manifest.csv exists. Without the guard, train.py falls back
    to silently hashing every image for 10-25 minutes and then aborts on the
    disclosed degenerate Guava copies."""
    for architecture in (
        "mobilenet_v1",
        "mobilenetv2",
        "mobilenetv3_small",
        "mobilenet_v3_large",
        "nasnet_mobile",
        "efficientnet_b0",
        "efficientnet_v2_b0",
        "shufflenetv2_05",
    ):
        path = ROOT / "notebooks" / f"colab_{architecture}_training.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code_cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        config_cell = next(
            source for source in code_cells if "apply_runtime_profile" in source
        )

        assert (
            'manifest_backed = os.path.isfile(os.path.join(DATASET_BASE, "split_manifest.csv"))'
            in config_cell
        )
        assert 'cfg["performance"]["validate_dataset_splits"] = True' in config_cell
        assert 'cfg["performance"]["validate_dataset_splits"] = False' in config_cell
        assert (
            'cfg["performance"]["allow_cross_split_duplicates"] = True' in config_cell
        )
        assert '"Run mode:"' in config_cell
        # The guard must run after the runtime profile copies the config...
        assert config_cell.index("apply_runtime_profile") < config_cell.index(
            "manifest_backed"
        )
        # ...and before the resolved config is written back to disk.
        assert config_cell.index("manifest_backed") < config_cell.index("yaml.safe_dump")
        compile(config_cell, f"{path}:config", "exec")


def test_kaggle_notebooks_expose_a_filelink_download_cell():
    """Kaggle's Output panel silently refuses files over ~20 MB, which is
    smaller than every exported model archive, so each Kaggle notebook must
    offer a FileLink that streams the archive from the live session.

    FileLink renders a *relative* href: passing an absolute path produces a
    link that resolves against the Kaggle page URL and 404s, so the cell must
    pass the bare basename. It also resolves the existence check against the
    kernel working directory, and the training cells chdir into the cloned
    repo, so the download cell must chdir back to the archive's folder -- or
    IPython prints a "Path doesn't exist" notice instead of the link.
    """
    architectures = [
        "mobilenet_v1",
        "mobilenetv2",
        "mobilenetv3_small",
        "mobilenet_v3_large",
        "nasnet_mobile",
        "efficientnet_b0",
        "efficientnet_v2_b0",
        "shufflenetv2_05",
    ]
    shell = InteractiveShell()
    for architecture in architectures:
        path = ROOT / "notebooks" / f"kaggle_{architecture}_training.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = notebook["cells"]

        archive_index = next(
            index
            for index, cell in enumerate(cells)
            if "make_archive" in "".join(cell.get("source", []))
        )
        archive_source = "".join(cells[archive_index]["source"])
        variable = "archive_path" if "archive_path" in archive_source else "zip_path"

        download_cell = cells[archive_index + 1]
        assert download_cell["cell_type"] == "code"
        source = "".join(download_cell["source"])

        # The link must be served from the session using a relative filename.
        assert "FileLink(os.path.basename(%s))" % variable in source
        assert "FileLink(%s)" % variable not in source
        assert "os.path.getsize(%s)" % variable in source
        # ...after moving back to the folder that holds the archive, because
        # FileLink checks existence against the kernel working directory.
        assert "os.chdir(os.path.dirname(%s))" % variable in source
        # An archive over the panel's limit is also split into 19 MB parts.
        assert "MAX_PART_BYTES" in source
        compile(shell.transform_cell(source), f"{path}:download", "exec")

        # The surrounding documentation must state the limitation and fallback.
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in cells
            if cell["cell_type"] == "markdown"
        )
        assert "Downloading the archive:" in markdown
        assert "kaggle kernels output" in markdown


def test_all_kaggle_notebooks_offer_a_working_archive_download():
    """Every Kaggle notebook that exports an archive must also render a
    FileLink for it, and the link must use a bare filename.

    FileLink emits a *relative* href, so ``FileLink("/kaggle/working/x.zip")``
    produces a link that resolves against the Kaggle page URL and 404s. It
    also checks existence against the kernel working directory, so every link
    cell must chdir to the archive's folder first and offer 19 MB part files
    for archives above the panel's download limit.
    """
    shell = InteractiveShell()
    notebooks = [
        "kaggle_training_notebook.ipynb",
        "kaggle_model_comparison.ipynb",
        "kaggle_harmonized_comparison.ipynb",
    ]
    for name in notebooks:
        path = ROOT / "notebooks" / name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code_cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        combined = "\n".join(code_cells)

        link_cells = [cell for cell in code_cells if "FileLink(" in cell]
        assert link_cells, f"{name} renders no download link"
        for source in link_cells:
            # Only basename-relative links resolve correctly on Kaggle.
            assert "FileLink(os.path.basename(" in source, name
            assert "/kaggle/working/" not in source.split("FileLink", 1)[1].split(")")[0], name
            # FileLink checks existence against the kernel CWD, which earlier
            # cells may have changed, so the cell pins it to the archive folder.
            assert "os.chdir(os.path.dirname(" in source, name
            # Oversized archives are additionally split into servable parts.
            assert "MAX_PART_BYTES" in source, name
            compile(shell.transform_cell(source), f"{path}:link", "exec")

        # An archive must actually be produced for the link to point at; the
        # comparison notebook delegates that to model_comparison_common.py.
        archiver = "make_archive" in combined or "run_comparison" in combined
        assert archiver, f"{name} archives nothing"
        # And the documentation must warn about the Output panel limitation.
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        assert "~20 MB" in markdown, name
        assert "kaggle kernels output" in markdown, name

