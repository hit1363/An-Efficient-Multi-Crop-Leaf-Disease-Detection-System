"""Dependency-light contract tests for the model and notebook expansion."""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_training_and_quantization_sources_parse():
    source_files = [
        *sorted((ROOT / "training").glob("*.py")),
        *sorted((ROOT / "quantization").glob("*.py")),
    ]
    for path in source_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_all_model_configs_exist():
    architectures = [
        "mobilenetv2",
        "efficientnet_b0",
        "efficientnet_lite0",
        "mobilenetv3_small",
        "shufflenetv2_05",
    ]
    for architecture in architectures:
        config_path = ROOT / "training" / f"config_{architecture}.yaml"
        assert config_path.exists(), config_path
        config_text = config_path.read_text(encoding="utf-8")
        assert f'architecture: "{architecture}"' in config_text


def test_efficientnet_lite0_uses_the_valid_tfhub_endpoint():
    endpoint = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"
    stale_endpoint = "https://tfhub.dev/google/efficientnet/lite0/feature-vector/2"
    config_text = (
        ROOT / "training" / "config_efficientnet_lite0.yaml"
    ).read_text(encoding="utf-8")
    model_text = (ROOT / "training" / "model.py").read_text(encoding="utf-8")

    assert endpoint in config_text
    assert endpoint in model_text
    assert stale_endpoint not in config_text
    assert stale_endpoint not in model_text


def test_tfhub_gcs_download_uses_the_tarball_suffix():
    model_text = (ROOT / "training" / "model.py").read_text(encoding="utf-8")
    assert "archive_url" in model_text
    assert 'f"{gcs_url}.tar.gz"' in model_text
    assert 'origin=archive_url' in model_text


def test_optimized_runtime_defaults_are_present():
    mobile_architectures = (
        "mobilenetv2",
        "mobilenetv3_small",
        "shufflenetv2_05",
    )
    efficient_architectures = ("efficientnet_b0", "efficientnet_lite0")

    for architecture in mobile_architectures:
        config_text = (
            ROOT / "training" / f"config_{architecture}.yaml"
        ).read_text(encoding="utf-8")
        assert "batch_size: 128" in config_text
        assert "batch_size_fallbacks: [128, 64, 32]" in config_text
        assert 'cache_mode: "auto"' in config_text
        assert "cache_memory_limit_gb: 8.0" in config_text
        assert "max_weight: 5.0" in config_text
        assert "finite_batch_check: true" in config_text
        assert "terminate_on_nan: true" in config_text
        assert "distribution: auto" in config_text
        assert "log_throughput: true" in config_text

    for architecture in efficient_architectures:
        config_text = (
            ROOT / "training" / f"config_{architecture}.yaml"
        ).read_text(encoding="utf-8")
        assert "batch_size: 64" in config_text
        assert "batch_size_fallbacks: [64, 32, 16]" in config_text
        assert 'cache_mode: "auto"' in config_text
        assert "cache_memory_limit_gb: 8.0" in config_text
        assert "max_weight: 5.0" in config_text
        assert "finite_batch_check: true" in config_text
        assert "terminate_on_nan: true" in config_text
        assert "distribution: auto" in config_text

    utils_text = (ROOT / "training" / "utils.py").read_text(encoding="utf-8")
    train_text = (ROOT / "training" / "train.py").read_text(encoding="utf-8")
    assert "MirroredStrategy" in utils_text
    assert "batch_size=batch_size" in utils_text
    assert "ThroughputCallback" in utils_text
    assert "validate_dataset_batch" in utils_text
    assert "choose_batch_size" in utils_text
    assert "TerminateOnNaN" in utils_text
    assert "with strategy.scope()" in train_text


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
        "mobilenetv2",
        "efficientnet_b0",
        "efficientnet_lite0",
        "mobilenetv3_small",
        "shufflenetv2_05",
    ]
    for platform in ("colab", "kaggle"):
        for architecture in architectures:
            path = ROOT / "notebooks" / f"{platform}_{architecture}_training.ipynb"
            notebook = json.loads(path.read_text(encoding="utf-8"))
            code_cells = [
                cell for cell in notebook["cells"] if cell["cell_type"] == "code"
            ]
            for index, cell in enumerate(code_cells):
                source = "".join(cell.get("source", []))
                compile(source, f"{path}:cell{index}", "exec")
            code = "\n".join(
                "".join(cell.get("source", [])) for cell in code_cells
            )
            assert f'ARCH = "{architecture}"' in code
            assert "--representative_data" in code
            assert "--arch" in code
            assert 'cfg["dataset"]["cache_mode"] = "disk"' in code
            assert 'cfg["dataset"]["cache_memory_limit_gb"] = 8.0' in code
            assert "capture_output=True" in code or "PYTHONUNBUFFERED" in code
            assert '"distribution": "auto"' in code


def test_standalone_notebooks_separate_clone_and_dependency_install_cells():
    for platform in ("colab", "kaggle"):
        for architecture in (
            "mobilenetv2",
            "efficientnet_b0",
            "efficientnet_lite0",
            "mobilenetv3_small",
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
    for path in sorted((ROOT / "notebooks").glob("colab*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        assert "files.upload()" in code, path
        assert "mdhasibultamim/multi-crop-leaf-disease" in code, path
        assert "zipfile.ZipFile" in code, path
        assert "KAGGLE_DATASET_BASE" in code, path


def test_comparison_notebooks_and_common_runner_contract():
    architectures = (
        "mobilenetv2",
        "efficientnet_b0",
        "efficientnet_lite0",
        "mobilenetv3_small",
        "shufflenetv2_05",
    )
    common_path = ROOT / "notebooks" / "model_comparison_common.py"
    ast.parse(common_path.read_text(encoding="utf-8"), filename=str(common_path))
    common_code = common_path.read_text(encoding="utf-8")
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
        assert "MAX_TEST_SAMPLES" in code
        assert "run_comparison" in code
        assert "MODEL_ROOT" in code
        assert "OUTPUT_DIR" in code
