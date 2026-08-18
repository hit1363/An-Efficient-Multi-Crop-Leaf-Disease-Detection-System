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
        assert 'cache_mode: "auto"' in config_text
        assert "distribution: auto" in config_text
        assert "log_throughput: true" in config_text

    for architecture in efficient_architectures:
        config_text = (
            ROOT / "training" / f"config_{architecture}.yaml"
        ).read_text(encoding="utf-8")
        assert "batch_size: 64" in config_text
        assert 'cache_mode: "auto"' in config_text
        assert "distribution: auto" in config_text

    utils_text = (ROOT / "training" / "utils.py").read_text(encoding="utf-8")
    train_text = (ROOT / "training" / "train.py").read_text(encoding="utf-8")
    assert "MirroredStrategy" in utils_text
    assert "batch_size=batch_size" in utils_text
    assert "ThroughputCallback" in utils_text
    assert "with strategy.scope()" in train_text


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
            assert 'cfg["dataset"]["cache_mode"] = "auto"' in code
            assert '"distribution": "auto"' in code
