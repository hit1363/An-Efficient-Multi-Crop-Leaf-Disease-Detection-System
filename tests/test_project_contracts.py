import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _function_default(source, function_name, argument_name):
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    positional_arguments = function.args.args
    defaults = function.args.defaults
    default_arguments = positional_arguments[-len(defaults) :]
    for argument, default in zip(default_arguments, defaults):
        if argument.arg == argument_name:
            return ast.literal_eval(default)
    raise AssertionError(f"{function_name} has no default for {argument_name}")


def test_model_config_class_counts_match_dataset_metadata():
    metadata = json.loads(
        (ROOT / "dataset" / "processed" / "dataset_info.json").read_text(
            encoding="utf-8"
        )
    )
    class_count = metadata["classes"]

    for config_path in (ROOT / "training").glob("config_*.yaml"):
        source = config_path.read_text(encoding="utf-8")
        match = re.search(r"(?m)^\s*num_classes:\s*(\d+)\s*$", source)
        assert match, f"{config_path.name} does not declare model.num_classes"
        assert int(match.group(1)) == class_count, config_path.name


def test_model_factory_defaults_match_dataset_metadata():
    metadata = json.loads(
        (ROOT / "dataset" / "processed" / "dataset_info.json").read_text(
            encoding="utf-8"
        )
    )
    source = (ROOT / "training" / "model.py").read_text(encoding="utf-8")

    assert _function_default(source, "create_mobilenetv2_model", "num_classes") == metadata["classes"]
    assert _function_default(source, "create_efficientnet_model", "num_classes") == metadata["classes"]
