"""Colab dataset discovery and validation helpers for the training notebooks."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from .taxonomy import CANONICAL_CLASSES, canonicalize_label
except ImportError:  # Running as a standalone module.
    from taxonomy import CANONICAL_CLASSES, canonicalize_label


def _validate_benchmark(dataset_dir: Path) -> list[str]:
    """Validate benchmark manifests, split folders, and referenced images."""
    class_manifest = dataset_dir / "class_names.json"
    split_manifest = dataset_dir / "split_manifest.csv"
    if not class_manifest.is_file() or not split_manifest.is_file():
        raise ValueError(
            "Existing benchmark is missing class_names.json or split_manifest.csv: "
            f"{dataset_dir}"
        )

    try:
        names = json.loads(class_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read canonical class manifest: {class_manifest}") from exc
    if names != list(CANONICAL_CLASSES):
        raise ValueError(
            f"Invalid canonical class manifest at {class_manifest}; expected the "
            "repository's ordered 72-class taxonomy."
        )

    expected = set(CANONICAL_CLASSES)
    for split in ("train", "val", "test"):
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            raise ValueError(f"Benchmark split is missing: {split_dir}")
        actual = {child.name for child in split_dir.iterdir() if child.is_dir()}
        if actual != expected:
            raise ValueError(
                f"{split} folders do not match the canonical taxonomy; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )

    seen_hashes: set[str] = set()
    covered_splits: set[str] = set()
    train_classes: set[str] = set()
    row_count = 0
    with split_manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "canonical_label", "source_sha256", "output_path"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Invalid split manifest columns: {split_manifest}")
        for row in reader:
            split = row["split"]
            label = row["canonical_label"]
            digest = row["source_sha256"]
            relative_path = Path(row["output_path"])
            if (
                split not in {"train", "val", "test"}
                or label not in expected
                or not digest
                or relative_path.is_absolute()
                or len(relative_path.parts) < 3
                or relative_path.parts[:2] != (split, label)
            ):
                raise ValueError(f"Invalid split manifest row: {row}")
            if not (dataset_dir / relative_path).is_file():
                raise ValueError(f"Split manifest points to a missing image: {relative_path}")
            if digest in seen_hashes:
                raise ValueError("Benchmark manifest contains duplicate source image hashes")
            seen_hashes.add(digest)
            covered_splits.add(split)
            if split == "train":
                train_classes.add(label)
            row_count += 1

    if row_count == 0 or covered_splits != {"train", "val", "test"}:
        raise ValueError(f"Split manifest is empty or missing a split: {split_manifest}")
    if train_classes != expected:
        raise ValueError("Training split manifest does not cover all canonical classes")
    return names


def _find_source_root(search_root: Path) -> Path:
    """Find one source root containing all splits or recognized class folders."""
    split_roots: list[Path] = []
    flat_roots: list[tuple[int, Path]] = []
    for root, directories, _ in os.walk(search_root):
        root_path = Path(root)
        if {"train", "val", "test"}.issubset(directories):
            split_roots.append(root_path)
            continue
        recognized = set()
        for name in directories:
            try:
                recognized.add(canonicalize_label(name))
            except ValueError:
                continue
        if recognized:
            flat_roots.append((len(recognized), root_path))

    if len(split_roots) == 1:
        return split_roots[0]
    if len(split_roots) > 1:
        raise RuntimeError(
            "Found multiple train/val/test dataset roots: "
            + ", ".join(str(path) for path in split_roots)
        )
    if not flat_roots:
        raise FileNotFoundError(
            f"No flat class-folder dataset or train/val/test tree found under {search_root}"
        )
    best_count = max(count for count, _ in flat_roots)
    best_roots = [path for count, path in flat_roots if count == best_count]
    if len(best_roots) != 1:
        raise RuntimeError(
            "Found multiple possible flat dataset roots: "
            + ", ".join(str(path) for path in best_roots)
        )
    return best_roots[0]


def prepare_colab_benchmark(repo_dir, source_dir, output_dir):
    """Reuse a valid runtime benchmark or build one from the downloaded source.

    Invalid existing outputs are preserved for inspection; rebuilding uses a new
    temporary location so a partial directory cannot block dataset preparation.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    try:
        class_names = _validate_benchmark(output_dir)
    except (OSError, ValueError) as exc:
        if output_dir.exists():
            print(f"Existing benchmark is invalid ({exc}); preserving it and rebuilding.")
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            parent = Path(tempfile.mkdtemp(prefix="leaf_benchmark_", dir=output_dir.parent))
            output_dir = parent / "processed_harmonized"
        else:
            output_dir.parent.mkdir(parents=True, exist_ok=True)
        source_root = source_dir / "processed" if (source_dir / "processed").is_dir() else source_dir
        source_root = _find_source_root(source_root)
        report_path = output_dir.with_name(output_dir.name + "_validation_report.json")
        preparation = subprocess.run(
            [
                sys.executable,
                str(Path(repo_dir) / "dataset" / "prepare_benchmark.py"),
                "--data-dir",
                str(source_root),
                "--output-dir",
                str(output_dir),
                "--seed",
                "42",
            ],
            text=True,
            capture_output=True,
        )
        if preparation.stdout:
            print(preparation.stdout)
        if preparation.stderr:
            print(preparation.stderr)
        if preparation.returncode:
            report_message = (
                f"See {report_path} for the full report."
                if report_path.is_file()
                else "Review the preparation output above for details."
            )
            raise RuntimeError(
                "Canonical benchmark preparation failed. The source must provide all "
                f"72 canonical classes. {report_message}"
            )
        class_names = _validate_benchmark(output_dir)
        print(f"Prepared and validated benchmark: {output_dir} ({len(class_names)} classes)")
    else:
        print(f"Reusing validated benchmark: {output_dir} ({len(class_names)} classes)")

    return str(output_dir), class_names
