import csv
import hashlib
import json
from pathlib import Path

from dataset.prepare_data import prepare_dataset
from dataset.prepare_benchmark import prepare_benchmark
from dataset.taxonomy import CANONICAL_CLASSES, EXPECTED_CLASS_COUNT

import pytest
from PIL import Image


def _make_raw_tree(root, images_per_class=3):
    for class_index, class_name in enumerate(CANONICAL_CLASSES):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for image_index in range(images_per_class):
            Image.new(
                "RGB",
                (8 + image_index, 7),
                (class_index, image_index * 40, (class_index + image_index) % 255),
            ).save(class_dir / f"sample_{image_index}.png")


def _read_manifest(output):
    with (output / "split_manifest.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_prepare_dataset_writes_benchmark_manifest_files(tmp_path):
    raw = tmp_path / "raw"
    output = tmp_path / "processed"
    _make_raw_tree(raw)

    stats = prepare_dataset(raw, output, write_manifest=True)

    class_names = json.loads((output / "class_names.json").read_text())
    assert class_names == list(CANONICAL_CLASSES)
    assert len(class_names) == EXPECTED_CLASS_COUNT == 72

    taxonomy = json.loads((output / "taxonomy.json").read_text())
    assert taxonomy["version"] == 1
    assert taxonomy["expected_class_count"] == EXPECTED_CLASS_COUNT

    audit = json.loads((output / "near_duplicate_audit.json").read_text())
    assert audit == {"status": "not_run", "reason": "exact duplicate audit only"}

    manifest = _read_manifest(output)
    assert len(manifest) == stats["processed_total"]
    assert len(manifest) == 216  # 72 classes x 3 images

    # Row fields must match the validator's contract exactly, and every
    # recorded digest must be the SHA-256 of the corresponding source file.
    assert list(manifest[0].keys()) == [
        "split",
        "canonical_label",
        "output_path",
        "source_sha256",
    ]
    for row in manifest:
        source = raw / row["canonical_label"] / (
            Path(row["output_path"]).name
        )
        # Output keeps the source stem; map back by stem instead of name.
        matches = [
            p
            for p in (raw / row["canonical_label"]).iterdir()
            if p.stem == source.stem
        ]
        assert len(matches) == 1
        digest = hashlib.sha256(matches[0].read_bytes()).hexdigest()
        assert row["source_sha256"] == digest
        assert row["split"] in {"train", "val", "test"}
        assert (output / row["output_path"]).is_file()


def test_prepare_dataset_manifest_matches_prepare_benchmark_shape(tmp_path):
    raw = tmp_path / "raw"
    _make_raw_tree(raw)

    pd_output = tmp_path / "pd_out"
    prepare_dataset(raw, pd_output, write_manifest=True)

    benchmark_output = tmp_path / "bench_out"
    prepare_benchmark(raw, benchmark_output, max_per_class=3)

    pd_names = json.loads((pd_output / "class_names.json").read_text())
    bench_names = json.loads((benchmark_output / "class_names.json").read_text())
    assert pd_names == bench_names

    pd_taxonomy = json.loads((pd_output / "taxonomy.json").read_text())
    bench_taxonomy = json.loads((benchmark_output / "taxonomy.json").read_text())
    assert pd_taxonomy == bench_taxonomy

    pd_manifest = _read_manifest(pd_output)
    bench_manifest_reader = csv.DictReader(
        (benchmark_output / "split_manifest.csv").open(newline="", encoding="utf-8")
    )
    assert list(pd_manifest[0].keys()) == list(bench_manifest_reader.fieldnames)


def test_prepare_dataset_write_manifest_false_skips_files(tmp_path):
    raw = tmp_path / "raw"
    output = tmp_path / "processed"
    _make_raw_tree(raw)

    prepare_dataset(raw, output, write_manifest=False)

    for name in (
        "class_names.json",
        "split_manifest.csv",
        "taxonomy.json",
        "near_duplicate_audit.json",
    ):
        assert not (output / name).exists(), f"{name} should not be written"
    # The split tree itself is still produced.
    assert (output / "train" / CANONICAL_CLASSES[0]).is_dir()


def test_prepare_dataset_manifest_is_deterministic(tmp_path):
    raw = tmp_path / "raw"
    _make_raw_tree(raw, images_per_class=4)

    first = tmp_path / "first"
    second = tmp_path / "second"
    prepare_dataset(raw, first, write_manifest=True)
    prepare_dataset(raw, second, write_manifest=True)

    assert (
        (first / "split_manifest.csv").read_text(encoding="utf-8")
        == (second / "split_manifest.csv").read_text(encoding="utf-8")
    )
    assert (
        (first / "class_names.json").read_text(encoding="utf-8")
        == (second / "class_names.json").read_text(encoding="utf-8")
    )


def test_cli_no_manifest_flag_exists():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "dataset.prepare_data", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--no-manifest" in result.stdout
