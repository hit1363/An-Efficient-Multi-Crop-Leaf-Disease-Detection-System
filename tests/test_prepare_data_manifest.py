import csv
import hashlib
import json
from pathlib import Path

from dataset.prepare_data import prepare_dataset
from dataset.taxonomy import CANONICAL_CLASSES, EXPECTED_CLASS_COUNT, canonicalize_label

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
    assert len(class_names) == EXPECTED_CLASS_COUNT == 67

    taxonomy = json.loads((output / "taxonomy.json").read_text())
    assert taxonomy["version"] == 1
    assert taxonomy["expected_class_count"] == EXPECTED_CLASS_COUNT

    audit = json.loads((output / "near_duplicate_audit.json").read_text())
    assert audit == {"status": "not_run", "reason": "exact duplicate audit only"}

    manifest = _read_manifest(output)
    assert len(manifest) == stats["processed_total"]
    assert len(manifest) == 3 * EXPECTED_CLASS_COUNT  # 67 classes x 3 images

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


def test_strict_and_tolerant_builds_share_the_manifest_shape(tmp_path):
    """Both modes of the single builder must emit the same manifest contract."""
    raw = tmp_path / "raw"
    _make_raw_tree(raw)

    tolerant_output = tmp_path / "tolerant_out"
    prepare_dataset(raw, tolerant_output, write_manifest=True)

    strict_output = tmp_path / "strict_out"
    prepare_dataset(raw, strict_output, write_manifest=True, strict=True)

    tolerant_names = json.loads((tolerant_output / "class_names.json").read_text())
    strict_names = json.loads((strict_output / "class_names.json").read_text())
    assert tolerant_names == strict_names == list(CANONICAL_CLASSES)

    tolerant_taxonomy = json.loads((tolerant_output / "taxonomy.json").read_text())
    strict_taxonomy = json.loads((strict_output / "taxonomy.json").read_text())
    assert tolerant_taxonomy == strict_taxonomy

    stricter_fields = csv.DictReader(
        (strict_output / "split_manifest.csv").open(newline="", encoding="utf-8")
    ).fieldnames
    assert list(_read_manifest(tolerant_output)[0].keys()) == list(stricter_fields)


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


def test_cli_exposes_no_manifest_and_strict_flags():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "dataset.prepare_data", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--no-manifest" in result.stdout
    assert "--strict" in result.stdout


def test_duplicate_heavy_class_is_disclosed_and_validates(tmp_path):
    """prepare_dataset must register classes whose copies cross splits.

    ``training.utils.validate_dataset_splits`` treats a source digest that
    appears in more than one split as leakage unless the label is listed in
    ``dataset_info.json`` under ``degenerate_classes``. Without the registry a
    tree built from duplicate-heavy source folders is rejected at training
    time, so the disclosure has to come from the builder itself.
    """
    from dataset.prepare_data import save_dataset_info
    from training.utils import validate_dataset_splits

    raw = tmp_path / "raw"
    base = tmp_path / "base"
    _make_raw_tree(base, images_per_class=3)

    raw.mkdir()
    for class_dir in base.iterdir():
        target = raw / class_dir.name
        target.mkdir()
        for image in class_dir.iterdir():
            (target / image.name).write_bytes(image.read_bytes())

    # 60 files but only two distinct images: both digests land in every split
    # (42/9/9), which the validator would otherwise report as leakage.
    duplicate_class = CANONICAL_CLASSES[3]
    duplicate_dir = raw / duplicate_class
    for image_index in range(2):
        source = duplicate_dir / f"sample_{image_index}.png"
        payload = source.read_bytes()
        for copy_index in range(29):
            (duplicate_dir / f"copy_{image_index}_{copy_index:02d}.png").write_bytes(
                payload
            )

    output = tmp_path / "processed"
    stats = prepare_dataset(raw, output, write_manifest=True)
    save_dataset_info(stats, output, 42)

    # Three distinct payloads (the class's three base images), each copied many
    # times, so the class is registered with its unique-image count.
    expected_registry = {canonicalize_label(duplicate_class): 3}
    assert stats["degenerate_classes"] == expected_registry
    info = json.loads((output / "dataset_info.json").read_text())
    assert info["degenerate_classes"] == expected_registry

    counts = validate_dataset_splits(
        str(output / "train"),
        str(output / "val"),
        str(output / "test"),
        class_names=list(CANONICAL_CLASSES),
    )
    assert (
        counts["train"]["classes"][canonicalize_label(duplicate_class)]
        == stats["train"]["classes"][duplicate_class]
    )

