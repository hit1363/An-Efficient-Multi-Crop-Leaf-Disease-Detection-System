"""Strict-mode coverage for the single dataset builder in ``prepare_data``.

``prepare_dataset(..., strict=True)`` is the former dedicated benchmark
builder: it validates a flat or already-split source tree, keeps
byte-identical copies (disclosing the classes whose copies cross splits), and
writes the benchmark manifest files.
"""

import csv
import json
import shutil

import pytest
from PIL import Image

from dataset.prepare_data import _split_counts, prepare_dataset
from dataset.taxonomy import (
    CANONICAL_CLASSES,
    EXPECTED_CLASS_COUNT,
    canonicalize_label,
)


def _make_source_tree(root, images_per_class=3):
    for class_index, class_name in enumerate(CANONICAL_CLASSES):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for image_index in range(images_per_class):
            image = Image.new(
                "RGB", (8 + image_index, 7),
                (class_index, image_index * 40, (class_index + image_index) % 255),
            )
            image.save(class_dir / f"sample_{image_index}.png")


def test_taxonomy_class_count_matches_declared_expected_count():
    assert len(CANONICAL_CLASSES) == EXPECTED_CLASS_COUNT == 67
    assert CANONICAL_CLASSES[-1] == "Invalid"
    assert canonicalize_label("Mango__Anthracnose") == "Mango___Anthracnose"
    assert canonicalize_label("Potato___Healthy") == "Potato___healthy"
    assert canonicalize_label("Invalid") == "Invalid"
    with pytest.raises(ValueError, match="Unknown source label"):
        canonicalize_label("Unmapped___Disease")


def test_split_count_rounding_preserves_coverage_when_possible():
    for total in range(3, 30):
        counts = _split_counts(total)
        assert sum(counts.values()) == total
        assert all(count > 0 for count in counts.values())


def test_strict_build_is_reproducible_rgb_224_and_disjoint(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "harmonized"
    _make_source_tree(source)

    first = prepare_dataset(source, output, strict=True)
    assert first["classes_count"] == EXPECTED_CLASS_COUNT
    assert first["train"]["total"] == EXPECTED_CLASS_COUNT
    assert first["val"]["total"] == EXPECTED_CLASS_COUNT
    assert first["test"]["total"] == EXPECTED_CLASS_COUNT

    class_names = json.loads((output / "class_names.json").read_text())
    assert class_names == list(CANONICAL_CLASSES)
    with (output / "split_manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    assert len(manifest) == 3 * EXPECTED_CLASS_COUNT
    assert len({row["source_sha256"] for row in manifest}) == len(manifest)

    for split in ("train", "val", "test"):
        for class_name in CANONICAL_CLASSES:
            image_path = next((output / split / class_name).glob("*.jpg"))
            with Image.open(image_path) as image:
                assert image.mode == "RGB"
                assert image.size == (224, 224)

    second_output = tmp_path / "harmonized_second"
    second = prepare_dataset(source, second_output, strict=True)
    assert first["train"]["total"] == second["train"]["total"]
    first_rows = (output / "split_manifest.csv").read_text(encoding="utf-8")
    second_rows = (second_output / "split_manifest.csv").read_text(encoding="utf-8")
    assert first_rows == second_rows


def test_exact_duplicate_is_kept_and_cross_label_duplicate_is_rejected(tmp_path):
    source = tmp_path / "source"
    _make_source_tree(source)
    duplicate_class = source / CANONICAL_CLASSES[0]
    (duplicate_class / "identical_copy.png").write_bytes(
        (duplicate_class / "sample_0.png").read_bytes()
    )
    output = tmp_path / "deduplicated"
    result = prepare_dataset(source, output, strict=True)
    assert result["exact_duplicates_kept"] == 1
    assert result["processed_total"] == 3 * EXPECTED_CLASS_COUNT + 1

    conflict_source = tmp_path / "conflict_source"
    _make_source_tree(conflict_source)
    original = conflict_source / CANONICAL_CLASSES[0] / "sample_0.png"
    conflict = conflict_source / CANONICAL_CLASSES[1] / "conflict.png"
    conflict.write_bytes(original.read_bytes())
    with pytest.raises(ValueError, match="Cross-label duplicate conflicts"):
        prepare_dataset(conflict_source, tmp_path / "conflict_output", strict=True)
    report = json.loads(
        (tmp_path / "conflict_output_validation_report.json").read_text()
    )
    assert report["cross_label_duplicate_conflicts"]


def test_split_tree_duplicates_are_kept_and_reuse_is_disclosed(tmp_path):
    source = tmp_path / "split_source"
    for split_index, split in enumerate(("train", "val", "test")):
        for class_index, class_name in enumerate(CANONICAL_CLASSES):
            class_dir = source / split / class_name
            class_dir.mkdir(parents=True)
            image = Image.new(
                "RGB",
                (8, 8),
                (class_index, split_index * 40, (class_index + split_index) % 255),
            )
            image.save(class_dir / "sample.png")

    duplicate_source = source / "train" / CANONICAL_CLASSES[0] / "sample.png"
    (source / "val" / CANONICAL_CLASSES[0] / "duplicate.png").write_bytes(
        duplicate_source.read_bytes()
    )

    output = tmp_path / "deduplicated"
    result = prepare_dataset(source, output, strict=True)

    assert result["exact_duplicates_kept"] == 1
    with (output / "split_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        manifest = list(csv.DictReader(handle))
    assert len(manifest) == 3 * EXPECTED_CLASS_COUNT + 1
    assert len({row["source_sha256"] for row in manifest}) == 3 * EXPECTED_CLASS_COUNT
    # Mirror the split validator's tolerance rule: a digest may only appear
    # in several splits when its class is registered in degenerate_classes.
    registry = set(result["degenerate_classes"])
    digest_splits = {}
    digest_labels = {}
    for row in manifest:
        digest_splits.setdefault(row["source_sha256"], set()).add(row["split"])
        digest_labels.setdefault(row["source_sha256"], set()).add(
            row["canonical_label"]
        )
    for digest, splits in digest_splits.items():
        if len(splits) > 1:
            assert digest_labels[digest] <= registry


def test_duplicate_heavy_class_stays_balanced(tmp_path):
    """The degenerate-class scenario: thousands of copies of a few photos
    must not collapse the class to its unique-image count."""
    source = tmp_path / "source"
    _make_source_tree(source, images_per_class=3)
    target = source / CANONICAL_CLASSES[5]
    base_images = sorted(target.glob("*.png"))
    for copy_index in range(60):
        shutil.copy(
            base_images[copy_index % len(base_images)],
            target / f"copy_{copy_index:03d}.png",
        )

    output = tmp_path / "balanced"
    first = prepare_dataset(source, output, strict=True)

    # 63 files: prepare_data.split_selected -> 44 / 9 / 10, not 3 unique.
    assert first["train"]["classes"][CANONICAL_CLASSES[5]] == 44
    assert first["val"]["classes"][CANONICAL_CLASSES[5]] == 9
    assert first["test"]["classes"][CANONICAL_CLASSES[5]] == 10
    assert first["degenerate_classes"] == {CANONICAL_CLASSES[5]: 3}
    # All-unique classes keep the non-empty fallback split (1/1/1).
    assert first["train"]["classes"][CANONICAL_CLASSES[6]] == 1

    info = json.loads((output / "dataset_info.json").read_text())
    assert info["degenerate_classes"] == {CANONICAL_CLASSES[5]: 3}
    assert info["exact_duplicates_kept"] == 60

    second_output = tmp_path / "balanced_second"
    second = prepare_dataset(source, second_output, strict=True)
    assert first["train"]["classes"] == second["train"]["classes"]
    assert (
        (output / "split_manifest.csv").read_text(encoding="utf-8")
        == (second_output / "split_manifest.csv").read_text(encoding="utf-8")
    )


def test_unknown_labels_write_a_validation_report(tmp_path):
    source = tmp_path / "unknown_source"
    class_dir = source / "UnknownCrop___Disease"
    class_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4), "red").save(class_dir / "sample.png")
    with pytest.raises(ValueError, match="taxonomy validation failed"):
        prepare_dataset(source, tmp_path / "unknown_output", strict=True)
    report = json.loads(
        (tmp_path / "unknown_output_validation_report.json").read_text()
    )
    assert report["unknown_source_labels"] == ["UnknownCrop___Disease"]
    assert not (tmp_path / "unknown_output").exists()


def test_degenerate_class_is_kept_with_disclosed_val_test_copies(tmp_path):
    source = tmp_path / "source"
    _make_source_tree(source, images_per_class=4)
    thin_class = source / CANONICAL_CLASSES[3]
    shutil.rmtree(thin_class)
    thin_class.mkdir()
    Image.new("RGB", (9, 9), (1, 2, 3)).save(thin_class / "only_a.png")
    Image.new("RGB", (9, 9), (4, 5, 6)).save(thin_class / "only_b.png")

    output = tmp_path / "dedup"
    result = prepare_dataset(source, output, strict=True)

    assert result["degenerate_classes"] == {CANONICAL_CLASSES[3]: 2}
    assert result["train"]["classes"][CANONICAL_CLASSES[3]] == 2
    assert result["val"]["classes"][CANONICAL_CLASSES[3]] == 1
    assert result["test"]["classes"][CANONICAL_CLASSES[3]] == 1

    info = json.loads((output / "dataset_info.json").read_text())
    assert info["degenerate_classes"] == {CANONICAL_CLASSES[3]: 2}

    with (output / "split_manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))

    thin_rows = [r for r in manifest if r["canonical_label"] == CANONICAL_CLASSES[3]]
    assert len(thin_rows) == 4
    assert len({r["source_sha256"] for r in thin_rows}) == 2

    other_rows = [r for r in manifest if r["canonical_label"] != CANONICAL_CLASSES[3]]
    assert len({r["source_sha256"] for r in other_rows}) == len(other_rows)


def test_strict_build_refuses_to_overwrite_and_requires_manifest(tmp_path):
    source = tmp_path / "source"
    _make_source_tree(source, images_per_class=3)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        prepare_dataset(source, output, strict=True)
    with pytest.raises(ValueError, match="write_manifest"):
        prepare_dataset(source, tmp_path / "fresh", strict=True, write_manifest=False)
    assert not (tmp_path / "fresh").exists()
