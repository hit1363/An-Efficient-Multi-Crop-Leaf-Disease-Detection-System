import csv
import json

import pytest
from PIL import Image

from dataset.prepare_benchmark import prepare_benchmark, _split_counts
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


def test_taxonomy_is_72_classes_and_known_aliases_are_explicit():
    assert len(CANONICAL_CLASSES) == EXPECTED_CLASS_COUNT == 72
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


def test_prepare_benchmark_is_reproducible_rgb_224_and_disjoint(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "harmonized"
    _make_source_tree(source)

    first = prepare_benchmark(source, output)
    assert first["classes"] == 72
    assert first["seed"] == 42
    assert first["counts"]["train"]["total"] == 72
    assert first["counts"]["val"]["total"] == 72
    assert first["counts"]["test"]["total"] == 72

    class_names = json.loads((output / "class_names.json").read_text())
    assert class_names == list(CANONICAL_CLASSES)
    with (output / "split_manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    assert len(manifest) == 216
    assert len({row["source_sha256"] for row in manifest}) == len(manifest)

    for split in ("train", "val", "test"):
        for class_name in CANONICAL_CLASSES:
            image_path = next((output / split / class_name).glob("*.jpg"))
            with Image.open(image_path) as image:
                assert image.mode == "RGB"
                assert image.size == (224, 224)

    second_output = tmp_path / "harmonized_second"
    second = prepare_benchmark(source, second_output)
    assert first["counts"] == second["counts"]
    first_rows = (output / "split_manifest.csv").read_text(encoding="utf-8")
    second_rows = (second_output / "split_manifest.csv").read_text(encoding="utf-8")
    assert first_rows == second_rows


def test_exact_duplicate_is_removed_and_cross_label_duplicate_is_rejected(tmp_path):
    source = tmp_path / "source"
    _make_source_tree(source)
    duplicate_class = source / CANONICAL_CLASSES[0]
    (duplicate_class / "identical_copy.png").write_bytes(
        (duplicate_class / "sample_0.png").read_bytes()
    )
    output = tmp_path / "deduplicated"
    result = prepare_benchmark(source, output)
    assert result["removed_exact_duplicates"] == 1
    assert sum(part["total"] for part in result["counts"].values()) == 216

    conflict_source = tmp_path / "conflict_source"
    _make_source_tree(conflict_source)
    original = conflict_source / CANONICAL_CLASSES[0] / "sample_0.png"
    conflict = conflict_source / CANONICAL_CLASSES[1] / "conflict.png"
    conflict.write_bytes(original.read_bytes())
    with pytest.raises(ValueError, match="Cross-label duplicate conflicts"):
        prepare_benchmark(conflict_source, tmp_path / "conflict_output")
    report = json.loads(
        (tmp_path / "conflict_output_validation_report.json").read_text()
    )
    assert report["cross_label_duplicate_conflicts"]


def test_unknown_labels_write_a_validation_report(tmp_path):
    source = tmp_path / "unknown_source"
    class_dir = source / "UnknownCrop___Disease"
    class_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4), "red").save(class_dir / "sample.png")
    with pytest.raises(ValueError, match="taxonomy validation failed"):
        prepare_benchmark(source, tmp_path / "unknown_output")
    report = json.loads(
        (tmp_path / "unknown_output_validation_report.json").read_text()
    )
    assert report["unknown_source_labels"] == ["UnknownCrop___Disease"]
