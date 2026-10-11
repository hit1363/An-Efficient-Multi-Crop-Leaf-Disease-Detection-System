from pathlib import Path

import pytest

from dataset.create_deduplicated_split import create_deduplicated_split


def test_deduplicated_split_keeps_exact_groups_together_and_preserves_source(
    tmp_path,
):
    source = tmp_path / "source"
    output = tmp_path / "corrected"
    contents = (b"same-image", b"image-b", b"image-c", b"image-d")
    for split in ("train", "val", "test"):
        (source / split / "healthy").mkdir(parents=True)
    for index, content in enumerate(contents):
        split = ("train", "val", "test")[index % 3]
        (source / split / "healthy" / f"image-{index}.jpg").write_bytes(content)
    (source / "val" / "healthy" / "duplicate.jpg").write_bytes(contents[0])

    info = create_deduplicated_split(source, output)

    assert info["removed_exact_duplicate_files"] == 1
    assert all(info["counts"][split]["classes"]["healthy"] > 0 for split in ("train", "val", "test"))
    assert len(list(source.rglob("*.jpg"))) == 5
    output_hash_splits = {}
    for split in ("train", "val", "test"):
        for path in (output / split / "healthy").glob("*.jpg"):
            output_hash_splits.setdefault(path.read_bytes(), set()).add(split)
    assert all(len(splits) == 1 for splits in output_hash_splits.values())


def test_deduplicated_split_refuses_to_overwrite_existing_output(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    for split in ("train", "val", "test"):
        (source / split / "healthy").mkdir(parents=True)
        (source / split / "healthy" / "image.jpg").write_bytes(split.encode())
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        create_deduplicated_split(source, output)

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_two_unique_class_images_keep_train_and_test_support(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "corrected"
    for split in ("train", "val", "test"):
        (source / split / "rare").mkdir(parents=True)
    (source / "train" / "rare" / "first.jpg").write_bytes(b"first")
    (source / "val" / "rare" / "copy.jpg").write_bytes(b"first")
    (source / "test" / "rare" / "second.jpg").write_bytes(b"second")

    info = create_deduplicated_split(source, output)

    assert info["counts"]["train"]["classes"]["rare"] == 1
    assert info["counts"]["test"]["classes"]["rare"] == 1
    assert info["classes_with_no_images_by_split"]["val"] == ["rare"]
