from pathlib import Path

import pytest

from training.dataset_audit import DatasetAuditError, audit_dataset_splits


def _image(path: Path, contents: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)


def _splits(tmp_path):
    return {split: tmp_path / split for split in ("train", "val", "test")}


def test_audit_returns_sorted_class_counts_for_clean_splits(tmp_path):
    splits = _splits(tmp_path)
    for split, contents in (("train", b"train"), ("val", b"val"), ("test", b"test")):
        _image(splits[split] / "zebra" / "image.jpg", contents)
        _image(splits[split] / "apple" / "image.png", contents + b"apple")

    result = audit_dataset_splits({key: str(value) for key, value in splits.items()})

    assert result["status"] == "passed"
    assert result["class_names"] == ["apple", "zebra"]
    assert result["splits"]["train"]["total"] == 2


def test_audit_rejects_mismatched_class_sets(tmp_path):
    splits = _splits(tmp_path)
    for split in splits:
        _image(splits[split] / "apple" / "image.jpg", split.encode())
    _image(splits["val"] / "banana" / "image.jpg", b"banana")

    with pytest.raises(DatasetAuditError, match="Class ordering mismatch"):
        audit_dataset_splits({key: str(value) for key, value in splits.items()})


def test_audit_rejects_cross_split_duplicate_images(tmp_path):
    splits = _splits(tmp_path)
    for split in splits:
        contents = b"shared" if split != "test" else b"unique"
        _image(splits[split] / "apple" / "image.jpg", contents)

    with pytest.raises(DatasetAuditError, match="byte-identical"):
        audit_dataset_splits({key: str(value) for key, value in splits.items()})
