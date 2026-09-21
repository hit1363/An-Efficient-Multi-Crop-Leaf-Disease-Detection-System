"""Integrity checks for directory-based image classification splits."""

from __future__ import annotations

import hashlib
import os
from itertools import combinations
from typing import Dict, Iterable, Mapping


class DatasetAuditError(ValueError):
    """Raised when split structure or isolation is unsafe for training."""


def _classes(split_dir: str) -> list[str]:
    if not os.path.isdir(split_dir):
        raise DatasetAuditError(f"Dataset split directory does not exist: {split_dir}")
    return sorted(
        entry.name
        for entry in os.scandir(split_dir)
        if entry.is_dir()
    )


def _iter_files(split_dir: str, class_names: Iterable[str]):
    for class_name in class_names:
        class_dir = os.path.join(split_dir, class_name)
        for root, _, filenames in os.walk(class_dir):
            for filename in sorted(filenames):
                yield class_name, os.path.join(root, filename)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as image_file:
        for block in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_dataset_splits(split_dirs: Mapping[str, str]) -> Dict[str, object]:
    """Validate class ordering and reject byte-identical cross-split images.

    ``split_dirs`` must contain the training, validation, and test directories.
    Files are hashed incrementally so the audit does not load complete images into
    memory. Duplicate files within the same split are reported but do not fail the
    audit; only cross-split duplicates contaminate validation or test metrics.
    """
    required_splits = ("train", "val", "test")
    missing = [split for split in required_splits if split not in split_dirs]
    if missing:
        raise DatasetAuditError(f"Missing required dataset splits: {', '.join(missing)}")

    class_names = {split: _classes(split_dirs[split]) for split in required_splits}
    expected = class_names["train"]
    for split in ("val", "test"):
        if class_names[split] != expected:
            raise DatasetAuditError(
                f"Class ordering mismatch between train and {split}: "
                f"train has {len(expected)} classes and {split} has "
                f"{len(class_names[split])}."
            )

    hashes_by_split: Dict[str, Dict[str, list[str]]] = {}
    split_counts: Dict[str, Dict[str, object]] = {}
    for split in required_splits:
        hashes: Dict[str, list[str]] = {}
        counts = {class_name: 0 for class_name in expected}
        for class_name, path in _iter_files(split_dirs[split], expected):
            counts[class_name] += 1
            hashes.setdefault(_sha256(path), []).append(path)
        hashes_by_split[split] = hashes
        split_counts[split] = {
            "total": sum(counts.values()),
            "classes": counts,
            "within_split_duplicate_files": sum(
                len(paths) - 1 for paths in hashes.values() if len(paths) > 1
            ),
        }

    cross_split_duplicates = []
    for first, second in combinations(required_splits, 2):
        shared_hashes = hashes_by_split[first].keys() & hashes_by_split[second].keys()
        for image_hash in sorted(shared_hashes):
            cross_split_duplicates.append(
                {
                    "splits": [first, second],
                    "sha256": image_hash,
                    "paths": {
                        first: hashes_by_split[first][image_hash],
                        second: hashes_by_split[second][image_hash],
                    },
                }
            )

    if cross_split_duplicates:
        examples = cross_split_duplicates[:3]
        raise DatasetAuditError(
            "Found %d byte-identical image groups across dataset splits. "
            "Correct the processed split before training. Examples: %s"
            % (len(cross_split_duplicates), examples)
        )

    return {
        "status": "passed",
        "class_names": expected,
        "splits": split_counts,
        "cross_split_duplicate_groups": 0,
    }
