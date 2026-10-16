"""Audit class balance and image overlap without modifying dataset files."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_REPORTED_PAIRS = 5000
_DCT_FREQUENCIES = np.arange(32, dtype=np.float32)
_DCT_SAMPLES = np.arange(32, dtype=np.float32)
_DCT_BASIS = np.cos(
    np.pi * (2 * _DCT_SAMPLES[None, :] + 1) * _DCT_FREQUENCIES[:, None] / 64
).astype(np.float32)
_DCT_BASIS[0] *= 1 / np.sqrt(2)
_DCT_BASIS *= np.sqrt(2 / 32)


def _image_paths(data_dir):
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    yield split, class_dir.name, path


def _perceptual_hash(path):
    with Image.open(path) as image:
        image = ImageOps.grayscale(image).resize((32, 32), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32)
    dct = _DCT_BASIS @ pixels @ _DCT_BASIS.T
    low_frequencies = dct[:8, :8].reshape(-1)
    median = np.median(low_frequencies[1:])
    value = 0
    for coefficient in low_frequencies[1:]:
        value = (value << 1) | int(coefficient > median)
    return value


def _hamming_distance(left, right):
    return (left ^ right).bit_count()


class _BKTree:
    """BK-tree index for bounded Hamming-distance perceptual-hash searches."""

    def __init__(self):
        self.root = None

    def add(self, value, record):
        if self.root is None:
            self.root = {"hash": value, "records": [record], "children": {}}
            return
        node = self.root
        while True:
            distance = _hamming_distance(value, node["hash"])
            if distance == 0:
                node["records"].append(record)
                return
            child = node["children"].get(distance)
            if child is None:
                node["children"][distance] = {
                    "hash": value,
                    "records": [record],
                    "children": {},
                }
                return
            node = child

    def find(self, value, max_distance):
        if self.root is None:
            return
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = _hamming_distance(value, node["hash"])
            if distance <= max_distance:
                for record in node["records"]:
                    yield distance, record
            lower = distance - max_distance
            upper = distance + max_distance
            pending.extend(
                child
                for edge, child in node["children"].items()
                if lower <= edge <= upper
            )


def audit_dataset(data_dir, near_duplicate_distance=8, include_near_duplicates=True):
    """Return split counts and cross-split exact/near duplicate candidates."""
    if not 0 <= near_duplicate_distance <= 63:
        raise ValueError("near_duplicate_distance must be between 0 and 63")
    data_dir = Path(data_dir).resolve()
    report = {
        "data_dir": str(data_dir),
        "class_names": {},
        "class_parity": {},
        "exact_duplicates_across_splits": [],
        "near_duplicate_candidates_across_splits": [],
        "exact_duplicate_pair_count": 0,
        "exact_duplicate_group_count": 0,
        "images_in_exact_duplicate_groups": 0,
        "near_duplicate_pair_count": 0,
        "duplicate_report_truncated": False,
        "errors": [],
    }

    class_names = {}
    for split, class_name, path in _image_paths(data_dir):
        class_names.setdefault(split, set()).add(class_name)

    reference_classes = class_names["train"]
    report["class_names"] = {
        split: sorted(classes) for split, classes in sorted(class_names.items())
    }
    report["class_parity"] = {
        split: {
            "missing": sorted(reference_classes - classes),
            "extra": sorted(classes - reference_classes),
        }
        for split, classes in sorted(class_names.items())
    }

    counts = {split: defaultdict(int) for split in ("train", "val", "test")}
    exact_index = defaultdict(list)
    perceptual_index = _BKTree()

    for split, class_name, path in _image_paths(data_dir):
        counts[split][class_name] += 1
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            for earlier in exact_index[digest]:
                if earlier["split"] != split:
                    report["exact_duplicate_pair_count"] += 1
                    if len(report["exact_duplicates_across_splits"]) < MAX_REPORTED_PAIRS:
                        report["exact_duplicates_across_splits"].append(
                            {"first": earlier["path"], "duplicate": str(path)}
                        )
                    else:
                        report["duplicate_report_truncated"] = True
            record = {
                "split": split,
                "class": class_name,
                "path": str(path),
                "sha256": digest,
            }
            exact_index[digest].append(record)

            if not include_near_duplicates:
                continue
            image_hash = _perceptual_hash(path)
            for distance, previous in perceptual_index.find(
                image_hash, near_duplicate_distance
            ):
                if previous["split"] == split or previous["sha256"] == digest:
                    continue
                report["near_duplicate_pair_count"] += 1
                if (
                    len(report["near_duplicate_candidates_across_splits"])
                    < MAX_REPORTED_PAIRS
                ):
                    report["near_duplicate_candidates_across_splits"].append(
                        {
                            "first": previous["path"],
                            "duplicate_candidate": str(path),
                            "hamming_distance": distance,
                        }
                    )
                else:
                    report["duplicate_report_truncated"] = True
            perceptual_index.add(image_hash, record)
        except (OSError, ValueError) as error:
            report["errors"].append({"path": str(path), "error": str(error)})

    for records in exact_index.values():
        split_names = {record["split"] for record in records}
        if len(split_names) > 1:
            report["exact_duplicate_group_count"] += 1
            report["images_in_exact_duplicate_groups"] += len(records)

    report["counts"] = {
        split: {
            "total": sum(class_counts.values()),
            "classes": dict(sorted(class_counts.items())),
        }
        for split, class_counts in sorted(counts.items())
    }
    report["summary"] = {
        "exact_duplicate_pairs": report["exact_duplicate_pair_count"],
        "exact_duplicate_groups": report["exact_duplicate_group_count"],
        "images_in_exact_duplicate_groups": report[
            "images_in_exact_duplicate_groups"
        ],
        "near_duplicate_candidates": report["near_duplicate_pair_count"],
        "read_errors": len(report["errors"]),
        "duplicate_report_truncated": report["duplicate_report_truncated"],
    }
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Audit train/val/test class counts and image overlap (read-only)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "processed",
        help="Directory containing train/, val/, and test/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path; stdout is used when omitted.",
    )
    parser.add_argument("--near-distance", type=int, default=8)
    parser.add_argument(
        "--no-near-duplicates",
        action="store_true",
        help="Skip perceptual hash matching and report only exact file duplicates.",
    )
    args = parser.parse_args()
    if not 0 <= args.near_distance <= 63:
        parser.error("--near-distance must be between 0 and 63")

    report = audit_dataset(
        args.data_dir,
        near_duplicate_distance=args.near_distance,
        include_near_duplicates=not args.no_near_duplicates,
    )
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(f"Audit report written to {args.output.resolve()}")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
