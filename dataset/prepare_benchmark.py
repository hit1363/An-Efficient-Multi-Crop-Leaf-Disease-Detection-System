"""Build the deterministic, canonical 72-class leaf disease benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

try:
    from .audit_splits import audit_dataset
    from .taxonomy import (
        CANONICAL_CLASSES,
        EXPECTED_CLASS_COUNT,
        canonicalize_label,
        taxonomy_document,
    )
except ImportError:  # Running as ``python dataset/prepare_benchmark.py``.
    from audit_splits import audit_dataset
    from taxonomy import (
        CANONICAL_CLASSES,
        EXPECTED_CLASS_COUNT,
        canonicalize_label,
        taxonomy_document,
    )


IMAGE_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".gif",
    ".ico",
    ".jfif",
    ".jpe",
    ".jpeg",
    ".jpg",
    ".pbm",
    ".pgm",
    ".png",
    ".pnm",
    ".ppm",
    ".tga",
    ".tif",
    ".tiff",
    ".webp",
}
SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_class_dirs(data_dir: Path):
    """Yield (source label, directory) for flat or already-split input trees."""
    split_dirs = [data_dir / split for split in SPLITS]
    if all(path.is_dir() for path in split_dirs):
        for split_dir in split_dirs:
            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                yield class_dir.name, class_dir
        return

    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        if class_dir.name in SPLITS:
            raise ValueError(
                f"Input has an incomplete split tree at {data_dir}; expected train, val, and test"
            )
        yield class_dir.name, class_dir


def _collect_images(data_dir: Path):
    """Collect recognized source classes and image files in stable order."""
    records = []
    unknown_labels = []
    source_classes = set()
    for source_label, class_dir in _source_class_dirs(data_dir):
        source_classes.add(source_label)
        try:
            canonical = canonicalize_label(source_label)
        except ValueError:
            unknown_labels.append(source_label)
            continue
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(
                    {
                        "source_label": source_label,
                        "canonical_label": canonical,
                        "path": path,
                    }
                )
    return records, sorted(source_classes), sorted(set(unknown_labels))


def _split_counts(count: int, ratios=DEFAULT_RATIOS):
    """Allocate per-class counts, preserving class coverage when possible."""
    if count == 1:
        return {"train": 1, "val": 0, "test": 0}
    if count == 2:
        return {"train": 1, "val": 0, "test": 1}
    train = int(count * ratios[0])
    val = int(count * ratios[1])
    test = count - train - val
    if count >= 3:
        while train == 0 or val == 0 or test == 0:
            if train == 0:
                train += 1
                if test > 1:
                    test -= 1
                else:
                    val -= 1
            elif val == 0:
                val += 1
                if train > 1:
                    train -= 1
                else:
                    test -= 1
            elif test == 0:
                test += 1
                if train > 1:
                    train -= 1
                else:
                    val -= 1
    return {"train": train, "val": val, "test": test}


def _partition_by_class(records, seed=42):
    grouped = {name: [] for name in CANONICAL_CLASSES}
    for record in records:
        grouped[record["canonical_label"]].append(record)

    assignments = []
    for class_name in CANONICAL_CLASSES:
        class_records = sorted(grouped[class_name], key=lambda item: str(item["path"]))
        random.Random(f"{seed}:{class_name}").shuffle(class_records)
        counts = _split_counts(len(class_records))
        start = 0
        for split in SPLITS:
            end = start + counts[split]
            assignments.extend((split, record) for record in class_records[start:end])
            start = end
    return assignments


def _write_image(source: Path, destination: Path):
    with Image.open(source) as image:
        image.load()
        rgb = image.convert("RGB")
        fitted = ImageOps.fit(
            rgb,
            (224, 224),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        fitted.save(destination, format="JPEG", quality=95, optimize=True)


def _write_json(path: Path, document):
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_benchmark(
    data_dir,
    output_dir,
    seed=42,
    expected_classes=EXPECTED_CLASS_COUNT,
    allow_class_count_override=False,
):
    """Canonicalize, remove exact duplicates, preprocess, and split images.

    The source tree is read-only. The output directory must not already exist.
    A neighboring validation report is written when label or image errors are
    found so the caller has an actionable record of the failure.
    """
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Input dataset directory does not exist: {data_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if not 0 <= seed:
        raise ValueError("seed must be a non-negative integer")
    if expected_classes != len(CANONICAL_CLASSES) and not allow_class_count_override:
        raise ValueError(
            f"Taxonomy defines {len(CANONICAL_CLASSES)} classes, not {expected_classes}"
        )

    records, source_classes, unknown_labels = _collect_images(data_dir)
    canonical_present = {record["canonical_label"] for record in records}
    missing_classes = sorted(set(CANONICAL_CLASSES) - canonical_present)
    report_path = output_dir.with_name(output_dir.name + "_validation_report.json")
    validation = {
        "input_directory": str(data_dir),
        "unknown_source_labels": unknown_labels,
        "missing_canonical_classes": missing_classes,
        "source_class_count": len(source_classes),
        "canonical_class_count": len(canonical_present),
        "errors": [],
    }
    if unknown_labels or missing_classes or len(canonical_present) != expected_classes:
        validation["errors"].append("Source labels do not resolve to the required taxonomy")
        _write_json(report_path, validation)
        raise ValueError(f"Dataset taxonomy validation failed; see {report_path}")
    if not records:
        validation["errors"].append("No supported image files were found")
        _write_json(report_path, validation)
        raise ValueError(f"No supported image files found; see {report_path}")

    # Hash raw source bytes before any lossy conversion. A byte-identical image
    # assigned to multiple labels is a taxonomy conflict, not a silently chosen
    # training label.
    unique_by_hash = {}
    duplicate_sources = []
    cross_label_conflicts = []
    corrupt_images = []
    for index, record in enumerate(records, start=1):
        try:
            record["source_sha256"] = _sha256(record["path"])
            with Image.open(record["path"]) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            corrupt_images.append({"path": str(record["path"]), "error": str(exc)})
            continue
        digest = record["source_sha256"]
        previous = unique_by_hash.get(digest)
        if previous is None:
            unique_by_hash[digest] = record
        elif previous["canonical_label"] != record["canonical_label"]:
            cross_label_conflicts.append(
                {
                    "sha256": digest,
                    "first_label": previous["canonical_label"],
                    "first_path": str(previous["path"]),
                    "conflicting_label": record["canonical_label"],
                    "conflicting_path": str(record["path"]),
                }
            )
        else:
            duplicate_sources.append(str(record["path"]))
        if index % 5000 == 0 or index == len(records):
            print(f"Validated source images: {index}/{len(records)}", flush=True)

    if cross_label_conflicts:
        validation["cross_label_duplicate_conflicts"] = cross_label_conflicts
        validation["corrupt_images"] = corrupt_images
        validation["errors"].append("Exact image duplicates have conflicting labels")
        _write_json(report_path, validation)
        raise ValueError(f"Cross-label duplicate conflicts found; see {report_path}")

    valid_records = list(unique_by_hash.values())
    empty_classes = sorted(
        name
        for name in CANONICAL_CLASSES
        if not any(item["canonical_label"] == name for item in valid_records)
    )
    if empty_classes:
        validation["empty_classes_after_deduplication"] = empty_classes
        validation["corrupt_images"] = corrupt_images
        validation["errors"].append("One or more canonical classes have no valid images")
        _write_json(report_path, validation)
        raise ValueError(f"Empty classes after validation; see {report_path}")

    assignments = _partition_by_class(valid_records, seed=seed)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    split_counts = {split: Counter() for split in SPLITS}
    manifest_rows = []
    try:
        for split in SPLITS:
            for class_name in CANONICAL_CLASSES:
                (staging_dir / split / class_name).mkdir(parents=True, exist_ok=True)

        for index, (split, record) in enumerate(assignments, start=1):
            digest = record["source_sha256"]
            output_name = f"{digest}.jpg"
            relative_path = Path(split) / record["canonical_label"] / output_name
            destination = staging_dir / relative_path
            try:
                _write_image(record["path"], destination)
            except (OSError, ValueError) as exc:
                corrupt_images.append({"path": str(record["path"]), "error": str(exc)})
                continue
            split_counts[split][record["canonical_label"]] += 1
            manifest_rows.append(
                {
                    "split": split,
                    "source_path": record["path"].relative_to(data_dir).as_posix(),
                    "source_label": record["source_label"],
                    "canonical_label": record["canonical_label"],
                    "source_sha256": digest,
                    "output_path": relative_path.as_posix(),
                }
            )
            if index % 5000 == 0 or index == len(assignments):
                print(f"Preprocessed images: {index}/{len(assignments)}", flush=True)

        # Drop class folders with failed conversions only if the class still has
        # valid training support; otherwise fail and preserve a diagnostic.
        empty_train = sorted(
            name for name in CANONICAL_CLASSES if split_counts["train"][name] == 0
        )
        if empty_train:
            raise ValueError(f"No converted training images for classes: {empty_train}")
        assigned_counts = {split: Counter() for split in SPLITS}
        for split, record in assignments:
            assigned_counts[split][record["canonical_label"]] += 1
        lost_assigned_classes = {
            split: sorted(
                name
                for name in CANONICAL_CLASSES
                if assigned_counts[split][name] > 0 and split_counts[split][name] == 0
            )
            for split in SPLITS
        }
        lost_assigned_classes = {
            split: names for split, names in lost_assigned_classes.items() if names
        }
        empty_splits = [split for split in SPLITS if sum(split_counts[split].values()) == 0]
        if lost_assigned_classes or empty_splits:
            validation["conversion_errors"] = corrupt_images
            validation["lost_assigned_classes"] = lost_assigned_classes
            validation["empty_splits"] = empty_splits
            validation["errors"].append(
                "Image conversion failures left assigned classes or splits without images"
            )
            _write_json(report_path, validation)
            raise ValueError(f"Image conversion validation failed; see {report_path}")

        (staging_dir / "class_names.json").write_text(
            json.dumps(list(CANONICAL_CLASSES), indent=2) + "\n", encoding="utf-8"
        )
        _write_json(staging_dir / "taxonomy.json", taxonomy_document())
        with (staging_dir / "split_manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "split",
                    "source_path",
                    "source_label",
                    "canonical_label",
                    "source_sha256",
                    "output_path",
                ],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

        counts_document = {
            split: {
                "total": sum(split_counts[split].values()),
                "classes": {name: split_counts[split][name] for name in CANONICAL_CLASSES},
            }
            for split in SPLITS
        }
        dataset_info = {
            "dataset_name": "Harmonized72LeafDiseaseBenchmark",
            "source_directory": str(data_dir),
            "source_class_names": source_classes,
            "classes": len(CANONICAL_CLASSES),
            "class_names": list(CANONICAL_CLASSES),
            "seed": seed,
            "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "preprocessing": {
                "color_mode": "RGB",
                "size": [224, 224],
                "crop": "center",
                "resample": "Lanczos",
                "output_format": "JPEG",
            },
            "source_image_files": len(records),
            "corrupt_or_unreadable_images": len(corrupt_images),
            "removed_exact_duplicates": len(duplicate_sources),
            "counts": counts_document,
        }
        _write_json(staging_dir / "dataset_info.json", dataset_info)
        _write_json(staging_dir / "corrupt_images.json", {"images": corrupt_images})
        staging_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    # pHash candidates are kept for review. Exact duplicates were already
    # removed before assignment, so this report focuses on near duplicates.
    try:
        print("Auditing exact and near-duplicate candidates...", flush=True)
        audit = audit_dataset(output_dir, near_duplicate_distance=8)
        _write_json(output_dir / "near_duplicate_audit.json", audit)
    except Exception as exc:  # Dataset build remains valid if the optional audit fails.
        _write_json(
            output_dir / "near_duplicate_audit.json",
            {"error": str(exc), "near_duplicate_candidates_across_splits": []},
        )
    return dataset_info


def main():
    parser = argparse.ArgumentParser(
        description="Create the canonical 72-class RGB 224x224 benchmark."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "raw",
        help="Flat class folders or a train/val/test folder tree.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "processed_harmonized",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = prepare_benchmark(args.data_dir, args.output_dir, seed=args.seed)
    print(json.dumps(result["counts"], indent=2))
    print(f"Benchmark created at {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
