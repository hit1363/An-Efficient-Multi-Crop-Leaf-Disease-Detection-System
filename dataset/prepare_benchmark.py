"""Build a deterministic, canonical 72-class leaf disease benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageOps

try:
    from .taxonomy import (
        CANONICAL_CLASSES,
        EXPECTED_CLASS_COUNT,
        canonicalize_label,
        taxonomy_document,
    )
except ImportError:
    from taxonomy import (
        CANONICAL_CLASSES,
        EXPECTED_CLASS_COUNT,
        canonicalize_label,
        taxonomy_document,
    )


IMAGE_EXTENSIONS = {
    ".avif", ".bmp", ".gif", ".ico", ".jfif", ".jpe", ".jpeg", ".jpg",
    ".pbm", ".pgm", ".png", ".pnm", ".ppm", ".tga", ".tif", ".tiff", ".webp",
}
SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)
PROGRESS_INTERVAL = 2000
MIN_IMAGES_PER_CLASS = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_class_dirs(data_dir: Path):
    """Yield source class directories from a flat or complete split tree."""
    split_dirs = [data_dir / split for split in SPLITS]
    if all(path.is_dir() for path in split_dirs):
        for split_dir in split_dirs:
            yield from (
                (class_dir.name, class_dir)
                for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir())
            )
        return
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        if class_dir.name in SPLITS:
            raise ValueError(
                f"Input has an incomplete split tree at {data_dir}; "
                "expected train, val, and test"
            )
        yield class_dir.name, class_dir


def _split_counts(total: int, ratios=DEFAULT_RATIOS):
    """Return deterministic split counts whose sum is ``total``."""
    if total < len(SPLITS):
        raise ValueError("Each class needs at least one image per split")
    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    for index in sorted(range(len(counts)), key=lambda item: raw[item] - counts[item], reverse=True):
        if sum(counts) >= total:
            break
        counts[index] += 1
    for index in range(len(counts)):
        if counts[index] == 0:
            donor = max(range(len(counts)), key=lambda item: counts[item])
            counts[donor] -= 1
            counts[index] = 1
    if sum(counts) != total or any(count < 1 for count in counts):
        raise ValueError(f"Unable to create non-empty splits for {total} images")
    return dict(zip(SPLITS, counts))


def _image_files(directory: Path):
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _write_validation_report(output_dir: Path, report: dict):
    report_path = output_dir.parent / f"{output_dir.name}_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _prepare_image(source: Path, destination: Path, size=(224, 224)):
    with Image.open(source) as image:
        converted = ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)
        converted.save(destination, format="JPEG", quality=95)


def prepare_benchmark(
    data_dir,
    output_dir,
    seed=42,
    max_per_class=300,
    ratios=DEFAULT_RATIOS,
):
    """Validate, deduplicate, and write a reproducible train/val/test tree.

    Classes left with fewer than :data:`MIN_IMAGES_PER_CLASS` unique images
    after exact de-duplication (a handful of PlantVillage classes are two or
    three near-identical photos duplicated hundreds of times) are not fatal:
    every unique image stays in train and one representative copy is placed
    in val and test so all 72 classes remain scorable. Those classes are
    disclosed in ``dataset_info.json`` under ``degenerate_classes`` and the
    split validator treats their registered copies as intentional reuse.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Input dataset directory does not exist: {data_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must contain three values summing to one")
    if isinstance(max_per_class, bool) or not isinstance(max_per_class, int) or max_per_class < 3:
        raise ValueError("max_per_class must be an integer of at least 3")

    report = {
        "unknown_source_labels": [],
        "cross_label_duplicate_conflicts": [],
        "removed_exact_duplicates": 0,
        "degenerate_classes": {},
    }
    by_class = {name: [] for name in CANONICAL_CLASSES}
    digest_owner = {}
    scanned = 0
    try:
        for source_label, class_dir in _source_class_dirs(data_dir):
            try:
                canonical_label = canonicalize_label(source_label)
            except ValueError:
                report["unknown_source_labels"].append(source_label)
                continue
            for path in _image_files(class_dir):
                digest = _sha256(path)
                scanned += 1
                if scanned % PROGRESS_INTERVAL == 0:
                    print(
                        f"[prepare_benchmark] hashed {scanned} source images",
                        flush=True,
                    )
                previous = digest_owner.get(digest)
                if previous is not None:
                    if previous != canonical_label:
                        report["cross_label_duplicate_conflicts"].append(
                            {
                                "sha256": digest,
                                "first_label": previous,
                                "conflicting_label": canonical_label,
                                "path": str(path),
                            }
                        )
                    else:
                        report["removed_exact_duplicates"] += 1
                    continue
                digest_owner[digest] = canonical_label
                by_class[canonical_label].append((path, digest))

        if report["unknown_source_labels"] or report["cross_label_duplicate_conflicts"]:
            report["error"] = "taxonomy validation failed"
            _write_validation_report(output_dir, report)
            if report["cross_label_duplicate_conflicts"]:
                raise ValueError(
                    "Cross-label duplicate conflicts detected; see validation report"
                )
            raise ValueError("taxonomy validation failed; see validation report")

        empty = [name for name, items in by_class.items() if not items]
        if empty:
            report["missing_classes"] = empty
            report["error"] = "classes need at least one image"
            _write_validation_report(output_dir, report)
            raise ValueError(f"Classes have no source images: {empty}")

        degenerate = {
            name: len(items)
            for name, items in sorted(by_class.items())
            if 0 < len(items) < MIN_IMAGES_PER_CLASS
        }
        report["degenerate_classes"] = degenerate
        for name, unique_count in degenerate.items():
            print(
                f"[prepare_benchmark] degenerate class {name}: "
                f"{unique_count} unique image(s); keeping all in train and "
                "placing one disclosed copy in val/test",
                flush=True,
            )

        output_dir.mkdir(parents=True)
        for split in SPLITS:
            (output_dir / split).mkdir()

        manifest_rows = []
        counts = {split: {"total": 0, "classes": {}} for split in SPLITS}
        rng = random.Random(seed)
        written = 0
        for class_name in CANONICAL_CLASSES:
            selected = list(by_class[class_name])
            rng.shuffle(selected)
            selected = selected[:max_per_class]
            if len(selected) < MIN_IMAGES_PER_CLASS:
                split_selection = {
                    "train": selected,
                    "val": selected[:1],
                    "test": selected[-1:],
                }
            else:
                split_counts = _split_counts(len(selected), ratios)
                split_selection = {}
                offset = 0
                for split in SPLITS:
                    split_selection[split] = selected[
                        offset : offset + split_counts[split]
                    ]
                    offset += split_counts[split]
            for split in SPLITS:
                class_output = output_dir / split / class_name
                class_output.mkdir(parents=True)
                items = split_selection[split]
                counts[split]["classes"][class_name] = len(items)
                counts[split]["total"] += len(items)
                for index, (source, digest) in enumerate(items):
                    destination = class_output / f"{index:05d}_{source.stem}.jpg"
                    _prepare_image(source, destination)
                    manifest_rows.append(
                        {
                            "split": split,
                            "canonical_label": class_name,
                            "output_path": str(destination.relative_to(output_dir)),
                            "source_sha256": digest,
                        }
                    )
                    written += 1
                    if written % PROGRESS_INTERVAL == 0:
                        print(
                            f"[prepare_benchmark] wrote {written} images",
                            flush=True,
                        )

        with (output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["split", "canonical_label", "output_path", "source_sha256"],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)
        (output_dir / "class_names.json").write_text(
            json.dumps(list(CANONICAL_CLASSES), indent=2), encoding="utf-8"
        )
        (output_dir / "taxonomy.json").write_text(
            json.dumps(taxonomy_document(), indent=2, sort_keys=True), encoding="utf-8"
        )
        (output_dir / "dataset_info.json").write_text(
            json.dumps(
                {
                    "dataset_name": "LeafDisease",
                    "classes": EXPECTED_CLASS_COUNT,
                    "class_names": list(CANONICAL_CLASSES),
                    "seed": seed,
                    "max_per_class": max_per_class,
                    "counts": counts,
                    "removed_exact_duplicates": report["removed_exact_duplicates"],
                    "degenerate_classes": degenerate,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (output_dir / "near_duplicate_audit.json").write_text(
            json.dumps({"status": "not_run", "reason": "exact duplicate audit only"}, indent=2),
            encoding="utf-8",
        )
        return {
            "classes": EXPECTED_CLASS_COUNT,
            "seed": seed,
            "counts": counts,
            "removed_exact_duplicates": report["removed_exact_duplicates"],
            "degenerate_classes": degenerate,
        }
    except Exception:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=300)
    args = parser.parse_args()
    result = prepare_benchmark(
        args.data_dir,
        args.output_dir,
        seed=args.seed,
        max_per_class=args.max_per_class,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
