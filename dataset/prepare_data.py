"""
Data Preparation Script - Split raw dataset, resize (center-crop), and save.

Writes the benchmark metadata files (class_names.json, split_manifest.csv,
taxonomy.json, near_duplicate_audit.json, dataset_info.json) alongside the
split tree so downstream validators and comparison runners work unchanged.

Two modes share one split engine so there is a single builder in the project:

* default (tolerant): every raw class folder is converted, an unknown folder
  spelling falls back to its own name instead of aborting, and per-class
  statistics are printed for thesis reporting.
* ``strict=True`` (CLI ``--strict``): the benchmark gates are enforced - every
  canonical class must be present, unknown source labels and cross-label
  byte-identical duplicates are rejected (with a validation report), an
  existing output directory is never overwritten, thin classes get a disclosed
  non-empty fallback split, and a failed build removes its partial output tree.
  Byte-identical copies inside the same class are kept, so a class is
  represented by its real file count instead of collapsing to its handful of
  unique photos; such reuse is disclosed under ``degenerate_classes``.
"""

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

# Ensure compatibility with older and newer Pillow versions
try:
    RESAMPLE_METHOD = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_METHOD = Image.LANCZOS

DEFAULT_MAX_PER_CLASS = 300
MANIFEST_FIELDS = ["split", "canonical_label", "output_path", "source_sha256"]
DEFAULT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
# Strict mode scans every extension Pillow can open, not just the common three.
IMAGE_EXTENSIONS = {
    ".avif", ".bmp", ".gif", ".ico", ".jfif", ".jpe", ".jpeg", ".jpg",
    ".pbm", ".pgm", ".png", ".pnm", ".ppm", ".tga", ".tif", ".tiff", ".webp",
}
SPLITS = ("train", "val", "test")
MIN_IMAGES_PER_CLASS = 3
PROGRESS_INTERVAL = 2000


def select_images(images, max_per_class, seed, class_name):
    """Select a reproducible, class-specific random subset of images."""
    selected = list(images)
    class_seed = hashlib.sha256(f"{seed}:{class_name}".encode("utf-8")).digest()
    random.Random(class_seed).shuffle(selected)
    return selected[:max_per_class]


def split_selected(items, train_ratio, val_ratio):
    """Split an already shuffled selection into train, validation, and test."""
    train_end = int(len(items) * train_ratio)
    val_end = train_end + int(len(items) * val_ratio)
    return items[:train_end], items[train_end:val_end], items[val_end:]


def _split_counts(total: int, ratios=(0.70, 0.15, 0.15)):
    """Return deterministic split counts whose sum is ``total``.

    Used by strict mode when the plain ratio cut would leave a split empty;
    largest-remainder rounding plus a donor pass keeps every split non-empty.
    """
    if total < len(SPLITS):
        raise ValueError("Each class needs at least one image per split")
    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    for index in sorted(
        range(len(counts)),
        key=lambda item: raw[item] - counts[item],
        reverse=True,
    ):
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


def get_image_files(directory: Path, extensions=None):
    """Get list of valid image file names from directory using pathlib."""
    valid_extensions = DEFAULT_IMAGE_EXTENSIONS if extensions is None else extensions
    if not directory.exists():
        return []

    return sorted(
        f.name
        for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    )


def _source_class_dirs(data_dir: Path):
    """Yield ``(source_label, class_dir)`` from a flat or complete split tree."""
    split_dirs = [data_dir / split for split in SPLITS]
    if all(path.is_dir() for path in split_dirs):
        for split_dir in split_dirs:
            yield from (
                (class_dir.name, class_dir)
                for class_dir in sorted(
                    path for path in split_dir.iterdir() if path.is_dir()
                )
            )
        return
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        if class_dir.name in SPLITS:
            raise ValueError(
                f"Input has an incomplete split tree at {data_dir}; "
                "expected train, val, and test"
            )
        yield class_dir.name, class_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_validation_report(output_dir: Path, report: dict):
    """Write the strict-mode validation report next to the output tree."""
    report_path = output_dir.parent / f"{output_dir.name}_validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report_path


def _write_manifest_files(output_dir, manifest_rows, class_names):
    """Write the benchmark manifest files next to the split tree.

    Shapes match what training/utils.py (validate_dataset_splits,
    load_benchmark_class_names) and training/compare_models.py consume.
    """
    output_dir = Path(output_dir)
    with (output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    (output_dir / "class_names.json").write_text(
        json.dumps(list(class_names), indent=2), encoding="utf-8"
    )
    (output_dir / "taxonomy.json").write_text(
        json.dumps(taxonomy_document(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "near_duplicate_audit.json").write_text(
        json.dumps(
            {"status": "not_run", "reason": "exact duplicate audit only"},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Manifest files written to: {output_dir}")


def process_and_save_image(src_path: Path, dst_path: Path, target_size=224):
    """
    Reads an image, converts to RGB, resizes/center-crops with explicit centering,
    and saves with high quality. Returns 1 if successful, 0 if it fails.
    """
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")

            # Explicit center cropping (0.5, 0.5)
            img = ImageOps.fit(
                img,
                (target_size, target_size),
                method=RESAMPLE_METHOD,
                centering=(0.5, 0.5),
            )

            # Save JPEGs with high quality to avoid compression artifacts
            if dst_path.suffix.lower() in (".jpg", ".jpeg"):
                img.save(dst_path, quality=95, optimize=True)
            else:
                img.save(dst_path)

        return 1
    except Exception as e:  # noqa: BLE001
        print(f"Error processing {src_path}: {e}")
        return 0


def _empty_stats(train_ratio, val_ratio, test_ratio, max_per_class, target_size):
    """Return the shared statistics structure both modes fill in."""
    return {
        "train": {"total": 0, "classes": {}},
        "val": {"total": 0, "classes": {}},
        "test": {"total": 0, "classes": {}},
        "original_total": 0,
        "selected_total": 0,
        "processed_total": 0,
        "failed_total": 0,
        "capped_classes": 0,
        "max_per_class": max_per_class,
        "per_class": {},
        "classes_count": 0,
        "target_size": target_size,
        "degenerate_classes": {},
        "ratios": (train_ratio, val_ratio, test_ratio),
    }


def _save_split_images(
    output_dir,
    split,
    class_dir_name,
    items,
    target_size,
    stats,
    class_stats,
    manifest_rows,
    canonical_label,
    indexed=False,
):
    """Resize/crop ``items`` into ``output_dir/split/class_dir_name``.

    ``items`` is a sequence of ``(source_path, source_sha256)`` pairs. Only
    successfully written images are counted and added to the manifest.
    ``indexed=True`` prefixes a per-split index to the file name, which strict
    mode needs because a reused source image can land in several splits.
    """
    class_dir = output_dir / split / class_dir_name
    class_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, (source, digest) in enumerate(items):
        if indexed:
            destination = class_dir / f"{index:05d}_{source.stem}.jpg"
        else:
            destination = class_dir / source.name
        if not process_and_save_image(
            src_path=source, dst_path=destination, target_size=target_size
        ):
            continue
        written += 1
        stats[split]["total"] += 1
        stats[split]["classes"][class_dir_name] += 1
        class_stats[split] += 1
        stats["processed_total"] += 1
        manifest_rows.append(
            {
                "split": split,
                "canonical_label": canonical_label,
                "output_path": str(destination.relative_to(output_dir)),
                "source_sha256": digest,
            }
        )
    return written


def prepare_dataset(
    raw_data_dir: Path,
    output_dir: Path,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=42,
    target_size=224,
    max_per_class=DEFAULT_MAX_PER_CLASS,
    write_manifest=True,
    strict=False,
):
    """Select up to max_per_class per class, split, and resize/crop images.

    With ``write_manifest=True`` (default) the four benchmark manifest files
    (class_names.json, split_manifest.csv, taxonomy.json,
    near_duplicate_audit.json) are written alongside the split tree.

    ``strict=True`` additionally enforces the benchmark gates: the output
    directory must not exist yet, every canonical class must have at least one
    image, unknown source labels and cross-label byte-identical duplicates
    abort the build (leaving a ``*_validation_report.json`` next to the output
    directory), thin classes receive a disclosed non-empty split, and a failure
    removes the partially written tree. ``dataset_info.json`` is written by
    strict mode itself so the degenerate-class registry travels with the tree.
    """
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)

    if not raw_data_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")

    test_ratio = 1.0 - train_ratio - val_ratio

    # Strict split ratio and parameter validation
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be strictly between 0 and 1")
    if not (0 <= val_ratio < 1):
        raise ValueError("val_ratio must be between 0 and 1")
    if not (0 <= test_ratio < 1):
        raise ValueError("Calculated test_ratio must be between 0 and 1")
    if target_size <= 0:
        raise ValueError("target_size must be greater than 0")
    if (
        isinstance(max_per_class, bool)
        or not isinstance(max_per_class, int)
        or max_per_class < 1
    ):
        raise ValueError("max_per_class must be a positive integer")

    if strict:
        if max_per_class < MIN_IMAGES_PER_CLASS:
            raise ValueError(
                f"max_per_class must be an integer of at least {MIN_IMAGES_PER_CLASS}"
            )
        return _prepare_dataset_strict(
            raw_data_dir,
            output_dir,
            (train_ratio, val_ratio, test_ratio),
            seed,
            target_size,
            max_per_class,
            write_manifest=write_manifest,
        )

    # Clear old output directory to prevent stale files
    if output_dir.exists():
        print(f"Clearing old output directory: {output_dir}")
        shutil.rmtree(output_dir)

    for split in SPLITS:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    stats = _empty_stats(
        train_ratio, val_ratio, test_ratio, max_per_class, target_size
    )

    disease_folders = sorted(
        [f for f in raw_data_dir.iterdir() if f.is_dir()]
    )

    if not disease_folders:
        print(f"Warning: No disease folders found in {raw_data_dir}")
        return stats

    stats["classes_count"] = len(disease_folders)

    print(f"Found {len(disease_folders)} disease classes.")
    print(f"Maximum images per class: {max_per_class}")
    print(f"Target size: {target_size}x{target_size} pixels.\n")

    manifest_rows = []

    for disease_folder in disease_folders:
        folder_name = disease_folder.name
        try:
            canonical_label = canonicalize_label(folder_name)
        except ValueError:
            canonical_label = folder_name
        images = get_image_files(disease_folder)

        if not images:
            print(f"Warning: No images found in {folder_name}")
            continue

        stats["original_total"] += len(images)

        for split in SPLITS:
            stats[split]["classes"][folder_name] = 0

        selected = select_images(images, max_per_class, seed, folder_name)
        stats["selected_total"] += len(selected)
        if len(images) > max_per_class:
            stats["capped_classes"] += 1
        class_stats = {"raw": len(images), "selected": len(selected)}
        for split in SPLITS:
            class_stats[split] = 0
        stats["per_class"][folder_name] = class_stats

        train_imgs, val_imgs, test_imgs = split_selected(
            selected, train_ratio, val_ratio
        )

        # Source digests let the manifest disclose classes whose identical
        # copies end up on both sides of a split; training's split validator
        # tolerates reuse only for labels registered as degenerate.
        digests = {img: _sha256(disease_folder / img) for img in selected}
        digest_splits = {}
        for split, img_list in (
            ("train", train_imgs),
            ("val", val_imgs),
            ("test", test_imgs),
        ):
            for img in img_list:
                digest_splits.setdefault(digests[img], set()).add(split)
        if any(len(splits) > 1 for splits in digest_splits.values()):
            unique_count = len(set(digests.values()))
            stats["degenerate_classes"][canonical_label] = unique_count
            print(
                f"[prepare_data] duplicate-heavy class {canonical_label}: "
                f"{unique_count} unique image(s) kept as {len(selected)} files; "
                "identical copies cross splits and are disclosed in dataset_info.json"
            )

        for split, img_list in [
            ("train", train_imgs),
            ("val", val_imgs),
            ("test", test_imgs),
        ]:
            _save_split_images(
                output_dir,
                split,
                folder_name,
                [(disease_folder / img, digests[img]) for img in img_list],
                target_size,
                stats,
                class_stats,
                manifest_rows,
                canonical_label,
            )

        print(
            f"{folder_name:30} | Raw: {len(images):5} | Selected: {len(selected):5} "
            f"| Train: {class_stats['train']:5} | Val: {class_stats['val']:5} "
            f"| Test: {class_stats['test']:5}"
        )

    stats["failed_total"] = stats["selected_total"] - stats["processed_total"]

    if write_manifest:
        # Canonical order (Invalid last, per dataset.taxonomy.CANONICAL_CLASSES)
        # so the manifest matches the order training/compare_models consume.
        class_names = list(CANONICAL_CLASSES)
        _write_manifest_files(output_dir, manifest_rows, class_names)

    return stats


def _prepare_dataset_strict(
    raw_data_dir,
    output_dir,
    ratios,
    seed,
    target_size,
    max_per_class,
    write_manifest=True,
):
    """Strict benchmark build: validate a flat or split source tree, then write.

    Every source image is hashed for taxonomy-conflict detection, but
    byte-identical copies inside one class are kept (the tolerant behaviour) so
    a class is represented by its real file count. Capping, splitting, and
    resizing reuse :func:`select_images`, :func:`split_selected`, and
    :func:`process_and_save_image`; a non-empty fallback split preserves the
    guarantee that every class is scorable in all three splits.
    """
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if not write_manifest:
        raise ValueError("strict builds require write_manifest=True")

    report = {
        "unknown_source_labels": [],
        "cross_label_duplicate_conflicts": [],
        "exact_duplicates_kept": 0,
        "degenerate_classes": {},
    }
    stats = _empty_stats(ratios[0], ratios[1], ratios[2], max_per_class, target_size)
    by_class = {name: [] for name in CANONICAL_CLASSES}
    digest_owner = {}
    scanned = 0

    try:
        for source_label, class_dir in _source_class_dirs(raw_data_dir):
            try:
                canonical_label = canonicalize_label(source_label)
            except ValueError:
                report["unknown_source_labels"].append(source_label)
                continue
            for name in get_image_files(class_dir, extensions=IMAGE_EXTENSIONS):
                path = class_dir / name
                digest = _sha256(path)
                scanned += 1
                if scanned % PROGRESS_INTERVAL == 0:
                    print(
                        f"[prepare_data] hashed {scanned} source images",
                        flush=True,
                    )
                previous = digest_owner.get(digest)
                if previous is None:
                    digest_owner[digest] = canonical_label
                elif previous != canonical_label:
                    report["cross_label_duplicate_conflicts"].append(
                        {
                            "sha256": digest,
                            "first_label": previous,
                            "conflicting_label": canonical_label,
                            "path": str(path),
                        }
                    )
                else:
                    report["exact_duplicates_kept"] += 1
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

        output_dir.mkdir(parents=True)
        for split in SPLITS:
            (output_dir / split).mkdir()

        stats["classes_count"] = len(CANONICAL_CLASSES)
        stats["original_total"] = scanned

        print(f"Found {len(CANONICAL_CLASSES)} canonical classes (strict build).")
        print(f"Maximum images per class: {max_per_class}")
        print(f"Target size: {target_size}x{target_size} pixels.\n")

        manifest_rows = []
        degenerate = {}
        written = 0

        for class_name in CANONICAL_CLASSES:
            items = by_class[class_name]
            unique_count = len({digest for _, digest in items})
            selected = select_images(items, max_per_class, seed, class_name)
            stats["selected_total"] += len(selected)
            if len(items) > max_per_class:
                stats["capped_classes"] += 1
            class_stats = {"raw": len(items), "selected": len(selected)}
            for split in SPLITS:
                class_stats[split] = 0
                stats[split]["classes"][class_name] = 0
            stats["per_class"][class_name] = class_stats

            if len(selected) < MIN_IMAGES_PER_CLASS:
                split_selection = {
                    "train": selected,
                    "val": selected[:1],
                    "test": selected[-1:],
                }
                degenerate[class_name] = unique_count
                print(
                    f"[prepare_data] degenerate class {class_name}: "
                    f"{unique_count} unique image(s); keeping all in train and "
                    "placing one disclosed copy in val/test",
                    flush=True,
                )
            else:
                train_items, val_items, test_items = split_selected(
                    selected, ratios[0], ratios[1]
                )
                if train_items and val_items and test_items:
                    split_selection = {
                        "train": train_items,
                        "val": val_items,
                        "test": test_items,
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

                digest_splits = {}
                for split, part in split_selection.items():
                    for _, digest in part:
                        digest_splits.setdefault(digest, set()).add(split)
                if any(len(splits) > 1 for splits in digest_splits.values()):
                    degenerate[class_name] = unique_count
                    print(
                        f"[prepare_data] duplicate-heavy class {class_name}: "
                        f"{unique_count} unique image(s) kept as {len(selected)} "
                        "files; identical copies cross splits and are disclosed "
                        "in dataset_info.json",
                        flush=True,
                    )

            written_before = written
            for split in SPLITS:
                written += _save_split_images(
                    output_dir,
                    split,
                    class_name,
                    split_selection[split],
                    target_size,
                    stats,
                    class_stats,
                    manifest_rows,
                    class_name,
                    indexed=True,
                )
            if written // PROGRESS_INTERVAL > written_before // PROGRESS_INTERVAL:
                print(f"[prepare_data] wrote {written} images", flush=True)

        stats["failed_total"] = stats["selected_total"] - stats["processed_total"]
        stats["degenerate_classes"] = degenerate
        stats["exact_duplicates_kept"] = report["exact_duplicates_kept"]
        stats["unique_source_images"] = len(digest_owner)
        report["degenerate_classes"] = degenerate

        if write_manifest:
            _write_manifest_files(
                output_dir, manifest_rows, list(CANONICAL_CLASSES)
            )
            save_dataset_info(stats, output_dir, seed)
        else:
            raise ValueError("strict builds require write_manifest=True")

        return stats
    except Exception:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        raise


def print_statistics(stats):
    """Print detailed split statistics suitable for thesis reporting."""
    tr, vr, ts = stats["ratios"]

    print("\n" + "=" * 80)
    print("DATASET SELECTION, SPLIT & RESIZE STATISTICS")
    print("=" * 80)
    print(f"  Number of Classes : {stats['classes_count']}")
    print(f"  Maximum per Class : {stats['max_per_class']}")
    print(f"  Classes Capped    : {stats['capped_classes']}")
    print(
        f"  Target Size       : {stats['target_size']}x{stats['target_size']} (Center-Cropped)"
    )
    print(
        f"  Split Ratio       : {tr*100:.0f}% Train / {vr*100:.0f}% Validation / {ts*100:.0f}% Test"
    )
    print("-" * 80)
    print(f"  Raw Images        : {stats['original_total']:,}")
    print(f"  Selected Images   : {stats['selected_total']:,}")
    print(f"  Processed Images  : {stats['processed_total']:,}")
    if stats["failed_total"]:
        print(f"  Failed Images     : {stats['failed_total']:,}")
    print("-" * 80)
    print(f"  Training Set      : {stats['train']['total']:,} images")
    print(f"  Validation Set    : {stats['val']['total']:,} images")
    print(f"  Test Set          : {stats['test']['total']:,} images")
    print("=" * 80 + "\n")


def save_dataset_info(stats, output_dir, seed):
    """Save dataset metadata and parameters to a JSON file."""
    tr, vr, ts = stats["ratios"]

    info = {
        "dataset_name": "LeafDisease",
        "classes": stats["classes_count"],
        "class_names": sorted(stats["train"]["classes"].keys()),
        "degenerate_classes": stats.get("degenerate_classes", {}),
        "selection": {
            "max_per_class": stats["max_per_class"],
            "method": "deterministic class-specific shuffle, keep first max_per_class images",
            "classes_capped": stats["capped_classes"],
        },
        "original_images": stats["original_total"],
        "selected_images": stats["selected_total"],
        "processed_images": stats["processed_total"],
        "failed_images": stats["failed_total"],
        "target_size": stats["target_size"],
        "color_mode": "RGB",
        "split": {
            "train": tr,
            "validation": vr,
            "test": ts,
        },
        "seed": seed,
        "per_class": stats["per_class"],
        "preprocessing": {
            "resize_method": "ImageOps.fit",
            "resample": "LANCZOS",
            "centering": [0.5, 0.5],
        },
    }
    if "exact_duplicates_kept" in stats:
        info["exact_duplicates_kept"] = stats["exact_duplicates_kept"]
        info["unique_source_images"] = stats["unique_source_images"]
        info["expected_classes"] = EXPECTED_CLASS_COUNT

    info_path = Path(output_dir) / "dataset_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
    print(f"Dataset metadata saved to: {info_path}")


def positive_int(value):
    """Parse a positive integer for command-line options."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be an integer, got {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {parsed}")
    return parsed


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset: Split, Resize (Center-Crop), and Save."
    )
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-per-class",
        type=positive_int,
        default=DEFAULT_MAX_PER_CLASS,
        help="Maximum images kept per class (default: 300)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=224,
        help="Target image size (e.g., 224 for 224x224)",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip writing class_names.json, split_manifest.csv, taxonomy.json, "
        "and near_duplicate_audit.json next to the split tree",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enforce benchmark gates: reject unknown labels and cross-label "
        "duplicates, require every canonical class, refuse to overwrite the "
        "output directory, and disclose degenerate classes",
    )
    args = parser.parse_args()
    if args.strict and args.no_manifest:
        parser.error("--strict requires the benchmark manifest files; drop --no-manifest")

    script_dir = Path(__file__).resolve().parent
    raw_dir = Path(args.raw_dir).resolve() if args.raw_dir else script_dir / "raw"
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else script_dir / "processed"
    )

    print(f"Raw data directory: {raw_dir}")
    print(f"Output directory: {output_dir}")

    stats = prepare_dataset(
        raw_dir,
        output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        target_size=args.target_size,
        max_per_class=args.max_per_class,
        write_manifest=not args.no_manifest,
        strict=args.strict,
    )
    print_statistics(stats)
    if not args.strict:
        # Strict builds write dataset_info.json themselves, together with a
        # degenerate-class registry that the split validator reads.
        save_dataset_info(stats, output_dir, args.seed)
    print("Dataset preparation completed successfully!")


if __name__ == "__main__":
    main()
