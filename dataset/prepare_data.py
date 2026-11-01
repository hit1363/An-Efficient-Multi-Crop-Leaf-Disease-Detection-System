"""
Data Preparation Script - Split raw dataset, resize (center-crop), and save.
"""

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageOps

# Ensure compatibility with older and newer Pillow versions
try:
    RESAMPLE_METHOD = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_METHOD = Image.LANCZOS

DEFAULT_MAX_PER_CLASS = 300


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


def get_image_files(directory: Path):
    """Get list of valid image files from directory using pathlib."""
    valid_extensions = {".jpg", ".jpeg", ".png"}
    if not directory.exists():
        return []

    return sorted(
        f.name
        for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    )


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


def prepare_dataset(
    raw_data_dir: Path,
    output_dir: Path,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=42,
    target_size=224,
    max_per_class=DEFAULT_MAX_PER_CLASS,
):
    """Select up to max_per_class per class, split, and resize/crop images."""
    if not raw_data_dir.exists():
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
    if isinstance(max_per_class, bool) or not isinstance(max_per_class, int) or max_per_class < 1:
        raise ValueError("max_per_class must be a positive integer")

    # Clear old output directory to prevent stale files
    if output_dir.exists():
        print(f"Clearing old output directory: {output_dir}")
        shutil.rmtree(output_dir)

    splits = ["train", "val", "test"]
    for split in splits:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    stats = {
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
        "ratios": (train_ratio, val_ratio, test_ratio),
    }

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

    for disease_folder in disease_folders:
        folder_name = disease_folder.name
        images = get_image_files(disease_folder)

        if not images:
            print(f"Warning: No images found in {folder_name}")
            continue

        stats["original_total"] += len(images)

        for split in splits:
            stats[split]["classes"][folder_name] = 0

        selected = select_images(images, max_per_class, seed, folder_name)
        stats["selected_total"] += len(selected)
        if len(images) > max_per_class:
            stats["capped_classes"] += 1
        class_stats = {"raw": len(images), "selected": len(selected)}
        for split in splits:
            class_stats[split] = 0
        stats["per_class"][folder_name] = class_stats

        train_imgs, val_imgs, test_imgs = split_selected(
            selected, train_ratio, val_ratio
        )

        for split, img_list in [
            ("train", train_imgs),
            ("val", val_imgs),
            ("test", test_imgs),
        ]:
            class_dir = output_dir / split / folder_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for img in img_list:
                src = disease_folder / img
                dst = class_dir / img

                success = process_and_save_image(
                    src_path=src, dst_path=dst, target_size=target_size
                )

                if success:
                    stats[split]["total"] += 1
                    stats[split]["classes"][folder_name] += 1
                    class_stats[split] += 1
                    stats["processed_total"] += 1

        print(
            f"{folder_name:30} | Raw: {len(images):5} | Selected: {len(selected):5} "
            f"| Train: {class_stats['train']:5} | Val: {class_stats['val']:5} "
            f"| Test: {class_stats['test']:5}"
        )

    stats["failed_total"] = stats["selected_total"] - stats["processed_total"]
    return stats


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

    info_path = output_dir / "dataset_info.json"
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
    args = parser.parse_args()

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
    )
    print_statistics(stats)
    save_dataset_info(stats, output_dir, args.seed)
    print("Dataset preparation completed successfully!")


if __name__ == "__main__":
    main()