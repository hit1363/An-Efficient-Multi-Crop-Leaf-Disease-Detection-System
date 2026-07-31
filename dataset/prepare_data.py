"""
Data Preparation Script - Split raw dataset, resize (center-crop), and save.
"""

import argparse
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


def split_data_once(data, train_ratio, val_ratio, seed=None):
    """
    Shuffles data once and splits into train, val, and test lists.
    """
    data_copy = list(data)
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(data_copy)
    else:
        random.shuffle(data_copy)

    n_total = len(data_copy)
    train_end = int(n_total * train_ratio)
    val_end = train_end + int(n_total * val_ratio)

    train_imgs = data_copy[:train_end]
    val_imgs = data_copy[train_end:val_end]
    test_imgs = data_copy[val_end:]

    return train_imgs, val_imgs, test_imgs


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
):
    """Split dataset into train/val/test and resize/crop images."""
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

        # Shuffle once and unpack into train, val, test
        train_imgs, val_imgs, test_imgs = split_data_once(
            images, train_ratio, val_ratio, seed=seed
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

        print(
            f"{folder_name:30} | Train: {stats['train']['classes'][folder_name]:5} "
            f"| Val: {stats['val']['classes'][folder_name]:5} "
            f"| Test: {stats['test']['classes'][folder_name]:5}"
        )

    return stats


def print_statistics(stats):
    """Print detailed split statistics suitable for thesis reporting."""
    total_processed = sum(stats[s]["total"] for s in ["train", "val", "test"])
    tr, vr, ts = stats["ratios"]

    print("\n" + "=" * 80)
    print("DATASET SPLIT & RESIZE STATISTICS")
    print("=" * 80)
    print(f"  Number of Classes : {stats['classes_count']}")
    print(
        f"  Target Size       : {stats['target_size']}x{stats['target_size']} (Center-Cropped)"
    )
    print(
        f"  Split Ratio       : {tr*100:.0f}% Train / {vr*100:.0f}% Validation / {ts*100:.0f}% Test"
    )
    print("-" * 80)
    print(f"  Original Images   : {stats['original_total']:,}")
    print(f"  Processed Images  : {total_processed:,}")
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
        "original_images": stats["original_total"],
        "processed_images": (
            stats["train"]["total"]
            + stats["val"]["total"]
            + stats["test"]["total"]
        ),
        "target_size": stats["target_size"],
        "color_mode": "RGB",
        "split": {
            "train": tr,
            "validation": vr,
            "test": ts,
        },
        "seed": seed,
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
    )
    print_statistics(stats)
    save_dataset_info(stats, output_dir, args.seed)
    print("Dataset preparation completed successfully!")


if __name__ == "__main__":
    main()