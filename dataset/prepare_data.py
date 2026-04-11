"""
Data Preparation Script - Split raw dataset into train/val/test sets.
"""

import os
import shutil
import argparse
import random


def simple_train_test_split(data, test_size, random_state=None):
    """Simple deterministic train/test split."""
    data_copy = list(data)
    if random_state is not None:
        rng = random.Random(random_state)
        rng.shuffle(data_copy)
    else:
        random.shuffle(data_copy)
    split_idx = int(len(data_copy) * (1 - test_size))
    return data_copy[:split_idx], data_copy[split_idx:]


def get_image_files(directory):
    """Get list of valid image files from directory."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    if not os.path.exists(directory):
        return []

    images = []
    for filename in os.listdir(directory):
        if any(filename.endswith(ext) for ext in valid_extensions):
            images.append(filename)
    return images


def prepare_dataset(
    raw_data_dir, output_dir, train_ratio=0.70, val_ratio=0.15, seed=42
):
    """Split dataset into train (70%), val (15%), test (15%)."""
    if not os.path.exists(raw_data_dir):
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError("train_ratio + val_ratio exceeds 1.0")
    if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Ratios must be non-negative, and train_ratio must be > 0")

    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    stats = {
        "train": {"total": 0, "classes": {}},
        "val": {"total": 0, "classes": {}},
        "test": {"total": 0, "classes": {}},
    }

    disease_folders = sorted(
        [
            f
            for f in os.listdir(raw_data_dir)
            if os.path.isdir(os.path.join(raw_data_dir, f))
        ]
    )

    if not disease_folders:
        print(f"Warning: No disease folders found in {raw_data_dir}")
        return stats

    print(f"Found {len(disease_folders)} disease classes")
    print(
        f"Split ratio - Train: {train_ratio * 100:.0f}%, Val: {val_ratio * 100:.0f}%, Test: {test_ratio * 100:.0f}%\n"
    )

    for disease_folder in disease_folders:
        disease_path = os.path.join(raw_data_dir, disease_folder)
        images = get_image_files(disease_path)

        if not images:
            print(f"Warning: No images found in {disease_folder}")
            continue

        train_imgs, temp_imgs = simple_train_test_split(
            images, test_size=(1 - train_ratio), random_state=seed
        )

        # Split remaining data into val/test while preserving requested global ratios.
        remaining_ratio = 1 - train_ratio
        test_size_in_remaining = (
            test_ratio / remaining_ratio if remaining_ratio > 0 else 0
        )
        val_imgs, test_imgs = simple_train_test_split(
            temp_imgs, test_size=test_size_in_remaining, random_state=seed
        )

        for split, img_list in [
            ("train", train_imgs),
            ("val", val_imgs),
            ("test", test_imgs),
        ]:
            class_dir = os.path.join(output_dir, split, disease_folder)
            os.makedirs(class_dir, exist_ok=True)

            for img in img_list:
                src = os.path.join(disease_path, img)
                dst = os.path.join(class_dir, img)
                shutil.copy2(src, dst)

            stats[split]["total"] += len(img_list)
            stats[split]["classes"][disease_folder] = len(img_list)

        print(
            f"{disease_folder:45} | Train: {len(train_imgs):6} | Val: {len(val_imgs):6} | Test: {len(test_imgs):6}"
        )

    return stats


def print_statistics(stats):
    """Print split statistics."""
    total = sum(s["total"] for s in stats.values())
    print("\n" + "=" * 80)
    print("DATASET SPLIT STATISTICS")
    print("=" * 80)
    print("\nOverall Split:")
    print(
        f"  Training:   {stats['train']['total']:,} images ({(stats['train']['total'] / total * 100) if total else 0:.1f}%)"
    )
    print(
        f"  Validation: {stats['val']['total']:,} images ({(stats['val']['total'] / total * 100) if total else 0:.1f}%)"
    )
    print(
        f"  Test:       {stats['test']['total']:,} images ({(stats['test']['total'] / total * 100) if total else 0:.1f}%)"
    )
    print(f"  Total:      {total:,} images")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset by splitting into train/val/test sets"
    )
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Use dataset-local defaults so the script works from any current directory.
    raw_dir = (
        os.path.abspath(args.raw_dir)
        if args.raw_dir is not None
        else os.path.join(script_dir, "raw")
    )
    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir is not None
        else os.path.join(script_dir, "processed")
    )

    print(f"Raw data directory: {raw_dir}")
    print(f"Output directory: {output_dir}")

    stats = prepare_dataset(
        raw_dir,
        output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print_statistics(stats)
    print("Dataset preparation completed successfully!")


if __name__ == "__main__":
    main()
