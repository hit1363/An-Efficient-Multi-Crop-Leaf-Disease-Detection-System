"""Create a new split with exact duplicates kept together and deduplicated."""

import argparse
import hashlib
import json
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLIT_NAMES = ("train", "val", "test")


def _image_records(data_dir):
    for split in SPLIT_NAMES:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    yield class_dir.name, path


def _group_exact_images(data_dir):
    groups = {}
    original_count = 0
    for class_name, path in _image_records(data_dir):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        group = groups.setdefault(digest, {})
        # Keep one representative per label for identical image bytes.
        group.setdefault(class_name, path)
        original_count += 1
    return groups, original_count


def _assign_groups(groups, ratios, seed):
    class_totals = Counter(
        class_name
        for group in groups.values()
        for class_name in group
    )
    targets = {
        split: {class_name: class_totals[class_name] * ratio for class_name in class_totals}
        for split, ratio in zip(SPLIT_NAMES, ratios)
    }
    assigned = {split: Counter() for split in SPLIT_NAMES}
    remaining_groups = Counter(
        class_name
        for group in groups.values()
        for class_name in group
    )
    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)
    group_items.sort(
        key=lambda item: max(
            1.0 / max(class_totals[class_name], 1)
            for class_name in item[1]
        ),
        reverse=True,
    )

    assignments = []
    for digest, classes in group_items:
        allowed_splits = set(SPLIT_NAMES)
        for class_name in classes:
            if class_totals[class_name] >= len(SPLIT_NAMES):
                required_splits = set(SPLIT_NAMES)
            elif class_totals[class_name] == 2:
                # Preserve at least one independent test example when a class
                # has only two unique images after exact deduplication.
                required_splits = {"train", "test"}
            else:
                required_splits = {"train"}
            missing_splits = {
                split
                for split in required_splits
                if assigned[split][class_name] == 0
            }
            if remaining_groups[class_name] <= len(missing_splits):
                allowed_splits.intersection_update(missing_splits)
        if not allowed_splits:
            raise ValueError(
                "Could not preserve class coverage in every split while keeping "
                "exact duplicate groups together. Try changing the split ratios."
            )

        candidate_scores = {}
        for split in allowed_splits:
            score = 0.0
            for class_name in classes:
                for candidate_split in SPLIT_NAMES:
                    count = assigned[candidate_split][class_name]
                    if candidate_split == split:
                        count += 1
                    target = targets[candidate_split][class_name]
                    score += (count - target) ** 2 / max(class_totals[class_name], 1)
            candidate_scores[split] = score
        best_score = min(candidate_scores.values())
        tied = [
            split
            for split, score in candidate_scores.items()
            if abs(score - best_score) < 1e-12
        ]
        selected_split = random.Random(f"{seed}:{digest}").choice(tied)
        for class_name in classes:
            assigned[selected_split][class_name] += 1
            remaining_groups[class_name] -= 1
        assignments.append((selected_split, digest, classes))

    return assignments, class_totals, assigned


def create_deduplicated_split(
    data_dir,
    output_dir,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=42,
):
    """Write a new dataset; never modifies or replaces the source dataset."""
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    test_ratio = 1.0 - train_ratio - val_ratio
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("All split ratios must be greater than zero")
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite: {output_dir}"
        )

    groups, original_count = _group_exact_images(data_dir)
    assignments, class_totals, assigned = _assign_groups(groups, ratios, seed)
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=parent)
    )

    try:
        split_totals = Counter()
        for split_name in SPLIT_NAMES:
            for class_name in class_totals:
                (staging_dir / split_name / class_name).mkdir(
                    parents=True, exist_ok=True
                )
        for split_name, digest, classes in assignments:
            for class_name, source_path in classes.items():
                class_dir = staging_dir / split_name / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                destination = class_dir / source_path.name
                if destination.exists():
                    destination = class_dir / f"{digest[:12]}_{source_path.name}"
                shutil.copy2(source_path, destination)
                split_totals[split_name] += 1

        class_names = sorted(class_totals)
        info = {
            "dataset_name": "LeafDiseaseExactDeduplicated",
            "source_directory": str(data_dir),
            "classes": len(class_names),
            "class_names": class_names,
            "original_image_files": original_count,
            "deduplicated_image_label_pairs": sum(class_totals.values()),
            "removed_exact_duplicate_files": original_count - sum(class_totals.values()),
            "exact_hash": "sha256 of original processed image bytes",
            "split": {
                "train": train_ratio,
                "validation": val_ratio,
                "test": test_ratio,
            },
            "seed": seed,
            "counts": {
                split: {
                    "total": split_totals[split],
                    "classes": dict(sorted(assigned[split].items())),
                }
                for split in SPLIT_NAMES
            },
            "classes_with_no_images_by_split": {
                split: sorted(
                    class_name
                    for class_name in class_names
                    if assigned[split][class_name] == 0
                )
                for split in SPLIT_NAMES
            },
        }
        (staging_dir / "dataset_info.json").write_text(
            json.dumps(info, indent=2) + "\n", encoding="utf-8"
        )
        staging_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return info


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a separate dataset split that deduplicates exact image copies "
            "and keeps each SHA-256 group in only one split."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "processed",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.train_ratio < 1 or not 0 <= args.val_ratio < 1:
        parser.error("train-ratio must be in (0,1) and val-ratio in [0,1)")
    if args.train_ratio + args.val_ratio >= 1:
        parser.error("train-ratio + val-ratio must be less than 1")

    info = create_deduplicated_split(
        args.data_dir,
        args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(json.dumps(info["counts"], indent=2))
    print(
        f"Created {Path(args.output_dir).resolve()} with "
        f"{info['removed_exact_duplicate_files']} exact duplicate files removed."
    )


if __name__ == "__main__":
    main()
