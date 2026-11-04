import re
from pathlib import Path

processed_dir = Path(__file__).resolve().parent / "processed"
renamed = 0

for path in processed_dir.rglob("*"):
    if not path.is_file():
        continue

    clean_name = re.sub(r"\s+(\.[^.]+)$", r"\1", path.name).rstrip()
    if clean_name == path.name:
        continue

    target = path.with_name(clean_name)
    if target.exists():
        raise FileExistsError(f"Cannot rename; target already exists: {target}")

    path.rename(target)
    renamed += 1
    print(f"Renamed: {path.name!r} -> {clean_name!r}")

print(f"Renamed files: {renamed}")

for split in ("train", "val", "test"):
    split_dir = processed_dir / split
    classes = {path.name for path in split_dir.iterdir() if path.is_dir()}
    print(f"{split}: {len(classes)} classes")

remaining = [
    path for path in processed_dir.rglob("*")
    if path.is_file() and path.name != path.name.rstrip()
]
print(f"Files with trailing whitespace: {len(remaining)}")
