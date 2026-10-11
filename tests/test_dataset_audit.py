from PIL import Image

from dataset.audit_splits import audit_dataset


def test_split_audit_finds_exact_cross_split_duplicate(tmp_path):
    image = Image.new("RGB", (16, 16), (20, 100, 40))
    for split in ("train", "val", "test"):
        class_dir = tmp_path / split / "healthy"
        class_dir.mkdir(parents=True)
        image.save(class_dir / "sample.png")

    report = audit_dataset(tmp_path, include_near_duplicates=False)

    assert report["summary"]["exact_duplicate_pairs"] == 3
    assert report["counts"]["train"]["classes"] == {"healthy": 1}


def test_split_audit_reports_class_mismatch(tmp_path):
    for split, class_name in (("train", "healthy"), ("val", "healthy"), ("test", "disease")):
        class_dir = tmp_path / split / class_name
        class_dir.mkdir(parents=True)
        Image.new("RGB", (8, 8), (40, 50, 60)).save(class_dir / "sample.png")

    report = audit_dataset(tmp_path, include_near_duplicates=False)

    assert report["class_parity"]["test"] == {
        "missing": ["healthy"],
        "extra": ["disease"],
    }


def test_split_audit_reports_near_duplicate_candidates(tmp_path):
    for split in ("train", "val", "test"):
        class_dir = tmp_path / split / "healthy"
        class_dir.mkdir(parents=True)
    base = Image.new("RGB", (32, 32), (100, 100, 100))
    changed = base.copy()
    changed.putpixel((0, 0), (255, 255, 255))
    base.save(tmp_path / "train" / "healthy" / "base.png")
    changed.save(tmp_path / "val" / "healthy" / "changed.png")
    Image.new("RGB", (8, 8), (0, 0, 0)).save(
        tmp_path / "test" / "healthy" / "different.png"
    )

    report = audit_dataset(tmp_path, near_duplicate_distance=4)

    assert report["summary"]["near_duplicate_candidates"] >= 1
    assert any(
        candidate["first"].endswith("base.png")
        and candidate["duplicate_candidate"].endswith("changed.png")
        for candidate in report["near_duplicate_candidates_across_splits"]
    )
