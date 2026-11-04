import numpy as np
from PIL import Image

from quantization.post_training_quant import (
    representative_dataset_generator,
    stratified_calibration_paths,
)


def _make_calibration_tree(root, counts):
    for class_index, (class_name, image_count) in enumerate(sorted(counts.items())):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for image_index in range(image_count):
            Image.new(
                "RGB",
                (8, 8),
                (class_index * 10, image_index * 20, (class_index + image_index) % 255),
            ).save(class_dir / f"img_{image_index}.png")


def test_stratified_paths_cover_every_class(tmp_path):
    _make_calibration_tree(
        tmp_path,
        {
            "Apple___A": 30,
            "Banana___B": 30,
            "Cherry___C": 30,
            "Guava___Degenerate": 2,
        },
    )

    paths = stratified_calibration_paths(tmp_path, num_samples=30)

    by_class = {}
    for path in paths:
        by_class[path.parent.name] = by_class.get(path.parent.name, 0) + 1
    assert set(by_class) == {
        "Apple___A",
        "Banana___B",
        "Cherry___C",
        "Guava___Degenerate",
    }
    assert by_class["Guava___Degenerate"] == 2
    assert sum(by_class.values()) == 30
    full_classes = [
        count for name, count in by_class.items() if name != "Guava___Degenerate"
    ]
    assert max(full_classes) - min(full_classes) <= 1

    replay = stratified_calibration_paths(tmp_path, num_samples=30)
    assert [str(path) for path in replay] == [str(path) for path in paths]


def test_stratified_paths_fall_back_to_flat_directory(tmp_path):
    for index in range(5):
        Image.new("RGB", (6, 6), (index * 30, 9, 9)).save(
            tmp_path / f"loose_{index}.png"
        )

    paths = stratified_calibration_paths(tmp_path, num_samples=3)

    assert len(paths) == 3
    assert {path.parent for path in paths} == {tmp_path}


def test_representative_generator_yields_preprocessed_batches(tmp_path):
    _make_calibration_tree(tmp_path, {"Alpha": 3, "Beta": 3})

    batches = list(
        representative_dataset_generator(
            tmp_path, num_samples=5, preprocess_fn=lambda images: images / 255.0
        )
    )

    assert len(batches) == 5
    for (batch,) in batches:
        assert tuple(batch.shape) == (1, 224, 224, 3)
        assert batch.dtype.name == "float32"
        assert float(np.max(batch.numpy())) <= 1.0
