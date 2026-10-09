import pytest

tf = pytest.importorskip("tensorflow")

from training.utils import load_dataset


def _write_image(path):
    image = tf.ones((4, 4, 3), dtype=tf.uint8)
    tf.io.write_file(str(path), tf.io.encode_jpeg(image))


def _create_split(root, image_count):
    for index in range(image_count):
        class_dir = root / ("healthy" if index % 2 == 0 else "diseased")
        class_dir.mkdir(parents=True, exist_ok=True)
        _write_image(class_dir / f"image_{index}.jpg")


def _batch_sizes(dataset):
    return [int(images.shape[0]) for images, _ in dataset]


def test_load_dataset_drops_partial_batches_when_requested(tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    _create_split(train_dir, image_count=3)
    _create_split(val_dir, image_count=2)

    full_batches, _, _ = load_dataset(
        train_dir,
        val_dir,
        batch_size=2,
        image_size=(4, 4),
        shuffle_buffer=0,
        drop_remainder=True,
    )
    all_batches, _, _ = load_dataset(
        train_dir,
        val_dir,
        batch_size=2,
        image_size=(4, 4),
        shuffle_buffer=0,
        drop_remainder=False,
    )

    assert _batch_sizes(full_batches) == [2]
    assert _batch_sizes(all_batches) == [2, 1]
