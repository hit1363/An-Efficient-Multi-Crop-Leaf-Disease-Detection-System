import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from training.utils import (
    dataset_batch_count,
    load_dataset,
    per_replica_batch_size,
)


def _write_images(directory, count):
    class_dir = directory / "leaf_disease"
    class_dir.mkdir(parents=True)
    for index in range(count):
        image = np.full((12, 12, 3), index * 20, dtype=np.uint8)
        tf.io.write_file(
            str(class_dir / f"{index}.png"), tf.io.encode_png(image)
        )


def test_training_stream_repeats_for_exact_steps_and_validation_stays_finite(tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    _write_images(train_dir, 5)
    _write_images(val_dir, 3)

    train_ds, val_ds, class_names = load_dataset(
        str(train_dir),
        str(val_dir),
        batch_size=2,
        image_size=(12, 12),
        shuffle_buffer=1,
    )
    train_steps = dataset_batch_count(train_ds, "training")
    val_steps = dataset_batch_count(val_ds, "validation")

    assert class_names == ["leaf_disease"]
    assert train_steps == 3
    assert val_steps == 2
    assert len(list(train_ds.repeat().take(train_steps * 4))) == train_steps * 4
    assert len(list(val_ds)) == val_steps


def test_dataset_batch_count_rejects_unknown_cardinality():
    unknown_dataset = tf.data.Dataset.from_generator(
        lambda: iter([1, 2]), output_signature=tf.TensorSpec((), tf.int32)
    )

    with pytest.raises(ValueError, match="Cannot determine the number of training batches"):
        dataset_batch_count(unknown_dataset, "training")


def test_dataset_smaller_than_batch_still_has_one_finite_step(tmp_path):
    small_dir = tmp_path / "small"
    _write_images(small_dir, 1)

    train_ds, val_ds, _ = load_dataset(
        str(small_dir),
        str(small_dir),
        batch_size=4,
        image_size=(12, 12),
        shuffle_buffer=0,
    )

    assert dataset_batch_count(train_ds, "training") == 1
    assert dataset_batch_count(val_ds, "validation") == 1
    assert next(iter(train_ds))[0].shape[0] == 1


@pytest.mark.parametrize(
    ("global_batch_size", "replicas", "expected"),
    [(32, 1, 32), (128, 2, 64), (96, 4, 24)],
)
def test_per_replica_batch_size(global_batch_size, replicas, expected):
    assert per_replica_batch_size(global_batch_size, replicas) == expected


def test_per_replica_batch_size_rejects_non_divisible_global_batch():
    with pytest.raises(ValueError, match="must be divisible"):
        per_replica_batch_size(127, 2)
