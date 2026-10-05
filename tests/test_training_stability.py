from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from training.model import unfreeze_base_model
from training.train import (
    FocalLoss,
    build_metrics,
    compile_model,
    create_fine_tune_optimizer,
    select_final_weights,
)
from training.utils import compute_class_weights, setup_callbacks
from training.metrics import MacroPrecisionRecall


def test_focal_loss_is_per_example_and_uses_target_class_alpha():
    y_true = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y_pred = tf.constant([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
    loss = FocalLoss(gamma=0.0, alpha=0.25)

    actual = loss.call(y_true, y_pred).numpy()
    expected = -0.25 * np.log([0.8, 0.7])

    assert actual.shape == (2,)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_focal_loss_accepts_per_example_sample_weights():
    y_true = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    y_pred = tf.constant([[0.9, 0.1], [0.4, 0.6]])
    loss = FocalLoss(gamma=0.0, alpha=0.25)

    actual = loss(y_true, y_pred, sample_weight=tf.constant([1.0, 0.0])).numpy()
    expected = -0.25 * np.log(0.9) / 2.0

    assert actual == pytest.approx(expected, rel=1e-6)


def test_macro_precision_recall_are_classwise_not_top_one_micro_scores():
    y_true = tf.constant(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
         [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    )
    y_pred = tf.constant(
        [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1],
         [0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    )
    precision = MacroPrecisionRecall(3, metric="precision", name="precision")
    recall = MacroPrecisionRecall(3, metric="recall", name="recall")

    precision.update_state(y_true, y_pred)
    recall.update_state(y_true, y_pred)

    assert precision.result().numpy() == pytest.approx(5.0 / 6.0)
    assert recall.result().numpy() == pytest.approx(2.0 / 3.0)


def test_build_metrics_uses_macro_precision_recall_and_one_vs_rest_auc():
    precision, recall, auc = build_metrics(["precision", "recall", "auc"], 3)

    assert isinstance(precision, MacroPrecisionRecall)
    assert precision.metric == "precision"
    assert isinstance(recall, MacroPrecisionRecall)
    assert recall.metric == "recall"
    assert auc.multi_label is True
    assert auc.num_labels == 3


def test_macro_auc_averages_one_vs_rest_class_auc():
    _, _, auc = build_metrics(["auc"], 3)
    y_true = tf.constant(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    y_pred = tf.constant(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    )

    auc.update_state(y_true, y_pred)

    assert auc.result().numpy() == pytest.approx(1.0)


def test_custom_loss_and_metric_are_registered_for_model_loading():
    loss = FocalLoss(gamma=1.5, alpha=0.4, label_smoothing=0.1)
    metric = MacroPrecisionRecall(3, metric="recall", name="recall")

    restored_loss = tf.keras.losses.deserialize(tf.keras.losses.serialize(loss))
    restored_metric = tf.keras.metrics.deserialize(tf.keras.metrics.serialize(metric))

    assert isinstance(restored_loss, FocalLoss)
    assert restored_loss.gamma == pytest.approx(1.5)
    assert isinstance(restored_metric, MacroPrecisionRecall)
    assert restored_metric.metric == "recall"


def _config():
    return {
        "model": {"num_classes": 2},
        "optimizer": {
            "name": "adam",
            "learning_rate": 3e-4,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-7,
            "clipnorm": 1.0,
        },
        "training": {"fine_tune_learning_rate": 5e-6},
        "loss": {"name": "categorical_crossentropy"},
        "metrics": ["accuracy"],
    }


def test_keeps_phase_one_weights_when_fine_tuning_is_worse():
    phase1 = SimpleNamespace(best=0.42)
    phase2 = SimpleNamespace(best=0.51, weights=["phase-two"])

    weights, phase, loss = select_final_weights(phase1, ["phase-one"], phase2)

    assert weights == ["phase-one"]
    assert phase == "phase_1"
    assert loss == pytest.approx(0.42)


def test_accepts_phase_two_weights_only_on_strict_improvement():
    phase1 = SimpleNamespace(best=0.42)
    phase2 = SimpleNamespace(best=0.41, weights=["phase-two"])

    weights, phase, loss = select_final_weights(phase1, ["phase-one"], phase2)

    assert weights == ["phase-two"]
    assert phase == "phase_2"
    assert loss == pytest.approx(0.41)


def test_unfreezing_keeps_batch_normalization_frozen():
    base_model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer((8, 8, 3)),
            tf.keras.layers.Conv2D(4, 1, name="early_conv"),
            tf.keras.layers.BatchNormalization(name="batch_norm"),
            tf.keras.layers.Conv2D(4, 1, name="late_conv"),
        ]
    )

    unfreeze_base_model(base_model, unfreeze_from_layer=1)

    assert base_model.get_layer("early_conv").trainable is False
    assert base_model.get_layer("batch_norm").trainable is False
    assert base_model.get_layer("late_conv").trainable is True


def test_gradient_clipping_is_applied_in_both_training_phases():
    config = _config()
    model = tf.keras.Sequential(
        [tf.keras.layers.InputLayer((4,)), tf.keras.layers.Dense(2, activation="softmax")]
    )

    compile_model(model, config)
    fine_tune_optimizer = create_fine_tune_optimizer(config)

    assert model.optimizer.clipnorm == pytest.approx(1.0)
    assert fine_tune_optimizer.clipnorm == pytest.approx(1.0)


def test_capped_class_weights_are_bounded_and_sample_normalized(tmp_path):
    counts = {"common": 100, "rare": 2}
    for class_name, count in counts.items():
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for index in range(count):
            (class_dir / f"{index}.jpg").touch()

    weights = compute_class_weights(tmp_path, max_weight=3.0, normalize=True)
    sample_weighted_mean = sum(
        counts[class_name] * weights[index]
        for index, class_name in enumerate(sorted(counts))
    ) / sum(counts.values())

    assert max(weights.values()) <= 3.0
    assert sample_weighted_mean == pytest.approx(1.0)


def test_phase_two_csv_logger_appends_to_phase_one_log(tmp_path):
    config = {"callbacks": {"csv_logger": {"enabled": True, "filename": str(tmp_path / "run.csv")}}}

    phase_one = setup_callbacks(config, csv_append=False)
    phase_two = setup_callbacks(config, csv_append=True)
    phase_one_logger = next(callback for callback in phase_one if isinstance(callback, tf.keras.callbacks.CSVLogger))
    phase_two_logger = next(callback for callback in phase_two if isinstance(callback, tf.keras.callbacks.CSVLogger))

    assert phase_one_logger.append is False
    assert phase_two_logger.append is True


def test_csv_logger_uses_the_configured_append_default(tmp_path):
    config = {"callbacks": {"csv_logger": {"enabled": True, "filename": str(tmp_path / "run.csv"), "append": True}}}

    callbacks = setup_callbacks(config)
    csv_logger = next(callback for callback in callbacks if isinstance(callback, tf.keras.callbacks.CSVLogger))

    assert csv_logger.append is True
