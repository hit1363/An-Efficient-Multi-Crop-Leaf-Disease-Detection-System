"""Metrics for single-label multiclass classification."""

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="LeafDisease")
class MacroPrecisionRecall(keras.metrics.Metric):
    """Macro-averaged precision or recall from an accumulated confusion matrix."""

    def __init__(self, num_classes, metric="precision", name=None, **kwargs):
        if metric not in {"precision", "recall"}:
            raise ValueError("metric must be 'precision' or 'recall'")
        self.num_classes = int(num_classes)
        self.metric = metric
        super().__init__(name=name or f"macro_{metric}", **kwargs)
        self.confusion_matrix = self.add_weight(
            name="confusion_matrix",
            shape=(self.num_classes, self.num_classes),
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1, output_type=tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.reshape(sample_weight, [-1])

        batch_confusion = tf.math.confusion_matrix(
            y_true,
            y_pred,
            num_classes=self.num_classes,
            weights=sample_weight,
            dtype=self.dtype,
        )
        self.confusion_matrix.assign_add(batch_confusion)

    def result(self):
        true_positives = tf.linalg.diag_part(self.confusion_matrix)
        if self.metric == "precision":
            denominator = tf.reduce_sum(self.confusion_matrix, axis=0)
        else:
            denominator = tf.reduce_sum(self.confusion_matrix, axis=1)
        per_class = tf.math.divide_no_nan(true_positives, denominator)
        return tf.reduce_mean(per_class)

    def reset_state(self):
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes, "metric": self.metric})
        return config
