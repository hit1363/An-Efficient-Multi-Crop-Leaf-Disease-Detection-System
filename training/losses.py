"""Loss functions used by the training pipeline."""

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="LeafDisease")
class FocalLoss(keras.losses.Loss):
    """Categorical focal loss returning one loss value per example."""

    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = (
                y_true * (1.0 - self.label_smoothing)
                + self.label_smoothing / num_classes
            )

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        alpha_t = tf.reduce_sum(y_true * self.alpha, axis=-1)

        # Keep the batch axis so Keras can apply per-example sample/class weights.
        return alpha_t * tf.pow(1.0 - p_t, self.gamma) * cross_entropy

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "gamma": self.gamma,
                "alpha": self.alpha,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config
