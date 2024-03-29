# -*- coding: utf-8 -*-

"""Custom callbacks."""

import tensorflow as tf
from tensorflow import keras
from utils import gif_summary


class PlayCallback(keras.callbacks.Callback):
    """Play callback."""

    def __init__(self, log_dir: str, dataset: tf.data.Dataset,
                 num_steps: int = 1) -> None:
        """Create PlayCallback.

        Parameters
        ----------
        log_dir: str
            Log directory
        dataset: tf.data.Dataset
            Dataset
        num_steps: int
            Number of steps
        """
        super().__init__()
        self.writer = None
        self.log_dir = log_dir
        self.dataset = dataset
        self.num_steps = num_steps

    def _ensure_writer_exists(self) -> None:
        if self.writer is not None:
            return
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def _close_writer(self) -> None:
        if self.writer is not None:
            return
        self.writer.close()
        self.writer = None

    def on_epoch_end(self, epoch: int, _logs=None) -> None:
        """Epoch end callback."""
        self._ensure_writer_exists()
        data = self.model.predict(
            self.dataset,
            steps=self.num_steps,
            verbose=0,
        )
        with self.writer.as_default():
            for key, val in data.items():
                if len(val.shape) == 5:
                    gif_summary(
                        name=key,
                        tensor=val,
                        step=epoch
                    )
                elif len(val.shape) == 4:
                    tf.summary.image(
                        name=key,
                        data=val[:, :, :, ::-1],
                        step=epoch,
                        max_outputs=val.shape[0]
                    )
                else:
                    raise ValueError(f"Unknown output type for {key}")
        self.writer.flush()

    def on_train_end(self, _logs=None) -> None:
        """Train end callback."""
        self._close_writer()


class TensorBoard(keras.callbacks.TensorBoard):
    """TensorBoard callback with improved weight logging."""

    def _log_weights(self, epoch: int) -> None:
        """Log weights."""
        weight_names = {}
        with self._train_writer.as_default():
            for weight in self.model.trainable_weights:
                weight_name = weight.name.replace(':', '_')
                if weight_name in weight_names:
                    weight_names[weight_name] += 1
                    weight_name += f"_{weight_names[weight_name]}"
                else:
                    weight_names[weight_name] = 0
                tf.summary.histogram(weight_name, weight, step=epoch)
                if self.write_images:
                    self._log_weight_as_image(weight, weight_name, epoch)
        self._train_writer.flush()
