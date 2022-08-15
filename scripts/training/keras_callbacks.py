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
        data = self.model.predict(self.dataset, steps=self.num_steps)
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
                        data=val,
                        step=epoch,
                        max_outputs=val.shape[0]
                    )
                else:
                    raise ValueError(f"Unknown output type for {key}")
        self.writer.flush()

    def on_train_end(self, _logs=None) -> None:
        """Train end callback."""
        self._close_writer()
