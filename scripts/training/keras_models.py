# -*- coding: utf-8 -*-

"""Custom models."""

import itertools
from typing import Any, Dict
import tensorflow as tf
from tensorflow import keras


class JoshUpscaleModel(keras.Model):
    """JoshUpscale model."""

    # pylint: disable=invalid-name

    def __init__(self, inference_model: keras.Model, **kwargs) -> None:
        """Create JoshUpscaleModel.

        Parameters
        ----------
        inference_model: keras.Model
            Inference model
        """
        super().__init__(**kwargs)
        self.inference_model = inference_model

    def predict_step(self, data: Any) -> Dict[str, tf.Tensor]:
        """Prediction step.

        Inputs:
        - (N x 10 x H x W x 3) - Input sequence
        Outputs:
        - gen_outputs (N x 18 x H*4 x W*4 x 3) - Generated outputs
        - pre_warp (N x 17 x H*4 x W*4 x 3) - Warped outputs

        Parameters
        ----------
        data: Any
            Data for prediction (keras format)

        Returns
        -------
        Dict[str, tf.Tensor]
            Predicted values
        """
        x, _, _ = keras.utils.unpack_x_y_sample_weight(data)
        inputs = x["input"]
        shape = tf.shape(inputs)
        batch_size = shape[0]
        height = shape[2]
        width = shape[3]
        last_frames = [tf.zeros((batch_size, height, width, 3))] * \
            (len(self.inference_model.inputs) - 2)
        last_output = tf.zeros((batch_size, height*4, width*4, 3))
        gen_outputs = []
        pre_warps = []
        for i in itertools.chain(range(10), range(8, 0, -1)):
            cur_frame = inputs[:, i, :, :, :]
            outputs = self.inference_model(
                [cur_frame, last_output] + last_frames,
                training=False
            )
            last_output = outputs["output_raw"]
            gen_outputs.append(last_output)
            if i > 0:
                pre_warps.append(outputs["pre_warp"])
            last_frames = outputs["last_frames"]
        gen_outputs = tf.stack(gen_outputs, axis=1)
        pre_warps = tf.stack(pre_warps[1:], axis=1)
        return {"gen_output": gen_outputs, "pre_warp": pre_warps}


class FRVSRModelSingle(keras.Model):
    """FRVSR model (single)."""

    # pylint: disable=invalid-name

    def __init__(self, inference_model: keras.Model, crop_size: int,
                 *args, **kwargs):
        """Create FRVSRModelSingle.

        Parameters
        ----------
        inference_model: keras.Model
            Inference model
        crop_size: int
            Image size
        """
        super().__init__(
            *args, **kwargs,
            **FRVSRModelSingle._build_model_args(
                inference_model=inference_model,
                crop_size=crop_size,
            )
        )
        self.gen_outputs_loss_tr = keras.metrics.Mean(name="gen_outputs_loss")
        self.target_warp_loss_tr = keras.metrics.Mean(name="target_warp_loss")
        self.loss_tr = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        """Get metrics."""
        return [
            self.loss_tr,
            self.gen_outputs_loss_tr,
            self.target_warp_loss_tr,
        ] + self.compiled_metrics.metrics

    def compile(self, learning_rate: Any = 0.0005, **kwargs) -> None:
        """Compile model.

        Parameters
        ----------
        learning_rate: Any
            Learning rate
        **kwargs
            keras.Model.compile args
        """
        super().compile(
            **kwargs,
            loss=None,
            optimizer=keras.optimizers.Adam(
                learning_rate=learning_rate
            )
        )

    def compute_loss(self, x, y, y_pred, sample_weight):
        """Compute loss."""
        assert y is None
        assert sample_weight is None
        inputs = x["input"]
        target = x["target"]
        gen_output = y_pred["gen_output"]
        pre_warp = y_pred["pre_warp"]

        # assuming that in distributed training batches are equal
        # across replicas
        batch_size = tf.shape(inputs)[0]
        num_replicas = tf.distribute.get_strategy().num_replicas_in_sync

        gen_outputs_loss = tf.math.squared_difference(gen_output, target)
        gen_outputs_loss = tf.math.reduce_sum(gen_outputs_loss, axis=-1)
        gen_outputs_loss = tf.math.reduce_mean(gen_outputs_loss)
        target_warp_loss = tf.math.squared_difference(pre_warp, target)
        target_warp_loss = tf.math.reduce_sum(target_warp_loss, axis=-1)
        target_warp_loss = tf.math.reduce_mean(target_warp_loss)
        loss = tf.add_n([gen_outputs_loss, target_warp_loss])

        self.gen_outputs_loss_tr.update_state(
            gen_outputs_loss, sample_weight=batch_size)
        self.target_warp_loss_tr.update_state(
            target_warp_loss, sample_weight=batch_size)
        self.loss_tr.update_state(loss, sample_weight=batch_size)

        # Average loss across all replicas is summed
        return loss / num_replicas

    @staticmethod
    def _build_model_args(
        inference_model: keras.Model,
        crop_size: int,
    ) -> Dict[str, Any]:
        """Build model arguments.

        Parameters
        ----------
        inference_model: keras.Model
            Inference model
        crop_size: int
            Image size

        Returns
        -------
        Dict[str, Any]
            Model arguments
        """
        num_frames = len(inference_model.inputs) - 1
        inputs = keras.Input(
            shape=[num_frames, crop_size, crop_size, 3],
            name="input",
            dtype="float32"
        )
        target = keras.Input(
            shape=[crop_size*4, crop_size*4, 3],
            name="target",
            dtype="float32"
        )
        last = keras.Input(
            shape=[crop_size*4, crop_size*4, 3],
            name="last",
            dtype="float32"
        )
        input_frames = [inputs[:, i] for i in range(num_frames)]
        output = inference_model([input_frames[-1], last] + input_frames[:-1])
        return {
            "inputs": [inputs, target, last],
            "outputs": {
                "gen_output": output["output_raw"],
                "pre_warp": output["pre_warp"],
            }
        }
