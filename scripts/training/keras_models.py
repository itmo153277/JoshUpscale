# -*- coding: utf-8 -*-

"""Custom models."""

import itertools
from typing import Any, Dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_layers import DenseWarpLayer


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
            ),
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
        """Build model arguments."""
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


class FRVSRModel(JoshUpscaleModel):
    """FRVSR model."""

    # pylint: disable=invalid-name

    def __init__(self, generator_model: keras.Model, flow_model: keras.Model,
                 crop_size: int, *args, **kwargs) -> None:
        """Create FRVSRModel.

        Parameters
        ----------
        generator_model: keras.Model
            Generator model
        flow_model: keras.Model
            Flow model
        crop_size: int
            Image size
        inference_model: keras.Model
            Inference model (for JoshUpscaleModel)
        """
        super().__init__(*args, **kwargs,
                         **FRVSRModel._build_model_args(
                             generator_model=generator_model,
                             flow_model=flow_model,
                             crop_size=crop_size
                         ))
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
            ),
        )

    def compute_loss(self, x, y, y_pred, sample_weight):
        """Compute loss."""
        assert y is None
        assert sample_weight is None
        inputs = x["input"]
        targets = x["target"]
        gen_outputs = y_pred["gen_outputs"]
        target_warp = y_pred["target_warp"]

        # assuming that in distributed training batches are equal
        # across replicas
        batch_size = tf.shape(inputs)[0]
        num_replicas = tf.distribute.get_strategy().num_replicas_in_sync

        gen_outputs_loss = tf.math.squared_difference(gen_outputs, targets)
        gen_outputs_loss = tf.math.reduce_sum(gen_outputs_loss, axis=-1)
        gen_outputs_loss = tf.math.reduce_mean(gen_outputs_loss)
        target_warp_loss = tf.math.squared_difference(
            target_warp, targets[:, 1:])
        target_warp_loss = tf.math.reduce_sum(target_warp_loss, axis=-1)
        target_warp_loss = tf.math.reduce_mean(target_warp_loss)
        loss = tf.add_n([gen_outputs_loss, target_warp_loss])

        self.gen_outputs_loss_tr.update_state(
            gen_outputs_loss, sample_weight=batch_size * 10)
        self.target_warp_loss_tr.update_state(
            target_warp_loss, sample_weight=batch_size * 9)
        self.loss_tr.update_state(loss, sample_weight=batch_size)

        # Average loss across all replicas is summed
        return loss / num_replicas

    @staticmethod
    def _build_model_args(
        generator_model: keras.Model,
        flow_model: keras.Model,
        crop_size: int,
    ) -> Dict[str, Any]:
        """Build model arguments."""
        inputs = keras.Input(shape=[10, crop_size, crop_size, 3],
                             name="input", dtype="float32")
        targets = keras.Input(shape=[10, crop_size*4, crop_size*4, 3],
                              name="target", dtype="float32")
        input_shape = tf.shape(inputs[:, 0, :, :, :])
        output_shape = tf.shape(targets[:, 0, :, :, :])
        input_frames = tf.reshape(inputs[:, 1:, :, :, :],
                                  [-1, crop_size, crop_size, 3])
        input_frames_pre = tf.reshape(inputs[:, :-1, :, :, :],
                                      [-1, crop_size, crop_size, 3])
        num_rand_frames = len(flow_model.inputs) - 2
        if num_rand_frames > 0:
            # Random last frames for flow generation
            rand_frames = tf.reshape(
                tf.stack([
                    tf.random.uniform(
                        input_shape,
                        minval=0,
                        maxval=1,
                        dtype="float32"
                    )
                    for _ in range(num_rand_frames)
                ], axis=1),
                [-1, num_rand_frames, crop_size, crop_size, 3]
            )
        flow_last_frames = [
            tf.reshape(
                tf.concat([
                    rand_frames[:, -(i + 1):, :, :, :],
                    inputs[:, :-(i + 2), :, :, :]
                ], axis=1),
                [-1, crop_size, crop_size, 3]
            )
            for i in range(num_rand_frames)
        ]
        target_frames_pre = tf.reshape(
            targets[:, :-1, :, :, :],
            [-1, crop_size*4, crop_size*4, 3]
        )
        flow = flow_model([input_frames, input_frames_pre] + flow_last_frames)
        target_warp = DenseWarpLayer()([target_frames_pre, flow])
        target_warp = tf.reshape(
            target_warp, [-1, 9, crop_size*4, crop_size*4, 3])
        target_warp = layers.Layer(name="target_warp",
                                   dtype="float32")(target_warp)
        flow = tf.reshape(flow, [-1, 9, crop_size*4, crop_size*4, 2])
        # First frame uses random pre_warp
        last_output = generator_model([
            inputs[:, 0, :, :, :],
            tf.random.uniform(output_shape, minval=0,
                              maxval=1, dtype="float32")
        ])
        gen_outputs = [last_output]
        for frame_i in range(9):
            cur_flow = flow[:, frame_i, :, :, :]
            gen_pre_output_warp = DenseWarpLayer()([last_output, cur_flow])
            last_output = generator_model([
                inputs[:, frame_i + 1, :, :, :],
                gen_pre_output_warp
            ])
            gen_outputs.append(last_output)
        gen_outputs = tf.reshape(
            tf.stack(gen_outputs, axis=1),
            [-1, 10, crop_size*4, crop_size*4, 3]
        )
        gen_outputs = layers.Layer(
            name="gen_outputs", dtype="float32")(gen_outputs)
        return {
            "inputs": [inputs, targets],
            "outputs": {
                "gen_outputs": gen_outputs,
                "target_warp": target_warp
            }
        }
