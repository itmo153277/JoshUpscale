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
