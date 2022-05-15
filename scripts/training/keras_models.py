# -*- coding: utf-8 -*-

"""Custom models."""

import itertools
from typing import Any, Dict
import tensorflow as tf
from tensorflow import keras


class JoshUpscaleModel(keras.Model):
    """JoshUpscale model."""

    # pylint: disable=invalid-name

    def __init__(self, full_model: keras.Model, **kwargs) -> None:
        """Create JoshUpscaleModel.

        Parameters
        ----------
        full_model: keras.Model
            Full model for inference
        """
        super().__init__(**kwargs)
        self.full_model = full_model

    def predict_step(self, data: Any) -> Dict[str, tf.Tensor]:
        """Prediction step.

        Input["input"]: (N, 10, H, W, 3)
        Output["gen_outputs"]: Generated outputs (N, 18, H * 4, W * 4, 3)
        Output["pre_warp"]: Warped outputs (N, 18, H * 4, W * 4, 3)

        Parameters
        ----------
        data: Any
            Data for prediction (keras format)

        Return
        ------
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
            (len(self.full_model.inputs) - 2)
        last_output = tf.zeros((batch_size, height*4, width*4, 3))
        gen_outputs = []
        pre_warps = []
        for i in itertools.chain(range(10), range(8, 0, -1)):
            cur_frame = inputs[:, i, :, :, :]
            outputs = self.full_model(
                [cur_frame, last_output] + last_frames,
                training=False
            )
            last_output = outputs["output_raw"]
            gen_outputs.append(last_output)
            pre_warps.append(outputs["pre_warp"])
            last_frames = outputs["last_frames"]
        gen_outputs = tf.stack(gen_outputs, axis=1)
        pre_warps = tf.stack(pre_warps[1:], axis=1)
        return {"gen_output": gen_outputs, "pre_warp": pre_warps}
