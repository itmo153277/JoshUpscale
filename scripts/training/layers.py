# -*- coding: utf-8 -*-

"""Layer definitions."""

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers


class DenseWarpLayer(layers.Layer):
    """Layer for dense warping."""

    # pylint: disable=arguments-differ

    def call(self, inputs):
        """
        Perform dense warping.

        Parameters
        ----------
        inputs : tf.Tensor array
            Images and displacement map (NHWC)

        Returns
        -------
        tf.Tensor
            Warped image (NHWC)
        """
        return tfa.image.dense_image_warp(inputs[0], inputs[1])


class UpscaleLayer(layers.Layer):
    """Layer for 2x upscaling."""

    # pylint: disable=arguments-differ

    def call(self, images):
        """
        Perform 2x upscaling (bilinear).

        Parameters
        ----------
        images: tf.Tensor
            Images to upscale (NHWC)

        Returns
        -------
        tf.Tensor
            Upscaled images (NHWC)
        """
        return tf.compat.v1.image.resize_bilinear(
            images=images,
            size=tf.shape(images)[1:3] * 2,
            align_corners=False,
            half_pixel_centers=False
        )
