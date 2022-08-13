# -*- coding: utf-8 -*-

"""Custom layers."""

from typing import Any, Dict, Sequence
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from tensorflow.keras import layers


class UpscaleLayer(layers.Layer):
    """Upscale layer."""

    def __init__(self, scale: int = 2, resize_type: str = "bilinear",
                 **kwargs) -> None:
        """Create UpscaleLayer.

        Parameters
        ----------
        scale: int
            Scale
        resize_type: str
            "bilinear" - bilinear resize
            "nearest" - nearest neighbour
        **kwargs
            keras.layers.Layer args
        """
        super().__init__(**kwargs)
        self.scale = scale
        self.resize_type = resize_type

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Perform upscaling.

        Parameters
        ----------
        inputs: tf.Tensor
            Tensor to upscale (N, H, W, C)

        Returns
        -------
        tf.Tensor
            Upscaled tensor (N, H, W, C)
        """
        if self.resize_type == "bilinear":
            return tf.compat.v1.image.resize_bilinear(
                images=inputs,
                size=tf.shape(inputs)[1:3] * self.scale,
                align_corners=False,
                half_pixel_centers=False
            )
        elif self.resize_type == "nearest":
            return tf.compat.v1.image.resize_nearest_neighbor(
                images=inputs,
                size=tf.shape(inputs)[1:3] * self.scale,
                align_corners=False,
                half_pixel_centers=False
            )
        else:
            raise ValueError(f"Invalid resize_type: {self.resize_type}")

    def get_config(self) -> Dict[str, Any]:
        """Get layer config.

        Returns
        -------
        Dict[str, any]
            Layer config
        """
        config = super().get_config()
        return {
            **config,
            "scale": self.scale,
            "resize_type": self.resize_type,
        }


class DenseWarpLayer(layers.Layer):
    """Dense warping."""

    def call(self, inputs: Sequence[tf.Tensor]) -> tf.Tensor:
        """Perform warping.

        Parameters
        ----------
        inputs: Sequence[tf.Tensor]
            inputs[0]: tensor (N, H, W, C)
            inputs[1]: tensor (N, H, W, 2)

        Returns
        -------
        tf.Tensor
            Warped tensor (N, H, W, C)
        """
        assert len(inputs) == 2
        return tfa.image.dense_image_warp(inputs[0], inputs[1])


class SpaceToDepth(layers.Layer):
    """Space to depth layer."""

    def __init__(self, block_size: int = 2, **kwargs) -> None:
        """Create SpaceToDepth layer.

        Parameters
        ----------
        block_size: int
            Block size
        **kwargs
            keras.layers.Layer args
        """
        super().__init__(**kwargs)
        self.block_size = block_size

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Perform space to depth operation.

        Parameters
        ----------
        inputs: tf.Tensor
            Input tensor (N, H, W, C)

        Returns
        -------
        tf.Tensor
            Output tensor (N, H, W, C * block_size)
        """
        return tf.nn.space_to_depth(inputs, self.block_size)

    def get_config(self) -> Dict[str, Any]:
        """Get layer config.

        Returns
        -------
        Dict[str, Any]
            Layer config
        """
        config = super().get_config()
        return {
            **config,
            "block_size": self.block_size,
        }


class DepthToSpace(layers.Layer):
    """Depth to space layer."""

    def __init__(self, block_size: int = 2, **kwargs) -> None:
        """Create DepthToSpaceLayer.

        Parameters
        ----------
        block_size: int
            Block size
        **kwargs
            keras.layers.Layer args
        """
        super().__init__(**kwargs)
        self.block_size = block_size

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Perform depth to space operation.

        Parameters
        ----------
        inputs: tf.Tensor
            Input tensor (N, H, W, C)

        Returns
        -------
        tf.Tensor
            Output tensor (N, H, W, C / block_size)
        """
        return tf.nn.depth_to_space(inputs, self.block_size)

    def get_config(self) -> Dict[str, Any]:
        """Get layer config.

        Returns
        -------
        Dict[str, Any]
            Layer config
        """
        config = super().get_config()
        return {
            **config,
            "block_size": self.block_size,
        }


class PreprocessLayer(layers.Layer):
    """Image preprocess layer."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Preprocess image.

        Parameters
        ----------
        inputs: tf.Tensor
            Input tensor

        Output
        ------
        tf.Tensor
            Output tensor
        """
        return inputs / 255


class PostprocessLayer(layers.Layer):
    """Image postprocess layer."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Postprocess image.

        Parameters
        ----------
        inputs: tf.Tensor
            Input tensor

        Output
        ------
        tf.Tensor
            Output tensor
        """
        out = inputs
        out = out * 255
        out = K.cast(out, "uint8")
        return out


class ClipLayer(layers.Layer):
    """Value clipping layer."""

    def __init__(self, min_val: float = 0, max_val: float = 1,
                 **kwargs) -> None:
        """Create ClipLayer.

        Parameters
        ----------
        min_val: float
            Minimum value
        max_val: float
            Maximum value
        **kwargs
            keras.layers.Layer args
        """
        super().__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Clip value.

        Parameters
        ----------
        inputs: tf.Tensor
            Input tensor

        Returns
        -------
        tf.Tensor
            Output tensor
        """
        return K.clip(inputs, self.min_val, self.max_val)

    def get_config(self) -> Dict[str, Any]:
        """Get layer config.

        Returns
        -------
        Dict[str, Any]
            Layer config
        """
        config = super().get_config()
        return {
            **config,
            "min_val": self.min_val,
            "max_val": self.max_val,
        }


CUSTOM_LAYERS = {
    "UpscaleLayer": UpscaleLayer,
    "DenseWarpLayer": DenseWarpLayer,
    "SpaceToDepth": SpaceToDepth,
    "DepthToSpace": DepthToSpace,
    "ClipLayer": ClipLayer,
    "PreprocessLayer": PreprocessLayer,
    "PostprocessLayer": PostprocessLayer,
}

__all__ = ["CUSTOM_LAYERS"] + list(CUSTOM_LAYERS)
