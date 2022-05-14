# -*- coding: utf-8 -*-

"""Utility function."""

from typing import Union
import tempfile
import numpy as np
import tensorflow as tf
from moviepy import editor as mpy
from matplotlib import pyplot as plt


def create_gif(images: np.array, fps: int = 15) -> bytes:
    """Create gif clip.

    Parameters
    ----------
    images: np.array
        Images (N, H, W, C) in BGR format
    fps: int
        FPS

    Returns
    -------
    bytes
        Encoded gif
    """
    # pylint: disable=invalid-name, unexpected-keyword-arg
    images = images[:, :, :, ::-1]
    images = (np.clip(images, 0, 1) * 255).astype(np.uint8)
    clip = mpy.ImageSequenceClip(list(images), fps=fps)
    with tempfile.NamedTemporaryFile(suffix=".gif") as f:
        try:
            clip.write_gif(f.name, verbose=False, progress_bar=False)
        except TypeError:
            clip.write_gif(f.name, verbose=False, logger=None)
        f.seek(0)
        return f.read()


def encode_gif_summary(images: np.array, name: str,
                       fps: int = 15) -> tf.compat.v1.Summary:
    """Encode images into GIF summary.

    Parameters
    ----------
    images: np.array
        Images (B, N, H, W, 3) or (N, H, W, 3) in BGR format
    name: str
        Name
    fps: int
        FPS

    Returns
    -------
    tf.compat.v1.Summary
        Encoded summary
    """
    # pylint: disable=no-member
    shape = images.shape
    if len(shape) == 4:
        shape = (1,) + shape
        images = [images]
    summary = tf.compat.v1.Summary()
    for i in range(shape[0]):
        img_summary = tf.compat.v1.Summary.Image()
        img_summary.height = shape[2]
        img_summary.width = shape[3]
        img_summary.colorspace = 3
        img_summary.encoded_image_string = create_gif(images[i], fps)
        if shape[0] == 1:
            tag = f"{name}/gif"
        else:
            tag = f"{name}/gif/{i}"
        summary.value.add(tag=tag, image=img_summary)
    return summary.SerializeToString()


def gif_summary(name: str, tensor: tf.Tensor, fps: int = 15,
                step: Union[None, int] = None) -> None:
    """Create GIF summary.

    Parameters
    ----------
    name: str
        Name
    tensor: tf.Tensor
        Tensor
    fps: int
        FPS
    step: Union[None, int]
        Step
    """
    tf.summary.experimental.write_raw_pb(
        encode_gif_summary(images=tensor, name=name, fps=fps),
        step
    )


def display_data(dataset: tf.data.Dataset, num_img: int) -> None:
    """Display dataset.

    Paramters
    ---------
    dataset: tf.data.Dataset
        Dataset
    num_img: int
        NUmber of images
    """
    # pylint: disable=invalid-name
    data = [x for x in dataset.unbatch().take(num_img).batch(num_img)][0]
    fig = plt.figure(figsize=(20, 4 * num_img))
    for ind in range(num_img):
        for i in range(10):
            sp = fig.add_subplot(2 * num_img, 10, ind * 20 + 1 + i)
            sp.axis("off")
            plt.imshow(data["input"][ind, i][:, :, ::-1])
        for i in range(10):
            sp = fig.add_subplot(2 * num_img, 10, ind * 20 + 11 + i)
            sp.axis("off")
            plt.imshow(data["target"][ind, i][:, :, ::-1])
    plt.show()
