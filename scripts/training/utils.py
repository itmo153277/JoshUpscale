# -*- coding: utf-8 -*-

"""Utility functions."""

from typing import Union
import os
import tempfile
import numpy as np
import tensorflow as tf
import imageio
from matplotlib import pyplot as plt


def create_gif(images: np.ndarray, fps: int = 15) -> bytes:
    """Create gif clip.

    Parameters
    ----------
    images: np.ndarray
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
    filename = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            filename = f.name
        with imageio.save(
            filename,
            format="GIF",
            duration=1.0/fps,
            loop=0,
        ) as writer:
            for img in images:
                writer.append_data(img)
        with open(filename, "rb") as f:
            gif = f.read()
    finally:
        if filename is not None and os.path.exists(filename):
            os.remove(filename)
    return gif


def encode_gif_summary(images: np.ndarray, name: str,
                       fps: int = 15) -> tf.compat.v1.Summary:
    """Encode images into GIF summary.

    Parameters
    ----------
    images: np.ndarray
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

    Parameters
    ---------
    dataset: tf.data.Dataset
        Dataset
    num_img: int
        Number of images
    """
    # pylint: disable=invalid-name
    spec = dataset.element_spec
    seq_len = spec["input"].shape[1]
    data = list(dataset.unbatch().take(num_img).batch(num_img))[0]
    fig = plt.figure(figsize=(2 * seq_len, 4 * num_img))
    for ind in range(num_img):
        for i in range(seq_len):
            sp = fig.add_subplot(2 * num_img, seq_len,
                                 ind * 2 * seq_len + 1 + i)
            sp.axis("off")
            plt.imshow(data["input"][ind, i][:, :, ::-1])
        if "last" in spec:
            sp = fig.add_subplot(2 * num_img, seq_len,
                                 (ind + 1) * 2 * seq_len - 1)
            sp.axis("off")
            plt.imshow(data["last"][ind][:, :, ::-1])
            sp = fig.add_subplot(2 * num_img, seq_len, (ind + 1) * 2 * seq_len)
            sp.axis("off")
            plt.imshow(data["target"][ind][:, :, ::-1])
        else:
            for i in range(seq_len):
                sp = fig.add_subplot(2 * num_img, seq_len,
                                     (ind * 2 + 1) * seq_len + 1 + i)
                sp.axis("off")
                plt.imshow(data["target"][ind, i][:, :, ::-1])
    plt.show()
