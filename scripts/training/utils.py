# -*- coding: utf-8 -*-

"""Utility functions."""

import os
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from moviepy import editor as mpy
import tensorflow as tf


def display_data(data):
    """Display some data from the dataset."""
    d = [x for x in data.take(1)][0]
    batch_size = d["input"].shape[0]
    fig = plt.figure(figsize=(20, 4 * batch_size // 4))
    for ind in range(batch_size // 4):
        for i in range(10):
            sp = fig.add_subplot(2 * batch_size // 4, 10, ind * 20 + 1 + i)
            sp.axis("off")
            plt.imshow(d["input"][ind, i])
        for i in range(10):
            sp = fig.add_subplot(2 * batch_size // 4, 10, ind * 20 + 11 + i)
            sp.axis("off")
            plt.imshow(d["target"][ind, i])
    plt.show()


def create_gif(images, fps=3):
    """Create gif."""
    images = (np.minimum(np.maximum(images, 0), 1) * 255
              ).astype(np.uint8)
    clip = mpy.ImageSequenceClip(list(images), fps=fps)
    with tempfile.NamedTemporaryFile() as f:
        filename = f.name + '.gif'
    clip.write_gif(filename, logger=None)
    with open(filename, 'rb') as f:
        gif = f.read()
    os.remove(filename)
    return gif


def encode_gif_summary(images, name, fps=3):
    """Encode TensorBoard summary for GIF images."""
    shape = images.shape
    if len(shape) == 4:
        shape = (1,) + shape
    summary = tf.compat.v1.Summary()
    for i in range(shape[0]):
        img_summary = tf.compat.v1.Summary.Image()
        img_summary.height = shape[2]
        img_summary.width = shape[3]
        img_summary.colorspace = 3
        img_summary.encoded_image_string = create_gif(images[i], fps)
        if shape[0] == 1:
            tag = "{}/gif".format(name)
        else:
            tag = "{}/gif/{}".format(name, i)
        summary.value.add(tag=tag, image=img_summary)
    return summary.SerializeToString()


def gif_summary(name, tensor, fps=3, step=None):
    """Create gif summary."""
    tf.summary.experimental.write_raw_pb(
        encode_gif_summary(images=tensor.numpy(), name=name, fps=fps),
        step
    )
