# -*- coding: utf-8 -*-

"""Utility functions."""

from typing import Union, Any, Tuple, Iterable, List
import os
import logging
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow import keras
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
    images = (np.clip(images + 0.5, 0, 1) * 255).astype(np.uint8)
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
                       fps: int = 15) -> Any:
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
    Summary
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
            plt.imshow(data["input"][ind, i][:, :, ::-1] + 0.5)
        if "last" in spec:
            sp = fig.add_subplot(2 * num_img, seq_len,
                                 (ind + 1) * 2 * seq_len - 1)
            sp.axis("off")
            plt.imshow(data["last"][ind][:, :, ::-1] + 0.5)
            sp = fig.add_subplot(2 * num_img, seq_len, (ind + 1) * 2 * seq_len)
            sp.axis("off")
            plt.imshow(data["target"][ind][:, :, ::-1] + 0.5)
        else:
            for i in range(seq_len):
                sp = fig.add_subplot(2 * num_img, seq_len,
                                     (ind * 2 + 1) * seq_len + 1 + i)
                sp.axis("off")
                plt.imshow(data["target"][ind, i][:, :, ::-1] + 0.5)
    plt.show()


BGR_LUMA = [0.1140, 0.5870, 0.2989]


def lcs(left: List[Any], right: List[Any]) -> Iterable[Tuple[int, int]]:
    """Find longest common subsequence."""
    m = len(left)
    n = len(right)
    lengths = [[None]*(n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                lengths[i][j] = 0
            elif left[i - 1] == right[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])
    best_val = 0
    start_idx = 0
    for i in range(m + 1):
        for j in range(start_idx, n + 1):
            if lengths[i][j] > best_val:
                best_val = lengths[i][j]
                start_idx = j + 1
                yield (i - 1, j - 1)
                break


def copy_model_variables(model_from: keras.Model, model_to: keras.Model) \
        -> None:
    """Copy savable variables from one model to another."""
    # pylint: disable=protected-access
    # pylint: disable=import-outside-toplevel
    # pylint: disable=unidiomatic-typecheck
    from keras.src.saving.saving_lib import _walk_saveable
    from keras.src.saving.keras_saveable import KerasSaveable

    visited_from = set()
    visited_to = set()
    # Note: touched dicts do not include aborted branches
    touched_from = {}
    touched_to = {}

    def walk(obj_from, obj_to):
        if isinstance(obj_from, (list, dict, tuple, set)):
            if type(obj_from) != type(obj_to):  # noqa
                return
            if isinstance(obj_from, dict):
                for k, v in obj_from.items():
                    target = obj_to.get(k)
                    if target is None:
                        continue
                    walk(v, target)
                touched_from.update({id(x): x
                                     for x in obj_from.values()
                                     if isinstance(x, KerasSaveable)})
                touched_to.update({id(x): x
                                   for x in obj_to.values()
                                   if isinstance(x, KerasSaveable)})
            else:
                if all(hasattr(x, "name") for x in obj_from) \
                        and all(hasattr(x, "name") for x in obj_to):
                    dict_to = {x.name: x for x in obj_to}
                    for v in obj_from:
                        target = dict_to.get(v.name)
                        if target is None:
                            continue
                        walk(v, target)
                else:
                    obj_from = list(obj_from)
                    obj_to = list(obj_to)
                    for idx_from, idx_to in lcs([type(x) for x in obj_from],
                                                [type(x) for x in obj_to]):
                        walk(obj_from[idx_from], obj_to[idx_to])
                touched_from.update({id(x): x
                                     for x in obj_from
                                     if isinstance(x, KerasSaveable)})
                touched_to.update({id(x): x
                                   for x in obj_to
                                   if isinstance(x, KerasSaveable)})
            return
        if not isinstance(obj_from, KerasSaveable):
            return
        if id(obj_from) in visited_from:
            return
        visited_from.add(id(obj_from))
        visited_to.add(id(obj_to))
        if id(obj_from) == id(obj_to):
            return
        if hasattr(obj_from, "save_own_variables") \
                and hasattr(obj_to, "save_own_variables"):
            vars_from = {}
            vars_to = {}
            use_internal_variables = False
            if hasattr(obj_from, "_variables") \
                    and hasattr(obj_to, "_variables"):
                use_internal_variables = True
                var_names = {}
                for v in obj_from._variables:
                    var_name = v.path
                    if var_name in var_names:
                        var_names[var_name] += 1
                        var_name = var_name + f"_{var_names[var_name]}"
                    else:
                        var_names[var_name] = 0
                    if var_name in vars_from:
                        use_internal_variables = False
                        break
                    vars_from[v.path] = v
                var_names = {}
                for v in obj_to._variables:
                    var_name = v.path
                    if var_name in var_names:
                        var_names[var_name] += 1
                        var_name = var_name + f"_{var_names[var_name]}"
                    else:
                        var_names[var_name] = 0
                    if var_name in vars_to:
                        use_internal_variables = False
                        break
                    vars_to[v.path] = v
            if not use_internal_variables:
                vars_from = {}
                vars_to = {}
                obj_from.save_own_variables(vars_from)
                obj_to.save_own_variables(vars_to)
            copied_from = set()
            copied_to = set()
            if not use_internal_variables \
                    and all(x.isdigit() for x in vars_from) \
                    and all(x.isdigit() for x in vars_to):
                vars_from_list = list(sorted(vars_from.keys(), key=int))
                vars_to_list = list(sorted(vars_to.keys(), key=int))
                for idx_from, idx_to in lcs(
                    [(vars_from[x].shape, vars_from[x].dtype)
                     for x in vars_from_list],
                    [(vars_to[x].shape, vars_to[x].dtype)
                     for x in vars_to_list]
                ):
                    if use_internal_variables:
                        vars_to[vars_to_list[idx_to]].assign(
                            vars_from[vars_from_list[idx_from]]
                        )
                    else:
                        vars_to[vars_to_list[idx_to]] = \
                            vars_from[vars_from_list[idx_from]]
                    copied_from.add(vars_from_list[idx_from])
                    copied_to.add(vars_to_list[idx_to])
            else:
                for k, v in vars_from.items():
                    target = vars_to.get(k)
                    if target is None:
                        continue
                    if target.dtype != v.dtype or target.shape != v.shape:
                        continue
                    copied_from.add(k)
                    copied_to.add(k)
                    if use_internal_variables:
                        vars_to[k].assign(v)
                    else:
                        vars_to[k] = v
            not_copied_from = set(vars_from.keys()) - copied_from
            not_copied_to = set(vars_to.keys()) - copied_to
            if len(not_copied_from) > 0:
                logging.warning(
                    "Not copied %d variables from %s: %s",
                    len(not_copied_from), obj_from, not_copied_from)
            if len(not_copied_to) > 0:
                logging.warning(
                    "Not copied %d variables to %s: %s",
                    len(not_copied_to), obj_to, not_copied_to)

            if not use_internal_variables:
                obj_to.load_own_variables(vars_to)

        dict_to = dict(_walk_saveable(obj_to))
        for k, v in _walk_saveable(obj_from):
            if isinstance(v, KerasSaveable):
                touched_from[id(v)] = v
            target = dict_to.get(k)
            if target is None:
                continue
            walk(v, target)
        touched_to.update({id(x): x
                           for x in dict_to.values()
                           if isinstance(x, KerasSaveable)})
    walk(model_from, model_to)
    not_copied_from = set(touched_from.keys()) - visited_from
    not_copied_to = set(touched_to.keys()) - visited_to
    if len(not_copied_from) > 0:
        logging.warning(
            "Not copied from %d savables: %s", len(not_copied_from),
            [x for x in touched_from.values() if id(x) in not_copied_from]
        )
    if len(not_copied_to) > 0:
        logging.warning(
            "Not copied to %d savables: %s", len(not_copied_to),
            [x for x in touched_to.values() if id(x) in not_copied_to]
        )
