# -*- coding: utf-8 -*-

"""Dataset routines."""

import random
import tensorflow as tf


def get_dataset(path, batch_size=64, num_test_set=100, crop_size=32):
    """
    Load dataset.

    Parameters
    ----------
    path : str
        Dataset path
    batch_size : int
        Batch size
    num_test_set : int
        Train/validation split
    crop_size : int
        Image cropping

    Returns
    -------
    tf.Dataset
        Train dataset
    tf.Dataset
        Validation dataset
    """
    def parse_example(example):
        return tf.io.parse_single_example(example, {
            "input": tf.io.FixedLenFeature([10], tf.string),
            "target": tf.io.FixedLenFeature([10], tf.string),
        })

    def parse_img(val):
        return {
            "input": tf.map_fn(tf.io.decode_image, val["input"],
                               fn_output_signature=tf.uint8),
            "target": tf.map_fn(tf.io.decode_image, val["target"],
                                fn_output_signature=tf.uint8),
        }

    def random_crop(val):
        inp_shape = tf.shape(val["input"])
        height = inp_shape[1]
        width = inp_shape[2]
        x0 = tf.random.uniform(
            shape=[], minval=0, maxval=width - crop_size, dtype=tf.int32)
        y0 = tf.random.uniform(
            shape=[], minval=0, maxval=height - crop_size, dtype=tf.int32)
        x1 = x0 + crop_size
        y1 = y0 + crop_size
        return {
            "input": val["input"][:, y0:y1, x0:x1, :],
            "target": val["target"][:, y0 * 2:y1 * 2, x0 * 2:x1 * 2, :],
        }

    def normalize(val):
        return {
            "input": tf.reshape(tf.cast(val["input"], tf.float32),
                                [10, crop_size, crop_size, 3]) / 255.0,
            "target": tf.reshape(tf.cast(val["target"], tf.float32),
                                 [10, crop_size*2, crop_size*2, 3]) / 255.0,
        }

    def filter_flat(val):
        d = val["input"]
        d = tf.math.reduce_std(d, axis=0)
        d = tf.math.reduce_sum(d) / crop_size / crop_size
        return d > 0.01

    records = tf.io.gfile.glob(path)
    random.shuffle(records)
    ds = tf.data.TFRecordDataset(
        records,
        num_parallel_reads=tf.data.experimental.AUTOTUNE
    )
    ds = ds.map(
        parse_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds = ds.map(
        parse_img,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    val_ds = ds.take(num_test_set)
    ds = ds.skip(num_test_set)
    ds = ds.repeat(20)
    ds = ds.shuffle(200)
    ds = ds.repeat()
    ds = ds.map(random_crop)
    ds = ds.map(normalize)
    ds = ds.filter(filter_flat)
    ds = ds.batch(batch_size, drop_remainder=True)
    val_ds = val_ds.repeat(10)
    val_ds = val_ds.map(random_crop)
    val_ds = val_ds.map(normalize)
    val_ds = val_ds.filter(filter_flat)
    val_ds = val_ds.batch(batch_size, drop_remainder=True)
    return ds, val_ds
