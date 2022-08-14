# -*- coding: utf-8 -*-

"""Dataset routines."""

import random
from typing import Any, Dict, List, Tuple,  Union
import tensorflow as tf


class DatasetOp:
    """Dataset operation."""

    def __call__(self, data: Any) -> Any:
        """Call dataset op."""
        del data
        raise NotImplementedError()


class GlobOp(DatasetOp):
    """Glob operation."""

    def __init__(self, glob_pattern: str) -> None:
        """Create GlobOp."""
        super().__init__()
        self.glob_pattern = glob_pattern

    def __call__(self, data: Any) -> Any:
        """Call operation."""
        del data
        return tf.io.gfile.glob(self.glob_pattern)


class ListShuffleOp(DatasetOp):
    """List shuffle operation."""

    def __call__(self, data: Any) -> Any:
        """Call operation."""
        new_list = list(data)
        random.shuffle(new_list)
        return new_list


class TFRecordDatasetOp(DatasetOp):
    """Dataset from TFRecord."""

    def __init__(self, path: Union[str, None] = None, **kwargs) -> None:
        """Create TFRecordDatasetOp."""
        super().__init__()
        self.path = path
        self.kwargs = kwargs

    def __call__(self, data: Any) -> Any:
        """Call operation."""
        if self.path is not None:
            path = self.path
        elif data is not None:
            path = data
        else:
            raise RuntimeError("Dataset path is not defined")
        return tf.data.TFRecordDataset(path, **self.kwargs)


class MapOp(DatasetOp):
    """Dataset map operation."""

    def __init__(self, **kwargs) -> None:
        """Create MapOp."""
        super().__init__()
        self.map_kwargs = kwargs

    def map_fn(self, data: Any) -> Any:
        """Map data."""
        return data

    def __call__(self, data: Any) -> Any:
        """Call operation."""
        assert isinstance(data, tf.data.Dataset), "Type mismatch"
        return data.map(self.map_fn, **self.map_kwargs)


class FilterOp(DatasetOp):
    """Dataset filter operation."""

    def __init__(self, **kwargs) -> None:
        """Create FilterOp."""
        super().__init__()
        self.filter_kwargs = kwargs

    def filter_fn(self, data: Any) -> bool:
        """Filter data."""
        del data
        return True

    def __call__(self, data: Any) -> Any:
        """Call operation."""
        assert isinstance(data, tf.data.Dataset), "Type mismatch"
        return data.filter(self.filter_fn, **self.filter_kwargs)


class RandomCondMapOp(MapOp):
    """Random map operation."""

    def __init__(self, threshold: float, **kwargs) -> None:
        """Create RandomCondMapOp."""
        super().__init__(**kwargs)
        self.threshold = threshold

    def true_fn(self, data: Any) -> Any:
        """Run if true."""
        return data

    def false_fn(self, data: Any) -> Any:
        """Run if false."""
        return data

    def map_fn(self, data: Any) -> Any:
        """Call dataset op."""
        rand = tf.random.uniform([])
        return tf.cond(
            rand < self.threshold,
            lambda: self.true_fn(data),
            lambda: self.false_fn(data),
        )


class ParsePairExampleOp(MapOp):
    """Example parsing op."""

    def map_fn(self, data: Any) -> Any:
        """Convert encoded proto to tensors."""
        parsed = tf.io.parse_single_example(data, {
            "input": tf.io.FixedLenFeature([10], tf.string),
            "target": tf.io.FixedLenFeature([10], tf.string),
        })
        return {
            "input": tf.map_fn(tf.io.decode_image, parsed["input"],
                               fn_output_signature=tf.uint8),
            "target": tf.map_fn(tf.io.decode_image, parsed["target"],
                                fn_output_signature=tf.uint8),
        }


class ParseSingleExampleOp(MapOp):
    """Example parsing op."""

    def map_fn(self, data: Any) -> Any:
        """Convert encoded proto to tensors."""
        parsed = tf.io.parse_single_example(data, {
            "images": tf.io.FixedLenFeature([10], tf.string),
        })["images"]
        images = tf.map_fn(tf.io.decode_image, parsed,
                           fn_output_signature=tf.uint8)
        shape = tf.shape(images)
        inputs = tf.compat.v1.image.resize_nearest_neighbor(
            images=images,
            size=shape[1:3] // 4,
            align_corners=False,
            half_pixel_centers=False,
        )
        return {
            "input": inputs,
            "target": images,
        }


class RandomCropOp(MapOp):
    """Random crop operation."""

    # pylint: disable=invalid-name

    def __init__(self, crop_size: int, **kwargs) -> None:
        """Create RandomCropOp."""
        super().__init__(**kwargs)
        self.crop_size = crop_size

    def map_fn(self, data: Any) -> Any:
        """Crop images."""
        inp_shape = tf.shape(data["input"])
        height = inp_shape[1]
        width = inp_shape[2]
        x0 = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=width - self.crop_size,
            dtype=tf.int32
        )
        y0 = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=height - self.crop_size,
            dtype=tf.int32
        )
        x1 = x0 + self.crop_size
        y1 = y0 + self.crop_size
        return {
            "input": data["input"][:, y0:y1, x0:x1, :],
            "target": data["target"][:, y0 * 4:y1 * 4, x0 * 4:x1 * 4, :],
        }


class NormalizeOp(MapOp):
    """Normalization op."""

    def __init__(self, crop_size: int, **kwargs) -> None:
        """Create RandomCropOp."""
        super().__init__(**kwargs)
        self.crop_size = crop_size

    def map_fn(self, data: Any) -> Any:
        """Normalize data."""
        return {
            "input": tf.reshape(
                tf.cast(data["input"], tf.float32),
                [10, self.crop_size, self.crop_size, 3]
            ) / 255.0,
            "target": tf.reshape(
                tf.cast(data["target"], tf.float32),
                [10, self.crop_size*4, self.crop_size*4, 3]
            ) / 255.0,
        }


class FilterFlatOp(FilterOp):
    """Filter out flat images."""

    def __init__(self, threshold: float, **kwargs) -> None:
        """Create FilterFlatOp."""
        super().__init__(**kwargs)
        self.threshold = threshold

    def filter_fn(self, data: Any) -> bool:
        """Filter out flat images."""
        val = data["input"]
        assert isinstance(val, tf.Tensor), "Type mismatch"
        # Calculate average Deviation across sequence
        val = tf.math.reduce_std(val, axis=0)
        val = tf.math.reduce_sum(val, axis=-1)
        val = tf.math.reduce_mean(val)
        return val > self.threshold


class RgbToBgrOp(MapOp):
    """Convert RGB to BGR."""

    def map_fn(self, data: Any) -> Any:
        """Convert RGB to BGR."""
        return {
            "input": data["input"][:, :, :, ::-1],
            "target": data["target"][:, :, :, ::-1],
        }


class RandomNoiseOp(MapOp):
    """Add random noise."""

    def __init__(self, stddev: float, **kwargs) -> None:
        """Create RandomNoiseOp."""
        super().__init__(**kwargs)
        self.stddev = stddev

    def map_fn(self, data: Any) -> Any:
        """Add random noise."""
        inp = data["input"]
        target = data["target"]
        inp = inp + tf.random.normal(shape=tf.shape(inp), stddev=self.stddev)
        return {
            "input": inp,
            "target": target,
        }


class RandomContrastOp(MapOp):
    """Random contrast op."""

    def __init__(self, stddev: float, base: float, **kwargs) -> None:
        """Create RandomContrastOp."""
        super().__init__(**kwargs)
        self.stddev = stddev
        self.base = base

    def map_fn(self, data: Any) -> Any:
        """Augment contrast."""
        inp = data["input"]
        target = data["target"]
        rate = tf.pow(float(self.base),
                      tf.random.normal(shape=[1], stddev=self.stddev))[0]
        mean = tf.math.reduce_mean(target, axis=[0, 1, 2])
        inp = (inp - mean) * rate + mean
        target = (target - mean) * rate + mean
        return {
            "input": inp,
            "target": target,
        }


class RandomBrightnessOp(MapOp):
    """Random brightness op."""

    def __init__(self, stddev: float, **kwargs) -> None:
        """Create RandomBrightnessOp."""
        super().__init__(**kwargs)
        self.stddev = stddev

    def map_fn(self, data: Any) -> Any:
        """Augment brightness."""
        inp = data["input"]
        target = data["target"]
        rate = tf.random.normal(shape=[1], stddev=self.stddev)
        inp = tf.image.adjust_brightness(inp, rate)
        target = tf.image.adjust_brightness(target, rate)
        return {
            "input": inp,
            "target": target,
        }


class RandomHorizontalFlipOp(RandomCondMapOp):
    """Random horizontal flip."""

    def true_fn(self, data: Any) -> Any:
        """Flip horizontally."""
        inp = data["input"]
        target = data["target"]
        inp = inp[:, :, ::-1, :]
        target = target[:, :, ::-1, :]
        return {
            "input": inp,
            "target": target,
        }


class RandomVerticalFlipOp(RandomCondMapOp):
    """Random vertical flip."""

    def true_fn(self, data: Any) -> Any:
        """Flip vertically."""
        inp = data["input"]
        target = data["target"]
        inp = inp[:, ::-1, :, :]
        target = target[:, ::-1, :, :]
        return {
            "input": inp,
            "target": target,
        }


class RandomTransposeOp(RandomCondMapOp):
    """Random transposition."""

    def true_fn(self, data: Any) -> Any:
        """Transpose data."""
        inp = data["input"]
        target = data["target"]
        inp = tf.transpose(inp, [0, 2, 1, 3])
        target = tf.transpose(target, [0, 2, 1, 3])
        return {
            "input": inp,
            "target": target,
        }


class ClipOp(MapOp):
    """Clip values."""

    def __init__(self, minval: float, maxval: float, **kwargs) -> None:
        """Create ClipOp."""
        super().__init__(**kwargs)
        self.minval = minval
        self.maxval = maxval

    def map_fn(self, data: Any) -> Any:
        """Clip values."""
        inp = data["input"]
        target = data["target"]
        inp = tf.minimum(inp, self.maxval)
        inp = tf.maximum(inp, self.minval)
        target = tf.minimum(target, self.maxval)
        target = tf.maximum(target, self.minval)
        return {
            "input": inp,
            "target": target,
        }


class SingleFrameMapOp(MapOp):
    """Convert dataset to single-frame."""

    def __init__(self, flow_frames: int, **kwargs) -> None:
        """Create SingleFrameMapOp."""
        super().__init__(**kwargs)
        self.flow_frames = flow_frames

    def map_fn(self, data: Any) -> Any:
        """Convert dataset to single-frame."""
        inp = data["inp"]
        target = data["target"]
        idx = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=10 - self.flow_frames,
            dtype=tf.int32
        )
        inp = inp[idx:idx+self.flow_frames]
        target = target[idx + (self.flow_frames - 1)]
        last = target[idx + (self.flow_frames - 2)]
        return {
            "input": inp,
            "target": target,
            "last": last,
        }


class SampleDatasetOp(DatasetOp):
    """Sample from datasets."""

    def __init__(self, configs: List[List[Dict[str, Any]]],
                 **kwargs) -> None:
        """Create SampleDatasetOp."""
        super().__init__()
        self.configs = configs
        self.kwargs = kwargs

    def __call__(self, data: Any) -> Any:
        """Sample from datasets."""
        del data
        return tf.data.Dataset.sample_from_datasets(
            datasets=[create_dataset(config)
                      for config in self.configs],
            **self.kwargs
        )


class BatchOp(DatasetOp):
    """Batch op."""

    def __init__(self, batch_size: int) -> None:
        """Create BatchOp."""
        super().__init__()
        self.batch_size = batch_size

    def __call__(self, data: Any) -> Any:
        """Batch data."""
        assert isinstance(data, tf.data.Dataset), "Type mismatch"
        return data.batch(self.batch_size, drop_remainder=True)


class RepeatOp(DatasetOp):
    """Repeat op."""

    def __call__(self, data: Any) -> Any:
        """Repeat data."""
        assert isinstance(data, tf.data.Dataset), "Type mismatch"
        return data.repeat()


class ShuffleOp(DatasetOp):
    """Shuffle op."""

    def __init__(self, shuffle_window: int, **kwargs) -> None:
        """Create ShuffleOp."""
        super().__init__()
        self.shuffle_window = shuffle_window
        self.kwargs = kwargs

    def __call__(self, data: Any) -> Any:
        """Shuffle data."""
        assert isinstance(data, tf.data.Dataset), "Type mismatch"
        return data.shuffle(self.shuffle_window, **self.kwargs)


class CacheOp(DatasetOp):
    """Cache op."""

    def __call__(self, data: Any) -> Any:
        """Cache data."""
        assert isinstance(data, tf.data.Dataset), "Type mismatch"
        return data.cache()


class PrefetchOp(DatasetOp):
    """Prefetch op."""

    def __init__(self, buffer_size: int) -> None:
        """Create PrefetchOp."""
        super().__init__()
        self.buffer_size = buffer_size

    def __call__(self, data: Any) -> Any:
        """Prefetch data."""
        assert isinstance(data, tf.data.Dataset), "Type mismatch"
        return data.prefetch(self.buffer_size)


class TakeOp(DatasetOp):
    """Take op."""

    def __init__(self, size: int) -> None:
        """Create TakeOp."""
        super().__init__()
        self.size = size

    def __call__(self, data: Any) -> Any:
        """Take data."""
        assert isinstance(data, tf.data.Dataset), "Type mismatch"
        return data.take(self.size)


DATASET_OPS = {
    "GlobOp": GlobOp,
    "ListShuffleOp": ListShuffleOp,
    "TFRecordDatasetOp": TFRecordDatasetOp,
    "ParsePairExampleOp": ParsePairExampleOp,
    "ParseSingleExampleOp": ParseSingleExampleOp,
    "RandomCropOp": RandomCropOp,
    "NormalizeOp": NormalizeOp,
    "FilterFlatOp": FilterFlatOp,
    "RgbToBgrOp": RgbToBgrOp,
    "RandomNoiseOp": RandomNoiseOp,
    "RandomContrastOp": RandomContrastOp,
    "RandomBrightnessOp": RandomBrightnessOp,
    "RandomHorizontalFlipOp": RandomHorizontalFlipOp,
    "RandomVerticalFlipOp": RandomVerticalFlipOp,
    "RandomTransposeOp": RandomTransposeOp,
    "ClipOp": ClipOp,
    "SampleDatasetOp": SampleDatasetOp,
    "SingleFrameMapOp": SingleFrameMapOp,
    "BatchOp": BatchOp,
    "RepeatOp": RepeatOp,
    "ShuffleOp": ShuffleOp,
    "CacheOp": CacheOp,
    "PrefetchOp": PrefetchOp,
    "TakeOp": TakeOp,
}


def create_dataset(config: List[Dict[str, Any]]) -> tf.data.Dataset:
    """Create dataset from config."""
    dataset = None
    for op_config in config:
        if "name" not in op_config:
            raise ValueError("Op name is not defined")
        name = op_config["name"]
        del op_config["name"]
        if name not in DATASET_OPS:
            raise ValueError(f"Unknown dataset op: {name}")
        dataset_op = DATASET_OPS[name](**op_config)
        dataset = dataset_op(dataset)
    if not isinstance(dataset, tf.data.Dataset):
        raise ValueError("Invalid dataset config")
    return dataset


def create_train_dataset(config: List[Dict[str, Any]],
                         batch_size: int) -> tf.data.Dataset:
    """Create training dataset."""
    return create_dataset(config + [
        {"name": "BatchOp", "batch_size": batch_size},
        {"name": "PrefetchOp", "buffer_size": -1},
    ])


def create_val_dataset(config: List[Dict[str, Any]], batch_size: int,
                       play_size: int, val_size: int) \
        -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Create validation dataset."""
    val_ds = create_dataset(config + [
        {"name": "TakeOp", "size": val_size},
        {"name": "BatchOp", "batch_size": batch_size},
        {"name": "CacheOp"},
    ])
    play_ds = create_dataset(config + [
        {"name": "TakeOp", "size": play_size},
        {"name": "BatchOp", "batch_size": play_size},
        {"name": "CacheOp"},
    ])
    list(play_ds.as_numpy_iterator())
    return (val_ds, play_ds)


__all__ = ["DATASET_OPS", "create_dataset", "create_train_dataset",
           "create_val_dataset"] + list(DATASET_OPS)
