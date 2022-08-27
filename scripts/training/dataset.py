# -*- coding: utf-8 -*-

"""Dataset routines."""

import os
import random
from glob import glob
from typing import Any, Dict, Iterator, List, Tuple,  Union
import tensorflow as tf
import cv2
import numpy as np


class DatasetOp:
    """Dataset operation."""

    def __init__(self, name: str) -> None:
        """Create DatasetOp."""
        self.name = name

    def __call__(self, data: Any) -> Any:
        """Call dataset op."""
        raise NotImplementedError()


class GlobOp(DatasetOp):
    """Glob operation."""

    def __init__(self, name: str, glob_pattern: str) -> None:
        """Create GlobOp."""
        super().__init__(name)
        self.glob_pattern = glob_pattern

    def __call__(self, data: Any) -> Any:
        """Call operation."""
        assert data is None
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

    def __init__(self, name: str, path: Union[str, None] = None,
                 **kwargs) -> None:
        """Create TFRecordDatasetOp."""
        super().__init__(name)
        self.path = path
        self.kwargs = kwargs

    def __call__(self, data: Any) -> Any:
        """Call operation."""
        if self.path is not None:
            path = self.path
        elif data is not None:
            path = data
        else:
            raise ValueError("Dataset path is not defined")
        return tf.data.TFRecordDataset(path, name=self.name, **self.kwargs)


class LocalDatasetOp(DatasetOp):
    """Local dataset."""

    def __init__(self, name: str, hr_path: str, lr_path: str,
                 shuffle: bool = False) -> None:
        """Create LocalDatasetOp."""
        super().__init__(name)
        hr_files = list(sorted(glob(hr_path, recursive=True)))
        lr_files = list(sorted(glob(lr_path, recursive=True)))
        if len(lr_files) != len(hr_files) or len(hr_files) % 10 != 0:
            raise ValueError("Invalid number of images")
        hr_files = [os.path.abspath(x) for x in hr_files]
        lr_files = [os.path.abspath(x) for x in lr_files]
        frames = zip(lr_files, hr_files)
        self.batch = list(zip(*(iter(frames), ) * 10))
        if shuffle:
            random.shuffle(self.batch)

    def _generator(self) -> Iterator[np.ndarray]:
        """Iterate over dataset."""
        for batch in self.batch:
            data = [
                np.array([
                    cv2.imread(batch[j][i], cv2.IMREAD_COLOR)
                    for j in range(10)
                ])
                for i in range(2)
            ]
            yield {
                "input": data[0],
                "target": data[1],
            }

    def __call__(self, data: Any) -> Any:
        """Create dataset."""
        assert data is None
        return tf.data.TFRecordDataset.from_generator(
            self._generator,
            output_signature={
                "input": tf.TensorSpec(dtype=tf.uint8, shape=None),
                "target": tf.TensorSpec(dtype=tf.uint8, shape=None),
            },
            name=self.name,
        )


class MapOp(DatasetOp):
    """Dataset map operation."""

    def __init__(self, name: str, **kwargs) -> None:
        """Create MapOp."""
        super().__init__(name)
        self.map_kwargs = kwargs

    def map_fn(self, data: Any) -> Any:
        """Map data."""
        return data

    def __call__(self, data: Any) -> Any:
        """Call operation."""
        assert isinstance(data, tf.data.Dataset), "Type mismatch"
        return data.map(self.map_fn, **self.map_kwargs, name=self.name)


class FlatMapOp(MapOp):
    """Dataset flat map operation."""

    def __call__(self, data: Any) -> Any:
        """Call operation."""
        return super().__call__(data).unbatch(name=f"{self.name}_unbatch")


class FilterOp(DatasetOp):
    """Dataset filter operation."""

    def __init__(self, name: str, **kwargs) -> None:
        """Create FilterOp."""
        super().__init__(name)
        self.filter_kwargs = kwargs

    def filter_fn(self, data: Any) -> bool:
        """Filter data."""
        del data
        return True

    def __call__(self, data: Any) -> Any:
        """Call operation."""
        assert isinstance(data, tf.data.Dataset), "Type mismatch"
        return data.filter(self.filter_fn, **self.filter_kwargs,
                           name=self.name)


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


class RandomCropOp(FlatMapOp):
    """Random crop operation."""

    # pylint: disable=invalid-name

    def __init__(self, crop_size: int, num_img: int, **kwargs) -> None:
        """Create RandomCropOp."""
        super().__init__(**kwargs)
        self.crop_size = crop_size
        self.num_img = num_img

    def map_fn(self, data: Any) -> Any:
        """Crop images."""
        inp_shape = tf.shape(data["input"])
        height = inp_shape[1]
        width = inp_shape[2]
        inputs = []
        targets = []
        for _ in range(self.num_img):
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
            inputs.append(data["input"][:, y0:y1, x0:x1, :])
            targets.append(data["target"][:, y0*4:y1*4, x0*4:x1*4, :])
        return {
            "input": tf.stack(inputs, axis=0),
            "target": tf.stack(targets, axis=0),
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
        inp = tf.clip_by_value(inp, self.minval, self.maxval)
        target = tf.clip_by_value(target, self.minval, self.maxval)
        return {
            "input": inp,
            "target": target,
        }


class SingleFrameMapOp(FlatMapOp):
    """Convert dataset to single-frame."""

    def __init__(self, flow_frames: int, **kwargs) -> None:
        """Create SingleFrameMapOp."""
        super().__init__(**kwargs)
        self.flow_frames = flow_frames

    def map_fn(self, data: Any) -> Any:
        """Convert dataset to single-frame."""
        inputs = []
        targets = []
        last = []
        for idx in range(11 - self.flow_frames):
            inputs.append(data["input"][idx:idx+self.flow_frames])
            targets.append(data["target"][idx + (self.flow_frames - 1)])
            last.append(data["target"][idx + (self.flow_frames - 2)])
        return {
            "input": tf.stack(inputs, axis=0),
            "target": tf.stack(targets, axis=0),
            "last": tf.stack(last, axis=0),
        }


class SampleDatasetOp(DatasetOp):
    """Sample from datasets."""

    def __init__(self, name: str, configs: List[List[Dict[str, Any]]],
                 **kwargs) -> None:
        """Create SampleDatasetOp."""
        super().__init__(name)
        self.configs = configs
        self.kwargs = kwargs

    def __call__(self, data: Any) -> Any:
        """Sample from datasets."""
        assert data is None
        return tf.data.Dataset.sample_from_datasets(
            datasets=[create_dataset(config)
                      for config in self.configs],
            **self.kwargs,
        )


class BatchOp(DatasetOp):
    """Batch op."""

    def __init__(self, name: str, batch_size: int) -> None:
        """Create BatchOp."""
        super().__init__(name)
        self.batch_size = batch_size

    def __call__(self, data: Any) -> Any:
        """Batch data."""
        assert isinstance(data, tf.data.Dataset), "Type mismatch"
        return data.batch(self.batch_size, drop_remainder=True,
                          name=self.name)


class RepeatOp(DatasetOp):
    """Repeat op."""

    def __call__(self, data: Any) -> Any:
        """Repeat data."""
        assert isinstance(data, tf.data.Dataset), "Type mismatch"
        return data.repeat(name=self.name)


class ShuffleOp(DatasetOp):
    """Shuffle op."""

    def __init__(self, name: str, shuffle_window: int, **kwargs) -> None:
        """Create ShuffleOp."""
        super().__init__(name)
        self.shuffle_window = shuffle_window
        self.kwargs = kwargs

    def __call__(self, data: Any) -> Any:
        """Shuffle data."""
        assert isinstance(data, tf.data.Dataset), "Type mismatch"
        return data.shuffle(self.shuffle_window, **self.kwargs,
                            name=self.name)


class CacheOp(DatasetOp):
    """Cache op."""

    def __call__(self, data: Any) -> Any:
        """Cache data."""
        assert isinstance(data, tf.data.Dataset), "Type mismatch"
        return data.cache(name=self.name)


class PrefetchOp(DatasetOp):
    """Prefetch op."""

    def __init__(self, name: str, buffer_size: int) -> None:
        """Create PrefetchOp."""
        super().__init__(name)
        self.buffer_size = buffer_size

    def __call__(self, data: Any) -> Any:
        """Prefetch data."""
        assert isinstance(data, tf.data.Dataset), "Type mismatch"
        return data.prefetch(self.buffer_size, name=self.name)


class TakeOp(DatasetOp):
    """Take op."""

    def __init__(self, name: str, size: int) -> None:
        """Create TakeOp."""
        super().__init__(name)
        self.size = size

    def __call__(self, data: Any) -> Any:
        """Take data."""
        assert isinstance(data, tf.data.Dataset), "Type mismatch"
        return data.take(self.size, name=self.name)


class OptionsOp(DatasetOp):
    """Options op."""

    def __init__(self, name: str, options: Dict[str, Any]) -> None:
        """Create OptionsOp."""
        super().__init__(name)

        def set_opt(obj: Any, opt: Dict[str, Any]) -> None:
            for key, val in opt.items():
                if isinstance(val, dict):
                    set_opt(getattr(obj, key), val)
                else:
                    setattr(obj, key, val)

        self.options = tf.data.Options()
        set_opt(self.options, options)

    def __call__(self, data: Any) -> Any:
        """Call operation."""
        assert isinstance(data, tf.data.Dataset), "Type mismatch"
        return data.with_options(self.options, name=self.name)


DATASET_OPS = {
    "GlobOp": GlobOp,
    "ListShuffleOp": ListShuffleOp,
    "TFRecordDatasetOp": TFRecordDatasetOp,
    "LocalDatasetOp": LocalDatasetOp,
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
    "OptionsOp": OptionsOp,
}


def create_dataset(config: List[Dict[str, Any]]) -> tf.data.Dataset:
    """Create dataset from config."""
    dataset = None
    for op_config in config:
        if "name" not in op_config:
            raise ValueError("Op name is not defined")
        name = op_config["name"]
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
    # Fill up cache for play and val datasets
    for _ in val_ds:
        pass
    for _ in play_ds:
        pass
    return (val_ds, play_ds)


__all__ = ["DATASET_OPS", "create_dataset", "create_train_dataset",
           "create_val_dataset"] + list(DATASET_OPS)
