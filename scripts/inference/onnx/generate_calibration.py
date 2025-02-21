#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate calibration data."""

import os
import sys
import logging
import argparse
import json
import tempfile
import enum
from typing import Dict, List, Union
import cv2
import numpy as np
import onnx
from onnxruntime import quantization


class CalibrationMethod(enum.Enum):
    """Calibration method."""

    ENTROPY = quantization.CalibrationMethod.Entropy
    MIN_MAX = quantization.CalibrationMethod.MinMax
    PERCENTILE = quantization.CalibrationMethod.Percentile

    def __str__(self) -> str:
        """Convert to str."""
        return self.name

    @staticmethod
    def argparse_value(val: str) -> "CalibrationMethod":
        """Create from string."""
        try:
            return CalibrationMethod[val.upper()]
        except KeyError as exc:
            raise argparse.ArgumentTypeError(f"Invalid value: {val}") from exc


LOG = logging.getLogger("generate_calibration")


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate calibration data")
    parser.add_argument("model_path",
                        help="Model",
                        type=str)
    parser.add_argument("lowres_path",
                        help="Low-res input images",
                        type=str)
    parser.add_argument("hires_path",
                        help="Hi-res input images",
                        type=str)
    parser.add_argument("output_path",
                        help="Output path",
                        type=str)
    parser.add_argument("-s", "--stride",
                        help="Step stride",
                        type=int,
                        default=None)
    parser.add_argument("-n", "--normalize-brightness",
                        help="Normalize brightness",
                        default=False,
                        action="store_true")
    parser.add_argument("-m", "--method",
                        help="Calibration method (default: %(default)s)",
                        dest="calibration_method",
                        type=CalibrationMethod.argparse_value,
                        choices=list(CalibrationMethod),
                        default=CalibrationMethod.ENTROPY)
    return parser.parse_args()


def read_data(path: str) -> List[np.ndarray]:
    """Read images from path."""
    imgs = []
    for img_name in sorted(os.listdir(path)):
        img = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_COLOR)
        assert img is not None, f"Could not read image: {img_name}"
        imgs.append(np.expand_dims(img, axis=0))
    return imgs


class DataReader(quantization.CalibrationDataReader):
    """Data reader for ONNX model calibration."""

    ONNX_TYPES = {
        onnx.TensorProto.FLOAT: np.float32,
        onnx.TensorProto.FLOAT16: np.float16,
        onnx.TensorProto.UINT8: np.uint8,
    }

    def __init__(self,
                 model: onnx.ModelProto,
                 hr_data: List[np.ndarray],
                 lr_data: List[np.ndarray],
                 normalize_brightness: bool = False,
                 **kwargs) -> None:
        """Create DataReader."""
        super().__init__(**kwargs)
        self.hr_data = hr_data
        self.lr_data = lr_data
        self.normalize_brightness = normalize_brightness
        self.names = [i.name for i in model.graph.input]
        self.dtypes = {
            i.name: DataReader.ONNX_TYPES[i.type.tensor_type.elem_type]
            for i in model.graph.input
        }
        self.pad_frame_shape = [
            d.dim_value
            for d in model.graph.input[2].type.tensor_type.shape.dim
        ]
        self.range_iter = None
        self.set_range(0, len(self))

    def _normalize(self, img: np.ndarray, dtype: np.dtype) -> np.ndarray:
        """Normalize data."""
        img = np.transpose(img / 255 - 0.5, [0, 3, 1, 2]).astype(dtype)
        if self.normalize_brightness:
            img -= (img * np.array([0.1140, 0.5870, 0.2989]
                                   ).reshape((1, 3, 1, 1))).mean() * 3
        return img

    def _normalize_and_pad(self, img: np.ndarray,
                           dtype: np.dtype) -> np.ndarray:
        """Normalize and pad data."""
        data = self._normalize(img, dtype)
        data = np.pad(
            data,
            [
                [b - a, 0]
                for a, b in zip(data.shape, self.pad_frame_shape)
            ]
        )
        return data

    def get_next(self) -> Union[None, Dict[str, np.ndarray]]:
        """Iterate over data."""
        idx = next(self.range_iter, None)
        if idx is None:
            return None
        LOG.info("Processing %d / %d", idx + 1, len(self))
        values = {}
        num_last_frames = len(self.names) - 2
        for i, name in enumerate(self.names[-1:1:-1]):
            values[name] = self._normalize_and_pad(
                self.lr_data[idx + i],
                self.dtypes[name]
            )
        values[self.names[1]] = self._normalize(
            self.hr_data[idx + num_last_frames - 1],
            self.dtypes[self.names[1]]
        )
        values[self.names[0]] = self.lr_data[idx + num_last_frames] \
            .astype(self.dtypes[self.names[0]])
        return values

    def __len__(self) -> int:
        return len(self.hr_data) - len(self.names) + 2

    def set_range(self, start_index: int, end_index: int) -> None:
        """Set DatReader range."""
        self.range_iter = iter(range(start_index, min(end_index, len(self))))


def main(
    model_path: str,
    lowres_path: str,
    hires_path: str,
    output_path: str,
    stride: Union[None, int] = None,
    normalize_brightness: bool = False,
    calibration_method: CalibrationMethod = CalibrationMethod.ENTROPY,
) -> int:
    """
    Run CLI.

    Parameters
    ----------
    model_path: str
        Path to model
    lowres_path: str
        Path to low-res input images
    hires_path: str
        Path to hi-res input images
    output_path: str
        Output path
    stride: int
        Step stride
    normalize_brightness: bool
        Normalize brightness
    calibration_method: CalibrationMethod
        Calibration method

    Returns
    -------
    int
        Exit code
    """
    model = onnx.load(model_path)
    LOG.info("Reading data")
    lr_data = read_data(lowres_path)
    hr_data = read_data(hires_path)
    num_images = len(hr_data)
    assert len(lr_data) == num_images, "Numbers of frames do not match"
    data_reader = DataReader(model, hr_data, lr_data, normalize_brightness)
    with tempfile.NamedTemporaryFile() as tmp:
        calibrator = quantization.create_calibrator(
            model_path,
            calibrate_method=calibration_method.value,
            augmented_model_path=tmp.name,
        )
        if stride is None:
            calibrator.collect_data(data_reader)
        else:
            for i in range(0, len(data_reader), stride):
                data_reader.set_range(i, i + stride)
                calibrator.collect_data(data_reader)
        LOG.info("Computing")
        calibration = calibrator.compute_data()
    with open(output_path, "wt", encoding="utf-8") as f:
        json.dump({k: (float(td.range_value[0][0]),
                       float(td.range_value[1][0]))
                   for k, td in calibration.items()}, f)
    return 0


if __name__ == "__main__":
    try:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] %(name)s: %(message)s",
            stream=sys.stderr,
        )
        sys.exit(main(**vars(parse_args())))
    except KeyboardInterrupt:
        sys.exit(-1)
    finally:
        logging.shutdown()
