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
    parser.add_argument("-w", "--whole",
                        help="Process whole data in one step",
                        dest="process_whole",
                        action="store_true",
                        default=False)
    parser.add_argument("-s", "--steps",
                        help="Max number of steps",
                        dest="max_steps",
                        type=int,
                        default=None)
    parser.add_argument("-m", "--method",
                        help="Calibration method (default: %(default)s)",
                        dest="calibration_method",
                        type=CalibrationMethod.argparse_value,
                        choices=list(CalibrationMethod),
                        default=CalibrationMethod.ENTROPY)
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error(f"Invalid steps argument: {args.max_steps}")
    return args


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
                 num_steps: Union[None, int] = None,
                 **kwargs) -> None:
        """Create DataReader."""
        super().__init__(**kwargs)
        self.hr_data_iter = iter(hr_data)
        self.lr_data_iter = iter(lr_data)
        self.names = [i.name for i in model.graph.input]
        self.dtypes = {
            i.name: DataReader.ONNX_TYPES[i.type.tensor_type.elem_type]
            for i in model.graph.input
        }
        self.states = {}
        last_gen = None
        for name in self.names[-1:1:-1]:
            self.states[name] = DataReader._normalize(
                next(self.lr_data_iter),
                self.dtypes[name]
            )
        last_gen = next(self.hr_data_iter)
        if len(self.names) > 1:
            self.states[self.names[1]] = DataReader._normalize(
                last_gen,
                self.dtypes[self.names[1]]
            )
        if num_steps is not None:
            self.step_iter = iter(range(num_steps))
        else:
            self.step_iter = None

    @staticmethod
    def _normalize(img: np.ndarray, dtype: np.dtype) -> np.ndarray:
        """Normalize data."""
        return np.transpose(img.astype(dtype) / 255, [0, 3, 1, 2])

    def get_next(self) -> Union[None, Dict[str, np.ndarray]]:
        """Iterate over data."""
        if self.step_iter is not None:
            i = next(self.step_iter, None)
            if i is None:
                return None
        last_gen = next(self.hr_data_iter, None)
        if last_gen is None:
            return None
        cur_frame = next(self.lr_data_iter)
        value = {
            self.names[0]: cur_frame,
            **self.states
        }
        for old_name, new_name in zip(self.names[-2:1:-1],
                                      self.names[-1:2:-1]):
            self.states[new_name] = self.states[old_name]
        if len(self.names) > 1:
            self.states[self.names[1]] = DataReader._normalize(
                last_gen,
                self.dtypes[self.names[1]]
            )
        if len(self.names) > 2:
            self.states[self.names[2]] = DataReader._normalize(
                cur_frame,
                self.dtypes[self.names[2]]
            )
        return value


def main(
    model_path: str,
    lowres_path: str,
    hires_path: str,
    output_path: str,
    process_whole: bool = False,
    max_steps: Union[None, int] = None,
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
    process_whole: bool
        Process whole data in one ste
    max_steps: int
        Max number of steps
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
    num_inputs = len(model.graph.input)
    if num_inputs > 1:
        num_images -= num_inputs - 2
    if max_steps is not None:
        num_images = min(num_images, max_steps)
    if process_whole:
        lr_data = lr_data[:num_images + num_inputs - 2]
        hr_data = hr_data[:num_images + num_inputs - 2]
    with tempfile.NamedTemporaryFile() as tmp:
        calibrator = quantization.create_calibrator(
            model,
            calibrate_method=calibration_method.value,
            augmented_model_path=tmp.name,
        )
        if process_whole:
            LOG.info("Calibration")
            data_reader = DataReader(model, hr_data, lr_data)
            calibrator.collect_data(data_reader)
        else:
            for i in range(num_images):
                LOG.info("Calibration %d / %d", i + 1, num_images)
                # Collecting data for one sample at a time due to possible OOM
                data_reader = DataReader(
                    model,
                    hr_data[i:],
                    lr_data[i:],
                    num_steps=1
                )
                calibrator.collect_data(data_reader)
        LOG.info("Computing")
        calibration = calibrator.compute_range()
    with open(output_path, "wt", encoding="utf-8") as f:
        json.dump(calibration, f)
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
