#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate calibration data."""

import sys
import argparse
import importlib
import json
import tempfile
from typing import Union, Dict
import numpy as np
import onnx
from onnxruntime import quantization
from onnxruntime.quantization.calibrate import TensorsData
from graph import Graph
from utils import simplify_model


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Quantize model to int8")
    parser.add_argument("model_path",
                        help="Model",
                        type=str)
    parser.add_argument("output_path",
                        help="Output path",
                        type=str)
    parser.add_argument("calibration_path",
                        help="Calibration data path",
                        type=str)
    parser.add_argument("--num-checks",
                        help="Number of simplifier checks",
                        type=int,
                        default=3)
    return parser.parse_args()


class DataReader(quantization.CalibrationDataReader):
    """Data reade."""

    ONNX_TYPES = {
        onnx.TensorProto.FLOAT: np.float32,
        onnx.TensorProto.FLOAT16: np.float16,
        onnx.TensorProto.UINT8: np.uint8,
    }

    def __init__(self,
                 calibration_path: str,
                 model: onnx.ModelProto,
                 **kwargs) -> None:
        """Create DataReader."""
        super().__init__(**kwargs)
        with open(calibration_path, "rt", encoding="utf-8") as f:
            calibration_dict = json.load(f)
        value_types = {
            v.name: DataReader.ONNX_TYPES[v.type.tensor_type.elem_type]
            for v in model.graph.value_info
        }
        for inp in model.graph.input:
            value_types[inp.name] = \
                DataReader.ONNX_TYPES[inp.type.tensor_type.elem_type]
        for out in model.graph.output:
            value_types[out.name] = \
                DataReader.ONNX_TYPES[out.type.tensor_type.elem_type]
        calibration_dict = {
            k: np.abs(np.array(v, dtype=value_types[k])).max()
            for k, v in calibration_dict.items()
            if k in value_types
        }
        self.data_iter = iter([calibration_dict])

    def get_next(self) -> Union[None, Dict[str, np.ndarray]]:
        """Iterate over data."""
        return next(self.data_iter, None)

    def __len__(self) -> int:
        raise NotImplementedError()

    def set_range(self, _start_index: int, _end_index: int) -> None:
        raise NotImplementedError()


class StaticCalibrator(quantization.CalibraterBase):
    """Static calibrator."""

    def __init__(self, *args, **kwargs):
        """Create StaticCalibrator."""
        super().__init__(*args, **kwargs)
        self.data = None

    def augment_graph(self):
        """Augment graph."""
        # no-op

    def collect_data(self, data_reader: quantization.CalibrationDataReader):
        """Read data from data reader."""
        if self.data is not None:
            raise ValueError()
        data = data_reader.get_next()
        if data is None:
            raise ValueError()
        self.data = data

    def compute_data(self) -> TensorsData:
        """Compute final data."""
        if self.data is None:
            raise ValueError()
        value_dict = {
            k: (-v, v)
            for k, v in self.data.items()
        }
        return TensorsData(quantization.CalibrationMethod.MinMax, value_dict)


def patch_onnxruntime() -> None:
    """Patch onnxruntime to run on calibration GPU"""
    qmod = importlib.import_module("onnxruntime.quantization.quantize")
    orig_create_calibrator = qmod.create_calibrator

    def create_calibrator(model, *args, **kwargs):
        if "calibrate_method" in kwargs and kwargs["calibrate_method"] is None:
            return StaticCalibrator(model)
        return orig_create_calibrator(model, *args, **kwargs)
    qmod.create_calibrator = create_calibrator


ADD_NODE = "final_1/full_1/generator_1/add_1/Add"


def main(
    model_path: str,
    calibration_path: str,
    output_path: str,
    num_checks: int,
) -> int:
    """
    Run CLI.

    Parameters
    ----------
    model_path: str
        Path to model
    calibration_path: str
        Calibration data path
    output_path: str
        Output pth
    num_checks: int
        Number of simplifier checks

    Returns
    -------
    int
        Exit code
    """
    patch_onnxruntime()
    model = onnx.load(model_path)
    graph = Graph(model)
    add_node = graph.find_node_by_name(ADD_NODE)
    resize_node = graph.find_node_by_output(add_node.input[0]).name
    data_reader = DataReader(calibration_path, model)
    with tempfile.NamedTemporaryFile() as tmp:
        quantization.quantize_static(
            model_input=model_path,
            model_output=tmp.name,
            calibration_data_reader=data_reader,
            quant_format=quantization.QuantFormat.QDQ,
            op_types_to_quantize=["Conv", "ConvTranspose", "Resize",
                                  "MaxPool", "LeakyRelu", "Relu", "Clip"],
            per_channel=True,
            reduce_range=False,
            activation_type=quantization.QuantType.QInt8,
            weight_type=quantization.QuantType.QInt8,
            nodes_to_exclude=["output_diff_mean", "scene_cond_s", resize_node],
            calibrate_method=None,
            extra_options={
                "AddQDQPairToWeight": True,
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
                "QuantizeBias": False,
            }
        )
        model = onnx.load(tmp.name)
    graph = Graph(model)
    for op_type in ["Conv"]:
        for chain in graph.find_node_chain([
            "DequantizeLinear",
            "QuantizeLinear",
            op_type
        ]):
            in_nodes = graph.find_nodes_by_input(chain[0].output[0])
            if len(in_nodes) != 1 \
                or in_nodes[0].op_type not in ["Clip", "Relu", "LeakyRelu",
                                               "Add"]:
                continue
            graph.remove_node(chain[0])
            graph.remove_node(chain[1])
            graph.remove_node(chain[2])
            chain[2].output[0] = chain[0].output[0]
            graph.insert_node(chain[2])
    model = graph.serialize()
    model = simplify_model(model, num_checks=num_checks)
    onnx.save(model, output_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(**vars(parse_args())))
    except KeyboardInterrupt:
        sys.exit(-1)
