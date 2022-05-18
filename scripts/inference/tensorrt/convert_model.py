#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert ONNX model to TensorRT graph."""

import os
import sys
import logging
import argparse
import json
import struct
from copy import deepcopy
import gzip
from typing import Any, Dict, List, Sequence, Tuple, Union
import yaml
import tensorrt as trt
import numpy as np

LOG = logging.getLogger("convert_model")


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Convert ONNX model to TensorRT graph")
    parser.add_argument("-c", "--calibration",
                        help="Calibration data",
                        type=str,
                        default=None)
    parser.add_argument("model",
                        help="Model",
                        type=str)
    parser.add_argument("output_path",
                        help="Output path",
                        type=str)
    return parser.parse_args()


def convert_to_list(val: Any) -> Union[None, List[Any]]:
    """Convert value to list."""
    try:
        return list(val)
    except ValueError:
        return None


class GraphSerializer:
    """TensorRT Graph."""

    def __init__(self) -> None:
        """Create graph."""
        self.weights = []
        self.calibration = {}

    def _append_weights(self, val: Union[np.ndarray, trt.Weights]) -> int:
        """Append weights to list."""
        idx = len(self.weights)
        if isinstance(val, trt.Weights):
            val = val.numpy()
        self.weights.append(val)
        return idx

    def _get_activation_params(self, layer: trt.ILayer) -> Dict[str, Any]:
        layer.__class__ = trt.IActivationLayer
        params = {
            "activation_type": layer.type.name,
            "alpha": layer.alpha,
            "beta": layer.beta,
        }
        return params

    def _get_concat_params(self, layer: trt.ILayer) -> Dict[str, Any]:
        layer.__class__ = trt.IConcatenationLayer
        return {
            "axis": layer.axis,
        }

    def _get_constant_params(self, layer: trt.ILayer) -> Dict[str, Any]:
        layer.__class__ = trt.IConstantLayer
        weights = layer.weights
        if isinstance(weights, trt.Weights):
            weights = weights.numpy()
        output_name = layer.get_output(0).name
        output_dtype = layer.get_output(0).dtype
        if (output_name not in self.calibration and
                output_dtype == trt.DataType.FLOAT):
            min_val = float(weights.min())
            max_val = float(weights.max())
            self.calibration[output_name] = [min_val, max_val]
        return {
            "weights": self._append_weights(weights),
            "shape": convert_to_list(layer.shape),
        }

    def _get_convolution_params(self, layer: trt.ILayer) -> Dict[str, Any]:
        layer.__class__ = trt.IConvolutionLayer
        return {
            "num_output_maps": layer.num_output_maps,
            "padding_mode": layer.padding_mode.name,
            "num_groups": layer.num_groups,
            "kernel": self._append_weights(layer.kernel),
            "bias": self._append_weights(layer.bias),
            "kernel_size_nd": convert_to_list(layer.kernel_size_nd),
            "stride_nd": convert_to_list(layer.stride_nd),
            "padding_nd": convert_to_list(layer.padding_nd),
            "dilation_nd": convert_to_list(layer.dilation_nd),
        }

    def _get_deconvolution_params(self, layer: trt.ILayer) -> Dict[str, Any]:
        layer.__class__ = trt.IDeconvolutionLayer
        return {
            "num_output_maps": layer.num_output_maps,
            "padding_mode": layer.padding_mode.name,
            "num_groups": layer.num_groups,
            "kernel": self._append_weights(layer.kernel),
            "bias": self._append_weights(layer.bias),
            "kernel_size_nd": convert_to_list(layer.kernel_size_nd),
            "stride_nd": convert_to_list(layer.stride_nd),
            "padding_nd": convert_to_list(layer.padding_nd),
        }

    def _get_elementwise_params(self, layer: trt.ILayer) -> Dict[str, Any]:
        layer.__class__ = trt.IElementWiseLayer
        return {
            "op": layer.op.name,
        }

    def _get_gather_params(self, layer: trt.ILayer) -> Dict[str, Any]:
        layer.__class__ = trt.IGatherLayer
        return {
            "axis": layer.axis,
            "num_elementwise_dims": layer.num_elementwise_dims,
            "mode": layer.mode.name,
        }

    def _get_identity_params(self, layer: trt.ILayer) -> Dict[str, Any]:
        input_name = layer.get_input(0).name
        output_name = layer.get_output(0).name
        output_dtype = layer.get_output(0).dtype
        if (input_name in self.calibration and
                output_name not in self.calibration and
                output_dtype == trt.DataType.FLOAT):
            self.calibration[output_name] = self.calibration[input_name]
        return {}

    def _get_resize_params(self, layer: trt.ILayer) -> Dict[str, Any]:
        layer.__class__ = trt.IResizeLayer
        params = {
            "resize_mode": layer.resize_mode.name,
            "coordinate_transformation": layer.coordinate_transformation.name,
            "selector_for_single_pixel": layer.selector_for_single_pixel.name,
            "nearest_rounding": layer.nearest_rounding.name,
        }
        if layer.num_inputs < 2:
            params["shape"] = convert_to_list(layer.shape),
            params["scales"] = layer.scales,
        return params

    def _get_scale_params(self, layer: trt.ILayer) -> Dict[str, Any]:
        layer.__class__ = trt.IScaleLayer
        return {
            "mode": layer.mode.name,
            "shift": self._append_weights(layer.shift),
            "scale": self._append_weights(layer.scale),
            "power": self._append_weights(layer.power),
            "channel_axis": layer.channel_axis,
        }

    def _get_shuffle_params(self, layer: trt.ILayer) -> Dict[str, Any]:
        layer.__class__ = trt.IShuffleLayer
        input_name = layer.get_input(0).name
        output_name = layer.get_output(0).name
        output_dtype = layer.get_output(0).dtype
        if (input_name in self.calibration and
                output_name not in self.calibration and
                output_dtype == trt.DataType.FLOAT):
            self.calibration[output_name] = self.calibration[input_name]
        return {
            "first_transpose": convert_to_list(layer.first_transpose),
            "second_transpose": convert_to_list(layer.second_transpose),
            "reshape_dims": convert_to_list(layer.reshape_dims),
            "zero_is_placeholder": layer.zero_is_placeholder,
        }

    def _get_slice_params(self, layer: trt.ILayer) -> Dict[str, Any]:
        layer.__class__ = trt.ISliceLayer
        return {
            "mode": layer.mode.name,
            "start": convert_to_list(layer.start),
            "shape": convert_to_list(layer.shape),
            "stride": convert_to_list(layer.stride),
        }

    def _get_unary_params(self, layer: trt.ILayer) -> Dict[str, Any]:
        layer.__class__ = trt.IUnaryLayer
        return {
            "op": layer.op.name,
        }

    def _get_layer_params(self, layer: trt.ILayer) -> Dict[str, Any]:
        layer_serializers = {
            trt.LayerType.ACTIVATION: self._get_activation_params,
            trt.LayerType.CONCATENATION: self._get_concat_params,
            trt.LayerType.CONSTANT: self._get_constant_params,
            trt.LayerType.CONVOLUTION: self._get_convolution_params,
            trt.LayerType.DECONVOLUTION: self._get_deconvolution_params,
            trt.LayerType.ELEMENTWISE: self._get_elementwise_params,
            trt.LayerType.GATHER: self._get_gather_params,
            trt.LayerType.IDENTITY: self._get_identity_params,
            trt.LayerType.RESIZE: self._get_resize_params,
            trt.LayerType.SCALE: self._get_scale_params,
            trt.LayerType.SHUFFLE: self._get_shuffle_params,
            trt.LayerType.SLICE: self._get_slice_params,
            trt.LayerType.UNARY: self._get_unary_params,
        }
        if layer.type not in layer_serializers:
            raise ValueError(f"Unsupported layer type: {layer.type.name}")
        return {
            "type": layer.type.name,
            "name": layer.name,
            **layer_serializers[layer.type](layer),
            "inputs": [
                layer.get_input(i).name
                for i in range(layer.num_inputs)
            ],
            "output_names":  [
                layer.get_output(i).name
                for i in range(layer.num_outputs)
            ],
            "output_dtypes": [
                layer.get_output(i).dtype.name
                for i in range(layer.num_outputs)
            ],
            "output_ranges": [
                self.calibration.get(layer.get_output(i).name, None)
                for i in range(layer.num_outputs)
            ],
        }

    def serialize(
        self,
        network: trt.INetworkDefinition,
        calibration: Union[None, Dict[str, Sequence[float]]] = None,
    ) -> Tuple[Dict[str, Any], List[np.ndarray]]:
        """Serialize TensorRT network."""
        inputs = []
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            inputs.append({
                "name": inp.name,
                "shape": list(inp.shape),
                "dtype": inp.dtype.name,
            })
        outputs = [
            network.get_output(i).name
            for i in range(network.num_outputs)
        ]
        self.weights = []
        self.calibration = deepcopy(calibration or {})
        layers = [
            self._get_layer_params(layer)
            for layer in network
        ]
        return {
            "inputs": inputs,
            "outputs": outputs,
            "layers": layers
        }, self.weights


def save_binary_weights(weights: List[np.ndarray], path: str) -> None:
    """Save weights to binary file."""
    with gzip.open(path, "wb") as f:
        for weight in weights:
            if weight.dtype == np.int32:
                f.write(struct.pack("I", 0))
            elif weight.dtype == np.float32:
                f.write(struct.pack("I", 1))
            else:
                raise ValueError(f"Unsupported dtype: {weight.dtype.str}")
            f.write(struct.pack("I", weight.size))
            f.write(weight.tobytes())


class CustomYAMLFormatter(yaml.Dumper):
    """Custom YAML Formatter."""

    def ignore_aliases(self, _data: Any) -> bool:
        """Ignore aliases."""
        return True

    def custom_represent_list(self, data: List[Any]) -> Any:
        """Represent list."""
        for val in data:
            if not (isinstance(val, int) or isinstance(val, float)):
                return super().represent_list(data)
        return self.represent_sequence("tag:yaml.org,2002:seq", data,
                                       flow_style=True)


CustomYAMLFormatter.add_representer(
    list, CustomYAMLFormatter.custom_represent_list)


def main(
    model: str,
    output_path: str,
    calibration: Union[str, None] = None,
) -> int:
    """
    Run CLI.

    Parameters
    ----------
    model: str
        Model
    output_path: str
        Output path
    calibration: Union[str, None]
        Calibration data

    Returns
    -------
    int
        Exit code
    """
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(model):
        error_str = "\n".join([
            f"  {parser.get_error(i).desc()}"
            for i in range(parser.num_errors)
        ]) or "  Unknown error"
        raise RuntimeError(f"Failed to parse ONNX model:\n{error_str}")
    if calibration is not None:
        with open(calibration, "rt", encoding="utf-8") as f:
            calibration_data = json.load(f)
    else:
        calibration_data = None
    graph, weights = GraphSerializer().serialize(network, calibration_data)
    output_name = os.path.splitext(os.path.basename(output_path))[0]
    weights_name = f"{output_name}-weights.bin"
    weights_path = os.path.join(os.path.dirname(output_path), weights_name)
    graph["weights"] = weights_name
    with open(output_path, "wt", encoding="utf-8") as f:
        yaml.dump(graph, f, Dumper=CustomYAMLFormatter, sort_keys=False)
    save_binary_weights(weights, weights_path)

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
