#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build TensorRT engine."""

import os
import sys
import logging
import argparse
import gzip
import struct
from typing import Any, Dict, List, Type, Union
import yaml
import numpy as np
import tensorrt as trt

LOG = logging.getLogger("convert_model")


class HumanReadableSize:
    """Human readable size."""

    UNIT_MAP = {
        "KB": 1 << 10,
        "MB": 1 << 20,
        "GB": 1 << 30,
        "TB": 1 << 40,
    }

    def __init__(self, val: Union[int, str]) -> None:
        """Create object."""
        if isinstance(val, int):
            self.val = val
        elif isinstance(val, str):
            self.val = HumanReadableSize._convert_value(val)
        else:
            raise ValueError("Unsupported type")

    def __repr__(self) -> str:
        """Convert to str."""
        val = self.val
        for cur_pf, mul in reversed(HumanReadableSize.UNIT_MAP.items()):
            if val >= mul:
                postfix = cur_pf
                val //= mul
                break
        else:
            postfix = "B"
        return f"{val}{postfix}"

    def __int__(self) -> int:
        """Convert to int."""
        return self.val

    @staticmethod
    def _convert_value(val: str) -> int:
        """Convert value."""
        val = val.upper().strip()
        for postfix, mul in HumanReadableSize.UNIT_MAP.items():
            if val.endswith(postfix):
                multiplier = mul
                val = val[:-2].strip()
                break
        else:
            if val.endswith("B"):
                val = val[:-1].strip()
            multiplier = 1
        return multiplier * int(val)


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Build TensorRT engine")
    parser.add_argument("model_path",
                        help="Model",
                        type=str)
    parser.add_argument("output_path",
                        help="Output path",
                        type=str)
    parser.add_argument("--int8",
                        help="Enable int8 quantization",
                        dest="quant_int8",
                        required=False,
                        default=False,
                        action="store_true")
    parser.add_argument("--fp16",
                        help="Enable fp16 quantization",
                        dest="quant_fp16",
                        required=False,
                        default=False,
                        action="store_true")
    parser.add_argument("-w", "--workspace-size",
                        help="Workspace size limit (default: %(default)s)",
                        type=HumanReadableSize,
                        required=False,
                        default=HumanReadableSize("2GB"))
    return parser.parse_args()


def load_weights(weights_path: str) -> List[np.ndarray]:
    """Load weights from file."""
    dtype_dict = {
        0: np.int32,
        1: np.float32,
    }
    weights = []
    with gzip.open(weights_path, "rb") as f:
        while f.peek(1):
            dtype = struct.unpack("=I", f.read(4))[0]
            assert dtype in dtype_dict
            size = struct.unpack("=I", f.read(4))[0]
            if size == 0:
                weights.append(None)
            else:
                data = np.frombuffer(f.read(size * 4), dtype=dtype_dict[dtype])
                weights.append(data)
    return weights


def enum_from_string(enum: Type, val: str) -> Any:
    """Get enum value from string."""
    return getattr(enum, val)


class GraphDeserializer:
    """TensorRT graph deserializer."""

    def __init__(self) -> None:
        """Create object."""
        self._network = None
        self._tensors = {}
        self._weights = []

    def _set_dynamic_range(self, name: str,
                           val_range: Union[List[float], None]) -> None:
        tensor = self._tensors[name]
        if val_range is None:
            if tensor.dtype == trt.DataType.FLOAT:
                LOG.warning("Missing calibration data for %s", name)
            return
        min_val, max_val = val_range
        if -min_val != max_val:
            # TensorRT supports symmetric ranges only
            max_val = max(abs(min_val), abs(max_val))
            min_val = -max_val
        tensor.set_dynamic_range(min_val, max_val)

    def _add_input(self, config: Dict[str, Any]) -> None:
        name = config["name"]
        tensor = self._network.add_input(
            name=name,
            dtype=enum_from_string(trt.DataType, config["dtype"]),
            shape=config["shape"]
        )
        tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
        self._tensors[name] = tensor
        self._set_dynamic_range(name, config.get("range", None))

    def _add_activation(self, config: Dict[str, Any]) -> trt.ILayer:
        assert len(config["inputs"]) == 1
        input_tensor = self._tensors[config["inputs"][0]]
        activation_type = enum_from_string(
            trt.ActivationType, config["activation_type"])
        layer = self._network.add_activation(input_tensor, activation_type)
        layer.alpha = config["alpha"]
        layer.beta = config["beta"]
        return layer

    def _add_concat(self, config: Dict[str, Any]) -> trt.ILayer:
        input_tensors = [
            self._tensors[x]
            for x in config["inputs"]
        ]
        layer = self._network.add_concatenation(input_tensors)
        layer.axis = config["axis"]
        return layer

    def _add_constant(self, config: Dict[str, Any]) -> trt.ILayer:
        assert len(config["inputs"]) == 0
        shape = config["shape"]
        weights = self._weights[config["weights"]]
        layer = self._network.add_constant(shape, weights)
        return layer

    def _add_convolution(self, config: Dict[str, Any]) -> trt.ILayer:
        assert len(config["inputs"]) == 1
        input_tensor = self._tensors[config["inputs"][0]]
        layer = self._network.add_convolution_nd(
            input_tensor,
            config["num_output_maps"],
            config["kernel_size_nd"],
            self._weights[config["kernel"]],
            self._weights[config["bias"]]
        )
        layer.padding_mode = enum_from_string(
            trt.PaddingMode, config["padding_mode"])
        layer.num_groups = config["num_groups"]
        layer.stride_nd = config["stride_nd"]
        layer.padding_nd = config["padding_nd"]
        layer.dilation_nd = config["dilation_nd"]
        return layer

    def _add_deconvolution(self, config: Dict[str, Any]) -> trt.ILayer:
        assert len(config["inputs"]) == 1
        input_tensor = self._tensors[config["inputs"][0]]
        layer = self._network.add_deconvolution_nd(
            input_tensor,
            config["num_output_maps"],
            config["kernel_size_nd"],
            self._weights[config["kernel"]],
            self._weights[config["bias"]]
        )
        layer.padding_mode = enum_from_string(
            trt.PaddingMode, config["padding_mode"])
        layer.num_groups = config["num_groups"]
        layer.stride_nd = config["stride_nd"]
        layer.padding_nd = config["padding_nd"]
        return layer

    def _add_elementwise(self, config: Dict[str, Any]) -> trt.ILayer:
        assert len(config["inputs"]) == 2
        input_tensor1 = self._tensors[config["inputs"][0]]
        input_tensor2 = self._tensors[config["inputs"][1]]
        op = enum_from_string(trt.ElementWiseOperation, config["op"])
        layer = self._network.add_elementwise(input_tensor1, input_tensor2, op)
        return layer

    def _add_gather(self, config: Dict[str, Any]) -> trt.ILayer:
        assert len(config["inputs"]) == 2
        input_tensor = self._tensors[config["inputs"][0]]
        indices = self._tensors[config["inputs"][1]]
        layer = self._network.add_gather(
            input_tensor,
            indices,
            enum_from_string(trt.GatherMode, config["mode"])
        )
        layer.axis = config["axis"]
        layer.num_elementwise_dims = config["num_elementwise_dims"]
        return layer

    def _add_identity(self, config: Dict[str, Any]) -> trt.ILayer:
        assert len(config["inputs"]) == 1
        input_tensor = self._tensors[config["inputs"][0]]
        layer = self._network.add_identity(input_tensor)
        layer.set_output_type(0, enum_from_string(
            trt.DataType, config["output_dtypes"][0]))
        return layer

    def _add_pooling(self, config: Dict[str, Any]) -> trt.ILayer:
        assert len(config["inputs"]) == 1
        input_tensor = self._tensors[config["inputs"][0]]
        layer = self._network.add_pooling_nd(
            input_tensor,
            enum_from_string(trt.PoolingType, config["pooling_type"]),
            config["window_size_nd"],
        )
        layer.padding_mode = enum_from_string(
            trt.PaddingMode, config["padding_mode"])
        layer.blend_factor = config["blend_factor"]
        layer.average_count_excludes_padding = \
            config["average_count_excludes_padding"]
        layer.stride_nd = config["stride_nd"]
        layer.padding_nd = config["padding_nd"]
        return layer

    def _add_reduce(self, config: Dict[str, Any]) -> trt.ILayer:
        assert len(config["inputs"]) == 1
        input_tensor = self._tensors[config["inputs"][0]]
        layer = self._network.add_reduce(
            input_tensor,
            enum_from_string(trt.ReduceOperation, config["op"]),
            config["axes"],
            config["keep_dims"],
        )
        return layer

    def _add_resize(self, config: Dict[str, Any]) -> trt.ILayer:
        assert len(config["inputs"]) in [1, 2]
        input_tensor = self._tensors[config["inputs"][0]]
        layer = self._network.add_resize(input_tensor)
        if len(config["inputs"]) > 1:
            scale_tensor = self._tensors[config["inputs"][1]]
            layer.set_input(1, scale_tensor)
        else:
            if "scales" in config:
                layer.scales = config["scales"]
            elif "shape" in config:
                layer.shape = config["shape"]
            else:
                raise ValueError(f"Unknown resize scale for {config['name']}")
        layer.resize_mode = enum_from_string(
            trt.ResizeMode, config["resize_mode"])
        layer.coordinate_transformation = enum_from_string(
            trt.ResizeCoordinateTransformation,
            config["coordinate_transformation"]
        )
        layer.selector_for_single_pixel = enum_from_string(
            trt.ResizeSelector,
            config["selector_for_single_pixel"]
        )
        layer.nearest_rounding = enum_from_string(
            trt.ResizeRoundMode,
            config["nearest_rounding"]
        )
        return layer

    def _add_scale(self, config: Dict[str, Any]) -> trt.ILayer:
        assert len(config["inputs"]) == 1
        input_tensor = self._tensors[config["inputs"][0]]
        layer = self._network.add_scale(
            input_tensor,
            enum_from_string(trt.ScaleMode, config["mode"]),
            self._weights[config["shift"]],
            self._weights[config["scale"]],
            self._weights[config["power"]],
        )
        layer.channel_axis = config["channel_axis"]
        return layer

    def _add_shuffle(self, config: Dict[str, Any]) -> trt.ILayer:
        assert len(config["inputs"]) == 1
        input_tensor = self._tensors[config["inputs"][0]]
        layer = self._network.add_shuffle(input_tensor)
        layer.first_transpose = config["first_transpose"]
        layer.second_transpose = config["second_transpose"]
        if config.get("reshape_dims", None) is not None:
            layer.reshape_dims = config["reshape_dims"]
        layer.zero_is_placeholder = config["zero_is_placeholder"]
        return layer

    def _add_slice(self, config: Dict[str, Any]) -> trt.ILayer:
        assert len(config["inputs"]) == 1
        input_tensor = self._tensors[config["inputs"][0]]
        layer = self._network.add_slice(
            input_tensor, config["start"], config["shape"], config["stride"])
        layer.mode = enum_from_string(trt.SliceMode, config["mode"])
        return layer

    def _add_unary(self, config: Dict[str, Any]) -> trt.ILayer:
        assert len(config["inputs"]) == 1
        input_tensor = self._tensors[config["inputs"][0]]
        op = enum_from_string(trt.UnaryOperation, config["op"])
        layer = self._network.add_unary(input_tensor, op)
        return layer

    def _add_layer(self, config: Dict[str, Any]) -> trt.ILayer:
        layer_deserializers = {
            trt.LayerType.ACTIVATION: self._add_activation,
            trt.LayerType.CONCATENATION: self._add_concat,
            trt.LayerType.CONSTANT: self._add_constant,
            trt.LayerType.CONVOLUTION: self._add_convolution,
            trt.LayerType.DECONVOLUTION: self._add_deconvolution,
            trt.LayerType.ELEMENTWISE: self._add_elementwise,
            trt.LayerType.GATHER: self._add_gather,
            trt.LayerType.IDENTITY: self._add_identity,
            trt.LayerType.POOLING: self._add_pooling,
            trt.LayerType.REDUCE: self._add_reduce,
            trt.LayerType.RESIZE: self._add_resize,
            trt.LayerType.SCALE: self._add_scale,
            trt.LayerType.SHUFFLE: self._add_shuffle,
            trt.LayerType.SLICE: self._add_slice,
            trt.LayerType.UNARY: self._add_unary,
        }
        layer_type = enum_from_string(trt.LayerType, config["type"])
        if layer_type not in layer_deserializers:
            raise ValueError(f"Unsupported layer type: {layer_type.name}")
        layer = layer_deserializers[layer_type](config)
        layer.name = config["name"]
        assert layer.num_outputs == len(config["output_names"])
        for i, out_name, out_dtype, out_range in zip(
            range(layer.num_outputs),
            config["output_names"],
            config["output_dtypes"],
            config["output_ranges"],
        ):
            out = layer.get_output(i)
            out.name = out_name
            assert layer.get_output_type(i) == \
                enum_from_string(trt.DataType, out_dtype), \
                f"Data types do not match for {out_name}: {out_dtype} " + \
                f"vs {out.dtype.name}"
            self._tensors[out_name] = out
            self._set_dynamic_range(out_name, out_range)
        return layer

    def deserialize(self, builder: trt.Builder, model: Dict[str, Any],
                    weights: List[np.ndarray]) -> trt.INetworkDefinition:
        """Deserialize network from model definition."""
        self._tensors = {}
        self._weights = weights
        self._network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        for inp in model["inputs"]:
            self._add_input(inp)
        for layer in model["layers"]:
            self._add_layer(layer)
        for out in model["outputs"]:
            self._network.mark_output(self._tensors[out])
            self._tensors[out].allowed_formats = 1 << int(
                trt.TensorFormat.LINEAR)
        return self._network


def main(
    model_path: str,
    output_path: str,
    quant_int8: bool = False,
    quant_fp16: bool = False,
    workspace_size: Union[None, HumanReadableSize] = None,
) -> int:
    """
    Run CLI.

    Parameters
    ----------
    model_path: str
        Model
    output_path: str
        Output path
    quant_int8: bool
        Int8 quantization
    quant_fp16: bool
        FP16 quantization
    workspace_size: Union[None, HumanReadableSize]
        Workspace size limit

    Returns
    -------
    int
        Exit code
    """
    with open(model_path, "rt", encoding="utf-8") as f:
        model = yaml.unsafe_load(f)
    weights_path = os.path.join(os.path.dirname(model_path), model["weights"])
    weights = load_weights(weights_path)
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = GraphDeserializer().deserialize(builder, model, weights)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_size or 2 << 30))
    config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    if quant_fp16 or quant_int8:
        if not builder.platform_has_fast_fp16:
            LOG.warning("FP16 is slow on this platform")
        config.set_flag(trt.BuilderFlag.FP16)
    if quant_int8:
        if not builder.platform_has_fast_int8:
            LOG.warning("INT8 is slow on this platform")
        config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)
    config.set_flag(trt.BuilderFlag.DIRECT_IO)
    if builder.num_DLA_cores > 0:
        config.default_device_type = trt.DeviceType.DLA
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    engine = builder.build_serialized_network(network, config)
    assert engine is not None
    with open(output_path, "wb") as f:
        f.write(engine)
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
