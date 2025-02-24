#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build TensorRT engine."""

import os
import sys
import logging
import argparse
import json
from typing import Any, Dict, Type, Union, Tuple
import yaml
import tensorrt as trt

LOG = logging.getLogger("build_engine")


def enum_from_string(enum: Type, val: str) -> Any:
    """Get enum value from string."""
    return getattr(enum, val)


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
    parser.add_argument("-c", "--config",
                        help="Builder configuration",
                        dest="config_path",
                        required=True,
                        type=str)
    return parser.parse_args()


def get_tensor_map(network: trt.INetworkDefinition) -> \
        Dict[str, Tuple[trt.ITensor, trt.ILayer, int]]:
    """Get tensor map from network definition."""
    tensors = {}
    for layer in network:
        for i in range(layer.num_outputs):
            tensor = layer.get_output(i)
            tensors[tensor.name] = (tensor, layer, i)
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        tensors[tensor.name] = (tensor, None, 0)
    return tensors


def set_dynamic_range(tensor: trt.ITensor,
                      calibration: Dict[str, Tuple[float]]) -> None:
    """Set tensor dynamic range."""
    if tensor.dtype not in [trt.DataType.FLOAT, trt.DataType.HALF]:
        return
    if tensor.name not in calibration:
        LOG.warning("Missing calibration data for %s", tensor.name)
        return
    min_val, max_val = calibration[tensor.name]
    value = max(abs(min_val), abs(max_val))
    tensor.set_dynamic_range(-value, value)


def set_dynamic_ranges(network: trt.INetworkDefinition,
                       calibration: Dict[str, Tuple[float]]) -> None:
    """Set dynamic ranges for network."""
    for i in range(network.num_inputs):
        set_dynamic_range(network.get_input(i), calibration)
    for layer in network:
        if layer.type == trt.LayerType.CONSTANT \
                and layer.get_output(0).name not in calibration \
                and layer.get_output(0).dtype in [trt.DataType.FLOAT,
                                                  trt.DataType.HALF]:
            layer.__class__ = trt.IConstantLayer
            weights = layer.weights
            min_val = float(weights.min())
            max_val = float(weights.max())
            calibration[layer.get_output(0).name] = (min_val, max_val)
        elif layer.type in [trt.LayerType.IDENTITY, trt.LayerType.SHUFFLE] \
                and layer.get_output(0).name not in calibration \
                and layer.get_output(0).dtype in [trt.DataType.FLOAT,
                                                  trt.DataType.HALF] \
                and layer.get_input(0).name in calibration:
            calibration[layer.get_output(0).name] = \
                calibration[layer.get_input(0).name]
        for i in range(layer.num_outputs):
            set_dynamic_range(layer.get_output(i), calibration)


def create_builder_config(builder: trt.Builder,
                          network: trt.INetworkDefinition,
                          config: Dict[str, Any]) -> \
        Tuple[trt.IBuilderConfig, Union[trt.ITimingCache, None]]:
    """Set builder config parameters."""
    builder_config = builder.create_builder_config()
    timing_cache = None
    layers = {x.name: x for x in network}
    tensors = get_tensor_map(network)
    if not builder.platform_has_tf32:
        builder_config.clear_flag(trt.BuilderFlag.TF32)
    quant_int8 = config.get("int8", False)
    quant_fp16 = config.get("fp16", False) or quant_int8
    if quant_fp16:
        if not builder.platform_has_fast_fp16:
            LOG.warning("FP16 is slow on this platform")
        builder_config.set_flag(trt.BuilderFlag.FP16)
    if quant_int8:
        if not builder.platform_has_fast_int8:
            LOG.warning("INT8 is slow on this platform")
        builder_config.set_flag(trt.BuilderFlag.INT8)
    if "optimization_level" in config:
        builder_config.builder_optimization_level = \
            config["optimization_level"]
    if config.get("direct_io", False):
        builder_config.set_flag(trt.BuilderFlag.DIRECT_IO)
    for k, v in config.get("allowed_formats", {}).items():
        fmts = 0
        if not isinstance(v, list):
            v = [v]
        for fmt in v:
            fmts |= 1 << int(enum_from_string(trt.TensorFormat, fmt))
        tensor = tensors[k][0]
        if not (tensor.is_network_input or tensor.is_network_output):
            LOG.warning("Adding extra output: %s", tensor.name)
            network.mark_output(tensor)
        tensor.allowed_formats = fmts
    precision_config = config.get("precision_constraints", None)
    if precision_config is not None:
        mode = precision_config.get("mode", None)
        if mode == "PREFER":
            builder_config.set_flag(
                trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        elif mode == "OBEY":
            builder_config.set_flag(
                trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        if quant_fp16:
            for k, v in precision_config.get("tensors", {}).items():
                dtype = enum_from_string(trt.DataType, v)
                tensor, layer, idx = tensors[k]
                if tensor.is_network_input or tensor.is_network_output:
                    tensor.dtype = dtype
                else:
                    layer.set_output_dtype(idx, dtype)
                assert tensor.dtype == dtype, "Failed to set dtype"
            for k, v in precision_config.get("layers", {}).items():
                layers[k].precision = enum_from_string(trt.DataType, v)
    timing_cache_config = config.get("timing_cache", None)
    if timing_cache_config is not None:
        if timing_cache_config.get("disable", False):
            builder_config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
        if timing_cache_config.get("editable", False):
            builder_config.set_flag(trt.BuilderFlag.EDITABLE_TIMING_CACHE)
        if timing_cache_config.get("error_or_miss", False):
            builder_config.set_flag(trt.BuilderFlag.ERROR_ON_TIMING_CACHE_MISS)
        if not timing_cache_config.get("compilation_cache", True):
            builder_config.set_flag(trt.BuilderFlag.DISABLE_COMPILATION_CACHE)
        timing_cache_path = timing_cache_config.get(
            "load_serialized_cache", None)
        if timing_cache_path is not None and os.path.exists(timing_cache_path):
            with open(timing_cache_path, "rb") as f:
                timing_cache = builder_config.create_timing_cache(f.read())
            assert timing_cache is not None, "Failed to read timing cache"
        else:
            LOG.warning("Creating new timing cache")
            timing_cache = builder_config.create_timing_cache(b"")
        for k, v in timing_cache_config.get("records", {}).items():
            key = trt.TimingCacheKey.parse(k)
            value = trt.TimingCacheValue(v["hash"], v["timing"])
            timing_cache.update(key, value)
        success = builder_config.set_timing_cache(
            timing_cache, ignore_mismatch=True)
        assert success, "Failed to load timing cache"
    profile_config = config.get("profiling", None)
    if profile_config is not None:
        verbosity = profile_config.get("verbosity", None)
        if verbosity is not None:
            builder_config.profiling_verbosity = enum_from_string(
                trt.ProfilingVerbosity, verbosity)
    compat_config = config.get("compatibility", None)
    if compat_config is not None:
        if compat_config.get("device", False):
            builder_config.hardware_compatibility_level = \
                trt.HardwareCompatibilityLevel.AMPERE_PLUS
        if compat_config.get("version", False):
            builder_config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
        if compat_config.get("exclude_lean_runtime", False):
            builder_config.set_flag(trt.BuilderFlag.EXCLUDE_LEAN_RUNTIME)
    tactics = builder_config.get_tactic_sources()
    for k, v in config.get("tactics", {}).items():
        if v:
            tactics |= 1 << int(enum_from_string(trt.TacticSource, k))
        else:
            tactics &= ~(1 << int(enum_from_string(trt.TacticSource, k)))
    builder_config.set_tactic_sources(tactics)
    if "calibration_path" in config and quant_int8:
        with open(config["calibration_path"], "rt", encoding="utf-8") as f:
            calibration = json.load(f)
        set_dynamic_ranges(network, calibration)
    return builder_config, timing_cache


def save_timing_cache(builder_config: trt.IBuilderConfig,
                      config: Dict[str, Any]) -> None:
    """Save timing cache."""
    timing_cache_config = config.get("timing_cache", None)
    if timing_cache_config is None:
        return
    timing_cache = builder_config.get_timing_cache()
    if not timing_cache:
        return
    timing_cache_report_path = timing_cache_config.get(
        "save_report_path", None)
    if timing_cache_report_path is not None:
        parsed_timing_cache = {}
        for key in timing_cache.queryKeys():
            value = timing_cache.query(key)
            parsed_timing_cache[str(key)] = {
                "hash": value.tacticHash,
                "timing": value.timingMSec
            }
        with open(timing_cache_report_path, "wt", encoding="utf-8") as f:
            yaml.dump(parsed_timing_cache, f, sort_keys=False)

    timing_cache_path = timing_cache_config.get("save_serialized_cache", None)
    if timing_cache_path is not None:
        if os.path.exists(timing_cache_path):
            with open(timing_cache_path, "rb") as f:
                old_timing_cache = builder_config.create_timing_cache(f.read())
            if old_timing_cache is not None:
                success = timing_cache.combine(
                    old_timing_cache, ignore_mismatch=True)
                assert success, "Cannot load timing cache"
        with open(timing_cache_path, "wb") as f:
            f.write(timing_cache.serialize())


def save_profile(engine: trt.ICudaEngine, config: Dict[str, Any]) -> None:
    """Save profile."""
    profile_config = config.get("profiling", None)
    if profile_config is None:
        return
    profile_report_path = profile_config.get("save_report_path", None)
    if profile_report_path is None:
        return
    inspector = engine.create_engine_inspector()
    with open(profile_report_path, "wt", encoding="utf-8") as f:
        f.write(inspector.get_engine_information(
            trt.LayerInformationFormat.JSON))


def main(
    config_path: str,
) -> int:
    """
    Run CLI.

    Parameters
    ----------
    config_path: str
        Builder configuration

    Returns
    -------
    int
        Exit code
    """
    with open(config_path, "rt", encoding="utf-8") as f:
        config = yaml.unsafe_load(f)
    trt_log = trt.Logger(trt.Logger.VERBOSE
                         if config.get("verbose", False)
                         else trt.Logger.INFO)
    builder = trt.Builder(trt_log)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_log)
    if not parser.parse_from_file(config["model_path"]):
        error_str = "\n".join([
            f"  {parser.get_error(i).desc()}"
            for i in range(parser.num_errors)
        ]) or "  Unknown error"
        raise RuntimeError(f"Failed to parse ONNX model:\n{error_str}")
    builder_config, timing_cache = create_builder_config(
        builder, network, config)
    built_engine = builder.build_serialized_network(network, builder_config)
    assert built_engine is not None, "Build failed"
    runtime = trt.Runtime(trt_log)
    engine = runtime.deserialize_cuda_engine(built_engine)
    outputs = [x for x in [engine.get_tensor_name(i)
                           for i in range(engine.num_io_tensors)]
               if engine.get_tensor_mode(x) == trt.TensorIOMode.OUTPUT]
    indices = [outputs.index(network.get_output(i).name)
               for i in range(min(network.num_outputs, network.num_inputs))]
    LOG.info("Output indices: %s", indices)
    with open(config["output_path"], "wb") as f:
        f.write(built_engine)
        if config.get("include_output_indices", True) \
                and indices != list(range(network.num_inputs)):
            f.write(bytes(indices + [len(indices)]))
    save_timing_cache(builder_config, config)
    del timing_cache
    save_profile(engine, config)
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
