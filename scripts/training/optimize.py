#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Optimize model for inference."""

# pylint: disable=no-name-in-module

import sys
import argparse
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants \
    import convert_variables_to_constants_v2
from tensorflow.python.framework import tensor_util
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.tools.optimize_for_inference_lib \
    import optimize_for_inference

import config
import training


def parse_args():
    """
    Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Optimize model for inference")
    parser.add_argument("-m", "--model",
                        dest="model_type",
                        help="Model type",
                        choices=["large"],
                        type=str,
                        default=None,
                        required=False)
    parser.add_argument("--config",
                        dest="config_override",
                        help="Path to model config override file",
                        type=str,
                        default=None,
                        required=False)
    parser.add_argument("--frvsr-weights",
                        help="Path to FRVSR weights",
                        type=str,
                        default=None,
                        required=False)
    parser.add_argument("--gan-weights",
                        help="Path to GAN weights",
                        type=str,
                        default=None,
                        required=False)
    parser.add_argument("output",
                        help="Output",
                        type=str,
                        default=None)

    args = parser.parse_args()
    if args.frvsr_weights is None and args.gan_weights is None:
        parser.error("should specify FRVSR or GAN weights")
    return args


def optimise_model(
    model,
    output
):
    """
    Optimise model.

    Parameters
    ----------
    model : keras.Model
        Model
    output : str
        Output
    """
    @tf.function
    def model_inference(cur_frame, last_frame, pre_gen):
        """Run model for inference."""
        gen_outputs, _ = model([
            cur_frame[:, :, :, ::-1],
            last_frame[:, :, :, ::-1],
            pre_gen[:, :, :, ::-1],
        ], training=False)
        return gen_outputs[:, :, :, ::-1]

    model_inference = model_inference.get_concrete_function(
        cur_frame=tf.TensorSpec(
            model.input[0].shape,
            model.input[0].dtype
        ),
        last_frame=tf.TensorSpec(
            model.input[1].shape,
            model.input[1].dtype
        ),
        pre_gen=tf.TensorSpec(
            model.input[2].shape,
            model.input[2].dtype
        ))
    model_inference = convert_variables_to_constants_v2(model_inference)
    opt_gd = model_inference.graph.as_graph_def()
    for node in opt_gd.node:
        if node.name == model_inference.structured_outputs[0].op.name:
            node.name = "output"
            inputs_to_delete = [
                inp for inp in node.input if inp[0] == "^"
            ]
            for inp in inputs_to_delete:
                node.input.remove(inp)

    opt_gd = optimize_for_inference(
        opt_gd,
        ["cur_frame", "last_frame", "pre_gen"],
        ["output"],
        [x.dtype.as_datatype_enum for x in model.input]
    )
    for node in opt_gd.node:
        if node.name.endswith("_bn_offset"):
            value = tensor_util.MakeNdarray(node.attr["value"].tensor)
            node.attr["dtype"].type = tf.half.as_datatype_enum
            node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(
                tensor=tensor_util.make_tensor_proto(
                    value,
                    tf.half.as_datatype_enum,
                    value.shape
                )
            ))
    with open(output, "wb") as out_file:
        out_file.write(opt_gd.SerializeToString())


def main(
    output,
    model_type="large",
    config_override=None,
    frvsr_weights=None,
    gan_weights=None,
):
    """
    Run CLI.

    Parameters
    ----------
    output : str
        Output
    model_type : str
        Model type
    config_override : str or dict
        Model config override
    frvsr_weights : str
        FRVSR weights
    gan_weights : str
        GAN weights

    Returns
    -------
    int
        Exit code
    """
    if config_override is not None and isinstance(config_override, str):
        with open(config_override, "rt") as config_file:
            config_override = json.load(config_file)
    model_config = config.get_config(
        model=model_type,
        config_override=config_override
    )
    model_config = config.merge_configs(
        model_config,
        {
            "generator": {
                "input_dtypes": ["float32", "float32"],
                "output_dtype": "float32"
            },
            "flow": {
                "input_dtypes": "float32"
            },
            "full_model": {
                "dtype": "float32"
            }
        }
    )
    keras.backend.set_floatx("float16")
    model = training.Training(model_config)
    model.init()
    if frvsr_weights is not None:
        model.frvsr_model.load_weights(frvsr_weights)
    if gan_weights is not None:
        model.gan_model.load_weights(gan_weights)
    optimise_model(model.full_model, output)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(**vars(parse_args())))
    except KeyboardInterrupt:
        sys.exit(-1)
