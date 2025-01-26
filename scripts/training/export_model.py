#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Export model."""

import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from keras_layers import CUSTOM_LAYERS
import onnx
import tf2onnx
from onnxsim import simplify

keras.config.enable_unsafe_deserialization()


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Export model for inference")
    parser.add_argument("config_path",
                        help="Model definition",
                        type=str)
    parser.add_argument("weights_path",
                        help="Model weights",
                        type=str)
    parser.add_argument("output_path",
                        help="Output file",
                        type=str)
    parser.add_argument("--opset",
                        help="ONNX opset version",
                        type=int,
                        default=12,
                        required=False)
    parser.add_argument("--num-checks",
                        help="Number of simplifier checks",
                        type=int,
                        default=3,
                        required=False)
    return parser.parse_args()


def load_model(config_path: str, weights_path: str) -> keras.Model:
    """Load serialized model.

    Parameters
    ----------
    config_path: str
        Path to model definition
    weights_path: str
        Path to model weights

    Returns
    -------
    keras.Model
        Deserialised model
    """
    # pylint: disable=invalid-name
    with open(config_path, "rt", encoding="utf-8") as f:
        config = f.read()
    model = keras.models.model_from_json(config, custom_objects=CUSTOM_LAYERS)
    model.load_weights(weights_path)
    return model


def wrap_model(model: keras.Model, name: str = "final") -> keras.Model:
    """Wrap model for export.

    Parameters
    ----------
    model: keras.Model
        Input model
    name: str
        Wrapped model name

    Returns
    -------
    keras.Model
        Wrapped model
    """
    inputs = [
        keras.Input(
            shape=np.array(x.shape)[[3, 1, 2]] if idx != 0 else x.shape[1:],
            name=x.name,
            dtype=x.dtype,
        )
        for idx, x in enumerate(model.inputs)
    ]
    inputs_pr = [
        layers.Lambda(lambda x: K.permute_dimensions(
            x, pattern=[0, 2, 3, 1]))(inp) if idx != 0 else inp
        for idx, inp in enumerate(inputs)
    ]
    outputs = model(inputs_pr)
    outputs = [outputs["output"]] + [
        layers.Lambda(lambda x: K.permute_dimensions(
            x, pattern=[0, 3, 1, 2]), dtype=output.dtype)(output)
        for output in [outputs["output_raw"]] + outputs["last_frames"]
    ]
    outputs = [
        layers.Identity(name=x, dtype=output.dtype)(output)
        for x, output in zip(
            ["output", "output_raw"] +
            [f"out_frame_{i}" for i in range(len(model.outputs) - 2)], outputs
        )
    ]
    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=name
    )
    return model


def create_onnx_model(model: keras.Model, opset: int = 12,
                      num_checks: int = 3) -> onnx.ModelProto:
    """Create ONNX model from keras model.

    Parameters
    ----------
    model: keras.Model
        Input model
    opset: int
        Opset version
    num_checks: int
        Number of checks for ONNX simplification

    Returns
    -------
    onnx.ModelProto
        Created ONNX model
    """
    model_proto, _ = tf2onnx.convert.from_keras(
        model=model,
        input_signature=[
            tf.TensorSpec(shape=x.shape, dtype=x.dtype, name=x.name)
            for x in model.inputs
        ],
        opset=opset,
    )
    model_proto, check = simplify(model_proto, overwrite_input_shapes={
        x.name: [1] + list(x.shape)[1:]
        for x in model.inputs
    }, check_n=num_checks)
    assert check
    return model_proto


def main(
    config_path: str,
    weights_path: str,
    output_path: str,
    opset: int = 2,
    num_checks: int = 3,
) -> int:
    """
    Run CLI.

    Parameters
    ----------
    config_path: str
        Path to model definition
    weights_path: str
        Path to model weights
    output_path: str
        Path to output file
    opset: int
        ONNX opset version
    num_checks: int
        Number of simplifier checks

    Returns
    -------
    int
        Exit code
    """
    model = load_model(config_path, weights_path)
    model.summary()
    model = wrap_model(model)
    onnx_model = create_onnx_model(model, opset=opset, num_checks=num_checks)
    onnx.save(onnx_model, output_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(**vars(parse_args())))
    except KeyboardInterrupt:
        sys.exit(-1)
