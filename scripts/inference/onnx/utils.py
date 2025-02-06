# -*- coding: utf-8 -*-

"""Utility functions."""

import onnx
from onnxsim import simplify


def simplify_model(model: onnx.ModelProto,
                   num_checks: int = 3,
                   convert_static_shape: bool = False) -> onnx.ModelProto:
    """Simplify onnx model.

    Parameters
    ----------
    model: onnx.ModelProto
        Input model
    num_checks: int
        Number of checks
    convert_static_shape: bool
        Convert shapes from dynamic to static

    Returns
    -------
    onnx.ModelProto
        Simplified model
    """
    overwrite_input_shapes = None
    if convert_static_shape:
        overwrite_input_shapes = {
            x.name: [1 if y.dim_param else y.dim_value
                     for y in x.type.tensor_type.shape.dim]
            for x in model.graph.input
        }
    model, check = simplify(model, check_n=num_checks,
                            overwrite_input_shapes=overwrite_input_shapes)
    assert check, "Model is broken"
    return model


def get_opset_version(model: onnx.ModelProto) -> int:
    """Get ONNX opset version.

    Parameters
    ----------
    model: onnx.ModelProto
        Input model

    Returns
    -------
    int
        opset version
    """
    for imp in model.opset_import:
        if imp.domain == "":
            return imp.version
    return 0
