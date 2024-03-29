# -*- coding: utf-8 -*-

"""Utility functions."""

import onnx
from onnxsim import simplify


def simplify_model(model: onnx.ModelProto,
                   num_checks: int = 3) -> onnx.ModelProto:
    """Simplify onnx model.

    Parameters
    ----------
    model: onnx.ModelProto
        Input model
    num_checks: int
        Number of checks

    Returns
    -------
    onnx.ModelProto
        Simplified model
    """
    model, check = simplify(model, check_n=num_checks)
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
