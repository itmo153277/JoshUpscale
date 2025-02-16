#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Simplify model."""

import sys
import argparse
from typing import Union
import onnx
from onnx import version_converter
from utils import simplify_model, get_opset_version


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Simplify model")
    parser.add_argument("model_path",
                        help="Model",
                        type=str)
    parser.add_argument("output_path",
                        help="Output path",
                        type=str)
    parser.add_argument("--num-checks",
                        help="Number of simplifier checks",
                        type=int,
                        default=3)
    parser.add_argument("--opset",
                        help="Opset",
                        type=int,
                        default=None)
    return parser.parse_args()


# Hardcoded nodes
INPUT_NODE = "final_1/full_1/generator_1/space_to_depth_1/SpaceToDepth"
TARGET_NODE = "final_1/full_1/generator_1/clip_1/clip_by_value"


def main(
    model_path: str,
    output_path: str,
    num_checks: int,
    opset: Union[int, None] = None,
) -> int:
    """
    Run CLI.

    Parameters
    ----------
    model_path: str
        Path to model
    output_path: str
        Output path
    num_checks: int
        Number of simplifier checks
    opset: Union[int, None]
        Opset

    Returns
    -------
    int
        Exit code
    """
    model = onnx.load(model_path)
    if opset is not None and get_opset_version(model) != opset:
        model = version_converter.convert_version(model, opset)
    model = simplify_model(
        model,
        num_checks=num_checks,
        convert_static_shape=True,
    )
    onnx.save(model, output_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(**vars(parse_args())))
    except KeyboardInterrupt:
        sys.exit(-1)
