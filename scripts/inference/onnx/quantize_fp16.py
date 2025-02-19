#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Quantize model to fp16."""

import sys
import argparse
import numpy as np
import onnx
from onnxconverter_common import float16
from onnx import numpy_helper
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
        description="Quantize model to fp16")
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
    return parser.parse_args()


def main(
    model_path: str,
    output_path: str,
    num_checks: int,
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

    Returns
    -------
    int
        Exit code
    """
    model = onnx.load(model_path)
    model = float16.convert_float_to_float16(
        model,
        keep_io_types=False,
        op_block_list=[],
    )
    graph = Graph(model)
    # Fix resize nodes
    for node in graph.nodes:
        if node.op_type != "Resize":
            continue
        scales = numpy_helper.to_array(graph.init_dict[node.input[2]])
        graph.create_value(node.input[2], scales.astype(np.float32))
    model = graph.serialize()
    model = simplify_model(model, num_checks=num_checks)
    onnx.save(model, output_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(**vars(parse_args())))
    except KeyboardInterrupt:
        sys.exit(-1)
