#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Change model output to flow frames."""

import sys
import argparse
import onnx
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
        description="Change model output to flow frames.")
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


# Hardcoded nodes
INPUT_NODE = "final/full/generator/space_to_depth/SpaceToDepth"
TARGET_NODE = "final/full/postprocess/mul"


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
    graph = Graph(model)

    input_node = graph.find_node_by_name(INPUT_NODE)
    target_node = graph.find_node_by_name(TARGET_NODE)
    target_node.input[0] = input_node.input[0]

    model = graph.serialize()
    model = simplify_model(model, num_checks=num_checks)
    onnx.save(model, output_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(**vars(parse_args())))
    except KeyboardInterrupt:
        sys.exit(-1)
