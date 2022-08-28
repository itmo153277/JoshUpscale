#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Change model output to moving average."""

import sys
import argparse
import onnx
import numpy as np
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
        description="Change model output to moving average")
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
    parser.add_argument("-s", "--strength",
                        help="Filter strength (default: %(default).3f)",
                        type=float,
                        default=0.1)
    return parser.parse_args()


# Hardcoded nodes
INPUT_NODE = "final/full/generator/space_to_depth/SpaceToDepth"
TARGET_NODE = "final/full/generator/clip/clip_by_value"


def main(
    model_path: str,
    output_path: str,
    num_checks: int,
    strength: float,
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
    strength: float
        Filter strength

    Returns
    -------
    int
        Exit code
    """
    model = onnx.load(model_path)
    graph = Graph(model)

    input_node = graph.find_node_by_name(INPUT_NODE)
    target_node = graph.find_node_by_name(TARGET_NODE)
    input_tensor = input_node.input[0]
    output_tensor = target_node.output[0]
    mask = np.array(strength, dtype=np.float32).reshape((1, 1, 1, 1))

    graph.remove_node(target_node)
    target_node.output[0] = "output_pre_mask"
    graph.insert_node(target_node)

    graph.create_constant("output_mask", mask)
    graph.create_constant("output_mask_2", 1 - mask)
    graph.create_node(
        "output_masked", "Mul", [input_tensor, "output_mask"],
    )
    graph.create_node(
        "output_masked_2", "Mul", ["output_pre_mask", "output_mask_2"]
    )
    graph.create_node(
        "output_masked_3", "Add", ["output_masked", "output_masked_2"],
        outputs=[output_tensor]
    )

    model = graph.serialize()
    model = simplify_model(model, num_checks=num_checks)
    onnx.save(model, output_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(**vars(parse_args())))
    except KeyboardInterrupt:
        sys.exit(-1)
