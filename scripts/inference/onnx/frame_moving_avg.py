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
                        default=0.25)
    parser.add_argument("-t", "--threshold",
                        help=("Scene detection threshold " +
                              "(default: %(default).3f)"),
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
    threshold: float,
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
    threshold: float
        Scene detection threshold

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

    graph.remove_node(target_node)
    target_node.output[0] = "output_pre_mask"
    graph.insert_node(target_node)

    graph.create_node(
        "output_diff", "Sub", ["output_pre_mask", input_tensor]
    )
    graph.create_node(
        "output_diff_abs", "Abs", ["output_diff"]
    )
    graph.create_node(
        "output_diff_mean", "ReduceMean", ["output_diff_abs"]
    )
    graph.create_constant("scene_threshold", np.float32(threshold))
    graph.create_node(
        "scene_th_diff", "Sub", ["output_diff_mean", "scene_threshold"]
    )
    graph.create_node(
        "scene_cond", "Sign", ["scene_th_diff"]
    )
    graph.create_constant("output_mask_const_1", np.float32(strength / 2))
    graph.create_constant("output_mask_const_2", np.float32(-strength / 2))
    graph.create_constant("output_mask_const_3", np.float32(1 - strength / 2))
    graph.create_node(
        "output_mask_a", "Mul", ["scene_cond", "output_mask_const_2"]
    )
    graph.create_node(
        "output_mask", "Add", ["output_mask_a", "output_mask_const_1"]
    )
    graph.create_node(
        "output_mask_b", "Mul", ["scene_cond", "output_mask_const_1"]
    )
    graph.create_node(
        "output_mask_2", "Add", ["output_mask_b", "output_mask_const_3"]
    )
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
