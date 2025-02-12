#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Replace tf-addons' dense_warp with native Microsoft implementation."""

import sys
import argparse
import numpy as np
import onnx
from onnx import version_converter
from graph import Graph
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
        description=(
            "Replace tf-addons' dense_warp with native " +
            "Microsoft implementation")
    )
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
OUT_NODE = "final_1/full_1/generator_1/space_to_depth_1/SpaceToDepth"
GRID_NODE = "final_1/full_1/dense_warp_1/StatefulPartitionedCall/" + \
    "dense_image_warp/Reshape"


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
    if get_opset_version(model) < 16:
        model = version_converter.convert_version(model, 16)
    graph = Graph(model)

    out_name = graph.find_node_by_name(OUT_NODE).input[0]
    grid_inp = graph.find_node_by_name(GRID_NODE).input[0]
    parent = graph.find_node_by_output(out_name)
    parent =  graph.find_node_by_output(parent.input[0])
    if parent.op_type == "Add":
        graph.remove_node(parent)
        parent.output[0] = out_name
        parent.input[0] = "grid_out"
        out_name = "grid_out"
        graph.insert_node(parent)
    img_inp = graph.inputs[1].name
    width = graph.inputs[1].type.tensor_type.shape.dim[3].dim_value
    height = graph.inputs[1].type.tensor_type.shape.dim[2].dim_value

    graph.create_constant("grid_sl_start", [-1])
    graph.create_constant("grid_sl_end", [-3])
    graph.create_constant("grid_sl_axis", [-1])
    graph.create_constant("grid_steps", [-1])
    graph.create_node(
        "grid_val", "Slice",
        [
            grid_inp, "grid_sl_start", "grid_sl_end", "grid_sl_axis",
            "grid_steps"
        ]
    )
    graph.create_constant("grid_norm_const", np.array(
        [width * 0.5, height * 0.5], dtype=np.float32))
    graph.create_node("grid_norm", "Div", ["grid_val", "grid_norm_const"])
    graph.create_constant("grid_shift_const", np.array(
        [-1 + 1 / width, -1 + 1 / height], dtype=np.float32))
    graph.create_node("grid_shift", "Add", ["grid_norm", "grid_shift_const"])
    graph.create_node(
        "grid_sample", "GridSample", [img_inp, "grid_shift"],
        outputs=[out_name],
        mode="bilinear",
        padding_mode="border",
        align_corners=0,
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
