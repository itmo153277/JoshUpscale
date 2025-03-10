#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Remove uint8 dtype from model."""

import sys
import argparse
from typing import List
import onnx
from onnx import helper
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
        description="Remove uint8 dtype from model")
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


def remove_uint8_values(values: List[onnx.ValueInfoProto]) \
        -> List[onnx.ValueInfoProto]:
    """Remove uint8 from list of tensor infos."""
    return [
        val
        if val.type.tensor_type.elem_type != onnx.TensorProto.UINT8
        else helper.make_tensor_value_info(
            name=val.name,
            elem_type=onnx.TensorProto.FLOAT,
            shape=[
                dim.dim_value
                for dim in val.type.tensor_type.shape.dim
            ]
        )
        for val in values
    ]


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
    for node in graph.nodes:
        if (node.op_type == "Cast" and
                node.attribute[0].i == onnx.TensorProto.UINT8):
            node.attribute[0].i = onnx.TensorProto.FLOAT
    # Switch up cast/transpose
    out_node = graph.find_node_by_output(graph.outputs[0].name)
    out_parent_node = graph.find_node_by_output(out_node.input[0])
    if out_node.op_type == "Transpose" and out_parent_node.op_type == "Cast":
        graph.remove_node(out_node)
        graph.remove_node(out_parent_node)
        out_parent_node.output[0] = out_node.output[0]
        out_node.input[0] = out_parent_node.input[0]
        out_parent_node.input[0] = "out_trans_switch"
        out_node.output[0] = "out_trans_switch"
        graph.insert_node(out_node)
        graph.insert_node(out_parent_node)
    in_node = graph.find_nodes_by_input(graph.inputs[0].name)[0]
    in_next_node = graph.find_nodes_by_input(in_node.output[0])[0]
    if in_node.op_type == "Transpose" and in_next_node.op_type == "Cast":
        graph.remove_node(in_node)
        graph.remove_node(in_next_node)
        in_next_node.input[0] = in_node.input[0]
        in_node.output[0] = in_next_node.output[0]
        in_next_node.output[0] = "inp_trans_switch"
        in_node.input[0] = "inp_trans_switch"
        graph.insert_node(in_node)
        graph.insert_node(in_next_node)
    model = graph.serialize(
        inputs=remove_uint8_values(graph.inputs),
        outputs=remove_uint8_values(graph.outputs))
    model = simplify_model(model, num_checks=num_checks)
    onnx.save(model, output_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(**vars(parse_args())))
    except KeyboardInterrupt:
        sys.exit(-1)
