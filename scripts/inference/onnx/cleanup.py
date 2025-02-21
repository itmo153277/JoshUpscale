#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Clean up final model."""

import sys
import argparse
import onnx
import numpy as np
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
        description="Clean up model")
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


def simplify_consts(graph: Graph) -> None:
    """Convert const to rank 0 if possible."""
    for name, init in graph.init_dict.items():
        if init.dims != [1, 1, 1, 1]:
            continue
        val = numpy_helper.to_array(init).ravel()[0]
        graph.create_value(name, val)


def merge_conv_trans_mul(graph: Graph) -> None:
    """Merge ConvTranspose with Mul."""
    for node in graph.nodes:
        if node.op_type != "Mul":
            continue
        parent = graph.find_node_by_output(node.input[0])
        if parent.op_type != "ConvTranspose":
            continue
        if len(parent.input) > 2:
            continue
        weights = numpy_helper.to_array(graph.init_dict[parent.input[1]])
        scales = numpy_helper.to_array(graph.init_dict[node.input[1]])
        if scales.shape != (1, weights.shape[1], 1, 1):
            continue
        output = node.output[0]
        graph.remove_node(node)
        graph.remove_node(parent)
        parent.output[0] = output
        graph.insert_node(parent)
        graph.create_value(parent.input[1], weights * scales)


def merge_conv_trans_add(graph: Graph) -> None:
    """Merge ConvTranspose with Add."""
    for node in graph.nodes:
        if node.op_type != "Add":
            continue
        parent = graph.find_node_by_output(node.input[0])
        if parent.op_type != "ConvTranspose":
            continue
        if len(parent.input) > 2:
            continue
        weights_shape = graph.init_dict[parent.input[1]].dims
        bias = numpy_helper.to_array(graph.init_dict[node.input[1]])
        if bias.shape != (1, weights_shape[1], 1, 1):
            continue
        output = node.output[0]
        graph.remove_node(node)
        graph.remove_node(parent)
        parent.output[0] = output
        parent.input.append(node.input[1])
        graph.insert_node(parent)
        graph.create_value(node.input[1], bias.ravel())


def transpose_node_const(node: onnx.NodeProto, graph: Graph) -> None:
    """Transpose node to NCHW by updating const."""
    for i in range(2):
        if node.input[i] in graph.init_dict:
            break
    else:
        return
    val = numpy_helper.to_array(graph.init_dict[node.input[i]])
    if len(val.shape) == 0:
        return
    if len(val.shape) == 1:
        val = val[np.newaxis, np.newaxis, np.newaxis, :]
    if len(val.shape) == 4:
        val = np.transpose(val, [0, 3, 1, 2])
        graph.create_value(node.input[i], val)
        return
    raise ValueError(f"Cannot transpose {node.op_type}: wrong rank")


def transpose_node(node: onnx.NodeProto, graph: Graph) -> None:
    """Transpose node to NCHW."""
    if node.op_type == "Cast":
        return
    if node.op_type == "Mul":
        transpose_node_const(node, graph)
        return
    if node.op_type == "Sub":
        transpose_node_const(node, graph)
        return
    if node.op_type == "Pad":
        pads = numpy_helper.to_array(graph.init_dict[node.input[1]])
        if pads.shape != (8,):
            raise ValueError("Cannot transpose Pad: wrong rank")
        pads = pads[[0, 3, 1, 2, 4, 7, 5, 6]]
        graph.create_value(node.input[1], pads)
        return
    raise ValueError(f"Cannot transpose {node.op_type}")


def transpose_branch(graph: Graph, inp: str) -> None:
    """Transpose graph branch to NCHW."""
    nodes = graph.find_nodes_by_input(inp)
    for node in nodes:
        if node.op_type == "ReduceMean":
            for attr in node.attribute:
                if attr.name == "axes":
                    axes = attr.ints
                    break
            else:
                continue
            if 1 in axes and 2 in axes:
                continue
        if node.op_type == "Transpose":
            perm = None
            for attr in node.attribute:
                if attr.name == "perm":
                    perm = attr.ints
                    break
            if perm == [0, 3, 1, 2]:
                for tr_node in graph.nodes:
                    for i, n_inp in enumerate(tr_node.input):
                        if n_inp == node.output[0]:
                            tr_node.input[i] = inp
                for out in graph.outputs:
                    if out.name == node.output[0]:
                        graph.create_node(
                            name=out.name,
                            op_type="Identity",
                            inputs=[inp]
                        )
                        break
                graph.remove_node(node)
                continue
        transpose_node(node, graph)
        transpose_branch(graph, node.output[0])


def optimize_input_transpose(graph: Graph) -> None:
    """Optimize Transpose for graph input."""
    inp = graph.inputs[0].name
    transpose_branch(graph, graph.inputs[0].name)
    for node in graph.nodes:
        for i, n_inp in enumerate(node.input):
            if n_inp == inp:
                node.input[i] = "inp_trans"
    graph.create_node(
        name="inp_trans",
        op_type="Transpose",
        inputs=[inp],
        perm=[0, 3, 1, 2]
    )


def optimize_grid_sample(graph: Graph) -> None:
    """Optimize GridSample calculation in graph."""
    chains = [["GridSample", "Add", "Div", "Slice", "Transpose", "Sub",
               "Slice", "DepthToSpace", "Conv"],
              ["GridSample", "Add", "Div", "Slice", "Transpose", "Sub",
               "DepthToSpace", "Conv"]]
    chain = graph.find_node_chain(chains[0])
    if chain:
        conv_node = chain[0][8]
    else:
        chain = graph.find_node_chain(chains[1])
        if chain:
            conv_node = chain[0][7]
        else:
            return
    grid_node = chain[0][0]
    add_node = chain[0][1]
    div_node = chain[0][2]
    slice_rev_node = chain[0][3]
    grid_transpose_node = chain[0][4]
    grid_shift_node = chain[0][5]

    add_const = numpy_helper.to_array(graph.init_dict[add_node.input[1]])
    div_const = numpy_helper.to_array(graph.init_dict[div_node.input[1]])
    grid_shift_const = numpy_helper.to_array(
        graph.init_dict[grid_shift_node.input[0]])
    weights = numpy_helper.to_array(graph.init_dict[conv_node.input[1]])
    bias = numpy_helper.to_array(graph.init_dict[conv_node.input[2]])

    grid_shift_const = (grid_shift_const[:, ::-1, :, :] /
                        div_const.reshape(1, 2, 1, 1))
    grid_shift_const = np.transpose(grid_shift_const, [0, 2, 3, 1])
    bias = (bias.reshape(-1, 2)[:, ::-1] / div_const - add_const).ravel()
    weights = (weights.reshape((-1, 2) + weights.shape[1:])[:, ::-1, :, :] /
               div_const.reshape(1, 2, 1, 1, 1)).reshape(weights.shape)

    graph.create_value(grid_shift_node.input[0], grid_shift_const)
    graph.create_value(conv_node.input[1], weights)
    graph.create_value(conv_node.input[2], bias)
    graph.remove_node(add_node)
    graph.remove_node(div_node)
    graph.remove_node(slice_rev_node)
    grid_transpose_node.input[0] = grid_shift_node.input[1]
    grid_shift_node.input[1] = grid_transpose_node.output[0]
    grid_node.input[1] = grid_shift_node.output[0]


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
    opset: Union[int, None]
        Opset

    Returns
    -------
    int
        Exit code
    """
    model = onnx.load(model_path)
    graph = Graph(model)
    simplify_consts(graph)
    merge_conv_trans_mul(graph)
    merge_conv_trans_add(graph)
    optimize_input_transpose(graph)
    optimize_grid_sample(graph)
    model = graph.serialize()
    model = simplify_model(model, num_checks=num_checks)
    onnx.save(model, output_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(**vars(parse_args())))
    except KeyboardInterrupt:
        sys.exit(-1)
