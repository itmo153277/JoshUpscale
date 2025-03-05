#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Change model output to moving average."""

import sys
import argparse
import enum
import onnx
import numpy as np
from graph import Graph
from utils import simplify_model


class NormType(enum.Enum):
    """Norm type."""

    L1 = 0
    L2 = 1

    def __str__(self) -> str:
        """Convert to str."""
        return self.name

    @staticmethod
    def argparse_value(val: str) -> "NormType":
        """Create from string."""
        try:
            return NormType[val.upper()]
        except KeyError as exc:
            raise argparse.ArgumentTypeError(f"Invalid value: {val}") from exc


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
    parser.add_argument("-w", "--window",
                        help=("Scene detection window (0=global) " +
                              "(default: %(default)d)"),
                        type=int,
                        default=0)
    parser.add_argument("-t", "--threshold",
                        help=("Scene detection threshold " +
                              "(default: %(default).3f)"),
                        type=float,
                        default=0.1)
    parser.add_argument("-g", "--gain",
                        help=("Scene detection gain (0=use sign fn) " +
                              "(default: %(default).3f)"),
                        type=float,
                        default=0)
    parser.add_argument("-n", "--norm",
                        help=("Scene detection norm type " +
                              "(default: %(default)s)"),
                        type=NormType.argparse_value,
                        choices=list(NormType),
                        default=NormType.L1)
    parser.add_argument("-l", "--limit",
                        help="Limit pre_warp output to output range",
                        action="store_true",
                        default=False)
    parser.add_argument("--luma-normalize",
                        help="Normalize by luma value",
                        action="store_true",
                        default=False)
    return parser.parse_args()


# Hardcoded nodes
INPUT_NODE = "final_1/full_1/generator_1/space_to_depth_1/SpaceToDepth"
TARGET_NODE = "final_1/full_1/generator_1/clip_1/clip_by_value"

LUMA_NORM = np.array([0.1140, 0.5870, 0.2989],
                     dtype=np.float32).reshape((1, 3, 1, 1)) * 3


def main(
    model_path: str,
    output_path: str,
    num_checks: int,
    strength: float,
    window: int,
    threshold: float,
    gain: float,
    norm: NormType,
    limit: bool,
    luma_normalize: bool,
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
    window: int
        Scene detection window
    threshold: float
        Scene detection threshold
    gain: float
        Scene detection gain
    norm: NormType
        Scene detection norm type
    limit: bool
        Limit pre_warp to output range
    luma_normalize: bool
        Normalize by luma value

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
    width = graph.inputs[1].type.tensor_type.shape.dim[3].dim_value
    height = graph.inputs[1].type.tensor_type.shape.dim[2].dim_value

    graph.remove_node(target_node)
    target_node.output[0] = "output_pre_mask"
    graph.insert_node(target_node)

    if limit:
        graph.create_constant(
            "output_pre_limit_min_val", np.float32(0.5))
        graph.create_constant(
            "output_pre_limit_max_val", np.float32(-0.5))
        graph.create_node("output_pre_limit_min", "Min",
                          [input_tensor, "output_pre_limit_min_val"])
        graph.create_node("output_pre_limit_max", "Max",
                          ["output_pre_limit_min", "output_pre_limit_max_val"])
        input_tensor = "output_pre_limit_max"

    gain_coef = 1.0 if gain == 0 else gain

    graph.create_node(
        "output_diff", "Sub", ["output_pre_mask", input_tensor]
    )
    if norm == NormType.L1:
        graph.create_node(
            "output_diff_abs", "Abs", ["output_diff"]
        )
    elif norm == NormType.L2:
        graph.create_node(
            "output_diff_abs", "Mul", ["output_diff", "output_diff"]
        )
    else:
        raise ValueError(f"Unknown norm type {norm}")
    if window == 0:
        if luma_normalize:
            kernel = LUMA_NORM * gain_coef
            if norm == NormType.L2:
                kernel *= LUMA_NORM
            graph.create_constant("output_diff_gain", kernel)
            graph.create_node(
                "output_diff_abs_g", "Mul",
                ["output_diff_abs", "output_diff_gain"]
            )
            graph.create_node(
                "output_diff_mean", "ReduceMean", ["output_diff_abs_g"]
            )
            output_diff_mean = "output_diff_mean"
        else:
            graph.create_node(
                "output_diff_mean", "ReduceMean", ["output_diff_abs"]
            )
            if gain == 0:
                output_diff_mean = "output_diff_mean"
            else:
                graph.create_constant("output_diff_gain",
                                      np.float32(gain_coef))
                graph.create_node(
                    "output_diff_mean_g", "Mul",
                    ["output_diff_mean", "output_diff_gain"]
                )
                output_diff_mean = "output_diff_mean_g"
    else:
        output_shape = [
            ((x + window - 1) // window) * window
            for x in [height, width]
        ]
        padding = [
            [(x - y) // 2, x - y - (x - y) // 2]
            for x, y in zip(output_shape, [height, width])
        ]
        kernel = (np.ones(shape=[1, 3, window, window]) /
                  3 / window / window * gain_coef).astype(np.float32)
        if luma_normalize:
            kernel *= LUMA_NORM
            if norm == NormType.L2:
                kernel *= LUMA_NORM
        graph.create_value("output_diff_mean_kernel", kernel)
        graph.create_node(
            "output_diff_mean", "Conv",
            ["output_diff_abs", "output_diff_mean_kernel"],
            strides=[window, window],
            pads=[x[0] for x in padding] + [x[1] for x in padding]
        )
        output_diff_mean = "output_diff_mean"
    graph.create_constant(
        "scene_threshold", np.float32(-threshold * gain_coef))
    graph.create_node(
        "scene_th_diff", "Add", [output_diff_mean, "scene_threshold"]
    )
    if gain == 0:
        graph.create_node(
            "scene_cond", "Sign", ["scene_th_diff"]
        )
    else:
        graph.create_node("scene_cond", "Tanh", ["scene_th_diff"])
    if window == 0:
        scene_cond = "scene_cond"
    else:
        graph.create_value(
            "output_mask_scales",
            np.float32([1, 1, window, window])
        )
        graph.create_value(
            "output_mask_roi",
            np.float32([])
        )
        graph.create_node(
            "scene_cond_s", "Resize",
            ["scene_cond", "output_mask_roi", "output_mask_scales"],
            coordinate_transformation_mode="asymmetric",
            mode="linear",
            nearest_mode="floor",
            exclude_outside=0,
        )
        if any(y != 0 for x in padding for y in x):
            graph.create_constant("scene_cond_start", [x[0] for x in padding])
            graph.create_constant(
                "scene_cond_end",
                [x - y[1] for x, y in zip(output_shape, padding)]
            )
            graph.create_constant("scene_cond_axis", [-2, -1])
            graph.create_node(
                "scene_cond_p", "Slice",
                ["scene_cond_s", "scene_cond_start", "scene_cond_end",
                 "scene_cond_axis"],
            )
            scene_cond = "scene_cond_p"
        else:
            scene_cond = "scene_cond_s"
    graph.create_constant("output_mask_const_1", np.float32(strength / 2))
    graph.create_constant("output_mask_const_2", np.float32(-strength / 2))
    graph.create_constant("output_mask_const_3", np.float32(1 - strength / 2))
    graph.create_node(
        "output_mask_a", "Mul", [scene_cond, "output_mask_const_2"]
    )
    graph.create_node(
        "output_mask", "Add", ["output_mask_a", "output_mask_const_1"]
    )
    graph.create_node(
        "output_mask_b", "Mul", [scene_cond, "output_mask_const_1"]
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
