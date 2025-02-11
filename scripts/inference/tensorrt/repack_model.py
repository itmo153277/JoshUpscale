#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Repack TensorRT models."""


import os
import sys
import logging
import argparse
import gzip
import struct
from typing import List, Any
import yaml


LOG = logging.getLogger("repack_model")


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Repack TensorRT models.")
    parser.add_argument("models",
                        help="Input model",
                        metavar="model",
                        nargs="+",
                        type=str)
    parser.add_argument("output_path",
                        help="Output path",
                        type=str)
    return parser.parse_args()


class Weights:
    """Network weights."""

    def __init__(self) -> None:
        """Create object."""
        self._weights = []
        self._weights_map = {}
        self._weights_out = []
        self._weights_out_map = {}

    def _append_weights(self, dtype: int, data: bytes) -> int:
        """Append weights."""
        weight_data = (dtype, data)
        idx = self._weights_map.get(weight_data, None)
        if idx is None:
            idx = len(self._weights)
            self._weights_map[weight_data] = idx
            self._weights.append(weight_data)
        return idx

    def load_weights(self, weights_path: str) -> List[int]:
        """Load weights from file."""

        indices = []
        with gzip.open(weights_path, "rb") as f:
            while f.peek(1):
                dtype = struct.unpack("=I", f.read(4))[0]
                size = struct.unpack("=I", f.read(4))[0]
                data = f.read(size * 4)
                indices.append(self._append_weights(dtype, data))
        return indices

    def get_weight_index(self, index: int) -> int:
        """Get weight index."""
        idx = self._weights_out_map.get(index, None)
        if idx is None:
            idx = len(self._weights_out)
            self._weights_out_map[index] = idx
            self._weights_out.append(index)
        return idx

    def save_weights(self, weights_path: str) -> None:
        """Save weights to binary file."""
        with gzip.open(weights_path, "wb") as f:
            for cur_idx in self._weights_out:
                weights = self._weights[cur_idx]
                f.write(struct.pack("=I", weights[0]))
                f.write(struct.pack("=I", len(weights[1]) // 4))
                f.write(weights[1])


class CustomYAMLFormatter(yaml.Dumper):
    """Custom YAML Formatter."""

    def ignore_aliases(self, _data: Any) -> bool:
        """Ignore aliases."""
        return True

    def custom_represent_list(self, data: List[Any]) -> Any:
        """Represent list."""
        for val in data:
            if not (isinstance(val, int) or isinstance(val, float)):
                return super().represent_list(data)
        return self.represent_sequence("tag:yaml.org,2002:seq", data,
                                       flow_style=True)


CustomYAMLFormatter.add_representer(
    list, CustomYAMLFormatter.custom_represent_list)
CustomYAMLFormatter.add_representer(
    tuple, CustomYAMLFormatter.custom_represent_list)


def main(
    models: List[str],
    output_path: str,
) -> int:
    """
    Run CLI.

    Parameters
    ----------
    models: list
        Models
    output_path: str
        Output path

    Returns
    -------
    int
        Exit code
    """

    weight_params = {
        "ACTIVATION": [],
        "CONCATENATION": [],
        "CAST": [],
        "CONSTANT": ["weights"],
        "CONVOLUTION": ["kernel", "bias"],
        "DECONVOLUTION": ["kernel", "bias"],
        "ELEMENTWISE": [],
        "GATHER": [],
        "GRID_SAMPLE": [],
        "IDENTITY": [],
        "POOLING": [],
        "REDUCE": [],
        "RESIZE": [],
        "SCALE": ["shift", "scale", "power"],
        "SHUFFLE": [],
        "SLICE": [],
        "UNARY": [],
    }

    weights = Weights()
    os.makedirs(output_path, exist_ok=True)
    for model_path in models:
        with open(model_path, "rt", encoding="utf-8") as f:
            model = yaml.unsafe_load(f)
        weights_path = os.path.join(
            os.path.dirname(model_path), model["weights"]
        )
        weight_indices = weights.load_weights(weights_path)
        renamed_tensors = {}
        for i, layer in enumerate(model["layers"]):
            if layer["name"].find("(Unnamed Layer*") >= 0:
                layer["name"] = f"layer_{i}"
            for j, output_name in enumerate(layer["output_names"]):
                if output_name.find("(Unnamed Layer*") >= 0:
                    new_name = f"{layer['name']}_{j}"
                    renamed_tensors[output_name] = new_name
                    layer["output_names"][j] = new_name
            for j, input_name in enumerate(layer["inputs"]):
                layer["inputs"][j] = renamed_tensors.get(
                    input_name, input_name)
            for param in weight_params[layer["type"]]:
                if param in layer:
                    layer[param] = weights.get_weight_index(
                        weight_indices[layer[param]])
        model["weights"] = "weights.bin"
        with open(os.path.join(output_path, os.path.basename(model_path)),
                  "wt", encoding="utf-8") as f:
            yaml.dump(model, f, Dumper=CustomYAMLFormatter, sort_keys=False)
    weights.save_weights(os.path.join(output_path, "weights.bin"))
    return 0


if __name__ == "__main__":
    try:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] %(name)s: %(message)s",
            stream=sys.stderr,
        )
        sys.exit(main(**vars(parse_args())))
    except KeyboardInterrupt:
        sys.exit(-1)
    finally:
        logging.shutdown()
