#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run inference."""

from typing import Iterator, List, Tuple
import os
import sys
import logging
import multiprocessing
from glob import glob
import argparse
import yaml
import cv2
import tvm.runtime as tvm
import numpy as np

LOG = logging.getLogger("inference")


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run inference")
    parser.add_argument("-m", "--model",
                        dest="model_path",
                        help="Model",
                        type=str,
                        required=True)
    parser.add_argument("-l", "--library",
                        dest="library_path",
                        help="Library",
                        type=str,
                        required=True)
    parser.add_argument("-d", "--device",
                        dest="device_type",
                        help="Device type",
                        type=str,
                        required=True)
    parser.add_argument("-o",
                        dest="output_dir",
                        help="Output directory",
                        type=str,
                        required=True)
    parser.add_argument("image_paths",
                        help="Input images",
                        type=str,
                        nargs="+")
    return parser.parse_args()


class Session:
    """Inference session."""

    def __init__(self, model_path: str, library_path: str,
                 device_type: str) -> None:
        """Create Session.

        Parameters
        ----------
        model_path: str
            Path to model
        library_path: str
            Path to library
        device_type: str
            Device type
        """
        with open(model_path, "rt", encoding="utf-8") as f:
            model_def = yaml.unsafe_load(f)
        self._device = tvm.device(dev_type=device_type)
        self._mod = tvm.load_module(library_path)["default"](self._device)
        self._mod_run = self._mod["run"]
        self._mod_set_input = self._mod["set_input_zero_copy"]
        self._mod_set_output = self._mod["set_output_zero_copy"]
        input_names = model_def["inputs"]
        self._output_buf = self._mod["get_output"](0)
        self._input_buf = self._mod["get_input"](input_names[0])
        self._bindings_idx = 0
        self._inter_bufs = []
        if len(input_names) == 1:
            self._bindings = [({}, []), ({}, [])]
        else:
            num_inter = len(input_names) - 1
            shape_data = self._mod["get_input_info"]()["shape"]
            mod_get_input_idx = self._mod["get_input_index"]
            input_idx = [mod_get_input_idx(x) for x in input_names[1:]]
            for _ in range(2):
                for i in range(num_inter):
                    self._inter_bufs.append(tvm.ndarray.empty(
                        shape_data[input_names[i + 1]],
                        device=self._device
                    ))
            self._bindings = [
                (
                    {
                        input_idx[i]: self._inter_bufs[i]
                        for i in range(num_inter)
                    },
                    self._inter_bufs[num_inter:]
                ),
                (
                    {
                        input_idx[i]: self._inter_bufs[i + num_inter]
                        for i in range(num_inter)
                    },
                    self._inter_bufs[:num_inter]
                )
            ]

    def _set_bindings(self) -> None:
        """Set bindings."""
        bindings = self._bindings[self._bindings_idx]
        for idx, val in bindings[0].items():
            self._mod_set_input(idx, val)
        for idx, val in enumerate(bindings[1], start=1):
            self._mod_set_output(idx, val)

    def run(self, image: np.ndarray) -> np.ndarray:
        """Run inference.

        Parameters
        ----------
        images: np.ndarray
            Input image

        Returns
        -------
        np.ndarray
            Output image
        """
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        self._input_buf.copyfrom(image)
        self._set_bindings()
        self._mod_run()
        self._bindings_idx ^= 1
        return np.squeeze(self._output_buf.numpy(), axis=0)


def list_images(image_paths: List[str]) -> Iterator[str]:
    """List image files."""
    for path in image_paths:
        for filename in glob(path, recursive=True):
            if os.path.isdir(filename):
                for subpath in list_images([os.path.join(filename, "*")]):
                    yield subpath
            else:
                yield filename


def get_data(image_paths: List[str], output_dir: str) \
        -> Iterator[Tuple[str, str]]:
    """Get input data."""
    for filename in sorted(list_images(image_paths)):
        img_name = os.path.splitext(os.path.basename(filename))[0]
        output_path = os.path.join(output_dir, f"{img_name}.png")
        yield filename, output_path


def main(
    model_path: str,
    library_path: str,
    device_type: str,
    output_dir: str,
    image_paths: List[str],
) -> int:
    """
    Run CLI.

    Parameters
    ----------
    model_path: str
        Model path
    library_path: str
        Library path
    device_type: str
        Device type
    output_dir: str
        Output directory
    image_paths: List[str]
        Input images

    Returns
    -------
    int
        Exit code
    """
    sess = Session(model_path, library_path, device_type)
    for input_path, output_path in get_data(image_paths, output_dir):
        LOG.info("Processing %s", input_path)
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        assert img is not None, f"Could not open image: {input_path}"
        img_out = sess.run(img)
        ret = cv2.imwrite(output_path, img_out)
        assert ret, f"Could not save image: {output_path}"
    return 0


if __name__ == "__main__":
    try:
        num_threads = multiprocessing.cpu_count()
        # TODO: handle HT
        # Note: TVM 0.10.0 autodetection basically is equivalent to
        #   multiprocessing.cpu_count() / 2
        # expecting CPU to always have HT
        os.environ["TVM_NUM_THREADS"] = str(num_threads)
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
