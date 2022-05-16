#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run inference."""

from typing import Iterator, List, Tuple
import os
import sys
import logging
from glob import glob
import argparse
import cv2
import numpy as np
import onnxruntime as rt

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
                        help="Model",
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

    ORT_TYPES = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
    }

    def __init__(self, model: str) -> None:
        """Create Session.

        Parameters
        ----------
        model: str
            Path to model
        """
        self.sess = rt.InferenceSession(model)
        inputs = self.sess.get_inputs()
        self.inp_name = inputs[0].name
        self.states = {x.name: np.zeros(
            shape=x.shape,
            dtype=Session.ORT_TYPES[x.type]
        ) for x in inputs[1:]}

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
        inp_dict = {
            self.inp_name: image,
            **self.states
        }
        out = self.sess.run(None, inp_dict)
        for val, name in zip(out[1:], self.states.keys()):
            self.states[name] = val
        return np.squeeze(out[0], axis=0)


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
    model: str,
    output_dir: str,
    image_paths: List[str],
) -> int:
    """
    Run CLI.

    Parameters
    ----------
    model: str
        Model
    output_dir: str
        Output directory
    image_paths: List[str]
        Input images

    Returns
    -------
    int
        Exit code
    """
    sess = Session(model)
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
