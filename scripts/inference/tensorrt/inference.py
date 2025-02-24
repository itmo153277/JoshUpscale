#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run inference."""

# pylint: disable=unused-import
# pylint: disable=no-member

from typing import Iterator, List, Sequence, Tuple
import os
import sys
import time
import logging
from glob import glob
import argparse
import struct
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # noqa
import tensorrt as trt

LOG = logging.getLogger("inference")
TRT_LOG = trt.Logger(trt.Logger.INFO)


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
    parser.add_argument("-e", "--engine",
                        dest="engine_path",
                        help="Engine",
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


def get_buf_size(shape: Sequence[int], dtype: np.dtype) -> int:
    """Get size of a buffer with shape."""
    return int(np.prod(shape)) * dtype(0).nbytes


class Session:
    """Inference session."""

    TRT_TYPES = {
        trt.DataType.UINT8: np.uint8,
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
    }

    def __init__(self, engine_path: str) -> None:
        """Create Session.

        Parameters
        ----------
        engine_path: str
            Path to engine
        """
        runtime = trt.Runtime(TRT_LOG)
        with open(engine_path, "rb") as f:
            data = f.read()
        trt_size = struct.unpack_from("<I", data, 16)[0]
        if trt_size + data[-1] + 1 != len(data):
            indices = None
        else:
            indices = list(data[trt_size:-1])
            data = data[:trt_size]
        self._engine = runtime.deserialize_cuda_engine(data)
        input_names = []
        output_names_raw = []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            tensor_mode = self._engine.get_tensor_mode(name)
            if tensor_mode == trt.TensorIOMode.INPUT:
                input_names.append(name)
            elif tensor_mode == trt.TensorIOMode.OUTPUT:
                output_names_raw.append(name)
        if indices is None:
            indices = list(range(len(input_names)))
        output_names = []
        for i in indices:
            output_names.append(output_names_raw[i])
        assert len(output_names) == len(input_names), "I/O mismatch"
        self._input_dtype = Session.TRT_TYPES[
            self._engine.get_tensor_dtype(input_names[0])]
        self._input_buf = self._malloc(input_names[0])
        output_shape = self._engine.get_tensor_shape(output_names[0])
        output_dtype = Session.TRT_TYPES[
            self._engine.get_tensor_dtype(output_names[0])]
        self._output_buf = self._malloc(output_names[0])
        self._output_buf_cpu = np.zeros(output_shape, dtype=output_dtype)
        self._inter_bufs = []
        if len(input_names) == 1:
            self._bindings = [{}, {}]
        else:
            num_inter = len(input_names) - 1
            for _ in range(2):
                for i in range(num_inter):
                    self._inter_bufs.append(self._malloc(output_names[i + 1]))
            self._bindings = [{
                **{
                    input_names[i + 1]: int(self._inter_bufs[i])
                    for i in range(num_inter)
                },
                **{
                    output_names[i + 1]: int(self._inter_bufs[num_inter + i])
                    for i in range(num_inter)
                },
            }, {
                **{
                    input_names[i + 1]: int(self._inter_bufs[num_inter + i])
                    for i in range(num_inter)
                },
                **{
                    output_names[i + 1]: int(self._inter_bufs[i])
                    for i in range(num_inter)
                },
            }]
        self._extra_bufs = []
        self._bindings_idx = 0
        self._stream = cuda.Stream()
        self._context = self._engine.create_execution_context()
        self._context.nvtx_verbosity = trt.ProfilingVerbosity.NONE
        self._context.enqueue_emits_profile = False
        self._extra_bufs = []
        for i, name in enumerate(output_names_raw):
            if i in indices:
                continue
            buf = self._malloc(name)
            self._extra_bufs.append(buf)
            self._context.set_tensor_address(name, int(buf))
        self._context.set_tensor_address(input_names[0],
                                         int(self._input_buf))
        self._context.set_tensor_address(output_names[0],
                                         int(self._output_buf))

    def _malloc(self, tensor_name: str) -> cuda.DeviceAllocation:
        """Memory allocation."""
        fmt = self._engine.get_tensor_format(tensor_name)
        assert fmt == trt.TensorFormat.LINEAR, "IO tensor format is not linear"
        shape = self._engine.get_tensor_shape(tensor_name)
        dtype = self._engine.get_tensor_dtype(tensor_name)
        size = get_buf_size(shape, Session.TRT_TYPES[dtype])
        data = cuda.mem_alloc(size)
        cuda.memset_d8(data, 0, size)
        return data

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
        inp = image.ravel().astype(self._input_dtype)
        cuda.memcpy_htod_async(self._input_buf, inp, self._stream)
        for tensor_name, tensor_addr in \
                self._bindings[self._bindings_idx].items():
            check = self._context.set_tensor_address(tensor_name, tensor_addr)
            assert check
        check = self._context.execute_async_v3(self._stream.handle)
        assert check
        cuda.memcpy_dtoh_async(self._output_buf_cpu,
                               self._output_buf, self._stream)
        self._stream.synchronize()
        self._bindings_idx ^= 1
        return np.squeeze(self._output_buf_cpu, axis=0).astype(np.uint8)


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
    engine_path: str,
    output_dir: str,
    image_paths: List[str],
) -> int:
    """
    Run CLI.

    Parameters
    ----------
    engine_path: str
        Engine path
    output_dir: str
        Output directory
    image_paths: List[str]
        Input images

    Returns
    -------
    int
        Exit code
    """
    sess = Session(engine_path)
    num_img = 0
    total_time = 0
    for input_path, output_path in get_data(image_paths, output_dir):
        LOG.info("Processing %s", input_path)
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        assert img is not None, f"Could not open image: {input_path}"
        start = time.time()
        img_out = sess.run(img)
        total_time += time.time() - start
        num_img += 1
        ret = cv2.imwrite(output_path, img_out)
        assert ret, f"Could not save image: {output_path}"
    LOG.info("average time: %fs", total_time / num_img)
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
