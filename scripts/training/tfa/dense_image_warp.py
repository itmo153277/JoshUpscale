# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Image warping using per-pixel flow vectors."""

import tensorflow as tf

# from tensorflow_addons.utils import types
from typing import Optional


@tf.function
def interpolate_bilinear(
    grid: tf.Tensor,
    query_points: tf.Tensor,
    indexing: str = "ij",
    name: Optional[str] = None,
) -> tf.Tensor:
    """Similar to Matlab's interp2 function.

    Finds values for query points on a grid using bilinear interpolation.

    Args:
      grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
      query_points: a 3-D float `Tensor` of N points with shape
        `[batch, N, 2]`.
      indexing: whether the query points are specified as row and column (ij),
        or Cartesian coordinates (xy).
      name: a name for the operation (optional).

    Returns:
      values: a 3-D `Tensor` with shape `[batch, N, channels]`

    Raises:
      ValueError: if the indexing mode is invalid, or if the shape of the
        inputs invalid.
    """
    return _interpolate_bilinear_with_checks(grid, query_points, indexing, name)


def _interpolate_bilinear_with_checks(
    grid: tf.Tensor,
    query_points: tf.Tensor,
    indexing: str,
    name: Optional[str],
) -> tf.Tensor:
    """Perform checks on inputs without tf.function decorator to avoid flakiness."""
    if indexing != "ij" and indexing != "xy":
        raise ValueError("Indexing mode must be 'ij' or 'xy'")

    grid = tf.convert_to_tensor(grid)
    query_points = tf.convert_to_tensor(query_points)
    grid_shape = tf.shape(grid)
    query_shape = tf.shape(query_points)

    with tf.control_dependencies(
        [
            tf.debugging.assert_equal(tf.rank(grid), 4, "Grid must be 4D Tensor"),
            tf.debugging.assert_greater_equal(
                grid_shape[1], 2, "Grid height must be at least 2."
            ),
            tf.debugging.assert_greater_equal(
                grid_shape[2], 2, "Grid width must be at least 2."
            ),
            tf.debugging.assert_equal(
                tf.rank(query_points), 3, "Query points must be 3 dimensional."
            ),
            tf.debugging.assert_equal(
                query_shape[2], 2, "Query points last dimension must be 2."
            ),
        ]
    ):
        return _interpolate_bilinear_impl(grid, query_points, indexing, name)


def _interpolate_bilinear_impl(
    grid: tf.Tensor,
    query_points: tf.Tensor,
    indexing: str,
    name: Optional[str],
) -> tf.Tensor:
    """tf.function implementation of interpolate_bilinear."""
    with tf.name_scope(name or "interpolate_bilinear"):
        grid_shape = tf.shape(grid)
        query_shape = tf.shape(query_points)

        batch_size, height, width, channels = (
            grid_shape[0],
            grid_shape[1],
            grid_shape[2],
            grid_shape[3],
        )

        num_queries = query_shape[1]

        query_type = query_points.dtype
        grid_type = grid.dtype

        alphas = []
        floors = []
        ceils = []
        index_order = [0, 1] if indexing == "ij" else [1, 0]
        unstacked_query_points = tf.unstack(query_points, axis=2, num=2)

        for i, dim in enumerate(index_order):
            with tf.name_scope("dim-" + str(dim)):
                queries = unstacked_query_points[dim]

                size_in_indexing_dimension = grid_shape[i + 1]

                # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                # is still a valid index into the grid.
                max_floor = tf.cast(size_in_indexing_dimension - 2, query_type)
                min_floor = tf.constant(0.0, dtype=query_type)
                floor = tf.math.minimum(
                    tf.math.maximum(min_floor, tf.math.floor(queries)), max_floor
                )
                int_floor = tf.cast(floor, tf.dtypes.int32)
                floors.append(int_floor)
                ceil = int_floor + 1
                ceils.append(ceil)

                # alpha has the same type as the grid, as we will directly use alpha
                # when taking linear combinations of pixel values from the image.
                alpha = tf.cast(queries - floor, grid_type)
                min_alpha = tf.constant(0.0, dtype=grid_type)
                max_alpha = tf.constant(1.0, dtype=grid_type)
                alpha = tf.math.minimum(tf.math.maximum(min_alpha, alpha), max_alpha)

                # Expand alpha to [b, n, 1] so we can use broadcasting
                # (since the alpha values don't depend on the channel).
                alpha = tf.expand_dims(alpha, 2)
                alphas.append(alpha)

            flattened_grid = tf.reshape(grid, [batch_size * height * width, channels])
            batch_offsets = tf.reshape(
                tf.range(batch_size) * height * width, [batch_size, 1]
            )

        # This wraps tf.gather. We reshape the image data such that the
        # batch, y, and x coordinates are pulled into the first dimension.
        # Then we gather. Finally, we reshape the output back. It's possible this
        # code would be made simpler by using tf.gather_nd.
        def gather(y_coords, x_coords, name):
            with tf.name_scope("gather-" + name):
                linear_coordinates = batch_offsets + y_coords * width + x_coords
                gathered_values = tf.gather(flattened_grid, linear_coordinates)
                return tf.reshape(gathered_values, [batch_size, num_queries, channels])

        # grab the pixel values in the 4 corners around each query point
        top_left = gather(floors[0], floors[1], "top_left")
        top_right = gather(floors[0], ceils[1], "top_right")
        bottom_left = gather(ceils[0], floors[1], "bottom_left")
        bottom_right = gather(ceils[0], ceils[1], "bottom_right")

        # now, do the actual interpolation
        with tf.name_scope("interpolate"):
            interp_top = alphas[1] * (top_right - top_left) + top_left
            interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
            interp = alphas[0] * (interp_bottom - interp_top) + interp_top

        return interp


def _get_dim(x, idx):
    if x.shape.ndims is None:
        return tf.shape(x)[idx]
    return x.shape[idx] or tf.shape(x)[idx]


@tf.function
def dense_image_warp(
    image: tf.Tensor, flow: tf.Tensor, name: Optional[str] = None
) -> tf.Tensor:
    """Image warping using per-pixel flow vectors.

    Apply a non-linear warp to the image, where the warp is specified by a
    dense flow field of offset vectors that define the correspondences of
    pixel values in the output image back to locations in the source image.
    Specifically, the pixel value at `output[b, j, i, c]` is
    `images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c]`.

    The locations specified by this formula do not necessarily map to an int
    index. Therefore, the pixel value is obtained by bilinear
    interpolation of the 4 nearest pixels around
    `(b, j - flow[b, j, i, 0], i - flow[b, j, i, 1])`. For locations outside
    of the image, we use the nearest pixel values at the image boundary.

    NOTE: The definition of the flow field above is different from that
    of optical flow. This function expects the negative forward flow from
    output image to source image. Given two images `I_1` and `I_2` and the
    optical flow `F_12` from `I_1` to `I_2`, the image `I_1` can be
    reconstructed by `I_1_rec = dense_image_warp(I_2, -F_12)`.

    Args:
      image: 4-D float `Tensor` with shape `[batch, height, width, channels]`.
      flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
      name: A name for the operation (optional).

      Note that image and flow can be of type `tf.half`, `tf.float32`, or
      `tf.float64`, and do not necessarily have to be the same type.

    Returns:
      A 4-D float `Tensor` with shape`[batch, height, width, channels]`
        and same type as input image.

    Raises:
      ValueError: if `height < 2` or `width < 2` or the inputs have the wrong
        number of dimensions.
    """
    with tf.name_scope(name or "dense_image_warp"):
        image = tf.convert_to_tensor(image)
        flow = tf.convert_to_tensor(flow)
        batch_size, height, width, channels = (
            _get_dim(image, 0),
            _get_dim(image, 1),
            _get_dim(image, 2),
            _get_dim(image, 3),
        )

        # The flow is defined on the image grid. Turn the flow into a list of query
        # points in the grid space.
        grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
        stacked_grid = tf.cast(tf.stack([grid_y, grid_x], axis=2), flow.dtype)
        batched_grid = tf.expand_dims(stacked_grid, axis=0)
        query_points_on_grid = batched_grid - flow
        query_points_flattened = tf.reshape(
            query_points_on_grid, [batch_size, height * width, 2]
        )
        # Compute values at the query points, then reshape the result back to the
        # image grid.
        interpolated = interpolate_bilinear(image, query_points_flattened)
        interpolated = tf.reshape(interpolated, [batch_size, height, width, channels])
        return interpolated


@tf.function(experimental_implements="addons:DenseImageWarp")
def dense_image_warp_annotated(
    image: tf.Tensor, flow: tf.Tensor
) -> tf.Tensor:
    """Similar to dense_image_warp but annotated with experimental_implements.

    IMPORTANT: This is a temporary function and will be removed after TensorFlow's
    next release.

    This annotation make the serialized function detectable by the TFLite MLIR
    converter and allow the converter to convert it to corresponding TFLite op.

    However, with the annotation, this function cannot be used with backprop
    under `tf.GradientTape` objects.
    """
    return dense_image_warp(image, flow)
