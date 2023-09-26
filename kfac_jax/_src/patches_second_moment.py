# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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
"""K-FAC optimized functions for patches second moment(PSM) computation."""
import functools
from typing import Optional, Sequence, TypeVar, Tuple, Union, List

import jax
from jax import interpreters
from jax import lax
import jax.numpy as jnp

from kfac_jax._src import utils

# Types for annotation
T = TypeVar("T")
Array = utils.Array
Shape = utils.Shape
TracedType = interpreters.partial_eval.DynamicJaxprTracer
DimNumbers = Tuple[Shape, Shape, Shape]
PaddingVariants = Union[str, int, Sequence[int], Sequence[Tuple[int, int]]]

# Special global variables
_USE_4D_CONVOLUTION: bool = True


def set_use_4d_convolution_in_psm_loop(value: bool):
  """Sets whether a 4D convolution is used for the PSM computation."""
  if not isinstance(value, bool):
    raise ValueError("The value provided must be a python bool.")
  global _USE_4D_CONVOLUTION
  _USE_4D_CONVOLUTION = value


def get_use_4d_convolution_in_psm_loop() -> bool:
  """Returns whether a 4D convolution is used for the PSM computation."""
  return _USE_4D_CONVOLUTION


def _ceil(x: int, y: int) -> int:
  """Computes `ceil(x / y)` with only integer operations."""
  return - (- x // y)


class _ConvSpec:
  """Layout specification for arrays that will be used in a convolution."""

  def __init__(self, order: Sequence[int]):
    """Initializes the array specification with the provided order."""
    self.order = tuple(order)

  def __len__(self):
    return len(self.order)

  @property
  def n_axis(self) -> int:
    """Returns the index of the batch axis."""
    return self.order[0]

  @property
  def c_axis(self) -> int:
    """Returns the index of the channel axis."""
    return self.order[1]

  @property
  def spatial_axes(self) -> Tuple[int]:
    """Returns the indices of the spatial axes."""
    return self.order[2:]

  def get_n(self, shape: Shape) -> int:
    """Returns the batch size of the given shape, under this spec layout."""
    return shape[self.n_axis]

  def get_c(self, shape: Shape) -> int:
    """Returns the channel size of the given shape, under this spec layout."""
    return shape[self.c_axis]

  def get_spatial(self, shape: Shape) -> Tuple[int, ...]:
    """Returns the spatial sizes of the given shape, under this spec layout."""
    return tuple(shape[i] for i in self.spatial_axes)

  def expand_spatial_axes(self) -> "_ConvSpec":
    """Expands the layout spatial axes by preserving `n` and `c` order."""
    n_axis = self.n_axis + sum(self.n_axis > axis for axis in self.spatial_axes)
    c_axis = self.c_axis + sum(self.c_axis > axis for axis in self.spatial_axes)
    spatial_axes = []
    for axis in self.spatial_axes:
      spatial_axes.append(axis + sum(axis > a for a in self.spatial_axes))
      spatial_axes.append(spatial_axes[-1] + 1)
    return _ConvSpec([n_axis, c_axis, *spatial_axes])

  def swap_n_and_c(self) -> "_ConvSpec":
    """Swaps the batch and channel indices of the layout."""
    return _ConvSpec([self.c_axis, self.n_axis, *self.spatial_axes])

  def create_shape(self, n: T, c: T, *spatial_dims: T) -> Tuple[T, ...]:
    """Creates a shape according to this layout specification."""
    if len(spatial_dims) != len(self.order) - 2:
      raise ValueError("Incorrect number of spatial dimensions.")
    result: List[T] = [None] * len(self)  # pytype: disable=annotation-type-mismatch
    result[self.n_axis] = n
    result[self.c_axis] = c
    for ax, dim in zip(self.spatial_axes, spatial_dims):
      result[ax] = dim
    assert all(r is not None for r in result)
    return tuple(result)

  def change_nhwc_to_ihwo(self) -> "_ConvSpec":
    """Changes the layout from `NHWC` to `IHWO` where `I=C`, `O=N`."""
    # Change the spec: NHWC -> IHWO where I=C, O=N
    order = [i - 2 if i > self.spatial_axes[1] else i for i in self.order[:4]]
    return _ConvSpec(order).swap_n_and_c()


def _slice_array(
    array: Array,
    indices: Sequence[Union[int, TracedType]],
    sizes: Sequence[int],
) -> Array:
  """Takes a slice from the array provided."""
  if any(isinstance(x, TracedType) for x in indices):
    # Any of the indices are dynamic values.
    return lax.dynamic_slice_p.bind(array, *indices, slice_sizes=sizes)
  else:
    # All indices are static values.
    index = tuple(slice(i, i + size) for i, size in zip(indices, sizes))
    return array[index]


def _output_spatial_shape(
    inputs_spatial_shape: Shape,
    kernel_spatial_shape: Shape,
    spatial_strides: Shape,
    padding: Union[str, Sequence[Tuple[int, int]]],
) -> Shape:
  """Returns the output spatial shape of the corresponding convolution."""
  if isinstance(padding, str):
    if padding.lower() == "valid":
      return tuple(_ceil(d - k + 1, s) for d, k, s in zip(inputs_spatial_shape,
                                                          kernel_spatial_shape,
                                                          spatial_strides))
    elif padding.lower() == "same":
      return tuple(_ceil(d, s) for d, s in zip(inputs_spatial_shape,
                                               spatial_strides))
    else:
      raise ValueError(f"Unrecognized padding string {padding}!")
  else:
    shapes_strides_padding = zip(
        inputs_spatial_shape, kernel_spatial_shape, spatial_strides, padding)
    return tuple(_ceil(d + p[0] + p[1] - k + 1, s)
                 for d, k, s, p in shapes_strides_padding)


def _normalize_padding(
    inputs_spatial_shape: Shape,
    kernel_spatial_shape: Shape,
    spatial_strides: Shape,
    padding: PaddingVariants,
) -> Tuple[Tuple[int, int], ...]:
  """Returns the padding as a tuple of pairs of integers."""
  n = len(kernel_spatial_shape)
  if isinstance(padding, str):
    if padding.lower() == "valid":
      return ((0, 0),) * n
    elif padding.lower() == "same":
      # https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/core/kernels/conv_ops.cc#L571
      output_shape = _output_spatial_shape(inputs_spatial_shape,
                                           kernel_spatial_shape,
                                           spatial_strides, "same")
      padding = []
      for out_d, d, k, s in zip(output_shape, inputs_spatial_shape,
                                kernel_spatial_shape, spatial_strides):
        pad = max(0, (out_d - 1) * s + k - d)
        padding.append((pad // 2, pad - pad // 2))
      return tuple(padding)
    else:
      raise ValueError(f"Unrecognized padding: {padding}!")
  elif isinstance(padding, int):
    return ((padding, padding),) * n
  else:
    final_padding = []
    for pad in padding:
      if isinstance(pad, int):
        final_padding.append((pad, pad))
      else:
        final_padding.append(pad)
    return tuple(final_padding)


def _normalize_strides(
    kernel_spatial_shape: Shape,
    strides: Union[int, Shape],
) -> Tuple[int, ...]:
  """Returns the strides as a tuple of integers."""
  n = len(kernel_spatial_shape)
  if strides is None:
    return (1,) * n
  elif isinstance(strides, int):
    return (strides,) * n
  else:
    assert len(strides) == n
    return tuple(strides)


def _data_format_to_dim_numbers(
    data_format: Optional[str],
    kernel_format: str = "HWIO",
) -> lax.ConvDimensionNumbers:
  """Converts the data format in dim numbers."""
  if data_format is None:
    data_format = "NHWC"
  if not isinstance(data_format, str):
    raise ValueError("data_format must be either a python string or `None`.")
  data_format = lax.conv_general_permutations([data_format,
                                               kernel_format,
                                               data_format])
  return lax.ConvDimensionNumbers(*data_format)


def _parse_simple_args(
    inputs_shape: Shape,
    kernel_spatial_shape: Union[int, Shape],
    strides: Union[int, Shape] = 1,
    padding: PaddingVariants = "VALID",
    data_format: Optional[str] = "NHWC",
    dim_numbers: Optional[Union[DimNumbers, lax.ConvDimensionNumbers]] = None,
) -> Tuple[
    Tuple[int, ...],
    Tuple[int, ...],
    Tuple[Tuple[int, int], ...],
    lax.ConvDimensionNumbers,
]:
  """Parses all convolutional arguments to a single unified format.

  Args:
    inputs_shape: A sequence of ints specifying the input's shape.
    kernel_spatial_shape: A sequence of ints specifying the kernel's shape.
    strides: A sequence of ints specifying strides in each spatial dimension,
      or a single int specifying the strides in every spatial dimension.
    padding: The padding can take one of the following formats:
      * str - Either 'VALID' or 'SAME'
      * int - Specifies the padding on both of sides of every spatial dimension.
      * sequence of ints - Specifies the padding on both sides of each spatial
        dimension.
      * sequence of pairs of ints - Specifies the padding on each side of each
        spatial dimension.
    data_format: The data format layout of the inputs.
    dim_numbers: If `data_format` is `None` this can specify the layout instead.

  Returns:
    A tuple of the (kernel shape, strides, padding, dim_numbers)
  """
  spatial_dims = len(inputs_shape) - 2

  if data_format is not None and dim_numbers is not None:
    raise ValueError("At least one of `data_format` and `dim_numbers` "
                     "must be None.")

  if dim_numbers is not None:

    if not isinstance(dim_numbers, lax.ConvDimensionNumbers):

      if not isinstance(dim_numbers, (list, tuple)):
        raise ValueError("The provided dim_numbers argument must be either a "
                         "list, tuple or lax.ConvDimensionNumbers.")

      if len(dim_numbers) != 3:
        raise ValueError("When the provided dim_numbers argument is a list or "
                         "tuple it must have length 3, but has length "
                         f"{len(dim_numbers)}.")

      lax_dim_numbers = lax.ConvDimensionNumbers(*dim_numbers)

    else:
      lax_dim_numbers: lax.ConvDimensionNumbers = dim_numbers

  else:
    lax_dim_numbers = _data_format_to_dim_numbers(data_format)

  if isinstance(kernel_spatial_shape, int):
    kernel_spatial_shape = (kernel_spatial_shape,) * spatial_dims

  if len(kernel_spatial_shape) != spatial_dims:
    raise ValueError("The provided argument `kernel_spatial_shape` must have "
                     f"length equal to the spatial dimensions {spatial_dims} of"
                     f" the inputs, but got {len(kernel_spatial_shape)}.")

  inputs_spatial_shape = _ConvSpec(lax_dim_numbers.lhs_spec).get_spatial(
      inputs_shape)

  kernel_spatial_shape = _ConvSpec(lax_dim_numbers.rhs_spec).get_spatial(
      kernel_spatial_shape)
  strides = _normalize_strides(kernel_spatial_shape, strides)

  padding = _normalize_padding(
      inputs_spatial_shape, kernel_spatial_shape, strides, padding)

  return kernel_spatial_shape, strides, padding, lax_dim_numbers


def _num_conv_locations_full_spec(
    input_spatial_shape: Shape,
    kernel_spatial_shape: Shape,
    spatial_strides: Shape,
    spatial_padding: Sequence[Tuple[int, int]],
) -> int:
  """The number of convolution locations from the unified spec for arguments."""
  if len(kernel_spatial_shape) != len(input_spatial_shape):
    raise ValueError("The `kernel_spatial_shape` and `input_spatial_shape` "
                     "must have the same number of elements, got "
                     f"{len(kernel_spatial_shape)} and "
                     f"{len(input_spatial_shape)}.")
  if len(spatial_strides) != len(input_spatial_shape):
    raise ValueError("The `spatial_strides` and `input_spatial_shape` "
                     "must have the same number of elements, got "
                     f"{len(spatial_strides)} and "
                     f"{len(input_spatial_shape)}.")
  if len(spatial_padding) != len(input_spatial_shape):
    raise ValueError("The `spatial_padding` and `input_spatial_shape` "
                     "must have the same number of elements, got "
                     f"{len(spatial_padding)} and "
                     f"{len(input_spatial_shape)}.")

  num_locations = 1
  for in_dim, k_dim, stride, padding in zip(
      input_spatial_shape, kernel_spatial_shape,
      spatial_strides, spatial_padding):
    num_locations *= _ceil(in_dim + padding[0] + padding[1] - k_dim + 1, stride)
  return num_locations


def num_conv_locations(
    inputs_spatial_shape: Shape,
    kernel_spatial_shape: Union[int, Shape],
    spatial_strides: Union[int, Shape],
    spatial_padding: Union[str, int, Sequence[Tuple[int, int]]],
) -> int:
  """Returns the number of convolution locations for the provided shapes."""
  inputs_spatial_shape = tuple(inputs_spatial_shape)
  n = len(inputs_spatial_shape)
  if isinstance(kernel_spatial_shape, int):
    kernel_spatial_shape = (kernel_spatial_shape,) * n
  spatial_strides = _normalize_strides(kernel_spatial_shape, spatial_strides)
  spatial_padding = _normalize_padding(
      inputs_spatial_shape, kernel_spatial_shape,
      spatial_strides, spatial_padding)
  return _num_conv_locations_full_spec(
      inputs_spatial_shape, kernel_spatial_shape,
      spatial_strides, spatial_padding)


@utils.auto_scope_function
def _the_conv4d(
    lhs: Array,
    lhs_spec: _ConvSpec,
    rhs: Array,
    rhs_spec: _ConvSpec,
    pad_h: int,
    pad_w: int,
    stride_h: int,
    stride_w: int,
    per_channel: bool = False,
    precision: Optional[jax.lax.Precision] = None,
) -> Array:
  """Performs a special conv4d or conv2d based on the global flag."""
  assert len(rhs_spec) == 6
  if get_use_4d_convolution_in_psm_loop():
    # Reshape lhs to 6D array - (n, extra_h, 1, extra_w, 1, c)
    lhs_shape = list(lhs.shape)
    lhs_shape.insert(lhs_spec.spatial_axes[1] + 1, 1)
    lhs_shape.insert(lhs_spec.spatial_axes[0] + 1, 1)
    lhs = jnp.reshape(lhs, lhs_shape)
    # Change the spec: NHAWBC -> CHAWBN
    lhs_spec = rhs_spec.swap_n_and_c()
    # Change the spec: NHAWBC -> IHAWBO where I=C, O=N
    rhs_spec = rhs_spec.swap_n_and_c()
    dim_specs = (lhs_spec.order, rhs_spec.order, lhs_spec.order)
    if per_channel:
      @functools.partial(jax.vmap,
                         in_axes=(lhs_spec.n_axis, rhs_spec.n_axis),
                         out_axes=-1)
      def single_conv(x, y):
        return lax.conv_general_dilated(
            lhs=jnp.expand_dims(x, lhs_spec.n_axis),
            rhs=jnp.expand_dims(y, rhs_spec.n_axis),
            window_strides=(1, 1, 1, 1),
            padding=((0, pad_h), (0, 0), (0, pad_w), (0, 0)),
            lhs_dilation=(1, 1, 1, 1),
            rhs_dilation=(stride_h, 1, stride_w, 1),
            dimension_numbers=lax.ConvDimensionNumbers(*dim_specs),
            precision=precision,
        )

      result = single_conv(lhs, rhs)
      assert result.shape[lhs_spec.n_axis] == 1
      result = jnp.squeeze(result, lhs_spec.n_axis)
      assert result.shape[2] == 1
      assert result.shape[4] == 1
      result = jnp.squeeze(result, (2, 4))
      return result[None]
    else:
      result = lax.conv_general_dilated(
          lhs=lhs,
          rhs=rhs,
          window_strides=(1, 1, 1, 1),
          padding=((0, pad_h), (0, 0), (0, pad_w), (0, 0)),
          lhs_dilation=(1, 1, 1, 1),
          rhs_dilation=(stride_h, 1, stride_w, 1),
          dimension_numbers=lax.ConvDimensionNumbers(*dim_specs),
          precision=precision,
      )
      # Order the result such that one of the channel dims is after spatial dims
      if lhs_spec != (5, 0, 1, 2, 3, 4):
        min_index = 0 if lhs_spec.n_axis < lhs_spec.c_axis else 1
        max_index = 1 - min_index
        axes = list(range(6))
        if lhs_spec.order[min_index] != 0:
          axes.insert(0, axes.pop(lhs_spec.order[min_index]))
        if lhs_spec.order[max_index] != 5:
          axes.insert(5, axes.pop(lhs_spec.order[max_index]))
        result = jnp.transpose(result, axes=axes)
      assert result.shape[2] == 1
      assert result.shape[4] == 1
      result = jnp.squeeze(result, (2, 4))
      return result[None, None]
  else:
    # Change the spec: NHWC -> CHWN
    lhs_spec = lhs_spec.swap_n_and_c()
    # Index rhs and remove the trivial dimensions
    rhs_slice: List[Union[slice, int]] = [slice(None)] * rhs.ndim
    rhs_slice[rhs_spec.spatial_axes[1]] = 0
    rhs_slice[rhs_spec.spatial_axes[3]] = 0
    rhs = rhs[tuple(rhs_slice)]
    rhs_spec = rhs_spec.change_nhwc_to_ihwo()
    dim_specs = (lhs_spec.order, rhs_spec.order, lhs_spec.order)
    if per_channel:
      vmap_single_conv = jax.vmap(lambda x, y: lax.conv_general_dilated(  # pylint: disable=g-long-lambda
          lhs=jnp.expand_dims(x, lhs_spec.n_axis),
          rhs=jnp.expand_dims(y, rhs_spec.n_axis),
          window_strides=(1, 1),
          padding=((0, pad_h), (0, pad_w)),
          lhs_dilation=(1, 1),
          rhs_dilation=(stride_h, stride_w),
          dimension_numbers=lax.ConvDimensionNumbers(*dim_specs),
          precision=precision,
      ), in_axes=(lhs_spec.n_axis, rhs_spec.n_axis), out_axes=-1)
      result = vmap_single_conv(lhs, rhs)
      assert result.shape[lhs_spec.n_axis] == 1
      result = jnp.squeeze(result, lhs_spec.n_axis)
      return result[None]
    else:
      result = lax.conv_general_dilated(
          lhs=lhs,
          rhs=rhs,
          window_strides=(1, 1),
          padding=((0, pad_h), (0, pad_w)),
          lhs_dilation=(1, 1),
          rhs_dilation=(stride_h, stride_w),
          dimension_numbers=lax.ConvDimensionNumbers(*dim_specs),
          precision=precision,
      )
      # Order the result such that one of the channel dims is after spatial dims
      if lhs_spec != (3, 0, 1, 2):
        min_index = 0 if lhs_spec.n_axis < lhs_spec.c_axis else 1
        max_index = 1 - min_index
        axes = list(range(4))
        if lhs_spec.order[min_index] != 0:
          axes.insert(0, axes.pop(lhs_spec.order[min_index]))
        if lhs_spec.order[max_index] != 5:
          axes.insert(5, axes.pop(lhs_spec.order[max_index]))
        result = jnp.transpose(result, axes=axes)
      return result[None, None]


def _validate_inputs_lengths(
    inputs: Array,
    kernel_spatial_shape: Shape,
    strides: Shape,
    padding: Tuple[Tuple[int, int], ...],
) -> None:
  """Checks that the provided arguments are valid."""
  spatial_dims = inputs.ndim - 2
  if spatial_dims != 2:
    raise ValueError("Currently `patches_second_moment` supports only 2D "
                     "convolution, hence the input is expected to have rank 4,"
                     f" but has rank {inputs.ndim}.")
  if len(kernel_spatial_shape) != spatial_dims:
    raise ValueError("The argument `kernel_spatial_shape` must have length "
                     f"equal to the number of spatial dimensions of the input -"
                     f" {spatial_dims}, but instead has length "
                     f"{len(kernel_spatial_shape)}.")
  if len(padding) != spatial_dims:
    raise ValueError("The argument `padding` must have length equal to the "
                     "number of spatial dimensions of the input - "
                     f"{spatial_dims}, but instead has length "
                     f"{len(kernel_spatial_shape)}.")
  if len(strides) != 2:
    raise ValueError("The argument `strides` must have length equal to the "
                     "number of spatial dimensions of the input - "
                     f"{spatial_dims}, but instead has length "
                     f"{len(kernel_spatial_shape)}.")


@functools.partial(jax.jit, static_argnums=list(range(1, 12)),
                   static_argnames=(
                       "kernel_spatial_shape", "strides", "padding",
                       "data_format", "dim_numbers", "inputs_dilation",
                       "kernel_dilation", "feature_group_count",
                       "batch_group_count", "unroll_loop", "precision"))
@utils.auto_scope_function
def patches_moments_explicit(
    inputs: Array,
    kernel_spatial_shape: Union[int, Shape],
    strides: Union[int, Shape] = 1,
    padding: PaddingVariants = "VALID",
    data_format: Optional[str] = "NHWC",
    dim_numbers: Optional[Union[DimNumbers, lax.ConvDimensionNumbers]] = None,
    inputs_dilation: Optional[Sequence[int]] = None,
    kernel_dilation: Optional[Sequence[int]] = None,
    feature_group_count: int = 1,
    batch_group_count: int = 1,
    unroll_loop: bool = False,
    precision: Optional[jax.lax.Precision] = None,
    weighting_array: Optional[Array] = None,
) -> Tuple[Array, Array]:
  """The exact same functionality as :func:`~patches_moments`, but explicitly extracts the patches via :func:`jax.lax.conv_general_dilated_patches`, potentially having a higher memory usage."""
  kernel_spatial_shape, strides, padding, dim_numbers = _parse_simple_args(
      inputs.shape, kernel_spatial_shape, padding=padding, strides=strides,
      data_format=data_format, dim_numbers=dim_numbers)
  _validate_inputs_lengths(inputs, kernel_spatial_shape, strides, padding)

  in_spec = _ConvSpec(dim_numbers.lhs_spec)
  out_spec = _ConvSpec(dim_numbers.out_spec)
  n = in_spec.get_n(inputs.shape)
  c = in_spec.get_c(inputs.shape)
  inputs_spatial_shape = in_spec.get_spatial(inputs.shape)
  spec = _ConvSpec(dim_numbers.out_spec).swap_n_and_c().order
  matmul_dim_numbers = lax.ConvDimensionNumbers(spec, spec, spec)

  if feature_group_count not in (1, in_spec.get_c(inputs.shape)):
    raise ValueError("`patches_moments_explicit` does not support "
                     "`feature_group_count` different from 1 or the number of "
                     "channels of the inputs.")
  if batch_group_count != 1:
    raise ValueError("`patches_moments_explicit` does not support "
                     "`batch_group_count` different from 1.")

  per_channel = feature_group_count != 1
  vector_target_shape = kernel_spatial_shape + (c,)
  leading_shape = kernel_spatial_shape if per_channel else vector_target_shape
  matrix_target_shape = leading_shape + vector_target_shape
  vector_axis = tuple(a for a in range(4) if a != out_spec.c_axis)

  # Broadcast the weighting function
  if weighting_array is not None:
    if weighting_array.ndim == inputs.ndim:
      pass
    elif weighting_array.ndim == inputs.ndim - 1:
      axis = dim_numbers.lhs_spec[1]
      weighting_array = jnp.expand_dims(weighting_array, axis=axis)
    elif weighting_array.ndim == 1:
      while weighting_array.ndim < inputs.ndim:
        weighting_array = weighting_array[:, None]
    else:
      raise ValueError(f"`weighting_array` shape {weighting_array.shape} is "
                       f"not compatible with the inputs shape {inputs.shape}"
                       ".")

  if not per_channel:
    vector_shape = (c,) + kernel_spatial_shape
    matrix_shape = vector_shape + vector_shape
    if weighting_array is None:
      weighting_array = jnp.ones([], dtype=inputs.dtype)

    # Standard explicit patches calculation
    extracted_patches = lax.conv_general_dilated_patches(
        inputs,
        filter_shape=kernel_spatial_shape,
        window_strides=strides,
        padding=padding,
        lhs_dilation=inputs_dilation,
        rhs_dilation=kernel_dilation,
        dimension_numbers=dim_numbers,
        precision=precision,
    )

    weighted_patches = extracted_patches * weighting_array
    matrix_results = lax.conv_general_dilated(
        extracted_patches,
        weighted_patches,
        window_strides=strides,
        padding="VALID",
        dimension_numbers=matmul_dim_numbers,
        precision=precision,
    )
    matrix_results = jnp.reshape(matrix_results, matrix_shape)
    vector_results = jnp.reshape(
        jnp.sum(weighted_patches, axis=vector_axis), vector_shape)

    if c > 1:
      # The output of `conv_general_dilated_patches` is ordered `chw`
      return (jnp.transpose(matrix_results, (1, 2, 0, 4, 5, 3)),
              jnp.transpose(vector_results, [1, 2, 0]))
    else:
      return (jnp.reshape(matrix_results, matrix_target_shape),
              jnp.reshape(vector_results, vector_target_shape))

  # Loop over channels
  def general_loop_body(i, image):
    index = in_spec.create_shape(0, i, 0, 0)
    sizes = in_spec.create_shape(n, 1, *inputs_spatial_shape)
    image_channel = _slice_array(image, index, sizes)

    # Index the weighting function
    if weighting_array is not None:
      if weighting_array.shape[in_spec.c_axis] == 1:
        wf_i = weighting_array
      else:
        wf_n = weighting_array[in_spec.n_axis]
        wf_spatial = [weighting_array.shape[a] for a in in_spec.spatial_axes]
        wf_sizes = in_spec.create_shape(wf_n, jnp.ones([]), *wf_spatial)  # pytype: disable=wrong-arg-types  # jnp-type
        wf_i = _slice_array(weighting_array, index, wf_sizes)
    else:
      wf_i = None

    matrix, vector = patches_moments_explicit(
        image_channel,
        kernel_spatial_shape=kernel_spatial_shape,
        strides=strides,
        padding=padding,
        data_format=None,
        dim_numbers=dim_numbers,
        precision=precision,
        weighting_array=wf_i,
    )
    return jnp.squeeze(matrix, axis=2), vector

  if unroll_loop:
    results = [general_loop_body(ii, inputs) for ii in range(c)]
    matrix_results, vector_results = zip(*results)
    matrix_results = jnp.concatenate(matrix_results, axis=-1)
    vector_results = jnp.concatenate(vector_results, axis=-1)
    return matrix_results, vector_results

  def loop_cond(args):
    return args[0] < c

  def loop_body(args):

    i, image, matrix_result, vector_result = args

    matrix_update, vector_update = general_loop_body(i, image)

    matrix_result = lax.dynamic_update_slice(
        matrix_result, matrix_update, (0, 0, 0, 0, i))

    vector_result = lax.dynamic_update_slice(
        vector_result, vector_update, (0, 0, i))

    return i + 1, image, matrix_result, vector_result

  init_vals = (0, inputs,
               jnp.zeros(matrix_target_shape, dtype=inputs.dtype),
               jnp.zeros(vector_target_shape, dtype=inputs.dtype))

  return lax.while_loop(loop_cond, loop_body, init_vals)[-2:]


@functools.partial(jax.jit, static_argnums=list(range(1, 12)),
                   static_argnames=(
                       "kernel_spatial_shape", "strides", "padding",
                       "data_format", "dim_numbers", "inputs_dilation",
                       "kernel_dilation", "feature_group_count",
                       "batch_group_count", "unroll_loop", "precision"))
@utils.auto_scope_function
def patches_moments(
    inputs: Array,
    kernel_spatial_shape: Union[int, Shape],
    strides: Union[int, Shape] = 1,
    padding: PaddingVariants = "VALID",
    data_format: Optional[str] = "NHWC",
    dim_numbers: Optional[Union[DimNumbers, lax.ConvDimensionNumbers]] = None,
    inputs_dilation: Optional[Sequence[int]] = None,
    kernel_dilation: Optional[Sequence[int]] = None,
    feature_group_count: int = 1,
    batch_group_count: int = 1,
    unroll_loop: bool = False,
    precision: Optional[jax.lax.Precision] = None,
    weighting_array: Optional[Array] = None,
) -> Tuple[Array, Array]:
  """Computes the first and second moment of the convolutional patches.

  Since the code is written to support arbitrary convolution data formats, e.g.
  both NHWC and NCHW, in comments above any of the procedures is written the
  simplified version of what the statements below do, if the data format
  was fixed to NHWC.

  Args:
    inputs: The batch of images.
    kernel_spatial_shape: The spatial dimensions of the filter (int or list of
      ints).
    strides: The spatial dimensions of the strides (int or list of ints).
    padding: The padding (str or list of pairs of ints).
    data_format: The data format of the inputs (None, NHWC, NCHW).
    dim_numbers: Instance of :class:`jax.lax.ConvDimensionNumbers` instead of
      data_format.
    inputs_dilation: An integer or sequence of integers, specifying the dilation
      for the image. Currently, `patches_moments` does not support dilation, so
      the only allowed values are `None, 1, (1,1)`.
    kernel_dilation: An integer or sequence of integers, specifying the dilation
      for the kernel. Currently, `patches_moments` does not support dilation, so
      the only allowed values are `None, 1, (1,1)`.
    feature_group_count: The feature grouping for grouped convolutions.
      Currently, `patches_moments` supports only 1 and number of channels of the
      inputs.
    batch_group_count: The batch grouping for grouped convolutions. Currently,
      `patches_moments` supports only 1.
    unroll_loop: Whether to unroll the loop in python.
    precision: In what precision to run the computation. For more details please
      read Jax documentation of :func:`jax.lax.conv_general_dilated`.
    weighting_array: A tensor specifying additional weighting of each element
      of the moment's average.

  Returns:
    The matrix of the patches' second and first moment as a pair. The tensor of
    the patches' second moment has a shape `kernel_spatial_shape + (, channels)
    + kernel_spatial_shape + (, channels)`. The tensor of the patches' first
    moment has a shape `kernel_spatial_shape + (, channels)`.
  """
  kernel_spatial_shape, strides, padding, dim_numbers = _parse_simple_args(
      inputs.shape, kernel_spatial_shape, padding=padding, strides=strides,
      data_format=data_format, dim_numbers=dim_numbers)
  _validate_inputs_lengths(inputs, kernel_spatial_shape, strides, padding)

  # Extract useful fixed integer values from the inputs
  in_spec = _ConvSpec(dim_numbers.lhs_spec)
  rhs_spec = _ConvSpec(dim_numbers.rhs_spec)
  inputs_spatial_shape = in_spec.get_spatial(inputs.shape)
  n = in_spec.get_n(inputs.shape)
  c = in_spec.get_c(inputs.shape)
  in_h, in_w = inputs_spatial_shape
  ker_h, ker_w = kernel_spatial_shape
  pad_h, pad_w = padding
  s_h, s_w = strides

  if inputs_dilation not in (None, 1, (1, 1)):
    raise ValueError("`patches_second_moment` does not support input dilation.")
  if kernel_dilation not in (None, 1, (1, 1)):
    raise ValueError("`patches_second_moment` does not support kernel "
                     "dilation.")
  if feature_group_count not in (1, in_spec.get_c(inputs.shape)):
    raise ValueError("`patches_second_moment` does not support "
                     "`feature_group_count` different from 1 or the number of "
                     "channels of the inputs.")
  if batch_group_count != 1:
    raise ValueError("PSM does not support `batch_group_count` different from "
                     "1.")
  per_channel = feature_group_count != 1

  # Sanity check
  if in_h + pad_h[0] + pad_h[1] < ker_h or in_w + pad_w[0] + pad_w[1] < ker_w:
    padded_h = in_h + pad_h[0] + pad_h[1]
    padded_w = in_w + pad_w[0] + pad_w[1]
    raise ValueError("The provided image has spatial padded shape "
                     f"({padded_h}, {padded_w}) while the kernel has a larger "
                     f"shape ({ker_h}, {ker_w}). This means a convolution is "
                     "not possible.")

  # First we calculate the maximum number of times the kernel can be applied
  # into the image, including the padding and ignoring the stride.
  ker_max_h = in_h + pad_h[0] + pad_h[1] - ker_h + 1
  ker_max_w = in_w + pad_w[0] + pad_w[1] - ker_w + 1
  # Second we calculate the size of the image that is covered when performing
  # a VALID convolution with the kernel, provided the padding.
  out_h = _ceil(ker_max_h, s_h) * s_h - s_h + ker_h
  out_w = _ceil(ker_max_w, s_w) * s_w - s_w + ker_w
  # Finally, we potentially add extra padding on the right in order to make the
  # padded image sizes divisible by their strides. This is needed so we can use
  # later reshape the image into multiples of the strides, which allows us to
  # execute a strided slice via XLA's dynamic slice.  Note that
  # in certain cases this could lead to negative padding, which is correct.
  # Example: image (9, 9), kernel (2, 2), strides (2, 2), padding (0, 0)
  # Then ker_max = 8, out_h = 8, padded_height = 8 and the padding is -1.
  padded_h = _ceil(out_h, s_h) * s_h
  padded_w = _ceil(out_w, s_w) * s_w
  # Actually pad the image (extra 0 for internal padding has to be added)
  extra_pad_h = (pad_h[0], padded_h - in_h - pad_h[0], 0)
  extra_pad_w = (pad_w[0], padded_w - in_w - pad_w[0], 0)
  spatial_padding = in_spec.create_shape(
      (0, 0, 0), (0, 0, 0), extra_pad_h, extra_pad_w)
  padded_image = lax.pad(inputs, jnp.asarray(0.0, dtype=inputs.dtype),
                         spatial_padding)

  # Reshape the input based on strides
  # rhs_shape = [n, out_h // str_h, str_h, out_w // str_w, str_w, c]
  rhs_spec = in_spec.expand_spatial_axes()
  rhs_shape = rhs_spec.create_shape(
      n, c, padded_h // s_h, s_h, padded_w // s_w, s_w)

  # sizes = (n, rhs_h, 1, rhs_w, 1, c)
  rhs_h = (padded_h - ker_h) // s_h + 1
  rhs_w = (padded_w - ker_w) // s_w + 1
  sizes = rhs_spec.create_shape(n, c, rhs_h, 1, rhs_w, 1)

  # Broadcast the weighting function
  if weighting_array is not None:
    if weighting_array.ndim == inputs.ndim:
      shape = rhs_spec.create_shape(n, c, rhs_h, 1, rhs_w, 1)
    elif weighting_array.ndim == inputs.ndim - 1:
      shape = rhs_spec.create_shape(n, 1, rhs_h, 1, rhs_w, 1)
    elif weighting_array.ndim == 1:
      shape = rhs_spec.create_shape(n, 1, 1, 1, 1, 1)
    else:
      raise ValueError(f"`weighting_array` shape {weighting_array.shape} is "
                       f"not compatible with the inputs shape {inputs.shape}"
                       ".")
    reshaped_weighting_array = jnp.reshape(weighting_array, shape)
  else:
    reshaped_weighting_array = 1

  def general_loop_body(i, image):
    reshaped_image = jnp.reshape(image, rhs_shape)

    # Slice the reshaped input
    iw = i % ker_w
    ih = i // ker_w

    # index = (0, ih // sh, ih % sh, iw // sw, iw % sw, 0)
    index = rhs_spec.create_shape(
        0, 0, ih // s_h, ih % s_h, iw // s_w, iw % s_w)
    conv_rhs = _slice_array(reshaped_image, index, sizes)
    conv_rhs = conv_rhs * reshaped_weighting_array

    # Compute the correct padding for the convolution
    dilated_bound_h = 0 if rhs_h == 0 else (rhs_h - 1) * s_h + 1
    dilated_bound_w = 0 if rhs_w == 0 else (rhs_w - 1) * s_w + 1
    conv_pad_h = ker_h - (padded_h - dilated_bound_h + 1)
    conv_pad_w = ker_w - (padded_w - dilated_bound_w + 1)

    # Compute matrix update
    matrix_update = _the_conv4d(
        lhs=image,
        lhs_spec=in_spec,
        rhs=conv_rhs,
        rhs_spec=rhs_spec,
        pad_h=conv_pad_h,
        pad_w=conv_pad_w,
        stride_h=s_h,
        stride_w=s_w,
        per_channel=per_channel,
        precision=precision,
    )

    # Compute vector update
    axis = tuple(i for i in range(len(rhs_spec)) if i != rhs_spec.c_axis)

    vector_update = jnp.sum(conv_rhs, axis=axis)
    vector_update = lax.broadcast_in_dim(vector_update, (1, 1, c), (2,))

    return ih, iw, matrix_update, vector_update

  vector_shape = kernel_spatial_shape + (c,)
  leading_shape = kernel_spatial_shape if per_channel else vector_shape
  matrix_shape = leading_shape + vector_shape

  if unroll_loop:

    matrix_results, vector_results = zip(
        *[general_loop_body(ii, padded_image)[-2:]
          for ii in range(ker_h * ker_w)])

    matrix_results = jnp.stack(matrix_results, axis=0)
    matrix_results = jnp.reshape(matrix_results, matrix_shape)

    vector_results = jnp.stack(vector_results, axis=0)
    vector_results = jnp.reshape(vector_results, vector_shape)

    return matrix_results, vector_results

  else:

    def loop_cond(args):
      return args[0] < ker_h * ker_w

    def loop_body(loop_inputs):

      i, image, matrix_result, vector_result = loop_inputs
      ih, iw, matrix_update, vector_update = general_loop_body(i, image)

      # Update matrix value
      indices = (ih, iw, 0, 0, 0) + (() if per_channel else (0,))
      matrix_result = lax.dynamic_update_slice_p.bind(
          matrix_result, matrix_update, *indices)

      # Update vector value
      vector_result = lax.dynamic_update_slice_p.bind(
          vector_result, vector_update, ih, iw, 0)

      return i + 1, image, matrix_result, vector_result

    # Initialize loop states with zeros
    matrix_init = jnp.zeros(matrix_shape, dtype=inputs.dtype)
    vector_init = jnp.zeros(vector_shape, dtype=inputs.dtype)
    init_vals = (0, padded_image, matrix_init, vector_init)

    return lax.while_loop(loop_cond, loop_body, init_vals)[-2:]
