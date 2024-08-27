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
"""Module containing the full matrix curvature blocks."""
import abc
import functools
from typing import Sequence

import jax
import jax.numpy as jnp
import jax.scipy
from kfac_jax._src import layers_and_loss_tags as tags
from kfac_jax._src import tracer
from kfac_jax._src import utils
from kfac_jax._src.curvature_blocks import curvature_block
from kfac_jax._src.curvature_blocks import utils as cb_utils

# Types for annotation
Array = utils.Array
Scalar = utils.Scalar
Numeric = utils.Numeric
PRNGKey = utils.PRNGKey
CurvatureBlock = curvature_block.CurvatureBlock


class Full(CurvatureBlock, abc.ABC):
  """An abstract class for approximating the block matrix with a full matrix."""

  @utils.register_state_class
  class State(CurvatureBlock.State):
    """Persistent state of the block.

    Attributes:
      matrix: A moving average of the estimated curvature matrix for all
        parameters that are part of the associated layer.
    """
    matrix: utils.WeightedMovingAverage

  def __init__(
      self,
      layer_tag_eq: tags.LayerTagEqn,
      eigen_decomposition_threshold: int | None = None,
  ):
    """Initializes the block.

    Args:
      layer_tag_eq: The Jax equation corresponding to the layer tag that this
        block will approximate the curvature to.
      eigen_decomposition_threshold: During calls to ``init`` and
       ``update_cache`` if higher number of matrix powers than this threshold
       are requested,  instead of computing individual approximate powers, will
       directly compute the eigen-decomposition instead (which provide access to
       any matrix power). If this is ``None`` will use the value returned from
       :func:`~get_default_eigen_decomposition_threshold()`.
    """

    if eigen_decomposition_threshold is None:
      threshold = cb_utils.get_default_eigen_decomposition_threshold()
      self._eigen_decomposition_threshold = threshold

    else:
      self._eigen_decomposition_threshold = eigen_decomposition_threshold

    super().__init__(layer_tag_eq)

  def parameters_list_to_single_vector(
      self,
      parameters_shaped_list: Sequence[Array],
  ) -> Array:
    """Converts values corresponding to parameters of the block to vector."""

    if len(parameters_shaped_list) != self.number_of_parameters:

      raise ValueError(f"Expected a list of {self.number_of_parameters} values,"
                       f" but got {len(parameters_shaped_list)} instead.")

    for array, shape in zip(parameters_shaped_list, self.parameters_shapes):

      if array.shape != shape:
        raise ValueError(f"Expected a value of shape {shape}, but got "
                         f"{array.shape} instead.")

    return jnp.concatenate([v.flatten() for v in parameters_shaped_list])

  def single_vector_to_parameters_list(
      self,
      vector: Array,
  ) -> tuple[Array, ...]:
    """Reverses the transformation ``self.parameters_list_to_single_vector``."""

    if vector.ndim != 1:
      raise ValueError(f"Expecting a vector, got {vector.ndim}-tensor.")

    if vector.size != self.dim:
      raise ValueError(f"Expected a vector of size {self.dim}, but got "
                       f"{vector.size} instead.")

    parameters_shaped_list = []
    index = 0

    for shape in self.parameters_shapes:

      size = utils.product(shape)
      parameters_shaped_list.append(vector[index: index + size].reshape(shape))
      index += size

    assert index == self.dim

    return tuple(parameters_shaped_list)

  def _init(
      self,
      rng: PRNGKey,
      exact_powers_to_cache: set[Scalar],
      approx_powers_to_cache: set[Scalar],
      cache_eigenvalues: bool,
  ) -> State:

    del rng

    # This block does not have any notion of "approximate" powers
    exact_powers_to_cache = exact_powers_to_cache | approx_powers_to_cache
    cache = {}

    if len(exact_powers_to_cache) > self._eigen_decomposition_threshold:
      cache["eigenvalues"] = jnp.zeros([self.dim], self.dtype)
      cache["eigen_vectors"] = jnp.zeros([self.dim, self.dim], self.dtype)

    elif cache_eigenvalues:
      cache["eigenvalues"] = jnp.zeros([self.dim], self.dtype)

    if len(exact_powers_to_cache) <= self._eigen_decomposition_threshold:
      for power in exact_powers_to_cache:
        cache[str(power)] = jnp.zeros([self.dim, self.dim], self.dtype)

    return Full.State(
        cache=cache,
        matrix=utils.WeightedMovingAverage.zeros_array(
            [self.dim, self.dim], self.dtype),
    )

  def sync(
      self,
      state: State,
      pmap_axis_name: str,
  ) -> State:

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    state.matrix.sync(pmap_axis_name)

    return state

  def _multiply_matpower_unscaled(
      self,
      state: State,
      vector: Sequence[Array],
      identity_weight: Numeric,
      power: Scalar,
      exact_power: bool,
      use_cached: bool,
  ) -> tuple[Array, ...]:

    vector = self.parameters_list_to_single_vector(vector)

    if power == 1:

      result = jnp.matmul(state.matrix.value, vector)

      if use_cached:
        # state_dependent_scale needs to be included here because it won't be by
        # the caller of this function (multiply_matpower) when use_cached=True.
        # This is not an issue for other powers because they bake in
        # state_dependent_scale.
        result *= self.state_dependent_scale(state)

      result += identity_weight * vector

    elif not use_cached:

      matrix = state.matrix.value + identity_weight * jnp.eye(self.dim)

      if power == -1:
        result = utils.psd_solve(matrix, vector)
      else:
        # TODO(jamesmartens,botev): investigate this for determinism on GPUs
        # NOTE: this function only works for integer powers
        result = jnp.matmul(jnp.linalg.matrix_power(matrix, power), vector)

    else:

      if str(power) in state.cache:
        result = jnp.matmul(state.cache[str(power)], vector)

      else:
        s = state.cache["eigenvalues"]
        q = state.cache["eigen_vectors"]

        result = jnp.matmul(jnp.transpose(q), vector)
        result = jnp.power(s + identity_weight, power) * result
        result = jnp.matmul(q, result)

    return self.single_vector_to_parameters_list(result)

  def _eigenvalues_unscaled(
      self,
      state: State,
      use_cached: bool,
  ) -> Array:

    if not use_cached:
      return utils.safe_psd_eigh(state.matrix.value)[0]

    else:
      return state.cache["eigenvalues"]

  def _update_cache(
      self,
      state: State,
      identity_weight: Numeric,
      exact_powers: set[Scalar],
      approx_powers: set[Scalar],
      eigenvalues: bool,
  ) -> State:

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    scale = self.state_dependent_scale(state)

    # This block does not have any notion of "approximate" powers
    exact_powers = exact_powers | approx_powers

    if len(exact_powers) > self._eigen_decomposition_threshold:

      s, q = utils.safe_psd_eigh(state.matrix.value)
      state.cache = dict(eigenvalues=scale * s, eigen_vectors=q)

    else:

      if eigenvalues:
        state.cache["eigenvalues"] = scale * utils.safe_psd_eigh(
            state.matrix.value)[0]

      for power in exact_powers:

        if power == -1:
          state.cache[str(power)] = utils.psd_inv(
              state.matrix.value + identity_weight * jnp.eye(self.dim)) / scale
        else:
          matrix = state.matrix.value + identity_weight * jnp.eye(self.dim)
          state.cache[str(power)] = (
              (scale ** power) * jnp.linalg.matrix_power(matrix, power))

    return state

  def _to_dense_unscaled(self, state: State) -> Array:

    # Permute the matrix according to the parameters canonical order
    return utils.block_permuted(
        state.matrix.value,
        block_sizes=[utils.product(shape) for shape in self.parameters_shapes],
        block_order=self.parameters_canonical_order
    )

  def _norm_unscaled(
      self,
      state: CurvatureBlock.State,
      norm_type: str
  ) -> Numeric:

    return utils.psd_matrix_norm(state.matrix.value, norm_type=norm_type)

  def _undamped_diagonal_unscaled(self, state: State) -> tuple[Array, ...]:
    diag_vec = jnp.diag(state.matrix.value)
    return self.single_vector_to_parameters_list(diag_vec)


class NaiveFull(Full):
  """Approximates the full curvature with in the most obvious way.

  The update to the curvature estimate is computed by
  ``(sum_i g_i) (sum_i g_i)^T / N``, where ``g_i`` is the gradient of each
  individual data point, and ``N`` is the batch size.
  """

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: Full.State,
      estimation_data: tracer.LayerVjpData[Array],
      ema_old: Numeric,
      ema_new: Numeric,
      identity_weight: Numeric,
      batch_size: Numeric,
  ) -> Full.State:
    del identity_weight

    # This method supports the case where the param tangents have an extra
    # leading dimension that should be summed over (after the outer products).
    # TODO(jamesmartens): add support for this to NaiveDiagonal

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    params_tangents = jax.tree_util.tree_leaves(
        estimation_data.tangents.params)

    params_tangents_flattened = []

    assert len(params_tangents) == self.number_of_parameters

    for p_shape, pt in zip(self.parameters_shapes, params_tangents):

      if p_shape:
        assert (
            pt.shape[-len(p_shape) :] == p_shape
        ), f"{pt.shape=} and {p_shape=}"

      p_size = utils.product(p_shape)

      params_tangents_flattened.append(pt.reshape([-1, p_size]))

    tangents = jnp.concatenate(params_tangents_flattened, axis=1)

    if jnp.iscomplexobj(tangents):
      stats = (
          jnp.einsum("ay,az->yz", tangents.real, tangents.real)
          - jnp.einsum("ay,az->yz", tangents.imag, tangents.imag)) / batch_size
    else:
      stats = jnp.einsum("ay,az->yz", tangents, tangents) / batch_size

    state.matrix.update(stats, ema_old, ema_new)

    return state


class DenseFull(Full):
  """A `Full` block specifically for dense layers."""

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: Full.State,
      estimation_data: tracer.LayerVjpData[Array],
      ema_old: Numeric,
      ema_new: Numeric,
      identity_weight: Numeric,
      batch_size: Numeric,
  ) -> Full.State:
    del identity_weight

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    [x] = estimation_data.primals.inputs
    [dy] = estimation_data.tangents.outputs

    assert utils.first_dim_is_size(batch_size, x, dy)

    params_tangents = x[:, :, None] * dy[:, None, :]

    if self.number_of_parameters == 2:
      params_tangents = jnp.concatenate([params_tangents, dy[:, None]], axis=1)

    params_tangents = jnp.reshape(params_tangents, [batch_size, -1])

    matrix_update = jnp.matmul(params_tangents.T, params_tangents) / batch_size
    state.matrix.update(matrix_update, ema_old, ema_new)

    return state


class Conv2DFull(Full):
  """A :class:`~Full` block specifically for 2D convolution layers."""

  def __init__(
      self,
      layer_tag_eq: tags.LayerTagEqn,
      max_elements_for_vmap: int | None = None,
  ):
    """Initializes the block.

    Since there is no 'nice' formula for computing the average of the
    tangents for a 2D convolution, what we do is that we have a function -
    ``self.conv2d_tangent_squared`` - that computes for a single feature map the
    square of the tangents for the kernel of the convolution. To average over
    the batch we have two choices - vmap or loop over the batch sequentially
    using scan. This utility function provides a trade-off by being able to
    specify the maximum batch that that will be handled in a single iteration
    of the loop. This means that the maximum memory usage will be
    ``max_batch_size_for_vmap`` times the memory needed when calling
    ``self.conv2d_tangent_squared``. And the actual ``vmap`` will be
    called ``ceil(total_batch_size / max_batch_size_for_vmap)`` number of times
    in a loop to find the final average.

    Args:
      layer_tag_eq: The Jax equation corresponding to the layer tag, that this
        block will approximate the curvature to.
      max_elements_for_vmap: The threshold used for determining how much
        computation to the in parallel and how much in serial manner. If
        ``None`` will use the value returned by
        :func:`~get_max_parallel_elements`.
    """

    self._averaged_tangents_outer_product = utils.loop_and_parallelize_average(
        func=self.conv2d_tangent_outer_product,
        max_parallel_size=max_elements_for_vmap or
        cb_utils.get_max_parallel_elements(),
    )

    super().__init__(layer_tag_eq)

  def conv2d_tangent_outer_product(
      self,
      inputs: Array,
      tangent_of_outputs: Array,
  ) -> Array:
    """Computes the outer product of a tangent for a single feature map."""

    extra_params = {k: v for k, v in self.layer_tag_extra_params.items()
                    if k not in ("lhs_shape", "rhs_shape", "meta")}

    _, vjp = jax.vjp(
        functools.partial(
            jax.lax.conv_general_dilated,
            **extra_params
        ),
        inputs[None], jnp.zeros(self.parameters_shapes[0])
    )

    tangents = (vjp(tangent_of_outputs[None])[1],)

    if self.number_of_parameters == 2:
      num_axis = tangent_of_outputs.ndim - len(self.parameters_shapes[1])
      sum_axis = tuple(range(num_axis))
      tangents += (jnp.sum(tangent_of_outputs, axis=sum_axis),)

    flat_tangents = self.parameters_list_to_single_vector(tangents)

    return jnp.outer(flat_tangents, flat_tangents)

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: Full.State,
      estimation_data: tracer.LayerVjpData[Array],
      ema_old: Numeric,
      ema_new: Numeric,
      identity_weight: Numeric,
      batch_size: Numeric,
  ) -> Full.State:
    del identity_weight

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    [x] = estimation_data.primals.inputs
    [dy] = estimation_data.tangents.outputs
    assert utils.first_dim_is_size(batch_size, x, dy)

    matrix_update = self._averaged_tangents_outer_product(x, dy)
    state.matrix.update(matrix_update, ema_old, ema_new)

    return state


class ScaleAndShiftFull(Full):
  """A full dense approximation specifically for a scale and shift layers."""

  @property
  def _has_scale(self) -> bool:
    """Whether this layer's equation has a scale."""
    return self._layer_tag_eq.params["has_scale"]

  @property
  def _has_shift(self) -> bool:
    """Whether this layer's equation has a shift."""
    return self._layer_tag_eq.params["has_shift"]

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: Full.State,
      estimation_data: tracer.LayerVjpData[Array],
      ema_old: Numeric,
      ema_new: Numeric,
      identity_weight: Numeric,
      batch_size: Numeric,
  ) -> Full.State:
    del identity_weight

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    [x] = estimation_data.primals.inputs
    [dy] = estimation_data.tangents.outputs
    assert utils.first_dim_is_size(batch_size, x, dy)

    tangents = []

    if self._has_scale:
      # Scale tangent
      scale_shape = estimation_data.primals.params[0].shape

      d_scale = cb_utils.compatible_sum(x * dy, scale_shape, skip_axes=[0])
      d_scale = d_scale.reshape([batch_size, -1])

      tangents.append(d_scale)

    if self._has_shift:
      # Shift tangent

      shift_shape = estimation_data.primals.params[-1].shape

      d_shift = cb_utils.compatible_sum(dy, shift_shape, skip_axes=[0])
      d_shift = d_shift.reshape([batch_size, -1])

      tangents.append(d_shift)

    tangents = jnp.concatenate(tangents, axis=1)
    matrix_update = jnp.matmul(tangents.T, tangents) / batch_size

    state.matrix.update(matrix_update, ema_old, ema_new)

    return state
