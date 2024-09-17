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
"""Module containing the diagonal curvature blocks."""
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


class Diagonal(CurvatureBlock, abc.ABC):
  """An abstract class for approximating only the diagonal of curvature."""

  @utils.register_state_class
  class State(CurvatureBlock.State):
    """Persistent state of the block.

    Attributes:
      diagonal_factors: A tuple of the moving averages of the estimated
        diagonals of the curvature for each parameter that is part of the
        associated layer.
    """
    diagonal_factors: tuple[utils.WeightedMovingAverage, ...]

  def _init(
      self,
      rng: PRNGKey,
      exact_powers_to_cache: set[Scalar],
      approx_powers_to_cache: set[Scalar],
      cache_eigenvalues: bool,
  ) -> State:

    del rng

    return Diagonal.State(
        cache=None,
        diagonal_factors=tuple(
            utils.WeightedMovingAverage.zeros_array(shape, self.dtype)
            for shape in self.parameters_shapes
        ),
    )

  def sync(
      self,
      state: State,
      pmap_axis_name: str,
  ) -> State:

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    for factor in state.diagonal_factors:
      factor.sync(pmap_axis_name)

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

    # state_dependent_scale needs to be included because it won't be by the
    # caller of this function (multiply_matpower) when use_cached=True
    scale = self.state_dependent_scale(state) if use_cached else 1.0

    factors = tuple(scale * f.value + identity_weight
                    for f in state.diagonal_factors)

    assert len(factors) == len(vector)

    if power == 1:
      return tuple(f * v for f, v in zip(factors, vector))
    elif power == -1:
      return tuple(v / f for f, v in zip(factors, vector))
    else:
      return tuple(jnp.power(f, power) * v for f, v in zip(factors, vector))

  def _eigenvalues_unscaled(
      self,
      state: State,
      use_cached: bool,
  ) -> Array:
    return jnp.concatenate([f.value.flatten() for f in state.diagonal_factors],
                           axis=0)

  def _update_cache(
      self,
      state: State,
      identity_weight: Numeric,
      exact_powers: set[Scalar],
      approx_powers: set[Scalar],
      eigenvalues: bool,
  ) -> State:

    return state.copy()

  def _to_dense_unscaled(self, state: State) -> Array:

    # Extract factors in canonical order
    factors = [state.diagonal_factors[i].value.flatten()
               for i in self.parameters_canonical_order]

    # Construct diagonal matrix
    return jnp.diag(jnp.concatenate(factors, axis=0))

  def _norm_unscaled(
      self,
      state: CurvatureBlock.State,
      norm_type: str
  ) -> Numeric:

    return utils.product(
        utils.psd_matrix_norm(f.value.flatten(), norm_type=norm_type)
        for f in state.diagonal_factors)


class NaiveDiagonal(Diagonal):
  """Approximates the diagonal of the curvature with in the most obvious way.

  The update to the curvature estimate is computed by ``(sum_i g_i) ** 2 / N``.
  where `g_i` is the gradient of each individual data point, and ``N`` is the
  batch size.
  """

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: Diagonal.State,
      estimation_data: tracer.LayerVjpData[Array],
      ema_old: Numeric,
      ema_new: Numeric,
      identity_weight: Numeric,
      batch_size: Numeric,
  ) -> Diagonal.State:
    del identity_weight

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    for factor, dw in zip(
        state.diagonal_factors, estimation_data.tangents.params
    ):
      factor.update(dw * dw / batch_size, ema_old, ema_new)

    return state


class DenseDiagonal(Diagonal):
  """A `Diagonal` block specifically for dense layers."""

  @property
  def has_bias(self) -> bool:
    """Whether the layer has a bias parameter."""
    return len(self.parameter_variables) == 2

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: Diagonal.State,
      estimation_data: tracer.LayerVjpData[Array],
      ema_old: Numeric,
      ema_new: Numeric,
      identity_weight: Numeric,
      batch_size: Numeric,
  ) -> Diagonal.State:
    del identity_weight

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    [x] = estimation_data.primals.inputs
    [dy] = estimation_data.tangents.outputs

    assert utils.first_dim_is_size(batch_size, x, dy)

    diagonals = (jnp.matmul((x * x).T, dy * dy) / batch_size,)
    if self.has_bias:
      diagonals += (jnp.mean(dy * dy, axis=0),)

    assert len(diagonals) == self.number_of_parameters

    for diagonal_factor, diagonal in zip(state.diagonal_factors, diagonals):
      diagonal_factor.update(diagonal, ema_old, ema_new)

    return state


class Conv2DDiagonal(Diagonal):
  """A :class:`~Diagonal` block specifically for 2D convolution layers."""

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
    specify the maximum number of batch size that we can vmap over. This means
    that the maximum memory usage will be ``max_batch_size_for_vmap`` times the
    memory needed when calling ``self.conv2d_tangent_squared``. And the actual
    ``vmap`` will be called ``ceil(total_batch_size / max_batch_size_for_vmap)``
    number of times in a loop to find the final average.

    Args:
      layer_tag_eq: The Jax equation corresponding to the layer tag, that this
        block will approximate the curvature to.
      max_elements_for_vmap: The threshold used for determining how much
        computation to the in parallel and how much in serial manner. If
        ``None`` will use the value returned by
        :func:`~get_max_parallel_elements`.
    """
    self._averaged_kernel_squared_tangents = utils.loop_and_parallelize_average(
        func=self.conv2d_tangent_squared,
        max_parallel_size=max_elements_for_vmap or
        cb_utils.get_max_parallel_elements(),
    )
    super().__init__(layer_tag_eq)

  @property
  def has_bias(self) -> bool:
    return len(self.parameter_variables) == 2

  def conv2d_tangent_squared(
      self,
      image_features_map: Array,
      output_tangent: Array,
  ) -> Array:
    """Computes the elementwise square of a tangent for a single feature map."""

    extra_params = {k: v for k, v in self.layer_tag_extra_params.items()
                    if k not in ("lhs_shape", "rhs_shape", "meta")}

    _, vjp = jax.vjp(
        functools.partial(
            jax.lax.conv_general_dilated,
            **extra_params
        ),
        image_features_map[None], jnp.zeros(self.parameters_shapes[0])
    )

    return jnp.square(vjp(output_tangent[None])[1])

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: Diagonal.State,
      estimation_data: tracer.LayerVjpData[Array],
      ema_old: Numeric,
      ema_new: Numeric,
      identity_weight: Numeric,
      batch_size: Numeric,
  ) -> Diagonal.State:
    del identity_weight

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    [x] = estimation_data.primals.inputs
    [dy] = estimation_data.tangents.outputs

    assert utils.first_dim_is_size(batch_size, x, dy)

    diagonals = (self._averaged_kernel_squared_tangents(x, dy),)

    if self.has_bias:
      sum_axis = tuple(range(1, dy.ndim - len(self.parameters_shapes[1])))
      bias_dy = jnp.sum(dy, axis=sum_axis)
      diagonals += (jnp.mean(bias_dy * bias_dy, axis=0),)

    assert len(diagonals) == self.number_of_parameters

    for diagonal_factor, diagonal in zip(state.diagonal_factors, diagonals):
      diagonal_factor.update(diagonal, ema_old, ema_new)

    return state


class ScaleAndShiftDiagonal(Diagonal):
  """A diagonal approximation specifically for a scale and shift layers."""

  @property
  def has_scale(self) -> bool:
    """Whether this layer's equation has a scale."""
    return self._layer_tag_eq.params["has_scale"]

  @property
  def has_shift(self) -> bool:
    """Whether this layer's equation has a shift."""
    return self._layer_tag_eq.params["has_shift"]

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: Diagonal.State,
      estimation_data: tracer.LayerVjpData[Array],
      ema_old: Numeric,
      ema_new: Numeric,
      identity_weight: Numeric,
      batch_size: Numeric,
  ) -> Diagonal.State:
    del identity_weight

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    [x] = estimation_data.primals.inputs
    [dy] = estimation_data.tangents.outputs

    assert utils.first_dim_is_size(batch_size, x, dy)

    if self.has_scale:

      assert state.diagonal_factors[0].shape == self.parameters_shapes[0]

      scale_shape = estimation_data.primals.params[0].shape

      d_scale = cb_utils.compatible_sum(x * dy, scale_shape, skip_axes=[0])

      scale_diag_update = jnp.sum(
          d_scale * d_scale,
          axis=0, keepdims=d_scale.ndim == len(scale_shape)
      ) / batch_size

      state.diagonal_factors[0].update(scale_diag_update, ema_old, ema_new)

    if self.has_shift:

      shift_shape = estimation_data.primals.params[-1].shape
      d_shift = cb_utils.compatible_sum(dy, shift_shape, skip_axes=[0])

      shift_diag_update = jnp.sum(
          d_shift * d_shift,
          axis=0, keepdims=d_shift.ndim == len(shift_shape)
      ) / batch_size

      state.diagonal_factors[-1].update(shift_diag_update, ema_old, ema_new)

    return state
