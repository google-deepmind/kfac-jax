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
"""Module containing the Kronecker factored curvature blocks."""
import abc
import math
from typing import Any, Sequence

import jax.numpy as jnp
from kfac_jax._src import layers_and_loss_tags as tags
from kfac_jax._src import patches_second_moment as psm
from kfac_jax._src import tracer
from kfac_jax._src import utils
from kfac_jax._src.curvature_blocks import curvature_block
from kfac_jax._src.curvature_blocks import utils as cb_utils
from typing_extensions import Self


# Types for annotation
Array = utils.Array
Scalar = utils.Scalar
Numeric = utils.Numeric
PRNGKey = utils.PRNGKey
Shape = utils.Shape
CurvatureBlock = curvature_block.CurvatureBlock


class KroneckerFactored(CurvatureBlock, abc.ABC):
  """An abstract class for approximating the block with a Kronecker product.

  The constructor takes two special arguments:
    - parameters_specs: A list, where each element specifies for each
      parameter a "rearrange string". This is in the format `abc->b(ca)`
      similar to `einops.rearrange`.
    - parameters_concat_axis: The axis along which the parameters will be
      concatenated to form a single array after each parameter has been
      rearranged according to its "rearrange string".

  The above implies that:
    - All parameters must have the same rank after they have been rearranged.
    - All parameters must have the same size along all axes except the
      concatenation axis after they have been rearranged.

  By default, each parameter is rearanged to a matrix, by merging all dimensions
  except the last one. If a parameter is a vector (rank 1), it is rearranged to
  a matrix with the first dimension being 1. Then concatenation is done along
  axis=0.
  """

  @utils.register_state_class
  class State(CurvatureBlock.State):
    """Persistent state of the block.

    Attributes:
      factors: A tuple of the moving averages of the estimated factors of the
        curvature for each axis group.
    """

    factors: tuple[utils.WeightedMovingAverage, ...]

    @classmethod
    def from_dict(cls, dict_rep: dict[str, Any]) -> Self:
      class_name = dict_rep.pop("__class__", cls.__name__)
      assert class_name == cls.__name__
      return cls(
          factors=tuple(
              utils.WeightedMovingAverage.from_dict(rep)
              for rep in dict_rep["factor"]
          )
      )

  def __init__(
      self,
      layer_tag_eq: tags.LayerTagEqn,
      parameters_specs: Sequence[str] | None = None,
      parameters_concat_axis: int = 0,
  ):

    # Even though the superclass constructor will set this later, we need to do
    # it now since it's used below.
    self._layer_tag_eq = layer_tag_eq

    if parameters_specs is None:
      parameters_specs = []

      for shape in self.parameters_shapes:

        if len(shape) == 1:
          parameters_specs.append("a -> 1a")

        else:
          in_str = cb_utils.ALPHABET[:len(shape)]
          out_str = f"({in_str[:-1]}){in_str[-1]}"
          parameters_specs.append(f"{in_str} -> {out_str}")

    else:
      assert len(parameters_specs) == self.number_of_parameters

    self.parameters_specs = parameters_specs
    self.parameters_concat_axis = parameters_concat_axis

    super().__init__(layer_tag_eq)

  def __str__(self):
    return (
        f"{self.__class__.__name__}(parameter_specs={self.parameters_specs}, "
        f"parameters_concat_axis={self.parameters_concat_axis}), "
        f"tag name: {self.name}, params shapes: {self.parameters_shapes!r}"
    )

  def parameters_shaped_list_to_array(
      self,
      parameters_shaped_list: Sequence[Array],
  ) -> Array:
    """Combines all parameters to a single array."""
    values = []
    for p, spec in zip(
        parameters_shaped_list,
        self.parameters_specs,
        strict=True,
    ):
      values.append(utils.rearrange(p, spec))

    return jnp.concatenate(values, axis=self.parameters_concat_axis)

  def array_to_parameters_shaped_list(self, array: Array) -> tuple[Array, ...]:
    """An inverse transformation of ``self.parameters_shaped_list_to_array``."""
    parameters_list = []
    n = 0
    index = [slice(None)] * array.ndim

    for shape, spec in zip(
        self.parameters_shapes,
        self.parameters_specs,
        strict=True,
    ):
      zero = utils.rearrange(jnp.zeros(shape), spec)
      d = zero.shape[self.parameters_concat_axis]
      index[self.parameters_concat_axis] = slice(n, n + d)
      p = array[tuple(index)]
      parameters_list.append(p.reshape(shape))
      n += d

    return tuple(parameters_list)

  @property
  def array_shape(self) -> Shape:
    """The shape of the single non axis grouped array."""
    avals = [jnp.zeros(shape) for shape in self.parameters_shapes]
    return self.parameters_shaped_list_to_array(avals).shape

  @property
  def array_ndim(self) -> int:
    """The number of dimensions of the single non axis grouped array."""
    return len(self.array_shape)

  def _init(
      self,
      rng: PRNGKey,
      exact_powers_to_cache: set[Scalar],
      approx_powers_to_cache: set[Scalar],
      cache_eigenvalues: bool,
  ) -> State:

    cache = {}
    factors = []

    for i, d in enumerate(self.array_shape):

      factors.append(
          utils.WeightedMovingAverage.zeros_array((d, d), self.dtype)
      )

      if cache_eigenvalues or exact_powers_to_cache:
        cache[f"{i}_factor_eigenvalues"] = jnp.zeros((d,), dtype=self.dtype)

      if exact_powers_to_cache:
        cache[f"{i}_factor_eigen_vectors"] = jnp.zeros((d, d), dtype=self.dtype)

      for power in approx_powers_to_cache:

        if power != -1:
          raise NotImplementedError(
              f"Approximations for power {power} is not yet implemented."
          )

        if str(power) not in cache:
          cache[str(power)] = {}

        cache[str(power)][f"{i}_factor"] = jnp.zeros((d, d), dtype=self.dtype)

    return KroneckerFactored.State(
        cache=cache,
        factors=tuple(factors),
    )

  def sync(
      self,
      state: State,
      pmap_axis_name: str,
  ) -> State:

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    for factor in state.factors:
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

    assert len(state.factors) == self.array_ndim

    vector = self.parameters_shaped_list_to_array(vector)

    if power == 1:

      factors = [f.value for f in state.factors]

      # state_dependent_scale needs to be included here because it won't be by
      # the caller of this function (multiply_matpower) when use_cached=True.
      # This is not an issue for other powers because they bake in
      # state_dependent_scale.
      scale = self.state_dependent_scale(state) if use_cached else 1.0

      if exact_power:
        result = scale * utils.kronecker_product_axis_mul_v(factors, vector)
        result = result + identity_weight * vector

      else:
        # If compute pi_adjusted_kronecker_factors used a more expensive matrix
        # norm in its computation, it might make sense to cache it. But we
        # currently don't do that.

        result = scale * utils.kronecker_product_axis_mul_v(
            utils.pi_adjusted_kronecker_factors(
                *factors, damping=identity_weight / scale),
            vector)

    elif exact_power:

      if use_cached:
        s = [
            state.cache[f"{i}_factor_eigenvalues"]
            for i in range(len(state.factors))
        ]
        q = [
            state.cache[f"{i}_factor_eigen_vectors"]
            for i in range(len(state.factors))
        ]

      else:
        s, q = zip(
            *[utils.safe_psd_eigh(factor.value) for factor in state.factors]
        )

      eigenvalues = utils.outer_product(*s) + identity_weight
      eigenvalues = jnp.power(eigenvalues, power)

      result = utils.kronecker_eigen_basis_axis_mul_v(q, eigenvalues, vector)

    else:

      if power != -1 and power != -0.5:
        raise NotImplementedError(
            f"Approximations for power {power} is not yet implemented."
        )

      if use_cached:

        assert power != -0.5

        factors = [
            state.cache[str(power)][f"{i}_factor"]
            for i in range(len(state.factors))
        ]

      else:

        factors = [factor.value for factor in state.factors]

        factors = utils.pi_adjusted_kronecker_factors(
            *factors, damping=identity_weight)

        if power == -1:
          factors = utils.invert_psd_matrices(factors)
        elif power == -0.5:
          factors = utils.inverse_sqrt_psd_matrices(factors)
        else:
          raise NotImplementedError()

      result = utils.kronecker_product_axis_mul_v(factors, vector)

    return self.array_to_parameters_shaped_list(result)

  def _eigenvalues_unscaled(
      self,
      state: State,
      use_cached: bool,
  ) -> Array:

    assert len(state.factors) == self.array_ndim

    if use_cached:
      s = [
          state.cache[f"{i}_factor_eigenvalues"]
          for i in range(len(state.factors))
      ]
    else:
      s_q = [utils.safe_psd_eigh(factor.value) for factor in state.factors]
      s, _ = zip(*s_q)

    return utils.outer_product(*s)

  def _update_cache(
      self,
      state: State,
      identity_weight: Numeric,
      exact_powers: set[Scalar],
      approx_powers: set[Scalar],
      eigenvalues: bool,
  ) -> State:

    assert len(state.factors) == self.array_ndim

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    scale = self.state_dependent_scale(state)
    factor_scale = jnp.power(scale, 1.0 / self.array_ndim)

    if eigenvalues or exact_powers:

      s_q = [utils.safe_psd_eigh(factor.value) for factor in state.factors]

      s, q = zip(*s_q)

      for i in range(len(state.factors)):
        state.cache[f"{i}_factor_eigenvalues"] = factor_scale * s[i]

        if exact_powers:
          state.cache[f"{i}_factor_eigen_vectors"] = q[i]

    for power in approx_powers:

      if power != -1:
        raise NotImplementedError(
            f"Approximations for power {power} is not yet implemented."
        )

      cache = state.cache[str(power)]

      # This computes the approximate inverse factors using the generalization
      # of the pi-adjusted inversion from the original KFAC paper.
      inv_factors = utils.pi_adjusted_kronecker_inverse(
          *[factor.value for factor in state.factors],
          damping=identity_weight,
      )

      for i in range(len(state.factors)):
        cache[f"{i}_factor"] = inv_factors[i] / factor_scale

    return state

  def _norm_unscaled(
      self,
      state: CurvatureBlock.State,
      norm_type: str
  ) -> Numeric:

    return utils.product(
        utils.psd_matrix_norm(f.value, norm_type=norm_type)
        for f in state.factors)

  def _to_dense_unscaled(self, state: "KroneckerFactored.State") -> Array:

    # We currently support this only for 2 parameters
    assert 0 < self.number_of_parameters <= 2
    inputs_factor = state.factors[0].value

    if (self.number_of_parameters == 2 and
        self.parameters_canonical_order[0] != 0):

      # Permute the matrix according to the parameters canonical order
      inputs_factor = utils.block_permuted(
          state.factors[0].value,
          block_sizes=[state.factors[0].shape[0] - 1, 1],
          block_order=(1, 0),
      )

    return jnp.kron(inputs_factor, state.factors[1].value)


class DenseTwoKroneckerFactored(KroneckerFactored):
  """A :class:`~TwoKroneckerFactored` block specifically for dense layers."""

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: KroneckerFactored.State,
      estimation_data: tracer.LayerVjpData[Array],
      ema_old: Numeric,
      ema_new: Numeric,
      identity_weight: Numeric,
      batch_size: Numeric,
  ) -> KroneckerFactored.State:
    del identity_weight
    assert 1 <= self.number_of_parameters <= 2

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    [x] = estimation_data.primals.inputs
    [dy] = estimation_data.tangents.outputs

    assert utils.first_dim_is_size(batch_size, x, dy)

    if self.number_of_parameters == 2:
      x_one = jnp.ones_like(x[:, :1])
      x = jnp.concatenate([x, x_one], axis=1)

    input_stats = jnp.einsum("ay,az->yz", x, x) / batch_size
    output_stats = jnp.einsum("ay,az->yz", dy, dy) / batch_size

    state.factors[0].update(input_stats, ema_old, ema_new)
    state.factors[1].update(output_stats, ema_old, ema_new)

    return state


class RepeatedDenseKroneckerFactored(DenseTwoKroneckerFactored):
  """Block for dense layers applied to tensors with extra time/loc dims."""

  @utils.register_state_class
  class State(KroneckerFactored.State):
    """Persistent state of the block.

    Attributes:
      average_repeats: A decayed average of the per-case number of non-masked
        repeats in the data used to compute the block's statistics. We use the
        same decayed averaging for this quantity that we do for the statistics,
        so that they "match".
    """

    average_repeats: utils.WeightedMovingAverage

  def __init__(
      self,
      layer_tag_eq: tags.LayerTagEqn,
      use_masking: bool = True,
      parameters_specs: Sequence[str] | None = None,
      parameters_concat_axis: int = 0,
  ):
    self._use_masking = use_masking
    super().__init__(
        layer_tag_eq=layer_tag_eq,
        parameters_specs=parameters_specs,
        parameters_concat_axis=parameters_concat_axis,
    )

  def _init(
      self,
      rng: PRNGKey,
      exact_powers_to_cache: set[Scalar],
      approx_powers_to_cache: set[Scalar],
      cache_eigenvalues: bool,
  ) -> "RepeatedDenseKroneckerFactored.State":

    super_state = super()._init(
        rng, exact_powers_to_cache, approx_powers_to_cache, cache_eigenvalues
    )

    return RepeatedDenseKroneckerFactored.State(
        average_repeats=utils.WeightedMovingAverage.zeros_array((), self.dtype),
        **super_state.__dict__,
    )

  def state_dependent_scale(
      self,
      state: "RepeatedDenseKroneckerFactored.State",
  ) -> Numeric:
    return 1.0 / state.average_repeats.value

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: KroneckerFactored.State,
      estimation_data: tracer.LayerVjpData[Array],
      ema_old: Numeric,
      ema_new: Numeric,
      identity_weight: Numeric,
      batch_size: Numeric,
  ) -> KroneckerFactored.State:
    del identity_weight

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    [x] = estimation_data.primals.inputs
    [dy] = estimation_data.tangents.outputs

    assert utils.first_dim_is_size(batch_size, x, dy)

    if self._use_masking:

      # hack: we identify masked repeats by checking if all corresponding
      # entries of dy are zero
      mask = 1.0 - jnp.all(dy == 0.0, axis=-1, keepdims=True)

      # zero out corresponding elts of x
      x = x * mask

      # compute total non-masked
      total = jnp.sum(mask)

    else:
      total = math.prod(dy.shape[:-1])

    x = x.reshape([-1, x.shape[-1]])
    dy = dy.reshape([-1, dy.shape[-1]])

    if self.number_of_parameters == 2:
      x_one = jnp.ones_like(x[:, :1])
      x = jnp.concatenate([x, x_one], axis=1)

    input_stats = jnp.einsum("ay,az->yz", x, x) / batch_size
    output_stats = jnp.einsum("ay,az->yz", dy, dy) / batch_size

    state.factors[0].update(input_stats, ema_old, ema_new)
    state.factors[1].update(output_stats, ema_old, ema_new)
    state.average_repeats.update(total / batch_size, ema_old, ema_new)

    return state


class Conv2DTwoKroneckerFactored(KroneckerFactored):
  """A :class:`~TwoKroneckerFactored` block specifically for 2D convolution layers."""

  def fixed_scale(self) -> Numeric:
    return float(self.num_locations)

  @property
  def kernel_output_axis(self) -> int:
    return self._layer_tag_eq.params["dimension_numbers"].rhs_spec[0]

  @property
  def outputs_channel_index(self) -> int:
    """The ``channels`` index in the outputs of the layer."""
    return self._layer_tag_eq.params["dimension_numbers"].out_spec[1]

  @property
  def inputs_channel_index(self) -> int:
    """The ``channels`` index in the inputs of the layer."""
    return self._layer_tag_eq.params["dimension_numbers"].lhs_spec[1]

  @property
  def weights_output_channel_index(self) -> int:
    """The ``channels`` index in weights of the layer."""
    return self._layer_tag_eq.params["dimension_numbers"].rhs_spec[0]

  @property
  def weights_spatial_shape(self) -> Shape:
    spatial_index = self._layer_tag_eq.params["dimension_numbers"].rhs_spec[2:]
    return tuple(self.parameters_shapes[0][i] for i in spatial_index)

  @property
  def weights_spatial_size(self) -> int:
    """The spatial filter size of the weights."""
    return utils.product(dim for dim in self.weights_spatial_shape)

  @property
  def inputs_spatial_shape(self) -> Shape:
    spatial_index = self._layer_tag_eq.params["dimension_numbers"].lhs_spec[2:]
    return tuple(self.inputs_shapes[0][i] for i in spatial_index)

  @property
  def num_locations(self) -> int:
    """The number of spatial locations that each filter is applied to."""
    return psm.num_conv_locations(
        self.inputs_spatial_shape,
        self.weights_spatial_shape,
        self._layer_tag_eq.params["window_strides"],
        self._layer_tag_eq.params["padding"])

  def input_size(self) -> int:
    if self.has_bias:
      return self.num_inputs_channels * self.weights_spatial_size + 1
    else:
      return self.num_inputs_channels * self.weights_spatial_size

  def output_size(self) -> int:
    return self.num_outputs_channels

  @property
  def num_inputs_channels(self) -> int:
    """The number of channels in the inputs to the layer."""
    return self._layer_tag_eq.invars[0].aval.shape[  # pytype: disable=attribute-error
        self.inputs_channel_index]

  @property
  def num_outputs_channels(self) -> int:
    """The number of channels in the outputs to the layer."""
    return self._layer_tag_eq.invars[1].aval.shape[  # pytype: disable=attribute-error
        self.weights_output_channel_index]

  def compute_inputs_stats(
      self,
      inputs: Array,
      weighting_array: Array | None = None,
  ) -> Array:
    """Computes the statistics for the inputs factor."""
    batch_size = inputs.shape[0]

    input_cov_m, input_cov_v = psm.patches_moments(
        inputs,
        kernel_spatial_shape=self.weights_spatial_shape,
        strides=self._layer_tag_eq.params["window_strides"],
        padding=self._layer_tag_eq.params["padding"],
        data_format=None,
        dim_numbers=self._layer_tag_eq.params["dimension_numbers"],
        precision=self._layer_tag_eq.params.get("precision"),
        weighting_array=weighting_array,
    )

    # Flatten the kernel and channels dimensions
    k, h, c = input_cov_v.shape
    input_cov_v = jnp.reshape(input_cov_v, (k * h * c,))
    input_cov_m = jnp.reshape(input_cov_m, (k * h * c, k * h * c))

    # Normalize by the `batch size` * `num_locations`
    normalizer = batch_size * self.num_locations
    input_cov_m = input_cov_m / normalizer
    input_cov_v = input_cov_v / normalizer

    if self.number_of_parameters == 1:
      return input_cov_m

    if weighting_array is None:
      corner = jnp.ones([1], dtype=input_cov_m.dtype)
    else:
      corner = jnp.mean(weighting_array).reshape([1])

    input_cov = jnp.concatenate([input_cov_m, input_cov_v[None]], axis=0)
    input_cov_v = jnp.concatenate([input_cov_v, corner], axis=0)

    return jnp.concatenate([input_cov, input_cov_v[:, None]], axis=1)

  def compute_outputs_stats(self, tangent_of_output: Array) -> Array:
    """Computes the statistics for the outputs factor."""
    lhs_str = utils.replace_char(
        cb_utils.ALPHABET[:4], "y", self.outputs_channel_index)
    rhs_str = utils.replace_char(
        cb_utils.ALPHABET[:4], "z", self.outputs_channel_index)
    ein_str = f"{lhs_str},{rhs_str}->yz"
    stats = jnp.einsum(ein_str, tangent_of_output, tangent_of_output)

    # Normalize by the `batch size` * `num_locations`
    normalizer = tangent_of_output.shape[0] * self.num_locations
    return stats / normalizer

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: KroneckerFactored.State,
      estimation_data: tracer.LayerVjpData[Array],
      ema_old: Numeric,
      ema_new: Numeric,
      identity_weight: Numeric,
      batch_size: Numeric,
  ) -> KroneckerFactored.State:
    del identity_weight
    assert 1 <= self.number_of_parameters <= 2

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    [x] = estimation_data.primals.inputs
    [dy] = estimation_data.tangents.outputs

    assert utils.first_dim_is_size(batch_size, x, dy)

    input_stats = self.compute_inputs_stats(x)
    output_stats = self.compute_outputs_stats(dy)

    state.factors[0].update(input_stats, ema_old, ema_new)
    state.factors[1].update(output_stats, ema_old, ema_new)

    return state
