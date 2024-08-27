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
"""Module containing TNT curvature blocks."""
from typing import Sequence

import jax
import jax.numpy as jnp
import jax.scipy
from kfac_jax._src import layers_and_loss_tags as tags
from kfac_jax._src import tracer
from kfac_jax._src import utils
from kfac_jax._src.curvature_blocks import kronecker_factored
from kfac_jax._src.curvature_blocks import utils as cb_utils


# Types for annotation
Array = utils.Array
Numeric = utils.Numeric
KroneckerFactored = kronecker_factored.KroneckerFactored


class NaiveTNT(KroneckerFactored):
  """A standard TNT block for a single parameter, or weights + bias.

  Each factor of the standard TNT curvature approximation estimates the expected
  value of the contraction of the gradients with themselves along all but a
  single axis `i`:
    ``F_i ~~ E[contract_all_but_one(g, g, i)].``
  where `contrat_all_but_one` is defined as the contraction over all axes except
  the i-th of its first two inputs, e.g.:
    ``contract_all_but_one(A, B, 1)[a,b] = sum_{i,j} A[i, a, j] B[i, b, j]``

  The estimation is performed in a naive way by contracting the sum of each
  examples' gradients and then dividing by the batch size:
    ``F_i = contract_all_but_one(sum_n g_n, sum_n g_n, i) / N``
  where `g_n` is the model gradient of a single example and `N` is the
  batch size. Since the expectations of the gradients over the model
  distribution is zero and they are independent across cases, this is still an
  unbiased estimator.
  """

  def state_dependent_scale(
      self,
      state: "NaiveTNT.State",
  ) -> Numeric:
    return utils.tnt_scale([factor.value for factor in state.factors])

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

    dw = self.parameters_shaped_list_to_array(estimation_data.tangents.params)

    assert dw.ndim == len(state.factors)

    in_str = cb_utils.ALPHABET[: dw.ndim]

    for i, factor in enumerate(state.factors):
      # For factor i we contract the gradient with itself along all axes,
      # except the i-th.

      lhs_str = utils.replace_char(in_str, "y", i)
      rhs_str = utils.replace_char(in_str, "z", i)

      # This is a rank-1 mod since it's like we flattened all but dim i together
      # and then did an outer product
      factor_update = (
          jnp.einsum(f"{lhs_str},{rhs_str}->yz", dw, dw) / batch_size
      )

      factor.update(factor_update, ema_old, ema_new)

    return state


class DenseTNT(kronecker_factored.DenseTwoKroneckerFactored):
  """A TNT block for dense layers.

  This TNT block modifies :class:`~NaiveTNTBlock` by the way it estimates each
  factor specifically for a dense layer. Instead of using the contraction over
  the summed gradient, it performs the contraction over each individual batch
  elements and then averages over the batch:
    ``F_i = sum_n contract_all_but_one(g_n, g_n, i) / N``
  The estimator is unbiased, and will have lower variance then the naive one.
  """

  def state_dependent_scale(self, state: "DenseTNT.State") -> Numeric:
    return utils.tnt_scale([factor.value for factor in state.factors])

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

    if self.number_of_parameters == 2:
      x_one = jnp.ones_like(x[:, :1])
      x = jnp.concatenate([x, x_one], axis=1)

    # We multiply each x by the norm_y, and each dy by the norm of x
    dy_norms = jnp.linalg.norm(dy, axis=-1, keepdims=True)
    x_norms = jnp.linalg.norm(x, axis=-1, keepdims=True)
    x = x * dy_norms
    dy = dy * x_norms

    input_stats = jnp.einsum("ay,az->yz", x, x) / batch_size
    output_stats = jnp.einsum("ay,az->yz", dy, dy) / batch_size

    state.factors[0].update(input_stats, ema_old, ema_new)
    state.factors[1].update(output_stats, ema_old, ema_new)

    return state


class Conv2DTNT(kronecker_factored.Conv2DTwoKroneckerFactored):
  """A TNT block for Conv2D layers.

  This TNT block modifies :class:`~NaiveTNTBlock` by the way it estimates each
  factor specifically for a conv2D layer. Importantly, it assumes "location
  independence" similar to :class:~`Conv2DTwoKroneckerFactored`. Given this
  assumption, instead of using the contraction over the summed gradient, it
  performs the contraction for each individual example in the batch, and each
  individual spatial location, and then averages over these:
    ``F_i = sum_n sum_t contract_all_but_one(g_{n,t}, g_{n,t}, i) / (N * T)``
  where T here is the number of spatial locations. The estimator is unbiased
  (under the "location independence" approximation), and will have lower
  variance then the naive one.

  If the argument `weighting_per_location` is set to `False`, then the block
  uses a mixture between location-independence and not, in the sense that it
  computes the contractions per example, while the matrix factor statistics
  still assume location independence.
  """

  def __init__(
      self,
      layer_tag_eq: tags.LayerTagEqn,
      weighting_per_location: bool = True,
      parameters_specs: Sequence[str] | None = None,
      parameters_concat_axis: int = 0,
  ):
    self.weighting_per_location = weighting_per_location
    super().__init__(
        layer_tag_eq=layer_tag_eq,
        parameters_specs=parameters_specs,
        parameters_concat_axis=parameters_concat_axis,
    )

  def state_dependent_scale(
      self, state: "Conv2DTNT.State"
  ) -> Numeric:
    return utils.tnt_scale([factor.value for factor in state.factors])

  def x_squared_spatial_norms(self, x: Array) -> Array:

    kernel_shape = list(self.parameters_shapes[0])
    kernel_shape[self.kernel_output_axis] = 1

    return jax.lax.conv_general_dilated(
        lhs=x * x,
        rhs=jnp.ones(kernel_shape),
        window_strides=self.layer_tag_extra_params["window_strides"],
        padding=self.layer_tag_extra_params["padding"],
        lhs_dilation=self.layer_tag_extra_params["lhs_dilation"],
        rhs_dilation=self.layer_tag_extra_params["rhs_dilation"],
        dimension_numbers=self.layer_tag_extra_params["dimension_numbers"],
        feature_group_count=self.layer_tag_extra_params["feature_group_count"],
        precision=self.layer_tag_extra_params["precision"],
        preferred_element_type=
        self.layer_tag_extra_params["preferred_element_type"],
    )

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

    # We multiply each x by the norm_y, and each dy by the norm of x
    dy_sq_norms = jnp.sum(dy * dy, axis=self.outputs_channel_index)
    x_sq_norms = self.x_squared_spatial_norms(x)

    if self.number_of_parameters == 2:
      # When we have a bias we need to add 1 coming from it to the squared norm
      x_sq_norms = x_sq_norms + 1

    if not self.weighting_per_location:
      dy_sq_norms = jnp.sum(dy_sq_norms, axis=[1, 2])
      x_sq_norms = jnp.sum(x_sq_norms, axis=[1, 2, 3], keepdims=True)

    input_cov = self.compute_inputs_stats(x, weighting_array=dy_sq_norms)
    output_cov = self.compute_outputs_stats(dy * jnp.sqrt(x_sq_norms))

    state.factors[0].update(input_cov, ema_old, ema_new)
    state.factors[1].update(output_cov, ema_old, ema_new)

    return state
