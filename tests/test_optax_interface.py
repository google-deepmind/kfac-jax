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
"""Module for testing the optax interface to K-FAC."""
import functools
from typing import Callable, Mapping

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import kfac_jax
from tests import estimator_test_utils
from tests import models
import numpy as np

Array = kfac_jax.utils.Array
PRNGKey = kfac_jax.utils.PRNGKey
Shape = kfac_jax.utils.Shape
StateType = kfac_jax.curvature_estimator.StateType

NON_LINEAR_MODELS_AND_CURVATURE_TYPE = (
    estimator_test_utils.NON_LINEAR_MODELS_AND_CURVATURE_TYPE
    )

compute_exact_approx_curvature = (
    estimator_test_utils.compute_exact_approx_curvature
    )


@functools.partial(jax.jit, static_argnums=(0,))
def compute_exact_approx_curvature_precon(
    preconditioner: kfac_jax.OptaxPreconditioner,
    rng: PRNGKey,
    func_args: kfac_jax.optimizer.FuncArgsVariants,
) -> kfac_jax.OptaxPreconditionState:
  """Computes the full Fisher matrix approximation for the estimator."""
  return preconditioner.maybe_update_estimator_curvature(
      state=preconditioner.init(func_args=func_args, rng=rng),
      func_args=func_args,
      rng=rng,
      decay_old_ema=True,
      sync=True,
  )


class TestOptaxPreconditioner(parameterized.TestCase):
  """Testing the optax interface to K-FAC."""

  def assert_trees_all_close(
      self,
      x: kfac_jax.utils.PyTree,
      y: kfac_jax.utils.PyTree,
      check_dtypes: bool = True,
      atol: float = 1e-6,
      rtol: float = 1e-6,
  ):
    """Asserts that the two PyTrees are close up to the provided tolerances."""
    if jax.devices()[0].platform in ("tpu", "gpu"):
      rtol = 3e3 * rtol
      atol = 3e3 * atol

    x_v, x_tree = jax.tree_util.tree_flatten(x)
    y_v, y_tree = jax.tree_util.tree_flatten(y)
    self.assertEqual(x_tree, y_tree)
    for xi, yi in zip(x_v, y_v):
      self.assertEqual(xi.shape, yi.shape)
      if check_dtypes:
        self.assertEqual(xi.dtype, yi.dtype)
      np.testing.assert_allclose(xi, yi, rtol=rtol, atol=atol, equal_nan=False)

  @parameterized.parameters(NON_LINEAR_MODELS_AND_CURVATURE_TYPE)
  def test_block_diagonal_full(
      self,
      init_func: Callable[..., models.hk.Params],
      model_func: Callable[..., Array],
      data_point_shapes: Mapping[str, Shape],
      seed: int,
      curvature_type: str,
      data_size: int = 4,
  ):
    """Tests that the block diagonal full is equal to the explicit curvature."""
    rng_key = jax.random.PRNGKey(seed)
    init_key, data_key, estimator_key = jax.random.split(rng_key, 3)

    # Generate data
    data = {}
    for name, shape in data_point_shapes.items():
      data_key, key = jax.random.split(data_key)
      data[name] = jax.random.uniform(key, (data_size, *shape))
      if name == "labels":
        data[name] = jnp.argmax(data[name], axis=-1)

    params = init_func(init_key, data)
    func_args = (params, data)

    # Compute curvature matrix using the block diagonal full estimator
    optax_estimator = kfac_jax.OptaxPreconditioner(
        model_func,
        damping=0.0,
        curvature_ema=0.0,
        layer_tag_to_block_ctor=dict(
            dense=kfac_jax.DenseFull,
            conv2d=kfac_jax.Conv2DFull,
            scale_and_shift=kfac_jax.ScaleAndShiftFull,
        ),
        pmap_axis_name="i",
        estimation_mode=f"{curvature_type}_exact",
    )

    precondition_state = compute_exact_approx_curvature_precon(
        preconditioner=optax_estimator,
        rng=estimator_key,
        func_args=func_args,
    )

    block_estimator = optax_estimator.estimator
    blocks = block_estimator.to_diagonal_block_dense_matrix(
        precondition_state.estimator_state
        )

    # Compute curvature matrix using the explicit exact curvature
    full_estimator = kfac_jax.ExplicitExactCurvature(
        model_func, default_estimation_mode="fisher_exact",
        param_order=block_estimator.param_order
    )
    state = compute_exact_approx_curvature(
        full_estimator,
        estimator_key,
        func_args,
        data_size,
        curvature_type,
    )
    full_matrix = full_estimator.to_dense_matrix(state)

    # Compare blocks
    d = 0
    for block in blocks:
      s = slice(d, d + block.shape[0])
      self.assert_trees_all_close(block, full_matrix[s, s])
      d = d + block.shape[0]
    self.assertEqual(d, full_matrix.shape[0])

  def test_damping_schedule(self):
    """Tests OptaxPreconditioner with a step-dependent damping_schedule."""
    layer_widths = [4, 4]
    init_func = models.autoencoder(layer_widths, output_dim=3).init
    model_func = functools.partial(
        models.autoencoder_deterministic_loss, layer_widths=layer_widths
    )
    init_key, data_key, estimator_key = jax.random.split(
        jax.random.PRNGKey(0), 3
    )
    data = {"images": jax.random.uniform(data_key, (4, 3))}
    params = init_func(init_key, data)
    func_args = (params, data)
    grads = jax.grad(model_func)(params, data)

    schedule = lambda count: 0.1 * (count + 1)
    sched_precon = kfac_jax.OptaxPreconditioner(
        model_func,
        damping_schedule=schedule,
        l2_reg=1e-3,
        curvature_ema=0.0,
        inverse_update_period=2,
        estimation_mode="fisher_exact",
        distributed_inverses=False,
        distributed_precon_apply=False,
    )
    state = sched_precon.init(func_args, estimator_key)
    state = sched_precon.maybe_update(state, func_args, estimator_key)

    fixed_step0 = kfac_jax.OptaxPreconditioner(
        model_func,
        damping=0.1,
        l2_reg=1e-3,
        curvature_ema=0.0,
        inverse_update_period=2,
        estimation_mode="fisher_exact",
        distributed_inverses=False,
        distributed_precon_apply=False,
    )
    fixed_state0 = fixed_step0.init(func_args, estimator_key)
    fixed_state0 = fixed_step0.maybe_update(
        fixed_state0, func_args, estimator_key
    )
    self.assert_trees_all_close(
        sched_precon.apply(grads, state),
        fixed_step0.apply(grads, fixed_state0),
    )

    # Advance two steps so count == 2 and the inverse cache updates at step 2.
    state = sched_precon.increment_count(sched_precon.increment_count(state))
    state = sched_precon.maybe_update(state, func_args, estimator_key)

    fixed_step2 = kfac_jax.OptaxPreconditioner(
        model_func,
        damping=0.3,
        l2_reg=1e-3,
        curvature_ema=0.0,
        inverse_update_period=2,
        estimation_mode="fisher_exact",
        distributed_inverses=False,
        distributed_precon_apply=False,
    )
    fixed_state2 = fixed_step2.init(func_args, estimator_key)
    fixed_state2 = fixed_step2.maybe_update(
        fixed_state2, func_args, estimator_key
    )
    self.assert_trees_all_close(
        sched_precon.apply(grads, state),
        fixed_step2.apply(grads, fixed_state2),
    )

  def test_use_sqrt_inv(self):
    """Tests OptaxPreconditioner with use_sqrt_inv=True."""
    layer_widths = [4, 4]
    init_func = models.autoencoder(layer_widths, output_dim=3).init
    model_func = functools.partial(
        models.autoencoder_deterministic_loss, layer_widths=layer_widths
    )
    init_key, data_key, estimator_key = jax.random.split(
        jax.random.PRNGKey(1), 3
    )
    data = {"images": jax.random.uniform(data_key, (4, 3))}
    params = init_func(init_key, data)
    func_args = (params, data)
    grads = jax.grad(model_func)(params, data)

    # Verify default approximate Kronecker cache initializes power -0.5.
    approx_cached = kfac_jax.OptaxPreconditioner(
        model_func,
        damping=0.1,
        use_sqrt_inv=True,
        distributed_inverses=False,
        distributed_precon_apply=False,
    )
    approx_init_state = approx_cached.init(func_args, estimator_key)
    zero_updates = approx_cached.apply(grads, approx_init_state)
    self.assert_trees_all_close(
        zero_updates, jax.tree_util.tree_map(jnp.zeros_like, grads)
    )

    # Verify exact cached inverse square root matches uncached and squares to
    # the full inverse: (F + damping * I)^(-1/2)^2 g == (F + damping * I)^(-1) g
    for register_only_generic in (False, True):
      sqrt_cached = kfac_jax.OptaxPreconditioner(
          model_func,
          damping=0.1,
          curvature_ema=0.0,
          inverse_update_period=5,
          use_exact_inverses=not register_only_generic,
          use_sqrt_inv=True,
          register_only_generic=register_only_generic,
          estimation_mode="fisher_exact",
          distributed_inverses=False,
          distributed_precon_apply=False,
      )
      sqrt_uncached = kfac_jax.OptaxPreconditioner(
          model_func,
          damping=0.1,
          curvature_ema=0.0,
          inverse_update_period=1,
          use_exact_inverses=not register_only_generic,
          use_sqrt_inv=True,
          register_only_generic=register_only_generic,
          estimation_mode="fisher_exact",
          distributed_inverses=False,
          distributed_precon_apply=False,
      )
      inv_cached = kfac_jax.OptaxPreconditioner(
          model_func,
          damping=0.1,
          curvature_ema=0.0,
          inverse_update_period=5,
          use_exact_inverses=not register_only_generic,
          use_sqrt_inv=False,
          register_only_generic=register_only_generic,
          estimation_mode="fisher_exact",
          distributed_inverses=False,
          distributed_precon_apply=False,
      )

      sqrt_cached_state = sqrt_cached.maybe_update(
          sqrt_cached.init(func_args, estimator_key), func_args, estimator_key
      )
      sqrt_uncached_state = sqrt_uncached.maybe_update(
          sqrt_uncached.init(func_args, estimator_key), func_args, estimator_key
      )
      inv_cached_state = inv_cached.maybe_update(
          inv_cached.init(func_args, estimator_key), func_args, estimator_key
      )

      cached_out = sqrt_cached.apply(grads, sqrt_cached_state)
      uncached_out = sqrt_uncached.apply(grads, sqrt_uncached_state)
      self.assert_trees_all_close(cached_out, uncached_out, atol=1e-5)

      twice_out = sqrt_cached.apply(cached_out, sqrt_cached_state)
      inv_out = inv_cached.apply(grads, inv_cached_state)
      self.assert_trees_all_close(twice_out, inv_out, atol=1e-5)

    # Verify exact cached power -1 on Full curvature blocks matches uncached.
    full_cached = kfac_jax.OptaxPreconditioner(
        model_func,
        damping=0.1,
        curvature_ema=0.0,
        inverse_update_period=5,
        use_exact_inverses=True,
        use_sqrt_inv=False,
        layer_tag_to_block_ctor=dict(dense=kfac_jax.DenseFull),
        estimation_mode="fisher_exact",
        distributed_inverses=False,
        distributed_precon_apply=False,
    )
    full_uncached = kfac_jax.OptaxPreconditioner(
        model_func,
        damping=0.1,
        curvature_ema=0.0,
        inverse_update_period=1,
        use_exact_inverses=True,
        use_sqrt_inv=False,
        layer_tag_to_block_ctor=dict(dense=kfac_jax.DenseFull),
        estimation_mode="fisher_exact",
        distributed_inverses=False,
        distributed_precon_apply=False,
    )
    full_cached_state = full_cached.maybe_update(
        full_cached.init(func_args, estimator_key), func_args, estimator_key
    )
    full_uncached_state = full_uncached.maybe_update(
        full_uncached.init(func_args, estimator_key), func_args, estimator_key
    )
    self.assert_trees_all_close(
        full_cached.apply(grads, full_cached_state),
        full_uncached.apply(grads, full_uncached_state),
        atol=1e-5,
    )



if __name__ == "__main__":
  absltest.main()
