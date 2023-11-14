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
"""Testing functionalities of the curvature estimation."""
import functools
from typing import Callable, Mapping

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import kfac_jax
from tests import models
import numpy as np


Array = kfac_jax.utils.Array
PRNGKey = kfac_jax.utils.PRNGKey
Shape = kfac_jax.utils.Shape
StateType = kfac_jax.curvature_estimator.StateType


NON_LINEAR_MODELS_AND_CURVATURE_TYPE = [
    model + ("ggn",) for model in models.NON_LINEAR_MODELS
] + [
    model + ("fisher",) for model in models.NON_LINEAR_MODELS
]


LINEAR_MODELS_AND_CURVATURE_TYPE = [
    model + ("ggn",) for model in models.LINEAR_MODELS
] + [
    model + ("fisher",) for model in models.LINEAR_MODELS
]


PIECEWISE_LINEAR_MODELS_AND_CURVATURE = [
    model + ("ggn",) for model in models.PIECEWISE_LINEAR_MODELS
] + [
    model + ("fisher",) for model in models.PIECEWISE_LINEAR_MODELS
]


@functools.partial(jax.jit, static_argnums=(0, 3, 4))
def compute_exact_approx_curvature(
    estimator: kfac_jax.CurvatureEstimator[StateType],
    rng: PRNGKey,
    func_args: kfac_jax.utils.FuncArgs,
    batch_size: int,
    curvature_type: str,
) -> StateType:
  """Computes the full Fisher matrix approximation for the estimator."""
  state = estimator.init(
      rng=rng,
      func_args=func_args,
      exact_powers_to_cache=None,
      approx_powers_to_cache=None,
      cache_eigenvalues=False,
  )
  state = estimator.update_curvature_matrix_estimate(
      state=state,
      ema_old=0.0,
      ema_new=1.0,
      batch_size=batch_size,
      rng=rng,
      func_args=func_args,
      estimation_mode=f"{curvature_type}_exact",
  )
  estimator.sync(state, pmap_axis_name="i")
  return state


class TestEstimator(parameterized.TestCase):
  """Testing of different curvature estimators."""

  def assertAllClose(
      self,
      x: kfac_jax.utils.PyTree,
      y: kfac_jax.utils.PyTree,
      check_dtypes: bool = True,
      atol: float = 1e-6,
      rtol: float = 1e-6,
  ):
    """Asserts that the two PyTrees are close up to the provided tolerances."""
    x_v, x_tree = jax.tree_util.tree_flatten(x)
    y_v, y_tree = jax.tree_util.tree_flatten(y)
    self.assertEqual(x_tree, y_tree)
    for xi, yi in zip(x_v, y_v):
      self.assertEqual(xi.shape, yi.shape)
      if check_dtypes:
        self.assertEqual(xi.dtype, yi.dtype)
      np.testing.assert_allclose(xi, yi, rtol=rtol, atol=atol, equal_nan=False)

  @parameterized.parameters(NON_LINEAR_MODELS_AND_CURVATURE_TYPE)
  def test_explicit_exact_full(
      self,
      init_func: Callable[..., models.hk.Params],
      model_func: Callable[..., Array],
      data_point_shapes: Mapping[str, Shape],
      seed: int,
      curvature_type: str,
      data_size: int = 4,
  ):
    """Tests the explicit exact estimator matches the implicit one."""
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

    # Compute curvature matrix using the explicit exact curvature
    explicit_estimator = kfac_jax.ExplicitExactCurvature(model_func)
    state = compute_exact_approx_curvature(
        explicit_estimator,
        estimator_key,
        func_args,
        data_size,
        curvature_type,
    )
    explicit_exact_matrix = explicit_estimator.to_dense_matrix(state)

    # Compute exact curvature matrix using the implicit curvature
    implicit = kfac_jax.ImplicitExactCurvature(model_func)
    zeros_vector = jnp.zeros([explicit_estimator.dim])
    @jax.jit
    def mul_e_i(index, *_):
      flat_v_e = zeros_vector.at[index].set(1.0)
      v_e_leaves = []
      i = 0
      for p in jax.tree_util.tree_leaves(params):
        v_e_leaves.append(flat_v_e[i: i + p.size].reshape(p.shape))
        i += p.size
      v_e = jax.tree_util.tree_unflatten(
          jax.tree_util.tree_structure(params), v_e_leaves)
      if curvature_type == "fisher":
        r_e = implicit.multiply_fisher(func_args, v_e)
      elif curvature_type == "ggn":
        r_e = implicit.multiply_ggn(func_args, v_e)
      else:
        raise ValueError(f"Unrecognized curvature_type={curvature_type}.")
      flat_r_e = jax.tree_util.tree_leaves(
          jax.tree_util.tree_map(lambda x: x.flatten(), r_e))
      return index + 1, jnp.concatenate(flat_r_e, axis=0)
    _, matrix = jax.lax.scan(mul_e_i, 0, None, length=explicit_estimator.dim)

    # Compare
    self.assertAllClose(matrix, explicit_exact_matrix)

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
    block_estimator = kfac_jax.BlockDiagonalCurvature(
        model_func,
        layer_tag_to_block_ctor=dict(
            dense_tag=kfac_jax.DenseFull,
            conv2d_tag=kfac_jax.Conv2DFull,
            scale_and_shift_tag=kfac_jax.ScaleAndShiftFull,
        )
    )
    block_state = compute_exact_approx_curvature(
        block_estimator,
        estimator_key,
        func_args,
        data_size,
        curvature_type,
    )
    blocks = block_estimator.to_diagonal_block_dense_matrix(block_state)

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
      self.assertAllClose(block, full_matrix[s, s])
      d = d + block.shape[0]
    self.assertEqual(d, full_matrix.shape[0])

  @parameterized.parameters(PIECEWISE_LINEAR_MODELS_AND_CURVATURE)
  def test_block_diagonal_full_to_hessian(
      self,
      init_func: Callable[..., models.hk.Params],
      model_func: Callable[..., Array],
      data_point_shapes: Mapping[str, Shape],
      seed: int,
      curvature_type: str,
      data_size: int = 4,
  ):
    """Tests for piecewise linear models that block equal to the Hessian."""
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

    block_estimator = kfac_jax.BlockDiagonalCurvature(
        model_func,
        layer_tag_to_block_ctor=dict(
            dense_tag=kfac_jax.DenseFull,
            conv2d_tag=kfac_jax.Conv2DFull,
            scale_and_shift_tag=kfac_jax.ScaleAndShiftFull,
        )
    )
    block_state = compute_exact_approx_curvature(
        block_estimator,
        estimator_key,
        func_args,
        data_size,
        curvature_type,
    )
    blocks = (block_estimator.to_diagonal_block_dense_matrix(block_state))

    # Compute exact curvature matrix using the implicit curvature
    implicit = kfac_jax.ImplicitExactCurvature(model_func)
    zeros_vector = jnp.zeros([block_estimator.dim])

    @jax.jit
    def mul_e_i(index, *_):
      flat_v_e = zeros_vector.at[index].set(1.0)
      v_e_leaves = []
      i = 0
      for p in jax.tree_util.tree_leaves(params):
        v_e_leaves.append(flat_v_e[i: i + p.size].reshape(p.shape))
        i += p.size
      v_e = jax.tree_util.tree_unflatten(
          jax.tree_util.tree_structure(params), v_e_leaves)
      r_e = implicit.multiply_hessian(func_args, v_e)
      flat_r_e = jax.tree_util.tree_leaves(
          jax.tree_map(lambda x: x.flatten(), r_e))
      return index + 1, jnp.concatenate(flat_r_e, axis=0)

    _, hessian = jax.lax.scan(mul_e_i, 0, None, length=block_estimator.dim)

    # Compare blocks
    d = 0
    for block in blocks:
      s = slice(d, d + block.shape[0])
      self.assertAllClose(block, hessian[s, s])
      d = d + block.shape[0]
    self.assertEqual(d, hessian.shape[0])

  @parameterized.parameters(NON_LINEAR_MODELS_AND_CURVATURE_TYPE)
  def test_diagonal(
      self,
      init_func: Callable[..., models.hk.Params],
      model_func: Callable[..., Array],
      data_point_shapes: Mapping[str, Shape],
      seed: int,
      curvature_type: str,
      data_size: int = 4,
  ):
    """Tests that the diagonal estimation is the diagonal of the full."""
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

    # Compute curvature matrix using the block diagonal diagonal estimator
    diagonal_estimator = kfac_jax.BlockDiagonalCurvature(
        model_func,
        layer_tag_to_block_ctor=dict(
            dense_tag=kfac_jax.DenseDiagonal,
            conv2d_tag=kfac_jax.Conv2DDiagonal,
            scale_and_shift_tag=kfac_jax.ScaleAndShiftDiagonal,
        )
    )
    diag_state = compute_exact_approx_curvature(
        diagonal_estimator,
        estimator_key,
        func_args,
        data_size,
        curvature_type,
    )
    diagonals = diagonal_estimator.to_diagonal_block_dense_matrix(diag_state)

    # Compute curvature matrix using the block diagonal full estimator
    block_estimator = kfac_jax.BlockDiagonalCurvature(
        model_func,
        layer_tag_to_block_ctor=dict(
            dense_tag=kfac_jax.DenseFull,
            conv2d_tag=kfac_jax.Conv2DFull,
            scale_and_shift_tag=kfac_jax.ScaleAndShiftFull,
        )
    )
    block_state = compute_exact_approx_curvature(
        block_estimator,
        estimator_key,
        func_args,
        data_size,
        curvature_type,
    )
    blocks = block_estimator.to_diagonal_block_dense_matrix(block_state)

    # Compare diagonals
    self.assertEqual(len(diagonals), len(blocks))
    for diagonal, block in zip(diagonals, blocks):
      self.assertAllClose(diagonal, jnp.diag(jnp.diag(block)))

  @parameterized.parameters(LINEAR_MODELS_AND_CURVATURE_TYPE)
  def test_kronecker_factored(
      self,
      init_func: Callable[..., models.hk.Params],
      model_func: Callable[..., Array],
      data_point_shapes: Mapping[str, Shape],
      seed: int,
      curvature_type: str = "fisher",
      data_size: int = 4,
  ):
    """Test for linear network if the KF blocks match the full."""
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

    # Compute curvature matrix using the kronecker factored blocks
    # Note that this identity does not hold for conv layers, as there the
    # KF approximation assumes independence between locations as well.
    kf_estimator = kfac_jax.BlockDiagonalCurvature(
        model_func,
        layer_tag_to_block_ctor=dict(
            dense_tag=kfac_jax.DenseTwoKroneckerFactored,
            conv2d_tag=None,
            scale_and_shift_tag=kfac_jax.ScaleAndShiftFull,
        )
    )

    kf_state = compute_exact_approx_curvature(
        kf_estimator,
        estimator_key,
        func_args,
        data_size,
        curvature_type,
    )
    kf_blocks = kf_estimator.to_diagonal_block_dense_matrix(kf_state)

    # Compute curvature matrix using the block diagonal full estimator
    full_estimator = kfac_jax.BlockDiagonalCurvature(
        model_func,
        layer_tag_to_block_ctor=dict(
            dense_tag=kfac_jax.DenseFull,
            conv2d_tag=kfac_jax.Conv2DFull,
            scale_and_shift_tag=kfac_jax.ScaleAndShiftFull,
        )
    )
    full_state = compute_exact_approx_curvature(
        full_estimator,
        estimator_key,
        func_args,
        data_size,
        curvature_type,
    )
    blocks = full_estimator.to_diagonal_block_dense_matrix(full_state)

    # Compare diagonals
    self.assertEqual(len(kf_blocks), len(blocks))
    for kf, block in zip(kf_blocks, blocks):
      self.assertAllClose(kf, block)

  @parameterized.parameters([
      (
          dict(images=(32, 32, 3), labels=(10,)),
          1230971,
          "ggn",
      ),
      (
          dict(images=(32, 32, 3), labels=(10,)),
          1230971,
          "fisher",
      ),
  ])
  def test_eigenvalues(
      self,
      data_point_shapes: Mapping[str, Shape],
      seed: int,
      curvature_type: str = "fisher",
      data_size: int = 4,
  ):
    """Test for linear network if the KF blocks match the full."""
    num_classes = data_point_shapes["labels"][0]
    init_func = models.conv_classifier(
        num_classes=num_classes, layer_channels=[8, 16, 32]).init
    model_func = functools.partial(
        models.conv_classifier_loss,
        num_classes=num_classes,
        layer_channels=[8, 16, 32])

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
    estimator = kfac_jax.BlockDiagonalCurvature(
        model_func,
        index_to_block_ctor={
            (0, 1): kfac_jax.Conv2DTwoKroneckerFactored,
            (3, 2): kfac_jax.Conv2DDiagonal,
            (4,): kfac_jax.ScaledIdentity,
            (6, 5): kfac_jax.ScaleAndShiftDiagonal,
            (8, 7): kfac_jax.DenseTwoKroneckerFactored,
            (10, 9): kfac_jax.DenseFull,
        }
    )

    state = compute_exact_approx_curvature(
        estimator,
        estimator_key,
        func_args,
        data_size,
        curvature_type,
    )

    cached_state = estimator.update_cache(
        state,
        identity_weight=1e-2,
        exact_powers=-1,
        approx_powers=None,
        eigenvalues=True,
        pmap_axis_name=None,
    )

    block_eigenvalues = estimator.block_eigenvalues(cached_state, True)
    scales = [block.fixed_scale() for block in estimator.blocks]

    self.assertLen(block_eigenvalues, estimator.num_blocks)
    for block_state, eigs, scale in zip(
        cached_state.blocks_states, block_eigenvalues, scales):
      if isinstance(block_state, kfac_jax.TwoKroneckerFactored.State):
        in_eigs, _ = kfac_jax.utils.safe_psd_eigh(
            block_state.factors[1].value)
        out_eigs, _ = kfac_jax.utils.safe_psd_eigh(
            block_state.factors[0].value)
        self.assertAllClose(scale * jnp.outer(out_eigs, in_eigs), eigs)
      elif isinstance(block_state, kfac_jax.Diagonal.State):
        diag_eigs = jnp.concatenate([factor.value.flatten() for factor in
                                     block_state.diagonal_factors])
        self.assertAllClose(diag_eigs, eigs)
      elif isinstance(block_state, kfac_jax.Full.State):
        matrix_eigs, _ = kfac_jax.utils.safe_psd_eigh(block_state.matrix.value)
        self.assertAllClose(matrix_eigs, eigs)
      elif isinstance(block_state, kfac_jax.CurvatureBlock.State):
        # ScaledIdentity
        self.assertAllClose(jnp.ones_like(eigs), eigs)
      else:
        raise NotImplementedError()

  @parameterized.parameters([
      (
          dict(images=(32, 32, 3), labels=(10,)),
          1230971,
          "ggn",
      ),
      (
          dict(images=(32, 32, 3), labels=(10,)),
          1230971,
          "fisher",
      ),
  ])
  def test_matmul(
      self,
      data_point_shapes: Mapping[str, Shape],
      seed: int,
      curvature_type: str,
      data_size: int = 4,
      e: float = 1.0,
  ):
    """Test for linear network if the KF blocks match the full."""
    num_classes = data_point_shapes["labels"][0]
    init_func = models.conv_classifier(
        num_classes=num_classes, layer_channels=[8, 16, 32]).init
    model_func = functools.partial(
        models.conv_classifier_loss,
        num_classes=num_classes,
        layer_channels=[8, 16, 32])

    rng_key = jax.random.PRNGKey(seed)
    init_key1, init_key2, data_key, estimator_key = jax.random.split(rng_key, 4)

    # Generate data
    data = {}
    for name, shape in data_point_shapes.items():
      data_key, key = jax.random.split(data_key)
      data[name] = jax.random.uniform(key, (data_size, *shape))
      if name == "labels":
        data[name] = jnp.argmax(data[name], axis=-1)

    params = init_func(init_key1, data)
    func_args = (params, data)
    estimator = kfac_jax.BlockDiagonalCurvature(
        model_func,
        index_to_block_ctor={
            (1, 0): kfac_jax.Conv2DTwoKroneckerFactored,
            (3, 2): kfac_jax.Conv2DDiagonal,
            (4,): kfac_jax.ScaledIdentity,
            (6, 5): kfac_jax.ScaleAndShiftDiagonal,
            (8, 7): kfac_jax.DenseTwoKroneckerFactored,
            (10, 9): kfac_jax.DenseFull,
        }
    )

    state = compute_exact_approx_curvature(
        estimator,
        estimator_key,
        func_args,
        data_size,
        curvature_type,
    )

    cached_state = estimator.update_cache(
        state,
        identity_weight=e,
        exact_powers=-1,
        approx_powers=None,
        eigenvalues=True,
        pmap_axis_name=None,
    )

    v = init_func(init_key2, data)
    m_v = estimator.multiply(state, v, e, True, True, None)
    m_inv_v = estimator.multiply_inverse(cached_state, v, e, True, True, None)

    # Check cached and non-cached are the same
    m_inv_v2 = estimator.multiply_inverse(state, v, e, True, False, None)
    self.assertAllClose(m_inv_v, m_inv_v2, atol=1e-5, rtol=1e-4)

    block_vectors = estimator.params_vector_to_blocks_vectors(v)
    results = estimator.params_vector_to_blocks_vectors(m_v)
    results_inv = estimator.params_vector_to_blocks_vectors(m_inv_v)
    block_matrices = estimator.to_diagonal_block_dense_matrix(state)

    for i in range(estimator.num_blocks):
      # In all modules the parameters are in reverse canonical order
      v_i_flat = jnp.concatenate([x.flatten() for x in block_vectors[i][::-1]])
      r_i_flat = jnp.concatenate([x.flatten() for x in results[i][::-1]])
      r2_i_flat = jnp.concatenate([x.flatten() for x in results_inv[i][::-1]])

      # Matrix multiplication
      computed = block_matrices[i] @ v_i_flat + e * v_i_flat
      self.assertAllClose(computed, r_i_flat)

      # Matrix inverse multiplication
      m_i_plus_eye = block_matrices[i] + e * jnp.eye(block_matrices[i].shape[0])
      computed2 = jnp.linalg.solve(m_i_plus_eye, v_i_flat)
      self.assertAllClose(computed2, r2_i_flat, atol=1e-5, rtol=1e-4)

  @parameterized.parameters([
      (
          dict(images=(32, 32, 3), labels=(10,)),
          1230971,
          "ggn",
      ),
      (
          dict(images=(32, 32, 3), labels=(10,)),
          1230971,
          "fisher",
      ),
  ])
  def test_implicit_factor_products(
      self,
      data_point_shapes: Mapping[str, Shape],
      seed: int,
      curvature_type: str,
      data_size: int = 4,
  ):
    """Tests that the products of the curvature factors are correct."""
    num_classes = data_point_shapes["labels"][0]
    init_func = models.conv_classifier(
        num_classes=num_classes, layer_channels=[8, 16, 32]).init
    model_func = functools.partial(
        models.conv_classifier_loss,
        num_classes=num_classes,
        layer_channels=[8, 16, 32])

    rng_key = jax.random.PRNGKey(seed)
    init_key1, init_key2, data_key = jax.random.split(rng_key, 3)

    # Generate data
    data = {}
    for name, shape in data_point_shapes.items():
      data_key, key = jax.random.split(data_key)
      data[name] = jax.random.uniform(key, (data_size, *shape))
      if name == "labels":
        data[name] = jnp.argmax(data[name], axis=-1)

    params = init_func(init_key1, data)
    func_args = (params, data)
    estimator = kfac_jax.ImplicitExactCurvature(model_func)

    v = init_func(init_key2, data)
    if curvature_type == "fisher":
      c_factor_v = estimator.multiply_fisher_factor_transpose(func_args, v)
      c_v_1 = estimator.multiply_fisher_factor(func_args, c_factor_v)
      c_v_2 = estimator.multiply_fisher(func_args, v)
    elif curvature_type == "ggn":
      c_factor_v = estimator.multiply_ggn_factor_transpose(func_args, v)
      c_v_1 = estimator.multiply_ggn_factor(func_args, c_factor_v)
      c_v_2 = estimator.multiply_ggn(func_args, v)
    else:
      raise NotImplementedError()

    self.assertAllClose(c_v_1, c_v_2, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
  absltest.main()
