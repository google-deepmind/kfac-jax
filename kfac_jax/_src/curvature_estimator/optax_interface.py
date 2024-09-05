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
"""Utilities for setting up different optimizers."""

import functools
from typing import Any, Callable, NamedTuple, Sequence

import jax
from jax import lax
import jax.numpy as jnp
from kfac_jax._src import utils
from kfac_jax._src.curvature_estimator import block_diagonal
from kfac_jax._src.curvature_estimator import curvature_estimator
import optax


Array = utils.Array
Numeric = utils.Numeric
PRNGKey = utils.PRNGKey
Params = utils.Params
Batch = utils.Batch
ValueFunc = utils.ValueFunc
FuncArgs = utils.FuncArgs
ScheduleType = utils.ScheduleType
EstimatorState = block_diagonal.BlockDiagonalCurvature.State


class OptaxPreconditionState(NamedTuple):
  count: Array
  estimator_state: EstimatorState


class OptaxPreconditioner:
  """An Optax-compatible K-FAC preconditioner."""

  def __init__(
      self,
      value_func: ValueFunc,
      l2_reg: Numeric = 0.0,
      damping: float | None = None,
      damping_schedule: ScheduleType | None = None,
      norm_constraint: Numeric | None = None,
      estimation_mode: str = "fisher_gradients",
      curvature_ema: Numeric = 0.95,
      curvature_update_period: int = 1,
      inverse_update_period: int = 5,
      use_exact_inverses: bool = False,
      use_sqrt_inv: bool = False,
      register_only_generic: bool = False,
      patterns_to_skip: Sequence[str] = (),
      auto_register_kwargs: dict[str, Any] | None = None,
      layer_tag_to_block_ctor: (
          dict[str, curvature_estimator.CurvatureBlockCtor] | None
      ) = None,
      pmap_axis_name: str = "kfac_axis",
      batch_size_extractor: Callable[
          [Batch], Numeric
      ] = utils.default_batch_size_extractor,
      distributed_inverses: bool = True,
      distributed_precon_apply: bool = True,
      num_samples: int = 1,
      should_vmap_samples: bool = False,
      norm_to_scale_identity_weight_per_block: str | None = None,
  ):
    """Initializes the curvature estimator and preconditioner.

    Args:
      value_func: Callable. The function should return the value of the loss to
        be optimized.
      l2_reg: Scalar. Set this value to tell the optimizer what L2
        regularization coefficient you are using (if any). Note the coefficient
        appears in the regularizer as ``coeff / 2 * sum(param**2)``. This adds
        an additional diagonal term to the curvature and hence will affect the
        quadratic model when using adaptive damping. Note that the user is still
        responsible for adding regularization to the loss. (Default: ``0.``)
      damping: Scalar. The fixed damping that will be used throughput the
        lifespan of Preconditioner. (Default: ``None``)
      damping_schedule: Callable. A schedule for the damping. This should take
        as input the current step number and return a single array that
        represents the learning rate. (Default: ``None``)
      norm_constraint: Scalar. If specified, the update is scaled down so that
        its approximate squared Fisher norm ``v^T F v`` is at most the specified
        value. (Note that here ``F`` is the approximate curvature matrix, not
        the exact.) (Default: ``None``)
      estimation_mode: String. The type of estimator to use for the curvature
        matrix. See the documentation for :class:`~CurvatureEstimator` for a
        detailed description of the possible options. (Default:
        ``fisher_gradients``).
      curvature_ema: The decay factor used when calculating the covariance
        estimate moving averages. (Default: ``0.95``)
      curvature_update_period: Int. The number of steps in between updating the
        the curvature estimates. (Default: ``1``)
      inverse_update_period: Int. The number of steps in between updating the
        the computation of the inverse curvature approximation. (Default: ``5``)
      use_exact_inverses: Bool. If ``True``, preconditioner inverses are
        computed "exactly" without the pi-adjusted factored damping approach.
        Note that this involves the use of eigendecompositions, which can
        sometimes be much more expensive. (Default: ``False``)
      use_sqrt_inv: Bool. If ``True``, we use inverse square roots for
        preconditioner instead of inverse. (Default: ``False``)
      register_only_generic: Boolean. Whether when running the auto-tagger to
        register only generic parameters, or allow it to use the graph matcher
        to automatically pick up any kind of layer tags. (Default: ``False``)
      patterns_to_skip: Tuple. A list of any patterns that should be skipped by
        the graph matcher when auto-tagging. (Default: ``()``)
      auto_register_kwargs: Any additional kwargs to be passed down to
        :func:`~auto_register_tags`, which is called by the curvature estimator.
        (Default: ``None``)
      layer_tag_to_block_ctor: Dictionary. A mapping from layer tags to block
        classes which to override the default choices of block approximation for
        that specific tag. See the documentation for
        :class:`~CurvatureEstimator` for a more detailed description. (Default:
        ``None``)
      pmap_axis_name: String. The name of the pmap axis to use when
        ``multi_device`` is set to True. (Default: ``kfac_axis``)
      batch_size_extractor: A function that takes as input the function
        arguments and returns the batch size for a single device. (Default:
        ``kfac.utils.default_batch_size_extractor``)
      distributed_inverses: Boolean. Whether to distribute the inverse
        computations (required to compute the preconditioner) across the
        different devices in a layer-wise fashion. If False, each device will
        (redundantly) perform the required computations for all of the layers.
        (Default: True)
      distributed_precon_apply: Boolean. Whether to distribute the application
        of the preconditioner across the different devices in a layer-wise
        fashion. If False, each device will (redundantly) perform the required
        operations for all of the layers. (Default: True)
      num_samples: Number of samples (per case) to use when computing stochastic
        curvature matrix estimates. This option is only used when
        ``estimation_mode == 'fisher_gradients'`` or ``estimation_mode ==
        '[fisher,ggn]_curvature_prop'``. (Default: 1)
      should_vmap_samples: Whether to use ``jax.vmap`` to compute samples
        when ``num_samples > 1``. (Default: False)
      norm_to_scale_identity_weight_per_block: The name of a norm to use to
        compute extra per-block scaling for the damping. See psd_matrix_norm()
        in utils/math.py for the definition of these. (Default: None)
    """
    self._l2_reg = l2_reg
    self._damping = damping
    self._damping_schedule = damping_schedule

    if (self._damping_schedule is None) == (self._damping is None):
      raise ValueError(
          "Only one of `damping_schedule` or `damping` has to be specified."
      )
    self._norm_constraint = norm_constraint
    self._curvature_ema = curvature_ema
    self._curvature_update_period = curvature_update_period
    self._inverse_update_period = inverse_update_period
    self._pmap_axis_name = pmap_axis_name
    self._batch_size_extractor = batch_size_extractor

    self._use_cached_inverses = self._inverse_update_period != 1
    self._use_exact_inverses = use_exact_inverses

    self._use_sqrt_inv = use_sqrt_inv

    self._norm_to_scale_identity_weight_per_block = (
        norm_to_scale_identity_weight_per_block
    )

    auto_register_kwargs = auto_register_kwargs or {}
    auto_register_kwargs.update(dict(
        register_only_generic=register_only_generic,
        patterns_to_skip=patterns_to_skip,
    ))
    # Curvature estimator
    self._estimator = block_diagonal.BlockDiagonalCurvature(
        func=value_func,
        default_estimation_mode=estimation_mode,
        params_index=0,
        layer_tag_to_block_ctor=layer_tag_to_block_ctor,
        distributed_multiplies=distributed_precon_apply,
        distributed_cache_updates=distributed_inverses,
        num_samples=num_samples,
        should_vmap_samples=should_vmap_samples,
        auto_register_kwargs=auto_register_kwargs,
    )

  def init(
      self,
      func_args: FuncArgs,
      rng: PRNGKey,
  ) -> OptaxPreconditionState:
    """Initializes the preconditioner and returns the state."""

    return OptaxPreconditionState(
        count=jnp.array(0, dtype=jnp.int32),
        estimator_state=self.estimator.init(
            rng=rng,
            func_args=func_args,
            exact_powers_to_cache=self._exact_powers_to_cache,
            approx_powers_to_cache=self._approx_powers_to_cache,
            cache_eigenvalues=False,
        ),
    )

  @property
  def _exact_powers_to_cache(self) -> int | None:
    if self._use_exact_inverses and self._use_cached_inverses:
      return -1
    return None

  @property
  def _approx_powers_to_cache(self) -> int | None:
    if not self._use_exact_inverses and self._use_cached_inverses:
      return -1
    return None

  @property
  def estimator(self) -> block_diagonal.BlockDiagonalCurvature:
    """The underlying curvature estimator used by the preconditioner."""
    return self._estimator

  @property
  def pmap_axis_name(self):
    return self._pmap_axis_name

  def get_identity_weight(
      self, state: OptaxPreconditionState
  ) -> Array | float:

    damping = self._damping

    if damping is None:
      damping = self._damping_schedule(state.count)

    return damping + self._l2_reg

  def sync_estimator_state(
      self,
      state: OptaxPreconditionState,
  ) -> OptaxPreconditionState:
    """Syncs the estimator state."""

    return OptaxPreconditionState(
        count=state.count,
        estimator_state=self.estimator.sync(
            state.estimator_state, pmap_axis_name=self.pmap_axis_name),
    )

  def should_update_estimator_curvature(
      self, state: OptaxPreconditionState
  ) -> Array | bool:
    """Whether at the current step the preconditioner should update the curvature estimates."""

    if self._curvature_update_period == 1:
      return True

    return state.count % self._curvature_update_period == 0

  def should_sync_estimate_curvature(
      self, state: OptaxPreconditionState
  ) -> Array | bool:
    """Whether at the current step the preconditioner should synchronize (pmean) the curvature estimates."""

    # sync only before inverses are calculated (either for updating the
    # cache or for preconditioning).
    if not self._use_cached_inverses:
      return True

    return self.should_update_inverse_cache(state)

  def should_update_inverse_cache(
      self, state: OptaxPreconditionState
  ) -> Array | bool:
    """Whether at the current step the preconditioner should update the inverse cache."""

    if not self._use_cached_inverses:
      return False

    return state.count % self._inverse_update_period == 0

  def maybe_update(
      self,
      state: OptaxPreconditionState,
      func_args: FuncArgs,
      rng: PRNGKey,
  ) -> OptaxPreconditionState:
    """Updates the estimates if it is the right iteration."""

    # NOTE: This maybe update curvatures and inverses at an iteration. But
    # if curvatures should be accumulated for multiple iterations
    # before updating inverses (for micro-batching), call
    # `maybe_update_estimator_curvature` and `maybe_update_inverse_cache`
    # separately, instead of calling this method.
    state = self.maybe_update_estimator_curvature(
        state=state,
        func_args=func_args,
        rng=rng,
        sync=self.should_sync_estimate_curvature(state),
    )

    state = self.maybe_update_inverse_cache(state)

    return OptaxPreconditionState(state.count, state.estimator_state)

  def _update_estimator_curvature(
      self,
      estimator_state: EstimatorState,
      func_args: FuncArgs,
      rng: PRNGKey,
      ema_old: Numeric,
      ema_new: Numeric,
      sync: Array | bool = True
  ) -> EstimatorState:
    """Updates the curvature estimator state."""

    state = self.estimator.update_curvature_matrix_estimate(
        state=estimator_state,
        ema_old=ema_old,
        ema_new=ema_new,
        # Note that the batch is always the last entry of FuncArgsVariantsdef
        batch_size=self._batch_size_extractor(func_args[-1]),
        identity_weight=self.get_identity_weight(estimator_state),
        rng=rng,
        func_args=func_args,
    )

    return jax.lax.cond(
        sync,
        functools.partial(self.estimator.sync,
                          pmap_axis_name=self.pmap_axis_name),
        lambda state_: state_,
        state,
    )

  def maybe_update_estimator_curvature(
      self,
      state: OptaxPreconditionState,
      func_args: FuncArgs,
      rng: PRNGKey,
      decay_old_ema: Array | bool = True,
      sync: Array | bool = True,
  ) -> OptaxPreconditionState:
    """Updates the curvature estimates if it is the right iteration."""

    ema_old = decay_old_ema * self._curvature_ema + (1.0 - decay_old_ema) * 1.0

    return self._maybe_update_estimator_state(
        state,
        self.should_update_estimator_curvature(state),
        self._update_estimator_curvature,
        func_args=func_args,
        rng=rng,
        ema_old=ema_old,
        ema_new=1.0,
        sync=sync,
    )

  def maybe_update_inverse_cache(
      self,
      state: OptaxPreconditionState,
  ) -> OptaxPreconditionState:
    """Updates the estimator state cache if it is the right iteration."""

    if state.count is None:
      raise ValueError(
          "PreconditionState is not initialized. Call"
          " `maybe_update_estimator_curvature` first."
      )

    return self._maybe_update_estimator_state(
        state,
        self.should_update_inverse_cache(state),
        self.estimator.update_cache,
        identity_weight=self.get_identity_weight(state),
        exact_powers=self._exact_powers_to_cache,
        approx_powers=self._approx_powers_to_cache,
        eigenvalues=False,
        pmap_axis_name=self.pmap_axis_name,
        norm_to_scale_identity_weight_per_block=self._norm_to_scale_identity_weight_per_block,
    )

  def _maybe_update_estimator_state(
      self,
      state: OptaxPreconditionState,
      should_update: Array | bool,
      update_func: Callable[..., EstimatorState],
      **update_func_kwargs,
  ) -> OptaxPreconditionState:
    """Updates the estimator state if it should update."""

    estimator_state = lax.cond(
        should_update,
        functools.partial(update_func, **update_func_kwargs),
        lambda s: s,
        state.estimator_state,
    )

    return OptaxPreconditionState(state.count, estimator_state)

  def apply(
      self,
      updates: optax.Updates,
      state: OptaxPreconditionState,
  ) -> optax.Updates:
    """Preconditions (= multiplies the inverse curvature estimation matrix to) updates."""

    new_updates = self.estimator.multiply_matpower(
        state=state.estimator_state,
        parameter_structured_vector=updates,
        identity_weight=self.get_identity_weight(state),
        power=-1 if not self._use_sqrt_inv else -0.5,
        exact_power=self._use_exact_inverses,
        use_cached=self._use_cached_inverses,
        pmap_axis_name=self.pmap_axis_name,
        norm_to_scale_identity_weight_per_block=self._norm_to_scale_identity_weight_per_block,
    )

    if self._norm_constraint is not None:

      sq_norm_grads = utils.inner_product(new_updates, updates)
      del updates

      max_coefficient = jnp.sqrt(self._norm_constraint / sq_norm_grads)
      coeff = jnp.minimum(max_coefficient, 1)

      new_updates = utils.scalar_mul(new_updates, coeff)

    else:
      del updates

    return new_updates

  def multiply_curvature(
      self,
      updates: optax.Updates,
      state: OptaxPreconditionState,
  ) -> optax.Updates:
    """Multiplies the (non-inverse) curvature estimation matrix to updates."""

    # NOTE: Currently, `exact_power` and `use_cached` arguments are not used
    # in `self.estimator.multiply()`, and the exact power (of 1) is always used.
    # Therefore, the way `identity_weight` (damping) is used with
    # `estimator.multiply()` is different from how it's used in
    # `estimator.multiply_inverse()` (in `Preconditioner.apply()`) when
    # `use_exact_inverses == False` (default). In particular, the former uses
    # non-factored damping while the latter uses factored one, and the two are
    # NOT the exact inverses of each other.
    return self.estimator.multiply(
        state=state.estimator_state,
        parameter_structured_vector=updates,
        identity_weight=self.get_identity_weight(state),
        exact_power=self._use_exact_inverses,
        use_cached=self._use_cached_inverses,
        pmap_axis_name=self.pmap_axis_name,
        norm_to_scale_identity_weight_per_block=self._norm_to_scale_identity_weight_per_block,
    )

  def as_gradient_transform(
      self, use_inverse: bool = True
  ) -> optax.GradientTransformationExtraArgs:
    """Multiplies the inverse or non-inverse curvature estimation matrix to updates."""

    def init_fn(params):
      del params
      return optax.EmptyState()

    multiply_fn = self.apply if use_inverse else self.multiply_curvature

    def update_fn(
        updates,
        state,
        params=None,
        *,
        precond_state: OptaxPreconditionState,
        **extra_args,
    ):
      del params, extra_args
      return multiply_fn(updates, precond_state), state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)

  def increment_count(self, state: OptaxPreconditionState):
    count_inc = optax.safe_int32_increment(state.count)
    return OptaxPreconditionState(count_inc, state.estimator_state)
