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
from typing import Any, Callable, Dict, Iterator, Mapping, NamedTuple, Optional, Sequence, Tuple, Type, Union

from absl import logging
import jax
from jax import lax
import jax.numpy as jnp
import kfac_jax
from ml_collections import config_dict
import optax


Array = kfac_jax.utils.Array
Numeric = kfac_jax.utils.Numeric
PRNGKey = kfac_jax.utils.PRNGKey
Params = kfac_jax.utils.Params
Batch = kfac_jax.utils.Batch
FuncState = kfac_jax.utils.FuncState
OptaxState = kfac_jax.utils.ArrayTree
ValueFunc = kfac_jax.optimizer.ValueFunc
FuncArgsVariants = kfac_jax.optimizer.FuncArgsVariants
ScheduleType = kfac_jax.optimizer.ScheduleType
OptaxCtor = Callable[[ScheduleType], optax.GradientTransformation]
EstimatorState = kfac_jax.curvature_estimator.BlockDiagonalCurvature.State
EmptyState = optax.EmptyState


class PreconditionState(NamedTuple):
  count: Array
  estimator_state: EstimatorState


class Preconditioner:
  """An Optax-compatible K-FAC preconditioner."""

  def __init__(
      self,
      value_func: ValueFunc,
      l2_reg: Numeric = 0.0,
      damping: Optional[float] = None,
      damping_schedule: Optional[ScheduleType] = None,
      norm_constraint: Optional[Numeric] = None,
      estimation_mode: str = "fisher_gradients",
      curvature_ema: Numeric = 0.95,
      curvature_update_period: int = 1,
      inverse_update_period: int = 5,
      use_exact_inverses: bool = False,
      use_sqrt_inv: bool = False,
      register_only_generic: bool = False,
      patterns_to_skip: Sequence[str] = (),
      auto_register_kwargs: Optional[Dict[str, Any]] = None,
      layer_tag_to_block_ctor: Optional[
          Dict[str, kfac_jax.curvature_estimator.CurvatureBlockCtor]
      ] = None,
      pmap_axis_name: str = "kfac_axis",
      batch_size_extractor: Callable[
          [Batch], Numeric
      ] = kfac_jax.utils.default_batch_size_extractor,
      distributed_inverses: bool = True,
      distributed_precon_apply: bool = True,
      num_samples: int = 1,
      should_vmap_samples: bool = False,
      norm_to_scale_identity_weight_per_block: Optional[str] = None,
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

    # Curvature estimator
    self._estimator = kfac_jax.curvature_estimator.BlockDiagonalCurvature(
        func=value_func,
        default_estimation_mode=estimation_mode,
        params_index=0,
        layer_tag_to_block_ctor=layer_tag_to_block_ctor,
        register_only_generic=register_only_generic,
        patterns_to_skip=patterns_to_skip,
        distributed_multiplies=distributed_precon_apply,
        distributed_cache_updates=distributed_inverses,
        num_samples=num_samples,
        should_vmap_samples=should_vmap_samples,
        **(auto_register_kwargs or {}),
    )

  def init(
      self,
      func_args: FuncArgsVariants,
      rng: PRNGKey,
  ) -> PreconditionState:
    """Initializes the preconditioner and returns the state."""

    return PreconditionState(
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
  def _exact_powers_to_cache(self) -> Optional[Union[int, Sequence[int]]]:

    if self._use_exact_inverses and self._use_cached_inverses:
      return -1
    else:
      return None

  @property
  def _approx_powers_to_cache(self) -> Optional[Union[int, Sequence[int]]]:

    if not self._use_exact_inverses and self._use_cached_inverses:
      return -1
    else:
      return None

  @property
  def estimator(self) -> kfac_jax.curvature_estimator.BlockDiagonalCurvature:
    """The underlying curvature estimator used by the preconditioner."""
    return self._estimator

  @property
  def pmap_axis_name(self):
    return self._pmap_axis_name

  def get_identity_weight(
      self, state: PreconditionState
  ) -> Union[Array, float]:

    damping = self._damping

    if damping is None:
      damping = self._damping_schedule(state.count)

    return damping + self._l2_reg

  def sync_estimator_state(
      self,
      state: PreconditionState,
  ) -> PreconditionState:
    """Syncs the estimator state."""

    return PreconditionState(
        count=state.count,
        estimator_state=self.estimator.sync(
            state.estimator_state, pmap_axis_name=self.pmap_axis_name),
    )

  def should_update_estimator_curvature(
      self, state: PreconditionState
  ) -> Union[Array, bool]:
    """Whether at the current step the preconditioner should update the curvature estimates."""

    if self._curvature_update_period == 1:
      return True

    return state.count % self._curvature_update_period == 0

  def should_sync_estimate_curvature(
      self, state: PreconditionState
  ) -> Union[Array, bool]:
    """Whether at the current step the preconditioner should synchronize (pmean) the curvature estimates."""

    # sync only before inverses are calculated (either for updating the
    # cache or for preconditioning).
    if not self._use_cached_inverses:
      return True

    return self.should_update_inverse_cache(state)

  def should_update_inverse_cache(
      self, state: PreconditionState
  ) -> Union[Array, bool]:
    """Whether at the current step the preconditioner should update the inverse cache."""

    if not self._use_cached_inverses:
      return False

    return state.count % self._inverse_update_period == 0

  def maybe_update(
      self,
      state: PreconditionState,
      func_args: FuncArgsVariants,
      rng: PRNGKey,
  ) -> PreconditionState:
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

    return PreconditionState(state.count, state.estimator_state)

  def _update_estimator_curvature(
      self,
      estimator_state: EstimatorState,
      func_args: FuncArgsVariants,
      rng: PRNGKey,
      ema_old: Numeric,
      ema_new: Numeric,
      sync: Union[Array, bool] = True
  ) -> EstimatorState:
    """Updates the curvature estimator state."""

    state = self.estimator.update_curvature_matrix_estimate(
        state=estimator_state,
        ema_old=ema_old,
        ema_new=ema_new,
        # Note that the batch is always the last entry of FuncArgsVariantsdef
        batch_size=self._batch_size_extractor(func_args[-1]),
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
      state: PreconditionState,
      func_args: FuncArgsVariants,
      rng: PRNGKey,
      decay_old_ema: Union[Array, bool] = True,
      sync: Union[Array, bool] = True,
  ) -> PreconditionState:
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
      state: PreconditionState,
  ) -> PreconditionState:
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
      state: PreconditionState,
      should_update: Union[Array, bool],
      update_func: Callable[..., EstimatorState],
      **update_func_kwargs,
  ) -> PreconditionState:
    """Updates the estimator state if it should update."""

    estimator_state = lax.cond(
        should_update,
        functools.partial(update_func, **update_func_kwargs),
        lambda s: s,
        state.estimator_state,
    )

    return PreconditionState(state.count, estimator_state)

  def apply(
      self,
      updates: optax.Updates,
      state: PreconditionState,
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

      sq_norm_grads = kfac_jax.utils.inner_product(new_updates, updates)
      del updates

      max_coefficient = jnp.sqrt(self._norm_constraint / sq_norm_grads)
      coeff = jnp.minimum(max_coefficient, 1)

      new_updates = kfac_jax.utils.scalar_mul(new_updates, coeff)

    else:
      del updates

    return new_updates

  def multiply_curvature(
      self,
      updates: optax.Updates,
      state: PreconditionState,
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
    updates = self.estimator.multiply(
        state=state.estimator_state,
        parameter_structured_vector=updates,
        identity_weight=self.get_identity_weight(state),
        exact_power=self._use_exact_inverses,
        use_cached=self._use_cached_inverses,
        pmap_axis_name=self.pmap_axis_name,
        norm_to_scale_identity_weight_per_block=self._norm_to_scale_identity_weight_per_block,
    )
    return updates

  def as_gradient_transform(
      self, use_inverse: bool = True
  ) -> optax.GradientTransformationExtraArgs:
    """Multiplies the inverse or non-inverse curvature estimation matrix to updates."""

    def init_fn(params):
      del params
      return EmptyState()

    multiply_fn = self.apply if use_inverse else self.multiply_curvature

    def update_fn(
        updates,
        state,
        params=None,
        *,
        precond_state: PreconditionState,
        **extra_args,
    ):
      del params, extra_args
      updates = multiply_fn(updates, precond_state)
      return updates, state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)

  def increment_count(self, state: PreconditionState):
    count_inc = optax.safe_int32_increment(state.count)
    return PreconditionState(count_inc, state.estimator_state)


class OptaxAndPreconditionState(NamedTuple):
  optax_state: OptaxState
  precond_state: Optional[PreconditionState] = None


class OptaxWrapper:
  """Wrapper class for Optax optimizers to have the same interface as KFAC."""

  def __init__(
      self,
      value_and_grad_func: kfac_jax.optimizer.ValueAndGradFunc,
      value_func_has_aux: bool,
      value_func_has_state: bool,
      value_func_has_rng: bool,
      learning_rate: ScheduleType,
      optax_optimizer_ctor: OptaxCtor,
      batch_process_func: Callable[[Batch], Batch] = lambda x: x,
      preconditioner: Optional[Preconditioner] = None,
      include_norms_in_stats: bool = False,
      include_per_param_norms_in_stats: bool = False,
  ):
    """Initializes the Optax wrapper.

    Args:
      value_and_grad_func: Python callable. The function should return the value
        of the loss to be optimized and its gradients. If the argument
        `value_func_has_aux` is `False` then the interface should be:
          loss, loss_grads = value_and_grad_func(params, batch)
        If `value_func_has_aux` is `True` then the interface should be:
          (loss, aux), loss_grads = value_and_grad_func(params, batch)
      value_func_has_aux: Boolean. Specifies whether the provided callable
        `value_and_grad_func` returns the loss value only, or also some
        auxiliary data. (Default: `False`)
      value_func_has_state: Boolean. Specifies whether the provided callable
        `value_and_grad_func` has a persistent state that is inputted and it
        also outputs an update version of it. (Default: `False`)
      value_func_has_rng: Boolean. Specifies whether the provided callable
        `value_and_grad_func` additionally takes as input an rng key. (Default:
        `False`)
      learning_rate: The learning rate or learning rate schedule.
      optax_optimizer_ctor: A callable that takes the learning rate schedule as
        an input and returns the optax optimizer.
      batch_process_func: Callable. A function which to be called on each batch
        before feeding to the KFAC on device. This could be useful for specific
        device input optimizations. (Default: `lambda x: x`)
      preconditioner: The optax-compatible K-FAC preconditioner.
      include_norms_in_stats: Boolean. It True, the vector norms of the
        gradient, preconditioned gradient, and parameter update are included in
        the statistics returned by the step function. (Default: ``False``)
      include_per_param_norms_in_stats: Boolean. It True, the per-parameter
        vector norms of the gradient, preconditioned gradient, and parameter
        update are included in the statistics returned by the step function.
        (Default: ``False``)
    """
    self._value_and_grad_func = value_and_grad_func
    self._value_func_has_aux = value_func_has_aux
    self._value_func_has_state = value_func_has_state
    self._value_func_has_rng = value_func_has_rng

    if not callable(learning_rate):
      self._learning_rate = lambda _: learning_rate
    else:
      self._learning_rate = learning_rate

    # Wraps the optax optimizer (gradient transformation), so that it ignores
    # extra args (i.e. `precond_state` for preconditioner) if not needed.
    self._optax_optimizer = optax.with_extra_args_support(
        optax_optimizer_ctor(self._learning_rate)
    )

    self._preconditioner = preconditioner
    self._include_norms_in_stats = include_norms_in_stats
    self._include_per_param_norms_in_stats = include_per_param_norms_in_stats
    self._batch_process_func = batch_process_func or (lambda x: x)
    self.pmap_axis_name = (
        "optax_axis"
        if self._preconditioner is None
        else self._preconditioner.pmap_axis_name
    )
    self._pmap_step = jax.pmap(
        self._step,
        axis_name=self.pmap_axis_name,
        donate_argnums=list(range(5)),
        in_axes=(0,) * 5 + (None,),
    )
    self._pmap_init = jax.pmap(
        lambda p, *_: OptaxAndPreconditionState(self._optax_optimizer.init(p)),
        axis_name=self.pmap_axis_name,
    )
    self._pmap_rng_split = jax.pmap(
        lambda rng, num: tuple(jax.random.split(rng, num)),
        axis_name=self.pmap_axis_name,
        static_broadcasted_argnums=1
    )

    if self._preconditioner is not None:

      if not isinstance(self._preconditioner, Preconditioner):
        raise ValueError(
            "preconditioner must be a {}, but {} is given.".format(
                Preconditioner, type(self._preconditioner)
            )
        )

      preconditioner: Preconditioner = self._preconditioner

      def _init_preconditioner(
          params: Params,
          rng: PRNGKey,
          batch: Batch,
          func_state: Optional[FuncState] = None,
      ) -> PreconditionState:
        """Maybe initializes the PreconditionState."""

        batch = self._batch_process_func(batch)

        func_args = kfac_jax.optimizer.make_func_args(
            params,
            func_state,
            rng,
            batch,
            has_state=self._value_func_has_state,
            has_rng=self._value_func_has_rng,
        )

        return preconditioner.init(func_args, rng)

      self._pmap_init_preconditioner = jax.pmap(
          _init_preconditioner,
          axis_name=self.pmap_axis_name,
      )

  def init(
      self,
      params: Params,
      rng: PRNGKey,
      batch: Batch,
      func_state: Optional[FuncState] = None,
  ) -> OptaxAndPreconditionState:
    """Initializes the optimizer and returns the appropriate optimizer state."""
    return self._pmap_init(params, rng, batch, func_state)

  def _step(
      self,
      params: Params,
      state: OptaxAndPreconditionState,
      rng: PRNGKey,
      batch: Batch,
      func_state: Optional[FuncState] = None,
      global_step_int: Optional[int] = None,
  ) -> Union[
      Tuple[Params, OptaxAndPreconditionState, FuncState, Mapping[str, Array]],
      Tuple[Params, OptaxAndPreconditionState, Mapping[str, Array]],
  ]:
    """A single step of optax."""

    rng_func, rng_precon = jax.random.split(rng)
    batch = self._batch_process_func(batch)

    func_args = kfac_jax.optimizer.make_func_args(
        params, func_state, rng_func, batch,
        has_state=self._value_func_has_state,
        has_rng=self._value_func_has_rng
    )

    optax_state, precond_state = state.optax_state, state.precond_state

    if self._preconditioner is not None:
      precond_state = self._preconditioner.maybe_update(
          precond_state,
          func_args,
          rng_precon,
      )
      precond_state = self._preconditioner.increment_count(precond_state)

    out, grads = self._value_and_grad_func(*func_args)

    loss, new_func_state, stats = kfac_jax.optimizer.extract_func_outputs(
        out,
        has_aux=self._value_func_has_aux,
        has_state=self._value_func_has_state,
    )

    loss, stats, grads = kfac_jax.utils.pmean_if_pmap(  # pytype: disable=wrong-keyword-args
        (loss, stats, grads), axis_name=self.pmap_axis_name
    )

    stats = stats or {}
    stats["loss"] = loss

    # Compute and apply updates via our optimizer.
    updates, new_optax_state = self._optax_optimizer.update(
        grads,
        optax_state,
        params,
        precond_state=precond_state,
    )
    new_state = OptaxAndPreconditionState(new_optax_state, precond_state)
    new_params = optax.apply_updates(params, updates)

    # Add step and batch size
    batch_size = jax.tree_util.tree_leaves(batch)[0].shape[0]
    stats["step"] = global_step_int + 1
    stats["batch_size"] = batch_size * jax.device_count()
    stats["data_seen"] = stats["step"] * stats["batch_size"]
    stats["learning_rate"] = self._learning_rate(global_step_int)

    if self._include_norms_in_stats:
      stats["grad_norm"] = kfac_jax.utils.norm(grads)
      stats["update_norm"] = kfac_jax.utils.norm(updates)
      stats["param_norm"] = kfac_jax.utils.norm(params)
      stats["rel_grad_norm"] = stats["grad_norm"] / stats["param_norm"]
      stats["rel_update_norm"] = stats["update_norm"] / stats["param_norm"]

    if self._include_per_param_norms_in_stats:
      stats.update(kfac_jax.utils.per_parameter_norm(grads, "grad_norm"))
      stats.update(kfac_jax.utils.per_parameter_norm(updates, "update_norm"))
      param_norms = kfac_jax.utils.per_parameter_norm(params, "param_norm")

      for key in param_norms:

        norm = param_norms[key]
        stats[key] = norm

        grad_key = key.replace("param", "grad")
        stats["rel_" + grad_key] = stats[grad_key] / norm

        upd_key = key.replace("param", "update")
        stats["rel_" + upd_key] = stats[upd_key] / norm

    if self._value_func_has_state:
      return new_params, new_state, new_func_state, stats
    else:
      return new_params, new_state, stats

  def step(
      self,
      params: Params,
      state: OptaxAndPreconditionState,
      rng: PRNGKey,
      data_iterator: Iterator[Batch],
      func_state: Optional[FuncState] = None,
      global_step_int: Optional[int] = None,
  ) -> Union[
      Tuple[Params, Any, FuncState, Mapping[str, Array]],
      Tuple[Params, Any, Mapping[str, Array]],
  ]:
    """A step with similar interface to KFAC."""

    rng_init, rng_step = self._pmap_rng_split(rng, 2)

    batch = next(data_iterator)

    if self._preconditioner is not None and state.precond_state is None:

      precond_state = self._pmap_init_preconditioner(
          params, rng_init, batch, func_state
      )
      state = OptaxAndPreconditionState(state.optax_state, precond_state)

    return self._pmap_step(
        params,
        state,
        rng_step,
        batch,
        func_state,
        global_step_int,
    )


def tf1_rmsprop(
    learning_rate_fn: Callable[[Numeric], Numeric],
    decay: float = .9,
    momentum: float = 0.,
    epsilon: float = 1e-8
) -> optax.GradientTransformation:
  """RMSProp update equivalent to tf.compat.v1.train.RMSPropOptimizer."""

  def tf1_scale_by_rms(decay_=0.9, epsilon_=1e-8):
    """Same as optax.scale_by_rms, but initializes second moment to one."""

    def init_fn(params):
      nu = jax.tree_util.tree_map(jnp.ones_like, params)  # second moment
      return optax.ScaleByRmsState(nu=nu)

    def _update_moment(updates, moments, decay, order):

      return jax.tree_util.tree_map(
          lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments)

    def update_fn(updates, state, params=None):

      del params

      nu = _update_moment(updates, state.nu, decay_, 2)

      updates = jax.tree_util.tree_map(
          lambda g, n: g / (jnp.sqrt(n + epsilon_)), updates, nu)

      return updates, optax.ScaleByRmsState(nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)

  return optax.chain(
      tf1_scale_by_rms(decay_=decay, epsilon_=epsilon),
      optax.trace(decay=momentum, nesterov=False),
      optax.scale_by_schedule(learning_rate_fn),
      optax.scale(-1.))


def linear_interpolation(
    x: Numeric,
    interpolation_points: Tuple[Tuple[float, float], ...]
) -> Array:
  """Performs linear interpolation between the interpolation points."""

  xs, ys = zip(*interpolation_points)
  masks = [x < ci for ci in xs[1:]]

  min_iter = jnp.zeros_like(x)
  max_iter = jnp.zeros_like(x)
  max_val = jnp.zeros_like(x)
  min_val = jnp.zeros_like(x)
  p = jnp.ones_like(x)

  for i in range(len(masks) - 1):
    pi = p * masks[i]

    min_iter = pi * xs[i] + (1 - pi) * min_iter
    max_iter = pi * xs[i + 1] + (1 - pi) * max_iter
    max_val = pi * ys[i] + (1 - pi) * max_val
    min_val = pi * ys[i + 1] + (1 - pi) * min_val

    p = p * (1 - masks[i])

  min_iter = p * xs[-2] + (1 - p) * min_iter
  max_iter = p * xs[-1] + (1 - p) * max_iter
  max_val = p * ys[-2] + (1 - p) * max_val
  min_val = p * ys[-1] + (1 - p) * min_val

  diff = (min_val - max_val)
  progress = (x - min_iter) / (max_iter - min_iter - 1)

  return max_val + diff * jnp.minimum(progress, 1.0)


def imagenet_sgd_schedule(
    global_step: Numeric,
    dataset_size: int,
    train_total_batch_size: Optional[int],
    **_: Any,
) -> Array:
  """Standard linear scaling schedule for ImageNet."""

  if train_total_batch_size is None:
    raise ValueError("Batch size must be known.")

  # Can be found in Section 5.1 of https://arxiv.org/pdf/1706.02677.pdf
  steps_per_epoch = dataset_size / train_total_batch_size
  current_epoch = global_step / steps_per_epoch

  lr = (0.1 * train_total_batch_size) / 256
  lr_linear_till = 5

  boundaries = jnp.array((30, 60, 80)) * steps_per_epoch
  values = jnp.array([1., 0.1, 0.01, 0.001]) * lr

  index = jnp.sum(boundaries < global_step)
  lr = jnp.take(values, index)

  return lr * jnp.minimum(1., current_epoch / lr_linear_till)


def fixed_schedule(
    global_step: Numeric,
    value: Numeric,
    **_: Any,
) -> Array:
  """Fixed/constant schedule."""
  return jnp.ones_like(global_step) * value


def kfac_resnet50_schedule(
    global_step: Numeric,
    **_: Any,
) -> Array:
  """Custom schedule for KFAC."""

  return jnp.power(10.0, linear_interpolation(
      x=global_step,
      interpolation_points=(
          (0, -6), (50, -3.1), (5000, -3.1), (11000, -3.23),
          (20000, -5.0), (200000, -5.7), (1000001, -6))
  ))


# TODO(jamesmartens,kazukiosawa,botev): Some possible future improvements to
# the schedules code:
# - Put the logic to calculate "warmup_data" (or "warmup_steps") and
#   "total_data" (or "total_steps") in a place so that we can apply warmup to
#   an arbitrary schedule.
# - Use existing `optax.schedule` operations (e.g. `exponential_decay`,
#   `piecewise_constant_schedule`) as much as possible to make the kfac_jax
#   codebase simple and compact.
# - Optax's `warmup_cosine_decay_schedule` and
#   `warmup_exponential_decay_schedule` are implemented by simply combining
#   `linear_schedule` and the corresponding schedule. So we can prepare a
#   general warmup scheduler factory that returns a combination of `linear_
#   schedule` and the given base scheduler based on the arguments e.g. warmup_
#   steps.


# TODO(jamesmartens,kazukiosawa,botev): change these argument names to be not be
# specific to learning rates.
def cosine_schedule(
    global_step: Numeric,
    dataset_size: int,
    train_total_batch_size: Optional[int],
    total_steps: Optional[int],
    total_epochs: Optional[float],
    peak_learning_rate: float,
    initial_learning_rate: float = 1e-7,
    end_learning_rate: float = 0.0,
    warmup_epochs: Optional[float] = None,
    warmup_steps: Optional[int] = None,
    warmup_fraction: Optional[float] = None,
    data_seen: Optional[Numeric] = None,
    **_: Any,
) -> Numeric:
  """A cosine schedule described in the TAT paper."""

  if (total_steps is None) == (total_epochs is None):
    raise ValueError("Exactly one of `total_steps` and `total_epochs` must be "
                     "set.")

  n = sum(x is not None for x in [warmup_epochs, warmup_steps, warmup_fraction])

  if n != 1:
    raise ValueError(f"Exactly one of warmup_steps={warmup_steps}, "
                     f"warmup_epochs={warmup_epochs} and warmup_fraction="
                     f"{warmup_fraction} must be set.")

  if warmup_epochs is not None or total_epochs is not None:

    if data_seen is None:

      if train_total_batch_size is not None:
        data_seen = global_step * train_total_batch_size

      else:
        raise ValueError("One of 'train_total_batch_size' or 'data_seen' must "
                         "passed when 'total_epochs' or 'warmup_epochs' are "
                         "passed.")

    if ((warmup_epochs is None or total_epochs is None)
        and train_total_batch_size is None):

      raise ValueError("'train_total_batch_size' must be passed if only one of "
                       "'total_epochs' or 'warmup_epochs' are passed.")

    if warmup_epochs is not None:
      warmup_data = warmup_epochs * dataset_size

    elif warmup_fraction is not None:
      warmup_data = warmup_fraction * total_steps * train_total_batch_size

    else:
      warmup_data = warmup_steps * train_total_batch_size

    if total_epochs is not None:
      total_data = total_epochs * dataset_size

    else:
      total_data = total_steps * train_total_batch_size

    # Optax uses chex which has an inconsistent definition of "Numeric" from
    # what we use here.
    return optax.warmup_cosine_decay_schedule(  # pytype: disable=bad-return-type
        init_value=initial_learning_rate,
        peak_value=peak_learning_rate,
        end_value=end_learning_rate,
        warmup_steps=warmup_data,
        decay_steps=total_data,
    )(data_seen)

  else:

    if warmup_fraction is not None:
      warmup_steps = warmup_fraction * total_steps

    # Optax uses chex which has an inconsistent definition of "Numeric" from
    # what we use here.
    return optax.warmup_cosine_decay_schedule(  # pytype: disable=bad-return-type
        init_value=initial_learning_rate,
        peak_value=peak_learning_rate,
        end_value=end_learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
    )(global_step)


# TODO(jamesmartens,kazukiosawa,botev): change these argument names to be not be
# specific to learning rates. Also, initial_learning_rate is misnamed since this
# is value is never actually used, but is just a "base" multiplying for the
# decay factors.
def stepwise_schedule(
    global_step: Numeric,
    dataset_size: int,
    train_total_batch_size: Optional[int],
    lr_decay_factors: Sequence[float],
    initial_learning_rate: float,
    epoch_boundaries: Optional[Sequence[float]] = None,
    warmup_epochs: Optional[float] = None,
    step_boundaries: Optional[Sequence[float]] = None,
    warmup_steps: Optional[int] = None,
    data_seen: Optional[Numeric] = None,
    **_: Any,
) -> Numeric:
  """A basic stepwise schedule."""

  if (epoch_boundaries is None) == (step_boundaries is None):
    raise ValueError("Exactly one of `epoch_boundaries` and `step_boundaries` "
                     "can must be passed.")

  if (warmup_epochs is None) == (warmup_steps is None):
    raise ValueError("Exactly one of `warmup_epochs` and `warmup_steps` must "
                     "be set.")

  values = jnp.array(lr_decay_factors) * initial_learning_rate

  if warmup_epochs is not None or epoch_boundaries is not None:

    if data_seen is None:

      if train_total_batch_size is not None:
        data_seen = global_step * train_total_batch_size

      else:
        raise ValueError("One of 'train_total_batch_size' or 'data_seen' must "
                         "passed when 'epoch_boundaries' or 'warmup_epochs' "
                         "are passed.")

    if ((warmup_epochs is None or epoch_boundaries is None)
        and train_total_batch_size is None):

      raise ValueError("'train_total_batch_size' must be passed if only one of "
                       "'epoch_boundaries' or 'warmup_epochs' are passed.")

    if warmup_epochs is not None:
      warmup_data = warmup_epochs * dataset_size

    else:
      warmup_data = warmup_steps * train_total_batch_size

    if epoch_boundaries is not None:
      data_boundaries = jnp.array(epoch_boundaries) * dataset_size

    else:
      data_boundaries = jnp.array(step_boundaries) * train_total_batch_size

    index = jnp.sum(data_boundaries <= data_seen)
    value = jnp.take(values, index)

    return value * jnp.minimum(1., data_seen / warmup_data)

  else:

    step_boundaries = jnp.array(step_boundaries)

    index = jnp.sum(step_boundaries <= global_step)
    value = jnp.take(values, index)

    return value * jnp.minimum(1., global_step / warmup_steps)


def exponential_decay_schedule(
    global_step: int,
    dataset_size: int,
    train_total_batch_size: Optional[int],
    total_steps: Optional[int],
    total_epochs: Optional[float],
    init_value: float,
    end_value: float,
    decay_epochs: Optional[float] = None,
    decay_steps: Optional[int] = None,
    decay_fraction: Optional[float] = None,
    **_: Any,
):
  """Exponential decay schedule."""
  if (total_steps is None) == (total_epochs is None):
    raise ValueError("Only one of `steps` and `epochs` can be set.")

  n = sum(x is not None for x in [decay_epochs, decay_steps, decay_fraction])

  if n != 1:
    raise ValueError(
        f"Exactly one of warmup_steps={decay_steps}, "
        f"warmup_epochs={decay_epochs} and warmpu_fraction="
        f"{decay_fraction} must be set."
    )

  if (
      decay_epochs is not None or total_epochs is not None
  ) and train_total_batch_size is None:
    raise ValueError(
        "Batch size must be known when passing epochs or warmup_epochs."
    )

  if decay_epochs is not None:
    decay_steps = decay_epochs * dataset_size / train_total_batch_size
  elif decay_fraction is not None:
    decay_steps = decay_fraction * total_steps

  return optax.exponential_decay(
      init_value=init_value,
      end_value=end_value,
      decay_rate=end_value / init_value,
      transition_steps=decay_steps,
  )(global_step)


def construct_schedule(
    name: str,
    **kwargs,
) -> Callable[[Numeric], Array]:
  """Constructs the actual schedule from its name and extra kwargs."""

  if name == "fixed":
    return functools.partial(fixed_schedule, **kwargs)
  elif name == "imagenet_sgd":
    return functools.partial(imagenet_sgd_schedule, **kwargs)
  elif name == "kfac_resnet50":
    return functools.partial(kfac_resnet50_schedule, **kwargs)
  elif name == "cosine":
    return functools.partial(cosine_schedule, **kwargs)
  elif name == "stepwise":
    return functools.partial(stepwise_schedule, **kwargs)
  elif name == "exponential_decay":
    return functools.partial(exponential_decay_schedule, **kwargs)
  else:
    raise NotImplementedError(name)


def kfac_bn_registration_kwargs(bn_registration: str) -> Mapping[
    str, Union[Tuple[str, ...], Mapping[str, Type[kfac_jax.CurvatureBlock]]]
]:
  """Constructs KFAC kwargs for the given batch-norm registration strategy."""

  if bn_registration == "generic":
    return dict(patterns_to_skip=("scale_and_shift", "scale_only"))

  elif bn_registration == "full":

    return dict(
        layer_tag_to_block_cls=dict(
            scale_and_shift_tag=kfac_jax.ScaleAndShiftFull,
        )
    )

  elif bn_registration != "diag":
    raise ValueError(f"Unknown batch_norm_registration={bn_registration}.")

  return {}


def create_optimizer(
    name: str,
    config: config_dict.ConfigDict,
    train_model_func: kfac_jax.optimizer.ValueFunc,
    l2_reg: Numeric,
    has_aux: bool,
    has_func_state: bool,
    has_rng: bool,
    dataset_size: int,
    train_total_batch_size: int,
    total_steps: Optional[int],
    total_epochs: Optional[float],
) -> Union[OptaxWrapper, kfac_jax.Optimizer]:
  """Creates an optimizer from the provided configuration."""

  value_and_grad_func = jax.value_and_grad(train_model_func, has_aux=has_aux)

  kwargs = dict(**config[name])

  logging.info("Using %s optimizer.", name)

  if "kfac" in name:

    # Update kwargs regarding batch norm registration
    extra_kwargs = kfac_bn_registration_kwargs(
        kwargs.pop("batch_norm_registration", "diag"))
    kwargs.update(extra_kwargs)

    if name == "kfac":

      for sched_name in ["learning_rate_schedule", "momentum_schedule",
                         "damping_schedule"]:

        if kwargs.get(sched_name) is not None:

          kwargs[sched_name] = construct_schedule(
              dataset_size=dataset_size,
              train_total_batch_size=train_total_batch_size,
              total_steps=total_steps,
              total_epochs=total_epochs,
              **kwargs[sched_name]
              )

    return kfac_jax.Optimizer(
        value_and_grad_func=value_and_grad_func,
        l2_reg=l2_reg,
        value_func_has_aux=has_aux,
        value_func_has_state=has_func_state,
        value_func_has_rng=has_rng,
        multi_device=True,
        **kwargs,
    )

  elif hasattr(optax, name):

    learning_rate_schedule = construct_schedule(
        dataset_size=dataset_size,
        train_total_batch_size=train_total_batch_size,
        total_steps=total_steps,
        total_epochs=total_epochs,
        **kwargs.pop("learning_rate_schedule")
    )
    optax_ctor = lambda lr: (getattr(optax, name)(learning_rate=lr, **kwargs))

    return OptaxWrapper(
        value_and_grad_func=value_and_grad_func,
        value_func_has_aux=has_aux,
        value_func_has_rng=has_rng,
        value_func_has_state=has_func_state,
        learning_rate=learning_rate_schedule,
        optax_optimizer_ctor=optax_ctor,
    )

  else:
    raise NotImplementedError()
