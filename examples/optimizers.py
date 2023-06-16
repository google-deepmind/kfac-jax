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
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Type, Union

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
EstimatorState = kfac_jax.curvature_estimator.BlockDiagonalCurvature.State


class PreconditionState(NamedTuple):
  count: Optional[Array]
  estimator_state: Optional[EstimatorState]


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

  def maybe_init(
      self, state: PreconditionState, func_args: FuncArgsVariants, rng: PRNGKey
  ) -> PreconditionState:
    if state.count is not None:
      return state
    return self.init(func_args, rng)

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

  def should_update_estimate_curvature(
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
    # TODO(kazukiosawa): Once an independent method for sync curvatures is
    # introduced, remove this method and modify to call the sync method at the
    # right timing (i.e., right before calling `self.estimator.update_cache()`
    # or `self.estimator.multiply_inverse()`).

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
        self.should_update_estimate_curvature(state),
        self.estimator.update_curvature_matrix_estimate,
        ema_old=ema_old,
        ema_new=1.0,
        # Note that the batch is always the last entry of FuncArgsVariantsdef
        batch_size=self._batch_size_extractor(func_args[-1]),
        rng=rng,
        func_args=func_args,
        pmap_axis_name=self.pmap_axis_name,
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
      coeff: Optional[optax.ScalarOrSchedule] = None,
  ) -> optax.Updates:
    """Preconditions the updates."""
    new_updates = self.estimator.multiply_inverse(
        state=state.estimator_state,
        parameter_structured_vector=updates,
        identity_weight=self.get_identity_weight(state),
        exact_power=self._use_exact_inverses,
        use_cached=self._use_cached_inverses,
        pmap_axis_name=self.pmap_axis_name,
    )
    if self._norm_constraint is not None:
      if coeff is None:
        raise ValueError(
            "coeff must be passed when norm_constraint is specified."
        )
      if callable(coeff):
        coeff = coeff(state.count)
      sq_norm_grads = kfac_jax.utils.inner_product(new_updates, updates)
      del updates
      sq_norm_scaled_grads = sq_norm_grads * coeff**2
      max_coefficient = jnp.sqrt(self._norm_constraint / sq_norm_scaled_grads)
      coeff = jnp.minimum(max_coefficient, 1)
      new_updates = kfac_jax.utils.scalar_mul(new_updates, coeff)
    else:
      del updates
    return new_updates

  def as_gradient_transform(
      self, learning_rate: Optional[optax.ScalarOrSchedule] = None
  ) -> optax.GradientTransformation:
    """Preconditions updates by the K-FAC preconditioner."""

    def init_fn(params):
      del params
      # The state must be initialized by `maybe_init()` with the appropriate
      # func_args (FuncArgsVariants).
      return PreconditionState(None, None)

    def update_fn(updates, state: PreconditionState, params=None):
      del params
      updates = self.apply(updates, state, learning_rate)
      count_inc = optax.safe_int32_increment(state.count)
      return updates, PreconditionState(count_inc, state.estimator_state)

    return optax.GradientTransformation(init_fn, update_fn)


def extract_precondition_state(state: OptaxState) -> PreconditionState:
  """Extracts the PreconditionState from OptaxState."""
  if isinstance(state, Iterable):
    num_precond_state = sum(isinstance(s, PreconditionState) for s in state)
    if num_precond_state != 1:
      raise ValueError(
          "The state must have only one PreconditionState. It has {}.".format(
              num_precond_state
          )
      )
    return next(s for s in state if isinstance(s, PreconditionState))
  elif isinstance(state, PreconditionState):
    return state
  else:
    raise ValueError("state {} is not a PreconditionState.".format(state))


def replace_precondition_state(
    state: OptaxState, new_precond_state: PreconditionState
) -> Union[List[Union[OptaxState, PreconditionState]], PreconditionState]:
  """Replaces the PreconditionState in OptaxState with new one."""
  extract_precondition_state(state)  # run for validating the state
  if isinstance(state, Iterable):
    new_state = [
        new_precond_state if isinstance(s, PreconditionState) else s
        for s in state
    ]
    return new_state
  else:
    return new_precond_state


class OptaxWrapper:
  """Wrapper class for Optax optimizers to have the same interface as KFAC."""

  def __init__(
      self,
      value_and_grad_func: kfac_jax.optimizer.ValueAndGradFunc,
      value_func_has_aux: bool,
      value_func_has_state: bool,
      value_func_has_rng: bool,
      optax_optimizer: optax.GradientTransformation,
      batch_process_func: Optional[Callable[[Batch], Batch]] = lambda x: x,
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
      optax_optimizer: The optax optimizer to be wrapped.
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
    self._optax_optimizer = optax_optimizer
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
        lambda p, *_: self._optax_optimizer.init(p),
        axis_name=self.pmap_axis_name,
    )
    if self._preconditioner is not None:
      if not isinstance(self._preconditioner, Preconditioner):
        raise ValueError(
            "preconditioner must be a {}, but {} is given.".format(
                Preconditioner, type(self._preconditioner)
            )
        )
      preconditioner: Preconditioner = self._preconditioner
      def _maybe_init_preconditioner(
          params: Params,
          state: OptaxState,
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
        precond_state = preconditioner.maybe_init(
            extract_precondition_state(state), func_args, rng
        )
        return precond_state

      self._pmap_maybe_init_preconditioner = jax.pmap(
          _maybe_init_preconditioner,
          axis_name=self.pmap_axis_name,
      )

  def init(
      self,
      params: Params,
      rng: PRNGKey,
      batch: Batch,
      func_state: Optional[FuncState] = None,
  ) -> OptaxState:
    """Initializes the optimizer and returns the appropriate optimizer state."""
    return self._pmap_init(params, rng, batch, func_state)

  def _step(
      self,
      params: Params,
      state: OptaxState,
      rng: PRNGKey,
      batch: Batch,
      func_state: Optional[FuncState] = None,
      global_step_int: Optional[int] = None,
  ) -> kfac_jax.optimizer.ReturnEither:
    """A single step of optax."""
    batch = self._batch_process_func(batch)
    func_args = kfac_jax.optimizer.make_func_args(
        params, func_state, rng, batch,
        has_state=self._value_func_has_state,
        has_rng=self._value_func_has_rng
    )

    if self._preconditioner is not None:
      precond_state = self._preconditioner.maybe_update(
          extract_precondition_state(state),
          func_args,
          rng,
      )
      state = replace_precondition_state(state, precond_state)
    out, grads = self._value_and_grad_func(*func_args)
    loss, new_func_state, stats = kfac_jax.optimizer.extract_func_outputs(
        out,
        has_aux=self._value_func_has_aux,
        has_state=self._value_func_has_state,
    )
    stats = stats or {}
    stats["loss"] = loss
    stats, grads = jax.lax.pmean((stats, grads), axis_name=self.pmap_axis_name)

    # Compute and apply updates via our optimizer.
    updates, new_state = self._optax_optimizer.update(grads, state, params)
    if self._include_norms_in_stats:
      stats["grad_norm"] = kfac_jax.utils.norm(grads)
      stats["update_norm"] = kfac_jax.utils.norm(updates)
      stats["param_norm"] = kfac_jax.utils.norm(params)
    if self._include_per_param_norms_in_stats:
      stats.update(kfac_jax.utils.per_parameter_norm(grads, "grad_norm"))
      stats.update(kfac_jax.utils.per_parameter_norm(updates, "update_norm"))
      stats.update(kfac_jax.utils.per_parameter_norm(params, "param_norm"))
    new_params = optax.apply_updates(params, updates)

    # Add step and batch size
    stats["step"] = global_step_int + 1
    batch_size = jax.tree_util.tree_leaves(batch)[0].shape[0]
    stats["batch_size"] = batch_size * jax.device_count()
    stats["data_seen"] = stats["step"] * stats["batch_size"]

    if self._value_func_has_state:
      return new_params, new_state, new_func_state, stats
    else:
      return new_params, new_state, stats

  def step(
      self,
      params: Params,
      state: OptaxState,
      rng: PRNGKey,
      data_iterator: Iterator[Batch],
      func_state: Optional[FuncState] = None,
      global_step_int: Optional[int] = None
  ) -> Union[
      Tuple[Params, Any, FuncState, Mapping[str, Array]],
      Tuple[Params, Any, Mapping[str, Array]],
  ]:
    """A step with similar interface to KFAC."""
    batch = next(data_iterator)
    if self._preconditioner is not None:
      precond_state = self._pmap_maybe_init_preconditioner(
          params, state, rng, batch, func_state
      )
      state = replace_precondition_state(state, precond_state)
    return self._pmap_step(
        params,
        state,
        rng,
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


def cosine_schedule(
    global_step: int,
    dataset_size: int,
    train_total_batch_size: Optional[int],
    epochs: Optional[float],
    steps: Optional[int],
    peak_learning_rate: float,
    initial_learning_rate: float = 1e-7,
    end_learning_rate: float = 0.0,
    warmup_epochs: Optional[float] = None,
    warmup_steps: Optional[int] = None,
    warmup_fraction: Optional[float] = None,
    **_: Any,
) -> Array:
  """A cosine schedule described in the TAT paper."""

  if (steps is None) == (epochs is None):
    raise ValueError("Only one of `steps` and `epochs` can be set.")

  n = sum(x is not None for x in [warmup_epochs, warmup_steps, warmup_fraction])

  if n != 1:
    raise ValueError(f"Exactly one of warmup_steps={warmup_steps}, "
                     f"warmup_epochs={warmup_epochs} and warmpu_fraction="
                     f"{warmup_fraction} must be set.")

  if ((warmup_epochs is not None or epochs is not None)
      and train_total_batch_size is None):
    raise ValueError("Batch size must be known when passing epochs or "
                     "warmup_epochs.")

  if warmup_epochs is not None:
    warmup_steps = warmup_epochs * dataset_size / train_total_batch_size
  elif warmup_fraction is not None:
    warmup_steps = warmup_fraction * steps

  if epochs is not None:
    total_steps = epochs * dataset_size / train_total_batch_size
  else:
    total_steps = steps

  return optax.warmup_cosine_decay_schedule(
      init_value=initial_learning_rate,
      peak_value=peak_learning_rate,
      end_value=end_learning_rate,
      warmup_steps=warmup_steps,
      decay_steps=total_steps,
  )(global_step)


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
    **_: Any,
) -> Array:
  """A basic stepwise schedule."""

  if (epoch_boundaries is None) == (step_boundaries is None):
    raise ValueError("Only one of `epoch_boundaries` and `step_boundaries` can "
                     "be set.")

  if (warmup_epochs is None) == (warmup_steps is None):
    raise ValueError("Only one of `warmup_epochs` and `warmup_steps` can be "
                     "set.")

  if step_boundaries is None or warmup_steps is None:

    if train_total_batch_size is None:
      raise ValueError("Batch size must be known when passing epoch_boundaries "
                       "or warmup_epochs.")

    steps_per_epoch = dataset_size / train_total_batch_size
    current_epoch = global_step / steps_per_epoch

  if step_boundaries is None:
    step_boundaries = jnp.array(epoch_boundaries) * steps_per_epoch
  else:
    step_boundaries = jnp.array(step_boundaries)

  values = jnp.array(lr_decay_factors) * initial_learning_rate
  index = jnp.sum(step_boundaries <= global_step)
  lr = jnp.take(values, index)

  if warmup_steps is None:
    return lr * jnp.minimum(1., current_epoch / warmup_epochs)
  else:
    return lr * jnp.minimum(1., global_step / warmup_steps)


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
    steps: Optional[int],
    epochs: Optional[float],
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

      # Set learning rate schedule
      if kwargs.get("learning_rate_schedule") is not None:
        kwargs["learning_rate_schedule"] = construct_schedule(
            dataset_size=dataset_size,
            train_total_batch_size=train_total_batch_size,
            steps=steps,
            epochs=epochs,
            **kwargs["learning_rate_schedule"]
        )

      # Set momentum schedule
      if kwargs.get("momentum_schedule") is not None:
        kwargs["momentum_schedule"] = construct_schedule(
            dataset_size=dataset_size,
            train_total_batch_size=train_total_batch_size,
            steps=steps,
            epochs=epochs,
            **kwargs["momentum_schedule"]
        )

      # Set damping schedule
      if kwargs.get("damping_schedule") is not None:
        kwargs["damping_schedule"] = construct_schedule(
            dataset_size=dataset_size,
            train_total_batch_size=train_total_batch_size,
            steps=steps,
            epochs=epochs,
            **kwargs["damping_schedule"]
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
        steps=steps,
        epochs=epochs,
        **kwargs.pop("learning_rate_schedule")
    )
    return OptaxWrapper(
        value_and_grad_func=value_and_grad_func,
        value_func_has_aux=has_aux,
        value_func_has_rng=has_rng,
        value_func_has_state=has_func_state,
        optax_optimizer=getattr(optax, name)(
            learning_rate=learning_rate_schedule, **kwargs,
        )
    )

  else:
    raise NotImplementedError()
