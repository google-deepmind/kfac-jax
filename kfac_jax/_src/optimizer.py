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

"""K-FAC optimizer."""

import functools
from typing import Callable, Iterator, Optional, Sequence, Any, Generic, Tuple, Union, Dict

import jax
from jax import lax
import jax.numpy as jnp
import optax

from kfac_jax._src import curvature_estimator
from kfac_jax._src import utils
from typing_extensions import TypeAlias

# Types for annotation
ArrayTree = utils.ArrayTree
Array = utils.Array
PRNGKey = utils.PRNGKey
Numeric = utils.Numeric
Params = utils.Params
Batch = utils.Batch
FuncState = Any
# FuncState = utils.FuncState
FuncAux = utils.FuncAux

OptimizerState: TypeAlias = "Optimizer.State"
ScheduleType = Union[Callable[[Numeric, Optional[Numeric]], Numeric],
                     Callable[[Numeric], Numeric]]
FuncArgsVariants = Union[
    Tuple[Params, Batch],
    Tuple[Params, FuncState, Batch],
    Tuple[Params, PRNGKey, Batch],
    Tuple[Params, FuncState, PRNGKey, Batch],
]
FuncOutputs = Union[
    Array,
    Tuple[Array, FuncState],
    Tuple[Array, FuncAux],
    Tuple[Array, Tuple[FuncState, FuncAux]],
]
ValueFunc = Callable[..., FuncOutputs]
ValueAndGradFunc = Callable[..., Tuple[FuncOutputs, Params]]
ReturnWithFuncState = Tuple[
    Params, OptimizerState, FuncState, Dict[str, Array]
]
ReturnWithoutFuncState = Tuple[
    Params, OptimizerState, Dict[str, Array]
]
ReturnEither = Union[ReturnWithFuncState, ReturnWithoutFuncState]


class Optimizer(utils.WithStagedMethods):
  """The K-FAC optimizer."""

  @utils.register_state_class
  class State(Generic[Params], utils.State):
    r"""Persistent state of the optimizer.

    Attributes:
      velocities: The update to the parameters from the previous step -
        :math:`\theta_t - \theta_{t-1}`.
      estimator_state: The persistent state for the curvature estimator.
      damping: When using damping adaptation, this will contain the current
        value.
      data_seen: The number of training cases that the optimizer has processed.
      step_counter: An integer giving the current step number :math:`t`.
    """
    optax_state: ArrayTree
    velocities: Params
    estimator_state: curvature_estimator.BlockDiagonalCurvature.State
    damping: Optional[Array]
    data_seen: Numeric
    step_counter: Numeric

    @classmethod
    def from_dict(cls, dict_representation: Dict[str, Any]) -> OptimizerState:
      dict_representation["estimator_state"] = (
          curvature_estimator.BlockDiagonalCurvature.State.from_dict(
              dict_representation["estimator_state"]
          )
      )
      return cls(**dict_representation)

  def __init__(
      self,
      value_and_grad_func: ValueAndGradFunc,
      l2_reg: Numeric,
      value_func_has_aux: bool = False,
      value_func_has_state: bool = False,
      value_func_has_rng: bool = False,
      internal_optimizer = None,
      use_adaptive_learning_rate: bool = False,
      learning_rate_schedule: Optional[ScheduleType] = None,
      use_adaptive_momentum: bool = False,
      momentum_schedule: Optional[ScheduleType] = None,
      use_adaptive_damping: bool = False,
      damping_schedule: Optional[ScheduleType] = None,
      initial_damping: Optional[Numeric] = None,
      min_damping: Numeric = 1e-8,
      max_damping: Numeric = jnp.inf,
      include_damping_in_quad_change: bool = False,
      damping_adaptation_interval: int = 5,
      damping_adaptation_decay: Numeric = 0.9,
      damping_lower_threshold: Numeric = 0.25,
      damping_upper_threshold: Numeric = 0.75,
      always_use_exact_qmodel_for_damping_adjustment: bool = False,
      norm_constraint_mode = 'fisher_scaled',
      norm_constraint: Optional[Numeric] = None,
      scale_nc_by_std_dev: bool = False,
      min_clip_nc: Optional[Numeric] = 3.0,
      max_clip_nc: Optional[Numeric] = 8.0,
      scale_by_std_deviation: bool = False,
      num_burnin_steps: int = 10,
      estimation_mode: str = "fisher_gradients",
      curvature_ema: Numeric = 0.95,
      curvature_update_period: int = 1,
      inverse_update_period: int = 5,
      use_exact_inverses: bool = False,
      batch_process_func: Optional[Callable[[Batch], Batch]] = None,
      register_only_generic: bool = False,
      patterns_to_skip: Sequence[str] = (),
      auto_register_kwargs: Optional[Dict[str, Any]] = None,
      layer_tag_to_block_ctor: Optional[
          Dict[str, curvature_estimator.CurvatureBlockCtor]
      ] = None,
      multi_device: bool = False,
      debug: bool = False,
      batch_size_extractor: Callable[
          [Batch], Numeric
      ] = utils.default_batch_size_extractor,
      pmap_axis_name: str = "kfac_axis",
      forbid_setting_attributes_after_finalize: bool = True,
      modifiable_attribute_exceptions: Sequence[str] = (),
      include_norms_in_stats: bool = False,
      include_per_param_norms_in_stats: bool = False,
      distributed_precon_apply: bool = True,
      distributed_inverses: bool = True,
  ):
    """Initializes the K-FAC optimizer with the provided settings.

    A note on damping:

    One of the main complications of using second-order optimizers like K-FAC is
    the "damping" parameter. This parameter is multiplied by the identity matrix
    and (approximately) added to the curvature matrix (i.e. the Fisher or GGN)
    before it is inverted and multiplied by the gradient when computing the
    update (before any learning rate scaling). The damping should follow the
    scale of the objective, so that if you multiply your loss by some factor you
    should do the same for the damping. Roughly speaking, larger damping values
    constrain the update vector to a smaller region around zero, which is needed
    in general since the second-order approximations that underly second-order
    methods can break down for large updates. (In gradient descent the learning
    rate plays an analogous role.) The relationship between the damping
    parameter and the radius of this region is complicated and depends on the
    scale of the objective amongst other things.

    The optimizer provides a system for adjusting the damping automatically via
    the ``use_adaptive_damping`` argument, although this system is not
    completely reliable. Using a fixed value or a manually tuned schedule can
    work as good or better for some problems, while it can be a very poor choice
    for others (like deep autoencoders). Empirically we have found that using
    a fixed value works well enough for common architectures like convnets and
    transformers.

    Args:
      value_and_grad_func: Python callable. The function should return the value
        of the loss to be optimized and its gradients, and optionally the model
        state and auxiliary information (usually statistics to log). The
        interface should be: ``out_args, loss_grads =
        value_and_grad_func(*in_args)``. Here, ``in_args`` is ``(params,
        func_state, rng, batch)``, with ``rng`` omitted if
        ``value_func_has_rng`` is ``False``, and with ``func_state`` omitted if
        ``value_func_has_state`` is ``False``. Meanwhile, ``out_args`` is
        ``(loss, func_state, aux)``, with ``func_state`` omitted if
        ``value_func_has_state`` is ``False``, and with ``aux`` omitted if
        ``value_func_has_aux`` is ``False``. If both ``value_func_has_state``
        and ``value_func_has_aux`` are ``False``, ``out_args`` should just be
        ``loss`` and not ``(loss,)``.
      l2_reg: Scalar. Set this value to tell the optimizer what L2
        regularization coefficient you are using (if any). Note the coefficient
        appears in the regularizer as ``coeff / 2 * sum(param**2)``. This adds
        an additional diagonal term to the curvature and hence will affect the
        quadratic model when using adaptive damping. Note that the user is still
        responsible for adding regularization to the loss.
      value_func_has_aux: Boolean. Specifies whether the provided callable
        ``value_and_grad_func`` returns auxiliary data. (Default: ``False``)
      value_func_has_state: Boolean. Specifies whether the provided callable
        ``value_and_grad_func`` has a persistent state that is passed in and
        out. (Default: ``False``)
      value_func_has_rng: Boolean. Specifies whether the provided callable
        ``value_and_grad_func`` additionally takes as input an rng key.
        (Default: ``False``)
      use_adaptive_learning_rate: Boolean. Specifies whether to use the special
        rule from the original K-FAC paper for picking the learning rate at each
        step. Note that this won't work well for stochastic objectives. If this
        is ``False``, the user must use the ``learning_rate`` argument of the
        step function, or the constructor argument ``learning_rate_schedule``.
        (Default: ``False``)
      learning_rate_schedule: Callable. A schedule for the learning rate. This
        should take as input the current step number, and optionally the amount
        of data seen so far as a keyword argument ``data_seen``, and return a
        single array that represents the learning rate. (Default: ``None``)
      use_adaptive_momentum: Boolean. Specifies whether to use the special rule
        from the original K-FAC paper for picking the momentum "decay" parameter
        at each step. Note that this won't work well for stochastic objectives.
        If this is ``False``, the user must use the ``momentum`` argument of the
        step function, or the constructor argument ``momentum_schedule``.
        (Default: ``False``)
      momentum_schedule: Callable. A schedule for the momentum parameter. This
        should take as input the current step number, and optionally the amount
        of data seen so far as a keyword argument ``data_seen``, and return a
        single array that represents the momentum. (Default: ``None``)
      use_adaptive_damping: Boolean. Specifies whether the optimizer will use
        the Levenberg-Marquardt method to automatically adjust the damping every
        ``damping_adaptation_interval`` iterations. If this is set to ``False``
        the user must provide a value to the damping argument of the step
        function at each iteration, or use the ``damping_schedule`` constructor
        argument. Note that the effectiveness of this technique seems to vary
        between problems. (Default: ``False``)
      damping_schedule: Callable. A schedule for the damping. This should take
        as input the current step number, and optionally the amount
        of data seen so far as a keyword argument ``data_seen``, and return a
        single array that represents the learning rate. (Default: ``None``)
      initial_damping: Scalar or None. This specifies the initial value of the
        damping that the optimizer will use when using automatic damping
        adaptation. (Default: ``None``)
      min_damping: Scalar. Minimum value the damping parameter can take when
        using automatic damping adaptation. Note that the default value of 1e-8
        is quite arbitrary, and you may have to adjust this up or down for your
        particular problem. If you are using a non-zero value of l2_reg you
        *may* be able to set this to zero. (Default: ``1e-8``)
      max_damping: Scalar. Maximum value the damping parameter can take when
        using automatic damping adaptation. (Default: ``Infinity``)
      include_damping_in_quad_change: Boolean. Whether to include the
        contribution of the damping in the quadratic model for the purposes
        computing the reduction ration ("rho") in the Levenberg-Marquardt scheme
        used for adapting the damping. Note that the contribution from the
        ``l2_reg`` argument is always included. (Default: ``False``)
      damping_adaptation_interval: Int. The number of steps in between adapting
        the damping parameter. (Default: ``5``)
      damping_adaptation_decay: Scalar. The damping parameter will be adjusted
        up or down by ``damping_adaptation_decay **
        damping_adaptation_interval``, or remain unchanged, every
        ``damping_adaptation_interval`` number of iterations. (Default: ``0.9``)
      damping_lower_threshold: Scalar. The damping parameter is increased if the
        reduction ratio is below this threshold. (Default: ``0.25``)
      damping_upper_threshold: Scalar. The damping parameter is decreased if the
        reduction ratio is below this threshold. (Default: ``0.75``)
      always_use_exact_qmodel_for_damping_adjustment: Boolean. When using
        learning rate and/or momentum adaptation, the quadratic model change
        used for damping adaption is always computed using the exact curvature
        matrix. Otherwise, there is an option to use either the exact or
        approximate curvature matrix to compute the quadratic model change,
        which is what this argument controls. When True, the exact curvature
        matrix will be used, which is more expensive, but could possibly produce
        a better damping schedule. (Default: ``False``)
      norm_constraint: Scalar. If specified, the update is scaled down so that
        its approximate squared Fisher norm ``v^T F v`` is at most the specified
        value. (Note that here ``F`` is the approximate curvature matrix, not
        the exact.) May only be used when ``use_adaptive_learning_rate`` is
        ``False``. (Default: ``None``)
      num_burnin_steps: Int. At the start of optimization, e.g. the first step,
        before performing the actual step the optimizer will perform this many
        times updates to the curvature approximation without updating the actual
        parameters. (Default: ``10``)
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
      batch_process_func: Callable. A function which to be called on each batch
        before feeding to the KFAC on device. This could be useful for specific
        device input optimizations. (Default: ``None``)
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
      multi_device: Boolean. Whether to use pmap and run the optimizer on
        multiple devices. (Default: ``False``)
      debug: Boolean. If neither the step or init functions should be jitted.
        Note that this also overrides ``multi_device`` and prevents using pmap.
        (Default: ``False``)
      batch_size_extractor: A function that takes as input the function
        arguments and returns the batch size for a single device. (Default:
        ``kfac.utils.default_batch_size_extractor``)
      pmap_axis_name: String. The name of the pmap axis to use when
        ``multi_device`` is set to True. (Default: ``kfac_axis``)
      forbid_setting_attributes_after_finalize: Boolean. By default after the
        object is finalized, you can not set any of its properties. This is done
        in order to protect the user from making changes to the object
        attributes that would not be picked up by various internal methods after
        they have been compiled. However, if you are extending this class, and
        clearly understand the risks of modifying attributes, setting this to
        ``False`` will remove the restriction. (Default: ``True``)
      modifiable_attribute_exceptions: Sequence of strings. Gives a list of
        names for attributes that can be modified after finalization even when
        ``forbid_setting_attributes_after_finalize`` is ``True``. (Default:
        ``()``)
      include_norms_in_stats: Boolean. It True, the vector norms of the
        gradient, preconditioned gradient, and parameter update are included in
        the statistics returned by the step function. (Default: ``False``)
      include_per_param_norms_in_stats: Boolean. It True, the per-parameter
        vector norms of the gradient, preconditioned gradient, and parameter
        update are included in the statistics returned by the step function.
        (Default: ``False``)
      distributed_precon_apply: Boolean. Whether to distribute the application
        of the preconditioner across the different devices in a layer-wise
        fashion. If False, each device will (redundantly) perform the required
        operations for all of the layers. (Default: True)
      distributed_inverses: Boolean. Whether to distribute the inverse
        computations (required to compute the preconditioner) across the
        different devices in a layer-wise fashion. If False, each device will
        (redundantly) perform the required computations for all of the layers.
        (Default: True)
    """
    super().__init__(
        multi_device=multi_device,
        pmap_axis_name=pmap_axis_name if multi_device else None,
        debug=debug,
        forbid_setting_attributes_after_finalize=
        forbid_setting_attributes_after_finalize,
        excluded_attribute_names=modifiable_attribute_exceptions,
    )

    if use_adaptive_damping and initial_damping is None:
      raise ValueError("When use_adaptive_damping is True you must provide a "
                       "value for initial_damping.")
    if not use_adaptive_damping and initial_damping is not None:
      raise ValueError("When use_adaptive_damping is False you should not "
                       "provide a value for initial_damping.")
    if use_adaptive_learning_rate and learning_rate_schedule is not None:
      raise ValueError("If you are using adaptive learning rate than "
                       "`learning_rate_schedule` should be None.")
    if use_adaptive_momentum and momentum_schedule is not None:
      raise ValueError("If you are using adaptive momentum than "
                       "`momentum_schedule` should be None.")
    if use_adaptive_damping and damping_schedule is not None:
      raise ValueError("If you are using adaptive damping than "
                       "`damping_schedule` should be None.")
    # Flip input gradients to internal optimizer, because kfac will return param-deltas (~= -gradient)
    self._internal_optimizer = optax.chain(optax.scale(-1), internal_optimizer or optax.sgd(learning_rate_schedule))
    self._value_and_grad_func = value_and_grad_func
    self._value_func_has_aux = value_func_has_aux
    self._value_func_has_state = value_func_has_state
    self._value_func_has_rng = value_func_has_rng
    self._value_func: ValueFunc = convert_value_and_grad_to_value_func(
        value_and_grad_func,
        has_aux=value_func_has_aux,
    )
    self._l2_reg = jnp.asarray(l2_reg)
    self._use_adaptive_learning_rate = use_adaptive_learning_rate
    self._learning_rate_schedule = learning_rate_schedule
    self._use_adaptive_momentum = use_adaptive_momentum

    if momentum_schedule is not None:

      def schedule_with_first_step_zero(
          global_step: Array,
          data_seen: Optional[Numeric] = None,
      ) -> Array:

        value = utils.call_func_with_conditional_kwargs(
            momentum_schedule, global_step, data_seen=data_seen)
        check = jnp.equal(global_step, 0)

        return check * jnp.zeros_like(value) + (1 - check) * value

      self._momentum_schedule = schedule_with_first_step_zero

    else:
      self._momentum_schedule = None

    self._use_adaptive_damping = use_adaptive_damping
    self._damping_schedule = damping_schedule
    self._initial_damping = initial_damping
    self._min_damping = min_damping
    self._max_damping = max_damping
    self._include_damping_in_quad_change = include_damping_in_quad_change
    self._damping_adaptation_decay = damping_adaptation_decay
    self._damping_adaptation_interval = damping_adaptation_interval
    self._damping_lower_threshold = damping_lower_threshold
    self._damping_upper_threshold = damping_upper_threshold
    self._always_use_exact_qmodel_for_damping_adjustment = (
        always_use_exact_qmodel_for_damping_adjustment)
    self._norm_constraint_mode = norm_constraint_mode
    self._norm_constraint = norm_constraint
    self._scale_nc_by_std_dev = scale_nc_by_std_dev
    self._min_clip_nc = min_clip_nc
    self._max_clip_nc = max_clip_nc
    self._scale_by_std_deviation = scale_by_std_deviation
    self._num_burnin_steps = num_burnin_steps
    self._estimation_mode = estimation_mode
    self._curvature_ema = curvature_ema
    if curvature_update_period > inverse_update_period:
      raise ValueError(
          "curvature_update_period ({}) cannot be larger than"
          " inverse_update_period ({}) as the identical matrix inversion would"
          " be redundantly performed. Set inverse_update_period larger instead."
          .format(curvature_update_period, inverse_update_period)
      )
    self._curvature_update_period = curvature_update_period
    self._inverse_update_period = inverse_update_period
    self._register_only_generic = register_only_generic
    self._layer_tag_to_block_cls = layer_tag_to_block_ctor
    self._patterns_to_skip = patterns_to_skip
    self._batch_process_func = batch_process_func or (lambda x: x)
    self._include_norms_in_stats = include_norms_in_stats
    self._include_per_param_norms_in_stats = include_per_param_norms_in_stats
    self._batch_size_extractor = batch_size_extractor

    self._use_cached_inverses = (self._inverse_update_period != 1)
    self._use_exact_inverses = use_exact_inverses

    # Curvature estimator
    self._estimator = curvature_estimator.BlockDiagonalCurvature(
        func=self._value_func,
        default_estimation_mode=estimation_mode,
        params_index=0,
        layer_tag_to_block_ctor=layer_tag_to_block_ctor,
        register_only_generic=register_only_generic,
        patterns_to_skip=patterns_to_skip,
        distributed_multiplies=distributed_precon_apply,
        distributed_cache_updates=distributed_inverses,
        **(auto_register_kwargs or {}),
    )
    self._implicit = curvature_estimator.ImplicitExactCurvature(
        self._value_func,
        params_index=0,
        batch_size_extractor=batch_size_extractor,
    )

    # Each subclass should call finalize on its own, so this gets called only
    # for instances of exactly this class type.
    if type(self) == Optimizer:  # pylint: disable=unidiomatic-typecheck
      self.finalize()

  @property
  def num_burnin_steps(self) -> int:
    """The number of burnin steps to run before the first parameter update."""
    return self._num_burnin_steps

  @property
  def l2_reg(self) -> Array:
    """The weight of the additional diagonal term added to the curvature."""
    return self._l2_reg

  @property
  def estimator(self) -> curvature_estimator.BlockDiagonalCurvature:
    """The underlying curvature estimator used by the optimizer."""
    return self._estimator

  @property
  def damping_decay_factor(self) -> Numeric:
    """How fast to decay the damping, when using damping adaptation."""
    return self._damping_adaptation_decay ** self._damping_adaptation_interval

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

  def should_update_damping(
      self,
      state: "Optimizer.State",
  ) -> Array:
    """Whether at the current step the optimizer should update the damping."""
    return (state.step_counter + 1) % self._damping_adaptation_interval == 0

  def should_update_estimate_curvature(
      self, state: "Optimizer.State"
  ) -> Union[Array, bool]:
    """Whether at the current step the optimizer should update the curvature estimates."""
    if self._curvature_update_period == 1:
      return True
    return state.step_counter % self._curvature_update_period == 0

  def should_update_inverse_cache(
      self, state: "Optimizer.State"
  ) -> Union[Array, bool]:
    """Whether at the current step the optimizer should update the inverse curvature approximation."""
    if self._inverse_update_period == 1:
      return True
    return state.step_counter % self._inverse_update_period == 0

  @functools.partial(utils.staged, static_argnums=1)
  def _rng_split(
      self,
      rng: PRNGKey,
      num: int,
  ) -> Tuple[Array, ...]:
    """Splits the ``rng`` key."""
    return tuple(jax.random.split(rng, num))

  @utils.auto_scope_method
  def compute_loss_value(self, func_args: FuncArgsVariants) -> Array:
    """Computes the value of the loss function being optimized."""
    return self._value_func(*func_args)

  def verify_args_and_get_step_counter(
      self,
      step_counter: Array,
      learning_rate: Optional[Array] = None,
      momentum: Optional[Array] = None,
      damping: Optional[Array] = None,
      global_step_int: Optional[int] = None,
  ) -> int:
    """Verifies that the arguments passed to the step function are correct."""

    # Verify correct arguments invocation
    if self._use_adaptive_learning_rate and learning_rate is not None:
      raise ValueError("When use_adaptive_learning_rate is set to True you "
                       "should not pass a value to the step function.")

    elif not self._use_adaptive_learning_rate and (
        self._learning_rate_schedule is None and learning_rate is None):
      raise ValueError("When use_adaptive_learning_rate is set to False and "
                       "`learning_rate_schedule` is None you must provide a "
                       "value to the step function.")

    elif self._learning_rate_schedule is not None and learning_rate is not None:
      raise ValueError("When you have passed a `learning_rate_schedule` you "
                       "should not pass a value to the step function.")

    if self._use_adaptive_momentum and momentum is not None:
      raise ValueError("When use_adaptive_momentum is set to True you "
                       "should not pass a value to the step function.")

    elif not self._use_adaptive_momentum and (
        self._momentum_schedule is None and momentum is None):
      raise ValueError("When use_adaptive_momentum is set to False and "
                       "`momentum_schedule` is None you must provide a value to"
                       " the step function.")

    elif self._momentum_schedule is not None and momentum is not None:
      raise ValueError("When you have passed a `momentum_schedule` you should "
                       "not pass a value to the step function.")

    if self._use_adaptive_damping and damping is not None:
      raise ValueError("When use_adaptive_damping is set to True you "
                       "should not pass a value to the step function.")

    elif not self._use_adaptive_damping and (
        self._damping_schedule is None and damping is None):
      raise ValueError("When use_adaptive_damping is set to False and "
                       "`damping_schedule` is None you must provide a value to "
                       "the step function.")

    elif self._damping_schedule is not None and damping is not None:
      raise ValueError("When you have passed a `damping_schedule` you should "
                       "not pass a value to the step function.")

    if global_step_int is None:
      if self.multi_device:
        return int(utils.get_first(step_counter))
      else:
        return int(step_counter)

    return global_step_int

  @utils.staged
  def _setup_state_and_schedules(
      self,
      learning_rate: Optional[Array],
      momentum: Optional[Array],
      damping: Optional[Array],
      step_counter: Array,
      data_seen: Array,
  ) -> Tuple[Optional[Array], Optional[Array], Array]:
    """Helper function for setting up learning rate, momentum and damping."""

    # Compute schedules if applicable
    if self._learning_rate_schedule is not None:

      assert learning_rate is None
      learning_rate = utils.call_func_with_conditional_kwargs(
          self._learning_rate_schedule, step_counter, data_seen=data_seen)

    if self._momentum_schedule is not None:
      assert momentum is None
      momentum = utils.call_func_with_conditional_kwargs(
          self._momentum_schedule, step_counter, data_seen=data_seen)

    if self._damping_schedule is not None:
      assert damping is None
      damping = utils.call_func_with_conditional_kwargs(
          self._damping_schedule, step_counter, data_seen=data_seen)

    else:
      assert damping is not None

    return learning_rate, momentum, damping

  def _setup_func_args_and_rng(
      self,
      params: Params,
      static_args: Any,
      rng: PRNGKey,
      batch: Batch,
      func_state: Optional[FuncState],
  ) -> Tuple[FuncArgsVariants, Array]:
    """Helper function for setting up the model function arguments correctly."""

    # Preprocess the batch and construct correctly the function arguments
    batch = self._batch_process_func(batch)

    # Correctly split rng
    if self._value_func_has_rng:
      rng, func_rng = jax.random.split(rng)
    else:
      func_rng = None
    # Make the function args
    func_args = make_func_args(
        params=params,
        func_state=func_state,
        rng=func_rng,
        static_args=static_args,
        batch=batch,
        has_state=self._value_func_has_state,
        has_rng=self._value_func_has_rng,
    )

    return func_args, rng

  def _maybe_update_estimator_state(
      self,
      state: "Optimizer.State",
      should_update: Union[Array, bool],
      update_func: Callable[
          ..., curvature_estimator.BlockDiagonalCurvature.State
      ],
      **update_func_kwargs,
  ) -> "Optimizer.State":
    """Updates the estimator state if it is the right iteration."""

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    state.estimator_state = lax.cond(
        should_update,
        functools.partial(update_func, **update_func_kwargs),
        lambda state_: state_,
        state.estimator_state,
    )
    return state

  def _update_estimator_curvature(
      self,
      estimator_state: curvature_estimator.BlockDiagonalCurvature.State,
      func_args: FuncArgsVariants,
      rng: PRNGKey,
      ema_old: Numeric,
      ema_new: Numeric,
      sync: Union[Array, bool] = True
  ) -> curvature_estimator.BlockDiagonalCurvature.State:
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

  def _maybe_update_estimator_curvature(
      self,
      state: "Optimizer.State",
      func_args: FuncArgsVariants,
      rng: PRNGKey,
      ema_old: Numeric,
      ema_new: Numeric,
      sync: Union[Array, bool] = True,
  ) -> "Optimizer.State":
    """Updates the curvature estimates if it is the right iteration."""
    return self._maybe_update_estimator_state(
        state,
        self.should_update_estimate_curvature(state),
        self._update_estimator_curvature,
        func_args=func_args,
        rng=rng,
        ema_old=ema_old,
        ema_new=ema_new,
        sync=sync,
    )

  @utils.auto_scope_method
  def _compute_loss_and_grads(
      self,
      func_args: FuncArgsVariants,
  ) -> Tuple[Array, Params, FuncState, FuncAux]:
    """Computes the model loss value and its gradients."""

    out, grads = self._value_and_grad_func(*func_args)

    loss, func_state, aux = extract_func_outputs(
        out, self._value_func_has_aux, self._value_func_has_state)

    return loss, grads, func_state, aux

  def _maybe_update_inverse_cache(
      self,
      state: "Optimizer.State",
      damping: Array,
  ) -> "Optimizer.State":
    """Updates the estimator state cache if it is the right iteration."""
    return self._maybe_update_estimator_state(
        state,
        self.should_update_inverse_cache(state),
        self.estimator.update_cache,
        identity_weight=self.l2_reg + damping,
        exact_powers=self._exact_powers_to_cache,
        approx_powers=self._approx_powers_to_cache,
        eigenvalues=False,
        pmap_axis_name=self.pmap_axis_name,
    )

  # TODO(jamesmartens, botev): It's ugly that this method implements the norm
  # constraint on top of computing the preconditioned gradient. Should refactor.
  @utils.staged
  def _compute_preconditioned_gradient(
      self,
      state: "Optimizer.State",
      grads: Params,
      coefficient: Optional[Array],
      damping: Array,
      norm_constraint,
  ) -> Tuple[Params, Optional[Array]]:
    """Computes the preconditioned gradient, maybe applying norm-constraint."""

    preconditioned_grads = self.estimator.multiply_inverse(
        state=state.estimator_state,
        parameter_structured_vector=grads,
        identity_weight=self.l2_reg + damping,
        exact_power=self._use_exact_inverses,
        use_cached=self._use_cached_inverses,
        pmap_axis_name=self.pmap_axis_name,
    )


    norm_constraint_factor = None
    if norm_constraint is not None:
      assert not self._use_adaptive_learning_rate
      assert coefficient is not None

      if self._norm_constraint_mode == 'fisher':
        sq_norm_grads = utils.inner_product(preconditioned_grads, grads)
      elif self._norm_constraint_mode == 'fisher_scaled':
        sq_norm_grads = utils.inner_product(preconditioned_grads, grads)
        sq_norm_grads = sq_norm_grads * coefficient ** 2
      elif self._norm_constraint_mode == 'precond_grad':
        sq_norm_grads = utils.inner_product(preconditioned_grads, preconditioned_grads)
      elif self._norm_constraint_mode == 'precond_grad_scaled':
        sq_norm_grads = utils.inner_product(preconditioned_grads, preconditioned_grads)
        sq_norm_grads = sq_norm_grads * coefficient ** 2
      else:
        raise ValueError(f"Unknown norm_constraint_mode: {self._norm_constraint_mode}")

      # We need to sync the norms here, because reduction can be
      # non-deterministic. They specifically are on GPUs by default for better
      # performance. Hence although grads and preconditioned_grads are synced,
      # the inner_product operation can still produce different answers on
      # different devices.
      sq_norm_grads = utils.pmean_if_pmap(sq_norm_grads, self.pmap_axis_name)
      norm_constraint_factor = jnp.sqrt(norm_constraint / sq_norm_grads)
      preconditioned_grads = utils.scalar_mul(preconditioned_grads, jnp.minimum(norm_constraint_factor, 1))

    return preconditioned_grads, norm_constraint_factor

  def _compute_quad_change_for_damping(
      self,
      state: "Optimizer.State",
      delta: Params,
      grads: Params,
      damping: Array,
      func_args: FuncArgsVariants,
  ) -> Array:
    """The quadratic model change, when lr and momentum are non-adaptive."""

    assert not (self._use_adaptive_learning_rate or self._use_adaptive_momentum)

    if self._always_use_exact_qmodel_for_damping_adjustment:
      quad_model = self.compute_exact_quad_model(
          [delta], grads, func_args)
    else:
      quad_model = self.compute_approx_quad_model(state, [delta], grads)

    w = jnp.ones([])
    return self._solve_quad_model(quad_model, damping, [delta], [w])[1]

  def _coefficients_and_quad_change(
      self,
      state: "Optimizer.State",
      vectors: Sequence[Params],
      grads: Params,
      learning_rate: Optional[Array],
      momentum: Optional[Array],
      damping: Array,
      func_args: Optional[FuncArgsVariants] = None,
  ) -> Tuple[Tuple[Optional[Array], Optional[Array]], Array]:
    """The correct update coefficients and corresponding quadratic change."""

    # Compute the coefficients of the update vectors
    # The learning rate is defined as the negative of the coefficient by which
    # we multiply the gradients, while the momentum is the coefficient by
    # which we multiply the velocities.
    neg_learning_rate = -learning_rate if learning_rate is not None else None
    coefficients = (neg_learning_rate, momentum)

    if self._use_adaptive_learning_rate or self._use_adaptive_momentum:

      quad_model = self.compute_exact_quad_model(vectors, grads, func_args)

      return self._solve_quad_model(quad_model, damping, vectors, coefficients)
    else:
      assert all(c is not None for c in coefficients)

      if self._use_adaptive_damping:
        delta = self.weighted_sum_of_objects(vectors, coefficients)

        quad_change = lax.cond(
            self.should_update_damping(state),
            lambda args: self._compute_quad_change_for_damping(*args),
            lambda args: jnp.nan,
            (state, delta, grads, damping, func_args),
        )

      else:
        quad_change = jnp.nan

      return coefficients, quad_change

  @utils.auto_scope_method
  def _update_damping(
      self,
      old_damping: Array,
      old_loss: Array,
      quad_change: Array,
      new_func_args: FuncArgsVariants,
  ) -> Tuple[Array, Array, Array]:
    """Updates the damping parameter."""

    new_loss = self.compute_loss_value(new_func_args)

    # Sync
    new_loss = utils.pmean_if_pmap(new_loss, self.pmap_axis_name)

    damping, rho = self._compute_new_damping_and_rho(
        old_loss, new_loss, quad_change, old_damping)

    return damping, rho, new_loss

  # @utils.staged
  @functools.partial(utils.staged, static_argnums=[2])
  def _init(
      self,
      params: Params,
      rng: PRNGKey,
      static_args: Any,
      batch: Batch,
      func_state: Optional[FuncState] = None,
  ) -> "Optimizer.State":
    """A staged function to initialize the optimizer state ."""

    return Optimizer.State(
        optax_state=self._internal_optimizer.init(params),
        velocities=jax.tree_util.tree_map(jnp.zeros_like, params),
        estimator_state=self.estimator.init(
            rng=rng,
            func_args=make_func_args(
                params=params,
                func_state=func_state,
                rng=rng,
                static_args=static_args,
                batch=self._batch_process_func(batch),
                has_state=self._value_func_has_state,
                has_rng=self._value_func_has_rng,
            ),
            exact_powers_to_cache=self._exact_powers_to_cache,
            approx_powers_to_cache=self._approx_powers_to_cache,
            cache_eigenvalues=False
        ),
        damping=(jnp.array(self._initial_damping, dtype=float)
                 if self._use_adaptive_damping else None),
        data_seen=jnp.array(0, dtype=int),
        step_counter=jnp.array(0, dtype=int)
    )

  def init(
      self,
      params: Params,
      rng: PRNGKey,
      batch: Batch,
      static_args: Optional[Any] = None,
      func_state: Optional[FuncState] = None,
  ) -> "Optimizer.State":
    """Initializes the optimizer and returns the appropriate optimizer state."""

    if not self.finalized:
      self.finalize(params, rng, static_args, batch, func_state)

    return self._init(params, rng, static_args, batch, func_state)

  @functools.partial(utils.staged, donate_argnums=[1, 3, 5])
  def _burnin(
      self,
      params: Params,
      state: "Optimizer.State",
      rng: Array,
      batch: Batch,
      func_state: Optional[FuncState],
      accumulator: utils.MultiChunkAccumulator
  ) -> Tuple["Optimizer.State", utils.MultiChunkAccumulator]:
    """A single burnin step, updating only the curvature estimate."""

    # Copy this first since we mutate it later in this function.
    accumulator = accumulator.copy()

    func_args, rng = self._setup_func_args_and_rng(
        params, rng, batch, func_state)

    # Update curvature estimate
    state.estimator_state = self._update_estimator_curvature(
        state.estimator_state, func_args, rng, 1.0, 1.0)

    # Optionally update func_state
    if func_state is not None:
      out, _ = self._value_and_grad_func(*func_args)
      _, func_state, _ = extract_func_outputs(
          out, self._value_func_has_aux, self._value_func_has_state)

    accumulator.add(func_state)

    return state, accumulator

  def burnin(
      self,
      num_steps: int,
      params: Params,
      state: "Optimizer.State",
      rng: PRNGKey,
      data_iterator: Iterator[Batch],
      func_state: Optional[FuncState] = None,
  ) -> Tuple["Optimizer.State", Optional[FuncState]]:
    """Runs all burnin steps required."""

    if num_steps > 0:
      rng = self._rng_split(rng, num_steps)

      accumulator = utils.MultiChunkAccumulator.zeros_like(
          func_state, self.multi_device)

      for rng_i in rng:
        batch = next(data_iterator)

        state, accumulator = self._burnin(
            params, state, rng_i, batch, func_state, accumulator)

      func_state = accumulator.value_and_clear()

    return state, func_state

  @functools.partial(utils.staged, static_argnums=(2,), donate_argnums=(1, 5))
  @utils.auto_scope_method
  def _step(
      self,
      params: Params,
      state: "Optimizer.State",
      static_args: any,
      rng: Array,
      batch: Batch,
      func_state: Optional[FuncState],
      learning_rate: Optional[Array],
      momentum: Optional[Array],
      damping: Optional[Array]
  )-> ReturnEither:
    """A single full step of the optimizer."""

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    # Setup arguments
    learning_rate, momentum, damping = self._setup_state_and_schedules(
        learning_rate, momentum,
        state.damping if self._use_adaptive_damping else damping,
        state.step_counter, state.data_seen)

    func_args, rng = self._setup_func_args_and_rng(
        params, static_args, rng, batch, func_state)

    # Update curvature estimate
    state = self._maybe_update_estimator_curvature(
        state,
        func_args,
        rng,
        self._curvature_ema,
        1.0,
        sync=self.should_update_inverse_cache(
            state
        ),  # sync curvature estimates only before inverses are updated.
    )

    del rng  # should not be used after this point!

    # Compute loss and gradients
    loss, grads, func_state, aux = self._compute_loss_and_grads(func_args)

    # Sync
    loss, grads = utils.pmean_if_pmap((loss, grads), self.pmap_axis_name)

    # Update the inverse curvature
    state = self._maybe_update_inverse_cache(state, damping)

    if self._scale_nc_by_std_dev:
        norm_constraint = jnp.clip(self._norm_constraint / jnp.sqrt(aux['E_var']), self._min_clip_nc, self._max_clip_nc)
    else:
        norm_constraint = self._norm_constraint

    # Compute proposed directions
    preconditioned_gradient, norm_constraint_factor = (
        self._compute_preconditioned_gradient(state, grads, learning_rate,
                                              damping, norm_constraint)
    )

    if self._scale_by_std_deviation:
      preconditioned_gradient = utils.scalar_mul(preconditioned_gradient, jnp.minimum(1 / jnp.sqrt(aux['E_var']), 1))

    vectors = (preconditioned_gradient, state.velocities)

    # Compute the coefficients for the vectors
    coefficients, quad_model_change = self._coefficients_and_quad_change(
        state=state,
        vectors=vectors,
        grads=grads,
        learning_rate=learning_rate,
        momentum=momentum,
        damping=damping,
        func_args=func_args)

    # Compute delta and update velocities
    delta = self.weighted_sum_of_objects(vectors, coefficients)
    delta, state.optax_state = self._internal_optimizer.update(delta, state.optax_state, params)
    state.velocities = delta

    # Update parameters
    params = jax.tree_util.tree_map(jnp.add, params, delta)

    # Optionally compute the reduction ratio and update the damping
    if self._use_adaptive_damping:

      state.damping, rho, new_loss = lax.cond(
          self.should_update_damping(state),
          lambda args: self._update_damping(*args),
          lambda args: (args[0], jnp.nan, jnp.nan),
          operand=(state.damping, loss, quad_model_change,
                   (params,) + func_args[1:])
      )
    else:
      new_loss, rho = jnp.nan, jnp.nan

    # Compute per-device and total batch size
    batch_size = self._batch_size_extractor(func_args[-1])

    if self.multi_device:
      total_batch_size = batch_size * jax.device_count()
    else:
      total_batch_size = batch_size

    # Update data seen and step counter
    state.data_seen = state.data_seen + total_batch_size
    state.step_counter = state.step_counter + 1

    # Statistics with useful information
    # Unlike other norm stats, sq_norm_scaled_grads has to be computed if
    # norm_constraint is not None, so log it by default even if the other
    # norm stats are not logged. This reduces the overall computational cost if
    # no other grad stats are desired.
    stats = dict(
        step=state.step_counter,
        batch_size=jnp.asarray(total_batch_size, dtype=jnp.int32),
        data_seen=state.data_seen,
        loss=loss,
        new_loss=new_loss,
        learning_rate=-coefficients[0],
        momentum=coefficients[1],
        damping=damping,
        rho=rho,
        quad_model_change=quad_model_change,
    )

    if self._value_func_has_aux:
      stats["aux"] = utils.pmean_if_pmap(aux, self.pmap_axis_name)

    if self._include_norms_in_stats:
      stats["param_norm"] = utils.norm(params)
      stats["grad_norm"] = utils.norm(grads)
      stats["precon_grad_norm"] = utils.norm(preconditioned_gradient)
      stats["update_norm"] = utils.norm(delta)
      stats["norm_constraint_factor"] = norm_constraint_factor
      stats["norm_constraint"] = norm_constraint

    if self._include_per_param_norms_in_stats:
      stats.update(utils.per_parameter_norm(params, "param_norm"))
      stats.update(utils.per_parameter_norm(grads, "grad_norm"))
      stats.update(
          utils.per_parameter_norm(preconditioned_gradient, "precon_grad_norm")
      )
      stats.update(utils.per_parameter_norm(delta, "update_norm"))

    if self._value_func_has_state:
      return params, state, func_state, stats
    else:
      assert func_state is None
      return params, state, stats

  def step(
      self,
      params: Params,
      state: "Optimizer.State",
      static_args: Any,
      rng: PRNGKey,
      data_iterator: Optional[Iterator[Batch]] = None,
      batch: Optional[Batch] = None,
      func_state: Optional[FuncState] = None,
      learning_rate: Optional[Array] = None,
      momentum: Optional[Array] = None,
      damping: Optional[Array] = None,
      global_step_int: Optional[int] = None
  )-> ReturnEither:
    """Performs a single update step using the optimizer.

    Args:
      params: The current parameters of the model.
      state: The current state of the optimizer.
      static_args: Static arguments passed to the (optimized) function
      rng: A Jax PRNG key. Should be different for each iteration and
        each Jax process/host.
      data_iterator: A data iterator to use (if not passing ``batch``).
      batch: A single batch used to compute the update. Should only pass one
        of ``data_iterator`` or ``batch``.
      func_state: Any function state that gets passed in and returned.
      learning_rate: Learning rate to use if the optimizer was created with
        ``use_adaptive_learning_rate=True``, ``None`` otherwise.
      momentum: Momentum to use if the optimizer was created with
        ``use_adaptive_momentum=True``, ``None`` otherwise.
      damping: Damping to use if the optimizer was created with
        ``use_adaptive_damping=True``, ``None`` otherwise. See discussion
        of constructor argument ``initial_damping`` for more information about
        damping.
      global_step_int: The global step as a python int. Note that this must
        match the step internal to the optimizer that is part of its state.

    Returns:
      (params, state, stats) if ``value_func_has_state=False`` and
      (params, state, func_state, stats) otherwise, where

          * params is the updated model parameters.

          * state is the updated optimizer state.

          * func_state is the updated function state.

          * stats is a dictionary of useful statistics including the loss.
    """

    if (data_iterator is None) == (batch is None):
      raise ValueError("Exactly one of the arguments ``data_iterator`` and "
                       "``batch`` must be provided.")

    step_counter_int = self.verify_args_and_get_step_counter(
        step_counter=state.step_counter,
        learning_rate=learning_rate,
        momentum=momentum,
        damping=damping,
        global_step_int=global_step_int,
    )

    if step_counter_int == 0:

      if data_iterator is not None:

        rng, burnin_rng = self._rng_split(rng, 2)
        state, func_state = self.burnin(
            num_steps=self.num_burnin_steps,
            params=params,
            state=state,
            rng=burnin_rng,
            data_iterator=data_iterator,
            func_state=func_state,
        )

    if data_iterator is not None:
      batch = next(data_iterator)

    return self._step(params, state, static_args, rng, batch, func_state,
                      learning_rate, momentum, damping)

  def compute_l2_quad_matrix(
      self,
      vectors: Sequence[Params]
  ) -> Array:
    """Computes the matrix corresponding to the prior/regularizer.

    Args:
      vectors: A sequence of parameter-like PyTree structures, each one
      representing a different vector.

    Returns:
      A matrix with i,j entry equal to ``self.l2_reg * v_i^T v_j``.
    """
    return self.l2_reg * utils.matrix_of_inner_products(vectors)

  @utils.auto_scope_method
  def compute_exact_quad_model(
      self,
      vectors: Sequence[Params],
      grads: Params,
      func_args: Optional[FuncArgsVariants] = None,
  ) -> Tuple[Array, Array, Array]:
    """Computes the components of the exact quadratic model."""
    if func_args is None:
      raise ValueError("When you have not provided `c_factor_v` you must "
                       "provide `func_args`.")
    if self.estimator.default_mat_type == "fisher":
      c_factor_v = tuple(self._implicit.multiply_fisher_factor_transpose
                         (func_args, vi) for vi in vectors)
    elif self.estimator.default_mat_type == "ggn":
      c_factor_v = tuple(self._implicit.multiply_ggn_factor_transpose
                         (func_args, vi) for vi in vectors)
    else:
      raise ValueError(f"Unrecognized estimator.mat_type="
                       f"{self.estimator.default_mat_type}.")

    return (utils.matrix_of_inner_products(c_factor_v),
            utils.matrix_of_inner_products(vectors),
            utils.vector_of_inner_products(grads, vectors))

  @functools.partial(utils.staged, donate_argnums=2)
  @utils.auto_scope_method
  def compute_approx_quad_model(
      self,
      state: "Optimizer.State",
      vectors: Sequence[Params],
      grads: Params,
  ) -> Tuple[Array, Array, Array]:
    """Computes the components of the approximate quadratic model."""

    # v_i^T C v_j
    def c_times_v(v):
      return self.estimator.multiply(
          state=state.estimator_state,
          parameter_structured_vector=v,
          identity_weight=0.0,
          exact_power=True,
          use_cached=False,
          pmap_axis_name=self.pmap_axis_name,
      )

    c_vectors = [c_times_v(v_i) for v_i in vectors]

    return (utils.symmetric_matrix_inner_products(c_vectors, vectors),
            utils.matrix_of_inner_products(vectors),
            utils.vector_of_inner_products(grads, vectors))

  @utils.staged
  def compute_quadratic_model_value(
      self,
      a: Array,
      a_damped: Array,
      b: Array,
      w: Array,
  ) -> Array:
    """Computes the quadratic model value from the inputs provided."""

    a_final = a_damped if self._include_damping_in_quad_change else a

    return jnp.dot(w, jnp.dot(a_final, w)) / 2 + jnp.dot(w, b)

  @utils.staged
  def _solve_quad_model(
      self,
      quad_model_parameters: Tuple[Array, Array, Array],
      damping: Array,
      vectors: Sequence[Params],
      fixed_coefficients: Optional[Sequence[Union[Numeric, None]]] = None,
  ) -> Tuple[Tuple[Array, ...], Array]:
    """Solves for the optimal learning rate and momentum of the quadratic model.

    The quadratic model is represented as:
      Q(w) = w^T V^T (C + damping * I) V w / 2.0 + w^T V^T g
    where (n - number of vectors, d - dimensions of each vector):
      w (n,) - the vector of free weights (learning rate and momentum)
      V (d, n) - the matrix of proposed vectors for each weight
      C (d, d) - the true curvature matrix (GGN/Fisher/Hessian)
      g (d,) - the true gradient
      damping - the damping value at the current iteration

    In the implementation we have:
      A = V^T C V
      D = V^T V
      b = V^T g

    Args:
      quad_model_parameters: The computed matrices A, D and vector b.
      damping: The damping to use for evaluating the quadratic model.
      vectors: The parameter-like vectors for which to evaluate.
      fixed_coefficients: A list of values and None indicating which weights are
          fixed, and the quadratic is solved only for those that aren't.
    Returns:
     A list of coefficients which are the solution (and include any values that
     are not None from fixed_weights) and the value of the quadratic model
     function for this solution (as a scalar).
    Raises:
      The function currently supports only up to two vectors, hence if you
      provide more, it will raise a ``NotImplementedError``.
    """
    # TODO(jamesmartens,botev): it would be better if this method didn't need
    # to have 'vectors' passed. We could instead use the 'D' matrix to get the
    # to get the matrix for the l2 regularization.

    if fixed_coefficients is None:
      fixed_coefficients = (None,) * len(vectors)

    if len(vectors) != len(fixed_coefficients):
      raise ValueError("The length of `vectors` must be equal to the length of "
                       "`fixed_coefficients`.")

    # pylint: disable=invalid-name
    A_no_diag, D, b = quad_model_parameters
    A = A_no_diag + self.compute_l2_quad_matrix(vectors)
    A_damped = A + damping * D

    # Sync.
    # TODO(jamesmartens, botev): we should perform this earlier since it's
    # dangerous to have the convention of doing it right before use (especially
    # since the convention everywhere else is to sync quantities immediately
    # after they are first computed).
    A, A_damped, b = utils.pmean_if_pmap((A, A_damped, b), self.pmap_axis_name)
    # This needs explicit annotation
    A_damped: Array

    if all(c is None for c in fixed_coefficients):
      # Adapt all coefficients

      if len(fixed_coefficients) == 1:
        # This special case arises at the first iteration, because all
        # velocities are zeros.

        special_case = jnp.logical_and(A_damped[0, 0] == 0, b[0] == 0)
        w = - lax.cond(special_case, lambda: b, lambda: b / A_damped[0])

      elif len(fixed_coefficients) == 2:
        # This special case arises at the first iteration, because all
        # velocities are zeros.

        to_check = jnp.asarray([A_damped[0, 1], A_damped[1, 0],
                                A_damped[1, 1], b[1]])

        w = - lax.cond(jnp.all(to_check == 0),
                       lambda: jnp.stack([b[0] / A_damped[0, 0], b[1]]),
                       lambda: jnp.linalg.solve(A_damped, b))
      else:
        raise NotImplementedError()

    elif all(c is not None for c in fixed_coefficients):
      # No coefficients adapted

      w = jnp.asarray(fixed_coefficients)

    elif len(vectors) == 2:
      # Exactly one adapted coefficient

      w = [None, None]
      index = fixed_coefficients.index(None)
      w[1 - index] = jnp.asarray([fixed_coefficients[1 - index]])

      b_extra = A_damped[1 - index, index: index + 1] * w[1 - index]
      A_solve = A_damped[index: index + 1, index: index + 1]
      b_solve = b[index: index + 1] + b_extra
      # pylint: enable=invalid-name

      w[index] = - b_solve / A_solve[0]
      w = jnp.concatenate(w, axis=0)

    else:
      raise NotImplementedError()

    quadratic_value = self.compute_quadratic_model_value(A, A_damped, b, w)

    return tuple(w), quadratic_value

  @utils.staged
  def _compute_new_damping_and_rho(
      self,
      old_loss: Array,
      new_loss: Array,
      quad_change: Array,
      current_damping: Array,
  ) -> Tuple[Array, Array]:
    """Computes the reduction ratio and the updated value of the damping."""

    # Reduction ratio
    rho = (new_loss - old_loss) / quad_change
    rho_not_nan = jnp.nan_to_num(rho, nan=-100.0)

    # Update damping
    should_increase = rho_not_nan < self._damping_lower_threshold
    increased_damping = current_damping / self.damping_decay_factor
    should_decrease = rho_not_nan > self._damping_upper_threshold
    decreased_damping = current_damping * self.damping_decay_factor

    # This is basically an if-else statement
    damping = (should_decrease * decreased_damping +
               should_increase * increased_damping +
               (1 - should_increase - should_decrease) * current_damping)

    return jnp.clip(damping, self._min_damping, self._max_damping), rho

  @utils.staged
  def weighted_sum_of_objects(
      self,
      objects: Sequence[utils.PyTree],
      coefficients: Sequence[Numeric],
  ) -> utils.PyTree:
    """Returns the weighted sum of the objects in the sequence."""
    return utils.weighted_sum_of_objects(objects, coefficients)


def convert_value_and_grad_to_value_func(
    value_and_grad_func: ValueAndGradFunc,
    has_aux: bool = False,
) -> ValueFunc:
  """Converts a value_and_grad function to value_func only.

  Args:
    value_and_grad_func: The function which computes the loss value and the
      gradients w.r.t. parameters.
    has_aux: Similar to the meaning in :func:`jax.grad`, whether the
      ``value_and_grad_func`` returns with the loss value any auxiliary data.

  Returns:
    A function that returns only the loss value.
  """
  def value_func(*args) -> Array:
    out, _ = value_and_grad_func(*args)
    return out[0] if has_aux else out

  return value_func


def make_func_args(
    params: Params,
    func_state: Optional[FuncState],
    rng: Optional[PRNGKey],
    static_args: Any,
    batch: Batch,
    has_state: bool,
    has_rng: bool,
) -> FuncArgsVariants:
  """Constructs the arguments to the model function in the pre-assumed order.

  The model function is assumed to take arguments in the following order:
    params, func_state, rng, batch
  If it has no function state or does not use an rng, those two arguments are
  discarded.

  Args:
    params: The model parameters.
    func_state: The function state, if ``has_state`` is ``True``, ``None``
      otherwise.
    rng: The PRNG, if ``has_rng`` is ``True``, ``None`` otherwise.
    static_args: Static arguments passed to the function
    batch: The batch of data.
    has_state: Whether the function has a function state.
    has_rng: Whether the function uses an rng.

  Returns:
    The arguments that need to be passed to the model function.
  """
  if has_state and func_state is None:
    raise ValueError("`func_state=None`, but argument `has_state=True`.")

  if has_rng and rng is None:
    raise ValueError("`rng=None`, but argument `has_rng=True`.")

  # make sure static_args is always the third returned argument
  # this is important for filtering static arguments out of jaxpr
  if not has_state and not has_rng:
    return params, batch, static_args

  elif not has_rng:
    return params, func_state, static_args, batch

  elif not has_state:
    return params, rng, static_args, batch

  else:
    return params, func_state, static_args, rng, batch


def extract_func_outputs(
    raw_outputs: FuncOutputs,
    has_aux: bool,
    has_state: bool,
) -> Tuple[Array, Optional[FuncState], Optional[FuncAux]]:
  """Converts the raw output of the model function into loss,func_state and aux.

  Args:
    raw_outputs: The direct output of the model function.
    has_aux: Whether the model function returns also some auxiliary data.
    has_state: Whether the model function has a function state.

  Returns:
    A triple ``(loss, func_state, aux)``. If the model function does not return
    any auxiliary data than ``aux`` will be ``None`` and if it does not have a
    state ``func_state`` will be ``None``.
  """

  if not has_aux and not has_state:
    return raw_outputs, None, None

  loss, other = raw_outputs

  if has_aux and has_state:
    func_state, aux = other
  elif has_aux:
    func_state, aux = None, other
  else:
    func_state, aux = other, None

  return loss, func_state, aux
