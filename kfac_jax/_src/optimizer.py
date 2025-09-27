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

"""The kfac_jax optimizer (supporting K-FAC and other methods)."""

import functools
from typing import Callable, Iterator, Sequence, Any, Generic

import jax
from jax import lax
import jax.numpy as jnp
from kfac_jax._src import curvature_estimator
from kfac_jax._src import utils
from typing_extensions import Self


# Types for annotation
Array = utils.Array
PRNGKey = utils.PRNGKey
Numeric = utils.Numeric
Params = utils.Params
Batch = utils.Batch
FuncState = Any
FuncAux = utils.FuncAux
Scalar = utils.Scalar
ScheduleType = utils.ScheduleType

FuncArgsVariants = (
    tuple[Params, Batch] |
    tuple[Params, FuncState, Batch] |
    tuple[Params, PRNGKey, Batch] |
    tuple[Params, FuncState, PRNGKey, Batch]
)
FuncOutputs = (
    Array |
    tuple[Array, FuncState] |
    tuple[Array, FuncAux] |
    tuple[Array, tuple[FuncState, FuncAux]]
)
ValueFunc = Callable[..., FuncOutputs]
ValueAndGradFunc = Callable[..., tuple[FuncOutputs, Params]]
BlockDiagonalCurvature = curvature_estimator.BlockDiagonalCurvature

ReturnEither = (
    tuple[Params, "Optimizer.State", FuncState, dict[str, Numeric]] |
    tuple[Params, "Optimizer.State", dict[str, Numeric]]
)

# See Optimizer._solve_quad_model for a description of these parameters.
QuadModelParams = tuple[Array, Array, Array, Array]


class Optimizer(utils.WithStagedMethods):
  """The kfac_jax optimizer (supporting K-FAC and other methods)."""

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
    velocities: Params
    estimator_state: BlockDiagonalCurvature.State
    damping: Array | None
    data_seen: Numeric
    step_counter: Numeric

    @classmethod
    def from_dict(cls, dict_representation: dict[str, Any]) -> Self:
      dict_representation["estimator_state"] = (
          BlockDiagonalCurvature.State.from_dict(
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
      value_func_for_estimator: ValueFunc | None = None,
      use_adaptive_learning_rate: bool = False,
      learning_rate_schedule: ScheduleType | None = None,
      use_adaptive_momentum: bool = False,
      momentum_schedule: ScheduleType | None = None,
      use_adaptive_damping: bool = False,
      damping_schedule: ScheduleType | None = None,
      initial_damping: Numeric | None = None,
      use_initial_damping_calibration: bool = False,
      min_damping: Numeric = 1e-8,
      max_damping: Numeric = jnp.inf,
      include_damping_in_quad_change: bool = False,
      damping_adaptation_interval: int = 5,
      damping_adaptation_decay: Numeric = 0.9,
      damping_lower_threshold: Numeric = 0.25,
      damping_upper_threshold: Numeric = 0.75,
      always_use_exact_qmodel_for_damping_adjustment: bool = False,
      precon_damping_mult: Numeric = 1.0,
      precon_damping_schedule: ScheduleType | None = None,
      use_step_rejection: bool = False,
      reject_damping_increase_factor: float = 1.0,
      norm_constraint: Numeric | None = None,
      num_burnin_steps: int = 10,
      estimation_mode: str | None = None,
      custom_estimator_ctor: (
          Callable[..., BlockDiagonalCurvature] | None) = None,
      curvature_ema: Numeric = 0.95,
      curvature_update_period: int = 1,
      inverse_update_period: int = 5,
      use_exact_inverses: bool = False,
      batch_process_func: Callable[[Batch], Batch] | None = None,
      register_only_generic: bool = False,
      patterns_to_skip: Sequence[str] = (),
      use_automatic_registration: bool = True,
      auto_register_kwargs: dict[str, Any] | None = None,
      layer_tag_to_block_ctor: (
          dict[str, curvature_estimator.CurvatureBlockCtor] | None) = None,
      multi_device: bool = False,
      debug: bool = False,
      invalid_metric_value: Numeric = jnp.nan,
      batch_size_extractor: Callable[
          [Batch], Numeric
      ] = utils.default_batch_size_extractor,
      pmap_axis_name: str = "kfac_axis",
      forbid_setting_attributes_after_finalize: bool = True,
      modifiable_attribute_exceptions: Sequence[str] = (),
      include_norms_in_stats: bool = False,
      include_per_param_norms_in_stats: bool = False,
      include_registered_loss_in_stats: bool = False,
      distributed_precon_apply: bool = True,
      distributed_inverses: bool = True,
      num_estimator_samples: int = 1,
      should_vmap_estimator_samples: bool = False,
      norm_to_scale_identity_weight_per_block: str | None = None,
      precon_power: Scalar = -1.0,
  ):
    """Initializes the kfac_jax optimizer with the provided settings.

    NOTE: Please read the docstring for this constructor carefully. Especially
    the description of ``value_and_grad_func``.

    A note on the "damping" parameter:

    One of the main complications of using second-order optimizers like K-FAC is
    the "damping" parameter. This parameter is multiplied by the identity matrix
    and (approximately) added to the curvature matrix (i.e. the Fisher or GGN)
    before it is inverted and multiplied by the gradient when computing the
    update (before any learning rate scaling). The damping should follow the
    scale of the objective, so that if you multiply your loss by some factor you
    should do the same for the damping. Roughly speaking, larger damping values
    constrain the update vector to a smaller region around zero, which is needed
    in general since the second-order approximations that underlie second-order
    methods can break down for large updates. (In gradient descent the learning
    rate plays an analogous role.) The relationship between the damping
    parameter and the radius of this region is complicated and depends on the
    scale of the objective amongst other things.

    The optimizer provides a system for adjusting the damping automatically via
    the ``use_adaptive_damping`` argument, although this system is not reliable,
    especially for highly stochastic objectives. Using a fixed value or a
    manually tuned schedule can work as good or better for some problems, while
    it can be a very poor choice for others (like deep autoencoders).
    Empirically we have found that using a fixed value works well enough for
    common architectures like convnets and transformers.

    Args:
      value_and_grad_func: Python callable. This function should return the
        value of the loss to be optimized and its gradients, and optionally the
        model state and auxiliary information in the form of a a dict mapping
        strings to scalar arrays (usually statistics to log). Note that it
        should *not* be jitted/pmapped or otherwise compiled by JAX, as this can
        lead to errors. (Compilation is done internally by the optimizer.) The
        interface of this function should be: ``out_args, loss_grads =
        value_and_grad_func(*in_args)``. Here, ``in_args`` is ``(params,
        func_state, rng, batch)``, with ``rng`` omitted if
        ``value_func_has_rng`` is ``False``, and with ``func_state`` omitted if
        ``value_func_has_state`` is ``False``. Meanwhile, ``out_args`` is
        ``(loss, (func_state, aux))`` if ``value_func_has_state`` and
        ``value_func_has_aux`` are both ``True``, ``(loss, func_state)`` if
        ``value_func_has_state`` is ``True`` and ``value_func_has_aux`` is
        ``False``, ``(loss, aux)`` if ``value_func_has_state`` is ``False`` and
        ``value_func_has_aux`` is ``True``, and finally ``loss`` if
        ``value_func_has_state`` and ``value_func_has_aux`` are both ``False``.
        This should be consistent with how JAX's ``value_and_grad`` API function
        is typically used.
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
      value_func_for_estimator: ValueFunc. If specified, this function will be
        used by the preconditioner estimator instead of ``value_and_grad_func``.
        This is useful for cases where the value function used for training is
        expensive to add to the preconditioner, e.g. because it has costly
        regularizers. (Default: ``None``)
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
        as input the current step number, and optionally the amount of data seen
        so far as a keyword argument ``data_seen``, and return a single array
        that represents the learning rate. (Default: ``None``)
      initial_damping: Scalar or None. This specifies the initial value of the
        damping that the optimizer will use when using automatic damping
        adaptation. (Default: ``None``)
      use_initial_damping_calibration: Boolean. If ``True``, the initial damping
        value, used to initialize the adaptive damping method, will be first
        calibrated (after any burnin steps to estimate the preconditioner) so
        that its value wouldn't be changed after the first step of optimization.
        This calibration is done by essentially running the step function
        multiple times without actually updating the parameters or sampling a
        new mini-batch. ``num_burnin_steps`` must be greater than 0 to use this
        option. (Default: ``False``)
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
      precon_damping_mult: Scalar. When ``precon_damping_schedule`` is unset,
        the regular damping is used for the preconditioner damping, multiplied
        by this value. (Default: ``1.0``)
      precon_damping_schedule: Similar to ``damping_schedule``, but for the
        preconditioner only. If ``None``, the preconditioner will use the
        regular damping, multiplied by ``precon_damping_mult``.
        (Default: ``None``)
      use_step_rejection: Whether or not to reject the step whenever the loss
        on the current batch goes up after the update. This option offers
        robustness at the cost of doing more work per step (unless adaptive
        damping with Levenberg-Marquardt is used). (Default: ``False``)
      reject_damping_increase_factor: The damping parameter is increased by this
        factor if the step is rejected. (Default: ``1.0``)
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
        matrix. See the documentation for :class:`~BlockDiagonalCurvature` for a
        detailed description of the possible options. If ``None`` will use
        default estimation_mode mode of the used CurvatureEstimator subclass,
        which is typically "ggn_curvature_prop". (Default: ``None``)
      custom_estimator_ctor: Optional constructor for subclass of
        :class:`~BlockDiagonalCurvature`. If specified, the optimizer will use
        this conastructor instead of the default
        :class:`~BlockDiagonalCurvature`. (Default: ``None``)
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
      patterns_to_skip: tuple. A list of any patterns that should be skipped by
        the graph matcher when auto-tagging. (Default: ``()``)
      use_automatic_registration: Bool. If ``True``, the optimizer will try to
        automatically register the layers of your network. (Default: ``True``)
      auto_register_kwargs: Any additional kwargs to be passed down to
        :func:`~auto_register_tags`, which is called by the curvature estimator.
        (Default: ``None``)
      layer_tag_to_block_ctor: dictionary. A mapping from layer tags to block
        classes which to override the default choices of block approximation for
        that specific tag. See the documentation for
        :class:`~CurvatureEstimator` for a more detailed description. (Default:
        ``None``)
      multi_device: Boolean. Whether to use pmap and run the optimizer on
        multiple devices. (Default: ``False``)
      debug: Boolean. If neither the step or init functions should be jitted.
        Note that this also overrides ``multi_device`` and prevents using pmap,
        instead using a "simulated pmap" that loops over the device index and
        does everything on the default device. (Default: ``False``)
      invalid_metric_value: Numeric. Certain metrics returned from the step
        function are not always computed at each iteration, or may otherwise
        be invalid. In such cases we need to return a value anyway. jnp.nan is
        a natural choice, but can sometimes cause problems (e.g. false positives
        JAX's automatic NaN checker). This argument allows the user to specify a
        different value to return in such cases. (Default: ``jnp.nan``)
      batch_size_extractor: A function that takes as input the function
        arguments and returns the batch size for a single device. (Default:
        ``kfac.utils.default_batch_size_extractor``)
      pmap_axis_name: String. The name of the pmap axis to use when
        ``multi_device`` is set to True. (Default: ``kfac_axis``)
      forbid_setting_attributes_after_finalize: Boolean. By default, after the
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
      include_registered_loss_in_stats: Boolean. If True, we include the loss,
        as computed from the registered losses, in the stats. Also included is
        the relative difference between this as the loss computed from
        ``value_and_grad_func``. This is useful for debugging registration
        errors. Note this for this option to work it's required that the targets
        are passed for each loss function registration. (Default: ``False``)
      distributed_precon_apply: Boolean. Whether to distribute the application
        of the preconditioner across the different devices in a layer-wise
        fashion. If False, each device will (redundantly) perform the required
        operations for all the layers. (Default: True)
      distributed_inverses: Boolean. Whether to distribute the inverse
        computations (required to compute the preconditioner) across the
        different devices in a layer-wise fashion. If False, each device will
        (redundantly) perform the required computations for all the layers.
        (Default: True)
      num_estimator_samples: Number of samples (per case) to use when computing
        stochastic curvature matrix estimates. This option is only used when
        ``estimation_mode == 'fisher_gradients'`` or ``estimation_mode ==
        '[fisher,ggn]_curvature_prop'``. (Default: 1)
      should_vmap_estimator_samples: Whether to use ``jax.vmap`` to compute
        samples when ``num_estimator_samples > 1``. (Default: False)
      norm_to_scale_identity_weight_per_block: The name of a norm to use to
        compute extra per-block scaling for the damping. See psd_matrix_norm()
        in utils/math.py for the definition of these. Note that this will not
        affect the exact quadratic model that is used as part of the "adaptive"
        learning rate, momentum, and damping methods. (Default: None)
      precon_power: The matrix power to use when computing the preconditioner.
        K-FAC use -1 by default, but ``kfac_jax`` can simulate other optimizers
        like RMSProp by using -0.5 (along with appropriate changes to
        ``layer_tag_to_block_ctor`` and  ``estimation_mode``). (Default: -1)
    """

    super().__init__(
        multi_device=multi_device,
        pmap_axis_name=pmap_axis_name if multi_device else None,
        debug=debug,
        forbid_setting_attributes_after_finalize=
        forbid_setting_attributes_after_finalize,
        excluded_attribute_names=modifiable_attribute_exceptions,
    )

    if use_adaptive_learning_rate and learning_rate_schedule is not None:
      raise ValueError("If you are using adaptive learning rate then "
                       "`learning_rate_schedule` should be None.")
    if use_adaptive_momentum and momentum_schedule is not None:
      raise ValueError("If you are using adaptive momentum then "
                       "`momentum_schedule` should be None.")
    if use_adaptive_learning_rate:
      if use_adaptive_damping and initial_damping is None:
        raise ValueError(
            "When use_adaptive_damping is True you must provide a "
            "value for initial_damping."
        )
      if use_adaptive_damping and damping_schedule is not None:
        raise ValueError(
            "If you are using adaptive damping then "
            "`damping_schedule` should be None."
        )

    if num_burnin_steps <= 0 and use_initial_damping_calibration:
      raise ValueError("num_burnin_steps must be > 0 if "
                       "use_initial_damping_calibration is True.")

    self._value_and_grad_func = value_and_grad_func
    self._value_func_has_aux = value_func_has_aux
    self._value_func_has_state = value_func_has_state
    self._value_func_has_rng = value_func_has_rng
    self._value_func: ValueFunc = convert_value_and_grad_to_value_func(
        value_and_grad_func,
        has_aux=value_func_has_aux or value_func_has_state,
    )

    self._l2_reg = l2_reg

    self._use_adaptive_learning_rate = use_adaptive_learning_rate
    self._learning_rate_schedule = learning_rate_schedule
    self._use_adaptive_momentum = use_adaptive_momentum
    self._momentum_schedule = momentum_schedule

    self._use_adaptive_damping = use_adaptive_damping
    self._damping_schedule = damping_schedule
    self._initial_damping = initial_damping
    self._use_initial_damping_calibration = use_initial_damping_calibration
    self._min_damping = min_damping
    self._max_damping = max_damping
    self._include_damping_in_quad_change = include_damping_in_quad_change
    self._damping_adaptation_decay = damping_adaptation_decay
    self._damping_adaptation_interval = damping_adaptation_interval
    self._damping_lower_threshold = damping_lower_threshold
    self._damping_upper_threshold = damping_upper_threshold
    self._always_use_exact_qmodel_for_damping_adjustment = (
        always_use_exact_qmodel_for_damping_adjustment)
    self._precon_damping_mult = precon_damping_mult
    self._precon_damping_schedule = precon_damping_schedule

    self._use_step_rejection = use_step_rejection
    self._reject_damping_increase_factor = reject_damping_increase_factor

    self._norm_constraint = norm_constraint
    self._num_burnin_steps = num_burnin_steps
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
    self._layer_tag_to_block_cls = layer_tag_to_block_ctor
    self._patterns_to_skip = patterns_to_skip
    self._batch_process_func = batch_process_func or (lambda x: x)
    self._include_norms_in_stats = include_norms_in_stats
    self._include_per_param_norms_in_stats = include_per_param_norms_in_stats
    self._include_registered_loss_in_stats = include_registered_loss_in_stats
    self._batch_size_extractor = batch_size_extractor

    self._invalid_metric_value = invalid_metric_value

    self._use_cached_inverses = (self._inverse_update_period != 1)
    self._use_exact_inverses = use_exact_inverses

    self._norm_to_scale_identity_weight_per_block = (
        norm_to_scale_identity_weight_per_block
    )

    self._precon_power = precon_power

    self._params_index = 0
    batch_index = int(value_func_has_state + value_func_has_rng + 1)

    if (norm_to_scale_identity_weight_per_block is not None
        and norm_to_scale_identity_weight_per_block != "none"):

      assert (not use_adaptive_learning_rate and not use_adaptive_momentum
              and not use_adaptive_damping)  # not currently supported

    estimator_ctor = (custom_estimator_ctor or BlockDiagonalCurvature)

    auto_register_kwargs = auto_register_kwargs or {}
    auto_register_kwargs.update(dict(
        register_only_generic=register_only_generic,
        patterns_to_skip=patterns_to_skip,
    ))

    # Curvature estimator
    self._estimator = estimator_ctor(
        func=(self._value_func if value_func_for_estimator is None else
              value_func_for_estimator),
        default_estimation_mode=estimation_mode,
        params_index=self._params_index,
        batch_index=batch_index,
        layer_tag_to_block_ctor=layer_tag_to_block_ctor,
        distributed_multiplies=distributed_precon_apply,
        distributed_cache_updates=distributed_inverses,
        num_samples=num_estimator_samples,
        should_vmap_samples=should_vmap_estimator_samples,
        auto_register_tags=use_automatic_registration,
        auto_register_kwargs=auto_register_kwargs,
    )
    self._implicit = curvature_estimator.ImplicitExactCurvature(
        self._value_func,
        params_index=self._params_index,
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
  def l2_reg(self) -> Numeric:
    """The weight of the additional diagonal term added to the curvature."""
    return self._l2_reg

  @property
  def estimator(self) -> BlockDiagonalCurvature:
    """The underlying curvature estimator used by the optimizer."""
    return self._estimator

  @property
  def damping_decay_factor(self) -> Numeric:
    """How fast to decay the damping, when using damping adaptation."""
    return self._damping_adaptation_decay ** self._damping_adaptation_interval

  @property
  def _exact_powers_to_cache(self) -> Numeric | Sequence[Numeric] | None:
    if self._use_exact_inverses and self._use_cached_inverses:
      return self._precon_power
    else:
      return None

  @property
  def _approx_powers_to_cache(self) -> Numeric | Sequence[Numeric] | None:
    if not self._use_exact_inverses and self._use_cached_inverses:
      return self._precon_power
    else:
      return None

  def should_update_damping(self, step_counter: int) -> bool:
    """Whether at the current step the optimizer should update the damping."""
    return ((step_counter + 1) % self._damping_adaptation_interval == 0) and (
        self._use_adaptive_damping
    )

  def should_update_estimate_curvature(self, step_counter: int) -> bool:
    """Whether at the current step the optimizer should update the curvature estimates."""
    return step_counter % self._curvature_update_period == 0

  def should_update_inverse_cache(self, state: State) -> Array | bool:
    """Whether at the current step the optimizer should update the inverse curvature approximation."""
    return self._use_cached_inverses and (
        state.step_counter % self._inverse_update_period == 0)

  def should_sync_estimator(self, state: State) -> Array | bool:
    """Whether at the current step the optimizer should update the inverse curvature approximation."""

    if self._use_cached_inverses:
      return self.should_update_inverse_cache(state)

    return True

  @functools.partial(utils.staged, static_argnums=1)
  def _rng_split(self, rng: PRNGKey, num: int) -> tuple[Array, ...]:
    """Splits the ``rng`` key."""
    return tuple(jax.random.split(rng, num))

  @utils.auto_scope_method
  def compute_loss_value(self, func_args: FuncArgsVariants) -> Array:
    """Computes the value of the loss function being optimized."""
    return self._value_func(*func_args)

  def verify_args_and_get_step_counter(
      self,
      step_counter: Array,
      learning_rate: Array | None = None,
      momentum: Array | None = None,
      damping: Array | None = None,
      global_step_int: int | None = None,
  ) -> int:
    """Verifies that the arguments passed to the step function are correct."""

    # Verify correct arguments invocation
    if self._use_adaptive_learning_rate and learning_rate is not None:
      raise ValueError("When use_adaptive_learning_rate is set to True you "
                       "should not pass a value to the step function.")

    elif not self._use_adaptive_learning_rate and (
        self._learning_rate_schedule is None and learning_rate is None):
      raise ValueError("When `use_adaptive_learning_rate` is set to False and "
                       "`learning_rate_schedule` is None you must provide a "
                       "value to the step function.")

    elif self._learning_rate_schedule is not None and learning_rate is not None:
      raise ValueError("When you have passed a `learning_rate_schedule` you "
                       "should not pass a value to the step function.")

    if self._use_adaptive_momentum and momentum is not None:
      raise ValueError("When `use_adaptive_momentum` is set to True you "
                       "should not pass a value to the step function.")

    elif not self._use_adaptive_momentum and (
        self._momentum_schedule is None and momentum is None):
      raise ValueError("When `use_adaptive_momentum` is set to False and "
                       "`momentum_schedule` is None you must provide a value to"
                       " the step function.")

    elif self._momentum_schedule is not None and momentum is not None:
      raise ValueError("When you have passed a `momentum_schedule` you should "
                       "not pass a value to the step function.")

    if self._use_adaptive_damping and damping is not None:
      raise ValueError("When `use_adaptive_damping` is set to True you "
                       "should not pass a value to the step function.")

    elif not self._use_adaptive_damping and (
        self._damping_schedule is None and damping is None):
      raise ValueError("When `use_adaptive_damping` is set to False and "
                       "`damping_schedule` is None you must provide a value to "
                       "the step function.")

    elif self._damping_schedule is not None and damping is not None:
      raise ValueError("When you have passed a `damping_schedule` you should "
                       "not pass a value to the step function.")

    if global_step_int is None:
      return int(self.get_first(step_counter))

    return global_step_int

  @utils.staged
  def _setup_state_and_schedules(
      self,
      learning_rate: Array | None,
      momentum: Array | None,
      damping: Array | None,
      step_counter: Array,
      data_seen: Array,
  ) -> tuple[Numeric | None, Numeric | None, Numeric, Numeric]:
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

    if self._precon_damping_schedule is not None:
      precon_damping = utils.call_func_with_conditional_kwargs(
          self._precon_damping_schedule, step_counter, data_seen=data_seen)
    else:
      precon_damping = damping * self._precon_damping_mult

    return learning_rate, momentum, damping, precon_damping

  def _setup_func_args_and_rng(
      self,
      params: Params,
      rng: PRNGKey,
      batch: Batch,
      func_state: FuncState | None,
  ) -> tuple[FuncArgsVariants, Array]:
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
        batch=batch,
        has_state=self._value_func_has_state,
        has_rng=self._value_func_has_rng,
    )

    return func_args, rng

  def _update_estimator_curvature(
      self,
      estimator_state: BlockDiagonalCurvature.State,
      func_args: FuncArgsVariants,
      rng: PRNGKey,
      ema_old: Numeric,
      ema_new: Numeric,
      precon_damping: Numeric,
      sync: Array | bool = True
  ) -> BlockDiagonalCurvature.State:
    """Updates the curvature estimator state."""

    state = self.estimator.update_curvature_matrix_estimate(
        state=estimator_state,
        ema_old=ema_old,
        ema_new=ema_new,
        identity_weight=self.l2_reg + precon_damping,
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

  @utils.auto_scope_method
  def _compute_loss_and_grads(
      self,
      func_args: FuncArgsVariants,
      state: State | None = None,
  ) -> tuple[Array, Params, FuncState | None, FuncAux | None]:
    """Computes the model loss value and its gradients."""

    del state

    out, grads = self._value_and_grad_func(*func_args)

    loss, func_state, aux = extract_func_outputs(
        out, self._value_func_has_aux, self._value_func_has_state)

    if self._include_registered_loss_in_stats:
      aux = aux or {}
      aux["loss_registered"] = self.compute_loss_from_registrations(func_args)

    return loss, grads, func_state, aux

  @functools.partial(utils.staged, donate_argnums=0)
  def _maybe_update_inverse_cache(
      self,
      state: State,
      precon_damping: Array
  ) -> State:
    """Updates the estimator state cache if it is the right iteration."""

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    state.estimator_state = lax.cond(
        self.should_update_inverse_cache(state),
        functools.partial(
            self.estimator.update_cache,
            identity_weight=self.l2_reg + precon_damping,
            exact_powers=self._exact_powers_to_cache,
            approx_powers=self._approx_powers_to_cache,
            eigenvalues=False,
            pmap_axis_name=self.pmap_axis_name,
        ),
        lambda state_: state_,
        state.estimator_state,
    )

    return state

  @utils.staged
  def _compute_preconditioned_gradient(
      self,
      state: State,
      grads: Params,
      precon_damping: Array,
  ) -> Params:
    """Computes the preconditioned gradient."""

    return self.estimator.multiply_matpower(
        state=state.estimator_state,
        parameter_structured_vector=grads,
        identity_weight=self.l2_reg + precon_damping,
        power=self._precon_power,
        exact_power=self._use_exact_inverses,
        use_cached=self._use_cached_inverses,
        pmap_axis_name=self.pmap_axis_name,
        norm_to_scale_identity_weight_per_block=self._norm_to_scale_identity_weight_per_block,
    )

  @utils.staged
  def _maybe_apply_norm_constraint(
      self, grads: Params, preconditioned_grads: Params, coefficient: Array
  ) -> tuple[Params, Params | None]:
    """Scales precon grad to have F-weighted norm <= norm_constraint."""

    if self._norm_constraint is None:
      return preconditioned_grads, None

    assert not self._use_adaptive_learning_rate

    sq_norm_grads = utils.inner_product(preconditioned_grads, grads)
    sq_norm_scaled_grads = sq_norm_grads * coefficient ** 2

    max_coefficient = jnp.sqrt(self._norm_constraint / sq_norm_scaled_grads)
    coefficient = jnp.minimum(max_coefficient, 1)

    precon_grad = utils.scalar_mul(preconditioned_grads, coefficient)

    return precon_grad, sq_norm_scaled_grads

  def _compute_quad_change_for_damping_adapt(
      self,
      state: State,
      delta: Params,
      grads: Params,
      damping: Array,
      func_args: FuncArgsVariants,
  ) -> Array:
    """The quadratic model change, when lr and momentum are non-adaptive."""

    assert not (self._use_adaptive_learning_rate or self._use_adaptive_momentum)

    if self._always_use_exact_qmodel_for_damping_adjustment:
      quad_model = self.compute_exact_quad_model_filtered(
          [delta], grads, func_args, state=state)
    else:
      quad_model = self.compute_approx_quad_model(state, [delta], grads)

    w = jnp.ones([])
    return self._solve_quad_model(quad_model, damping, [w])[1]

  def _coefficients_and_quad_change(
      self,
      state: State,
      vectors: Sequence[Params],
      grads: Params,
      learning_rate: Numeric | None,
      momentum: Numeric | None,
      damping: Numeric,
      func_args: FuncArgsVariants,
      should_update_damping: bool,
  ) -> tuple[tuple[Numeric, Numeric], Numeric]:
    """The correct update coefficients and corresponding quadratic change."""

    # Compute the coefficients of the update vectors
    # The learning rate is defined as the negative of the coefficient by which
    # we multiply the gradients, while the momentum is the coefficient by
    # which we multiply the velocities.
    neg_learning_rate = -learning_rate if learning_rate is not None else None
    fixed_coefficients = (neg_learning_rate, momentum)

    if self._use_adaptive_learning_rate or self._use_adaptive_momentum:

      assert fixed_coefficients[0] is None or fixed_coefficients[1] is None

      quad_model = self.compute_exact_quad_model_filtered(
          vectors, grads, func_args, state=state,
          fixed_coefficients=fixed_coefficients)

      return self._solve_quad_model(quad_model, damping, fixed_coefficients)

    else:

      assert all(c is not None for c in fixed_coefficients)
      fixed_coefficients: tuple[Numeric, Numeric]

      if should_update_damping:

        delta = self.weighted_sum_of_objects(vectors, fixed_coefficients)

        quad_change = self._compute_quad_change_for_damping_adapt(
            state, delta, grads, damping, func_args)

      else:
        quad_change = self._invalid_metric_value

      return fixed_coefficients, quad_change

  @utils.staged
  def compute_loss_from_registrations(
      self,
      func_args: FuncArgsVariants
  ) -> Array:

    loss = self.estimator.compute_func_from_registered(
        func_args, self._batch_size_extractor(func_args[-1]))

    if self.l2_reg > 0.0:

      l2_reg_val = self.l2_reg / 2 * utils.squared_norm(
          func_args[self._params_index])

      loss += l2_reg_val

    return loss

  @utils.staged
  def _init(
      self,
      params: Params,
      rng: PRNGKey,
      batch: Batch,
      func_state: FuncState | None = None,
  ) -> State:
    """A staged function to initialize the optimizer state ."""

    # Note that we can reuse the rng in the func_args construction below, as
    # these are just dummy values used to perform the tracing.

    return Optimizer.State(
        velocities=jax.tree_util.tree_map(jnp.zeros_like, params),
        estimator_state=self.estimator.init(
            rng=rng,
            func_args=make_func_args(
                params=params,
                func_state=func_state,
                rng=rng,
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
      func_state: FuncState | None = None,
  ) -> State:
    """Initializes the optimizer and returns the appropriate optimizer state.

    NOTE: please do not jit/pmap or otherwise compile this function with JAX,
    as this can lead to errors. Compilation is handled internally by the
    optimizer.

    NOTE: when ``multi_device`` is ``True``, all of the JAX array arguments to
    this function (including arrays inside of trees), should have an extra
    leading axis the size of the number of local devices.

    Args:
      params: Example models parameters (used for tracing and shape info).
      rng: A Jax PRNG key. Unlike the ``rng`` in the step function, should be
        the same for each host and for each slice in the leading axis (i.e.
        corresponding to devices) when ``multi_device`` is ``True``.
      batch: An example batch of the same size as the one passed to ``step``
        (or returned from the ``data_iterator``). Used for tracing and shape
        info.
      func_state: Example function state (used for tracing and shape info).

    Returns:
      The initialized optimizer state.
    """

    if not self.finalized:
      self.finalize(params, rng, batch, func_state)

    return self._init(params, rng, batch, func_state)

  @functools.partial(utils.staged, donate_argnums=[1, 3, 5])
  def _burnin(
      self,
      params: Params,
      state: State,
      rng: Array,
      batch: Batch,
      func_state: FuncState | None,
      damping: Array | None,
      accumulator: utils.MultiChunkAccumulator,
      sync: Array | bool,
  ) -> tuple[State, utils.MultiChunkAccumulator]:
    """A single burnin step, updating only the curvature estimate."""

    _, _, _, precon_damping = self._setup_state_and_schedules(
        None, None,
        state.damping if self._use_adaptive_damping else damping,
        state.step_counter, state.data_seen)

    # Copy this first since we mutate it later in this function.
    accumulator = accumulator.copy()

    func_args, rng = self._setup_func_args_and_rng(
        params, rng, batch, func_state)

    # Update curvature estimate
    state.estimator_state = self._update_estimator_curvature(
        state.estimator_state,
        func_args,
        rng,
        ema_old=1.0,
        ema_new=1.0,
        precon_damping=precon_damping,
        sync=sync,
    )

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
      state: State,
      rng: PRNGKey,
      data_iterator: Iterator[Batch],
      func_state: FuncState | None = None,
      damping: Array | None = None,
  ) -> tuple[State, FuncState | None]:
    """Runs all burnin steps required."""

    if num_steps > 0:

      rng = self._rng_split(rng, num_steps)

      accumulator = utils.MultiChunkAccumulator.zeros_like(
          func_state, self.multi_device)

      for i, rng_i in enumerate(rng):
        batch = next(data_iterator)

        state, accumulator = self._burnin(
            params, state, rng_i, batch, func_state, damping, accumulator,
            i == num_steps - 1)

      func_state = accumulator.value_and_clear()

    return state, func_state

  @functools.partial(
      utils.staged, donate_argnums=(0, 1, 4), static_argnums=(8, 9))
  @utils.auto_scope_method
  def _step(
      self,
      params: Params,
      state: State,
      rng: Array,
      batch: Batch,
      func_state: FuncState | None,
      learning_rate: Array | None,
      momentum: Array | None,
      damping: Array | None,
      should_update_estimate_curvature: bool,
      should_update_damping: bool,
  )-> ReturnEither:
    """A single full step of the optimizer."""

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    # Setup arguments
    (learning_rate, momentum, damping,
     precon_damping) = self._setup_state_and_schedules(
         learning_rate, momentum,
         state.damping if self._use_adaptive_damping else damping,
         state.step_counter, state.data_seen)

    func_args, rng = self._setup_func_args_and_rng(
        params, rng, batch, func_state)

    # Update curvature estimate
    if should_update_estimate_curvature:
      state.estimator_state = self._update_estimator_curvature(
          state.estimator_state,
          func_args,
          rng,
          ema_old=self._curvature_ema,
          ema_new=1.0,
          precon_damping=precon_damping,
          sync=self.should_sync_estimator(state),
      )

    del rng  # should not be used after this point!

    # Compute loss and gradients
    loss, grads, func_state, aux = self._compute_loss_and_grads(
        func_args, state=state)

    # Sync
    loss, grads = utils.pmean_if_pmap((loss, grads), self.pmap_axis_name)

    # Update the inverse curvature
    state = self._maybe_update_inverse_cache(state, precon_damping)

    # Compute proposed directions
    preconditioned_gradient = self._compute_preconditioned_gradient(
        state, grads, precon_damping
    )

    # constrain the norms
    preconditioned_gradient, scaled_grad_norm_sq = (
        self._maybe_apply_norm_constraint(
            grads, preconditioned_gradient, learning_rate,
        )
    )

    vectors = (preconditioned_gradient, state.velocities)

    # Compute the coefficients for the vectors
    coefficients, quad_model_change = self._coefficients_and_quad_change(
        state=state,
        vectors=vectors,
        grads=grads,
        learning_rate=learning_rate,
        momentum=momentum,
        damping=damping,
        func_args=func_args,
        should_update_damping=should_update_damping,
        )

    # Compute the parameter update (delta)
    delta = self.weighted_sum_of_objects(vectors, coefficients)

    # Update parameters
    new_params = jax.tree_util.tree_map(jnp.add, params, delta)

    if should_update_damping or self._use_step_rejection:

      new_loss = self.compute_loss_value((new_params,) + func_args[1:])
      # Sync
      new_loss = utils.pmean_if_pmap(new_loss, self.pmap_axis_name)

    else:
      new_loss = self._invalid_metric_value

    # Optionally compute the reduction ratio and update the damping
    if should_update_damping:

      state.damping, rho = self._compute_new_damping_and_rho(
          loss, new_loss, quad_model_change, state.damping)

    else:
      # If not adjusting the damping we don't compute these here and just set
      # them to self._invalid_metric_value.
      new_loss, rho = self._invalid_metric_value, self._invalid_metric_value

    if self._use_step_rejection:

      reject_step = jnp.logical_or(jnp.isnan(new_loss), new_loss > loss)

      params, state.velocities, state.damping = lax.cond(
          reject_step,
          lambda: (params, state.velocities,
                   self._reject_damping_increase_factor * state.damping),
          lambda: (new_params, delta, state.damping))

    else:
      # stop the linter from complaining about uninitialized variable
      reject_step = False
      params, state.velocities = new_params, delta

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
        precon_damping=precon_damping,
        rho=rho,
        quad_model_change=quad_model_change,
        scaled_grad_norm_sq=scaled_grad_norm_sq,
    )

    if self._use_step_rejection:
      stats["step_rejected"] = reject_step

    if aux is not None:
      aux = utils.pmean_if_pmap(aux, self.pmap_axis_name)
      stats["aux"] = aux

    if self._include_norms_in_stats:
      stats["param_norm"] = utils.norm(params)
      stats["grad_norm"] = utils.norm(grads)
      stats["precon_grad_norm"] = utils.norm(preconditioned_gradient)
      stats["update_norm"] = utils.norm(delta)

    if self._include_per_param_norms_in_stats:
      stats.update(utils.per_parameter_norm(params, "param_norm"))
      stats.update(utils.per_parameter_norm(grads, "grad_norm"))
      stats.update(
          utils.per_parameter_norm(preconditioned_gradient, "precon_grad_norm")
      )
      stats.update(utils.per_parameter_norm(delta, "update_norm"))

    if self._include_registered_loss_in_stats:
      assert aux is not None
      stats["loss_registered"] = aux.pop("loss_registered")
      stats["loss_registered"] = utils.pmean_if_pmap(stats["loss_registered"],
                                                     self.pmap_axis_name)
      stats["loss_registered_reldiff"] = (
          stats["loss_registered"] - loss) / loss

    if self._value_func_has_state:
      return params, state, func_state, stats

    assert func_state is None

    return params, state, stats

  def step(
      self,
      params: Params,
      state: State,
      rng: PRNGKey,
      data_iterator: Iterator[Batch] | None = None,
      batch: Batch | None = None,
      func_state: FuncState | None = None,
      learning_rate: Array | None = None,
      momentum: Array | None = None,
      damping: Array | None = None,
      global_step_int: int | None = None
  )-> ReturnEither:
    """Performs a single update step using the optimizer.

    NOTE: please do not jit/pmap or otherwise compile this function with JAX,
    as this can lead to errors. Compilation is handled internally by the
    optimizer.

    NOTE: when ``multi_device`` is ``True``, all of the JAX array arguments to
    this function (including arrays inside of trees), should have an extra
    leading axis the size of the number of local devices. Slices of ``batch``
    and ``rng`` should be different for each device, whereas the other arugments
    should be identical for each slice. Passing the arguments any other way will
    result in an exception, or possibly undefined behavior.

    Args:
      params: The current parameters of the model.
      state: The current state of the optimizer.
      rng: A Jax PRNG key. Should be different for each iteration, each host,
        and for each slice in the leading axis (i.e. corresponding to devices)
        when ``multi_device`` is ``True``.
      data_iterator: A data iterator to use (if not passing ``batch``).
      batch: A single batch used to compute the update. Should only pass one
        of ``data_iterator`` or ``batch``.
      func_state: Any function state that gets passed in and returned.
      learning_rate: Learning rate to use if the optimizer was created with
        ``use_adaptive_learning_rate=False`` and
        ``learning_rate_schedule=None``. Should be ``None`` otherwise.
      momentum: Momentum to use if the optimizer was created with
        ``use_adaptive_momentum=False`` and ``momentum_schedule=None``. Should
        be ``None`` otherwise.
      damping: Damping to use if the optimizer was created with
        ``use_adaptive_damping=False`` and ``damping_schedule=None``. Should be
        ``None`` otherwise. See discussion of constructor argument
        ``initial_damping`` for more information about damping.
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

      if self.num_burnin_steps > 0:

        if data_iterator is None:
          raise ValueError("If num_burnin_steps > 0, data_iterator must be "
                           "provided.")

        rng, burnin_rng = self._rng_split(rng, 2)

        state, func_state = self.burnin(
            num_steps=self.num_burnin_steps,
            params=params,
            state=state,
            rng=burnin_rng,
            data_iterator=data_iterator,
            func_state=func_state,
            damping=damping,
        )

    if data_iterator is not None:
      batch = next(data_iterator)

    if (step_counter_int == 0 and self._use_adaptive_damping
        and self._use_initial_damping_calibration):

      assert self.num_burnin_steps > 0

      state = self.calibrate_initial_damping(
          params, state, rng, batch, func_state, learning_rate, momentum)

    should_update_estimate_curvature = self.should_update_estimate_curvature(
        step_counter_int
    )
    should_update_damping = self.should_update_damping(step_counter_int)

    return self._step(
        params, state, rng, batch, func_state, learning_rate, momentum, damping,
        should_update_estimate_curvature, should_update_damping)

  def calibrate_initial_damping(
      self,
      params: Params,
      state: State,
      rng: PRNGKey,
      batch: Batch,
      func_state: FuncState | None = None,
      learning_rate: Array | None = None,
      momentum: Array | None = None,
  ) -> State:
    """Calibrates the initial damping parameter."""

    # Instead of writing a custom compiled function to compute rho and update
    # the damping, we're going to be lazy and just call the step function
    # repeatedly, throwing out the new optimizer state, params, and stats, while
    # keeping the rng and batch the same at each call. This is a bit hacky and
    # somewhat wasteful, both in terms of a few extra (minor) computations done
    # in step() that are pointless, as well as the extra memory required to
    # store temporary copies of the optimizer state and model params.

    # TODO(jamesmartens): Improve the implementation if this feature is commonly
    # used?

    while True:

      prev_damping = float(self.get_first(state.damping))

      # Note that we need to copy params and func_state since _step() will
      # donate them. A bette option might be to recompile _step() to not donate
      # these arguments.
      ret = self._step(
          self.copy_obj(params), self.copy_obj(state), rng, batch,
          self.copy_obj(func_state), learning_rate, momentum, None, False, True)

      new_state = ret[1]

      new_damping = float(self.get_first(new_state.damping))
      state.damping = new_state.damping

      del new_state

      if prev_damping == new_damping:
        return state

  @utils.auto_scope_method
  def compute_exact_quad_model_filtered(
      self,
      vectors: Sequence[Params],
      grads: Params,
      func_args: FuncArgsVariants,
      state: State | None = None,
      fixed_coefficients: Sequence[Numeric | None] | None = None,
  ) -> QuadModelParams:
    """Computes the components of the exact quadratic model."""

    # We check the fixed_coefficients for zeros to save computing the expensive
    # matrix vector products for vectors that will eventually be multiplied by
    # zero. If fixed_coefficients is None, we assume that all coefficients are
    # free and compute the full model.

    if fixed_coefficients is None:  # can we get rid of this?
      return self.compute_exact_quad_model(
          vectors, grads, func_args, state=state)

    assert len(vectors) == len(fixed_coefficients)
    assert len(vectors) == 2  # only deal with the two vector case

    def if_momentum_coeff_zero():

      # Only pass in the vectors that won't be multiplied by zero
      quad_model = self.compute_exact_quad_model(
          vectors[:1], grads, func_args, state=state)

      # Repad the quad model with zeroes for the removed entries
      return tuple(
          jnp.pad(arr, [(0, 1)] * arr.ndim, constant_values=0.0)
          for arr in quad_model
      )

    # Add a check here to save compiling both branches in the static case
    if (isinstance(fixed_coefficients[1], float)
        and fixed_coefficients[1] == 0.0):

      return if_momentum_coeff_zero()

    # Due to how XLA cannot share computations across cond boundaries, such as
    # network forward and backwards passes, we cannot use a cond here and remain
    # effecicient. If this behavior ever changes we can uncomment the block
    # below.

    # return jax.lax.cond(
    #     fixed_coefficients[1] == 0.0,
    #     if_momentum_coeff_zero,
    #     lambda: self.compute_exact_quad_model(
    #         vectors, grads, func_args, state=state),
    # )

    return self.compute_exact_quad_model(vectors, grads, func_args, state=state)

  @utils.auto_scope_method
  def compute_exact_quad_model(
      self,
      vectors: Sequence[Params],
      grads: Params,
      func_args: FuncArgsVariants,
      state: State | None = None,
  ) -> QuadModelParams:
    """Computes the components of the exact quadratic model."""

    del state

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
            utils.matrix_of_inner_products(vectors),
            utils.vector_of_inner_products(grads, vectors))

  @functools.partial(utils.staged, donate_argnums=2)
  @utils.auto_scope_method
  def compute_approx_quad_model(
      self,
      state: State,
      vectors: Sequence[Params],
      grads: Params,
  ) -> QuadModelParams:
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
          norm_to_scale_identity_weight_per_block=self._norm_to_scale_identity_weight_per_block,
      )

    c_vectors = [c_times_v(v_i) for v_i in vectors]

    return (utils.symmetric_matrix_inner_products(c_vectors, vectors),
            utils.matrix_of_inner_products(vectors),
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
      quad_model_parameters: QuadModelParams,
      damping: Array,
      fixed_coefficients: Sequence[Numeric | None],
  ) -> tuple[tuple[Numeric, ...], Array]:
    """Solves for the optimal learning rate and momentum of the quadratic model.

    The quadratic model is represented as:
      Q(w) = w^T V^T (C + damping * I + l2_reg * I) V w / 2.0 + w^T V^T g
    where (n - number of vectors, d - dimensions of each vector):
      w (n,) - the vector of free weights (learning rate and momentum)
      V (d, n) - the matrix of proposed vectors for each weight
      C (d, d) - the true curvature matrix (GGN/Fisher/Hessian)
      g (d,) - the true gradient
      damping - the damping value at the current iteration

    In the implementation we have:
      A = V^T C V
      D = V^T V  (for damping)
      R = V^T V  (for L2 regularization)
      b = V^T g

    Args:
      quad_model_parameters: The computed matrices A, D, R, and vector b.
      damping: The damping to use for evaluating the quadratic model.
      fixed_coefficients: A list over the vectors of the fixed numerical values
        to use for their coefficients. For each of these that is None, the
        quadratic model is minimized to compute the 'optimal' coefficient value.
    Returns:
     A list of coefficients which are the solution (and include any values that
     are not None from fixed_weights) and the value of the quadratic model
     function for this solution (as a scalar).
    Raises:
      The function currently supports only up to two vectors, hence if you
      provide more, it will raise a ``NotImplementedError``.
    """

    # TODO(jamesmartens): should revise the above docstring and move most of it
    # to compute_exact_quad_model.

    # pylint: disable=invalid-name
    A_no_diag, D, R, b = quad_model_parameters
    A = A_no_diag + self.l2_reg * R
    A_damped = A + damping * D

    # Sync.
    # TODO(jamesmartens): we should perform this earlier since it's
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
        w = - utils.psd_solve_maybe_zero_last_idx(A_damped, b)

      else:
        raise NotImplementedError()

    elif all(c is not None for c in fixed_coefficients):
      # No coefficients adapted

      w = jnp.asarray(fixed_coefficients)

    elif len(fixed_coefficients) == 2:
      # Exactly one adapted coefficient

      w = [None, None]
      index = fixed_coefficients.index(None)
      w[1 - index] = fixed_coefficients[1 - index]

      b_extra = A_damped[1 - index, index] * w[1 - index]
      # pylint: enable=invalid-name

      w[index] = -(b[index] + b_extra) / A_damped[index, index]

    else:
      raise NotImplementedError()

    w = tuple(w)
    w: tuple[Numeric, ...]

    quad_model_change = self.compute_quadratic_model_value(
        A, A_damped, b, jnp.array(w))

    return w, quad_model_change

  @utils.staged
  def _compute_new_damping_and_rho(
      self,
      old_loss: Array,
      new_loss: Array,
      quad_change: Array,
      current_damping: Array,
  ) -> tuple[Array, Array]:
    """Computes the reduction ratio and the updated value of the damping."""

    # Reduction ratio
    rho = (new_loss - old_loss) / quad_change
    rho_not_nan = jnp.nan_to_num(rho, nan=-100.0)

    # Update damping
    should_increase = rho_not_nan < self._damping_lower_threshold
    increased_damping = current_damping / self.damping_decay_factor
    should_decrease = rho_not_nan > self._damping_upper_threshold
    decreased_damping = current_damping * self.damping_decay_factor

    damping = jnp.select([should_decrease, should_increase],
                         [decreased_damping, increased_damping],
                         default=current_damping)

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

  def value_func(*args, **kwargs) -> Array:
    out, _ = value_and_grad_func(*args, **kwargs)
    return out[0] if has_aux else out

  return value_func


def make_func_args(
    params: Params,
    func_state: FuncState | None,
    rng: PRNGKey | None,
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

  if not has_state and not has_rng:
    return params, batch

  elif not has_rng:
    return params, func_state, batch

  elif not has_state:
    return params, rng, batch

  else:
    return params, func_state, rng, batch


def extract_func_outputs(
    raw_outputs: FuncOutputs,
    has_aux: bool,
    has_state: bool,
) -> tuple[Array, FuncState | None, FuncAux | None]:
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
    assert isinstance(raw_outputs, Array)
    return raw_outputs, None, None

  loss, other = raw_outputs

  if has_aux and has_state:
    func_state, aux = other
  elif has_aux:
    func_state, aux = None, other
  else:
    func_state, aux = other, None

  return loss, func_state, aux
