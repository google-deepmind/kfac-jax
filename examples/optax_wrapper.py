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
from typing import Callable, Iterator, Mapping, NamedTuple

import jax
import kfac_jax
import optax


Array = kfac_jax.utils.Array
PRNGKey = kfac_jax.utils.PRNGKey
Params = kfac_jax.utils.Params
Batch = kfac_jax.utils.Batch
FuncState = kfac_jax.utils.FuncState
OptaxState = kfac_jax.utils.ArrayTree
ScheduleType = kfac_jax.optimizer.ScheduleType
OptaxCtor = Callable[[ScheduleType], optax.GradientTransformation]

PreconditionState = kfac_jax.OptaxPreconditionState
Preconditioner = kfac_jax.OptaxPreconditioner


class OptaxAndPreconditionState(NamedTuple):
  optax_state: OptaxState
  precond_state: PreconditionState | None = None


class OptaxWrapper:
  """Wrapper for Optax optimizers to have the same interface as kfac_jax's optimizer."""

  def __init__(
      self,
      value_and_grad_func: kfac_jax.optimizer.ValueAndGradFunc,
      value_func_has_aux: bool,
      value_func_has_state: bool,
      value_func_has_rng: bool,
      learning_rate: ScheduleType,
      optax_optimizer_ctor: OptaxCtor,
      batch_process_func: Callable[[Batch], Batch] = lambda x: x,
      preconditioner: Preconditioner | None = None,
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
          func_state: FuncState | None = None,
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
      func_state: FuncState | None = None,
  ) -> OptaxAndPreconditionState:
    """Initializes the optimizer and returns the appropriate optimizer state."""
    return self._pmap_init(params, rng, batch, func_state)

  def _step(
      self,
      params: Params,
      state: OptaxAndPreconditionState,
      rng: PRNGKey,
      batch: Batch,
      func_state: FuncState | None = None,
      global_step_int: int | None = None,
  ) -> (
      tuple[Params, OptaxAndPreconditionState, FuncState, Mapping[str, Array]] |
      tuple[Params, OptaxAndPreconditionState, Mapping[str, Array]]
  ):
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

    loss, stats, grads = kfac_jax.utils.pmean_if_pmap(
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
      func_state: FuncState | None = None,
      global_step_int: int | None = None,
  ) -> (
      tuple[Params, OptaxAndPreconditionState, FuncState, Mapping[str, Array]] |
      tuple[Params, OptaxAndPreconditionState, Mapping[str, Array]]
  ):
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
