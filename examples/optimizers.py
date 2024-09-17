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
from typing import Callable, Mapping, Type

from absl import logging
import jax
import jax.numpy as jnp
import kfac_jax
from examples import optax_wrapper
from examples import schedules
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


def kfac_bn_registration_kwargs(bn_registration: str) -> Mapping[
    str, tuple[str, ...] | Mapping[str, Type[kfac_jax.CurvatureBlock]]
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
    model_func_for_estimator: kfac_jax.optimizer.ValueFunc | None,
    dataset_size: int,
    train_total_batch_size: int,
    total_steps: int | None,
    total_epochs: float | None,
    schedule_free_config: config_dict.ConfigDict,
) -> optax_wrapper.OptaxWrapper | kfac_jax.Optimizer:
  """Creates an optimizer from the provided configuration."""

  is_optax = "kfac" not in name and hasattr(optax, name)

  if not is_optax and schedule_free_config.enabled:
    raise ValueError(
        "Schedule Free is only supported for optax optimizers."
    )

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

          kwargs[sched_name] = schedules.construct_schedule(
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
        value_func_for_estimator=model_func_for_estimator,
        multi_device=True,
        **kwargs,
    )

  elif is_optax:

    learning_rate_schedule = schedules.construct_schedule(
        dataset_size=dataset_size,
        train_total_batch_size=train_total_batch_size,
        total_steps=total_steps,
        total_epochs=total_epochs,
        **kwargs.pop("learning_rate_schedule")
    )

    if schedule_free_config.enabled:
      optax_ctor = lambda lr: optax.contrib.schedule_free(
          base_optimizer=getattr(optax, name)(learning_rate=lr, **kwargs),
          learning_rate=lr,
          **schedule_free_config.kwargs
      )
    else:
      optax_ctor = lambda lr: (getattr(optax, name)(learning_rate=lr, **kwargs))

    return optax_wrapper.OptaxWrapper(
        value_and_grad_func=value_and_grad_func,
        value_func_has_aux=has_aux,
        value_func_has_rng=has_rng,
        value_func_has_state=has_func_state,
        learning_rate=learning_rate_schedule,
        optax_optimizer_ctor=optax_ctor,
    )

  else:
    raise NotImplementedError()
