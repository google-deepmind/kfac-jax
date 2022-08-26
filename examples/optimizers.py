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
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Tuple, Type, Union

from absl import logging
import chex
import jax
import jax.numpy as jnp
import kfac_jax
from ml_collections import config_dict
import optax


OptaxState = Any


class OptaxWrapper:
  """Wrapper class for Optax optimizers to have the same interface as KFAC."""

  def __init__(
      self,
      value_and_grad_func: kfac_jax.optimizer.ValueAndGradFunc,
      value_func_has_aux: bool,
      value_func_has_state: bool,
      value_func_has_rng: bool,
      optax_optimizer: optax.GradientTransformation,
      batch_process_func: Optional[Callable[[Any], Any]] = lambda x: x,
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
    """
    self._value_and_grad_func = value_and_grad_func
    self._value_func_has_aux = value_func_has_aux
    self._value_func_has_state = value_func_has_state
    self._value_func_has_rng = value_func_has_rng
    self._optax_optimizer = optax_optimizer
    self._batch_process_func = batch_process_func or (lambda x: x)
    self.pmap_axis_name = "optax_axis"
    self._jit_step = jax.pmap(
        self._step,
        axis_name=self.pmap_axis_name,
        donate_argnums=list(range(5))
    )
    self._jit_init = jax.pmap(
        lambda p, *_: self._optax_optimizer.init(p),
        axis_name=self.pmap_axis_name,
    )

  def init(
      self,
      params: kfac_jax.utils.Params,
      rng: jnp.ndarray,
      batch: kfac_jax.utils.Batch,
      func_state: Optional[kfac_jax.utils.FuncState] = None
  ) -> OptaxState:
    """Initializes the optimizer and returns the appropriate optimizer state."""
    return self._jit_init(params, rng, batch, func_state)

  def _step(
      self,
      params: kfac_jax.utils.Params,
      state: OptaxState,
      rng: chex.PRNGKey,
      batch: kfac_jax.utils.Batch,
      func_state: Optional[kfac_jax.utils.FuncState] = None,
  ) -> kfac_jax.optimizer.ReturnEither:
    """A single step of optax."""
    batch = self._batch_process_func(batch)
    func_args = kfac_jax.optimizer.make_func_args(
        params, func_state, rng, batch,
        has_state=self._value_func_has_state,
        has_rng=self._value_func_has_rng
    )
    out, grads = self._value_and_grad_func(*func_args)
    loss, new_func_state, stats = kfac_jax.optimizer.extract_func_outputs(
        out,
        has_aux=self._value_func_has_aux,
        has_state=self._value_func_has_state,
    )
    stats["loss"] = loss
    stats, grads = jax.lax.pmean((stats, grads), axis_name="optax_axis")

    # Compute and apply updates via our optimizer.
    updates, new_state = self._optax_optimizer.update(grads, state, params)
    new_params = optax.apply_updates(params, updates)

    # Add batch size
    batch_size = jax.tree_util.tree_leaves(batch)[0].shape[0]
    stats["batch_size"] = batch_size * jax.device_count()

    if self._value_func_has_state:
      return new_params, new_state, new_func_state, stats
    else:
      return new_params, new_state, stats

  def step(
      self,
      params: kfac_jax.utils.Params,
      state: OptaxState,
      rng: jnp.ndarray,
      data_iterator: Iterator[kfac_jax.utils.Batch],
      func_state: Optional[kfac_jax.utils.FuncState] = None,
      global_step_int: Optional[int] = None
  ) -> Union[Tuple[kfac_jax.utils.Params, Any, kfac_jax.utils.FuncState,
                   Mapping[str, jnp.ndarray]],
             Tuple[kfac_jax.utils.Params, Any,
                   Mapping[str, jnp.ndarray]]]:
    """A step with similar interface to KFAC."""
    result = self._jit_step(
        params=params,
        state=state,
        rng=rng,
        batch=next(data_iterator),
        func_state=func_state,
    )
    step = jnp.asarray(global_step_int + 1)
    step = kfac_jax.utils.replicate_all_local_devices(step)
    result[-1]["step"] = step
    result[-1]["data_seen"] = step * result[-1]["batch_size"]

    return result


def tf1_rmsprop(
    learning_rate_fn: Callable[[chex.Numeric], chex.Numeric],
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
    x: chex.Numeric,
    interpolation_points: Tuple[Tuple[float, float], ...]
) -> chex.Array:
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
    global_step: chex.Numeric,
    dataset_size: int,
    train_total_batch_size: int,
    **_: Any,
) -> chex.Array:
  """Standard linear scaling schedule for ImageNet."""
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
    global_step: chex.Numeric,
    value: chex.Numeric,
    **_: Any,
) -> chex.Array:
  """Fixed/constant schedule."""
  return jnp.ones_like(global_step) * value


def kfac_resnet50_schedule(
    global_step: chex.Numeric,
    **_: Any,
) -> chex.Array:
  """Custom schedule for KFAC."""
  return jnp.power(10.0, linear_interpolation(
      x=global_step,
      interpolation_points=(
          (0, -6), (50, -3.1), (5000, -3.1), (11000, -3.23),
          (20000, -5.0), (200000, -5.7), (1000001, -6))
  ))


def cosine_schedule(
    global_step: chex.Numeric,
    dataset_size: int,
    train_total_batch_size: int,
    epochs: Optional[int],
    steps: Optional[int],
    initial_learning_rate: float,
    warmup_epochs: int,
    **_: Any,
) -> chex.Array:
  """A cosine schedule described in the TAT paper."""
  if (steps is None) == (epochs is None):
    raise ValueError("Only one of `steps` and `epochs` can be set.")

  warmup_steps = warmup_epochs * dataset_size / train_total_batch_size

  if epochs is not None:
    total_steps = epochs * dataset_size / train_total_batch_size
  else:
    total_steps = steps

  scaled_step = (jnp.maximum(global_step - warmup_steps, 0) /
                 (total_steps - warmup_steps))

  warmup_factor = jnp.minimum(1., global_step / warmup_steps)
  factor = (1.0 + jnp.cos(jnp.pi * scaled_step)) / 2

  return initial_learning_rate * warmup_factor * factor


def stepwise_schedule(
    global_step: chex.Numeric,
    dataset_size: int,
    train_total_batch_size: int,
    lr_decay_factors: Sequence[float],
    initial_learning_rate: float,
    epoch_boundaries: Optional[Sequence[float]] = None,
    warmup_epochs: Optional[int] = None,
    step_boundaries: Optional[Sequence[float]] = None,
    warmup_steps: Optional[int] = None,
    **_: Any,
) -> chex.Array:
  """A basic stepwise schedule."""

  if (epoch_boundaries is None) == (step_boundaries is None):
    raise ValueError("Only one of `epoch_boundaries` and `step_boundaries` can "
                     "be set.")

  if (warmup_epochs is None) == (warmup_steps is None):
    raise ValueError("Only one of `warmup_epochs` and `warmup_steps` can be "
                     "set.")

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
) -> Callable[[chex.Numeric], chex.Array]:
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


def kfac_bn_registration_kwargs(bn_registration: str) -> Mapping[str, Union[
    Tuple[str, ...],
    Mapping[str, Type[kfac_jax.CurvatureBlock]]]]:
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
    l2_reg: chex.Numeric,
    has_aux: bool,
    has_func_state: bool,
    has_rng: bool,
    dataset_size: int,
    train_total_batch_size: int,
    steps: Optional[int],
    epochs: Optional[int],
) -> Union[OptaxWrapper, kfac_jax.Optimizer]:
  """Creates an optimizer from the provided configuration."""
  value_and_grad_func = jax.value_and_grad(train_model_func, has_aux=has_aux)

  kwargs = dict(**config[name])
  logging.info("Using %s kfac_jax.", name)
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
  elif name == "sgd":
    lr_schedule = construct_schedule(
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
        optax_optimizer=optax.chain(
            optax.trace(**kwargs),
            optax.scale_by_schedule(lr_schedule),
            optax.scale(-1))
    )
  else:
    raise NotImplementedError()
