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
from typing import Callable, Sequence

import jax.numpy as jnp
import kfac_jax
import optax


Array = kfac_jax.utils.Array
Numeric = kfac_jax.utils.Numeric


def linear_interpolation(
    x: Numeric,
    interpolation_points: tuple[tuple[float, float], ...]
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
    train_total_batch_size: int | None,
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
) -> Numeric:
  """Fixed/constant schedule."""
  del global_step
  return value


def kfac_resnet50_schedule(
    global_step: Numeric,
) -> Array:
  """Custom schedule for KFAC."""

  return jnp.power(10.0, linear_interpolation(
      x=global_step,
      interpolation_points=(
          (0, -6), (50, -3.1), (5000, -3.1), (11000, -3.23),
          (20000, -5.0), (200000, -5.7), (1000001, -6))
  ))


# TODO(jamesmartens,timothycnguyen,joeljennings): Some possible future
# improvements to the schedules code:
# - Put the logic to calculate "warmup_data" (or "warmup_steps") and
#   "total_data" (or "total_steps") in a place so that we can apply warmup to
#   an arbitrary schedule.
# - Use existing `optax.schedule` operations (e.g. `exponential_decay`,
#   `piecewise_constant_schedule`) as much as possible to make the kfac_jax
#   codebase simple and compact.
# - Optax's `warmup_cosine_decay_schedule` and
#   `warmup_exponential_decay_schedule` are implemented by simply combining
#   `polynomial_schedule` and the corresponding schedule. So we can prepare a
#   general warmup scheduler factory that returns a combination of `polynomial_
#   schedule` and the given base scheduler based on the arguments e.g. warmup_
#   steps.
# - Abstract out the logic to compute data_seen and global_step from the
#   arguments to the schedule functions.


# TODO(jamesmartens,kazukiosawa,botev): change these argument names to be not be
# specific to learning rates.
def cosine_schedule(
    global_step: Numeric,
    dataset_size: int,
    train_total_batch_size: int | None,
    total_steps: int | None,
    total_epochs: float | None,
    peak_learning_rate: float,
    initial_learning_rate: float = 1e-7,
    end_learning_rate: float = 0.0,
    warmup_epochs: float | None = None,
    warmup_steps: int | None = None,
    warmup_fraction: float | None = None,
    data_seen: Numeric | None = None,
) -> Numeric:
  """A cosine schedule, similar to Optax."""

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
      warmup_data = warmup_fraction * total_epochs * dataset_size

    else:
      warmup_data = warmup_steps * train_total_batch_size

    if total_epochs is not None:
      total_data = total_epochs * dataset_size

    else:
      total_data = total_steps * train_total_batch_size

    val = optax.warmup_cosine_decay_schedule(
        init_value=initial_learning_rate,
        peak_value=peak_learning_rate,
        end_value=end_learning_rate,
        warmup_steps=warmup_data,
        decay_steps=total_data,
    )(data_seen)

  else:

    if warmup_fraction is not None:
      warmup_steps = warmup_fraction * total_steps

    val = optax.warmup_cosine_decay_schedule(
        init_value=initial_learning_rate,
        peak_value=peak_learning_rate,
        end_value=end_learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
    )(global_step)

  assert isinstance(val, Numeric)
  return val


# TODO(jamesmartens,kazukiosawa,botev): change these argument names to be not be
# specific to learning rates. Also, initial_learning_rate is misnamed since this
# is value is never actually used, but is just a "base" multiplying for the
# decay factors.
def stepwise_schedule(
    global_step: Numeric,
    dataset_size: int,
    train_total_batch_size: int | None,
    lr_decay_factors: Sequence[float],
    initial_learning_rate: float,
    epoch_boundaries: Sequence[float] | None = None,
    warmup_epochs: float | None = None,
    step_boundaries: Sequence[float] | None = None,
    warmup_steps: int | None = None,
    data_seen: Numeric | None = None,
) -> Numeric:
  """A basic stepwise schedule."""

  if (epoch_boundaries is None) == (step_boundaries is None):
    raise ValueError("Exactly one of 'epoch_boundaries' and 'step_boundaries' "
                     "can must be passed.")

  if (warmup_epochs is None) == (warmup_steps is None):
    raise ValueError("Exactly one of 'warmup_epochs' and 'warmup_steps' must "
                     "be set.")

  num_boundaries = len(epoch_boundaries or step_boundaries)
  if len(lr_decay_factors) != num_boundaries:
    raise ValueError(f"len(lr_decay_factors)={len(lr_decay_factors)} must be "
                     f"equal to the number of boundaries={num_boundaries}.")

  values = jnp.concatenate(
      [jnp.array([1.0]), jnp.array(lr_decay_factors)]) * initial_learning_rate

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

    if warmup_data > 0.0:
      return value * jnp.minimum(1., data_seen / warmup_data)
    else:
      return value

  else:

    step_boundaries = jnp.array(step_boundaries)

    index = jnp.sum(step_boundaries <= global_step)
    value = jnp.take(values, index)

    if warmup_steps > 0.0:
      return value * jnp.minimum(1., global_step / warmup_steps)
    else:
      return value


def exponential_decay_schedule(
    global_step: int,
    dataset_size: int,
    train_total_batch_size: int | None,
    total_steps: int | None,
    total_epochs: float | None,
    init_value: float,
    end_value: float,
    start_epochs: float | None = None,
    start_steps: int | None = None,
    start_fraction: float | None = None,
    data_seen: Numeric | None = None,
) -> Numeric:
  """Exponential decay schedule, similar to Optax."""

  if (total_steps is None) == (total_epochs is None):
    raise ValueError("Exactly one of 'total_steps' and 'total_epochs' must be "
                     "set.")

  n = sum(x is not None for x in [start_epochs, start_steps, start_fraction])

  if n != 1:
    raise ValueError(f"Exactly one of start_steps={start_steps}, "
                     f"start_epochs={start_epochs} and start_fraction="
                     f"{start_fraction} must be set.")

  if start_epochs is not None or total_epochs is not None:

    if data_seen is None:

      if train_total_batch_size is not None:
        data_seen = global_step * train_total_batch_size

      else:
        raise ValueError("One of 'train_total_batch_size' or 'data_seen' must "
                         "passed when 'total_epochs' or 'start_epochs' are "
                         "passed.")

    if ((start_epochs is None or total_epochs is None)
        and train_total_batch_size is None):

      raise ValueError("'train_total_batch_size' must be passed if only one of "
                       "'total_epochs' or 'start_epochs' are passed.")

    if start_epochs is not None:
      start_data = start_epochs * dataset_size

    elif start_fraction is not None:
      start_data = start_fraction * total_epochs * dataset_size

    else:
      start_data = start_steps * train_total_batch_size

    if total_epochs is not None:
      total_data = total_epochs * dataset_size

    else:
      total_data = total_steps * train_total_batch_size

    val = optax.exponential_decay(
        init_value=init_value,
        end_value=end_value,
        decay_rate=end_value / init_value,
        transition_begin=start_data,
        transition_steps=total_data - start_data,
    )(data_seen)

  else:

    if start_fraction is not None:
      start_steps = start_fraction * total_steps

    val = optax.exponential_decay(
        init_value=init_value,
        end_value=end_value,
        decay_rate=end_value / init_value,
        transition_begin=start_steps,
        transition_steps=total_steps - start_steps,
    )(global_step)

  assert isinstance(val, Numeric)
  return val


def _custom_polynomial_schedule(
    init_value: Numeric,
    end_value: Numeric,
    power: Numeric,
    transition_steps: int,
    transition_begin: int = 0
) -> Callable[[Numeric], Numeric]:
  """A polynomial schedule similar to Optax that works even when init_value < end_value."""

  # See the Optax docstring for polynomial_schedule for more information about
  # what this computation is doing.

  def schedule(count):

    count = jnp.clip(count - transition_begin, 0, transition_steps)

    if init_value >= end_value:
      frac = 1.0 - count / transition_steps
      return (init_value - end_value) * (frac**power) + end_value
    else:
      frac = count / transition_steps
      return (end_value - init_value) * (frac**power) + init_value

  return schedule


def polynomial_schedule(
    global_step: int,
    dataset_size: int,
    train_total_batch_size: int | None,
    total_steps: int | None,
    total_epochs: float | None,
    init_value: float,
    end_value: float,
    power: Numeric = 1,
    start_epochs: float | None = None,
    start_steps: int | None = None,
    start_fraction: float | None = None,
    data_seen: Numeric | None = None,
):
  """Polynomial schedule (defaults to linear), similar to Optax."""

  if (total_steps is None) == (total_epochs is None):
    raise ValueError("Exactly one of 'total_steps' and 'total_epochs' must be "
                     "set.")

  n = sum(x is not None for x in [start_epochs, start_steps, start_fraction])

  if n != 1:
    raise ValueError(f"Exactly one of start_steps={start_steps}, "
                     f"start_epochs={start_epochs} and start_fraction="
                     f"{start_fraction} must be set.")

  if start_epochs is not None or total_epochs is not None:

    if data_seen is None:

      if train_total_batch_size is not None:
        data_seen = global_step * train_total_batch_size

      else:
        raise ValueError("One of 'train_total_batch_size' or 'data_seen' must "
                         "passed when 'total_epochs' or 'start_epochs' are "
                         "passed.")

    if ((start_epochs is None or total_epochs is None)
        and train_total_batch_size is None):

      raise ValueError("'train_total_batch_size' must be passed if only one of "
                       "'total_epochs' or 'start_epochs' are passed.")

    if start_epochs is not None:
      start_data = start_epochs * dataset_size

    elif start_fraction is not None:
      start_data = start_fraction * total_epochs * dataset_size

    else:
      start_data = start_steps * train_total_batch_size

    if total_epochs is not None:
      total_data = total_epochs * dataset_size

    else:
      total_data = total_steps * train_total_batch_size

    val = _custom_polynomial_schedule(
        init_value=init_value,
        end_value=end_value,
        power=power,
        transition_begin=start_data,
        transition_steps=total_data - start_data,
    )(data_seen)

  else:

    if start_fraction is not None:
      start_steps = start_fraction * total_steps

    val = _custom_polynomial_schedule(
        init_value=init_value,
        end_value=end_value,
        power=power,
        transition_begin=start_steps,
        transition_steps=total_steps - start_steps,
    )(global_step)

  assert isinstance(val, Numeric)
  return val


def construct_schedule(
    name: str,
    dataset_size: int,
    train_total_batch_size: int,
    total_steps: int | None,
    total_epochs: float | None,
    **kwargs,
) -> Callable[[Numeric], Numeric]:
  """Constructs the actual schedule from its name and extra kwargs."""

  name_to_ctor = {
      "fixed": fixed_schedule,
      "imagenet_sgd": imagenet_sgd_schedule,
      "kfac_resnet50": kfac_resnet50_schedule,
      "cosine": cosine_schedule,
      "stepwise": stepwise_schedule,
      "exponential_decay": exponential_decay_schedule,
      "polynomial": polynomial_schedule,
  }

  if name not in name_to_ctor:
    raise NotImplementedError(name)

  return lambda *a, **kw: kfac_jax.utils.call_func_with_conditional_kwargs(
      functools.partial(name_to_ctor[name], *a, **(kw | kwargs)),
      dataset_size=dataset_size,
      train_total_batch_size=train_total_batch_size,
      total_steps=total_steps,
      total_epochs=total_epochs,
      )
