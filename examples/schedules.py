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
"""Utilities for setting up different optimizers with unified warmup support."""

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import kfac_jax
import numpy as np
import optax

Array = kfac_jax.utils.Array
Numeric = kfac_jax.utils.Numeric
PyTree = kfac_jax.utils.PyTree

# Note that there is no specific interpretation of the argument to these
# schedules, unlike kfac_jax.utils.ScheduleType (returned by
# construct_schedule).
GenericSchedule = Callable[[Numeric], Numeric]


def _linear_interpolate(start: float, end: float, pct: float):
  return (end - start) * pct + start


def _cosine_interpolate(start: float, end: float, pct: float):
  return end + (start - end) / 2.0 * (jnp.cos(jnp.pi * pct) + 1)


def piecewise_interpolated_schedule(
    count: Numeric,
    vals: Sequence[Numeric],
    boundaries: Sequence[Numeric],
    interpolate_type: str = "linear",
) -> Numeric:
  """Piecewise interpolated schedule.

  Computes a schedule that interpolates between the given values at the given
  boundaries, with vals[0] being the value at count=0, vals[1] being the value
  at count=boundaries[0], etc. For counts past the last boundary, vals[-1] is
  returned.

  Args:
    count: The current count.
    vals: The values to interpolate between.
    boundaries: The boundaries between the intervals.
    interpolate_type: The type of interpolation to use. Must be either "linear"
      or "cosine". (See optax.piecewise_interpolate_schedule() for details.)

  Returns:
    The value of the schedule at the current count.
  """

  # This is essentially a reimplementation of Optax's
  # piecewise_interpolate_schedule() since that function has a really weird
  # signature that takes the values to be a cumulative product of positive
  # scales.

  if interpolate_type == "linear":
    interpolate_fn = _linear_interpolate
  elif interpolate_type == "cosine":
    interpolate_fn = _cosine_interpolate
  else:
    raise ValueError("`interpolate_type` must be either 'cos' or 'linear'")

  if len(vals) != len(boundaries) + 1:
    raise ValueError("`vals` must have one more element than `boundaries`.")

  bounds = np.stack((0,) + tuple(boundaries))
  vals = np.array(vals)
  interval_sizes = bounds[1:] - bounds[:-1]

  indicator = (bounds[:-1] <= count) & (count < bounds[1:])
  pct = (count - bounds[:-1]) / interval_sizes
  interp_vals = interpolate_fn(vals[:-1], vals[1:], pct)

  return indicator.dot(interp_vals) + (bounds[-1] <= count) * vals[-1]


def fixed_schedule(
    count: Numeric,
    value: Numeric,
) -> Numeric:
  """Fixed/constant schedule."""
  del count
  return value


def kfac_resnet50_schedule(
    count: Numeric,
) -> Numeric:
  """Custom schedule for KFAC ResNet50 experiment."""

  # We linearly interpolate in log space
  exponent = piecewise_interpolated_schedule(
      count,
      vals=[-6.0, -3.1, -3.1, -3.23, -5.0, -5.7, -6.0],
      boundaries=[50, 5000, 11000, 20000, 200000, 1000001],
      interpolate_type="linear",
  )
  return jnp.power(10.0, exponent)


def cosine_schedule(
    count: Numeric,
    total: Numeric,
    peak_value: float,
    end_value: float = 0.0,
) -> Numeric:
  """Cosine schedule."""

  val = optax.cosine_decay_schedule(
      init_value=peak_value,
      decay_steps=total,
      alpha=end_value / peak_value if peak_value != 0 else 0.0,
  )(count)

  assert isinstance(val, Numeric)
  return val


def stepwise_schedule(
    count: Numeric,
    boundaries: Array,
    decay_factors: Sequence[float],
    init_value: float,
) -> Numeric:
  """A basic stepwise schedule.

  Returns init_value until boundaries[0], init_value * decay_factors[0]
  until boundaries[1], init_value * decay_factors[2] until boundaries[2], etc.
  After boundaries[-1], returns init_value * decay_factors[-1].

  Args:
    count: The current count.
    boundaries: The boundaries between the intervals.
    decay_factors: The decay factors for each interval.
    init_value: The initial value of the schedule.

  Returns:
    The value of the schedule at the current count.
  """

  if len(boundaries) != len(decay_factors):
    raise ValueError("`boundaries` and `decay_factors` must have the same "
                     "length.")

  values = jnp.concatenate(
      [jnp.array([1.0]), jnp.array(decay_factors)]) * init_value

  index = jnp.sum(boundaries <= count)

  return jnp.take(values, index)


def exponential_decay_schedule(
    count: int,
    start: Numeric,
    total: Numeric,
    init_value: float,
    end_value: float,
) -> Numeric:
  """Exponential decay schedule, similar to Optax."""

  val = optax.exponential_decay(
      init_value=init_value,
      end_value=end_value,
      decay_rate=end_value / init_value if init_value != 0 else 0.0,
      transition_begin=start,
      transition_steps=total - start,
  )(count)

  assert isinstance(val, Numeric)
  return val


def _custom_polynomial_schedule(
    init_value: Numeric,
    end_value: float,
    power: Numeric,
    transition_steps: int,
    transition_begin: int = 0
) -> GenericSchedule:
  """Polynomial schedule similar to Optax, but works even when init_value < end_value."""

  def schedule(count):

    count = jnp.clip(count - transition_begin, 0, transition_steps)

    if init_value >= end_value:
      frac = 1.0 - count / transition_steps
      return (init_value - end_value) * (frac ** power) + end_value
    else:
      frac = count / transition_steps
      return (end_value - init_value) * (frac ** power) + init_value

  return schedule


def polynomial_schedule(
    count: int,
    start: int,
    total: int,
    init_value: float,
    end_value: float,
    power: Numeric = 1,
) -> Numeric:
  """Polynomial schedule (defaults to linear), similar to Optax."""

  val = _custom_polynomial_schedule(
      init_value=init_value,
      end_value=end_value,
      power=power,
      transition_begin=start,
      transition_steps=total - start,
  )(count)

  assert isinstance(val, Numeric)
  return val


# For each schedule we specify:
#  - "params_to_convert": list of parameters to convert (excluding
#     warmup-related ones)
#  - "include_total": whether a total duration should be injected
#  - "warmup_end_value_key": the key whose value represents the schedule’s
#     regular starting value, which becomes the peak value reached at the end of
#     warmup (if warmup is used).
SCHEDULE_METADATA = {
    "fixed": {
        "ctor": fixed_schedule,
        "params_to_convert": [],
        "include_total": False,
        "warmup_end_value_key": "value",
    },
    "kfac_resnet50": {
        "ctor": kfac_resnet50_schedule,
        "params_to_convert": [],
        "include_total": False,
        "warmup_end_value_key": "value",
    },
    "cosine": {
        "ctor": cosine_schedule,
        "params_to_convert": [],
        "include_total": True,
        "warmup_end_value_key": "peak_value",
    },
    "stepwise": {
        "ctor": stepwise_schedule,
        "params_to_convert": ["boundaries"],
        "include_total": False,
        "warmup_end_value_key": "init_value",
    },
    "exponential_decay": {
        "ctor": exponential_decay_schedule,
        "params_to_convert": ["start"],
        "include_total": True,
        "warmup_end_value_key": "init_value",
    },
    "polynomial": {
        "ctor": polynomial_schedule,
        "params_to_convert": ["start"],
        "include_total": True,
        "warmup_end_value_key": "init_value",
    },
    "piecewise_interpolated": {
        "ctor": piecewise_interpolated_schedule,
        "params_to_convert": ["boundaries"],
        "include_total": False,
        "warmup_end_value_key": "vals",
    },
}


def with_warmup(
    base_schedule_fn: GenericSchedule,
    warmup_duration: Numeric,
    warmup_start_value: float,
    warmup_end_value: float
) -> GenericSchedule:
  """Wraps a base schedule with a linear warmup phase."""

  warmup_sched = optax.linear_schedule(
      init_value=warmup_start_value,
      end_value=warmup_end_value,
      transition_steps=warmup_duration,
  )

  return optax.join_schedules([warmup_sched, base_schedule_fn],  # pytype: disable=bad-return-type
                              [warmup_duration])


def construct_schedule(
    name: str,
    dataset_size: int,
    train_total_batch_size: int | None,
    total_steps: int | None,
    total_epochs: float | None,
    mode: str = "steps",
    **kwargs,
) -> kfac_jax.utils.ScheduleType:
  """Constructs the schedule from its name and extra kwargs.

  The `mode` argument (one of 'epochs', 'steps', or 'fraction') indicates how
  certain parameters (given in PARAM_CONVERSION and also warmup_duration) are
  interpreted:
    - 'epochs': values (e.g. boundaries, start) are in epochs.
    - 'fraction': values are fractions of total epochs.
    - 'steps': values are in optimizer steps.

  This function will also add linear warmup into the schedule if the 'warmup'
  argument is provided in kwargs. In that case, the starting value of the
  warmup phase is specified by 'warmup_start_value' if it's passed, and
  defaults to 0.0 otherwise. The schedule’s regular starting value (specified
  by the key defined in PARAM_CONVERSION as "warmup_peak_key") is taken as the
  peak value reached at the end of warmup. Note that all of the schedule's
  parameters (e.g. boundaries) are then interpretered relative to the end of the
  warmup phase.

  Args:
    name: The name of the schedule to construct.
    dataset_size: The size of the dataset. Only used for 'epochs' and 'fraction'
      modes.
    train_total_batch_size: The total batch size used for training. Must be set
      if mode is 'epochs' or 'fraction' in cases where data_seen is not passed
      to the schedule.
    total_steps: The total number of optimizer steps. Must be set if mode is
      'steps'. Must be None if total_epochs is set.
    total_epochs: The total number of epochs. Must be set if mode is 'epochs' or
      'fraction'. Must be None if total_steps is set.
    mode: The mode of the schedule (see above).
    **kwargs: Extra keyword arguments to pass to the schedule constructor.

  Returns:
    The constructed schedule function, which maps global_step and possibly
    data_seen (i.e. the total amount of training data seen so far) to the
    current value of the schedule.
  """

  if total_steps is not None and total_epochs is not None:
    raise ValueError("Only one of total_steps and total_epochs can be set.")

  if mode == "fraction" and total_epochs is None and total_steps is None:
    raise ValueError(
        "One of total_steps or total_epochs must be set when mode is"
        f" 'fraction' for schedule '{name}'."
    )

  if name not in SCHEDULE_METADATA:
    raise ValueError(f"Schedule '{name}' is not valid.")

  if mode not in ("epochs", "steps", "fraction"):
    raise ValueError("Mode must be one of 'epochs', 'steps', or 'fraction'.")

  if mode == "epochs":
    conversion_fn = lambda x: x * dataset_size
  elif mode == "steps":
    conversion_fn = lambda x: x
  elif mode == "fraction":
    if total_epochs is not None:
      conversion_fn = lambda x: x * total_epochs * dataset_size
    else:
      conversion_fn = lambda x: x * total_steps * train_total_batch_size

  # Convert all FieldReferences to their values. This is supposed to happen
  # automatically when the config is finalized, but doesn't work recursively for
  # list/tuple fields that contain FieldReferences for some reason.
  # new_kwargs = _convert_fieldreferences(kwargs)  # this also makes a copy
  new_kwargs = kwargs.copy()

  for param in SCHEDULE_METADATA[name]["params_to_convert"]:

    if param not in new_kwargs:
      raise ValueError(f"Parameter '{param}' is required for schedule "
                       f"'{name}'.")

    new_kwargs[param] = jax.tree.map(conversion_fn, new_kwargs.pop(param))

  if SCHEDULE_METADATA[name]["include_total"]:

    if mode == "steps":

      if total_steps is None:
        raise ValueError(
            "total_steps must be set when mode is 'steps' for "
            f"schedule '{name}'."
        )

      new_kwargs["total"] = total_steps

    elif mode == "epochs":

      if total_epochs is None:
        raise ValueError(
            "total_epochs must be set when mode is 'epochs' for schedule"
            " '{name}'."
        )

      # Convert to data seen in this case
      new_kwargs["total"] = total_epochs * dataset_size

    elif mode == "fraction":

      # Convert to data seen in this case
      if total_steps:
        new_kwargs["total"] = total_steps * train_total_batch_size
      else:
        new_kwargs["total"] = total_epochs * dataset_size

  # Create the base schedule (which does not include warmup).
  base_schedule = lambda count: SCHEDULE_METADATA[name]["ctor"](count,
                                                                **new_kwargs)

  # If a warmup is asked for, wrap the base schedule with it.
  if "warmup_duration" in kwargs:

    warmup_duration = conversion_fn(new_kwargs.pop("warmup_duration"))

    if SCHEDULE_METADATA[name]["include_total"]:
      new_kwargs["total"] -= warmup_duration

    if "warmup_start_value" in new_kwargs:
      warmup_start_value = new_kwargs.pop("warmup_start_value")
    else:
      warmup_start_value = 0.0

    warmup_end_value_key = SCHEDULE_METADATA[name]["warmup_end_value_key"]

    if warmup_end_value_key not in new_kwargs:
      raise ValueError(
          f"When 'warmup_duration' is provided, '{warmup_end_value_key}' "
          f"must be provided for schedule '{name}'.")

    warmup_end_value = new_kwargs[warmup_end_value_key]
    if isinstance(warmup_end_value, (list, tuple)):
      warmup_end_value = warmup_end_value[0]

    schedule = with_warmup(base_schedule, warmup_duration, warmup_start_value,
                           warmup_end_value)

  else:
    schedule = base_schedule

  # Convert the input to schedule to use data_seen for 'epochs' and 'fraction'
  # modes. In these cases, the 'total' argument of the schedule is the total
  # data seen at end of training.
  def schedule_with_input_conversion(global_step, data_seen=None):

    if mode in ("epochs", "fraction"):

      if data_seen is None:

        if train_total_batch_size is not None:
          data_seen = global_step * train_total_batch_size
        else:
          raise ValueError("One of 'train_total_batch_size' or 'data_seen' "
                           "must passed when mode is 'epochs' or 'fraction'.")

      return schedule(data_seen)

    else:

      return schedule(global_step)

  return schedule_with_input_conversion
