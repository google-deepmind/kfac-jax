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
"""K-FAC for accumulating statistics."""
from typing import Any, Optional, Generic

import jax
import jax.numpy as jnp

from kfac_jax._src.utils import misc
from kfac_jax._src.utils import parallel
from kfac_jax._src.utils import types

Array = types.Array
Numeric = types.Numeric
Shape = types.Shape
DType = types.DType
ArrayTree = types.ArrayTree
TArrayTree = types.TArrayTree


@misc.register_state_class
class WeightedMovingAverage(Generic[TArrayTree], misc.State):
  """A wrapped class for an arbitrary weighted moving average."""

  weight: Numeric
  raw_value: Optional[TArrayTree]

  @property
  def value(self) -> TArrayTree:
    """The value of the underlying arrays data structure."""
    return jax.tree_util.tree_map(lambda x: x / self.weight, self.raw_value)

  def update(
      self,
      value: TArrayTree,
      old_weight_multiplier: Numeric,
      new_weight: Numeric,
  ):
    """Updates the underlying array and weight accordingly."""

    assert self.raw_value is not None

    # A negative value of new_weight means we should only update the value
    # (with -new_weight) and not the total running weight. This roughly
    # corresponds to summation instead of averaging, and is useful in a few
    # contexts.
    new_weight_for_value = jnp.abs(new_weight)
    new_weight_for_weight = jax.nn.relu(new_weight)

    self.weight = self.weight * old_weight_multiplier + new_weight_for_weight

    self.raw_value = jax.tree_util.tree_map(
        lambda x, y: x * old_weight_multiplier + y * new_weight_for_value,
        self.raw_value,
        value,
    )

  def sync(self, pmap_axis_name: Optional[str]):
    """Syncs the underlying array across devices."""

    if self.raw_value is None:
      raise ValueError("`raw_value` has not been set yet.")

    self.raw_value = parallel.pmean_if_pmap(self.raw_value, pmap_axis_name)

  def clear(self, value_to_none: bool = False):
    """Resets the weighted average."""

    self.weight = jnp.zeros_like(self.weight)
    self.raw_value = None if value_to_none else jnp.zeros_like(self.raw_value)

  def value_and_clear(self) -> TArrayTree:
    """Retrieves the value of the weighted average and clears it."""

    value = self.value
    self.clear()

    return value

  @classmethod
  def zeros_array(
      cls,
      shape: Shape,
      dtype: Optional[DType] = None,
  ) -> "WeightedMovingAverage[Array]":
    """Initializes a `WeightedMovingAverage` with a single array of zeros."""

    return cls(  # pytype: disable=wrong-keyword-args
        weight=jnp.zeros([], dtype=dtype),
        raw_value=jnp.zeros(shape, dtype=dtype),
    )

  @classmethod
  def zeros_like(cls, value: TArrayTree) -> "WeightedMovingAverage[TArrayTree]":
    """Initializes a `WeightedMovingAverage` with zeros structure like `value`."""

    return cls(  # pytype: disable=wrong-keyword-args
        weight=jnp.array(
            0.0, dtype=types.get_float_dtype_and_check_consistency(value)
        ),
        raw_value=jax.tree_util.tree_map(jnp.zeros_like, value),
    )


class MultiChunkAccumulator(Generic[TArrayTree]):
  """Statistics accumulation, abstracted over multiple chunks."""

  def __init__(
      self,
      init_obj_value: Optional[TArrayTree],
      weight: Numeric,
      multi_device: bool,
  ):
    """Initializes an accumulator instance with the provided object and counter.

    Args:
      init_obj_value: The initial value of the accumulator.
      weight: The initial weight, which specifies how many samples are assumed
        to have been already counted in the initial value of the accumulator.
      multi_device: Whether the objects that are accumulated are outputs of a
        multi-device computation (e.g. `jax.pmap`).
    """
    self._accumulator = init_obj_value
    self._weight = weight
    self._multi_device = multi_device

  @property
  def accumulator(self) -> TArrayTree:
    """The current value of the underlying not-normalized accumulator."""
    return self._accumulator

  @property
  def weight(self) -> Numeric:
    """The current normalization weight of the underlying accumulator."""
    return self._weight

  @property
  def multi_device(self) -> bool:
    """Whether the accumulator is the output of a multi-device computation."""
    return self._multi_device

  @property
  def value(self) -> TArrayTree:
    """The current normalized value of the accumulator."""

    if types.tree_is_empty(self.accumulator):
      return self.accumulator

    if self._multi_device:
      return parallel.pmap_sync_and_divide_value(self.accumulator, self.weight)
    else:
      return parallel.jit_sync_and_divide_value(self.accumulator, self.weight)

  def clear(self) -> None:
    """Sets the underlying accumulator and weight to `None`."""
    self._accumulator = None
    self._weight = None

  def value_and_clear(self) -> TArrayTree:
    """Retrieves the normalized value of the accumulator and clears it."""

    value = self.value
    self.clear()

    return value

  def add(self, value_obj: TArrayTree, weight: Numeric = 1):
    """Adds an element to the moving average and the max.

    The exact update equation for the statistics are:
      raw_value_t = raw_value_{t-1} + value_obj * weight
      weight_t = weight_{t-1} + weight

    Args:
      value_obj: The value of the object, which scaled by `weight` will be added
        to the accumulator.
      weight: The relative weight of the `value_obj`.
    """

    value_obj = jax.tree_util.tree_map(lambda x: x * weight, value_obj)

    if self._accumulator is None:

      self._accumulator = value_obj

      if isinstance(weight, types.SCALAR_TYPES):
        self._weight = jnp.full_like(self._weight, weight)

      elif not isinstance(weight, jax.Array):
        raise ValueError("`weight` should be an instance of float, int or "
                         "jax.Array.")

      elif self._weight.shape != weight.shape:  # pytype: disable=attribute-error  # numpy-scalars
        raise ValueError("If `weight` is an `jnp.ndarray` then should have the "
                         "same shape as the weight of the accumulator.")
      else:
        self._weight = weight

      return

    if not types.tree_is_empty(self._accumulator):

      if types.tree_is_empty(value_obj):
        raise ValueError("The provided `value_obj` has an empty PyTree "
                         "structure, but the accumulator has been initialized "
                         "with a non-empty PyTree object.")

      self._accumulator = jax.tree_util.tree_map(
          jnp.add, self._accumulator, value_obj)

    elif not types.tree_is_empty(value_obj):

      raise ValueError("The provided `value_obj` has a non-empty PyTree "
                       "structure, but the accumulator has been initialized "
                       "with an empty PyTree object.")

    self._weight = self._weight + weight

  @classmethod
  def zeros_like(
      cls,
      obj: TArrayTree,
      multi_device: bool
  ) -> "MultiChunkAccumulator[TArrayTree]":
    """Creates a zero initialized accumulator as `obj`."""

    if multi_device:
      value = (parallel.pmap_zeros_like(obj)
               if not types.tree_is_empty(obj) else obj)
      weight = parallel.replicate_all_local_devices(
          jnp.zeros([], dtype=jnp.int32))
    else:
      value = (parallel.jit_zeros_like(obj)
               if not types.tree_is_empty(obj) else obj)
      weight = jnp.zeros([], dtype=jnp.int32)

    return cls(value, weight, multi_device)

  @classmethod
  def empty(cls, multi_device: bool) -> "MultiChunkAccumulator[Any]":
    """Creates an empty accumulator."""

    weight = jnp.zeros([], dtype=jnp.int32)

    if multi_device:
      weight = parallel.replicate_all_local_devices(weight)

    return cls(None, weight, multi_device)

  def __repr__(self):
    return (f"{self.__class__.__name__}({self._accumulator!r}, "
            f"{self._weight!r}, {self._multi_device})")

  def copy(self):
    """Returns a copy of the PyTree structure (but not the JAX arrays)."""

    (flattened, structure) = jax.tree_util.tree_flatten(self)

    return jax.tree_util.tree_unflatten(structure, flattened)


jax.tree_util.register_pytree_node(
    MultiChunkAccumulator,
    lambda x: ((x.accumulator, x.weight), (x.multi_device,)),
    lambda fixed, arrays: MultiChunkAccumulator(*arrays, *fixed)
)
