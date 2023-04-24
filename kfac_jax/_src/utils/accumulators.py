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
from typing import Optional, Generic, Any, Dict

import jax
import jax.numpy as jnp

from kfac_jax._src.utils import misc
from kfac_jax._src.utils import types

Array = types.Array
Numeric = types.Numeric
Shape = types.Shape
DType = types.DType
ArrayTree = types.ArrayTree
TArrayTree = types.TArrayTree


@misc.pytree_dataclass
class WeightedMovingAverage(Generic[TArrayTree], misc.State):
  """A wrapped class for an arbitrary weighted moving average."""
  weight: Numeric
  raw_value: Optional[TArrayTree]

  @property
  def value(self) -> Optional[TArrayTree]:
    """The value of the underlying arrays data structure."""
    if types.tree_is_empty(self.raw_value):
      return self.raw_value

    return jax.tree_util.tree_map(lambda x: x / self.weight, self.raw_value)

  def update(
      self,
      value: TArrayTree,
      old_weight_multiplier: Numeric,
      new_weight: Numeric,
  ):
    """Updates the underlying array and weight accordingly."""
    if self.raw_value is None:
      self.raw_value = value
      self.weight = jnp.asarray(new_weight).astype(self.weight.dtype)

    else:
      self.weight = self.weight * old_weight_multiplier + new_weight
      self.raw_value = jax.tree_util.tree_map(
          lambda x, y: x * old_weight_multiplier + y * new_weight,
          self.raw_value,
          value,
      )

  def clear(self, value_to_none: bool = False):
    """Resets the weighted average."""
    self.weight = jnp.zeros_like(self.weight)
    self.raw_value = None if value_to_none else jnp.zeros_like(self.raw_value)

  def value_and_clear(self) -> TArrayTree:
    """Retrieves the value of the weighted average and clears it."""
    value = self.value
    self.clear()
    return value

  def copy(self):
    """Returns a copy of the PyTree structure (but not the JAX arrays)."""
    (flattened, structure) = jax.tree_util.tree_flatten(self)
    return jax.tree_util.tree_unflatten(structure, flattened)

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
    dtype = types.get_float_dtype_and_check_consistency(value)
    weight = jnp.array(0.0, dtype=dtype)
    if value is not None:
      weight = jax.device_put(weight, jax.sharding.NamedSharding(
          jax.tree_leaves(value)[0].sharding.mesh,
          jax.sharding.PartitionSpec(),
      ))
    return cls(  # pytype: disable=wrong-keyword-args
        weight=weight,
        raw_value=misc.zeros_like_with_sharding(value),
    )

  @classmethod
  def empty(cls, dtype: Optional[DType] = None) -> "WeightedMovingAverage[Any]":
    """Returns an empty moving average instance."""
    weight = jnp.zeros([]) if dtype is None else jnp.zeros([], dtype=dtype)
    return cls(weight=weight, raw_value=None)  # pytype: disable=wrong-keyword-args

  @classmethod
  def state_sharding(
      cls,
      sharding: jax.sharding.NamedSharding,
  ) -> "WeightedMovingAverage[jax.sharding.NamedSharding]":
    return cls(  # pytype: disable=wrong-keyword-args
        weight=jax.sharding.NamedSharding(
            sharding.mesh, jax.sharding.PartitionSpec()
        ),
        raw_value=sharding,
    )

  def as_dict(self) -> Dict[str, Any]:
    return {"weight": self.weight, "raw_value": self.raw_value}

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}({self.weight!r}, {self.raw_value!r})"


@misc.pytree_dataclass
class MultiChunkAccumulator(WeightedMovingAverage[TArrayTree]):
  """Statistics accumulation, abstracted over multiple chunks."""

  def add(self, value: TArrayTree, weight: Numeric = 1):
    """Adds an element to the moving average and the max."""
    return self.update(value, 1, weight)

  def clear(self, value_to_none: bool = True):
    return super().clear(value_to_none=value_to_none)
