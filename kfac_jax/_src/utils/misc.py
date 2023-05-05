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
"""K-FAC related utility classes and functions."""
import abc
import dataclasses
import functools
from typing import Any, Iterator, Sequence, Type, Tuple, Union, Dict

import jax
import jax.numpy as jnp
from kfac_jax._src.utils import types

Array = types.Array
Numeric = types.Numeric
ArrayTree = types.ArrayTree
TArrayTree = types.TArrayTree
ShardingTree = types.ShardingTree


def get_sharding(x: ArrayTree) -> types.ShardingTree:
  return jax.tree_util.tree_map(lambda x: x.sharding, x)


def zeros_like_with_sharding(x: TArrayTree) -> TArrayTree:
  def zero_like_array(x: Array) -> Array:
    return jax.device_put(jnp.zeros_like(x), x.sharding)

  return jax.tree_util.tree_map(zero_like_array, x)


def fake_element_from_iterator(
    iterator: Iterator[TArrayTree],
) -> Tuple[TArrayTree, Iterator[TArrayTree]]:
  """Returns a zeroed-out initial element of the iterator "non-destructively".

  This function mutates the input iterator, hence after calling this function
  it will be advanced by one. An equivalent to the original iterator (e.g. not
  advanced by one) is returned as the second element of the returned pair. The
  advised usage of the function is:
    `fake_element, iterator = fake_element_from_iterator(iterator)`

  Args:
    iterator: A PyTree iterator. Must yield at least one element.

  Returns:
    A pair `(element, output_iterator)` where `element` is a zeroed-out version
    of the first element of the iterator, and `output_iterator` is an
    equivalent iterator to the input one.
  """
  init_element = next(iterator)
  fake_element = zeros_like_with_sharding(init_element)

  def equivalent_iterator() -> Iterator[ArrayTree]:
    yield init_element
    # For some reason unknown to us, "yield from" can fail in certain
    # circumstances
    while True:
      yield next(iterator)
  return fake_element, equivalent_iterator()


def to_tuple_or_repeat(
    x: Union[Numeric, Sequence[Numeric]],
    length: int,
) -> Tuple[Numeric, ...]:
  """Converts `x` to a tuple of fixed length.

  If `x` is an array, it is split along its last axis to a tuple (assumed to
  have `x.shape[-1] == length`). If it is a scalar, the scalar is repeated
  `length` times into a tuple, and if it is a list or a tuple it is just
  verified that its length is the same.

  Args:
    x: The input array, scalar, list or tuple.
    length: The length of the returned tuple.

  Returns:
    A tuple constructed by either replicating or splitting `x`.
  """
  if isinstance(x, jnp.ndarray) and x.size > 1:  # pytype: disable=attribute-error
    assert x.shape[-1] == length  # pytype: disable=attribute-error
    return tuple(x[..., i] for i in range(length))
  elif isinstance(x, (list, tuple)):
    assert len(x) == length
    return tuple(x)
  elif isinstance(x, (int, float, jnp.ndarray)):
    return (x,) * length
  else:
    raise ValueError(f"Unrecognized type for `x` - {type(x)}.")


def first_dim_is_size(size: int, *args: Array) -> bool:
  """Checks that each element of `args` has first axis size equal to `size`."""
  return all(arg.shape[0] == size for arg in args)


def pytree_dataclass(class_type: Type[Any]) -> Type[Any]:
  """Extended dataclass decorator, which also registers the class as a PyTree.

  The function is equivalent to `dataclasses.dataclass`, but additionally
  registers the `class_type` as a PyTree. This is done done by setting the
  PyTree nodes to all of the `dataclasses.fields` of the class.

  Args:
    class_type: The class type to transform.

  Returns:
    The transformed `class_type` which is now a dataclass and also registered as
    a PyTree.
  """
  class_type = dataclasses.dataclass(class_type)
  fields_names = tuple(field.name for field in dataclasses.fields(class_type))

  def flatten(instance) -> Tuple[Tuple[Any, ...], Any]:
    return tuple(getattr(instance, name) for name in fields_names), None

  def unflatten(_: Any, args: Sequence[Any]) -> Any:
    return class_type(*args)

  jax.tree_util.register_pytree_node(class_type, flatten, unflatten)

  return class_type


@pytree_dataclass
class State(abc.ABC):
  """Abstract class for optimizer state."""

  def copy(self):
    """Returns a copy of the PyTree structure (but not the JAX arrays)."""
    (flattened, structure) = jax.tree_util.tree_flatten(self)
    return jax.tree_util.tree_unflatten(structure, flattened)

  @abc.abstractmethod
  def as_dict(self) -> Dict[str, Any]:
    """Returns a recursively constructed dictionary of the state."""

  @classmethod
  def from_dict(cls, dict_rep: Dict[str, Any]) -> "State":
    """Returns a recursively reconstructed dictionary of the state."""
    return cls(**dict_rep)


class Finalizable(abc.ABC):
  """A mixin for classes that can "finalize" their attributes.

  The class provides the function `finalize` which freezes all attributes of the
  instance after its call. Any attributes assignment thereafter will raise an
  error. All subclasses must always call `super().__init__()` for the mixin to
  function properly, and they must set any attributes before any call to
  `finalize` has happened.
  """

  def __init__(
      self,
      forbid_setting_attributes_after_finalize: bool = True,
      excluded_attribute_names: Sequence[str] = (),
      **parent_kwargs: Any,
  ):
    """Initializes the instance.

    Args:
      forbid_setting_attributes_after_finalize: If `True`, trying to set
        attributes (via direct obj.attr = ...) after `finalize` was called on
        the instance will raise an error. If `False`, this is not checked.
      excluded_attribute_names: When `forbid_setting_attributes_after_finalize`
        is set to `True` this specifies any attributes names that can still be
        set.
      **parent_kwargs: Any keyword arguments to be passed to any parent class.
    """
    self._finalized = False
    self._forbid_setting_attributes = forbid_setting_attributes_after_finalize
    self._excluded_attribute_names = frozenset(excluded_attribute_names)
    super().__init__(**parent_kwargs)

  @property
  def finalized(self) -> bool:
    """Whether the object has already been finalized."""
    return self._finalized  # pytype: disable=attribute-error

  def finalize(self, *args: Any, **kwargs: Any):
    """Finalizes the object, after which no attributes can be set."""

    if self.finalized:
      raise ValueError("Object has already been finalized.")

    self._finalize(*args, **kwargs)
    self._finalized = True

  def _finalize(self, *args: Any, **kwargs: Any):
    """Any logic that a child class needs to do during the finalization."""

  def __setattr__(self, name: str, value: Any):

    if (not getattr(self, "_finalized", False) or
        not getattr(self, "_forbid_setting_attributes", True) or
        name in getattr(self, "_excluded_attribute_names", ())):

      super().__setattr__(name, value)

    else:
      raise AttributeError("Can't set attributes after finalization.")


def auto_scope_method(method):
  """Wraps the method call to have automatically generated Jax name scope."""
  @functools.wraps(method)
  def wrapped(instance, *args, **kwargs):
    class_name = type(instance).__name__
    method_name = method.__name__
    if method_name.startswith("_"):
      method_name = method_name[1:]
    with jax.named_scope(f"{class_name}_{method_name}"):
      return method(instance, *args, **kwargs)

  return wrapped


def auto_scope_function(func):
  """Wraps the function call to have automatically generated Jax name scope."""
  @functools.wraps(func)
  def wrapped(*args, **kwargs):
    with jax.named_scope(func.__name__):
      return func(*args, **kwargs)

  return wrapped


def default_batch_size_extractor(
    batch: types.Batch,
) -> Numeric:
  return jax.tree_util.tree_leaves(batch)[0].shape[0]
