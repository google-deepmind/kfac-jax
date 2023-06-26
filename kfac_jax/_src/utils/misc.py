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
import inspect
from typing import Any, Callable, Iterator, Sequence, Type, Tuple, Union, Dict, TypeVar

import jax
import jax.numpy as jnp
from kfac_jax._src.utils import types

Array = types.Array
Numeric = types.Numeric
ArrayTree = types.ArrayTree
TArrayTree = types.TArrayTree
StateType = TypeVar("StateType")
StateTree = types.PyTree["State"]


STATE_CLASSES_SERIALIZATION_DICT = {}


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
  fake_element = jax.tree_util.tree_map(jnp.zeros_like, init_element)
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


class State(abc.ABC):
  """Abstract class for state classes."""

  @classmethod
  def field_names(cls) -> Tuple[str, ...]:
    return tuple(field.name for field in dataclasses.fields(cls))  # pytype: disable=wrong-arg-types

  @classmethod
  def field_types(cls) -> Dict[str, Type[Any]]:
    return {field.name: field.type for field in dataclasses.fields(cls)}  # pytype: disable=wrong-arg-types

  @property
  def field_values(self) -> Tuple[ArrayTree, ...]:
    return tuple(getattr(self, name) for name in self.field_names())

  def copy(self: StateType) -> StateType:
    """Returns a copy of the PyTree structure (but not the JAX arrays)."""
    (flattened, structure) = jax.tree_util.tree_flatten(self)
    return jax.tree_util.tree_unflatten(structure, flattened)

  def tree_flatten(self) -> Tuple[Tuple[ArrayTree, ...], Tuple[str, ...]]:
    return self.field_values, self.field_names()

  @classmethod
  def tree_unflatten(
      cls,
      aux_data: Tuple[str, ...],
      children: Tuple[ArrayTree, ...],
  ):
    return cls(**dict(zip(aux_data, children)))

  def __repr__(self) -> str:
    return (f"{self.__class__.__name__}(" +
            ",".join(f"{name}={v!r}" for name, v in self.field_values) +
            ")")


def register_state_class(class_type: Type[Any]) -> Type[Any]:
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
  if not issubclass(class_type, State):
    raise ValueError(
        f"Class {class_type} is not a subclass of kfac_jax.utils.State."
    )

  class_type = dataclasses.dataclass(class_type)
  class_type = jax.tree_util.register_pytree_node_class(class_type)
  class_name = f"{class_type.__module__}.{class_type.__qualname__}"
  STATE_CLASSES_SERIALIZATION_DICT[class_name] = class_type
  return class_type


def serialize_state_tree(instance: StateTree) -> ArrayTree:
  """Returns a recursively constructed dictionary of the state."""
  if isinstance(instance, State):
    result_dict = {name: serialize_state_tree(getattr(instance, name))
                   for name in instance.field_names()}
    cls = instance.__class__
    result_dict["__class__"] = f"{cls.__module__}.{cls.__qualname__}"
    return result_dict

  elif isinstance(instance, list):
    return [serialize_state_tree(v) for v in instance]

  elif isinstance(instance, tuple):
    return tuple(serialize_state_tree(v) for v in instance)

  elif isinstance(instance, set):
    return set(serialize_state_tree(v) for v in instance)

  elif isinstance(instance, dict):
    return {k: serialize_state_tree(v) for k, v in instance.items()}

  else:
    return instance


def deserialize_state_tree(representation: ArrayTree) -> StateTree:
  """Returns the state class using a recursively constructed."""
  if isinstance(representation, list):
    return [deserialize_state_tree(v) for v in representation]

  elif isinstance(representation, tuple):
    return tuple(deserialize_state_tree(v) for v in representation)

  elif isinstance(representation, set):
    return set(deserialize_state_tree(v) for v in representation)

  elif isinstance(representation, dict):
    if "__class__" not in representation:
      return {k: deserialize_state_tree(v) for k, v in representation.items()}

    class_name = representation.pop("__class__")
    if class_name not in STATE_CLASSES_SERIALIZATION_DICT:
      raise ValueError(f"Did not find how to reconstruct class {class_name}.")

    dict_rep = deserialize_state_tree(representation)
    return STATE_CLASSES_SERIALIZATION_DICT[class_name](**dict_rep)

  else:
    return representation


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


def default_batch_size_extractor(batch: types.Batch) -> Numeric:
  """Computes the batch size as the size of axis `0` of the first element."""
  return jax.tree_util.tree_leaves(batch)[0].shape[0]


def replace_char(original: str, new_str: str, index: int) -> str:
  """Replaces the character at a given location."""
  return original[:index] + new_str + original[index + 1 :]


def call_func_with_conditional_kwargs(
    func: Callable[..., Any],
    *func_args: Any,
    **kwargs: Any) -> Any:

  sig = inspect.signature(func)
  func_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

  return func(*func_args, **func_kwargs)
