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
"""K-FAC utilities for classes with staged methods."""
import functools
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import jax

from kfac_jax._src.utils import misc
from kfac_jax._src.utils import parallel
from kfac_jax._src.utils import types

TArrayTree = types.TArrayTree
ArrayTree = types.ArrayTree


class WithStagedMethods(misc.Finalizable):
  """An mixin for classes which can have staged/compiled methods."""

  class StagingContext:
    """A context manager for handling methods that are staged/compiled."""

    def __init__(self, wsm_instance: "WithStagedMethods"):
      """Initializes the context manager.

      Args:
        wsm_instance: The corresponding `WithStagedMethods` instance.
      """
      self._wsm_instance = wsm_instance

    def __enter__(self):
      """Enters the staging context."""
      if self._wsm_instance._in_staging:
        raise RuntimeError("Cannot enter staging context while already in "
                           "staging context.")
      self._wsm_instance._in_staging = True

    def __exit__(self, *_):
      """Exits the staging context."""
      assert self._wsm_instance._in_staging, "Exiting while not in staging."
      self._wsm_instance._in_staging = False

  def __init__(
      self,
      debug: bool = False,
      **parent_kwargs: Any,
  ):
    """Initializes the instance.

    Args:
      debug: If this is set `True` than any call to a stage method would
        directly call the method and would not stage/compile it.
      **parent_kwargs: Any additional keyword arguments for the parent class.
    """
    if "excluded_attribute_names" in parent_kwargs:
      parent_kwargs["excluded_attribute_names"] = (
          ("_in_staging",) + tuple(parent_kwargs["excluded_attribute_names"]))
    else:
      parent_kwargs["excluded_attribute_names"] = ("_in_staging",)

    super().__init__(**parent_kwargs)
    self._debug = debug
    self._in_staging = False

  @property
  def debug(self) -> bool:
    """Whether staged methods would be run in 'debug' mode."""
    return self._debug

  @property
  def in_staging(self) -> bool:
    """Whether we are in a staging context while compiling staged methods."""
    return self._in_staging

  def staging_context(self) -> "StagingContext":
    """Returns a staging context manager, linked to this instance."""
    return self.StagingContext(self)

  def get_first(self, obj: TArrayTree) -> TArrayTree:
    """Indexes the `obj` PyTree leaves over leading axis if `multi_device`."""
    return obj

  def copy_obj(self, obj: TArrayTree) -> TArrayTree:
    """Copies the object."""
    return parallel.copy_obj(obj)

  # def replicate(self, obj: PyTree) -> PyTree:
  #   """Replicates the object to all local devices if `multi_device`."""
  #   if self.multi_device:
  #     return parallel.replicate_all_local_devices(obj)
  #   else:
  #     return obj


def staged(
    method: Callable[..., TArrayTree],
    static_argnums: Optional[Union[int, Sequence[int]]] = None,
    donate_argnums: Optional[Union[int, Sequence[int]]] = None,
) -> Callable[..., TArrayTree]:
  """Makes the instance method staged.

  This decorator **should** only be applied to instance methods of classes that
  inherit from the `WithStagedMethods` class. The decorator makes the decorated
  method staged, which is equivalent to `jax.jit` if `instance.multi_device` is
  `False` and to `jax.pmap` otherwise. When specifying static and donated
  argunms, the `self` reference **must not** be counted. Example:

    @functools.partial(staged, donate_argunms=0)
    def try(self, x):
      ...

    then `instance.try(x)` is equivalent to
    `jax.jit(instance.try, donate_argnums=0)(x)` if `instance.multi_device` is
    `False` and to `jax.pmap(instance.try, donate_argnums=0)(x)` otherwise.

  Args:
    method: The method to be transformed into a staged method.
    static_argnums: The static argument numbers, as defined in `jax.jit/pmap`.
    donate_argnums: The donated argument numbers, as defined in
      `jax.jit/pmap`.

  Returns:
    The transformed method, which will now be a staged function.
  """

  if isinstance(static_argnums, int):
    static_argnums = (static_argnums,)

  # This is needed because of b/147015762
  if donate_argnums is None:
    donate_argnums = ()
  if isinstance(donate_argnums, int):
    donate_argnums = (donate_argnums,)
  else:
    donate_argnums: Tuple[int, ...] = tuple(donate_argnums)

  # shift static_argnums by 1 and include instance (self)
  static_argnums = (0,) + tuple(i + 1 for i in (static_argnums or ()))
  # shift donate_argnums by 1 and include state
  donate_argnums = tuple(i + 1 for i in donate_argnums)

  jitted_method = jax.jit(
      method, donate_argnums=donate_argnums, static_argnums=static_argnums)

  @functools.wraps(method)
  def decorated(
      instance: "WithStagedMethods",
      *args: Any,
  ) -> ArrayTree:

    if instance.in_staging or instance.debug:
      return method(instance, *args)

    with instance.staging_context():
      return jitted_method(instance, *args)

  return decorated
