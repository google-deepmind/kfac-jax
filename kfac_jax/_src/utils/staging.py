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
import numbers
import operator
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from kfac_jax._src.utils import misc
from kfac_jax._src.utils import parallel
from kfac_jax._src.utils import types

TArrayTree = types.TArrayTree


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
      multi_device: bool = False,
      pmap_axis_name: Optional[str] = None,
      debug: bool = False,
      **parent_kwargs: Any,
  ):
    """Initializes the instance.

    Args:
      multi_device: Whether any of decorated staged methods are to be run on a
        single or multiple devices. If this is set to `True` than any call
        would internally be delegated to `jax.pmap` and otherwise to  `jax.jit`.
      pmap_axis_name: The name of the pmap axis to use when running on
        multiple devices. This is required if `multi_device=True`.
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

    if multi_device and not isinstance(pmap_axis_name, str):
      raise ValueError("When `multi_device=True` you must pass in a string for "
                       "`pmap_axis_name`.")

    self._multi_device = multi_device
    self._pmap_axis_name = pmap_axis_name
    self._debug = debug
    self._in_staging = False

  @property
  def multi_device(self) -> bool:
    """Indicates whether staged method will be run across multiple devices."""
    return self._multi_device

  @property
  def pmap_axis_name(self) -> Optional[str]:
    """The name of the `jax.pmap` axis to use for staged methods."""
    return self._pmap_axis_name

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
    return parallel.get_first(obj) if self.multi_device else obj

  def copy_obj(self, obj: Optional[TArrayTree]) -> Optional[TArrayTree]:
    """Copies the object."""
    if self.multi_device:
      return parallel.pmap_copy_obj(obj)
    else:
      return parallel.copy_obj(obj)

  def replicate(self, obj: TArrayTree) -> TArrayTree:
    """Replicates the object to all local devices if `multi_device`."""
    if self.multi_device:
      return parallel.replicate_all_local_devices(obj)
    else:
      return obj


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

  bcast_argnums = static_argnums or ()

  # shift static_argnums by 1 and include instance (self)
  static_argnums = (0,) + tuple(i + 1 for i in (static_argnums or ()))
  # shift donate_argnums by 1 and include state
  donate_argnums = tuple(i + 1 for i in donate_argnums)

  pmap_funcs = {}
  jitted_func = jax.jit(method,
                        static_argnums=static_argnums,
                        donate_argnums=donate_argnums)

  @functools.wraps(method)
  def decorated(instance: "WithStagedMethods", *args: Any) -> TArrayTree:

    if instance.in_staging:
      return method(instance, *args)

    with instance.staging_context():
      if instance.multi_device and instance.debug:
        # In this case we want to call `method` once for each device index.
        # Note that this might not always produce sensible behavior, and will
        # depend on the details of the method and if it has side effects on the
        # state of the class.

        outs = []
        non_bcast_args = [args[i] if i not in bcast_argnums else None
                          for i in range(len(args))]

        for i in range(jax.local_device_count()):

          non_bcast_args_i = jax.tree_util.tree_map(
              operator.itemgetter(i), non_bcast_args)

          args_i = [
              non_bcast_args_i[j] if j not in bcast_argnums else args[j]
              for j in range(len(args))
          ]

          outs.append(method(instance, *args_i))

        outs = jax.tree_util.tree_map(lambda *args_: jnp.stack(args_), *outs)

      elif instance.debug:
        outs = method(instance, *args)

      elif instance.multi_device:
        # Compute in_axes so we broadcast any argument that is a scalar
        in_axes = [None]
        for i in range(len(args)):
          if (isinstance(args[i], numbers.Number) or
              (isinstance(args[i], jax.Array) and not args[i].shape)):
            # Single scalar
            in_axes.append(None)
          else:
            in_axes.append(0)

        in_axes = tuple(in_axes)
        key = (instance.pmap_axis_name, in_axes)
        func = pmap_funcs.get(key)

        if func is None:
          func = jax.pmap(
              method,
              static_broadcasted_argnums=static_argnums,
              donate_argnums=donate_argnums,
              axis_name=instance.pmap_axis_name,
              in_axes=in_axes,
          )
          pmap_funcs[key] = func

        outs = func(instance, *args)

      else:
        outs = jitted_func(instance, *args)

    return outs

  return decorated
