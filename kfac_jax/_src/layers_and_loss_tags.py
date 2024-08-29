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
"""K-FAC losses and layers tagging Jax primitives."""
import dataclasses
import functools
from typing import Any, Generic, Sequence, TypeVar

import jax
from jax import core


# Types for annotation
T = TypeVar("T")
Array = jax.Array
Arrays = tuple[Array, ...]


@dataclasses.dataclass(frozen=True, kw_only=True, unsafe_hash=True)
@jax.tree_util.register_pytree_node_class
class LayerData(Generic[T]):
  """A compact class for all data related to a single layer."""
  inputs: tuple[T, ...]
  outputs: tuple[T, ...]
  params: tuple[T, ...]

  def tree_flatten(self) -> tuple[
      tuple[tuple[T, ...], tuple[T, ...], tuple[T, ...]],
      None,
  ]:
    return (self.inputs, self.outputs, self.params), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    assert aux_data is None
    inputs, outputs, params = children
    return cls(inputs=inputs, outputs=outputs, params=params)


@dataclasses.dataclass(kw_only=True, unsafe_hash=True)
class LayerMetaData:
  """A compact class for all metadata related to a single layer."""
  variant: str
  outputs_index: tuple[int, ...]
  inputs_index: tuple[int, ...]
  params_index: tuple[int, ...]
  name: str | None = None
  nesting: tuple[str, ...] = ()


@dataclasses.dataclass(kw_only=True, unsafe_hash=True)
class LossMetaData(Generic[T]):
  """A compact class for all metadata related to a single layer."""

  loss_class: type[T]
  parameter_dependants: tuple[str, ...]
  parameter_independants: tuple[str, ...]
  argument_names: tuple[str, ...]


def get_and_verify_loss_meta(
    args: Sequence[Any],
    params: Any,
    err_suffix: str = "",
) -> LossMetaData:
  """Verifies that the number of arguments matches expectations."""
  meta = params.get("meta")
  if meta is None or not isinstance(meta, LossMetaData):
    raise ValueError(f"Meta must be LossMetaData, but found {meta=}.")
  if len(args) != len(meta.argument_names):
    raise ValueError(f"Number of arguments {len(args)} must match the "
                     f"number of argument names {len(meta.argument_names)} for"
                     f" {err_suffix}.")
  return meta


def get_loss_outputs(
    args: Sequence[T],
    params: dict[str, Any],
    err_suffix: str = "",
) -> tuple[T, ...]:
  meta = get_and_verify_loss_meta(args, params, err_suffix)
  kwargs = dict(zip(meta.argument_names, args))
  return tuple(kwargs[name] for name in meta.parameter_dependants)


class LossTag(core.Primitive):
  """A Jax primitive for tagging K-FAC losses.

  The primitive is no-op at runtime, however its goal is to tag (annotate) the
  Jax computation graph what expression exactly is the loss and what type of
  loss it represents. This is the only way for K-FAC to know how to compute the
  curvature matrix.
  """

  # Whether the primitive returns multiple outputs (from core.Primitive)
  multiple_results = True

  def __init__(self):
    """Initializes a loss tag primitive for the given :class:`~LossFunction` class.

    When the primitive is created, the constructor automatically registers it
    with the standard Jax machinery for differentiation, :func:`jax.vmap` and
    XLA lowering. For further details see please take a look at the JAX
    documentation on `primitives
    <https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html>`__.
    """
    super().__init__("loss_tag")

    jax.interpreters.mlir.register_lowering(self, self._mlir_lowering)
    jax.interpreters.ad.primitive_jvps[self] = self._jvp

    # This line defines how does the tag behave under vmap. It is required for
    # any primitive that can be used inside a vmap. The reason why we want to
    # allow this is two fold - one to not break user code when the tags are not
    # used at all, and two - to be able to define a network with code for a
    # single example which is the vmap-ed for a batch.
    jax.interpreters.batching.primitive_batchers[self] = self._batching

  def impl(self, *args: Array, **params: Any) -> tuple[Array, ...]:
    return get_loss_outputs(args, params)

  def abstract_eval(
      self,
      *args: Array,
      **params: Any,
  ) -> tuple[Arrays, jax.core.Effects]:

    return get_loss_outputs(args, params), jax.core.no_effects

  def _mlir_lowering(
      self,
      _: jax.interpreters.mlir.LoweringRuleContext,
      *args,
      **params: Any,
  ) -> tuple[Any, ...]:
    """The XLA translation rule for this primitive (creates a no-op tuple)."""
    return get_loss_outputs(args, params)

  def _jvp(
      self,
      arg_values: Sequence[Array],
      arg_tangents: Sequence[Array],
      **params: Any,
  ) -> tuple[Arrays, Arrays]:
    """Computes the Jacobian-vector product for the primitive."""

    if len(arg_values) != len(arg_tangents):
      raise ValueError("Values and tangents are not the same length.")

    primal_output = self.bind(*arg_values, **params)
    tangent_output = get_loss_outputs(arg_tangents, params)

    return primal_output, tangent_output

  def _batching(
      self,
      batched_args: Sequence[Array],
      batched_dims: int | tuple[int, ...],
      **params: Any,
  ) -> tuple[Array, int | tuple[int, ...]]:
    """Defines how the primitive behaves under :func:`jax.vmap`."""

    return self.bind(*batched_args, **params), batched_dims[:1]


def loss_eqn_parameter_dependants(
    eqn: jax.core.JaxprEqn,
    raise_an_error: bool = True,
) -> list[jax.core.Var]:
  """Returns the parameter dependants variables from the give loss equation."""
  if not isinstance(eqn.primitive, LossTag):
    if raise_an_error:
      raise ValueError("Primitive must be a LossTag.")
    return []

  meta = eqn.params.get("meta")
  assert meta is not None and isinstance(meta, LossMetaData)
  assert len(eqn.invars) == len(meta.argument_names)
  kwargs = dict(zip(meta.argument_names, eqn.invars))
  return [kwargs[name] for name in meta.parameter_dependants]


def loss_eqn_construct_loss(
    eqn: jax.core.JaxprEqn,
    *args: Array,
) -> Any:
  """Constructs an instance of the corresponding :class:`~LossFunction` class."""
  if not isinstance(eqn.primitive, LossTag):
    raise ValueError("Primitive must be a LossTag.")

  meta: LossMetaData[T] = eqn.params.get("meta")
  assert meta is not None and isinstance(meta, LossMetaData)
  assert len(eqn.invars) == len(meta.argument_names)
  kwargs = dict(zip(meta.argument_names, args))
  return meta.loss_class(**kwargs)


def loss_eqn_class_name(eqn: jax.core.JaxprEqn) -> str:
  """The name of the underlying `~LossFunction` class."""

  if not isinstance(eqn.primitive, LossTag):
    raise ValueError("Primitive must be a LossTag.")

  meta: LossMetaData[T] = eqn.params.get("meta")
  assert meta is not None and isinstance(meta, LossMetaData)

  return meta.loss_class.__name__


def get_and_verify_layer_meta(
    args: Sequence[Any],
    params: dict[str, Any],
    err_suffix: str = "",
) -> LayerMetaData:
  """Verifies that the number of arguments matches expectations."""

  meta = params.get("meta")

  if meta is None or not isinstance(meta, LayerMetaData):
    raise ValueError(f"Meta must be LayerMetaData, but found {meta=}.")

  n = len(args)

  for i in meta.inputs_index:
    if i >= n:
      raise ValueError(
          f"Meta data has {meta.input_index=}, but only {n} "
          f"arguments passed for {err_suffix}.")

  for i in meta.outputs_index:
    if i >= n:
      raise ValueError(
          f"Meta data has {meta.output_index=}, but only {n} "
          f"arguments passed for {err_suffix}.")

  for i in meta.params_index:
    if i >= n:
      raise ValueError(
          f"Meta data has {meta.params_index=}, but only {n} "
          f"arguments passed for {err_suffix}.")

  return meta


class LayerTag(core.Primitive):
  """A Jax primitive for tagging K-FAC layers.

  The primitive is no-op at runtime, however its goal is to tag (annotate) the
  Jax computation graph what expressions represents a single unique layer type.
  This is the only way for K-FAC to know how to compute the curvature matrix.
  """

  def __init__(self):
    """Initializes a layer tag primitive with the given name.

    Any layer tag primitive must have the following interface `layer_tag(
    *outputs, *inputs, *parameters, **kwargs)`. We refer collectively to
    ``inputs`` , ``outputs`` and ``parameters`` as operands. All operands must
    be Jax arrays, while any of the values in ``kwargs`` must be hashable fixed
    constants.

    When the primitive is created, the constructor automatically registers it
    with the standard Jax machinery for differentiation, :func:`jax.vmap` and
    XLA lowering. For further details see please take a look at the JAX
    documentation on `primitives
    <https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html>`__.

    """
    super().__init__(name="layer_tag")
    jax.interpreters.mlir.register_lowering(self, self._mlir_lowering)
    jax.interpreters.ad.deflinear(self, self._transpose)
    jax.interpreters.ad.primitive_transposes[self] = self._transpose
    # This line defines how does the tag behave under vmap. It is required for
    # any primitive that can be used inside a vmap. The reason why we want to
    # allow this is two fold - one to not break user code when the tags are not
    # used at all, and two - to be able to define a network with code for a
    # single example which is the vmap-ed for a batch.
    jax.interpreters.batching.primitive_batchers[self] = self._batching

  def layer_data(  # pytype: disable=invalid-annotation
      self,
      args: Sequence[T],
      params: dict[str, Any],
      err_suffix: str = "",
  ) -> LayerData[T]:
    """Splits the operands of the primitive into ``(outputs, inputs, params)``."""
    meta = get_and_verify_layer_meta(args, params, err_suffix)
    return LayerData(
        inputs=tuple(args[i] for i in meta.inputs_index),
        outputs=tuple(args[i] for i in meta.outputs_index),
        params=tuple(args[i] for i in meta.params_index),
    )

  def _mlir_lowering(
      self,
      _: jax.interpreters.mlir.LoweringRuleContext,
      *args: T,
      **params: Any,
  ) -> tuple[T, ...]:
    """The XLA translation rule for this primitive - returns the ``outputs`` ."""
    return self.layer_data(args, params).outputs

  @classmethod
  def _transpose(
      cls,
      cotangent: Array,
      *args: Array,
      **_: Any,
  ) -> tuple[Array | None, ...]:
    """Computes the cotangents of the operands from those of the primitive."""
    del cls  # not used
    return (cotangent,) + (None,) * (len(args) - 1)

  def impl(self, *args: Array, **params: Any) -> Array:
    # For now we support only single output
    [output] = self.layer_data(args, params).outputs
    return output

  def abstract_eval(
      self,
      *args: Array,
      **params: Any,
  ) -> tuple[Array, jax.core.Effects]:
    # For now we support only single output
    [output] = self.layer_data(args, params).outputs
    return output, jax.core.no_effects

  def _batching(
      self,
      batched_args: Sequence[Array],
      batched_dims: int | tuple[int, ...],
      **params: Any,
  ) -> tuple[Array, int]:
    """Defines how the primitive behaves under :func:`jax.vmap`."""
    return self.bind(*batched_args, **params), batched_dims[0]


def layer_eqn_data(  # pytype: disable=invalid-annotation
    eqn: jax.core.JaxprEqn,
    raise_an_error: bool = True,
) -> LayerData[jax.core.Var]:

  if isinstance(eqn.primitive, LayerTag):
    return eqn.primitive.layer_data(eqn.invars, eqn.params, str(eqn))

  if raise_an_error:
    raise ValueError("Primitive must be a LayerTag.")
  else:
    return LayerData(inputs=(), outputs=(), params=())


def layer_eqn_name(eqn: jax.core.JaxprEqn) -> str:
  meta = get_and_verify_layer_meta(eqn.invars, eqn.params)
  if meta.name is None:
    raise ValueError("Layer name must be provided at this stage.")
  return meta.name


loss_tag = LossTag()
layer_tag = LayerTag()


def register_generic(*args: Array) -> Array:
  """Registers a generic tag around the provided parameters array."""
  return layer_tag.bind(
      *args,
      meta=LayerMetaData(
          variant="generic",
          inputs_index=(),
          outputs_index=(0,),
          params_index=tuple(range(len(args))),
      ),
  )


def register_dense(
    y: Array,
    x: Array,
    w: Array,
    b: Array | None = None,
    variant: str = "dense",
    **kwargs,
) -> Array:
  """Registers a dense layer: ``y = matmul(x, w) + b``."""
  args = (y, x, w) if b is None else (y, x, w, b)
  return layer_tag.bind(
      *args,
      meta=LayerMetaData(
          variant=variant,
          outputs_index=(0,),
          inputs_index=(1,),
          params_index=tuple(i + 2 for i in range(len(args) - 2)),
      ),
      **kwargs,
  )


def register_conv2d(
    y: Array,
    x: Array,
    w: Array,
    b: Array | None = None,
    variant: str = "conv2d",
    **kwargs: Any
) -> Array:
  """Registers a 2d convolution layer: ``y = conv2d(x, w) + b``."""
  args = (y, x, w) if b is None else (y, x, w, b)
  return layer_tag.bind(
      *args,
      meta=LayerMetaData(
          variant=variant,
          outputs_index=(0,),
          inputs_index=(1,),
          params_index=tuple(i + 2 for i in range(len(args) - 2)),
      ),
      **kwargs,
  )


def register_scale_and_shift(
    y: Array,
    x: Array,
    scale: Array | None = None,
    shift: Array | None = None,
    variant: str = "scale_and_shift",
    **kwargs: Any,
) -> Array:
  """Registers a scale and shift layer: ``y = x * scale + shift``."""
  args = tuple(a for a in (y, x, scale, shift) if a is not None)
  if len(args) < 3:
    raise ValueError("At least one of `scale` and `shift` must be provided.")
  return layer_tag.bind(
      *args,
      meta=LayerMetaData(
          variant=variant,
          outputs_index=(0,),
          inputs_index=(1,),
          params_index=tuple(i + 2 for i in range(len(args) - 2)),
      ),
      has_scale=scale is not None,
      has_shift=shift is not None,
      **kwargs,
  )


register_repeated_dense = functools.partial(
    register_dense,
    variant="repeated_dense",
)


class LossTagEqn(core.JaxprEqn):
  """A class used only for annotation purposes."""
  primitive: LossTag


class LayerTagEqn(core.JaxprEqn):
  """A class used only for annotation purposes."""
  primitive: LayerTag
