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
import types
from typing import Any, Generic, Optional, Sequence, Type, TypeVar, Tuple, Union

import jax
from jax import core
from jax.interpreters import batching as jax_batching

# Types for annotation
T = TypeVar("T")
Array = jax.Array
Arrays = Tuple[Array, ...]
ArrayOrXla = TypeVar("ArrayOrXla", Array, jax.interpreters.xla.XlaOp)


class LossTag(core.Primitive, Generic[T]):
  """A Jax primitive for tagging K-FAC losses.

  The primitive is no-op at runtime, however its goal is to tag (annotate) the
  Jax computation graph what expression exactly is the loss and what type of
  loss it represents. This is the only way for K-FAC to know how to compute the
  curvature matrix.
  """

  # Whether the primitive returns multiple outputs (from core.Primitive)
  multiple_results = True

  def __init__(
      self,
      cls: Type[T],
      parameter_dependants: Sequence[str],
      parameter_independants: Sequence[str],
  ):
    """Initializes a loss tag primitive for the given :class:`~LossFunction` class.

    When the primitive is created, the constructor automatically registers it
    with the standard Jax machinery for differentiation, :func:`jax.vmap` and
    XLA lowering. For further details see please take a look at the JAX
    documentation on `primitives
    <https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html>`__.

    Args:
      cls: The corresponding class of :class:`~LossFunction` that this tag
        represents.
      parameter_dependants: The names of each of the parameter **dependent**
        inputs to the tag.
      parameter_independants: The names of each of the parameter **independent**
        inputs to the tag.
    """
    super().__init__(cls.__name__ + "_tag")
    self._cls = cls
    self._parameter_dependants = tuple(parameter_dependants)
    self._parameter_independants = tuple(parameter_independants)

    jax.interpreters.xla.register_translation(self, self._xla_translation)
    jax.interpreters.ad.primitive_jvps[self] = self._jvp
    # This line defines how does the tag behave under vmap. It is required for
    # any primitive that can be used inside a vmap. The reason why we want to
    # allow this is two fold - one to not break user code when the tags are not
    # used at all, and two - to be able to define a network with code for a
    # single example which is the vmap-ed for a batch.
    jax_batching.primitive_batchers[self] = self._batching

  @property
  def parameter_dependants_names(self) -> Tuple[str, ...]:
    """The number of parameter dependent inputs to the tag primitive."""
    return self._parameter_dependants

  @property
  def parameter_independants_names(self) -> Tuple[str, ...]:
    """The number of parameter **independent** inputs to the tag primitive."""
    return self._parameter_independants

  @property
  def arguments_names(self):
    return self.parameter_dependants_names + self.parameter_independants_names

  def extract_parameter_dependants(
      self,
      *args: T,
      args_names: Sequence[str],
  ) -> Tuple[T, ...]:
    assert len(args) == len(args_names)
    arg_map = dict(zip(args_names, args))
    return tuple(arg_map[name] for name in self.parameter_dependants_names)

  def loss(self, *args: Array, args_names: Sequence[str]) -> T:
    """Constructs an instance of the corresponding :class:`~LossFunction` class."""
    assert len(args) == len(args_names)
    arg_map = dict(zip(args_names, args))
    return self._cls(**arg_map)

  def get_outputs(
      self,
      *args: ArrayOrXla,
      args_names: Sequence[str],
  ) -> Tuple[ArrayOrXla, ...]:
    """Verifies that the number of arguments matches expectations."""
    assert len(args) == len(args_names)
    return tuple(arg for name, arg in zip(args_names, args)
                 if name in self.parameter_dependants_names)

  def impl(self, *operands: Array, args_names: Sequence[str]) -> Arrays:
    return self.get_outputs(*operands, args_names=args_names)

  def abstract_eval(
      self,
      *operands: Array,
      args_names: Sequence[str],
  ) -> Tuple[Arrays, jax.core.Effects]:
    return (self.get_outputs(*operands, args_names=args_names),
            jax.core.no_effects)

  def _xla_translation(
      self,
      xla_context: jax.interpreters.xla.TranslationContext,
      avals_in: Sequence[core.AbstractValue],
      avals_out: Sequence[core.AbstractValue],
      *args: jax.interpreters.xla.XlaOp,
      args_names: Sequence[str],
  ) -> Tuple[jax.interpreters.xla.XlaOp, ...]:
    """The XLA translation rule for this primitive (creates a no-op tuple)."""
    del avals_in, avals_out  # not used
    return self.get_outputs(*args, args_names=args_names)

  def _jvp(
      self,
      arg_values: Sequence[Array],
      arg_tangents: Sequence[Array],
      args_names: Sequence[str],
  ) -> Tuple[Arrays, Arrays]:
    """Computes the Jacobian-vector product for the primitive."""
    if len(arg_values) != len(arg_tangents):
      raise ValueError("Values and tangents are not the same length.")
    primal_output = self.bind(*arg_values, args_names=args_names)
    tangent_output = self.get_outputs(*arg_tangents, args_names=args_names)
    return primal_output, tangent_output

  def _batching(
      self,
      batched_args: Sequence[Array],
      batched_dims: Union[int, Tuple[int, ...]],
      args_names: Sequence[str],
  ) -> Tuple[Array, Union[int, Tuple[int, ...]]]:
    """Defines how the primitive behaves under :func:`jax.vmap`."""
    return self.bind(*batched_args, args_names=args_names), batched_dims


class LayerTag(core.Primitive):
  """A Jax primitive for tagging K-FAC layers.

  The primitive is no-op at runtime, however its goal is to tag (annotate) the
  Jax computation graph what expressions represents a single unique layer type.
  This is the only way for K-FAC to know how to compute the curvature matrix.
  """

  def __init__(self, name: str, num_inputs: int, num_outputs: int):
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

    Args:
      name: The name of the layer primitive.
      num_inputs: The number of inputs to the layer.
      num_outputs: The number of outputs to the layer.
    """
    super().__init__(name)
    if num_outputs > 1:
      raise NotImplementedError(
          f"Only single outputs are supported, got: num_outputs={num_outputs}.")
    self._num_outputs = num_outputs
    self._num_inputs = num_inputs

    jax.interpreters.xla.register_translation(self, self._xla_translation)  # pytype: disable=wrong-arg-types  # numpy-scalars
    jax.interpreters.ad.deflinear(self, self._transpose)
    jax.interpreters.ad.primitive_transposes[self] = self._transpose
    # This line defines how does the tag behave under vmap. It is required for
    # any primitive that can be used inside a vmap. The reason why we want to
    # allow this is two fold - one to not break user code when the tags are not
    # used at all, and two - to be able to define a network with code for a
    # single example which is the vmap-ed for a batch.
    jax_batching.primitive_batchers[self] = self._batching

  @property
  def num_outputs(self) -> int:
    """The number of outputs of the layer tag that this primitive represents."""
    return self._num_outputs

  @property
  def num_inputs(self) -> int:
    """The number of inputs of the layer tag that this primitive represents."""
    return self._num_inputs

  def split_all_inputs(
      self,
      all_inputs: Sequence[T],
  ) -> Tuple[
      Tuple[T, ...],
      Tuple[T, ...],
      Tuple[T, ...]
  ]:
    """Splits the operands of the primitive into ``(outputs, inputs, params)``."""
    outputs = tuple(all_inputs[:self.num_outputs])
    inputs = tuple(all_inputs[self.num_outputs:self.num_outputs +
                              self.num_inputs])
    params = tuple(all_inputs[self.num_outputs + self.num_inputs:])
    return outputs, inputs, params

  def get_outputs(self, *operands: Array, **_: Any) -> Array:
    """Extracts the ``outputs`` of a layer from the operands of the primitive."""
    outputs = self.split_all_inputs(operands)[0]
    assert self.num_outputs == len(outputs) == 1
    return outputs[0]

  def _xla_translation(
      self,
      xla_context: jax.interpreters.xla.TranslationContext,
      avals_in: Sequence[core.AbstractValue],
      avals_out: Sequence[core.AbstractValue],
      *args: jax.interpreters.xla.XlaOp,
      **_: Any,
  ) -> Tuple[Array, ...]:
    """The XLA translation rule for this primitive - returns the ``outputs`` ."""
    del xla_context, avals_in, avals_out  # not used
    # Need to return a sequence
    return (self.get_outputs(*args),)

  @classmethod
  def _transpose(
      cls,
      cotangent: Array,
      *operands: Array,
      **_: Any,
  ) -> Tuple[Union[Array, None], ...]:
    """Computes the cotangents of the operands from those of the primitive."""
    del cls  # not used
    return (cotangent,) + (None,) * (len(operands) - 1)

  def impl(self, *operands: Array, **_: Any) -> Array:
    return self.get_outputs(*operands)

  def abstract_eval(self, *operands: Array, **_: Any) -> Array:
    jax_version = (
        jax.__version_info__ if hasattr(jax, "__version_info__")
        else tuple(map(int, jax.__version__.split("."))))
    if jax_version > (0, 3, 4):
      return self.get_outputs(*operands), jax.core.no_effects  # pytype: disable=bad-return-type  # numpy-scalars
    return self.get_outputs(*operands)

  def _batching(
      self,
      batched_operands: Sequence[Array],
      batched_dims: Union[int, Tuple[int, ...]],
      **kwargs: Any
  ) -> Tuple[Array, int]:
    """Defines how the primitive behaves under :func:`jax.vmap`."""
    return self.bind(*batched_operands, **kwargs), batched_dims[0]


def generic_get_outputs(
    self: LayerTag,
    *operands: Array,
) -> Array:
  """Special logic for generic tag's ``get_outputs``."""
  # The generic tags have no `inputs` and `outputs` so instead they return just
  # the parameters.
  assert self.num_inputs == self.num_outputs == 0
  params = self.split_all_inputs(operands)[2]
  if len(params) != 1:
    raise ValueError("A generic tag can have only one parameter.")
  return params[0]


generic = LayerTag(name="generic_tag", num_inputs=0, num_outputs=0)
setattr(generic, "get_outputs",
        types.MethodType(generic_get_outputs, generic))


def register_generic(parameter: Array) -> Array:
  """Registers a generic tag around the provided parameter array."""
  return generic.bind(parameter)


dense = LayerTag(name="dense_tag", num_inputs=1, num_outputs=1)


def register_dense(
    y: Array,
    x: Array,
    w: Array,
    b: Optional[Array] = None,
    **kwargs,
) -> Array:
  """Registers a dense layer: ``y = matmul(x, w) + b``."""
  if b is None:
    return dense.bind(y, x, w, **kwargs)
  return dense.bind(y, x, w, b, **kwargs)


conv2d = LayerTag(name="conv2d_tag", num_inputs=1, num_outputs=1)


def register_conv2d(
    y: Array,
    x: Array,
    w: Array,
    b: Optional[Array] = None,
    **kwargs: Any
) -> Array:
  """Registers a 2d convolution layer: ``y = conv2d(x, w) + b``."""
  if b is None:
    return conv2d.bind(y, x, w, **kwargs)
  return conv2d.bind(y, x, w, b, **kwargs)


scale_and_shift = LayerTag(
    name="scale_and_shift_tag", num_inputs=1, num_outputs=1)


def register_scale_and_shift(
    y: Array,
    x: Array,
    scale: Optional[Array] = None,
    shift: Optional[Array] = None,
) -> Array:
  """Registers a scale and shift layer: ``y = x * scale + shift``."""
  if scale is not None and shift is not None:
    args = (scale, shift)
  elif scale is not None:
    args = (scale,)
  elif shift is not None:
    args = (shift,)
  else:
    raise ValueError("At least one of `scale` and `shift` must be provided.")
  return scale_and_shift.bind(
      y, x, *args, has_scale=scale is not None, has_shift=shift is not None)


class LossTagEqn(core.JaxprEqn):
  """A class used only for annotation purposes."""
  primitive: LossTag


class LayerTagEqn(core.JaxprEqn):
  """A class used only for annotation purposes."""
  primitive: LayerTag
