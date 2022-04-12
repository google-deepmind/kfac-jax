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
from typing import Any, Generic, Optional, Sequence, Tuple, Type, TypeVar, Union

import chex
import jax
from jax import core
from jax.interpreters import batching as jax_batching

# Types for annotation
T = TypeVar("T")
ArrayOrXLA = TypeVar("ArrayOrXLA", chex.Array, jax.xla.XlaOp)


class LossTag(core.Primitive, Generic[T]):
  """A Jax primitive for tagging K-FAC losses.

  The primitive is no-op at runtime, however its goal is to tag (annotate) the
  Jax computation graph what expression exactly is the loss and what type of
  loss it represents. This is the only way for K-FAC to know how to compute the
  curvature matrix.
  """

  # Whether the primitive returns multiple outputs (from core.Primitive)
  multiple_results = True

  def __init__(self, cls: Type[T], num_inputs: int, num_targets: int = 1):
    """Initializes a loss tag primitive for the given :class:`~LossFunction` class.

    When the primitive is created, the constructor automatically registers it
    with the standard Jax machinery for differentiation, :func:`jax.vmap` and
    XLA lowering. For further details see please take a look at the JAX
    documentation on `primitives
    <https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html>`__.

    Args:
      cls: The corresponding class of :class:`~LossFunction` that this tag
        represents.
      num_inputs: The number of inputs to the tag primitive that are assumed
        to be parameter dependent.
      num_targets: The number of inputs to the tag primitive, which are assumed
        to be parameter **independent**.
    """
    super().__init__(cls.__name__ + "_tag")
    self._cls = cls
    self._num_inputs = num_inputs
    self._num_targets = num_targets

    jax.xla.register_translation(self, self._xla_translation)
    jax.ad.primitive_jvps[self] = self._jvp
    # This line defines how does the tag behave under vmap. It is required for
    # any primitive that can be used inside a vmap. The reason why we want to
    # allow this is two fold - one to not break user code when the tags are not
    # used at all, and two - to be able to define a network with code for a
    # single example which is the vmap-ed for a batch.
    jax_batching.primitive_batchers[self] = self._batching

  @property
  def num_inputs(self) -> int:
    """The number of parameter dependent inputs to the tag primitive."""
    return self._num_inputs

  @property
  def num_targets(self) -> int:
    """The number of parameter **independent** inputs to the tag primitive."""
    return self._num_targets

  def loss(self, *args: chex.Array, **kwargs: Any) -> T:
    """Constructs an instance of the corresponding :class:`~LossFunction` class."""
    return self._cls(*args, **kwargs)

  def get_outputs(
      self,
      *args: ArrayOrXLA
  ) -> Tuple[ArrayOrXLA, ...]:
    """Verifies that the number of arguments matches expectations."""
    if len(args) < self.num_inputs:
      raise ValueError("Inputs to the tag are not enough.")
    if len(args) > self.num_inputs + self.num_targets:
      raise ValueError("Inputs to the tag are too many.")
    if self.num_inputs < len(args) < self.num_inputs + self.num_targets:
      raise ValueError("Inputs to the tag are not quite enough.")
    return args

  def impl(self, *operands: chex.Array, **_: Any) -> Tuple[chex.Array, ...]:
    return self.get_outputs(*operands)

  def abstract_eval(self, *operands: chex.Array, **_) -> Tuple[chex.Array, ...]:
    jax_version = (
        jax.__version_info__ if hasattr(jax, "__version_info__")
        else tuple(map(int, jax.__version__.split("."))))
    if jax_version > (0, 3, 4):
      return self.get_outputs(*operands), jax.core.no_effects
    return self.get_outputs(*operands)

  def _xla_translation(
      self,
      xla_context: jax.xla.TranslationContext,
      avals_in: Sequence[core.AbstractValue],
      avals_out: Sequence[core.AbstractValue],
      *args: jax.xla.XlaOp,
      **_: Any,
  ) -> Tuple[jax.xla.XlaOp, ...]:
    """The XLA translation rule for this primitive (creates a no-op Tuple)."""
    del avals_in, avals_out  # not used
    return self.get_outputs(*args)

  def _jvp(
      self,
      arg_values: Sequence[chex.Array],
      arg_tangents: Sequence[chex.Array],
      **kwargs: Any,
  ) -> Tuple[Tuple[chex.Array, ...], Tuple[chex.Array, ...]]:
    """Computes the Jacobian-vector product for the primitive."""
    if len(arg_values) != len(arg_tangents):
      raise ValueError("Values and tangents are not the same length.")
    primal_output = self.bind(*arg_values, **kwargs)
    return primal_output, tuple(arg_tangents)

  def _batching(
      self,
      batched_args: Sequence[chex.Array],
      batched_dims: Union[int, Tuple[int, ...]],
      **kwargs: Any
  ) -> Tuple[chex.Array, Union[int, Tuple[int, ...]]]:
    """Defines how the primitive behaves under :func:`jax.vmap`."""
    return self.bind(*batched_args, **kwargs), batched_dims


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

    jax.xla.register_translation(self, self._xla_translation)
    jax.ad.deflinear(self, self._transpose)
    jax.ad.primitive_transposes[self] = self._transpose
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

  def get_outputs(self, *operands: chex.Array, **_: Any) -> chex.Array:
    """Extracts the ``outputs`` of a layer from the operands of the primitive."""
    outputs = self.split_all_inputs(operands)[0]
    assert self.num_outputs == len(outputs) == 1
    return outputs[0]

  def _xla_translation(
      self,
      xla_context: jax.xla.TranslationContext,
      avals_in: Sequence[core.AbstractValue],
      avals_out: Sequence[core.AbstractValue],
      *args: jax.xla.XlaOp,
      **_: Any,
  ) -> Tuple[chex.Array, ...]:
    """The XLA translation rule for this primitive - returns the ``outputs`` ."""
    del xla_context, avals_in, avals_out  # not used
    # Need to return a sequence
    return (self.get_outputs(*args),)

  @classmethod
  def _transpose(
      cls,
      cotangent: chex.Array,
      *operands: chex.Array,
      **_: Any,
  ) -> Tuple[Union[chex.Array, None], ...]:
    """Computes the cotangents of the operands from those of the primitive."""
    del cls  # not used
    return (cotangent,) + (None,) * (len(operands) - 1)

  def impl(self, *operands: chex.Array, **_: Any) -> chex.Array:
    return self.get_outputs(*operands)

  def abstract_eval(self, *operands: chex.Array, **_: Any) -> chex.Array:
    jax_version = (
        jax.__version_info__ if hasattr(jax, "__version_info__")
        else tuple(map(int, jax.__version__.split("."))))
    if jax_version > (0, 3, 4):
      return self.get_outputs(*operands), jax.core.no_effects
    return self.get_outputs(*operands)

  def _batching(
      self,
      batched_operands: Sequence[chex.Array],
      batched_dims: Union[int, Tuple[int, ...]],
      **kwargs: Any
  ) -> Tuple[chex.Array, int]:
    """Defines how the primitive behaves under :func:`jax.vmap`."""
    return self.bind(*batched_operands, **kwargs), batched_dims[0]


def generic_get_outputs(
    self: LayerTag,
    *operands: chex.Array,
) -> chex.Array:
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


def register_generic(parameter: chex.Array) -> chex.Array:
  """Registers a generic tag around the provided parameter array."""
  return generic.bind(parameter)


dense = LayerTag(name="dense_tag", num_inputs=1, num_outputs=1)


def register_dense(
    y: chex.Array,
    x: chex.Array,
    w: chex.Array,
    b: Optional[chex.Array] = None,
    **kwargs,
) -> chex.Array:
  """Registers a dense layer: ``y = matmul(x, w) + b``."""
  if b is None:
    return dense.bind(y, x, w, **kwargs)
  return dense.bind(y, x, w, b, **kwargs)


conv2d = LayerTag(name="conv2d_tag", num_inputs=1, num_outputs=1)


def register_conv2d(
    y: chex.Array,
    x: chex.Array,
    w: chex.Array,
    b: Optional[chex.Array] = None,
    **kwargs: Any
) -> chex.Array:
  """Registers a 2d convolution layer: ``y = conv2d(x, w) + b``."""
  if b is None:
    return conv2d.bind(y, x, w, **kwargs)
  return conv2d.bind(y, x, w, b, **kwargs)


scale_and_shift = LayerTag(
    name="scale_and_shift_tag", num_inputs=1, num_outputs=1)


def register_scale_and_shift(
    y: chex.Array,
    x: chex.Array,
    scale: Optional[chex.Array] = None,
    shift: Optional[chex.Array] = None,
) -> chex.Array:
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
