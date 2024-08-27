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
"""Module containing utility functions for curvature blocks."""
import collections
import string
from typing import Sequence

import jax.numpy as jnp
from kfac_jax._src import utils


# Types for annotation
Scalar = utils.Scalar
ScalarOrSequence = Scalar | Sequence[Scalar]

# Special global variables
# This is used for einsum strings
ALPHABET = string.ascii_lowercase
# The default value that would be used for the argument
# ``max_elements_for_vmap``, when it is set to ``None`` in the
# ``Conv2DDiagonal`` and ``Conv2DFull` curvature blocks.
_MAX_PARALLEL_ELEMENTS: int = 2 ** 23
# The default value that would be used for the argument
# ``eigen_decomposition_threshold``, when it is set to ``None`` in any of the
# curvature blocks that inherit from ``Full`.
_DEFAULT_EIGEN_DECOMPOSITION_THRESHOLD = 5


def set_max_parallel_elements(value: int):
  """Sets the default value of maximum parallel elements in the module.

  This value is used to determine the parallel-to-memory tradeoff in the
  curvature estimation procedure of :class:`~Conv2DDiagonal` and
  :class:`~Conv2DFull`. See their corresponding docs for further details.

  Args:
    value: The default value for maximum number of parallel elements.
  """
  global _MAX_PARALLEL_ELEMENTS
  _MAX_PARALLEL_ELEMENTS = value


def get_max_parallel_elements() -> int:
  """Returns the default value of maximum parallel elements in the module.

  This value is used to determine the parallel-to-memory tradeoff in the
  curvature estimation procedure of :class:`~Conv2DDiagonal` and
  :class:`~Conv2DFull`. See their corresponding docs for further details.

  Returns:
    The default value for maximum number of parallel elements.
  """
  return _MAX_PARALLEL_ELEMENTS


def set_default_eigen_decomposition_threshold(value: int):
  """Sets the default value of the eigen decomposition threshold.

  This value is used in :class:`~Full` to determine when updating the cache,
  at what number of different powers to switch the implementation from a simple
  matrix power to an eigenvector decomposition.

  Args:
    value: The default value for eigen decomposition threshold.
  """
  global _DEFAULT_EIGEN_DECOMPOSITION_THRESHOLD
  _DEFAULT_EIGEN_DECOMPOSITION_THRESHOLD = value


def get_default_eigen_decomposition_threshold() -> int:
  """Returns the default value of the eigen decomposition threshold.

  This value is used in :class:`~Full` to determine when updating the cache,
  at what number of different powers to switch the implementation from a simple
  matrix power to an eigenvector decomposition.

  Returns:
    The default value of the eigen decomposition threshold.
  """
  return _DEFAULT_EIGEN_DECOMPOSITION_THRESHOLD


def to_real_set(
    number_or_sequence: ScalarOrSequence | None
) -> set[Scalar]:
  """Converts the optional number or sequence to a set."""
  if number_or_sequence is None:
    return set()
  elif isinstance(number_or_sequence, set):
    return number_or_sequence
  elif isinstance(number_or_sequence, (float, int)):
    return {number_or_sequence}
  elif (isinstance(number_or_sequence, collections.abc.Sequence) and
        all(isinstance(x, (int, float)) for x in number_or_sequence)):
    return set(number_or_sequence)
  else:
    raise ValueError(f"Expecting a real-number or a sequence of reals, but got "
                     f"{type(number_or_sequence)}.")


def compatible_shapes(ref_shape, target_shape):

  if len(target_shape) > len(ref_shape):
    raise ValueError("Target shape should be smaller.")

  for ref_d, target_d in zip(reversed(ref_shape), reversed(target_shape)):
    if ref_d != target_d and target_d != 1:
      raise ValueError(f"{target_shape} is incompatible with {ref_shape}.")


def compatible_sum(tensor, target_shape, skip_axes):
  """Compute sum over ``tensor`` to achieve shape given by ``target_shape``."""

  compatible_shapes(tensor.shape, target_shape)

  n = tensor.ndim - len(target_shape)

  axis = [i + n for i, t in enumerate(target_shape)
          if t == 1 and i + n not in skip_axes]

  tensor = jnp.sum(tensor, axis=axis, keepdims=True)

  axis = [i for i in range(tensor.ndim - len(target_shape))
          if i not in skip_axes]

  return jnp.sum(tensor, axis=axis)
