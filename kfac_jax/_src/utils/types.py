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
"""K-FAC annotation types and general tree operations."""
from typing import Callable, Mapping, Sequence, TypeVar

import jax
import jax.numpy as jnp

# Types for annotation
T = TypeVar("T")
Array = jax.Array
PRNGKey = Array
Scalar = float | int
Numeric = Array | Scalar
Shape = tuple[int, ...]
DType = jax.typing.DTypeLike
PyTree = T | Sequence["PyTree[T]"] | Mapping[str, "PyTree[T]"]
ArrayTree = PyTree[Array]
TArrayTree = TypeVar("TArrayTree", bound=ArrayTree)
Params = TypeVar("Params", bound=ArrayTree)
Batch = TypeVar("Batch", bound=ArrayTree)
FuncState = TypeVar("FuncState", bound=ArrayTree)
FuncAux = dict[str, ArrayTree]
PyTreeDef = jax.tree_util.PyTreeDef
FuncArgs = Sequence[ArrayTree]
FuncOuts = Array | tuple[Array, FuncAux]
Func = Callable[..., FuncOuts]
ValueFunc = Callable[..., Array]
ValueAndGradFunc = Callable[..., tuple[Array, Params]]
AssumedFuncOutput = (Array | tuple[Array, FuncAux] |
                     tuple[Array, tuple[FuncState, FuncAux]])
SCALAR_TYPES = (float, int)
ScheduleType = (
    Callable[[Numeric, Numeric | None], Numeric] |
    Callable[[Numeric], Numeric]
    )


def tree_is_empty(obj: ArrayTree) -> bool:
  """Returns whether the given PyTree is empty."""
  return not jax.tree_util.tree_leaves(obj)


def abstract_objects_equal(
    obj1: ArrayTree,
    obj2: ArrayTree,
    check_dtype: bool = True
) -> bool:
  """`True` if the objects have the same PyTree structure, shapes and dtypes."""
  return (jax.tree_util.tree_structure(obj1) ==
          jax.tree_util.tree_structure(obj2) and
          all(e1.shape == e2.shape and (e1.dtype == e2.dtype or not check_dtype)
              for e1, e2 in zip(jax.tree_util.tree_leaves(obj1),
                                jax.tree_util.tree_leaves(obj2))))


def get_float_dtype_and_check_consistency(obj: ArrayTree) -> DType | None:
  """Checks that all leaves have the same float dtype, and returns this."""

  leaves = jax.tree_util.tree_leaves(obj)

  dtype = None

  for leaf in leaves:

    if (leaf.dtype == jnp.float16 or leaf.dtype == jnp.bfloat16
        or leaf.dtype == jnp.float32 or leaf.dtype == jnp.float64):

      if dtype is not None and leaf.dtype != dtype:
        raise ValueError("Inconsistent dtypes detected.")
      else:
        dtype = leaf.dtype

    else:
      raise ValueError("Non-float dtype detected.")

  return dtype
