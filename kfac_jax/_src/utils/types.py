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
from typing import Any, Callable, Sequence, TypeVar, Union, Tuple

import chex
import jax

# Types for annotation
T = TypeVar("T")
Params = chex.ArrayTree
Batch = chex.ArrayTree
FuncState = chex.ArrayTree
FuncAux = chex.ArrayTree
PyTreeDef = chex.PyTreeDef
PyTreeType = Any
PyTree = chex.ArrayTree
TPyTree = TypeVar("TPyTree", bound=PyTree)
FuncArgs = Sequence[PyTree]
Func = Callable[..., Union[chex.Array, Tuple[chex.Array, FuncAux]]]
ValueFunc = Callable[..., chex.Array]
ValueAndGradFunc = Callable[..., Tuple[chex.Array, Params]]

AssumedFuncOutput = Union[
    chex.Array,
    Tuple[chex.Array, FuncAux],
    Tuple[chex.Array, Tuple[FuncState, FuncAux]],
]

CHEX_SCALAR_TYPES = (float, int)


def tree_is_empty(obj: PyTree) -> bool:
  """Returns whether the given PyTree is empty."""
  return not jax.tree_util.tree_leaves(obj)


def abstract_objects_equal(
    obj1: PyTree,
    obj2: PyTree,
    check_dtype: bool = True
) -> bool:
  """`True` if the objects have the same PyTree structure, shapes and dtypes."""
  return (jax.tree_util.tree_structure(obj1) ==
          jax.tree_util.tree_structure(obj2) and
          all(e1.shape == e2.shape and (e1.dtype == e2.dtype or not check_dtype)
              for e1, e2 in zip(jax.tree_util.tree_leaves(obj1),
                                jax.tree_util.tree_leaves(obj2))))
