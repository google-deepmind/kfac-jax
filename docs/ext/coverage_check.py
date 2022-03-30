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
"""Asserts all public symbols are covered in the docs."""
import inspect
import types
from typing import Any, Mapping, Set, Sequence, Tuple

import kfac_jax
from sphinx import application
from sphinx import builders
from sphinx import errors


def get_public_functions(
    root_module: types.ModuleType) -> Sequence[Tuple[str, types.FunctionType]]:
  """Returns `(function_name, function)` for all functions of `root_module`."""
  fns = []
  for name in dir(root_module):
    o = getattr(root_module, name)
    if inspect.isfunction(o):
      fns.append((name, o))
  return fns


def get_public_symbols(
    root_module: types.ModuleType) -> Sequence[Tuple[str, types.FunctionType]]:
  """Returns `(symbol_name, symbol)` for all symbols of `root_module`."""
  fns = []
  for name in getattr(root_module, "__all__"):
    o = getattr(root_module, name)
    fns.append((name, o))
  return fns


class CoverageCheck(builders.Builder):
  """Builder that checks all public symbols are included."""

  name = "coverage_check"

  def get_outdated_docs(self) -> str:
    return "coverage_check"

  def write(self, *ignored: Any) -> None:
    pass

  def finish(self) -> None:

    def public_symbols() -> Set[str]:
      symbols = set()
      for symbol_name, _ in get_public_symbols(kfac_jax):
        symbols.add("kfac_jax." + symbol_name)
      return symbols

    documented_objects = frozenset(self.env.domaindata["py"]["objects"])
    undocumented_objects = public_symbols() - documented_objects
    if undocumented_objects:
      undocumented_objects = tuple(sorted(undocumented_objects))
      raise errors.SphinxError(
          "All public symbols must be included in our documentation, did you "
          "forget to add an entry to `api.rst`?\n"
          f"Undocumented symbols: {undocumented_objects}.")


def setup(app: application.Sphinx) -> Mapping[str, Any]:
  app.add_builder(CoverageCheck)
  return dict(version=kfac_jax.__version__, parallel_read_safe=True)
