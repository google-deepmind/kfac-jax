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
"""K-FAC functionality for auto-detecting layer tags and graph matching."""

import collections
import dataclasses
import functools
import itertools
import pprint
from typing import Any, Callable, Mapping, Sequence, Set, TypeVar

from absl import logging
import immutabledict
import jax
import jax.extend as jex

import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
from kfac_jax._src import layers_and_loss_tags as tags
from kfac_jax._src import utils
import numpy as np

jax_version = (
    jax.__version_info__ if hasattr(jax, "__version_info__")
    else tuple(map(int, jax.__version__.split("."))))

if jax_version >= (0, 5, 1):
  DebugInfo = jax.core.DebugInfo
else:
  DebugInfo = jax.core.JaxprDebugInfo  #  pytype: disable=module-attr


HIGHER_ORDER_NAMES = ("cond", "while", "scan", "pjit", "xla_call", "xla_pmap")
ITERATIVE_HIGHER_ORDER_NAMES = ("while", "scan")

# Types for annotation
Array = utils.Array
PyTreeDef = utils.PyTreeDef
Var = jex.core.Var
Vars = Sequence[Var]
Jaxpr = jex.core.Jaxpr
ClosedJaxpr = jex.core.ClosedJaxpr
JaxprEqn = jex.core.JaxprEqn
JaxprEqns = Sequence[JaxprEqn]
T = TypeVar("T")
J = TypeVar("J", Jaxpr, ClosedJaxpr)
JaxprOrClosedJaxpr = Jaxpr | ClosedJaxpr
EquivalenceFunction = Callable[[JaxprEqn, JaxprEqn], bool]
MakeVarFunc = Callable[[jax.core.AbstractValue], Var]
VarProcessor = Callable[[Vars, MakeVarFunc], tuple[Vars, JaxprEqns]]
PatternComputeFunc = Callable[[Array, Sequence[Array]], Array]
ParameterExtractorFunc = Callable[[JaxprEqns], Mapping[str, Any]]
TagCtor = Callable[[Vars, Vars, JaxprEqns, MakeVarFunc], JaxprEqn]


def eval_jaxpr_eqn(eqn: JaxprEqn, in_values: list[T]) -> list[T]:
  """Computes the outputs of the given Jaxpr equation."""

  subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)

  user_context = jex.source_info_util.user_context

  with user_context(eqn.source_info.traceback):
    output = eqn.primitive.bind(*subfuns, *in_values, **bind_params)

  if not isinstance(output, list):
    return [output]
  else:
    return output


def reshape_equivalent(
    equation1: JaxprEqn,
    equation2: JaxprEqn,
) -> bool:
  """Equivalence rule for :func:`~jax.numpy.reshape` primitives."""
  if not (equation1.primitive.name == "reshape" and
          equation2.primitive.name == "reshape"):
    raise ValueError("This is only applicable to `reshape` primitive.")

  return equation1.params["dimensions"] == equation2.params["dimensions"]


def broadcast_in_dim_equivalent(
    equation1: JaxprEqn,
    equation2: JaxprEqn,
) -> bool:
  """Equivalence rule for :func:`~jax.numpy.broadcast` primitives."""
  if not (equation1.primitive.name == "broadcast_in_dim" and
          equation2.primitive.name == "broadcast_in_dim"):
    raise ValueError("This is only applicable to `broadcast_in_dim` primitive.")
  return True


def conv_general_dilated_equivalent(
    equation1: JaxprEqn,
    equation2: JaxprEqn,
) -> bool:
  """Equivalence rule for :func:`~jax.lax.conv_general_dilated` primitives."""
  if not (equation1.primitive.name == "conv_general_dilated" and
          equation2.primitive.name == "conv_general_dilated"):
    raise ValueError("This is only applicable to `conv_general_dilated` "
                     "primitive.")
  params1 = equation1.params
  params2 = equation2.params
  for k in ("window_strides", "padding",
            "lhs_dilation", "rhs_dilation"):
    if len(params1[k]) != len(params2[k]):
      return False
  # pytype: disable=attribute-error
  if (len(params1["dimension_numbers"].lhs_spec) !=
      len(params2["dimension_numbers"].lhs_spec)):
    return False
  if (len(params1["dimension_numbers"].rhs_spec) !=
      len(params2["dimension_numbers"].rhs_spec)):
    return False
  if (len(params1["dimension_numbers"].out_spec) !=
      len(params2["dimension_numbers"].out_spec)):
    return False
  if ((params1["feature_group_count"] > 1) !=
      (params2["feature_group_count"] > 1)):
    return False
  if ((params1["batch_group_count"] > 1) !=
      (params2["batch_group_count"] > 1)):
    return False
  # pytype: enable=attribute-error
  return True


def dot_general_equivalent(
    equation1: JaxprEqn,
    equation2: JaxprEqn,
) -> bool:
  if not (equation1.primitive.name == "dot_general" and
          equation2.primitive.name == "dot_general"):
    raise ValueError("This is only applicable to `dot_general_equivalent` "
                     "primitive.")
  # We ignore precision and preferred_element_type
  return (equation1.params["dimension_numbers"] ==
          equation2.params["dimension_numbers"])


DEFAULT_SPECIAL_EQUIVALENCE_RULES = immutabledict.immutabledict({
    "reshape": reshape_equivalent,
    "broadcast_in_dim": broadcast_in_dim_equivalent,
    "conv_general_dilated": conv_general_dilated_equivalent,
    "dot_general": dot_general_equivalent,
})


class GraphMatcherComparator:
  """A class to compare and determine equivalence of abstract Jax equations."""

  def __init__(
      self,
      commutative_ops_names: Sequence[str] = ("add", "mul"),
      special_eqn_equivalence_rules:
      Mapping[str, EquivalenceFunction] = DEFAULT_SPECIAL_EQUIVALENCE_RULES,
  ):
    """Initializes the instance.

    Args:
      commutative_ops_names: A sequence of all Jax primitive names, which are
        consider commutative ops and the order of their arguments is irrelevant.
      special_eqn_equivalence_rules: A mapping of a Jax primitive names to a
        comparison rule, which to be used instead of the default comparator,
        which looks that the whole dictionaries of extra parameters to the
        primitives match.
    """
    self._commutative_ops_names = set(commutative_ops_names)
    self._special_eqn_equivalence_rules = dict(**special_eqn_equivalence_rules)

  @property
  def commutative_ops_names(self) -> set[str]:
    """The set of commutative ops."""
    return self._commutative_ops_names

  @property
  def special_eqn_equivalence_rules(self) -> Mapping[str, EquivalenceFunction]:
    """The special equivalence rules."""
    return self._special_eqn_equivalence_rules

  def add_commutative_op_name(self, name: str):
    """Adds a name to the set of primitive ops considered to be commutative."""
    if name in self.commutative_ops_names:
      raise ValueError(f"Commutative op {name!r} has already been added.")
    self._commutative_ops_names.add(name)

  def add_special_equivalence_rule(
      self,
      name: str,
      equivalence_rule: EquivalenceFunction,
  ):
    """Adds the special equivalence rule for ``name`` to the global store."""
    if name in self.special_eqn_equivalence_rules:
      raise ValueError(
          f"Special equation equivalence rule already exists for name: {name}")
    self._special_eqn_equivalence_rules[name] = equivalence_rule

  def are_equivalent(
      self,
      equation1: JaxprEqn,
      equation2: JaxprEqn,
  ) -> bool:
    """Returns whether the two equations are considered equivalent."""

    if equation1.primitive.name != equation2.primitive.name:
      return False

    equivalence_rule = self.special_eqn_equivalence_rules.get(
        equation1.primitive.name)

    if equivalence_rule is not None:
      return equivalence_rule(equation1, equation2)

    # Default comparison
    return equation1.params == equation2.params


@dataclasses.dataclass(frozen=True)
class JaxprGraph:
  """A wrapper around Jaxpr as a graph for pattern matching.

  Attributes:
    name: The name for this Jaxpr graph.
    closed_jaxpr: The original `ClosedJaxpr` that is being wrapped.
    params_tree: The PyTreeDef of the parameter variables.
    params_vars: A flat list of all the abstract parameter variables.
    out_tree: The PyTreeDef of the outputs of the function.
    tag_ctor: This is an optional attribute, that defines if this is used during
      automatic layer tag registration, how to construct the corresponding layer
      tag primitive from the subgraph matching this pattern.
    losses_eqns: A tuple of all the Jaxpr equations corresponding to a loss
      tag.
    var_to_creation_op: A mapping of variables to the Jax equation that created
      it.
    manual_registrations: Any layer tag equations that have been manually
      registered.
    jaxpr: The underlying :class:`Jaxpr` part of ``self.closed_jaxpr``.
    consts: The underlying constants part ``self.closed_jaxpr``.
    outvars: The output variables of the underlying :class:`Jaxpr` part
      of ``self.closed_jaxpr``.
  """
  name: str
  closed_jaxpr: ClosedJaxpr
  params_tree: PyTreeDef
  params_vars: Vars
  out_tree: PyTreeDef
  tag_ctor: TagCtor | None

  @property
  def jaxpr(self) -> Jaxpr:
    return self.closed_jaxpr.jaxpr

  @property
  def consts(self) -> Sequence[Any]:
    return self.closed_jaxpr.consts

  @property
  def outvars(self) -> Vars:
    return self.jaxpr.outvars  # pytype:disable=bad-return-type

  def sub_graph_eqns(
      self,
      root_vars: Sequence[Var],
      leaf_vars: Sequence[Var],
  ) -> JaxprEqns:
    """Returns the sub-graph equations between root vars and leaf vars."""

    eqns = []
    # Extract the subgraph equations such that they both depend on root_vars and
    # leaf_vars depends on them

    if any(v in self.params_vars for v in leaf_vars):
      # The special case of a generic tag, where the output is a parameter
      assert all(v in self.params_vars for v in leaf_vars)
      return ()

    to_process_eqns = [self.var_to_creation_op[v] for v in leaf_vars]
    processed_vars = set()

    while to_process_eqns:

      next_eqn = to_process_eqns.pop()
      eqns.append(next_eqn)

      for v in next_eqn.invars:
        if (not isinstance(v, jex.core.Literal) and v not in root_vars and
            v not in processed_vars and v in self.var_to_creation_op):
          to_process_eqns.append(self.var_to_creation_op[v])
          processed_vars.add(v)

    return tuple(eqns)

  @functools.cached_property
  def losses_eqns(self) -> tuple[tags.LossTagEqn, ...]:
    # Note that this function won't look inside higher order primitives of this
    # graph to find loss tags.
    return tuple(
        eqn for eqn in self.closed_jaxpr.jaxpr.eqns
        if isinstance(eqn.primitive, tags.LossTag)
    )

  @functools.cached_property
  def var_to_creation_op(self) -> immutabledict.immutabledict:
    return immutabledict.immutabledict(
        sum(([(var, eqn) for var in eqn.outvars]
             for eqn in self.jaxpr.eqns), []))

  @functools.cached_property
  def manual_registrations(self) -> tuple[tags.LayerTagEqn, ...]:
    """Returns all manually registered tags."""

    # Note that this function won't look inside higher order primitives of this
    # graph to find layer tags.

    registered_tags = []

    for eqn in self.jaxpr.eqns:

      if isinstance(eqn.primitive, tags.LayerTag):

        for param in tags.layer_eqn_data(eqn).params:
          if param not in self.params_vars:
            raise ValueError("One of the parameters of the manual layer "
                             f"registration equation: {eqn} is not part of "
                             "the parameters of the global function.")

        registered_tags.append(eqn)

    return tuple(registered_tags)


def make_jax_graph(
    func: utils.Func,
    func_args: utils.FuncArgs,
    params_index: int | Sequence[int],
    name: str,
    compute_only_loss_tags: bool,
    clean_broadcasts: bool,
    tag_ctor: TagCtor | None = None,
) -> JaxprGraph:
  """Creates a :class:`~JaxGraph` instance from the provided function and arguments."""

  in_tree = jax.tree_util.tree_structure(func_args)
  closed_jaxpr, out_shapes = jax.make_jaxpr(func, return_shape=True)(*func_args)

  if compute_only_loss_tags:

    make_var_func = jax.core.gensym()
    eqns = []
    sub_graph_vars = set()
    loss_tags_output_vars = []

    for eqn in reversed(closed_jaxpr.jaxpr.eqns):

      if (isinstance(eqn.primitive, tags.LossTag) or
          any(v in sub_graph_vars for v in eqn.outvars)):

        if isinstance(eqn.primitive, tags.LossTag):

          new_out_vars = []
          for v in eqn.outvars:

            if isinstance(v, jax.core.DropVar):
              new_out_vars.append(make_var_func(v.aval))
            else:
              new_out_vars.append(v)

          loss_tags_output_vars.extend(new_out_vars[::-1])
          eqns.append(eqn.replace(outvars=new_out_vars))

        else:
          eqns.append(eqn)

        sub_graph_vars.update(
            v for v in eqn.invars if not isinstance(v, jex.core.Literal)
        )

    consts_i = [
        i
        for i, c in enumerate(closed_jaxpr.jaxpr.constvars)
        if c in sub_graph_vars
    ]

    debug_info = closed_jaxpr.jaxpr.debug_info
    if debug_info is not None:
      debug_info = DebugInfo(
          debug_info.traced_for,
          debug_info.func_src_info,
          debug_info.arg_names,
          tuple([f"{i}" for i in range(len(loss_tags_output_vars))]),
      )

    closed_jaxpr = ClosedJaxpr(
        jaxpr=closed_jaxpr.jaxpr.replace(
            eqns=eqns[::-1],
            constvars=[closed_jaxpr.jaxpr.constvars[i] for i in consts_i],
            outvars=loss_tags_output_vars[::-1],
            debug_info=debug_info,
        ),
        consts=[closed_jaxpr.consts[i] for i in consts_i],
    )
    out_shapes = [jax.ShapeDtypeStruct(shape=v.aval.shape, dtype=v.aval.dtype)
                  for v in closed_jaxpr.jaxpr.outvars]  # pytype:disable=attribute-error

  closed_jaxpr = clean_jaxpr(closed_jaxpr)

  if clean_broadcasts:
    closed_jaxpr = merge_broadcasts_jaxpr(closed_jaxpr)
    closed_jaxpr = clean_jaxpr(closed_jaxpr)

  in_vars = jax.tree_util.tree_unflatten(in_tree, closed_jaxpr.jaxpr.invars)

  if isinstance(params_index, int):
    params_vars = in_vars[params_index]
  else:
    params_vars = tuple(in_vars[i] for i in params_index)

  params_vars, params_tree = jax.tree_util.tree_flatten(params_vars)

  return JaxprGraph(
      name=name,
      closed_jaxpr=closed_jaxpr,
      params_tree=params_tree,
      params_vars=params_vars,
      out_tree=jax.tree_util.tree_structure(out_shapes),
      tag_ctor=tag_ctor
  )


@dataclasses.dataclass(frozen=True)
class GraphPattern:
  """A graph pattern used for automatically detecting layers.

  The graph matcher needs to trace at least once the full function, which
  means the caller needs to provide it with dummy arguments. The shapes of the
  arguments do not matter, as the graph matcher ignores their values, however
  the rank does. Especially if there is some broadcasting happening you should
  register with every possible broadcast pattern. As a general advice avoid
  using a shape to be 1, unless you want the pattern to specifically match
  that, as some operations, like squeeze for example, can have special
  behaviour then.

  Attributes:
    name: The name of the pattern that is being registered to.
    tag_primitive: The primitive tag to bind.
    compute_func: The function that performs the computation.
    parameters_extractor_func: A function that extracts from the traced Jaxpr
      any parameters that are passed into the tag.
    example_args: Example arguments that can be inputted into ``func``.
    in_values_preprocessor: A function that can optionally modify the in_vals
      passed to the tag_primitive, from those that are usually the input to
      the jaxpr.
    jaxpr: The underlying :class:`Jaxpr` represented by the pattern.
    param_vars: The list of :class:`Var` that correspond to parameters
      in the pattern.
    graph: A :class:`JaxprGraph` representation of the pattern.
  """
  name: str
  tag_primitive: tags.LayerTag
  compute_func: PatternComputeFunc
  parameters_extractor_func: ParameterExtractorFunc
  example_args: utils.FuncArgs
  in_values_preprocessor: VarProcessor | None = None

  @property
  def jaxpr(self) -> Jaxpr:
    return self.graph.jaxpr

  @property
  def param_vars(self) -> Vars:
    return self.graph.params_vars

  @functools.cached_property
  def graph(self) -> JaxprGraph:
    """A :class:`JaxprGraph` representation of the pattern."""
    jnp_args = jax.tree_util.tree_map(jnp.asarray, self.example_args)
    return make_jax_graph(
        func=self.compute_func,
        func_args=jnp_args,
        params_index=1,
        name=self.name,
        compute_only_loss_tags=False,
        clean_broadcasts=True,
    )

  def tag_ctor(
      self,
      in_vars: Vars,
      out_vars: Vars,
      graph_eqns: JaxprEqns,
      make_var_func: MakeVarFunc,
  ) -> JaxprEqns:
    """Registers the layer tag for this graph pattern.

    Args:
      in_vars: The input variables to the pattern.
      out_vars: The output variables to the pattern.
      graph_eqns: The real graph equations corresponding to the pattern.
      make_var_func: A function to create correctly new variables.
    Returns:
      A sequence of any additional equations that are created from creating the
      tag.
    """
    assert len(out_vars) == 1

    if self.in_values_preprocessor is not None:
      in_vars, eqns = self.in_values_preprocessor(in_vars, make_var_func)
    else:
      eqns = []

    new_out_vars = [make_var_func(v.aval) for v in out_vars]

    tag_eqn = jax.core.new_jaxpr_eqn(
        invars=[*out_vars, *in_vars],
        outvars=new_out_vars,
        primitive=tags.layer_tag,
        params=self.parameters_extractor_func(graph_eqns),
        effects=set(),
    )

    return [*eqns, tag_eqn]


@dataclasses.dataclass(frozen=True)
class GraphMatch:
  """Represents a match of the pattern on some graph.

  Attributes:
    pattern: The pattern that has been matched.
    variables_map: Mapping of variables from the pattern to the original graph,
      on which it has been matched.
    graph_eqns: All the equations in the original graph, that correspond to
      computation of the pattern.
    output_var: The variable in the original graph, that correspond to the
      output variable of the pattern.
    param_graph_variables: All variables in the original graph, that correspond
      to parameters of the pattern.
    name: The name of the pattern that has been matched.
  """
  pattern: GraphPattern
  variables_map: Mapping[Var, Var]
  graph_eqns: JaxprEqns

  @property
  def name(self) -> str:
    return self.pattern.name

  @functools.cached_property
  def output_var(self) -> Var:
    return self.variables_map[self.pattern.jaxpr.outvars[0]]

  @functools.cached_property
  def param_graph_variables(self) -> Vars:
    return [self.variables_map[p] for p in self.pattern.graph.params_vars]

  def create_eqn(
      self,
      env: dict[Var, Var],
      make_var_func: MakeVarFunc,
  ) -> JaxprEqns:
    """Creates a new ``JaxprEqn`` for this match."""

    in_vars = [self.variables_map[k] for k in self.pattern.graph.jaxpr.invars]
    in_vars = [env.get(v, v) for v in in_vars]
    out_vars = [self.variables_map[k]
                for k in self.pattern.graph.jaxpr.outvars]
    out_vars = [env.get(v, v) for v in out_vars]

    eqns = self.pattern.tag_ctor(
        in_vars, out_vars, self.graph_eqns, make_var_func)
    assert len(out_vars) == len(eqns[-1].outvars)

    # Reinsert the output in the environment
    for k, v in zip(out_vars, eqns[-1].outvars):
      env[k] = v

    return eqns


def match_equations(
    graph: JaxprGraph,
    current_variables_map: Mapping[Var, Var],
    reversed_eqns_to_match: Sequence[JaxprEqn],
    input_vars: Vars,
    param_variables: Vars,
    graph_matcher_rules: GraphMatcherComparator,
    matchable_graph_params: Set[Var],
) -> dict[Var, Var] | None:
  """Tries to continue matching the remaining equations to the Jaxpr graph.

  Args:
    graph: The :class:`~JaxprGraph` on which we are searching for matching
      equations.
    current_variables_map: A mapping from a pattern variables to graph
      variables, which describes what is the current partial mapping between
      the pattern and the graph.
    reversed_eqns_to_match: The remaining equations of the pattern that have
      not yet been matched to the graph.
    input_vars: The input variables of the pattern.
    param_variables: The parameter variables of the pattern.
    graph_matcher_rules: A :class:`~GraphMatcherRules` instance, which is used
      for determining equivalence of individual Jax primitives.
    matchable_graph_params: A subset of graph.params_vars consisting of
      parameters that may appear in matches as parameters (not merely input
      variables).

  Returns:
    ``None`` if it is not possible to finish matching the remaining equations
    in the graph. Otherwise, returns the full match of the pattern onto the
    graph, in terms of a variable to variable mapping.
  """

  # Copy the variables mapping
  current_variables_map = dict(current_variables_map)

  def add_vars_if_possible(
      eqn_vars: Sequence[Var],
      graph_vars: Sequence[Var]
  ) -> bool:
    """Tries to update the current variables map.

    If at least one of the pattern variables is a parameter, but the
    corresponding graph variable is not or vise-versa, the method does not
    update the current variables map and returns ``False``. Similarly, if at
    least one of the graph variables is a :class:`iteral` (meaning a
    constant, independent of the function inputs) and the corresponding
    pattern variable is not an input to the pattern, it returns ``False``. In
    all other cases it updates the map and returns ``True``.

    Args:
      eqn_vars: The variables from a single equation of the pattern.
      graph_vars: The variables from a corresponding equation of the graph.

    Returns:
      A boolean describing whether the method succeeded to update the
      current variables map.
    """
    for var1, var2 in zip(eqn_vars, graph_vars):

      var2_matchable = isinstance(var2, jex.core.Var) and (
          var2 in matchable_graph_params)

      if (var1 in param_variables and not var2_matchable or
          var1 not in param_variables and var2_matchable or
          (isinstance(var2, jex.core.Literal) and var1 not in input_vars)):
        return False

    current_variables_map.update(zip(eqn_vars, graph_vars))

    return True

  # Loop over all remaining equations to match
  for i, eqn in enumerate(reversed_eqns_to_match):

    assert all(v in current_variables_map for v in eqn.outvars)

    # Retrieve the graph equation, whose output currently corresponds to the
    # first output variable of the pattern equation.
    first_output_var = current_variables_map[eqn.outvars[0]]
    graph_eqn = graph.var_to_creation_op.get(first_output_var)

    if graph_eqn is None:
      assert first_output_var in (graph.jaxpr.invars + graph.jaxpr.constvars)
      # Clearly the pattern equation is not an input or parameter
      return None

    assert isinstance(graph_eqn, JaxprEqn)

    # For equations with more than one output, make sure all output variables
    # in the graph are generated from the same graph equation.
    for v in eqn.outvars[1:]:
      if graph_eqn != graph.var_to_creation_op.get(current_variables_map[v]):
        return None

    # Check that the graph and pattern equation are equivalent
    if not graph_matcher_rules.are_equivalent(graph_eqn, eqn):
      return None

    # Sanity check
    assert len(eqn.invars) == len(graph_eqn.invars)

    if eqn.primitive.name in graph_matcher_rules.commutative_ops_names:

      # For commutative ops we search through all possible pair alignments.
      # This requires a recursive solution, on top of the iterative one.
      results = []
      for permutation in itertools.permutations(range(len(eqn.invars))):

        pattern_vars = [eqn.invars[j] for j in permutation]

        # Check if this ordering is feasible
        if not add_vars_if_possible(pattern_vars, graph_eqn.invars):
          continue

        # Recursively continue by trying to match the remaining equations.
        candidate_map = match_equations(
            graph=graph,
            current_variables_map=current_variables_map,
            reversed_eqns_to_match=reversed_eqns_to_match[i + 1:],
            input_vars=input_vars,
            param_variables=param_variables,
            graph_matcher_rules=graph_matcher_rules,
            matchable_graph_params=matchable_graph_params,
        )

        if candidate_map is not None:
          # Sanity check
          assert all(candidate_map[p] in matchable_graph_params
                     for p in param_variables)
          results.append(candidate_map)

      # Return appropriately
      if len(results) > 1:
        raise ValueError("Found multiple branch matches in pattern at "
                         f"associative op {eqn.primitive.name}.")
      elif len(results) == 1:
        return results[0]
      else:
        return None

    elif not add_vars_if_possible(eqn.invars, graph_eqn.invars):
      # In the case where we can't update the current variables map directly
      # return
      return None

  return current_variables_map


def match_pattern(
    graph: JaxprGraph,
    root_eqn: JaxprEqn,
    pattern: GraphPattern,
    graph_matcher_rules: GraphMatcherComparator,
    matchable_graph_params: Set[Var],
) -> GraphMatch | None:
  """Tries to match the ``pattern`` in the Jaxpr graph from the ``root_eqn``.

  Args:
    graph: The :class:`~JaxprGraph` on which we are searching for matching
      equations.
    root_eqn: The equation in the graph, which is assumed to match the output
      equation of the pattern.
    pattern: The pattern, which we are trying to match.
    graph_matcher_rules: A :class:`~GraphMatcherRules` instance, which is used
      for determining equivalence of individual Jax primitives.
    matchable_graph_params: A subset of graph.params_vars consisting of
      parameters that may appear in matches as parameters (not merely input
      variables).

  Returns:
    The variable to variable mapping between the pattern and graph variable,
    if the pattern can be matched to the root equation, otherwise ``None``.
  """

  # Check the number of output variables match.
  if len(pattern.jaxpr.outvars) != len(root_eqn.outvars):
    return None

  # Set the current variables mapping to the output variables and then try to
  # check the match from there.
  match_variables_map = match_equations(
      graph=graph,
      current_variables_map=dict(zip(pattern.jaxpr.outvars,
                                     root_eqn.outvars)),
      reversed_eqns_to_match=tuple(reversed(pattern.jaxpr.eqns)),
      input_vars=pattern.jaxpr.invars,
      param_variables=pattern.param_vars,
      graph_matcher_rules=graph_matcher_rules,
      matchable_graph_params=matchable_graph_params,
  )

  if match_variables_map is None:
    return None

  # Extract all the graph equations corresponding to the pattern.
  graph_eqns = []
  for k, v in match_variables_map.items():

    if (k not in pattern.graph.jaxpr.invars and
        not isinstance(v, jex.core.Literal)):

      creation_op = graph.var_to_creation_op[v]

      assert isinstance(creation_op, JaxprEqn)

      graph_eqns.append(creation_op)

  return GraphMatch(
      pattern=pattern,
      variables_map=match_variables_map,
      graph_eqns=graph_eqns,
  )


def find_layer_tags_and_patterns(
    graph: JaxprGraph,
    eqns_for_patterns: Sequence[JaxprEqn],
    graph_matcher_rules: GraphMatcherComparator,
    graph_patterns: Sequence[GraphPattern],
    matchable_params: Set[Var],
) -> dict[Var, GraphMatch]:
  """Tries to automatically match ``patterns_to_match`` in the Jaxpr graph.

  The method returns all newly discovered matches of any pattern. Each entry has
  as a key the variable of the graph corresponding to the output of the pattern,
  while each value is a triple ``(pattern, match_map, eqns)`` where ``pattern``
  is the :class:`~JaxprGraph` of the pattern that has been matched,
  ``match_map`` is mapping the pattern variables to the corresponding graph
  variables and ``eqns`` is the sequence of all graph equations corresponding to
  the pattern equations.

  Args:
    graph: The :class:`~JaxprGraph` on which we are searching for matching
      equations.
    eqns_for_patterns: All equation that should be considered for finding
      a pattern.
    graph_matcher_rules: A :class:`~GraphMatcherRules` instance, which is used
      for determining equivalence of individual Jax primitives.
    graph_patterns: A sequence of :class:`~GraphPattern` objects, which contain
      all patterns to use, in order of precedence, which to try to find in the
      graph before registering a parameter with a generic layer tag.
    matchable_params: A subset of graph.params_vars consisting of parameters
      that may appear in matches as parameters (not merely input variables).

  Returns:
    See above.
  """

  # This list keeps track to any equations that are already in a pattern and
  # hence should not be part of any other.
  registered_equations = []

  # First add any manual registrations to this.
  for eqn in graph.manual_registrations:

    layer_data = tags.layer_eqn_data(eqn)

    for manual_eqn in graph.sub_graph_eqns(
        layer_data.inputs + layer_data.params, layer_data.outputs
    ):
      registered_equations.append(manual_eqn)

  matches = {}

  # Loop through all equations in reverse, and for each one check every pattern
  for eqn in reversed(eqns_for_patterns):

    if eqn in registered_equations or eqn.primitive.name in HIGHER_ORDER_NAMES:
      continue

    for pattern in graph_patterns:

      match = match_pattern(
          graph=graph,
          root_eqn=eqn,
          pattern=pattern,
          graph_matcher_rules=graph_matcher_rules,
          matchable_graph_params=matchable_params,
      )

      if match is not None:

        # Add all the match equations to the registered equations
        registered_equations.extend(match.graph_eqns)

        # Add the match to the mapping of graph matches
        matches[match.output_var] = match

        break

  return matches


def read_env(
    env: dict[jex.core.Var, T],
    variables: list[jax.core.Atom],
) -> list[T]:
  """Reads from the variable-to-array environment during tracing."""
  result = []
  assert isinstance(variables, list)
  for v in variables:
    if isinstance(v, jex.core.Literal):
      # Literals are values baked into the Jaxpr
      result.append(v.val)
    elif isinstance(v, jax.core.DropVar):
      result.append(None)
    else:
      result.append(env[v])
  return result


def write_env(
    env: dict[jex.core.Var, T],
    variables: list[jex.core.Var],
    values: list[T],
) -> None:
  """Writes to the variable-to-array environment during tracing."""
  assert len(variables) == len(values)
  for variables, val in zip(variables, values):
    env[variables] = val


def to_closed_jaxpr(jaxpr: JaxprOrClosedJaxpr) -> ClosedJaxpr:
  if isinstance(jaxpr, Jaxpr):
    return ClosedJaxpr(jaxpr=jaxpr, consts=[])
  return jaxpr


def to_jaxpr_or_closed_jaxpr(closed_jaxpr: ClosedJaxpr, original: J) -> J:
  if isinstance(original, Jaxpr):
    return closed_jaxpr.jaxpr
  return closed_jaxpr


def apply_to_higher_order_primitives(
    eqn: JaxprEqn,
    func: Callable[[J], J]):
  """Applies `func` only to higher order Jax primitives."""

  if eqn.primitive.name not in HIGHER_ORDER_NAMES:
    return eqn

  elif eqn.primitive.name == "cond":
    params = dict(**eqn.params)
    params["branches"] = tuple(
        func(branch) for branch in params["branches"]
    )
    return eqn.replace(params=params)

  elif eqn.primitive.name == "while":
    params = dict(**eqn.params)
    params["body_jaxpr"] = func(params["body_jaxpr"])
    return eqn.replace(params=params)

  elif eqn.primitive.name in ("scan", "pjit"):
    params = dict(**eqn.params)
    params["jaxpr"] = func(params["jaxpr"])
    return eqn.replace(params=params)

  elif eqn.primitive.name in ("xla_call", "xla_pmap"):
    params = dict(**eqn.params)
    params["call_jaxpr"] = func(params["call_jaxpr"])
    return eqn.replace(params=params)

  else:
    raise NotImplementedError()


def clean_jaxpr(
    jaxpr: J,
    preserve_tags: bool = True,
    outvar_is_dep: tuple[bool, ...] | None = None,
) -> J:
  """Runs dead code elimination on a Jaxpr, retaining loss and layer tags."""

  closed_jaxpr = to_closed_jaxpr(jaxpr)
  eqns = []

  if outvar_is_dep is None:
    outvar_is_dep = (True,) * len(closed_jaxpr.jaxpr.outvars)

  final_outvars = []
  dependants = set()

  for var, is_dep in zip(closed_jaxpr.jaxpr.outvars, outvar_is_dep,
                         strict=True):
    if is_dep:

      final_outvars.append(var)

      if not isinstance(var, jex.core.Literal):
        dependants.add(var)

  for eqn in reversed(closed_jaxpr.jaxpr.eqns):

    # It's much more complicated to trace dependencies through *iterative*
    # higher order primitives, so we don't do it.
    if eqn.primitive.name in ITERATIVE_HIGHER_ORDER_NAMES:
      outvar_is_dep_for_eqn = (True,) * len(eqn.outvars)
    else:
      outvar_is_dep_for_eqn = tuple(var in dependants for var in eqn.outvars)

    # Note that we currently only trace dependencies into higher order
    # primitives, but not *through* them. If a single output of a higher order
    # primitive is a dependency, then all of its inputs are treated as such too.
    eqn = apply_to_higher_order_primitives(
        eqn,
        functools.partial(
            clean_jaxpr,
            outvar_is_dep=outvar_is_dep_for_eqn,
            preserve_tags=preserve_tags
            ),
        )

    if eqn.primitive.name in HIGHER_ORDER_NAMES:

      params = dict(**eqn.params)

      if "out_shardings" in params:
        params["out_shardings"] = utils.filter_sequence(params["out_shardings"],
                                                        outvar_is_dep_for_eqn)
      if "out_layouts" in params:
        params["out_layouts"] = utils.filter_sequence(params["out_layouts"],
                                                      outvar_is_dep_for_eqn)

      eqn = eqn.replace(
          outvars=utils.filter_sequence(eqn.outvars, outvar_is_dep_for_eqn),
          params=params,
          )

    # else:
    #   assert all(outvar_is_dep_for_eqn) or not any(outvar_is_dep_for_eqn)

    check = False

    for v in eqn.outvars:
      if v in dependants:
        dependants.remove(v)
        check = True

    if isinstance(eqn.primitive, (tags.LossTag, tags.LayerTag)):
      check = check or preserve_tags

    if check:
      eqns.append(eqn)
      new_dependants = set(v for v in eqn.invars
                           if not isinstance(v, jex.core.Literal))
      dependants = dependants.union(new_dependants)

  # Dependants should only be invars
  dependants = dependants - set(closed_jaxpr.jaxpr.invars +
                                closed_jaxpr.jaxpr.constvars)

  if dependants:
    raise ValueError("Something went wrong with the dead code elimination.")

  closed_jaxpr = ClosedJaxpr(
      jaxpr=closed_jaxpr.jaxpr.replace(eqns=list(reversed(eqns)),
                                       outvars=final_outvars),
      consts=closed_jaxpr.consts,
  )

  return to_jaxpr_or_closed_jaxpr(closed_jaxpr, jaxpr)


def clean_layer_tags_jaxpr(
    jaxpr: J,
    only_remove_auto_tags: bool = False,
) -> tuple[J, tuple[tags.LayerTagEqn | JaxprEqn, ...]]:
  """Removes layer tags from a Jaxpr."""

  closed_jaxpr = to_closed_jaxpr(jaxpr)
  eqns = []
  layer_tag_eqns = []
  var_map = {}

  for eqn in closed_jaxpr.jaxpr.eqns:
    if isinstance(eqn.primitive, tags.LayerTag) and (
        not only_remove_auto_tags
        or (
            eqn.params["meta"].name is not None
            and "Auto" in eqn.params["meta"].name
        )
    ):
      for ind1, ind2 in enumerate(eqn.params["meta"].outputs_index):
        var_map[eqn.outvars[ind1]] = eqn.invars[ind2]
    else:
      eqns.append(eqn)
    if isinstance(eqn.primitive, tags.LayerTag):
      layer_tag_eqns.append(eqn)

  def remap_input_vars(
      eqns: list[JaxprEqn], var_map: dict[jex.core.Var, jex.core.Var]
  ) -> list[JaxprEqn]:
    """Remaps the input variables of a JaxprEqn.

    Args:
      eqns: The list of JaxprEqns to remap.
      var_map: A mapping from variables to new variables.

    Returns:
      A new list of JaxprEqns with remapped input variables.
    """
    eqns_new = []
    for eqn in eqns:
      new_invars = []
      for var in eqn.invars:
        if not isinstance(var, jex.core.Literal) and var in var_map.keys():
          new_invars.append(var_map[var])
        else:
          new_invars.append(var)
      eqns_new.append(eqn.replace(invars=new_invars))
    return eqns_new

  eqns_new = remap_input_vars(eqns, var_map)
  layer_tag_eqns_new = remap_input_vars(layer_tag_eqns, var_map)

  closed_jaxpr = ClosedJaxpr(
      jaxpr=closed_jaxpr.jaxpr.replace(eqns=eqns_new),
      consts=closed_jaxpr.consts,
  )

  return (
      to_jaxpr_or_closed_jaxpr(closed_jaxpr, jaxpr),
      tuple(layer_tag_eqns_new),
  )


# Prototype for clean_jaxpr using JAX's dce_jaxpr. Doesn't work because
# dce_jaxpr will remove any equations with no used outputs, regardless of the
# dce_rule for that equation's primitive. Adding an "effect" to loss/layer
# tags also won't work, because we sometimes actually do want to remove them
# from the graph (when preserve_tags is False).
# def clean_jaxpr(
#     jaxpr: J,
#     preserve_tags: bool = True,
# ) -> J:
#   """Runs dead code elimination on a Jaxpr, retaining loss and layer tags."""

#   def dce_jaxpr_tag_rule(
#       used_outputs: list[bool],
#       eqn: JaxprEqn
#   ) -> tuple[list[bool], JaxprEqn | None]:

#     assert len(used_outputs) == len(eqn.outvars)

#     if any(used_outputs) or preserve_tags:
#       return [True] * len(eqn.invars), eqn
#     else:
#       return [False] * len(eqn.invars), None

#   closed_jaxpr = to_closed_jaxpr(jaxpr)

#   pe.dce_rules[tags.LossTag] = dce_jaxpr_tag_rule
#   pe.dce_rules[tags.LayerTag] = dce_jaxpr_tag_rule

#   cleaned_jaxpr, _ = pe.dce_jaxpr(
#       closed_jaxpr.jaxpr,
#       used_outputs=(True,) * len(closed_jaxpr.jaxpr.outvars),
#       instantiate=True)

#   pe.dce_rules.pop(tags.LossTag)
#   pe.dce_rules.pop(tags.LayerTag)

#   closed_jaxpr = ClosedJaxpr(
#       jaxpr=cleaned_jaxpr,
#       consts=closed_jaxpr.consts,
#   )

#   return to_jaxpr_or_closed_jaxpr(closed_jaxpr, jaxpr)


def merge_broadcasts_jaxpr(jaxpr: J) -> J:
  """Merges consecutive broadcasts in the given Jaxpr."""

  closed_jaxpr = to_closed_jaxpr(jaxpr)

  broadcasts_outputs = {}
  eqns = list()

  for eqn in closed_jaxpr.jaxpr.eqns:

    eqn = apply_to_higher_order_primitives(eqn, merge_broadcasts_jaxpr)

    # We ignore broadcasting of constants
    if (eqn.primitive.name == "broadcast_in_dim" and
        not all(isinstance(v, jex.core.Literal) for v in eqn.invars)):

      if eqn.invars[0] in broadcasts_outputs:
        # Construct a merged equation from the previous and current one
        prev_eqn = broadcasts_outputs[eqn.invars[0]]

        broadcasts_outputs[eqn.outvars[0]] = prev_eqn.replace(
            params={
                "shape": eqn.params["shape"],
                "broadcast_dimensions": tuple(
                    eqn.params["broadcast_dimensions"][d]
                    for d in prev_eqn.params["broadcast_dimensions"]
                ),
                "sharding": None,
            },
            outvars=eqn.outvars,
        )

      else:
        broadcasts_outputs[eqn.outvars[0]] = eqn

      if eqn.outvars[0] in closed_jaxpr.jaxpr.outvars:
        # We must preserve output equations
        eqns.append(broadcasts_outputs[eqn.outvars[0]])

    else:
      for v in eqn.invars:
        if not isinstance(v, jex.core.Literal) and v in broadcasts_outputs:
          eqns.append(broadcasts_outputs[v])

      eqns.append(eqn)

  closed_jaxpr = ClosedJaxpr(
      jaxpr=closed_jaxpr.jaxpr.replace(eqns=eqns),
      consts=closed_jaxpr.consts
  )
  return to_jaxpr_or_closed_jaxpr(closed_jaxpr, jaxpr)


def num_unique_inputs(eqns: Sequence[JaxprEqn]) -> int:
  n = 0
  vars_so_far = set()
  for eqn in eqns:
    for v in eqn.invars:
      if v not in vars_so_far:
        vars_so_far.add(v)
        n += 1
    for v in eqn.outvars:
      vars_so_far.add(v)
  return n

#  _____            _     _             _   _
# |  __ \          (_)   | |           | | (_)
# | |__) |___  __ _ _ ___| |_ _ __ __ _| |_ _  ___  _ __  ___
# |  _  // _ \/ _` | / __| __| '__/ _` | __| |/ _ \| '_ \/ __|
# | | \ \  __/ (_| | \__ \ |_| | | (_| | |_| | (_) | | | \__ \
# |_|  \_\___|\__, |_|___/\__|_|  \__,_|\__|_|\___/|_| |_|___/
#              __/ |
#             |___/


def _dense(
    x: Array,
    params: Sequence[Array],
    axes: int,
    with_reshape: bool,
    ) -> Array:
  """Example of a dense layer function."""
  # NOTE: This function uses `tensordot` so contracts over the last
  # `axes` dimensions of `x` and the first `axes` dimensions of `params`.
  match params:
    case [w, b]:
      y = jnp.tensordot(x, w, axes=axes)
      if with_reshape:
        return y + b.reshape((1,) * (y.ndim - b.ndim) + b.shape)
      else:
        return y + b
    case [w]:
      return jnp.tensordot(x, w, axes=axes)
    case _:
      raise ValueError("Unsupported parameters list")


def _dense_parameter_extractor(
    reversed_eqns: Sequence[JaxprEqn],
    variant: str = "dense",
) -> Mapping[str, Any]:
  """Extracts all parameters from the `dot_general` operator."""
  n = num_unique_inputs(reversed_eqns[::-1])

  for eqn in reversed_eqns:
    if eqn.primitive.name == "dot_general":
      return dict(
          meta=tags.LayerMetaData(
              variant=variant,
              outputs_index=(0,),
              inputs_index=(1,),
              params_index=tuple(i + 2 for i in range(n - 1)),
          ),
          **eqn.params,
      )
  assert False


def _make_general_dense_pattern(
    with_bias: bool,
    with_reshape: bool,
    num_repeated_axes: int,
    num_in_dims: int,
    num_out_dims: int,
) -> GraphPattern:
  """Creates a pattern for a dense or repeated dense layer."""
  batch_dim = (2,)
  repeating_dims = tuple(itertools.repeat(7, num_repeated_axes))

  out_dims = tuple(i + 2 for i in range(num_out_dims))
  in_dims = tuple(i + 2 for i in range(num_in_dims))
  x_shape = batch_dim + repeating_dims + in_dims
  weight_shape = in_dims + out_dims
  p_shapes = [weight_shape, out_dims] if with_bias else [weight_shape]

  name = "dense_with_bias" if with_bias else "dense_no_bias"
  name = name + ("_with_reshape" if with_reshape else "_no_reshape")

  if num_repeated_axes > 0:
    name = f"repeated[{num_repeated_axes}]_{name}"
    variant = "repeated_dense"
  else:
    variant = "dense"

  return GraphPattern(
      name=name,
      tag_primitive=tags.layer_tag,
      compute_func=functools.partial(
          _dense, axes=num_in_dims, with_reshape=with_reshape),
      parameters_extractor_func=functools.partial(
          _dense_parameter_extractor, variant=variant),
      example_args=[np.zeros(x_shape), [np.zeros(s) for s in p_shapes]],
  )


def _conv2d(x: Array, params: Sequence[Array], flax_style: bool) -> Array:
  """Example of a conv2d layer function."""

  w = params[0]

  y = jax.lax.conv_general_dilated(
      x,
      w,
      window_strides=(2, 2),
      padding="SAME",
      dimension_numbers=("NHWC", "HWIO", "NHWC"))

  if len(params) == 1:
    # No bias
    return y

  # Add bias
  if flax_style:
    bias = params[1]
    return y + bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
  return y + params[1][None, None, None]


def _conv2d_parameter_extractor(
    reversed_eqns: Sequence[JaxprEqn],
    variant: str = "conv2d",
) -> Mapping[str, Any]:
  """Extracts all parameters from the `conv_general_dilated` operator."""

  n = num_unique_inputs(reversed_eqns[::-1])

  for eqn in reversed_eqns:
    if eqn.primitive.name == "conv_general_dilated":
      return dict(
          meta=tags.LayerMetaData(
              variant=variant,
              outputs_index=(0,),
              inputs_index=(1,),
              params_index=tuple(i + 2 for i in range(n - 1)),
          ),
          **eqn.params,
      )

  assert False


def _make_conv2d_pattern(
    with_bias: bool,
    flax_style: bool,
) -> GraphPattern:

  x_shape = [2, 8, 8, 5]

  p_shapes = ([[3, 3, 5, 4], [4]] if with_bias else
              [[3, 3, 5, 4]])

  return GraphPattern(
      name="conv2d_with_bias" if with_bias else "conv2d_no_bias",
      tag_primitive=tags.layer_tag,
      compute_func=functools.partial(_conv2d, flax_style=flax_style),
      parameters_extractor_func=_conv2d_parameter_extractor,
      example_args=[np.zeros(x_shape), [np.zeros(s) for s in p_shapes]],
  )


def _scale_and_shift(
    x: Array,
    params: Sequence[Array],
    has_scale: bool,
    has_shift: bool,
) -> Array:
  """Example of a scale and shift function."""

  if has_scale and has_shift:
    scale, shift = params
    return x * scale + shift

  elif has_scale:
    [scale] = params
    return x * scale

  elif has_shift:
    [shift] = params
    return x + shift

  else:
    raise ValueError("You must have either `has_scale` or `has_shift` set "
                     "to True.")


def _scale_and_shift_parameter_extractor(
    reversed_eqns: Sequence[JaxprEqn],
    variant: str = "scale_and_shift",
) -> Mapping[str, Any]:
  """Extracts all parameters from the `conv_general_dilated` operator."""

  has_scale = False

  has_shift = False

  for eqn in reversed_eqns:
    if eqn.primitive.name == "mul":
      has_scale = True
    elif eqn.primitive.name == "add":
      has_shift = True

  return dict(
      meta=tags.LayerMetaData(
          variant=variant,
          outputs_index=(0,),
          inputs_index=(1,),
          params_index=tuple(i + 2 for i in range(has_scale + has_shift)),
      ),
      has_scale=has_scale,
      has_shift=has_shift,
  )


def _make_scale_and_shift_pattern(
    broadcast_ndim: int,
    has_scale: bool,
    has_shift: bool,
    p_dim: int = 13,
) -> GraphPattern:
  """Creates a scale and shift graph pattern."""

  assert broadcast_ndim >= 0

  assert has_scale or has_shift

  x_shape = [i + 2 for i in range(broadcast_ndim)] + [p_dim]
  p_shapes = [[p_dim], [p_dim]] if (has_scale and has_shift) else [[p_dim]]

  if has_scale and has_shift:
    name = f"scale_and_shift_broadcast_{broadcast_ndim}"
  elif has_scale:
    name = f"scale_only_broadcast_{broadcast_ndim}"
  elif has_shift:
    name = f"shift_only_broadcast_{broadcast_ndim}"
  else:
    raise ValueError("Unreachable.")

  return GraphPattern(
      name=name,
      tag_primitive=tags.layer_tag,
      compute_func=functools.partial(
          _scale_and_shift, has_scale=has_scale, has_shift=has_shift),
      parameters_extractor_func=_scale_and_shift_parameter_extractor,
      example_args=[np.zeros(x_shape), [np.zeros(s) for s in p_shapes]],
  )


def _normalization_haiku_flax(
    inputs: Sequence[Array],
    params: Sequence[Array],
    has_scale: bool,
    has_shift: bool,
    has_reshape: bool,
) -> Array:
  """Example of normalization as is defined in Haiku/Flax."""

  if len(params) not in (1, 2):
    raise ValueError("The inputs to the `normalization_haiku` computation must "
                     f"have either 1 or 2 parameters, but got {len(params)}.")

  [inputs, rsqrt_var] = inputs

  if has_scale:
    scale = params[0]
    if has_reshape:
      scale = scale.reshape(
          [1] * (inputs.ndim - scale.ndim) + list(scale.shape))
    inv = scale * rsqrt_var
  else:
    inv = rsqrt_var

  outputs = inputs * inv

  if has_shift:
    shift = params[1]
    if has_reshape:
      shift = shift.reshape(
          [1] * (inputs.ndim - shift.ndim) + list(shift.shape))
    return outputs + shift
  return outputs


def _normalization_haiku_preprocessor(
    in_vars: Vars,
    make_var_func: MakeVarFunc,
) -> tuple[tuple[Var, ...], JaxprEqns]:
  """Preprocesses the inputs to a Haiku normalization layer.

  The standard ``scale_and_shift`` represents the following canonical
  computation:
    y = x * scale + shift
  Normalization performs a similar computation, where the `normalized_x` below
  represents the standard ``x`` input to ``scale_and_shift``:
    normalized_x = (x - m) / sqrt(var(x) + eps)
    y = normalized_x * scale + shift
  Each ``layer_tag`` represents a specific computation and hence it expects its
  inputs to be in canonical form. For ``scale_and_shift`` the input must be
  the array that gets multiplied by the ``scale`` before the ``shift`` addition
  as shown above. However, Haiku performs normalization slightly out of order:
    y = [(x - m) * scale] / sqrt(var(x) + eps) + shift
  As a result, in the Jax computation graph the canonical input (normalized_x)
  does not exist, because of the ordering of the multiplication and division.
  To remedy this we have to add this additional function, which to be able to
  compute from the variables in the Haiku normalization computation, the
  canonical input to ``scale_and_shift`` tag.

  Args:
    in_vars: The input variables to the pattern.
    make_var_func: A function to create correctly new variables.

  Returns:
    The canonical input to ``scale_and_shift`` pattern.
  """

  [in_var, rsqrt_var, *param_vars] = in_vars

  # The equation below corresponds to the computation:
  # normalized_inputs = inputs * rsqrt_var

  normalized_inputs_var = make_var_func(in_var.aval)

  normalized_inputs_eqn = jax.core.new_jaxpr_eqn(
      invars=[in_var, rsqrt_var],
      outvars=[normalized_inputs_var],
      primitive=jax.lax.mul_p,
      params=dict(),
      effects=set(),
  )

  return (normalized_inputs_var, *param_vars), [normalized_inputs_eqn]


def _make_normalization_haiku_flax_pattern(
    broadcast_ndim: int,
    has_reshape: bool,
    p_dim: int = 13,
    has_shift: bool = True,
) -> GraphPattern:
  """Creates a pattern for a Haiku/Flax normalization layer."""

  assert broadcast_ndim >= 0

  x_shape = [i + 2 for i in range(broadcast_ndim)] + [p_dim]

  example_params = [np.zeros([p_dim])]
  if has_shift:
    example_params.append(np.zeros([p_dim]))

  return GraphPattern(
      name=f"normalization_haiku_broadcast_{broadcast_ndim}",
      tag_primitive=tags.layer_tag,
      compute_func=functools.partial(
          _normalization_haiku_flax,
          has_scale=True,
          has_shift=has_shift,
          has_reshape=has_reshape),
      parameters_extractor_func=_scale_and_shift_parameter_extractor,
      example_args=[[np.zeros(x_shape), np.zeros(x_shape)], example_params],
      in_values_preprocessor=_normalization_haiku_preprocessor
  )

# NOTE: itertools iterates the last iterator first
# i.e. [(True, False), 0, 1, 1] [(True, False), 0, 1, 2] ...
DENSE_GRAPH_PATTERNS = tuple(
    _make_general_dense_pattern(
        with_bias=b,
        with_reshape=r,
        num_repeated_axes=rep,
        num_in_dims=n_ins,
        num_out_dims=n_outs)
    for (b, r), rep, n_ins, n_outs in itertools.product(
        ((True, False), (True, True), (False, False)),
        range(3),
        range(1, 3),
        range(1, 3)
    )
)

NORMALIZATION_GRAPH_PATTERNS = tuple(
    _make_normalization_haiku_flax_pattern(
        broadcast_ndim=n,
        has_reshape=r,
        has_shift=s)
    for n, r, s in itertools.product(
        range(2),
        (False, True),
        (False, True),
    )
)

DEFAULT_GRAPH_PATTERNS = DENSE_GRAPH_PATTERNS + (
    _make_conv2d_pattern(True, False),
    _make_conv2d_pattern(True, True),
    _make_conv2d_pattern(False, False),
    _make_scale_and_shift_pattern(1, True, True),
    _make_scale_and_shift_pattern(0, True, True)
    )

DEFAULT_GRAPH_PATTERNS += NORMALIZATION_GRAPH_PATTERNS

DEFAULT_GRAPH_PATTERNS += (
    _make_scale_and_shift_pattern(1, True, False),
    _make_scale_and_shift_pattern(0, True, False),
    _make_scale_and_shift_pattern(1, False, True),
    _make_scale_and_shift_pattern(0, False, True),
)


class TagLocation:
  """Represents a tag location inside a function graph."""

  def __init__(
      self,
      tag_eqn: JaxprEqn,
      parent_equations: Sequence[tuple[JaxprEqn, int]] = (),
  ):
    self.tag_eqn = tag_eqn
    self.parent_equations = list(parent_equations)

  @property
  def base_name(self) -> str:
    meta = self.tag_eqn.params.get("meta")
    assert meta is not None and isinstance(meta, tags.LayerMetaData)
    assert meta.name is not None, self.tag_eqn
    return meta.name

  @property
  def full_name(self) -> str:
    """The full name of the tag location."""

    prefix = ""
    param_vars = self.bottom_level_parameters

    for eqn, n in reversed(self.parent_equations):

      assert eqn.primitive.name in HIGHER_ORDER_NAMES

      # Prefix for this higher order primitive
      prefix = prefix + f"{eqn.primitive.name}_{n}/"

      if eqn.primitive.name == "cond":
        raise NotImplementedError()

      elif eqn.primitive.name == "scan":

        p_indexes = [eqn.params["jaxpr"].jaxpr.invars.index(p)
                     for p in param_vars]
        checks = [pi < eqn.params["num_consts"] for pi in p_indexes]

        if not (all(checks) or all(not ci for ci in checks)):
          raise ValueError("Parameters inside scan of the same tag are not both"
                           " carry or const.")

        if all(checks):
          prefix = prefix + "const/"
        else:
          prefix = prefix + "carry/"

      elif eqn.primitive.name == "pjit":
        p_indexes = [eqn.params["jaxpr"].jaxpr.invars.index(p)
                     for p in param_vars]

      elif eqn.primitive.name == "while":
        p_indexes = [eqn.params["body_jaxpr"].jaxpr.invars.index(p)
                     for p in param_vars]

      elif eqn.primitive.name in ("xla_call", "xla_pmap"):
        p_indexes = [eqn.params["call_jaxpr"].invars.index(p)
                     for p in param_vars]

      else:
        raise NotImplementedError()

      param_vars = [eqn.invars[pi] for pi in p_indexes]

    return prefix + self.base_name

  @property
  def bottom_level_parameters(self) -> tuple[Var, ...]:
    """The bottom level variables of the tag location."""
    return tags.layer_eqn_data(self.tag_eqn).params

  @property
  def top_level_parameters(self) -> tuple[Var, ...]:
    """The top level parameter variables of the tag location."""

    param_vars = self.bottom_level_parameters

    for eqn, _ in reversed(self.parent_equations):

      assert eqn.primitive.name in HIGHER_ORDER_NAMES

      if eqn.primitive.name == "cond":
        raise NotImplementedError()

      elif eqn.primitive.name in ("scan", "pjit"):
        invars = eqn.params["jaxpr"].jaxpr.invars

      elif eqn.primitive.name == "while":
        invars = eqn.params["body_jaxpr"].jaxpr.invars

      elif eqn.primitive.name in ("xla_call", "xla_pmap"):
        invars = eqn.params["call_jaxpr"].invars

      else:
        raise NotImplementedError()

      # Indices inside of the higher order primitive
      p_indexes = [invars.index(p) for p in param_vars]

      # Inputs (to the higher order primitive) corresponding to those indices
      param_vars = tuple(eqn.invars[pi] for pi in p_indexes)

    return param_vars

  def add_parent_eqn(self, eqn: JaxprEqn, counter: int):
    assert eqn.primitive.name in HIGHER_ORDER_NAMES
    self.parent_equations.append((eqn, counter))


class TaggedFunction:
  """Represents a function that has been processed and auto tagged."""

  def __init__(
      self,
      func_graph: JaxprGraph,
      tag_locations: Sequence[TagLocation],
  ):
    self._func_graph = func_graph
    self._tag_locations = tag_locations
    self._flat_func = jex.core.jaxpr_as_fun(func_graph.closed_jaxpr)
    self._param_labels = self._compute_parameter_labels()

  def __call__(self, *args, **kwargs):
    flat_args = jax.tree_util.tree_leaves(args)
    flat_output = self._flat_func(*flat_args)
    return jax.tree_util.tree_unflatten(self._func_graph.out_tree, flat_output)

  def _compute_parameter_labels(self) -> Mapping[Var, Sequence[str]]:
    """Computes the parameter labels as a dict from params to strings."""

    # Collect all registrations for every tagged parameter
    tagged_params = {}

    for tag_l in self._tag_locations:
      for p in tag_l.top_level_parameters:

        assert p in self._func_graph.params_vars

        if p not in tagged_params:
          tagged_params[p] = []

        tagged_params[p].append(tag_l.full_name)

    return tagged_params

  def print_parameter_tags(self):
    """Prints all the parameter registrations."""
    # Print all tag parameter registrations
    labels = ["|".join(self._param_labels.get(p, ["Orphan"]))
              for p in self._func_graph.params_vars]
    logging.info("=" * 50)
    logging.info(
        "Graph parameter registrations:\n%s",
        pprint.pformat(jax.tree_util.tree_unflatten(
            self._func_graph.params_tree, labels,
        ))
    )
    logging.info("=" * 50)

  def check_multiple_registrations(self):
    for p in self._func_graph.params_vars:
      if len(self._param_labels[p]) > 1:
        raise ValueError(f"Parameter {p} has been registered to multiple tags: "
                         f"{self._param_labels[p]}.")


def _auto_register_tags(
    graph: JaxprGraph,
    graph_matcher_rules: GraphMatcherComparator,
    graph_patterns: Sequence[GraphPattern],
    register_orphans: bool,
    register_only_until_losses: bool,
    matchable_params: Set[Var],
) -> tuple[JaxprGraph, Sequence[TagLocation]]:
  """Internal function for automatic registration of layer tags."""

  higher_counters = {
      "cond": 0,
      "while": 0,
      "scan": 0,
      "pjit": 0,
      "xla_call": 0,
      "xla_pmap": 0,
    }

  # Extract the sub-graph that leads to losses
  if register_only_until_losses:

    eqns_for_registration = []
    sub_graph_vars = set()
    for eqn in reversed(graph.jaxpr.eqns):

      # Note that graph.losses_eqns won't recurse into higher order primitives
      # to find loss tags, so any losses defined inside such primitives will be
      # effectively chopped out.

      if (eqn in graph.losses_eqns or
          any(v in sub_graph_vars for v in eqn.outvars)):

        eqns_for_registration.append(eqn)
        sub_graph_vars.update(
            v for v in eqn.invars if not isinstance(v, jex.core.Literal))

    eqns_for_registration = eqns_for_registration[::-1]

  else:
    eqns_for_registration = graph.jaxpr.eqns

  # Count number of uses of each parameter and if it exceeds 1, we don't do any
  # automatic registration. Note that we don't have to recurse into higher order
  # primitives to count uses because we only care about whether there is more
  # than one use of a given parameter. If there is, and they are not all inside
  # of one higher order primitive, we will catch it here. If they're all in one
  # higher order primitive, then we will catch that when we recursively call
  # _auto_register_tags, and no registrations will happen at that level.
  # Finally, the parameter in question will be seen as an orphan and registered
  # as generic *only* at the top-level call of _auto_register_tags, as intended
  # (since register_orphans is True only at the top-level call).
  param_uses = collections.Counter()
  for eqn in eqns_for_registration:
    if not isinstance(eqn.primitive, tags.LayerTag):
      for v in eqn.invars:
        if v in graph.params_vars:
          param_uses[v] += 1

  manual_registrations = graph.manual_registrations
  manually_tagged_params = set()
  for eqn in manual_registrations:
    for p in tags.layer_eqn_data(eqn).params:
      manually_tagged_params.add(p)

  # Parameters that are eligible for auto-matching are those that have exactly
  # one use in the graph and are not manually tagged.
  single_use_params = {p for p in graph.params_vars if param_uses[p] <= 1}
  matchable_params = (single_use_params & matchable_params) - manually_tagged_params  # pylint: disable=line-too-long

  # Process all higher order primitives
  eqns = []
  tag_locations = []
  for eqn in graph.jaxpr.eqns:

    if not (eqn in eqns_for_registration
            and eqn.primitive.name in HIGHER_ORDER_NAMES):

      eqns.append(eqn)

      continue

    eqn_name = eqn.primitive.name
    if eqn_name == "cond":
      sub_jaxprs = eqn.params["branches"]
    elif eqn_name == "while":
      sub_jaxprs = [eqn.params["body_jaxpr"]]
    elif eqn_name in ("scan", "pjit"):
      sub_jaxprs = [eqn.params["jaxpr"]]
    elif eqn_name in ("xla_call", "xla_pmap"):
      sub_jaxprs = [eqn.params["call_jaxpr"]]
    else:
      raise NotImplementedError()

    final_jaxprs = []
    final_tag_locations = []
    for original_jaxpr in sub_jaxprs:

      sub_jaxpr = to_closed_jaxpr(original_jaxpr)

      sub_params_vars = []
      sub_matchable_params = set()
      for outer_v, inner_v in zip(eqn.invars, sub_jaxpr.jaxpr.invars):

        if outer_v in graph.params_vars:
          sub_params_vars.append(inner_v)

        if isinstance(outer_v, Var) and outer_v in matchable_params:
          assert isinstance(inner_v, Var)
          sub_matchable_params.add(inner_v)

      sub_graph, sub_tag_locations = _auto_register_tags(
          graph=JaxprGraph(
              name=graph.name + f"_{eqn_name}",
              closed_jaxpr=sub_jaxpr,
              params_tree=jax.tree_util.tree_structure(sub_params_vars),
              params_vars=sub_params_vars,
              out_tree=jax.tree_util.tree_structure(sub_jaxpr.jaxpr.outvars),
              tag_ctor=None,
          ),
          graph_matcher_rules=graph_matcher_rules,
          graph_patterns=graph_patterns,
          register_orphans=False,
          register_only_until_losses=False,
          matchable_params=sub_matchable_params,
      )

      final_jaxprs.append(
          to_jaxpr_or_closed_jaxpr(sub_graph.closed_jaxpr, original_jaxpr))

      final_tag_locations.append(sub_tag_locations)

    if eqn_name == "cond":
      if final_tag_locations[0] or final_tag_locations[1]:
        # TODO(botev): We need to check each branch has identical registrations
        raise NotImplementedError()
      sub_tag_locations = []
    else:
      # Extract the sub jaxpr parameter tag registrations and input vars
      [sub_tag_locations] = final_tag_locations  # pylint:disable=unbalanced-tuple-unpacking

    del final_tag_locations

    # Update the jaxpr parameter in the equation
    eqn_params = dict(**eqn.params)
    if eqn_name == "cond":
      eqn_params["branches"] = tuple(final_jaxprs)
    elif eqn_name == "while":
      [eqn_params["body_jaxpr"]] = final_jaxprs  # pylint:disable=unbalanced-tuple-unpacking
    elif eqn_name in ("scan", "pjit"):
      [eqn_params["jaxpr"]] = final_jaxprs  # pylint:disable=unbalanced-tuple-unpacking
    elif eqn_name in ("xla_call", "xla_pmap"):
      [eqn_params["call_jaxpr"]] = final_jaxprs  # pylint:disable=unbalanced-tuple-unpacking
    else:
      raise NotImplementedError()

    eqns.append(eqn.replace(params=eqn_params))

    del final_jaxprs

    # Insert the sub-registrations into the tagged_params
    for tag_l in sub_tag_locations:
      tag_l.add_parent_eqn(eqns[-1], higher_counters[eqn_name])
      higher_counters[eqn_name] = higher_counters[eqn_name] + 1
      tag_locations.append(tag_l)

  # Make a new graph with the replaced higher order equations
  mid_graph = JaxprGraph(
      name=graph.name,
      closed_jaxpr=ClosedJaxpr(
          jaxpr=graph.jaxpr.replace(eqns=eqns),
          consts=graph.consts,
      ),
      params_tree=graph.params_tree,
      params_vars=graph.params_vars,
      out_tree=graph.out_tree,
      tag_ctor=None,
  )
  del graph

  # Find matches
  matches = find_layer_tags_and_patterns(
      graph=mid_graph,
      eqns_for_patterns=eqns_for_registration,
      graph_matcher_rules=graph_matcher_rules,
      graph_patterns=graph_patterns,
      matchable_params=matchable_params,
  )

  tagged_params = set()

  # Registrations in higher order primitives
  for tag_l in tag_locations:
    for p in tag_l.top_level_parameters:
      tagged_params.add(p)

  # Manual registrations
  for manual_eqn in manual_registrations:
    for p in tags.layer_eqn_data(manual_eqn).params:
      tagged_params.add(p)

  # Automatically detected registrations
  for match in matches.values():
    for p in match.param_graph_variables:
      tagged_params.add(p)

  # Create the Jaxpr with all the tag registrations
  make_var_func = jax.core.gensym()
  eqns = list()
  env = {}
  pattern_counters = {}

  if register_orphans:

    for param in mid_graph.params_vars:

      if param not in tagged_params:

        orphan_p = make_var_func(param.aval)

        n = pattern_counters.get("generic", 0)
        pattern_counters["generic"] = n + 1

        eqns.append(
            jax.core.new_jaxpr_eqn(
                invars=[param],
                outvars=[orphan_p],
                primitive=tags.layer_tag,
                params=dict(meta=tags.LayerMetaData(
                    variant="generic",
                    inputs_index=(),
                    outputs_index=(0,),
                    params_index=(0,),
                    name=f"Auto[generic({n})]",
                )),
                effects=set(),
            )
        )

        env[param] = orphan_p
        tag_locations.append(TagLocation(eqns[-1]))

  for eqn in mid_graph.jaxpr.eqns:

    invars = [env.get(v, v) if isinstance(v, Var) else v
              for v in eqn.invars]

    eqns.append(eqn.replace(invars=invars))

    if isinstance(eqn.primitive, tags.LayerTag):

      # Mark manual registrations
      meta = eqns[-1].params.get("meta")
      assert meta is not None and isinstance(meta, tags.LayerMetaData)

      if meta.name is None:
        n = pattern_counters.get(meta.variant, 0)
        pattern_counters[meta.variant] = n + 1
        meta.name = f"Manual[{meta.variant}({n})]"

      tag_locations.append(TagLocation(eqn))

    for var in eqn.outvars:

      # Check if this is a match of a graph pattern
      match = matches.get(var)

      if match is not None:

        for additional_eqn in match.create_eqn(env, make_var_func):
          eqns.append(additional_eqn)

        # Mark automatic registration
        meta = eqns[-1].params.get("meta")
        assert meta is not None and isinstance(meta, tags.LayerMetaData)
        assert meta.name is None
        n = pattern_counters.get(meta.variant, 0)
        pattern_counters[meta.variant] = n + 1
        meta.name = (f"Auto[tag_variant={meta.variant}({n})|"
                     f"match_type={match.name}]")
        tag_locations.append(TagLocation(eqns[-1]))

  final_outvars = [env.get(v, v) if isinstance(v, Var) else v
                   for v in mid_graph.jaxpr.outvars]

  final_graph = JaxprGraph(
      name=mid_graph.name,
      closed_jaxpr=ClosedJaxpr(
          jaxpr=mid_graph.jaxpr.replace(eqns=eqns, outvars=final_outvars),
          consts=mid_graph.closed_jaxpr.consts
      ),
      params_tree=mid_graph.params_tree,
      params_vars=mid_graph.params_vars,
      out_tree=mid_graph.out_tree,
      tag_ctor=None,
  )

  return final_graph, tag_locations


def auto_register_tags(
    func: utils.Func,
    func_args: utils.FuncArgs,
    params_index: int = 0,
    register_only_generic: bool = False,
    compute_only_loss_tags: bool = True,
    patterns_to_skip: Sequence[str] = (),
    graph_matcher_rules: GraphMatcherComparator = GraphMatcherComparator(),
    graph_patterns: Sequence[GraphPattern] = DEFAULT_GRAPH_PATTERNS,
) -> TaggedFunction:
  """Transforms the function by automatically registering layer tags.

  Args:
    func: The original function to transform.
    func_args: Example arguments to ``func`` which to be used for tracing it.
    params_index: Specifies, which inputs to the function are to be considered
      a parameter variable. Specifically - ``inputs[params_index]``.
    register_only_generic: If ``True`` registers all parameters not already in a
      layer tag with a generic tag, effectively ignoring ``graph_patterns``.
    compute_only_loss_tags: If set to ``True`` (default) the resulting function
      will only compute the loss tags in ``func``, not its full computation and
      actual output.
    patterns_to_skip: The names of any patterns from the provided list, which to
      be skipped/not used during the pattern matching.
    graph_matcher_rules: A :class:`~GraphMatcherRules` instance, which is used
      for determining equivalence of individual Jax primitives.
    graph_patterns: A sequence of :class:`~GraphPattern` objects, which contain
      all patterns to use, in order of precedence, which to try to find in the
      graph before registering a parameter with a generic layer tag.
  Returns:
    A transformed function as described above.
  """

  graph = make_jax_graph(
      func=func,
      func_args=func_args,
      params_index=params_index,
      name="main",
      compute_only_loss_tags=compute_only_loss_tags,
      clean_broadcasts=True,
  )

  patterns = () if register_only_generic else tuple(
      pattern for pattern in graph_patterns
      if pattern.name not in patterns_to_skip)

  func_graph, tagged_locations = _auto_register_tags(
      graph=graph,
      graph_matcher_rules=graph_matcher_rules,
      graph_patterns=patterns,
      register_orphans=True,
      register_only_until_losses=True,
      matchable_params=set(graph.params_vars),
  )

  func = TaggedFunction(
      func_graph=func_graph,
      tag_locations=tagged_locations,
  )
  func.print_parameter_tags()

  func.check_multiple_registrations()

  return func
