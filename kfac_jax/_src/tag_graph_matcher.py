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
import functools
import itertools
import pprint
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, TypeVar, Union

from absl import logging
import chex
import immutabledict
import jax
from jax import core
from jax import lax
from jax import util as jax_util
import jax.numpy as jnp
from kfac_jax._src import layers_and_loss_tags as tags
from kfac_jax._src import utils
import numpy as np

# Types for annotation
T = TypeVar("T")
EquivalenceFunction = Callable[[core.JaxprEqn, core.JaxprEqn], bool]
GraphMatch = Tuple[
    "JaxprGraph",
    Dict[core.Var, core.Var],
    Tuple[core.JaxprEqn, ...]
]
TagCtor = Callable[
    [Sequence[core.JaxprEqn], Mapping[core.Var, chex.Array]],
    chex.Array
]
PatternComputeFunc = Callable[[chex.Array, Sequence[chex.Array]], chex.Array]
ParameterExtractorFunc = Callable[[Sequence[core.JaxprEqn]], Mapping[str, Any]]
ValuesProcessor = Callable[[Sequence[chex.Array]], Sequence[chex.Array]]


def eval_jaxpr_eqn(eqn, in_values):
  """Computes the outputs of the given Jaxpr equation."""
  subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
  with jax.core.source_info_util.user_context(
      eqn.source_info.traceback):
    return eqn.primitive.bind(*subfuns, *in_values, **bind_params)


def reshape_equivalent(
    equation1: core.JaxprEqn,
    equation2: core.JaxprEqn
) -> bool:
  """Equivalence rule for :func:`~jax.numpy.reshape` primitives."""
  if not (equation1.primitive.name == "reshape" and
          equation2.primitive.name == "reshape"):
    raise ValueError("This is only applicable to `reshape` primitive.")

  return equation1.params["dimensions"] == equation2.params["dimensions"]


def broadcast_in_dim_equivalent(
    equation1: core.JaxprEqn,
    equation2: core.JaxprEqn
) -> bool:
  """Equivalence rule for :func:`~jax.numpy.broadcast` primitives."""
  if not (equation1.primitive.name == "broadcast_in_dim" and
          equation2.primitive.name == "broadcast_in_dim"):
    raise ValueError("This is only applicable to `broadcast_in_dim` primitive.")
  return True


def conv_general_dilated_equivalent(
    equation1: core.JaxprEqn,
    equation2: core.JaxprEqn
) -> bool:
  """Equivalence rule for :func:`~jax.lax.conv_general_dilated` primitives."""
  if not (equation1.primitive.name == "conv_general_dilated" and
          equation2.primitive.name == "conv_general_dilated"):
    raise ValueError("This is only applicable to `conv_general_dilated` "
                     "primitive.")
  params1 = equation1.params
  params2 = equation2.params
  for k in ("window_strides", "padding",
            "lhs_dilation", "rhs_dilation",
            "lhs_shape", "rhs_shape"):
    if len(params1[k]) != len(params2[k]):
      return False
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
  return True


def dot_general_equivalent(
    equation1: core.JaxprEqn,
    equation2: core.JaxprEqn
) -> bool:
  if not (equation1.primitive.name == "dot_general" and
          equation2.primitive.name == "dot_general"):
    raise ValueError("This is only applicable to `conv_general_dilated` "
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
  def commutative_ops_names(self) -> Set[str]:
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
      equivalence_rule: EquivalenceFunction
  ):
    """Adds the special equivalence rule for ``name`` to the global store."""
    if name in self.special_eqn_equivalence_rules:
      raise ValueError(
          f"Special equation equivalence rule already exists for name: {name}")
    self._special_eqn_equivalence_rules[name] = equivalence_rule

  def are_equivalent(
      self,
      equation1: core.JaxprEqn,
      equation2: core.JaxprEqn
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


class JaxprGraph:
  """A wrapper around Jaxpr as a graph for pattern matching.

  Attributes:
    name: The name for this Jaxpr graph.
    jaxpr: The original Jaxpr that is being wrapped.
    consts: The constants needed for evaluation of the raw Jaxpr.
    params_vars: A flat list of all of the abstract parameter variables.
    params_tree: The PyTree structure of the parameter variables.
    out_tree: The PyTree structure of the outputs of the Jaxpr function.
    losses_eqns: A tuple of all of the Jaxpr equations corresponding to a loss
      tag.
    var_to_creation_op: A mapping of variables to the Jax equation that created
      it.
    tag_ctor: This is an optional attribute, that defines if this is used during
      automatic layer tag registration, how to construct the corresponding layer
      tag primitive from the subgraph matching this pattern.
  """

  def __init__(
      self,
      name: str,
      jaxpr: core.Jaxpr,
      consts: Sequence[Any],
      params_vars: Sequence[core.Var],
      params_tree: utils.PyTreeDef,
      out_tree: utils.PyTreeDef,
      tag_ctor: Optional[TagCtor],
  ):
    """Initializes the instance.

    Args:
      name: The name for this Jaxpr graph.
      jaxpr: The original Jaxpr that is being wrapped.
      consts: The constants needed for evaluation of the raw Jaxpr.
      params_vars: A flat list of all of the abstract parameter variables.
      params_tree: The PyTree structure of the parameter variables.
      out_tree: The PyTree structure of the outputs of the Jaxpr function.
      tag_ctor: This is an optional attribute, that defines if this is used
        during automatic layer tag registration, how to construct the
        corresponding layer tag primitive from the subgraph matching this
        pattern.
    """
    self.name = name
    self.jaxpr = jaxpr
    self.params_vars = list(params_vars)
    self.params_tree = params_tree
    self.out_tree = out_tree
    self.consts = list(consts)
    self.tag_ctor = tag_ctor
    self.losses_eqns = tuple(
        eqn for eqn in jaxpr.eqns if isinstance(eqn.primitive, tags.LossTag))
    self.var_to_creation_op = immutabledict.immutabledict(
        sum(([(var, eqn) for var in eqn.outvars] for eqn in jaxpr.eqns), []))

  def __repr__(self):
    return (f"{self.__class__.__name__}({self.name!r}, "
            f"{self.jaxpr!r}, {self.consts!r}, {self.params_vars!r}, "
            f"{self.params_tree!r}, {self.out_tree!r}, {self.tag_ctor!r})")

  @property
  def outvars(self) -> Sequence[core.Atom]:
    """A sequence of all of the output variables of the Jaxpr graph."""
    return self.jaxpr.outvars

  def ancestors_sub_graph(self, eqns: Iterable[core.JaxprEqn]) -> "JaxprGraph":
    """Constructs a subgraph of all the ancestors(self-inclusive) of ``eqns``."""
    sub_graph_eqns = []
    sub_graph_vars = set()
    for eqn in reversed(self.jaxpr.eqns):
      if eqn in eqns or any(v in sub_graph_vars for v in eqn.outvars):
        sub_graph_eqns.append(eqn)
        sub_graph_vars.update(
            v for v in eqn.invars if not isinstance(v, core.Literal))
    outvars, out_tree = jax.tree_flatten(tuple(
        eqn.outvars for eqn in self.losses_eqns))
    return JaxprGraph(
        name="sub_" + self.name,
        jaxpr=core.Jaxpr(
            constvars=self.jaxpr.constvars,
            invars=self.jaxpr.invars,
            outvars=outvars,
            eqns=tuple(reversed(sub_graph_eqns))
        ),
        consts=self.consts,
        params_vars=self.params_vars,
        params_tree=self.params_tree,
        out_tree=out_tree,
        tag_ctor=self.tag_ctor
    )

  def extract_manual_registrations(self) -> Tuple[tags.LayerTagEqn, ...]:
    """Returns all manually registered tags."""
    registered_tags = []
    for eqn in self.jaxpr.eqns:
      if isinstance(eqn.primitive, tags.LayerTag):
        for param in eqn.primitive.split_all_inputs(eqn.invars)[2]:
          if param not in self.params_vars:
            raise ValueError(f"One of the parameters of the manual layer "
                             f"registration equation: {eqn} is not part of the "
                             f"parameters of the global function.")
        registered_tags.append(eqn)
    return tuple(registered_tags)


def make_jax_graph(
    func: utils.Func,
    func_args: Sequence[Any],
    params_index: Union[int, Sequence[int]],
    graph_name: str,
    tag_ctor: Optional[TagCtor] = None,
) -> JaxprGraph:
  """Creates a :class:`~JaxGraph` instance from the provided function and arguments."""
  in_tree = jax.tree_structure(func_args)
  typed_jaxpr, out_shapes = jax.make_jaxpr(func, return_shape=True)(*func_args)
  in_vars = jax.tree_unflatten(in_tree, typed_jaxpr.jaxpr.invars)
  if isinstance(params_index, int):
    params_vars = in_vars[params_index]
  else:
    params_vars = tuple(in_vars[i] for i in params_index)
  params_vars, params_tree = jax.tree_flatten(params_vars)
  return JaxprGraph(
      name=graph_name,
      jaxpr=typed_jaxpr.jaxpr,
      consts=typed_jaxpr.literals,
      params_vars=params_vars,
      params_tree=params_tree,
      out_tree=jax.tree_structure(out_shapes),
      tag_ctor=tag_ctor
  )


class GraphPattern:
  """A graph pattern used for automatically detecting layers."""

  def __init__(
      self,
      name: str,
      tag_primitive: tags.LayerTag,
      precedence: int,
      compute_func: PatternComputeFunc,
      parameters_extractor_func: ParameterExtractorFunc,
      example_args: utils.FuncArgs,
      in_values_preprocessor: ValuesProcessor = lambda in_values: in_values,
  ):
    """Instantiates the graph pattern.

    The graph matcher needs to trace at least once the full function, which
    means the caller needs to provide it with dummy arguments. The shapes of the
    arguments do not matter, as the graph matcher ignores their values, however
    the rank does. Especially if there is some broadcasting happening you should
    register with every possible broadcast pattern. As a general advice avoid
    using a shape to be 1, unless you want the pattern to specifically match
    that, as some operations, like squeeze for example, can have special
    behaviour then.

    Args:
      name: The name of the pattern that is being registered to.
      tag_primitive: The primitive tag to bind.
      precedence: This specifies what precedence the graph matcher is going to
        assign to the provided pattern. The graph matcher will go from lowest
        to highest precedence, randomly breaking ties, when matching. Note that
        the pattern that matches a parameter with the lowest precedence will get
        registered and no other will. Specifically useful when there is a
        pattern for a layer with and without bias, in which case the with bias
        registration always should go with lower precedence.
      compute_func: The function that performs the computation.
      parameters_extractor_func: A function that extracts from the traced Jaxpr
        any parameters that are passed into the tag.
      example_args: Example arguments that can be inputted into ``func``.
      in_values_preprocessor: A function that can optionally modify the in_vals
        passed to the tag_primitive, from those that are usually the input to
        the jaxpr.
    """
    self._name = name
    self._tag_primitive = tag_primitive
    self._precedence = precedence
    self._compute_func = compute_func
    self._parameters_extractor_func = parameters_extractor_func
    self._example_args = example_args
    self._in_values_preprocessor = in_values_preprocessor
    self._graph = None

  @property
  def name(self) -> str:
    """Name of this graph pattern."""
    return self._name

  @property
  def tag_primitive(self) -> tags.LayerTag:
    """The layer tag primitive that this pattern corresponds to."""
    return self._tag_primitive

  @property
  def graph(self) -> JaxprGraph:
    """The Jaxpr graph representing the computation of this pattern."""
    if self._graph is None:
      jnp_args = jax.tree_map(jnp.asarray, self._example_args)
      self._graph = make_jax_graph(
          broadcast_merger(self._compute_func), jnp_args, 1, self._name)
    return self._graph

  def tag_ctor(
      self,
      eqns: Sequence[core.JaxprEqn],
      values_map: Mapping[core.Var, chex.Array]
  ) -> chex.Array:
    """Registers the layer tag for this graph pattern.

    Args:
      eqns: The equations in the function, where this pattern is inserted.
      values_map: A mapping between variables of the pattern and corresponding
        concrete Jax arrays.
    Returns:
      The output value of the layer tag, after its registration.
    """
    primitive_params = self._parameters_extractor_func(eqns)
    in_values = [values_map[v] for v in self.graph.jaxpr.invars]
    out_values = [values_map[v] for v in self.graph.jaxpr.outvars]
    in_values = self._in_values_preprocessor(in_values)
    return self.tag_primitive.bind(
        out_values[0], *in_values, **primitive_params)


def match_equations(
    graph: JaxprGraph,
    current_variables_map: Mapping[core.Var, core.Var],
    reversed_eqns_to_match: Sequence[core.JaxprEqn],
    input_vars: Sequence[core.Var],
    param_variables: Sequence[core.Var],
    graph_matcher_rules: GraphMatcherComparator,
) -> Optional[Dict[core.Var, core.Var]]:
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

  Returns:
    ``None`` if it is not possible to finish matching the remaining equations
    in the graph. Otherwise returns the full match of the pattern onto the
    graph, in terms of a variable to variable mapping.
  """
  # Copy the variables mapping
  current_variables_map = dict(current_variables_map)
  def add_vars_if_possible(
      eqn_vars: Sequence[core.Var],
      graph_vars: Sequence[core.Var]
  ) -> bool:
    """Tries to update the current variables map.

    If at least one of the pattern variables is a parameter, but the
    corresponding graph variable is not or vise-versa, the method does not
    update the current variables map and returns ``False``. Similarly if at
    least one of the graph variables is a :class:`~jax.core.Literal` (meaning a
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
      if (var1 in param_variables and var2 not in graph.params_vars or
          var1 not in param_variables and var2 in graph.params_vars or
          (isinstance(var2, core.Literal) and var1 not in input_vars)):
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
      assert first_output_var in graph.jaxpr.invars
      # Clearly the pattern equation is not an input or parameter
      return None

    assert isinstance(graph_eqn, jax.core.JaxprEqn)
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
        )
        if candidate_map is not None:
          # Sanity check
          assert all(candidate_map[p] in graph.params_vars
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
    root_eqn: core.JaxprEqn,
    pattern: "JaxprGraph",
    graph_matcher_rules: GraphMatcherComparator,
) -> Optional[Dict[core.Var, core.Var]]:
  """Tries to match the ``pattern`` in the Jaxpr graph from the ``root_eqn``.

  Args:
    graph: The :class:`~JaxprGraph` on which we are searching for matching
      equations.
    root_eqn: The equation in the graph, which is assumed to match the output
      equation of the pattern.
    pattern: The pattern, which we are trying to match.
    graph_matcher_rules: A :class:`~GraphMatcherRules` instance, which is used
      for determining equivalence of individual Jax primitives.

  Returns:
    The variable to variable mapping between the pattern and graph variable,
    if the pattern can be matched to the root equation, otherwise ``None``.
  """
  # Check the number of output variables match.
  if len(pattern.jaxpr.outvars) != len(root_eqn.outvars):
    return None
  # Set the current variables mapping to the output variables and the try to
  # check the match from there.
  return match_equations(
      graph=graph,
      current_variables_map=dict(zip(pattern.jaxpr.outvars,
                                     root_eqn.outvars)),
      reversed_eqns_to_match=tuple(reversed(pattern.jaxpr.eqns)),
      input_vars=pattern.jaxpr.invars,
      param_variables=pattern.params_vars,
      graph_matcher_rules=graph_matcher_rules,
  )


def find_layer_tags_and_patterns(
    graph: JaxprGraph,
    patterns_to_match: Sequence[GraphPattern],
    graph_matcher_rules: GraphMatcherComparator,
) -> Tuple[Tuple[tags.LayerTagEqn, ...],
           Dict[core.Var,
                Tuple[GraphPattern,
                      Dict[core.Var, core.Var],
                      Tuple[core.JaxprEqn, ...]]]]:
  """Tries to automatically match ``patterns_to_match`` in the Jaxpr graph.

  The method returns a pair of ``(manual_registrations, matches)``, where
  ``manual_registrations`` is a tuple of all layer tags that are already
  present in the graph and ``matches`` contains all newly discovered matches
  of any of the patterns. Each entry has as a key the variable of the graph
  corresponding to the output of the pattern, while each value is a triple
  ``(pattern, match_map, eqns)`` where ``pattern`` is the :class:`~JaxprGraph`
  of the pattern that has been matched, ``match_map`` is mapping the pattern
  variables to the corresponding graph variables and ``eqns`` is the sequence
  of all graph equations corresponding to the pattern equations.

  Args:
    graph: The :class:`~JaxprGraph` on which we are searching for matching
      equations.
    patterns_to_match: A sequence of different patterns that we want to find
    matches for in the graph.
    graph_matcher_rules: A :class:`~GraphMatcherRules` instance, which is used
      for determining equivalence of individual Jax primitives.

  Returns:
    The pair ``(manual_registrations, matches)``.
  """
  manual_registrations = graph.extract_manual_registrations()
  # This keeps track to any equations that are already in a pattern and hence
  # should not be part of any other.
  registered_equations = []
  # First add any manual registrations to this.
  for eqn in manual_registrations:
    assert isinstance(eqn.primitive, tags.LayerTag)
    for root_var in eqn.primitive.split_all_inputs(eqn.invars)[0]:
      assert root_var in graph.var_to_creation_op
      registered_equations.append(graph.var_to_creation_op[root_var])

  matches = {}
  # Loop through all equations in reverse and for each one check every pattern
  for eqn in reversed(graph.jaxpr.eqns):
    if eqn in registered_equations:
      continue
    for pattern in patterns_to_match:
      match_map = match_pattern(graph, eqn, pattern.graph, graph_matcher_rules)
      if match_map is not None:
        assert len(pattern.graph.outvars) == 1
        output_variable = match_map[pattern.graph.outvars[0]]
        # Extract all equations from the pattern and add them to the already
        # registered equations.
        match_eqns = []
        for k, v in match_map.items():
          if k not in pattern.graph.jaxpr.invars:
            creation_op = graph.var_to_creation_op[v]
            assert isinstance(creation_op, core.JaxprEqn)
            match_eqns.append(creation_op)
            registered_equations.append(match_eqns[-1])
        # Add the match
        matches[output_variable] = (pattern, match_map, tuple(match_eqns))
        break

  return manual_registrations, matches


def read_env(
    env: Mapping[core.Var, chex.Array],
    var: Union[core.Literal, core.Var, Sequence[core.Var]],
) -> Union[float, chex.Array, Sequence[chex.Array]]:
  """Reads from the variable-to-array environment during tracing."""
  if isinstance(var, (list, tuple)):
    return jax.tree_map(lambda x: read_env(env, x), var)
  elif isinstance(var, core.Literal):
    # Literals are values baked into the Jaxpr
    return var.val
  elif isinstance(var, core.Var):
    return env[var]
  else:
    raise NotImplementedError()


def write_env(
    env: MutableMapping[core.Var, chex.Array],
    var: Union[core.Var, List[core.Var]],
    val: Union[chex.Array, List[chex.Array]],
) -> None:
  """Writes to the variable-to-array environment during tracing."""
  if isinstance(var, tuple):
    raise NotImplementedError()
  if isinstance(var, list):
    if not isinstance(val, list):
      val = [val]
    return jax.tree_map(lambda x, y: write_env(env, x, y), var, val)
  elif isinstance(var, (core.Literal, core.Var)):
    env[var] = val
  else:
    raise NotImplementedError()


def clean_jaxpr_eqns(
    jaxpr: core.Jaxpr,
    preserve_tags: bool = True
) -> Iterator[core.JaxprEqn]:
  """Runs dead code elimination on a Jaxpr, retaining loss and layer tags."""
  eqns = []
  dependants = set(jaxpr.outvars)
  for eqn in reversed(jaxpr.eqns):
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
                           if not isinstance(v, core.Literal))
      dependants = dependants.union(new_dependants)
  # Dependants should only be invars
  dependants = dependants - set(jaxpr.invars + jaxpr.constvars)

  if dependants:
    raise ValueError("Something went wrong with the dead code elimination.")
  return reversed(eqns)


def broadcast_merger(f: utils.Func) -> utils.Func:
  """Transforms ``f`` by merging any consecutive broadcasts in its Jaxpr."""

  def read_with_delayed_evaluation(env, var):
    if isinstance(var, (list, tuple)):
      return jax.tree_map(lambda x: read_with_delayed_evaluation(env, x), var)
    elif isinstance(var, core.Literal):
      # Literals are values baked into the Jaxpr
      return var.val
    elif isinstance(var, core.Var):
      r = env[var]
      if isinstance(r, (jnp.ndarray, np.ndarray)):
        return r
      elif isinstance(r, Callable):
        y = r()
        if isinstance(y, list):
          assert len(y) == 1
          y = y[0]
        assert isinstance(y, jnp.ndarray)
        env[var] = y
        return y
    raise NotImplementedError()

  @functools.wraps(f)
  def merged_func(*func_args: Any) -> Any:
    typed_jaxpr, out_avals = jax.make_jaxpr(f, return_shape=True)(*func_args)
    out_tree = jax.tree_structure(out_avals)
    jaxpr, consts = typed_jaxpr.jaxpr, typed_jaxpr.literals

    # Mapping from variable -> value
    env = {}
    read = functools.partial(read_with_delayed_evaluation, env)
    write = functools.partial(write_env, env)

    # Bind args and consts to environment
    flat_args = jax.tree_flatten(func_args)[0]
    write(jaxpr.invars, flat_args)
    write(jaxpr.constvars, consts)

    # Bind args and consts to environment
    write(jaxpr.invars, flat_args)
    write(jaxpr.constvars, consts)

    # Loop through equations and evaluate primitives using `bind`
    broadcasts_outputs = {}
    for eqn in clean_jaxpr_eqns(jaxpr):
      # We ignore broadcasting of constants
      if (eqn.primitive.name == "broadcast_in_dim" and
          not all(isinstance(v, core.Literal) for v in eqn.invars)):
        if eqn.invars[0] in broadcasts_outputs:
          x, dims = broadcasts_outputs[eqn.invars[0]]
          kept_dims = eqn.params["broadcast_dimensions"]
          kept_dims = [kept_dims[d] for d in dims]
          # In order not to compute any un-needed broadcasting we instead put
          # in a function for delayed evaluation.
          write(eqn.outvars, [functools.partial(
              lax.broadcast_in_dim, x, eqn.params["shape"], kept_dims)])
          broadcasts_outputs[eqn.outvars[0]] = (x, kept_dims)
        else:
          input_values = read(eqn.invars)
          # In order not to compute any un-needed broadcasting we instead put
          # in a function for delayed evaluation.
          write(eqn.outvars, [functools.partial(
              eval_jaxpr_eqn, eqn, input_values)])
          broadcasts_outputs[eqn.outvars[0]] = (
              (input_values[0], eqn.params["broadcast_dimensions"]))
      else:
        write(eqn.outvars, eval_jaxpr_eqn(eqn, read(eqn.invars)))
    return jax.tree_unflatten(out_tree, read(jaxpr.outvars))

  return merged_func


#  _____            _     _             _   _
# |  __ \          (_)   | |           | | (_)
# | |__) |___  __ _ _ ___| |_ _ __ __ _| |_ _  ___  _ __  ___
# |  _  // _ \/ _` | / __| __| '__/ _` | __| |/ _ \| '_ \/ __|
# | | \ \  __/ (_| | \__ \ |_| | | (_| | |_| | (_) | | | \__ \
# |_|  \_\___|\__, |_|___/\__|_|  \__,_|\__|_|\___/|_| |_|___/
#              __/ |
#             |___/


def _dense(x: chex.Array, params: Sequence[chex.Array]) -> chex.Array:
  """Example of a dense layer function."""
  w, *opt_b = params
  y = jnp.matmul(x, w)
  return y if not opt_b else y + opt_b[0]


def _dense_parameter_extractor(
    eqns: Sequence[core.JaxprEqn],
) -> Mapping[str, Any]:
  """Extracts all parameters from the conv_general_dilated operator."""
  for eqn in eqns:
    if eqn.primitive.name == "dot_general":
      return dict(**eqn.params)
  assert False


dense_with_bias_pattern = GraphPattern(
    name="dense_with_bias",
    tag_primitive=tags.dense,
    precedence=0,
    compute_func=_dense,
    parameters_extractor_func=_dense_parameter_extractor,
    example_args=[np.zeros([11, 13]), [np.zeros([13, 7]), np.zeros([7])]],
)

dense_no_bias_pattern = GraphPattern(
    name="dense_no_bias",
    tag_primitive=tags.dense,
    precedence=1,
    compute_func=_dense,
    parameters_extractor_func=_dense_parameter_extractor,
    example_args=[np.zeros([11, 13]), [np.zeros([13, 7])]],
)


def _conv2d(x: chex.Array, params: Sequence[chex.Array]) -> chex.Array:
  """Example of a conv2d layer function."""
  w = params[0]
  y = lax.conv_general_dilated(
      x,
      w,
      window_strides=(2, 2),
      padding="SAME",
      dimension_numbers=("NHWC", "HWIO", "NHWC"))
  if len(params) == 1:
    # No bias
    return y
  # Add bias
  return y + params[1][None, None, None]


def _conv2d_parameter_extractor(
    eqns: Sequence[core.JaxprEqn],
) -> Mapping[str, Any]:
  """Extracts all parameters from the conv_general_dilated operator."""
  for eqn in eqns:
    if eqn.primitive.name == "conv_general_dilated":
      return dict(**eqn.params)
  assert False


conv2d_with_bias_pattern = GraphPattern(
    name="conv2d_with_bias",
    tag_primitive=tags.conv2d,
    precedence=0,
    compute_func=_conv2d,
    parameters_extractor_func=_conv2d_parameter_extractor,
    example_args=[np.zeros([2, 8, 8, 5]),
                  [np.zeros([3, 3, 5, 4]), np.zeros([4])]],
)

conv2d_no_bias_pattern = GraphPattern(
    name="conv2d_no_bias",
    tag_primitive=tags.conv2d,
    precedence=1,
    compute_func=_conv2d,
    parameters_extractor_func=_conv2d_parameter_extractor,
    example_args=[np.zeros([2, 8, 8, 5]), [np.zeros([3, 3, 5, 4])]],
)


def _scale_and_shift(
    x: chex.Array,
    params: Sequence[chex.Array],
    has_scale: bool,
    has_shift: bool,
) -> chex.Array:
  """Example of a scale and shift function."""
  if has_scale and has_shift:
    scale, shift = params
    return x * scale + shift
  elif has_scale:
    assert len(params) == 1
    return x * params[0]
  elif has_shift:
    assert len(params) == 1
    return x + params[0]
  else:
    raise ValueError("You must have either `has_scale` or `has_shift` set "
                     "to True.")


scale_and_shift_with_broadcast_pattern = GraphPattern(
    name="scale_and_shift_with_broadcast",
    tag_primitive=tags.scale_and_shift,
    precedence=0,
    compute_func=functools.partial(_scale_and_shift,
                                   has_scale=True, has_shift=True),
    parameters_extractor_func=
    lambda jaxpr: dict(has_scale=True, has_shift=True),
    example_args=[np.zeros([2, 13]), [np.zeros([13]), np.zeros([13])]],
)


scale_and_shift_no_broadcast_pattern = GraphPattern(
    name="scale_and_shift_no_broadcast",
    tag_primitive=tags.scale_and_shift,
    precedence=0,
    compute_func=functools.partial(_scale_and_shift,
                                   has_scale=True, has_shift=True),
    parameters_extractor_func=
    lambda jaxpr: dict(has_scale=True, has_shift=True),
    example_args=[np.zeros([13]), [np.zeros([13]), np.zeros([13])]],
)

scale_only_pattern = GraphPattern(
    name="scale_only",
    tag_primitive=tags.scale_and_shift,
    precedence=1,
    compute_func=functools.partial(_scale_and_shift,
                                   has_scale=True, has_shift=False),
    parameters_extractor_func=
    lambda jaxpr: dict(has_scale=True, has_shift=False),
    example_args=[np.zeros([2, 13]), [np.zeros([13])]],
)

shift_only_pattern = GraphPattern(
    name="shift_only",
    tag_primitive=tags.scale_and_shift,
    precedence=2,
    compute_func=functools.partial(_scale_and_shift,
                                   has_scale=False, has_shift=True),
    parameters_extractor_func=
    lambda jaxpr: dict(has_scale=False, has_shift=True),
    example_args=[np.zeros([2, 13]), [np.zeros([13])]],
)


def _normalization_haiku(
    inputs: Sequence[chex.Array],
    params: Sequence[chex.Array],
    has_scale: bool,
    has_shift: bool,
) -> chex.Array:
  """Example of normalization as is defined in Haiku."""
  if len(params) not in (1, 2):
    raise ValueError("The inputs to the `normalization_haiku` computation must "
                     f"have either 1 or 2 parameters, but got {len(params)}.")
  [inputs, rsqrt_var] = inputs
  inv = params[0] * rsqrt_var if has_scale else rsqrt_var
  outputs = inputs * inv
  return outputs + params[-1] if has_shift else outputs


def _normalization_haiku_preprocessor(
    in_values: Sequence[chex.Array],
) -> Tuple[chex.Array, ...]:
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
    in_values: The standard Haiku normalization inputs.

  Returns:
    The canonical input to ``scale_and_shift`` and the parameters.
  """
  [inputs, rsqrt_var, *params] = in_values
  normalized_inputs = inputs * rsqrt_var
  return (normalized_inputs,) + tuple(params)


normalization_haiku_with_broadcast_pattern = GraphPattern(
    name="normalization_haiku_with_broadcast",
    tag_primitive=tags.scale_and_shift,
    precedence=0,
    compute_func=functools.partial(_normalization_haiku,
                                   has_scale=True, has_shift=True),
    parameters_extractor_func=
    lambda jaxpr: dict(has_scale=True, has_shift=True),
    example_args=[[np.zeros([2, 13]), np.zeros([2, 13])],
                  [np.zeros([13]), np.zeros([13])]],
    in_values_preprocessor=_normalization_haiku_preprocessor
)


normalization_haiku_no_broadcast_pattern = GraphPattern(
    name="normalization_haiku_no_broadcast",
    tag_primitive=tags.scale_and_shift,
    precedence=0,
    compute_func=functools.partial(_normalization_haiku,
                                   has_scale=True, has_shift=True),
    parameters_extractor_func=
    lambda jaxpr: dict(has_scale=True, has_shift=True),
    example_args=[[np.zeros([13]), np.zeros([13])],
                  [np.zeros([13]), np.zeros([13])]],
    in_values_preprocessor=_normalization_haiku_preprocessor
)


normalization_haiku_scale_only_pattern = GraphPattern(
    name="normalization_haiku_scale_only",
    tag_primitive=tags.scale_and_shift,
    precedence=1,
    compute_func=functools.partial(_normalization_haiku,
                                   has_scale=True, has_shift=False),
    parameters_extractor_func=
    lambda jaxpr: dict(has_scale=True, has_shift=False),
    example_args=[[np.zeros([2, 13]), np.zeros([2, 13])], [np.zeros([13])]],
)


DEFAULT_GRAPH_PATTERNS = (
    dense_with_bias_pattern,
    dense_no_bias_pattern,
    conv2d_with_bias_pattern,
    conv2d_no_bias_pattern,
    scale_and_shift_with_broadcast_pattern,
    scale_and_shift_no_broadcast_pattern,
    normalization_haiku_with_broadcast_pattern,
    normalization_haiku_no_broadcast_pattern,
    scale_only_pattern,
    normalization_haiku_scale_only_pattern,
    shift_only_pattern,
)


def auto_register_tags(
    func: utils.Func,
    func_args: utils.FuncArgs,
    params_index: int = 0,
    register_only_generic: bool = False,
    compute_only_loss_tags: bool = True,
    patterns_to_skip: Sequence[str] = (),
    allow_multiple_registrations: bool = False,
    graph_matcher_rules: GraphMatcherComparator = GraphMatcherComparator(),
    graph_patterns: Sequence[GraphPattern] = DEFAULT_GRAPH_PATTERNS,
) -> utils.Func:
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
    allow_multiple_registrations: Whether to raise an error if a parameter is
      registered with more than one layer tag.
    graph_matcher_rules: A :class:`~GraphMatcherRules` instance, which is used
      for determining equivalence of individual Jax primitives.
    graph_patterns: A sequence of :class:`~GraphPattern` objects, which contain
      all patterns to use, in order of precedence, which to try to find in the
      graph before registering a parameter with a generic layer tag.
  Returns:
    A transformed function as described above.
  """
  graph = make_jax_graph(
      func=broadcast_merger(func),
      func_args=func_args,
      params_index=params_index,
      graph_name="main",
  )

  # Extract the sub-graph that leads to losses
  sub_graph = graph.ancestors_sub_graph(graph.losses_eqns)
  patterns = () if register_only_generic else  tuple(
      pattern for pattern in graph_patterns
      if pattern.name not in patterns_to_skip)
  manual, matches = find_layer_tags_and_patterns(
      sub_graph, patterns, graph_matcher_rules)

  tagged_params = {}
  pattern_counters = {}
  # Manual registrations
  for manual_eqn in manual:
    assert isinstance(manual_eqn.primitive, tags.LayerTag)
    n = pattern_counters.get(manual_eqn.primitive.name, 0)
    pattern_counters[manual_eqn.primitive.name] = n + 1
    for p in manual_eqn.primitive.split_all_inputs(manual_eqn.invars)[2]:
      assert p in sub_graph.params_vars
      tag_str = f"Manual[{manual_eqn.primitive.name}_{n}]"
      if p in tagged_params:
        if not allow_multiple_registrations:
          raise ValueError(f"Parameter {p} has been registered manually more "
                           f"than once - {tagged_params[p]} and {tag_str}, but "
                           f"`allow_multiple_registrations=False`.")
        tag_str = f"{tagged_params[p]}|{tag_str}"
      tagged_params[p] = tag_str
  # Automatically detect registrations
  for pattern, variables_map, _ in matches.values():
    n = pattern_counters.get(pattern.name, 0)
    pattern_counters[pattern.name] = n + 1
    for pattern_p in pattern.graph.params_vars:
      p = variables_map[pattern_p]
      assert p in sub_graph.params_vars
      tag_str = f"Auto[{pattern.name}_{n}]"
      if p in tagged_params:
        if not allow_multiple_registrations:
          raise ValueError(f"Parameter {p} has been matched a second time - "
                           f"{tagged_params[p]} and {tag_str}, but "
                           f"`allow_multiple_registrations=False`.")
        tag_str = f"{tagged_params[p]}|{tag_str}"
      tagged_params[p] = tag_str

  params_labels = [tagged_params.get(p, "Orphan") for p in graph.params_vars]
  logging.info("=" * 50)
  logging.info("Graph parameter registrations:")
  logging.info(pprint.pformat(
      jax.tree_unflatten(graph.params_tree, params_labels)))
  logging.info("=" * 50)

  # Construct a function with all of the extra tag registrations
  @functools.wraps(func)
  def wrapped_auto_registered(*args: Any) -> Any:
    flat_args, _ = jax.tree_flatten(args)
    # Mapping from variable -> value
    env = {}

    read = functools.partial(read_env, env)
    write = functools.partial(write_env, env)

    def tag(var):
      match = matches.get(var)
      if match is not None:
        pattern_, variables_map_, match_eqns_ = match
        values_map = {k: read(variables_map_[k]) for k in variables_map_}
        val = pattern_.tag_ctor(match_eqns_, values_map)
        env[var] = val

    # Bind args and consts to environment
    write(graph.jaxpr.invars, flat_args)
    write(graph.jaxpr.constvars, graph.consts)

    # Register any orphan parameters as generic
    for param in graph.params_vars:
      if param not in tagged_params:
        write(param, tags.register_generic(read(param)))

    # Set the correct output variables
    if compute_only_loss_tags:
      output_vars = []
      for eqn in graph.losses_eqns:
        # Do not include any dropped variables as they are always mapped to
        # the same value.
        output_vars.append(
            [v for v in eqn.outvars if not isinstance(v, jax.core.DropVar)])
      output_vars, out_tree = jax.tree_flatten(output_vars)
    else:
      output_vars = graph.jaxpr.outvars
      out_tree = graph.out_tree

    # Loop through equations and evaluate primitives using `bind`
    losses_evaluated = 0
    for eqn in graph.jaxpr.eqns:
      out = eqn.outvars if eqn.primitive.multiple_results else eqn.outvars[0]
      write(out, eval_jaxpr_eqn(eqn, read(eqn.invars)))
      jax_util.safe_map(tag, eqn.outvars)

      # If we want to output only tagged losses
      if isinstance(eqn.primitive, tags.LossTag):
        losses_evaluated += 1
      if compute_only_loss_tags and len(graph.losses_eqns) == losses_evaluated:
        break

    outputs = read(output_vars)
    return jax.tree_unflatten(out_tree, outputs)
  return wrapped_auto_registered
