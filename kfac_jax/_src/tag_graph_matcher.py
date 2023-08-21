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
import dataclasses
import functools
import itertools
import pprint
from typing import Any, Callable, Mapping, Optional, Sequence, TypeVar, Tuple, Union, Dict, Set

from absl import logging
import immutabledict
import jax
import jax.numpy as jnp
from kfac_jax._src import layers_and_loss_tags as tags
from kfac_jax._src import utils
import numpy as np

HIGHER_ORDER_NAMES = ("cond", "while", "scan", "xla_call", "xla_pmap")

# Types for annotation
Array = utils.Array
PyTreeDef = utils.PyTreeDef
Var = jax.core.Var
Vars = Sequence[Var]
Jaxpr = jax.core.Jaxpr
ClosedJaxpr = jax.core.ClosedJaxpr
JaxprEqn = jax.core.JaxprEqn
JaxprEqns = Sequence[JaxprEqn]
T = TypeVar("T")
J = TypeVar("J", Jaxpr, ClosedJaxpr)
JaxprOrClosedJaxpr = Union[Jaxpr, ClosedJaxpr]
EquivalenceFunction = Callable[[JaxprEqn, JaxprEqn], bool]
MakeVarFunc = Callable[[jax.core.AbstractValue], Var]
VarProcessor = Callable[[Vars, MakeVarFunc], Tuple[Vars, JaxprEqns]]
PatternComputeFunc = Callable[[Array, Sequence[Array]], Array]
ParameterExtractorFunc = Callable[[JaxprEqns], Mapping[str, Any]]
TagCtor = Callable[[Vars, Vars, JaxprEqns, MakeVarFunc], JaxprEqn]


def eval_jaxpr_eqn(eqn: JaxprEqn, in_values: Vars) -> Var:
  """Computes the outputs of the given Jaxpr equation."""
  subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
  with jax.core.source_info_util.user_context(
      eqn.source_info.traceback):
    return eqn.primitive.bind(*subfuns, *in_values, **bind_params)


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
    equation1: JaxprEqn,
    equation2: JaxprEqn,
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
    jaxpr: The underlying :class:`jax.core.Jaxpr` part of ``self.closed_jaxpr``.
    consts: The underlying constants part ``self.closed_jaxpr``.
    outvars: The output variables of the underlying :class:`jax.core.Jaxpr` part
      of ``self.closed_jaxpr``.
  """
  name: str
  closed_jaxpr: ClosedJaxpr
  params_tree: PyTreeDef
  params_vars: Vars
  out_tree: PyTreeDef
  tag_ctor: Optional[TagCtor]
  # Until we stop supporting Python 3.7 we can't use @functools.cached_property,
  # so we set these attributes in __post_init__
  losses_eqns: Tuple[tags.LossTagEqn, ...] = ()
  var_to_creation_op: immutabledict.immutabledict = None  # pytype:disable=annotation-type-mismatch
  manual_registrations: Tuple[tags.LayerTagEqn, ...] = ()

  def __post_init__(self):
    losses_eqns = tuple(
        eqn for eqn in self.closed_jaxpr.jaxpr.eqns
        if isinstance(eqn.primitive, tags.LossTag)
    )
    var_to_creation_op = immutabledict.immutabledict(
        sum(([(var, eqn) for var in eqn.outvars]
             for eqn in self.jaxpr.eqns), [])
    )
    registered_tags = []
    for eqn in self.jaxpr.eqns:
      if isinstance(eqn.primitive, tags.LayerTag):
        for param in eqn.primitive.split_all_inputs(eqn.invars)[2]:
          if param not in self.params_vars:
            raise ValueError(f"One of the parameters of the manual layer "
                             f"registration equation: {eqn} is not part of the "
                             f"parameters of the global function.")
        registered_tags.append(eqn)
    manual_registrations = tuple(registered_tags)

    object.__setattr__(self, "losses_eqns", losses_eqns)
    object.__setattr__(self, "var_to_creation_op", var_to_creation_op)
    object.__setattr__(self, "manual_registrations", manual_registrations)

  @property
  def jaxpr(self) -> Jaxpr:
    return self.closed_jaxpr.jaxpr

  @property
  def consts(self) -> Sequence[Any]:
    return self.closed_jaxpr.consts

  @property
  def outvars(self) -> Vars:
    return self.jaxpr.outvars  # pytype:disable=bad-return-type

  def sub_graph_eqns(self, root_vars: Vars, leaf_vars: Vars) -> JaxprEqns:
    """Returns the sub-graph equations between root vars and leaf vars."""
    eqns = []
    # Extract the subgraph equations such that they both depend on root_vars and
    # leaf_vars depends on them

    to_process_eqns = [self.var_to_creation_op[v] for v in leaf_vars]
    processed_vars = set()
    while to_process_eqns:
      next_eqn = to_process_eqns.pop()
      eqns.append(next_eqn)
      for v in next_eqn.invars:
        if (not isinstance(v, jax.core.Literal) and v not in root_vars and
            v not in processed_vars and v in self.var_to_creation_op):
          to_process_eqns.append(self.var_to_creation_op[v])
          processed_vars.add(v)
    return tuple(eqns)
  #
  # @functools.cached_property
  # def losses_eqns(self) -> Tuple[tags.LossTagEqn, ...]:
  #   return tuple(
  #       eqn for eqn in self.closed_jaxpr.jaxpr.eqns
  #       if isinstance(eqn.primitive, tags.LossTag)
  #   )
  #
  # @functools.cached_property
  # def var_to_creation_op(self) -> immutabledict.immutabledict:
  #   return immutabledict.immutabledict(
  #       sum(([(var, eqn) for var in eqn.outvars]
  #            for eqn in self.jaxpr.eqns), []))
  #
  # @functools.cached_property
  # def manual_registrations(self) -> Tuple[tags.LayerTagEqn, ...]:
  #   """Returns all manually registered tags."""
  #   registered_tags = []
  #   for eqn in self.jaxpr.eqns:
  #     if isinstance(eqn.primitive, tags.LayerTag):
  #       for param in eqn.primitive.split_all_inputs(eqn.invars)[2]:
  #         if param not in self.params_vars:
  #           raise ValueError("One of the parameters of the manual layer "
  #                            f"registration equation: {eqn} is not part of "
  #                            "the parameters of the global function.")
  #       registered_tags.append(eqn)
  #   return tuple(registered_tags)


def make_jax_graph(
    func: utils.Func,
    func_args: utils.FuncArgs,
    params_index: Union[int, Sequence[int]],
    name: str,
    compute_only_loss_tags: bool,
    clean_broadcasts: bool,
    tag_ctor: Optional[TagCtor] = None,
) -> JaxprGraph:
  """Creates a :class:`~JaxGraph` instance from the provided function and arguments."""
  # we always put static_args as the third argument
  func_args_without_static_args = tuple([arg for idx, arg in enumerate(list(func_args)) if idx != 2])
  in_tree = jax.tree_util.tree_structure(func_args_without_static_args)
  closed_jaxpr, out_shapes = jax.make_jaxpr(func, return_shape=True, static_argnums=[2])(*func_args)
  # closed_jaxpr, out_shapes = jax.make_jaxpr(func, return_shape=True)(*func_args)

  if compute_only_loss_tags:
    make_var_func = jax.core.gensym([closed_jaxpr.jaxpr])
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
            v for v in eqn.invars if not isinstance(v, jax.core.Literal))

    consts_i = [i for i, c in enumerate(closed_jaxpr.jaxpr.constvars)
                if c in sub_graph_vars]
    closed_jaxpr = ClosedJaxpr(
        jaxpr=closed_jaxpr.jaxpr.replace(
            eqns=eqns[::-1],
            constvars=[closed_jaxpr.jaxpr.constvars[i] for i in consts_i],
            outvars=loss_tags_output_vars[::-1],
        ),
        consts=[closed_jaxpr.consts[i] for i in consts_i],
    )
    out_shapes = [jax.ShapeDtypeStruct(shape=v.aval.shape, dtype=v.aval.dtype)
                  for v in closed_jaxpr.jaxpr.outvars]  # pytype:disable=attribute-error

  if clean_broadcasts:
    closed_jaxpr: ClosedJaxpr = merge_broadcasts_jaxpr(closed_jaxpr)  # pytype:disable=annotation-type-mismatch
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
    jaxpr: The underlying :class:`jax.core.Jaxpr` represented by the pattern.
    param_vars: The list of :class:`jax.core.Var` that correspond to parameters
      in the pattern.
    graph: A :class:`JaxprGraph` representation of the pattern.
  """
  name: str
  tag_primitive: tags.LayerTag
  compute_func: PatternComputeFunc
  parameters_extractor_func: ParameterExtractorFunc
  example_args: utils.FuncArgs
  in_values_preprocessor: Optional[VarProcessor] = None
  # Until we stop supporting Python 3.7 we can't use @functools.cached_property,
  # so we set this attribute in the property
  _graph: Optional[JaxprGraph] = None

  @property
  def jaxpr(self) -> Jaxpr:
    return self.graph.jaxpr

  @property
  def param_vars(self) -> Vars:
    return self.graph.params_vars

  @property
  def graph(self) -> JaxprGraph:
    """A :class:`JaxprGraph` representation of the pattern."""
    if self._graph is None:
      jnp_args = jax.tree_util.tree_map(jnp.asarray, self.example_args)
      graph = make_jax_graph(
          func=self.compute_func,
          func_args=jnp_args,
          params_index=1,
          name=self.name,
          compute_only_loss_tags=False,
          clean_broadcasts=True,
      )
      object.__setattr__(self, "_graph", graph)
    assert self._graph is not None
    return self._graph

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
        primitive=self.tag_primitive,
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
  # Until we stop supporting Python 3.7 we can't use @functools.cached_property,
  # so we set these attributes in __post_init__
  output_var: Var = None  # pytype:disable=annotation-type-mismatch
  param_graph_variables: Vars = ()

  def __post_init__(self):
    # Until we stop supporting Python 3.7 we can't use
    # @functools.cached_property, so we set here additional attributes.
    output_var = self.variables_map[self.pattern.jaxpr.outvars[0]]
    param_graph_variables = [self.variables_map[p]
                             for p in self.pattern.graph.params_vars]

    object.__setattr__(self, "output_var", output_var)
    object.__setattr__(self, "param_graph_variables", param_graph_variables)

  @property
  def name(self) -> str:
    return self.pattern.name
  #
  # @functools.cached_property
  # def output_var(self) -> Var:
  #   return self._variables_map[self.pattern.jaxpr.outvars[0]]
  #
  # @functools.cached_property
  # def param_graph_variables(self) -> Vars:
  #   return [self._variables_map[p] for p in self.pattern.graph.params_vars]

  def create_eqn(
      self,
      env: Dict[Var, Var],
      make_var_func: MakeVarFunc,
  ) -> JaxprEqns:
    """Creates a new ``JaxprEqn`` for the this match."""
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
) -> Optional[Dict[Var, Var]]:
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
          (isinstance(var2, jax.core.Literal) and var1 not in input_vars)):
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
    root_eqn: JaxprEqn,
    pattern: GraphPattern,
    graph_matcher_rules: GraphMatcherComparator,
) -> Optional[GraphMatch]:
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
  match_variables_map = match_equations(
      graph=graph,
      current_variables_map=dict(zip(pattern.jaxpr.outvars,
                                     root_eqn.outvars)),
      reversed_eqns_to_match=tuple(reversed(pattern.jaxpr.eqns)),
      input_vars=pattern.jaxpr.invars,
      param_variables=pattern.param_vars,
      graph_matcher_rules=graph_matcher_rules,
  )
  if match_variables_map is None:
    return None

  # Extract all the graph equations corresponding to the pattern.
  graph_eqns = []
  for k, v in match_variables_map.items():
    if (k not in pattern.graph.jaxpr.invars and
        not isinstance(v, jax.core.Literal)):
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
) -> Tuple[Tuple[tags.LayerTagEqn, ...], Dict[Var, GraphMatch]]:
  """Tries to automatically match ``patterns_to_match`` in the Jaxpr graph.

  The method returns a pair of ``(manual_registrations, matches)``, where
  ``manual_registrations`` is a tuple of all layer tags that are already
  present in the graph and ``matches`` contains all newly discovered matches
  of any pattern. Each entry has as a key the variable of the graph
  corresponding to the output of the pattern, while each value is a triple
  ``(pattern, match_map, eqns)`` where ``pattern`` is the :class:`~JaxprGraph`
  of the pattern that has been matched, ``match_map`` is mapping the pattern
  variables to the corresponding graph variables and ``eqns`` is the sequence
  of all graph equations corresponding to the pattern equations.

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

  Returns:
    The pair ``(manual_registrations, matches)``.
  """
  # This list keeps track to any equations that are already in a pattern and
  # hence should not be part of any other.
  registered_equations = []

  # First add any manual registrations to this.
  for eqn in graph.manual_registrations:
    assert isinstance(eqn.primitive, tags.LayerTag)
    outputs, inputs, params = eqn.primitive.split_all_inputs(eqn.invars)
    for manual_eqn in graph.sub_graph_eqns(inputs + params, outputs):
      registered_equations.append(manual_eqn)

  matches = {}
  # Loop through all equations in reverse and for each one check every pattern
  for eqn in reversed(eqns_for_patterns):
    if eqn in registered_equations or eqn.primitive.name in HIGHER_ORDER_NAMES:
      continue

    for pattern in graph_patterns:
      match = match_pattern(
          graph=graph,
          root_eqn=eqn,
          pattern=pattern,
          graph_matcher_rules=graph_matcher_rules,
      )
      if match is not None:
        # Add all the match equations to the registered equations
        registered_equations.extend(match.graph_eqns)
        # Add the match to the mapping of graph matches
        matches[match.output_var] = match
        break

  return graph.manual_registrations, matches


def read_env(
    env: Mapping[Var, Array],
    var: Union[jax.core.Literal, Vars],
) -> Union[float, Array, Sequence[Array]]:
  """Reads from the variable-to-array environment during tracing."""
  if isinstance(var, (list, tuple)):
    return jax.tree_util.tree_map(lambda x: read_env(env, x), var)
  elif isinstance(var, jax.core.Literal):
    # Literals are values baked into the Jaxpr
    return var.val
  elif isinstance(var, Var):
    return env[var]
  else:
    raise NotImplementedError()


def write_env(
    env: Dict[Var, Array],
    var: Union[Var, Vars],
    val: Union[Array, Sequence[Array]],
):
  """Writes to the variable-to-array environment during tracing."""
  if isinstance(var, tuple):
    raise NotImplementedError()
  if isinstance(var, list):
    if not isinstance(val, list):
      val = [val]
    return jax.tree_util.tree_map(lambda x, y: write_env(env, x, y), var, val)
  elif isinstance(var, (jax.core.Literal, Var)):
    env[var] = val  # pytype: disable=container-type-mismatch  # numpy-scalars
  else:
    raise NotImplementedError()


def to_closed_jaxpr(jaxpr: JaxprOrClosedJaxpr) -> ClosedJaxpr:
  if isinstance(jaxpr, Jaxpr):
    return ClosedJaxpr(jaxpr=jaxpr, consts=[])
  else:
    return jaxpr


def to_jaxpr_or_closed_jaxpr(
    closed_jaxpr: ClosedJaxpr,
    original: J,
) -> J:
  if isinstance(original, Jaxpr):
    return closed_jaxpr.jaxpr
  else:
    return closed_jaxpr


def apply_to_higher_order_primitives(eqn, func, *args, **kwargs):
  """Applies `func` only to higher order Jax primitives."""
  if eqn.primitive.name not in HIGHER_ORDER_NAMES:
    return eqn
  elif eqn.primitive.name == "cond":
    params = dict(**eqn.params)
    params["branches"] = tuple(
        func(branch, *args, **kwargs) for branch in params["branches"]
    )
    return eqn.replace(params=params)
  elif eqn.primitive.name == "while":
    params = dict(**eqn.params)
    params["body_jaxpr"] = func(params["body_jaxpr"], *args, **kwargs)
    return eqn.replace(params=params)
  elif eqn.primitive.name == "scan":
    params = dict(**eqn.params)
    params["jaxpr"] = func(params["jaxpr"], *args, **kwargs)
    return eqn.replace(params=params)
  elif eqn.primitive.name in ("xla_call", "xla_pmap"):
    params = dict(**eqn.params)
    params["call_jaxpr"] = func(params["call_jaxpr"], *args, **kwargs)
    return eqn.replace(params=params)
  else:
    raise NotImplementedError()


def clean_jaxpr(jaxpr: J, preserve_tags: bool = True) -> J:
  """Runs dead code elimination on a Jaxpr, retaining loss and layer tags."""
  closed_jaxpr = to_closed_jaxpr(jaxpr)
  eqns = []
  dependants = set(closed_jaxpr.jaxpr.outvars)
  for eqn in reversed(closed_jaxpr.jaxpr.eqns):
    eqn = apply_to_higher_order_primitives(
        eqn, clean_jaxpr, preserve_tags=preserve_tags)

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
                           if not isinstance(v, jax.core.Literal))
      dependants = dependants.union(new_dependants)
  # Dependants should only be invars
  dependants = dependants - set(closed_jaxpr.jaxpr.invars +
                                closed_jaxpr.jaxpr.constvars)

  if dependants:
    raise ValueError("Something went wrong with the dead code elimination.")

  closed_jaxpr = ClosedJaxpr(
      jaxpr=closed_jaxpr.jaxpr.replace(eqns=list(reversed(eqns))),
      consts=closed_jaxpr.consts,
  )
  return to_jaxpr_or_closed_jaxpr(closed_jaxpr, jaxpr)


def merge_broadcasts_jaxpr(jaxpr: J) -> J:
  """Merges consecutive broadcasts in the given Jaxpr."""
  closed_jaxpr = clean_jaxpr(to_closed_jaxpr(jaxpr))

  broadcasts_outputs = {}
  eqns = list()

  for eqn in closed_jaxpr.jaxpr.eqns:
    eqn = apply_to_higher_order_primitives(eqn, merge_broadcasts_jaxpr)

    # We ignore broadcasting of constants
    if (eqn.primitive.name == "broadcast_in_dim" and
        not all(isinstance(v, jax.core.Literal) for v in eqn.invars)):
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
        if not isinstance(v, jax.core.Literal) and v in broadcasts_outputs:
          eqns.append(broadcasts_outputs[v])
      eqns.append(eqn)
  closed_jaxpr = ClosedJaxpr(
      jaxpr=closed_jaxpr.jaxpr.replace(eqns=eqns),
      consts=closed_jaxpr.consts
  )
  return to_jaxpr_or_closed_jaxpr(closed_jaxpr, jaxpr)


#  _____            _     _             _   _
# |  __ \          (_)   | |           | | (_)
# | |__) |___  __ _ _ ___| |_ _ __ __ _| |_ _  ___  _ __  ___
# |  _  // _ \/ _` | / __| __| '__/ _` | __| |/ _ \| '_ \/ __|
# | | \ \  __/ (_| | \__ \ |_| | | (_| | |_| | (_) | | | \__ \
# |_|  \_\___|\__, |_|___/\__|_|  \__,_|\__|_|\___/|_| |_|___/
#              __/ |
#             |___/


def _dense(x: Array, params: Sequence[Array]) -> Array:
  """Example of a dense layer function."""
  w, *opt_b = params
  y = jnp.matmul(x, w)
  return y if not opt_b else y + opt_b[0]


def _dense_with_reshape(x: Array, params: Sequence[Array],) -> Array:
  w, b = params
  y = jnp.matmul(x, w)
  return y + b.reshape([1, b.size])


def _dense_parameter_extractor(
    eqns: Sequence[JaxprEqn],
) -> Mapping[str, Any]:
  """Extracts all parameters from the `dot_general` operator."""
  for eqn in eqns:
    if eqn.primitive.name == "dot_general":
      return dict(**eqn.params)
  assert False


def _make_dense_pattern(
    with_bias: bool,
    reshape: bool,
    in_dim: int = 13,
    out_dim: int = 7,
) -> GraphPattern:
  x_shape = [2, in_dim]
  p_shapes = ([[in_dim, out_dim], [out_dim]] if with_bias else
              [[in_dim, out_dim]])
  return GraphPattern(
      name="dense_with_bias" if with_bias else "dense_no_bias",
      tag_primitive=tags.dense,
      compute_func=_dense_with_reshape if reshape else _dense,
      parameters_extractor_func=_dense_parameter_extractor,
      example_args=[np.zeros(x_shape), [np.zeros(s) for s in p_shapes]],
  )


def _conv2d(x: Array, params: Sequence[Array]) -> Array:
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
  return y + params[1][None, None, None]


def _conv2d_parameter_extractor(
    eqns: Sequence[JaxprEqn],
) -> Mapping[str, Any]:
  """Extracts all parameters from the `conv_general_dilated` operator."""
  for eqn in eqns:
    if eqn.primitive.name == "conv_general_dilated":
      return dict(**eqn.params)
  assert False


def _make_conv2d_pattern(
    with_bias: bool,
) -> GraphPattern:
  x_shape = [2, 8, 8, 5]
  p_shapes = ([[3, 3, 5, 4], [4]] if with_bias else
              [[3, 3, 5, 4]])
  return GraphPattern(
      name="conv2d_with_bias" if with_bias else "conv2d_no_bias",
      tag_primitive=tags.conv2d,
      compute_func=_conv2d,
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
      tag_primitive=tags.scale_and_shift,
      compute_func=functools.partial(
          _scale_and_shift, has_scale=has_scale, has_shift=has_shift),
      parameters_extractor_func=
      lambda jaxpr: dict(has_scale=has_scale, has_shift=has_shift),
      example_args=[np.zeros(x_shape), [np.zeros(s) for s in p_shapes]],
  )


def _normalization_haiku(
    inputs: Sequence[Array],
    params: Sequence[Array],
    has_scale: bool,
    has_shift: bool,
) -> Array:
  """Example of normalization as is defined in Haiku."""
  if len(params) not in (1, 2):
    raise ValueError("The inputs to the `normalization_haiku` computation must "
                     f"have either 1 or 2 parameters, but got {len(params)}.")
  [inputs, rsqrt_var] = inputs
  inv = params[0] * rsqrt_var if has_scale else rsqrt_var
  outputs = inputs * inv
  return outputs + params[-1] if has_shift else outputs


def _normalization_haiku_preprocessor(
    in_vars: Vars,
    make_var_func: MakeVarFunc,
) -> Tuple[Vars, JaxprEqns]:
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


def _make_normalization_haiku_pattern(
    broadcast_ndim: int,
    p_dim: int = 13,
):
  assert broadcast_ndim >= 0
  x_shape = [i + 2 for i in range(broadcast_ndim)] + [p_dim]
  return GraphPattern(
      name=f"normalization_haiku_broadcast_{broadcast_ndim}",
      tag_primitive=tags.scale_and_shift,
      compute_func=functools.partial(_normalization_haiku,
                                     has_scale=True, has_shift=True),
      parameters_extractor_func=
      lambda jaxpr: dict(has_scale=True, has_shift=True),
      example_args=[[np.zeros(x_shape), np.zeros(x_shape)],
                    [np.zeros([p_dim]), np.zeros([p_dim])]],
      in_values_preprocessor=_normalization_haiku_preprocessor
  )


DEFAULT_GRAPH_PATTERNS = (
    _make_dense_pattern(True, False),
    _make_dense_pattern(True, True),
    _make_dense_pattern(False, False),
    _make_conv2d_pattern(True),
    _make_conv2d_pattern(False),
    _make_scale_and_shift_pattern(1, True, True),
    _make_scale_and_shift_pattern(0, True, True),
    _make_normalization_haiku_pattern(1),
    _make_normalization_haiku_pattern(0),
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
      base_name: str,
      parent_equations: Sequence[Tuple[JaxprEqn, int]] = (),
  ):
    # assert isinstance(tag_eqn.primitive, tags.LayerTag)
    self.tag_eqn = tag_eqn
    self.base_name = base_name
    self.parent_equations = list(parent_equations)

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
  def bottom_level_parameters(self) -> Vars:
    """The bottom level variables of the tag location."""
    return self.tag_eqn.primitive.split_all_inputs(self.tag_eqn.invars)[2]  # pytype:disable=attribute-error

  @property
  def top_level_parameters(self) -> Vars:
    """The top level parameter variables of the tag location."""
    param_vars = self.bottom_level_parameters
    for eqn, _ in reversed(self.parent_equations):
      assert eqn.primitive.name in HIGHER_ORDER_NAMES
      if eqn.primitive.name == "cond":
        raise NotImplementedError()
      elif eqn.primitive.name == "scan":
        invars = eqn.params["jaxpr"].jaxpr.invars
      elif eqn.primitive.name == "while":
        invars = eqn.params["body_jaxpr"].jaxpr.invars
      elif eqn.primitive.name in ("xla_call", "xla_pmap"):
        invars = eqn.params["call_jaxpr"].invars
      else:
        raise NotImplementedError()
      p_indexes = [invars.index(p) for p in param_vars]
      param_vars = [eqn.invars[pi] for pi in p_indexes]
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
    self._flat_func = jax.core.jaxpr_as_fun(func_graph.closed_jaxpr)
    self._param_labels = self._compute_parameter_labels()

  def __call__(self, *args, **kwargs):
    # we always put static_args as the third argument
    args_without_static_args = tuple([arg for idx, arg in enumerate(list(args)) if idx != 2])
    flat_args = jax.tree_util.tree_leaves(args_without_static_args)
    flat_output = self._flat_func(*flat_args)
    return jax.tree_util.tree_unflatten(self._func_graph.out_tree, flat_output)

  def _compute_parameter_labels(self) -> Mapping[Var, Sequence[str]]:
    # Collect all registration for every tagged parameter
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
    logging.info("Graph parameter registrations:")
    logging.info(pprint.pformat(jax.tree_util.tree_unflatten(
        self._func_graph.params_tree, labels)))
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
) -> Tuple[JaxprGraph, Sequence[TagLocation]]:
  """Internal function for automatic registration of layer tags."""
  higher_counters = {
      "cond": 0,
      "while": 0,
      "scan": 0,
      "xla_call": 0,
      "xla_pmap": 0,
    }

  # Extract the sub-graph that leads to losses
  if register_only_until_losses:
    eqns_for_registration = []
    sub_graph_vars = set()
    for eqn in reversed(graph.jaxpr.eqns):
      if (eqn in graph.losses_eqns or
          any(v in sub_graph_vars for v in eqn.outvars)):
        eqns_for_registration.append(eqn)
        sub_graph_vars.update(
            v for v in eqn.invars if not isinstance(v, jax.core.Literal))
    eqns_for_registration = eqns_for_registration[::-1]
  else:
    eqns_for_registration = graph.jaxpr.eqns

  # Process all higher order primitives
  eqns = []
  tag_locations = []
  for eqn in graph.jaxpr.eqns:
    if (eqn not in eqns_for_registration or
        eqn.primitive.name not in HIGHER_ORDER_NAMES):
      eqns.append(eqn)
      continue

    eqn_name = eqn.primitive.name
    if eqn_name == "cond":
      sub_jaxprs = eqn.params["branches"]
    elif eqn_name == "while":
      sub_jaxprs = [eqn.params["body_jaxpr"]]
    elif eqn_name == "scan":
      sub_jaxprs = [eqn.params["jaxpr"]]
    elif eqn_name in ("xla_call", "xla_pmap"):
      sub_jaxprs = [eqn.params["call_jaxpr"]]
    else:
      raise NotImplementedError()

    final_jaxprs = []
    final_tag_locations = []
    for original_jaxpr in sub_jaxprs:
      sub_jaxpr = to_closed_jaxpr(original_jaxpr)
      params_vars = []
      for out_v, in_v in zip(eqn.invars, sub_jaxpr.jaxpr.invars):
        if out_v in graph.params_vars:
          params_vars.append(in_v)
      sub_graph, sub_tag_locations = _auto_register_tags(
          graph=JaxprGraph(
              name=graph.name + f"_{eqn_name}",
              closed_jaxpr=sub_jaxpr,
              params_tree=jax.tree_util.tree_structure(params_vars),
              params_vars=params_vars,
              out_tree=jax.tree_util.tree_structure(sub_jaxpr.jaxpr.outvars),
              tag_ctor=None,
          ),
          graph_matcher_rules=graph_matcher_rules,
          graph_patterns=graph_patterns,
          register_orphans=False,
          register_only_until_losses=False,
      )
      final_jaxprs.append(
          to_jaxpr_or_closed_jaxpr(sub_graph.closed_jaxpr, original_jaxpr))
      final_tag_locations.append(sub_tag_locations)

    if eqn_name == "cond":
      # TODO(botev): We need to check each branch has identical registrations
      raise NotImplementedError()
    else:
      # Extract the sub jaxpr parameter tag registrations and input vars
      [sub_tag_locations] = final_tag_locations  # pylint:disable=unbalanced-tuple-unpacking

    # Update the jaxpr parameter in the equation
    eqn_params = dict(**eqn.params)
    if eqn_name == "cond":
      eqn_params["branches"] = final_jaxprs
    elif eqn_name == "while":
      [eqn_params["body_jaxpr"]] = final_jaxprs  # pylint:disable=unbalanced-tuple-unpacking
    elif eqn_name == "scan":
      [eqn_params["jaxpr"]] = final_jaxprs  # pylint:disable=unbalanced-tuple-unpacking
    elif eqn_name in ("xla_call", "xla_pmap"):
      [eqn_params["call_jaxpr"]] = final_jaxprs  # pylint:disable=unbalanced-tuple-unpacking
    else:
      raise NotImplementedError()

    eqns.append(eqn.replace(params=eqn_params))
    del sub_graph, final_jaxprs, final_tag_locations

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
  manual_registrations, matches = find_layer_tags_and_patterns(
      graph=mid_graph,
      eqns_for_patterns=eqns_for_registration,
      graph_matcher_rules=graph_matcher_rules,
      graph_patterns=graph_patterns
  )

  tagged_params = set()
  # Automatically detected registrations in higher order primitives
  for tag_l in tag_locations:
    for p in tag_l.top_level_parameters:
      tagged_params.add(p)

  # Manual registrations
  for manual_eqn in manual_registrations:
    assert isinstance(manual_eqn.primitive, tags.LayerTag)
    for p in manual_eqn.primitive.split_all_inputs(manual_eqn.invars)[2]:
      tagged_params.add(p)

  # Automatically detect registrations
  for match in matches.values():
    for p in match.param_graph_variables:
      tagged_params.add(p)

  # Create the Jaxpr with all the tag registrations
  make_var_func = jax.core.gensym([mid_graph.jaxpr])
  eqns = list()
  env = {}
  pattern_counters = {}

  if register_orphans:
    for param in mid_graph.params_vars:
      if param not in tagged_params:
        orphan_p = make_var_func(param.aval)
        eqns.append(jax.core.new_jaxpr_eqn(
            invars=[param],
            outvars=[orphan_p],
            primitive=tags.generic,
            params={},
            effects=set(),
        ))
        env[param] = orphan_p
        tag_locations.append(TagLocation(eqns[-1], "Orphan"))

  for eqn in mid_graph.jaxpr.eqns:
    invars = [env.get(v, v) if isinstance(v, Var) else v
              for v in eqn.invars]
    eqns.append(eqn.replace(invars=invars))

    if isinstance(eqn.primitive, tags.LayerTag):
      # Mark manual registrations
      tag_name = eqn.primitive.name
      n = pattern_counters.get(tag_name, 0)
      pattern_counters[tag_name] = n + 1
      tag_locations.append(TagLocation(eqn, f"Manual[{tag_name}_{n}]"))

    for var in eqn.outvars:
      # Check if this is a match of a graph pattern
      match = matches.get(var)
      if match is not None:
        for additional_eqn in match.create_eqn(env, make_var_func):
          eqns.append(additional_eqn)

        # Mark automatic registration
        tag_name = eqns[-1].primitive.name
        n = pattern_counters.get(tag_name, 0)
        pattern_counters[tag_name] = n + 1
        tag_locations.append(TagLocation(eqns[-1], f"Auto[{tag_name}_{n}]"))

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
    allow_multiple_registrations: bool = False,
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
      func=func,
      func_args=func_args,
      params_index=params_index,
      name="main",
      compute_only_loss_tags=compute_only_loss_tags,
      clean_broadcasts=True,
  )

  patterns = () if register_only_generic else  tuple(
      pattern for pattern in graph_patterns
      if pattern.name not in patterns_to_skip)
  func_graph, tagged_locations = _auto_register_tags(
      graph=graph,
      graph_matcher_rules=graph_matcher_rules,
      graph_patterns=patterns,
      register_orphans=True,
      register_only_until_losses=True
  )

  func = TaggedFunction(
      func_graph=func_graph,
      tag_locations=tagged_locations,
  )
  func.print_parameter_tags()

  if not allow_multiple_registrations:
    func.check_multiple_registrations()

  return func
