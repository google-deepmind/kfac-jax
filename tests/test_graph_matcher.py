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
"""Testing the functionality of the graph matcher."""
import functools
from typing import Callable, Mapping

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import kfac_jax
from tests import models

Array = kfac_jax.utils.Array
Shape = kfac_jax.utils.Shape


class TestGraphMatcher(parameterized.TestCase):
  """Test class for the functions in `tag_graph_matcher.py`."""

  def check_equation_match(self, eqn1, vars_to_vars, vars_to_eqn):
    """Checks that equation is matched in the other graph."""
    eqn1_out_vars = [v for v in eqn1.outvars
                     if not isinstance(v, jax.core.DropVar)]
    eqn2_out_vars = [vars_to_vars[v] for v in eqn1_out_vars]
    eqns = [vars_to_eqn[v] for v in eqn2_out_vars]
    self.assertTrue(all(e == eqns[0] for e in eqns[1:]))
    eqn2 = eqns[0]

    self.assertEqual(eqn1.primitive, eqn2.primitive)
    if eqn1.primitive.name == "conv2d_tag":
      # params removed in https://github.com/google/jax/pull/14211
      skip_params = ["lhs_shape", "rhs_shape"]
    else:
      skip_params = []
    if eqn1.primitive.name == "cond":
      raise NotImplementedError()
    elif eqn1.primitive.name == "while":
      exclude_param = "body_jaxpr"
    elif eqn1.primitive.name == "scan":
      exclude_param = "jaxpr"
    elif eqn1.primitive.name in ("xla_call", "xla_pmap"):
      exclude_param = "call_jaxpr"
    else:
      exclude_param = ""
    # Check all eqn parameters
    for k in eqn1.params:
      if k != exclude_param and k not in skip_params:
        self.assertEqual(eqn1.params[k], eqn2.params[k])

    # For higher order primitive check the jaxpr match
    if exclude_param:
      j1 = eqn1.params[exclude_param]
      j2 = eqn2.params[exclude_param]
      if isinstance(j1, jax.core.ClosedJaxpr):
        assert isinstance(j2, jax.core.ClosedJaxpr)
        self.assertEqual(len(j1.consts), len(j2.consts))
        j1 = j1.jaxpr
        j2 = j2.jaxpr
      self.check_jaxpr_equal(j1, j2, True)

    # Check variables
    for v1, v2 in zip(eqn1.invars, eqn2.invars):
      if isinstance(v1, jax.core.Literal):
        self.assertIsInstance(v2, jax.core.Literal)
        self.assertEqual(v1.aval, v2.aval)
      else:
        self.assertEqual(v1.aval.shape, v2.aval.shape)
        self.assertEqual(v1.aval.dtype, v2.aval.dtype)
        vars_to_vars[v1] = v2
    return vars_to_vars

  def check_jaxpr_equal(self, jaxpr_1, jaxpr_2, map_output_vars: bool):
    """Checks that the two jaxpr match."""
    self.assertEqual(len(jaxpr_1.invars), len(jaxpr_2.invars))
    self.assertEqual(len(jaxpr_1.constvars), len(jaxpr_2.constvars))
    self.assertEqual(len(jaxpr_1.outvars), len(jaxpr_2.outvars))
    self.assertEqual(len(jaxpr_1.eqns), len(jaxpr_2.eqns))

    # Extract all loss tags from both jax expressions
    l1_eqns = []
    for eqn in jaxpr_1.eqns:
      if isinstance(eqn.primitive, kfac_jax.layers_and_loss_tags.LossTag):
        l1_eqns.append(eqn)
    l2_eqns = []
    vars_to_eqn = {}
    for eqn in jaxpr_2.eqns:
      if isinstance(eqn.primitive, kfac_jax.layers_and_loss_tags.LossTag):
        l2_eqns.append(eqn)
      for v in eqn.outvars:
        vars_to_eqn[v] = eqn
    self.assertEqual(len(l1_eqns), len(l2_eqns))

    # Match all losses output variables
    if map_output_vars:
      vars_to_vars = dict(zip(jaxpr_1.outvars, jaxpr_2.outvars))
    else:
      vars_to_vars = {}
      for eqn1, eqn2 in zip(l1_eqns, l2_eqns):
        self.assertEqual(len(eqn1.outvars), len(eqn2.outvars))
        for v1, v2 in zip(eqn1.outvars, eqn2.outvars):
          if isinstance(v1, jax.core.DropVar):
            self.assertIsInstance(v2, jax.core.DropVar)
          elif isinstance(v1, jax.core.Literal):
            self.assertIsInstance(v2, jax.core.Literal)
            self.assertEqual(v1.aval, v2.aval)
          else:
            self.assertEqual(v1.aval.shape, v2.aval.shape)
            self.assertEqual(v1.aval.dtype, v2.aval.dtype)
            vars_to_vars[v1] = v2

    # Match all others equations
    for eqn in reversed(jaxpr_1.eqns):
      vars_to_vars = self.check_equation_match(eqn, vars_to_vars, vars_to_eqn)

    for v1, v2 in zip(jaxpr_1.invars, jaxpr_2.invars):
      if v1 in vars_to_vars:
        self.assertEqual(v2, vars_to_vars[v1])
      self.assertEqual(v1.aval.shape, v2.aval.shape)
      self.assertEqual(v1.aval.dtype, v2.aval.dtype)
      self.assertEqual(v1.count, v2.count)

  @parameterized.parameters(models.NON_LINEAR_MODELS + models.SCAN_MODELS)
  def test_auto_register_tags_jaxpr(
      self,
      init_func: Callable[..., models.hk.Params],
      model_func: Callable[..., Array],
      data_point_shapes: Mapping[str, Shape],
      seed: int,
      data_size: int = 4,
  ):
    """Tests the tags auto registration."""
    rng = jax.random.PRNGKey(seed)
    init_key, data_key = jax.random.split(rng)

    data = {}
    for name, shape in data_point_shapes.items():
      data_key, key = jax.random.split(data_key)
      data[name] = jax.random.uniform(key, (data_size,) + shape)
      if name == "labels":
        data[name] = jnp.argmax(data[name], axis=-1)

    params = init_func(init_key, data)
    func = kfac_jax.tag_graph_matcher.auto_register_tags(
        model_func, (params, data))
    jaxpr = jax.make_jaxpr(func)(params, data).jaxpr
    tagged_func = functools.partial(
        model_func,
        explicit_tagging=True,
        return_losses_outputs=True,
    )
    tagged_jaxpr = jax.make_jaxpr(tagged_func)(params, data).jaxpr
    self.check_jaxpr_equal(jaxpr, tagged_jaxpr, False)


if __name__ == "__main__":
  absltest.main()
