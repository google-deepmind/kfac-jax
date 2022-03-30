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
import chex
import jax
import jax.numpy as jnp
import kfac_jax
from tests import models

tag_graph_matcher = kfac_jax.tag_graph_matcher


class TestGraphMatcher(parameterized.TestCase):
  """Test class for the functions in `tag_graph_matcher.py`."""

  @parameterized.parameters(models.NON_LINEAR_MODELS)
  def test_auto_register_tags_jaxpr(
      self,
      init_func: Callable[..., models.hk.Params],
      model_func: Callable[..., chex.Array],
      data_point_shapes: Mapping[str, chex.Shape],
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
    func = tag_graph_matcher.auto_register_tags(model_func, (params, data))
    jaxpr = jax.make_jaxpr(func)(params, data).jaxpr
    tagged_func = functools.partial(
        model_func,
        explicit_tagging=True,
        return_registered_losses_inputs=True,
    )
    tagged_jaxpr = jax.make_jaxpr(tagged_func)(params, data).jaxpr
    self.assertEqual(len(jaxpr.invars), len(tagged_jaxpr.invars))
    self.assertEqual(len(jaxpr.constvars), len(tagged_jaxpr.constvars))
    self.assertEqual(len(jaxpr.outvars), len(tagged_jaxpr.outvars))
    # Note that since the auto registered computation finishes at the loss tags
    # it will always have the same or less number of equations.
    self.assertLessEqual(len(jaxpr.eqns), len(tagged_jaxpr.eqns))

    for eq, tagged_eq in zip(jaxpr.eqns, tagged_jaxpr.eqns):
      eq_in_vars = [v for v in eq.invars
                    if not isinstance(v, jax.core.UnitVar)]
      tagged_in_vars = [v for v in tagged_eq.invars
                        if not isinstance(v, jax.core.UnitVar)]
      self.assertEqual(len(eq_in_vars), len(tagged_in_vars))
      self.assertEqual(len(eq.outvars), len(tagged_eq.outvars))
      self.assertEqual(eq.primitive, tagged_eq.primitive)
      for variable, t_variable in zip(eq_in_vars + eq.outvars,
                                      tagged_in_vars + tagged_eq.outvars):
        if isinstance(variable, jax.core.Literal):
          self.assertEqual(variable.aval, t_variable.aval)
        else:
          self.assertEqual(variable.count, t_variable.count)


if __name__ == "__main__":
  absltest.main()
