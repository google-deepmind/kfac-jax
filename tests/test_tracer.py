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
"""Testing all the tracing mechanisms from tracer.py."""
from typing import Any, Callable, Mapping, Tuple

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import kfac_jax
from tests import models
import numpy as np

tracer = kfac_jax.tracer
utils = kfac_jax.utils
Array = utils.Array
PRNGKey = utils.PRNGKey
Shape = utils.Shape


class TestTracer(parameterized.TestCase):
  """Test class for the functions in `tracer.py`."""

  def assertAllClose(
      self,
      x: utils.PyTree,
      y: utils.PyTree,
      check_dtypes: bool = True,
      atol: float = 5e-6,
      rtol: float = 1e-6,
  ):
    """Asserts that the two PyTrees are close up to the provided tolerances."""
    x_v, x_tree = jax.tree_util.tree_flatten(x)
    y_v, y_tree = jax.tree_util.tree_flatten(y)
    self.assertEqual(x_tree, y_tree)
    for xi, yi in zip(x_v, y_v):
      self.assertEqual(xi.shape, yi.shape)
      if check_dtypes:
        self.assertEqual(xi.dtype, yi.dtype)
      np.testing.assert_allclose(xi, yi, rtol=rtol, atol=atol)

  @staticmethod
  def generate_data(
      init_func: Callable[..., models.hk.Params],
      model_func: Callable[..., Array],
      data_point_shapes: Mapping[str, Shape],
      rng: PRNGKey,
      data_size: int = 4,
  ) -> Tuple[
      models.hk.Params,
      Mapping[str, Array],
      models.hk.Params,
      Tuple[Tuple[Array, ...], ...]
  ]:
    """Generates random data for any testing."""
    data = {}
    for name, shape in data_point_shapes.items():
      rng, key = jax.random.split(rng)
      data[name] = jax.random.uniform(key, (data_size, *shape))
      if name == "labels":
        data[name] = jnp.argmax(data[name], axis=-1)

    rng, key = jax.random.split(rng)
    params = init_func(key, data)
    rng, key = jax.random.split(rng)
    p_tangents = init_func(key, data)

    loss_values, layers_values = model_func(
        params, data, return_layer_values=True)
    last_layer_output = layers_values[-1][1]
    keys = tuple(jax.random.split(key, len(loss_values)))
    output_tangents = tuple(
        (jax.random.normal(key, last_layer_output.shape),) for key in keys)
    return params, data, p_tangents, output_tangents

  def compare_multi_batch(
      self,
      func: Callable[[Any], Any],
      data: Any,
      data_size: int,
      combine: str,
      atol: float = 1e-6,
      rtol: float = 1e-6,
  ):
    """Compares `func` with a single large batch and multiple small one."""
    # Single batch computation
    single_output = func(data)

    # Different batch computation
    data1 = jax.tree_util.tree_map(lambda x: x[:data_size // 2], data)
    data2 = jax.tree_util.tree_map(lambda x: x[data_size // 2:], data)
    outputs = list()
    for d in (data1, data2):
      outputs.append(func(d))
    if combine == "concatenate":
      outputs = jax.tree_util.tree_map(
          lambda x, y: jnp.concatenate([x, y], axis=0), *outputs)
    elif combine == "sum":
      outputs = jax.tree_util.tree_map(lambda x, y: x + y, *outputs)
    else:
      raise NotImplementedError()

    self.assertAllClose(single_output, outputs, atol=atol, rtol=rtol)

  @parameterized.parameters(models.NON_LINEAR_MODELS)
  def test_loss_tags_jvp(
      self,
      init_func: Callable[..., models.hk.Params],
      model_func: Callable[..., Array],
      data_point_shape: Mapping[str, Shape],
      seed: int,
      dataset_size: int = 4,
  ):
    """Tests `tracer.py:loss_tags.jvp`."""
    # Model and data setup
    rng = jax.random.PRNGKey(seed)
    params, data, p_tangents, _ = self.generate_data(
        init_func, model_func, data_point_shape, rng, dataset_size,
    )

    # True computation
    (primals_out, tangents_out) = jax.jvp(
        lambda p: model_func(p, data, return_layer_values=True),
        [params], [p_tangents])
    loss_values, _ = primals_out
    _, layers_tangents = tangents_out
    last_layer_output_tangents = layers_tangents[-1][1]
    loss_tangents = ((last_layer_output_tangents,),) * len(loss_values)

    # Tracer computation
    tracer_jvp = tracer.loss_tags_jvp(model_func)
    tracer_losses, tracer_loss_tangents = tracer_jvp((params, data), p_tangents)  # pytype: disable=attribute-error  # always-use-return-annotations
    tracer_losses_values = [loss.evaluate() for loss in tracer_losses]

    self.assertAllClose(loss_values, tracer_losses_values)
    self.assertAllClose(loss_tangents, tracer_loss_tangents)

  @parameterized.parameters(models.NON_LINEAR_MODELS)
  def test_loss_tags_jvp_diff_batch_size(
      self,
      init_func: Callable[..., models.hk.Params],
      model_func: Callable[..., Array],
      data_point_shape: Mapping[str, Shape],
      seed: int,
      data_size: int = 4,
  ):
    """Tests `tracer.py:loss_tags.jvp` for multiple batches."""
    # Model and data setup
    rng = jax.random.PRNGKey(seed)
    params, data, p_tangents, _ = self.generate_data(
        init_func, model_func, data_point_shape, rng, data_size,
    )

    jvp = tracer.loss_tags_jvp(model_func)
    def func(data_):
      losses, loss_tangents = jvp((params, data_), p_tangents)  # pytype: disable=attribute-error  # always-use-return-annotations
      losses = [loss.evaluate() for loss in losses]
      return losses, loss_tangents

    self.compare_multi_batch(func, data, data_size, "concatenate")

  @parameterized.parameters(models.NON_LINEAR_MODELS)
  def test_loss_tags_vjp(
      self,
      init_func: Callable[..., models.hk.Params],
      model_func: Callable[..., Array],
      data_point_shape: Mapping[str, Shape],
      seed: int,
      dataset_size: int = 4,
  ):
    """Tests `tracer.py:loss_tags_vjp`."""
    # Model and data setup
    rng = jax.random.PRNGKey(seed)
    params, data, _, output_tangents = self.generate_data(
        init_func, model_func, data_point_shape, rng, dataset_size,
    )
    def no_data_func(p):
      losses, layers_values = model_func(p, data, return_layer_values=True)
      last_layer_output = layers_values[-1][1]
      return losses, last_layer_output

    # True computation
    (loss_values, _), vjp_func = jax.vjp(no_data_func, params)
    loss_tangents = jax.tree_util.tree_map(jnp.zeros_like, loss_values)
    summed_output_tangents = sum(jax.tree_util.tree_leaves(output_tangents))
    p_tangents, = vjp_func((loss_tangents, summed_output_tangents))

    # Tracer computation
    trace_vjp = tracer.loss_tags_vjp(model_func)
    tracer_losses, tracer_vjp_func = trace_vjp((params, data))  # pytype: disable=attribute-error  # always-use-return-annotations
    tracer_losses = [loss.evaluate() for loss in tracer_losses]
    tracer_p_tangents = tracer_vjp_func(output_tangents)

    # Comparison
    self.assertAllClose(loss_values, tracer_losses)
    self.assertAllClose(p_tangents, tracer_p_tangents)

  @parameterized.parameters(models.NON_LINEAR_MODELS)
  def test_loss_tags_vjp_diff_batch_size(
      self,
      init_func: Callable[..., models.hk.Params],
      model_func: Callable[..., Array],
      data_point_shape: Mapping[str, Shape],
      seed: int,
      data_size: int = 4,
  ):
    """Tests `tracer.py:loss_tags_vjp` for multiple batches."""
    # Model and data setup
    rng = jax.random.PRNGKey(seed)
    params, data, _, output_tangents = self.generate_data(
        init_func, model_func, data_point_shape, rng, data_size,
    )

    # Tracer computation
    vjp = tracer.loss_tags_vjp(model_func)

    def func1(data_):
      losses, _ = vjp((params, data_))  # pytype: disable=attribute-error  # always-use-return-annotations
      return [loss.evaluate() for loss in losses]

    self.compare_multi_batch(func1, data, data_size, "concatenate")

    def func2(data_and_output_tangents):
      data_, output_tangents_ = data_and_output_tangents
      _, vjp_func = vjp((params, data_))  # pytype: disable=attribute-error  # always-use-return-annotations
      return vjp_func(output_tangents_)

    self.compare_multi_batch(func2, (data, output_tangents), data_size, "sum")

  @parameterized.parameters(models.NON_LINEAR_MODELS)
  def test_loss_tags_hvp(
      self,
      init_func: Callable[..., models.hk.Params],
      model_func: Callable[..., Array],
      data_point_shape: Mapping[str, Shape],
      seed: int,
      dataset_size: int = 4,
  ):
    """Tests `tracer.py:loss_tags_hvp`."""
    # Model and data setup
    rng = jax.random.PRNGKey(seed)
    params, data, p_tangents, _ = self.generate_data(
        init_func, model_func, data_point_shape, rng, dataset_size,
    )
    def no_data_func(p):
      return sum(jax.tree_util.tree_map(jnp.sum, model_func(p, data)))

    # True computation
    grad_func = jax.grad(no_data_func)
    def grad_time_tangents(args):
      return utils.inner_product(grad_func(args), p_tangents)
    hvp = jax.grad(grad_time_tangents)
    hvp_vectors = hvp(params)

    # Tracer computation
    tracer_hvp = tracer.loss_tags_hvp(model_func)
    tracer_hvp_vectors, _ = tracer_hvp((params, data), p_tangents)  # pytype: disable=attribute-error  # always-use-return-annotations

    # Comparison
    self.assertAllClose(hvp_vectors, tracer_hvp_vectors, atol=5e-6)

  @parameterized.parameters(models.NON_LINEAR_MODELS)
  def test_loss_tags_hvp_diff_batch_size(
      self,
      init_func: Callable[..., models.hk.Params],
      model_func: Callable[..., Array],
      data_point_shape: Mapping[str, Shape],
      seed: int,
      data_size: int = 4,
  ):
    """Tests `tracer.py:loss_tags_hvp` for multiple batches."""
    # Model and data setup
    rng = jax.random.PRNGKey(seed)
    params, data, p_tangents, _ = self.generate_data(
        init_func, model_func, data_point_shape, rng, data_size
    )

    hvp = tracer.loss_tags_hvp(model_func)

    def func(data_):
      return hvp((params, data_), p_tangents)[0]

    self.compare_multi_batch(func, data, data_size, "sum", rtol=1e-4)

  @parameterized.parameters(models.NON_LINEAR_MODELS)
  def test_layer_tags_vjp(
      self,
      init_func: Callable[..., models.hk.Params],
      model_func: Callable[..., Array],
      data_point_shape: Mapping[str, Shape],
      seed: int,
      dataset_size: int = 4,
  ):
    """Tests `tracer.py:layer_tags_vjp`."""
    # Model and data setup
    rng = jax.random.PRNGKey(seed)
    params, data, _, output_tangents = self.generate_data(
        init_func, model_func, data_point_shape, rng, dataset_size,
    )
    def aux_no_data_func(aux, p):
      _, layers_values = model_func(
          p, data, aux=aux, return_layer_values=True)
      last_layer_output = layers_values[-1][1]
      return last_layer_output

    # True computation
    loss_values, layer_values = model_func(
        params, data, return_layer_values=True)
    layer_outputs = tuple(v[1] for v in layer_values)
    aux_values = jax.tree_util.tree_map(jnp.zeros_like, layer_outputs)
    _, vjp = jax.vjp(aux_no_data_func, aux_values, params)
    summed_output_tangents = sum(jax.tree_util.tree_leaves(output_tangents))
    aux_tangents, p_tangents = vjp(summed_output_tangents)
    self.assertEqual(len(layer_values), len(params))
    self.assertEqual(len(aux_tangents), len(p_tangents))

    layers_info = list()
    for (x, y), aux_t, param, param_tangent in zip(
        layer_values, aux_tangents,
        list(params.values()), list(p_tangents.values())
    ):
      info = dict()
      info["inputs"] = (x,)
      info["outputs"] = (y,)
      info["outputs_tangent"] = (aux_t,)
      general_names = ("w", "b") if "w" in param else ("scale", "offset")
      p_names = tuple(name for name in general_names if name in param)
      self.assertLessEqual(len(p_names), len(param))
      info["params"] = tuple(param[name] for name in p_names)
      info["params_tangent"] = tuple(param_tangent[name] for name in p_names)
      layers_info.append(info)

    layers_info = tuple(layers_info)

    # Tracer computation
    tracer_losses, tracer_vjp_func = tracer.layer_tags_vjp(model_func)(  # pytype: disable=attribute-error  # always-use-return-annotations
        (params, data))
    tracer_losses = [loss.evaluate() for loss in tracer_losses]
    tracer_info = tracer_vjp_func(output_tangents)

    # We don't support testing of inputs_tangent currently
    for info in tracer_info:
      info.pop("inputs_tangent")

    # Comparison
    self.assertAllClose(loss_values, tracer_losses)
    self.assertAllClose(layers_info, tracer_info)

  @parameterized.parameters(models.NON_LINEAR_MODELS)
  def test_layer_tags_vjp_diff_batch_size(
      self,
      init_func: Callable[..., models.hk.Params],
      model_func: Callable[..., Array],
      data_point_shape: Mapping[str, Shape],
      seed: int,
      data_size: int = 4,
  ):
    """Tests `tracer.py:layer_tags_vjp` for multiple batches."""
    # Model and data setup
    rng = jax.random.PRNGKey(seed)
    params, data, _, output_tangents = self.generate_data(
        init_func, model_func, data_point_shape, rng, data_size,
    )

    vjp = tracer.layer_tags_vjp(model_func)

    def func(data_and_output_tangents):
      data_, output_tangents_ = data_and_output_tangents
      losses, vjp_func = vjp((params, data_))  # pytype: disable=attribute-error  # always-use-return-annotations
      losses = [loss.evaluate() for loss in losses]
      layers_info = vjp_func(output_tangents_)
      for info in layers_info:
        # These quantities are not per-batch, but averaged, so we skip them
        info.pop("params")
        info.pop("params_tangent")
      return losses, layers_info

    self.compare_multi_batch(
        func, (data, output_tangents), data_size, "concatenate")


if __name__ == "__main__":
  absltest.main()
