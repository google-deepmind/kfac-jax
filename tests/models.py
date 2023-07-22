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
"""Module with models used for testing."""
import functools
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import kfac_jax

tags = kfac_jax.layers_and_loss_tags
loss_functions = kfac_jax.loss_functions
utils = kfac_jax.utils

Array = kfac_jax.utils.Array
PRNGKey = kfac_jax.utils.PRNGKey
LayerValues = List[Tuple[Array, Array]]
LayerInputs = Tuple[Array, LayerValues, Optional[Tuple[Array, ...]]]
LossOutputs = Union[
    List[List[Array]],
    List[Array],
    Tuple[List[Array], LayerValues]
]


def _extract_params(
    instance: hk.Module,
    names: Sequence[str],
) -> Tuple[Array, Optional[Array]]:
  """Extracts the weights and bias parameters or `None` if don't exists."""
  params = [None] * len(names)
  for name, v in instance.params_dict().items():
    found = False
    for i, k in enumerate(names):
      # In the tests, parameter names are guaranteed to have the form
      # 'layer_name/parameter_name'.
      if "/" + k in name:
        params[i] = v
        found = True
        break
    if not found:
      raise ValueError(f"Did not match parameter {name}.")
  assert len(params) == 2 and params[0] is not None
  return tuple(params)


class _Linear(hk.Linear):
  """A linear layer which also can register and return intermediate values."""

  def __init__(
      self,
      *args: Any,
      explicit_tagging: bool = False,
      **kwargs: Any,
  ):
    """Initializes the instance.

    Args:
      *args: Arguments to pass to the `hk.Linear` constructor.
      explicit_tagging: Whether to explicitly tag the computation of the layer
        with a `dense_tag`.
      **kwargs: Keywords arguments to pass to the `hk.Conv2D` constructor.
    """
    self._explicit_tagging = explicit_tagging
    super().__init__(*args, **kwargs)

  def __call__(self, inputs: LayerInputs, *_) -> LayerInputs:  # pytype: disable=signature-mismatch  # overriding-parameter-name-checks
    jax_version = tuple(map(int, jax.__version__.split(".")[:3]))
    x, layer_values, aux = inputs
    y = super().__call__(x, precision=jax.lax.Precision.HIGHEST)
    if aux is not None:
      y, aux = y + aux[0], aux[1:]

    if self._explicit_tagging:
      params = _extract_params(self, ("w", "b"))

      if jax_version < (0, 4, 14):
        preferred_element_type = None
      else:
        assert all(p.dtype == y.dtype for p in params if p is not None)
        preferred_element_type = y.dtype

      y = tags.register_dense(
          y, x, *params,
          dimension_numbers=(((1,), (0,)), ((), ())),
          precision=(jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
          preferred_element_type=preferred_element_type,
      )
    layer_values.append((x, y))

    return y, layer_values, aux


class _Conv2D(hk.Conv2D):
  """A conv2d layer which also can register and return intermediate values."""

  def __init__(
      self,
      *args: Any,
      explicit_tagging: bool = False,
      **kwargs: Any,
  ):
    """Initializes the instance.

    Args:
      *args: Arguments to pass to the `hk.Conv2D` constructor.
      explicit_tagging: Whether to explicitly tag the computation of the layer
        with a `dense_tag`.
      **kwargs: Keywords arguments to pass to the `hk.Conv2D` constructor.
    """
    self._explicit_tagging = explicit_tagging
    super().__init__(*args, **kwargs)

  def __call__(self, inputs: LayerInputs, *_) -> LayerInputs:
    x, layer_values, aux = inputs
    y = super().__call__(x, precision=jax.lax.Precision.HIGHEST)
    if aux is not None:
      y, aux = y + aux[0], aux[1:]
    layer_values.append((x, y))
    if self._explicit_tagging:
      params = _extract_params(self, ("w", "b"))
      y = tags.register_conv2d(
          y, x, *params,
          batch_group_count=1,
          dimension_numbers=jax.lax.ConvDimensionNumbers(
              lhs_spec=(0, 3, 1, 2),
              rhs_spec=(3, 2, 0, 1),
              out_spec=(0, 3, 1, 2),
          ),
          feature_group_count=1,
          lhs_dilation=(1, 1),
          padding=((0, 1), (0, 1)),
          precision=(jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
          preferred_element_type=None,
          rhs_dilation=(1, 1),
          window_strides=self.stride,
      )
    return y, layer_values, aux


class _LayerNorm(hk.LayerNorm):
  """A layer norm layer which can register and return intermediate values."""

  def __init__(
      self,
      *args: Any,
      explicit_tagging: bool = False,
      **kwargs: Any,
  ):
    """Initializes the instance.

    Args:
      *args: Arguments to pass to the `hk.LayerNorm` constructor.
      explicit_tagging: Whether to explicitly tag the computation of the layer
        with a `dense_tag`.
      **kwargs: Keywords arguments to pass to the `hk.Conv2D` constructor.
    """
    self._explicit_tagging = explicit_tagging
    super().__init__(*args, create_scale=True, create_offset=True, **kwargs)

  def __call__(self, inputs: LayerInputs, *_) -> LayerInputs:  # pytype: disable=signature-mismatch  # jax-ndarray
    x, layer_values, aux = inputs

    mean = jnp.mean(x, axis=self.axis, keepdims=True)
    variance = jnp.var(x, axis=self.axis, keepdims=True)
    param_shape = x.shape[-1:]
    scale = hk.get_parameter("scale", param_shape, x.dtype,
                             init=self.scale_init)
    offset = hk.get_parameter("offset", param_shape, x.dtype,
                              init=self.offset_init)
    scale = jnp.broadcast_to(scale, x.shape)
    offset = jnp.broadcast_to(offset, x.shape)
    mean = jnp.broadcast_to(mean, x.shape)

    rsqrt = jax.lax.rsqrt(variance + self.eps)
    # This is specially implemented to preserve correct ordering in the jaxpr
    multiplier = scale * rsqrt
    diff = x - mean
    y = multiplier * diff + offset
    normalized_inputs = diff * rsqrt

    if aux is not None:
      y, aux = y + aux[0], aux[1:]
    layer_values.append((normalized_inputs, y))
    if self._explicit_tagging:
      params = _extract_params(self, ("scale", "offset"))
      y = tags.register_scale_and_shift(y, normalized_inputs, *params)
    return y, layer_values, aux


class _VanillaRNN(hk.VanillaRNN):
  """Modified Vanilla RNN to handle layer values and auxiliary inputs."""

  def __init__(
      self,
      hidden_size: int,
      activation: Callable[[LayerInputs], LayerInputs],
      explicit_tagging: bool = False,
      double_bias: bool = True,
      name: Optional[str] = None
  ):
    super().__init__(hidden_size, double_bias, name=name)
    self.activation = activation
    self.explicit_tagging = explicit_tagging

  def __call__(
      self,
      inputs: LayerInputs,
      prev_state: Array,
      *_,
  ) -> Tuple[Tuple[Array, LayerValues], Array]:
    x, layer_values, aux = inputs
    input_to_hidden = _Linear(
        self.hidden_size, explicit_tagging=self.explicit_tagging)
    hidden_to_hidden = _Linear(
        self.hidden_size, explicit_tagging=self.explicit_tagging,
        with_bias=self.double_bias)
    ih, layer_values, aux = input_to_hidden((x, layer_values, aux))
    hh, layer_values, aux = hidden_to_hidden((x, layer_values, aux))
    out, layer_values, aux = self.activation((ih + hh, layer_values, aux))
    assert aux is None or not aux
    return (out, layer_values), out


def _modify_func(
    func: Callable[[Array], Array]
) -> Callable[[LayerInputs], LayerInputs]:
  """Functorially maps f: x -> y to f2: (x, p, q) -> (f(x), p, q)."""

  def func2(inputs: LayerInputs) -> LayerInputs:
    """Applies `func` only to the first argument of `inputs`."""
    if not (isinstance(inputs, tuple) and len(inputs) == 3):
      raise ValueError("Transformed activations take a tuple of length 3 as an "
                       "argument.")
    return func(inputs[0]), inputs[1], inputs[2]

  return func2

_special_tanh = _modify_func(jax.nn.tanh)
_special_relu = _modify_func(jax.nn.relu)
_special_flatten = _modify_func(lambda x: x.reshape([x.shape[0], -1]))
_special_identity = _modify_func(lambda x: x)


class _DeterministicBernoulli(distrax.Bernoulli):
  """A fake deterministic bernoulli distribution, to make KFAC deterministic."""

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    del key  # not used
    return jnp.repeat(jnp.round(self.probs)[None], n, axis=0)


class _DeterministicBernoulliNegativeLogProbLoss(
    loss_functions.MultiBernoulliNegativeLogProbLoss):
  """A negative log-likelihood loss using the `DeterministicBernoulli`."""

  @property
  def dist(self):
    return _DeterministicBernoulli(logits=self._logits, dtype=jnp.int32)

_DeterministicBernoulliNegativeLogProbLoss_tag = loss_functions.tags.LossTag(
    _DeterministicBernoulliNegativeLogProbLoss,
    parameter_dependants=["logits"],
    parameter_independants=["targets", "weight"],
)


def _register_deterministic_bernoulli(
    logits: Array,
    targets: Array,
    weight=1.0
):
  """Registers a deterministic bernoulli loss."""
  if targets is None:
    args = [logits, weight]
    args_names = ["logits", "weight"]
  else:
    args = [logits, targets, weight]
    args_names = ["logits", "targets", "weight"]
  _DeterministicBernoulliNegativeLogProbLoss_tag.bind(*args,
                                                      args_names=args_names)


class _DeterministicCategorical(distrax.Categorical):
  """A fake deterministic bernoulli distribution, to make KFAC deterministic."""

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    del key  # not used
    return jnp.repeat(jnp.round(self.probs)[None], n, axis=0)


class _DeterministicCategoricalNegativeLogProbLoss(
    loss_functions.CategoricalLogitsNegativeLogProbLoss):
  """A negative log-likelihood loss using the `DeterministicBernoulli`."""

  @property
  def dist(self) -> _DeterministicCategorical:
    return _DeterministicCategorical(logits=self._logits, dtype=jnp.int32)

_DeterministicCategoricalNegativeLogProbLoss_tag = loss_functions.tags.LossTag(
    _DeterministicCategoricalNegativeLogProbLoss,
    parameter_dependants=["logits"],
    parameter_independants=["targets", "weight"],
)


def _register_deterministic_categorical(
    logits: Array,
    targets: Array,
    weight=1.0
) -> Array:
  """Registers a deterministic categorical loss."""
  if targets is None:
    args = [logits, weight]
    args_names = ["logits", "weight"]
  else:
    args = [logits, targets, weight]
    args_names = ["logits", "targets", "weight"]
  return _DeterministicCategoricalNegativeLogProbLoss_tag.bind(
      *args, args_names=args_names)[0]


def squared_error_loss(
    params: utils.Params,
    batch: utils.Batch,
    model_func: Callable[..., hk.Transformed],
    l2_reg: float = 0.0,
    explicit_tagging: bool = False,
    return_losses_outputs: bool = False,
    return_layer_values: bool = False,
) -> LossOutputs:
  """A squared error loss computed for the given model function."""
  x, y = batch["images"], batch["targets"]

  y_hat, layer_values = model_func(
      explicit_tagging=explicit_tagging, output_dim=y.shape[-1],  # pytype: disable=attribute-error  # numpy-scalars
  ).apply(params, x)

  assert y_hat.shape == y.shape  # pytype: disable=attribute-error  # numpy-scalars
  y = y.reshape((-1, y.shape[-1]))  # pytype: disable=attribute-error  # numpy-scalars
  y_hat = y_hat.reshape((-1, y_hat.shape[-1]))

  loss_functions.register_squared_error_loss(y_hat, y, weight=0.5)

  if return_losses_outputs:
    return [[y_hat]]

  loss = jnp.mean(jnp.sum((y_hat - y) ** 2, axis=-1)) / 2
  loss = loss + l2_reg * utils.norm(params)

  if return_layer_values:
    return [loss], layer_values
  else:
    return [loss]


def autoencoder(
    layer_widths: Sequence[int],
    output_dim: int,
    explicit_tagging: bool = False,
    activation: Callable[[LayerInputs], LayerInputs] = _special_tanh,
) -> hk.Transformed:
  """Constructs a Haiku transformed object of the autoencoder network."""
  def func(
      batch: Union[Array, Mapping[str, Array]],
      aux: Optional[Tuple[Array, ...]] = None,
  ) -> Tuple[Array, LayerValues]:
    images = batch["images"] if isinstance(batch, Mapping) else batch
    images = images.reshape([images.shape[0], -1])
    layers = []
    for width in layer_widths:
      layers.append(_Linear(output_size=width,
                            explicit_tagging=explicit_tagging))
      layers.append(activation)
    layers.append(_Linear(output_size=output_dim,
                          explicit_tagging=explicit_tagging))
    model = hk.Sequential(layers)
    output, layer_values, aux = model((images, list(), aux))
    assert aux is None or not aux
    return output, layer_values
  return hk.without_apply_rng(hk.transform(func))


def linear_squared_error_autoencoder_loss(
    params: utils.Params,
    batch: utils.Batch,
    layer_widths: Sequence[int],
    l2_reg: float = 0.0,
    explicit_tagging: bool = False,
    return_losses_outputs: bool = False,
    return_layer_values: bool = False,
) -> LossOutputs:
  """A linear autoencoder with squared error."""
  batch["images"] = batch["images"].reshape(batch["images"].shape[0], -1)  # type: ignore  # numpy-scalars
  batch["targets"] = batch["images"]  # pytype: disable=unsupported-operands  # numpy-scalars
  model_func = functools.partial(
      autoencoder, layer_widths=layer_widths, activation=_special_identity)
  return squared_error_loss(
      params=params,
      batch=batch,
      model_func=model_func,
      l2_reg=l2_reg,
      explicit_tagging=explicit_tagging,
      return_losses_outputs=return_losses_outputs,
      return_layer_values=return_layer_values,
  )


def autoencoder_deterministic_loss(
    params: utils.Params,
    batch: utils.Batch,
    layer_widths: Sequence[int],
    l2_reg: float = 0.0,
    explicit_tagging: bool = False,
    activation: Callable[[LayerInputs], LayerInputs] = _special_tanh,
) -> Array:
  """Evaluate the autoencoder with a deterministic loss."""
  x = batch["images"].reshape((batch["images"].shape[0], -1))  # pytype: disable=attribute-error  # numpy-scalars
  logits, _ = autoencoder(
      layer_widths, x.shape[-1], explicit_tagging, activation=activation,
  ).apply(params, x)
  _register_deterministic_bernoulli(logits, x)
  loss = - distrax.Bernoulli(logits=logits).log_prob(x)
  loss = jnp.mean(jnp.sum(loss, axis=-1)).astype(logits.dtype)
  return loss + l2_reg * utils.norm(params)


def autoencoder_with_two_losses(
    params: utils.Params,
    batch: utils.Batch,
    layer_widths: Sequence[int],
    aux: Optional[Tuple[Array, ...]] = None,
    explicit_tagging: bool = False,
    return_losses_outputs: bool = False,
    return_layer_values: bool = False,
    activation: Callable[[LayerInputs], LayerInputs] = _special_tanh,
) -> LossOutputs:
  """Evaluate the autoencoder with two losses."""
  x = batch["images"].reshape((batch["images"].shape[0], -1))  # pytype: disable=attribute-error  # numpy-scalars

  logits, layer_values = autoencoder(
      layer_widths, x.shape[-1], explicit_tagging, activation=activation,
  ).apply(params, x, aux)

  # Register both losses in KFAC
  loss_functions.register_multi_bernoulli_predictive_distribution(
      logits, x)
  loss_functions.register_normal_predictive_distribution(
      logits, x, weight=0.1)

  if return_losses_outputs:
    return [[logits], [logits]]

  loss_1: Array = - distrax.Bernoulli(logits=logits).log_prob(x)  # pytype: disable=annotation-type-mismatch
  scale_diag = jnp.ones_like(logits) * jnp.sqrt(0.5)
  loss_2: Array = - distrax.MultivariateNormalDiag(  # pytype: disable=annotation-type-mismatch
      loc=logits, scale_diag=scale_diag).log_prob(x)

  if return_layer_values:
    return [loss_1, 0.1 * loss_2], layer_values
  else:
    return [loss_1, 0.1 * loss_2]


def conv_classifier(
    num_classes: int,
    layer_channels: Sequence[int],
    explicit_tagging: bool = False,
    kernel_size: int = 3,
    stride: int = 2,
    activation: Callable[[LayerInputs], LayerInputs] = _special_tanh,
) -> hk.Transformed:
  """Constructs a Haiku transformed object of a convolutional classifier."""
  def func(
      batch: Union[Array, Mapping[str, Array]],
      aux: Optional[Tuple[Array, ...]] = None,
  ) -> Tuple[Array, LayerValues]:
    images = batch["images"] if isinstance(batch, Mapping) else batch
    layers = []
    # Conv channels
    for num_channels in layer_channels[:-1]:
      layers.append(_Conv2D(
          output_channels=num_channels,
          kernel_shape=kernel_size,
          stride=stride,
          explicit_tagging=explicit_tagging))
      layers.append(activation)

    # Last conv has layer norm
    layers.append(_Conv2D(
        output_channels=layer_channels[-1],
        kernel_shape=kernel_size,
        stride=stride,
        with_bias=False,
        explicit_tagging=explicit_tagging))
    layers.append(_LayerNorm(
        axis=-1,
        explicit_tagging=explicit_tagging))
    layers.append(activation)

    # Flatten
    layers.append(_special_flatten)

    # One Linear layer with activation and final layer
    layers.append(_Linear(output_size=layer_channels[-1],
                          explicit_tagging=explicit_tagging))
    layers.append(activation)
    layers.append(_Linear(output_size=num_classes,
                          explicit_tagging=explicit_tagging))
    model = hk.Sequential(layers)

    output, layer_values, aux = model((images, list(), aux))
    assert aux is None or not aux
    return output, layer_values
  return hk.without_apply_rng(hk.transform(func))


def conv_classifier_deterministic_loss(
    params: utils.Params,
    batch: utils.Batch,
    num_classes: int,
    layer_channels: Sequence[int],
    l2_reg: float = 0.0,
    explicit_tagging: bool = False,
    activation: Callable[[LayerInputs], LayerInputs] = _special_tanh,
) -> Array:
  """Evaluate the convolutional classifier with a deterministic loss."""
  logits, _ = conv_classifier(
      num_classes, layer_channels, explicit_tagging, activation=activation
  ).apply(params, batch["images"])
  _register_deterministic_categorical(logits, batch["labels"])
  loss = - distrax.Categorical(logits=logits).log_prob(batch["labels"])
  loss = jnp.mean(jnp.sum(loss, axis=-1)).astype(logits.dtype)
  return loss + l2_reg * utils.norm(params)


def conv_classifier_loss(
    params: utils.Params,
    batch: utils.Batch,
    num_classes: int,
    layer_channels: Sequence[int],
    aux: Optional[Tuple[Array, ...]] = None,
    l2_reg: float = 0.0,
    explicit_tagging: bool = False,
    return_losses_outputs: bool = False,
    return_layer_values: bool = False,
    activation: Callable[[LayerInputs], LayerInputs] = _special_tanh,
) -> LossOutputs:
  """Evaluates the convolutional classifier loss."""
  logits, layer_values = conv_classifier(
      num_classes, layer_channels, explicit_tagging, activation=activation
  ).apply(params, batch["images"], aux=aux)
  loss_functions.register_categorical_predictive_distribution(
      logits, batch["labels"])

  if return_losses_outputs:
    return [[logits]]

  loss = - distrax.Categorical(logits=logits).log_prob(batch["labels"])
  loss = loss + l2_reg * utils.norm(params)
  if return_layer_values:
    return [loss], layer_values
  else:
    return [loss]


def layer_stack_with_scan_mlp(
    layer_widths: Sequence[int],
    output_dim: int,
    explicit_tagging: bool = False,
    activation: Callable[[LayerInputs], LayerInputs] = _special_tanh,
) -> hk.Transformed:
  """A model that uses ``hk.experimental.layer_stack`` with scan."""
  def scan_fn(
      x: Array,
      aux: Optional[Tuple[Array, ...]] = None,
  ) -> Tuple[Array, LayerValues]:
    layers = []
    for w in layer_widths:
      layers.append(_Linear(w, explicit_tagging=explicit_tagging))
      layers.append(activation)
    layers.append(_Linear(x.shape[-1], explicit_tagging=explicit_tagging))
    model = hk.Sequential(layers)

    output, layer_values, aux = model((x, list(), aux))

    assert aux is None or not aux
    return output, layer_values

  def func(
      batch: Union[Array, Mapping[str, Array]],
      aux: Optional[Tuple[Array, ...]] = None,
  ) -> Tuple[Array, LayerValues]:
    x = batch["images"] if isinstance(batch, Mapping) else batch

    stack = hk.experimental.layer_stack(2, with_per_layer_inputs=True)(scan_fn)

    if aux is None:
      aux = None
      x, layer_values = stack(x)
    else:
      aux_scan, aux = aux
      x, layer_values = stack(scan_fn)(x, aux_scan)

    y_hat, layer_values, aux = _Linear(
        output_dim, explicit_tagging=explicit_tagging)((x, layer_values, aux))
    assert aux is None or not aux
    return y_hat, layer_values

  return hk.without_apply_rng(hk.transform(func))


def layer_stack_mlp_loss(
    params: utils.Params,
    batch: utils.Batch,
    layer_widths: Sequence[int],
    l2_reg: float = 0.0,
    explicit_tagging: bool = False,
    return_losses_outputs: bool = False,
    return_layer_values: bool = False,
    activation: Callable[[LayerInputs], LayerInputs] = _special_tanh,
) -> LossOutputs:
  """A layer stack mlp loss."""
  return squared_error_loss(
      params=params,
      batch=batch,
      model_func=functools.partial(
          layer_stack_with_scan_mlp,
          layer_widths=layer_widths,
          activation=activation,
      ),
      l2_reg=l2_reg,
      explicit_tagging=explicit_tagging,
      return_losses_outputs=return_losses_outputs,
      return_layer_values=return_layer_values,
  )


def vanilla_rnn_with_scan(
    hidden_size: int,
    output_dim: int,
    explicit_tagging: bool = False,
    activation: Callable[[LayerInputs], LayerInputs] = _special_tanh,
) -> hk.Transformed:
  """A model that uses an RNN with scan."""
  def func(
      batch: Union[Array, Mapping[str, Array]],
      aux: Optional[Tuple[Array, ...]] = None,
  ) -> Tuple[Array, LayerValues]:
    x = batch["images"] if isinstance(batch, Mapping) else batch

    core = _VanillaRNN(
        hidden_size, activation=activation, explicit_tagging=explicit_tagging)
    init_state = core.initial_state(x.shape[1])

    if aux is None:
      aux = None
      unroll_in = (x, list(), None)
      (x, layer_values), _ = hk.dynamic_unroll(core, unroll_in, init_state)
    else:
      aux_rnn, aux = aux
      unroll_in = (x, list(), aux_rnn)
      (x, layer_values), _ = hk.dynamic_unroll(core, unroll_in, init_state)

    layer_values = list()
    # We need this in order the dense tag to recognize things correctly
    x_r = x.reshape((-1, x.shape[-1]))
    y_hat, layer_values, aux = _Linear(
        output_dim, explicit_tagging=explicit_tagging)((x_r, layer_values, aux))
    y_hat = y_hat.reshape(x.shape[:2] + (output_dim,))
    assert aux is None or not aux
    return y_hat, layer_values

  return hk.without_apply_rng(hk.transform(func))


def vanilla_rnn_with_scan_loss(
    params: utils.Params,
    batch: utils.Batch,
    hidden_size: int,
    l2_reg: float = 0.0,
    explicit_tagging: bool = False,
    return_losses_outputs: bool = False,
    return_layer_values: bool = False,
    activation: Callable[[LayerInputs], LayerInputs] = _special_tanh,
) -> LossOutputs:
  """A layer stack mlp loss."""
  return squared_error_loss(
      params=params,
      batch=batch,
      model_func=functools.partial(
          vanilla_rnn_with_scan,
          hidden_size=hidden_size,
          activation=activation,
      ),
      l2_reg=l2_reg,
      explicit_tagging=explicit_tagging,
      return_losses_outputs=return_losses_outputs,
      return_layer_values=return_layer_values,
  )

NON_LINEAR_MODELS = [
    (
        autoencoder([20, 10, 20], output_dim=8).init,
        functools.partial(
            autoencoder_with_two_losses,
            layer_widths=[20, 10, 20]),
        dict(images=(8,)),
        1231987,
    ),
    (
        conv_classifier(
            num_classes=10,
            layer_channels=[8, 16]
        ).init,
        functools.partial(
            conv_classifier_loss,
            num_classes=10,
            layer_channels=[8, 16]),
        dict(images=(8, 8, 3), labels=(10,)),
        354649831,
    ),
]


LINEAR_MODELS = [
    (
        autoencoder([20, 10, 20], output_dim=8).init,
        functools.partial(
            linear_squared_error_autoencoder_loss,
            layer_widths=[20, 10, 20]),
        dict(images=(8,)),
        1240982837,
    ),
]


PIECEWISE_LINEAR_MODELS = [
    (
        autoencoder([20, 10, 20], output_dim=8).init,
        functools.partial(
            autoencoder_with_two_losses,
            layer_widths=[20, 10, 20],
            activation=_special_relu,
        ),
        dict(images=(8,)),
        1231987,
    ),
]


SCAN_MODELS = [
    (
        layer_stack_with_scan_mlp([20, 15, 10], output_dim=2).init,
        functools.partial(
            layer_stack_mlp_loss,
            layer_widths=[20, 15, 10],
            activation=_special_tanh,
        ),
        dict(images=(13,), targets=(2,)),
        9812386123,
    ),
    (
        vanilla_rnn_with_scan(20, output_dim=2).init,
        functools.partial(
            vanilla_rnn_with_scan_loss,
            hidden_size=20,
            activation=_special_tanh,
        ),
        dict(images=(7, 13), targets=(7, 2)),
        650981239,
    ),
]
