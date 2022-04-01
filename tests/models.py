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

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import kfac_jax

tags = kfac_jax.layers_and_loss_tags
loss_functions = kfac_jax.loss_functions
utils = kfac_jax.utils


LayerValues = List[Tuple[chex.Array, chex.Array]]
LayerInputs = Tuple[chex.Array, LayerValues, Optional[Tuple[chex.Array, ...]]]
LossOutputs = Union[
    List[List[chex.Array]],
    List[chex.Array],
    Tuple[List[chex.Array], LayerValues]
]


def _extract_params(
    instance: hk.Module,
    names: Sequence[str],
) -> Tuple[chex.Array, Optional[chex.Array]]:
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

  def __call__(self, inputs: LayerInputs, *_) -> LayerInputs:
    x, layer_values, aux = inputs
    y = super().__call__(x, precision=jax.lax.Precision.HIGHEST)
    if aux is not None:
      y, aux = y + aux[0], aux[1:]
    layer_values.append((x, y))
    if self._explicit_tagging:
      params = _extract_params(self, ("w", "b"))
      y = tags.register_dense(
          y, x, *params,
          dimension_numbers=(((1,), (0,)), ((), ())),
          precision=(jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
          preferred_element_type=None,
      )
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
          lhs_shape=x.shape,
          padding=((0, 1), (0, 1)),
          precision=(jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
          preferred_element_type=None,
          rhs_dilation=(1, 1),
          rhs_shape=params[0].shape,
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

  def __call__(self, inputs: LayerInputs, *_) -> LayerInputs:
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


def _modify_func(
    func: Callable[[chex.Array], chex.Array]
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

  def _sample_n(self, key: chex.PRNGKey, n: int) -> chex.Array:
    del key  # not used
    return jnp.repeat(jnp.round(self.probs)[None], n, axis=0)


class _DeterministicBernoulliNegativeLogProbLoss(
    loss_functions.MultiBernoulliNegativeLogProbLoss):
  """A negative log-likelihood loss using the `DeterministicBernoulli`."""

  @property
  def dist(self):
    return _DeterministicBernoulli(logits=self._logits, dtype=jnp.int32)

DeterministicBernoulliNegativeLogProbLoss_tag = loss_functions.tags.LossTag(
    _DeterministicBernoulliNegativeLogProbLoss, num_inputs=1)


def _register_deterministic_bernoulli(
    logits: chex.Array,
    targets: chex.Array,
    weight=1.0
) -> Tuple[chex.Array, chex.Array]:
  return DeterministicBernoulliNegativeLogProbLoss_tag.bind(
      logits, targets, weight=weight)


class _DeterministicCategorical(distrax.Categorical):
  """A fake deterministic bernoulli distribution, to make KFAC deterministic."""

  def _sample_n(self, key: chex.PRNGKey, n: int) -> chex.Array:
    del key  # not used
    return jnp.repeat(jnp.round(self.probs)[None], n, axis=0)


class _DeterministicCategoricalNegativeLogProbLoss(
    loss_functions.CategoricalLogitsNegativeLogProbLoss):
  """A negative log-likelihood loss using the `DeterministicBernoulli`."""

  @property
  def dist(self) -> _DeterministicCategorical:
    return _DeterministicCategorical(logits=self._logits, dtype=jnp.int32)

_DeterministicCategoricalNegativeLogProbLoss_tag = loss_functions.tags.LossTag(
    _DeterministicCategoricalNegativeLogProbLoss, num_inputs=1)


def _register_deterministic_categorical(
    logits: chex.Array,
    targets: chex.Array,
    weight=1.0
) -> Tuple[chex.Array, chex.Array]:
  return DeterministicBernoulliNegativeLogProbLoss_tag.bind(
      logits, targets, weight=weight)


def autoencoder(
    layer_widths: Sequence[int],
    explicit_tagging: bool = False,
    activation: Callable[[LayerInputs], LayerInputs] = _special_tanh,
) -> hk.Transformed:
  """Constructs a Haiku transformed object of the autoencoder network."""
  def func(
      batch: Union[chex.Array, Mapping[str, chex.Array]],
      aux: Optional[Tuple[chex.Array, ...]] = None,
  ) -> Tuple[chex.Array, LayerValues]:
    images = batch["images"] if isinstance(batch, Mapping) else batch
    images = images.reshape([images.shape[0], -1])
    layers = []
    for width in layer_widths:
      layers.append(_Linear(output_size=width,
                            explicit_tagging=explicit_tagging))
      layers.append(activation)
    layers.append(_Linear(output_size=images.shape[-1],
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
) -> chex.Array:
  """A linear autoencoder with squared error."""
  outputs, _ = autoencoder(
      layer_widths, explicit_tagging, activation=_special_identity,
  ).apply(params, batch["images"])
  outputs, _ = loss_functions.register_squared_error_loss(
      outputs, batch["images"])
  loss = jnp.mean(jnp.sum((outputs - batch["images"]) ** 2, axis=-1))
  return loss + l2_reg * utils.norm(params)


def autoencoder_deterministic_loss(
    params: utils.Params,
    batch: utils.Batch,
    layer_widths: Sequence[int],
    l2_reg: float = 0.0,
    explicit_tagging: bool = False,
    activation: Callable[[LayerInputs], LayerInputs] = _special_tanh,
) -> chex.Array:
  """Evaluate the autoencoder with a deterministic loss."""
  logits, _ = autoencoder(
      layer_widths, explicit_tagging, activation=activation,
  ).apply(params, batch["images"])
  logits, _ = _register_deterministic_bernoulli(logits, batch["images"])
  loss = - distrax.Bernoulli(logits=logits).log_prob(batch["images"])
  loss = jnp.mean(jnp.sum(loss, axis=-1)).astype(logits.dtype)
  return loss + l2_reg * utils.norm(params)


def autoencoder_with_two_losses(
    params: utils.Params,
    batch: utils.Batch,
    layer_widths: Sequence[int],
    aux: Optional[Tuple[chex.Array, ...]] = None,
    explicit_tagging: bool = False,
    return_registered_losses_inputs: bool = False,
    return_layer_values: bool = False,
    activation: Callable[[LayerInputs], LayerInputs] = _special_tanh,
) -> LossOutputs:
  """Evaluate the autoencoder with two losses."""
  logits, layer_values = autoencoder(
      layer_widths, explicit_tagging, activation=activation,
  ).apply(params, batch["images"], aux)

  # Register both losses in KFAC
  logits1, _ = loss_functions.register_multi_bernoulli_predictive_distribution(
      logits, batch["images"])
  logits2, _ = loss_functions.register_normal_predictive_distribution(
      logits, batch["images"], weight=0.1)

  if return_registered_losses_inputs:
    return [[logits1], [logits2]]
  else:
    loss_1 = - distrax.Bernoulli(logits=logits1).log_prob(batch["images"])
    scale_diag = jnp.ones_like(logits2) * jnp.sqrt(0.5)
    loss_2 = - distrax.MultivariateNormalDiag(
        loc=logits2, scale_diag=scale_diag).log_prob(batch["images"])

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
  """Constructs a Haiku transformed object of the autoencoder network."""
  def func(
      batch: Union[chex.Array, Mapping[str, chex.Array]],
      aux: Optional[Tuple[chex.Array, ...]] = None,
  ) -> Tuple[chex.Array, LayerValues]:
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
) -> chex.Array:
  """Evaluate the autoencoder with a deterministic loss."""
  logits, _ = conv_classifier(
      num_classes, layer_channels, explicit_tagging, activation=activation
  ).apply(params, batch["images"])
  logits, _ = _register_deterministic_categorical(logits, batch["labels"])
  loss = - distrax.Categorical(logits=logits).log_prob(batch["labels"])
  loss = jnp.mean(jnp.sum(loss, axis=-1)).astype(logits.dtype)
  return loss + l2_reg * utils.norm(params)


def conv_classifier_loss(
    params: utils.Params,
    batch: utils.Batch,
    num_classes: int,
    layer_channels: Sequence[int],
    aux: Optional[Tuple[chex.Array, ...]] = None,
    l2_reg: float = 0.0,
    explicit_tagging: bool = False,
    return_registered_losses_inputs: bool = False,
    return_layer_values: bool = False,
    activation: Callable[[LayerInputs], LayerInputs] = _special_tanh,
) -> LossOutputs:
  """Evaluates the autoencoder with a deterministic loss."""
  logits, layer_values = conv_classifier(
      num_classes, layer_channels, explicit_tagging, activation=activation
  ).apply(params, batch["images"], aux=aux)
  logits, _ = loss_functions.register_categorical_predictive_distribution(
      logits, batch["labels"])
  loss = - distrax.Categorical(logits=logits).log_prob(batch["labels"])
  loss = loss + l2_reg * utils.norm(params)

  if return_registered_losses_inputs:
    return [[logits]]
  else:
    if return_layer_values:
      return [loss], layer_values
    else:
      return [loss]


NON_LINEAR_MODELS = [
    (
        autoencoder([20, 10, 20]).init,
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
        autoencoder([20, 10, 20]).init,
        functools.partial(
            linear_squared_error_autoencoder_loss,
            layer_widths=[20, 10, 20]),
        dict(images=(8,)),
        1240982837,
    ),
]


PIECEWISE_LINEAR_MODELS = [
    (
        autoencoder([20, 10, 20]).init,
        functools.partial(
            autoencoder_with_two_losses,
            layer_widths=[20, 10, 20],
            activation=_special_relu,
        ),
        dict(images=(8,)),
        1231987,
    ),
]
