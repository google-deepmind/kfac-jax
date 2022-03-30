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
"""Vanilla network (derived from a ResNet) with LReLU from the TAT paper."""
import functools
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import chex
import haiku as hk
from jax import nn
import jax.numpy as jnp
from examples import losses
from examples import training
from ml_collections import config_dict
import numpy as np

FloatStrOrBool = Union[str, float, bool]


class ScaledUniformOrthogonal(hk.initializers.Initializer):
  """SUO (+ Delta) initializer for fully-connected and convolutional layers."""

  def __init__(self, scale: float = 1.0, axis: int = -1):
    """Construct a Haiku initializer which uses the SUO distribution.

    Args:
      scale: A float giving an additional scale factor applied on top of the
        standard rescaling used in the SUO distribution. This should be left
        at its default value when using DKS/TAT. (Default: 1.0)
      axis: An int giving the axis corresponding to the "output dimension" of
        the parameter tensor. (Default: -1)
    """

    if axis != -1:
      raise ValueError("Invalid axis value for Delta initializations. "
                       "Must be -1.")
    self.scale = scale
    self.axis = axis

  def __call__(self, shape: chex.Shape, dtype: chex.ArrayDType) -> chex.Array:
    # This has essentially copied from https://github.com/deepmind/dks

    if self.axis != -1:
      raise ValueError("Invalid axis value for Delta initializations. "
                       "Must be -1.")

    if len(shape) != 2:
      # We assume 'weights' is a filter bank when len(shape) != 2

      # In JAX, conv filter banks have the shape
      # [loc_dim_1, loc_dim_2, in_dim, out_dim]
      in_dim = shape[-2]
      out_dim = shape[-1]

      rescale_factor = np.maximum(np.sqrt(out_dim / in_dim), 1.0)

      nonzero_part = hk.initializers.Orthogonal(
          scale=(rescale_factor * self.scale),
          axis=-1)(shape[-2:], dtype)

      if any(s % 2 != 1 for s in shape[:-2]):
        raise ValueError("All spatial axes must have odd length for Delta "
                         "initializations.")

      midpoints = tuple((s - 1) // 2 for s in shape[:-2])

      return jnp.zeros(shape, dtype).at[midpoints].set(nonzero_part)

    else:

      in_dim = np.prod(np.delete(shape, self.axis))
      out_dim = shape[self.axis]

      rescale_factor = np.maximum(np.sqrt(out_dim / in_dim), 1.0)

      return hk.initializers.Orthogonal(
          scale=(rescale_factor * self.scale),
          axis=self.axis)(shape, dtype)


class BlockV2(hk.Module):
  """ResNet V2 block without batch norm or residual connections."""

  def __init__(
      self,
      channels: int,
      stride: Union[int, Sequence[int]],
      bottleneck: bool,
      activation: Callable[[jnp.ndarray], jnp.ndarray],
      w_init: Optional[Any],
      name: Optional[str] = None,
  ):
    """Initializes the module instance."""
    super().__init__(name=name)

    channel_div = 4 if bottleneck else 1
    conv_0 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=1 if bottleneck else 3,
        stride=1 if bottleneck else stride,
        w_init=w_init,
        with_bias=True,
        padding="SAME",
        name="conv_0")

    conv_1 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=3,
        stride=stride if bottleneck else 1,
        w_init=w_init,
        with_bias=True,
        padding="SAME",
        name="conv_1")

    layers = (conv_0, conv_1)

    if bottleneck:
      conv_2 = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=1,
          w_init=w_init,
          with_bias=True,
          padding="SAME",
          name="conv_2")

      layers = layers + (conv_2,)

    self.layers = layers
    self.activation = activation

  def __call__(self, inputs: chex.Array, **_: Any) -> chex.Array:
    out = inputs

    for conv_i in self.layers:
      out = self.activation(out)
      out = conv_i(out)

    return out


class BlockGroup(hk.Module):
  """Higher level block for network implementation."""

  def __init__(
      self,
      channels: int,
      num_blocks: int,
      stride: Union[int, Sequence[int]],
      bottleneck: bool,
      activation: Callable[[jnp.ndarray], jnp.ndarray],
      w_init: Optional[Any],
      name: Optional[str] = None,
  ):
    """Initializes the block group."""

    super().__init__(name=name)

    self.blocks = []
    for i in range(num_blocks):
      self.blocks.append(BlockV2(
          channels=channels,
          stride=(1 if i else stride),
          bottleneck=bottleneck,
          activation=activation,
          w_init=w_init,
          name=f"block_{i}"
      ))

  def __call__(self, inputs: chex.Array, **kwargs: Any) -> chex.Array:
    out = inputs
    for block in self.blocks:
      out = block(out, **kwargs)
    return out


def _check_length(length: int, value: Sequence[int], name: str):
  """Verifies the length of the model."""
  if len(value) != length:
    raise ValueError(f"`{name}` must be of length 4 not {len(value)}")


# The values below are generated using the TAT method with parameter eta=0.9
_ACTIVATIONS_DICT = {
    50: lambda x: nn.leaky_relu(x, 0.4259071946144104) * 1.301119175166785,
    101: lambda x: nn.leaky_relu(x, 0.5704395323991776) * 1.2284042441106242,
    152: lambda x: nn.leaky_relu(x, 0.6386479139328003) * 1.1918827706862754,
}


class LReLUNet(hk.Module):
  """Vanilla network (derived from a ResNet) with LReLU from the TAT paper."""

  CONFIGS = {
      50: {
          "blocks_per_group": (3, 4, 6, 3),
          "bottleneck": True,
          "channels_per_group": (256, 512, 1024, 2048),
      },
      101: {
          "blocks_per_group": (3, 4, 23, 3),
          "bottleneck": True,
          "channels_per_group": (256, 512, 1024, 2048),
      },
      152: {
          "blocks_per_group": (3, 8, 36, 3),
          "bottleneck": True,
          "channels_per_group": (256, 512, 1024, 2048),
      },
  }

  def __init__(
      self,
      num_classes: int,
      depth: int,
      w_init: Optional[Any] = ScaledUniformOrthogonal(),
      logits_config: Optional[Mapping[str, Any]] = None,
      initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
      dropout_rate: float = 0.0,
      name: Optional[str] = None,
  ):
    """Initializes the network module.

    The model has been used in ...
    It mimics a ResNet, but has all batch normalization and residual connections
    removed.

    Args:
      num_classes: The number of classes to classify the inputs into.
      depth: The number of layers.
      w_init: Haiku initializer used to initialize the weights.
      logits_config: A dictionary of keyword arguments for the logits layer.
      initial_conv_config: Keyword arguments passed to the constructor of the
        initial :class:`~haiku.Conv2D` module.
      dropout_rate: A float giving the dropout rate for penultimate layer of the
        network (i.e. right before the layer which produces the class logits).
        (Default: 0.0)
      name: Name of the Sonnet module.
    """
    if depth not in _ACTIVATIONS_DICT:
      raise ValueError(f"Depth {depth} not supported.")

    super().__init__(name=name)
    self.depth = depth
    self.dropout_rate = dropout_rate

    blocks_per_group = LReLUNet.CONFIGS[depth]["blocks_per_group"]
    channels_per_group = LReLUNet.CONFIGS[depth]["channels_per_group"]
    bottleneck = LReLUNet.CONFIGS[depth]["bottleneck"]

    logits_config = dict(logits_config or {})
    logits_config.setdefault("w_init", w_init)
    logits_config.setdefault("name", "logits")

    # Number of blocks in each group.
    _check_length(4, blocks_per_group, "blocks_per_group")
    _check_length(4, channels_per_group, "channels_per_group")

    initial_conv_config = dict(initial_conv_config or {})
    initial_conv_config.setdefault("output_channels", 64)
    initial_conv_config.setdefault("kernel_shape", 7)
    initial_conv_config.setdefault("stride", 2)
    initial_conv_config.setdefault("with_bias", True)
    initial_conv_config.setdefault("padding", "SAME")
    initial_conv_config.setdefault("name", "initial_conv")
    initial_conv_config.setdefault("w_init", w_init)

    self.activation = _ACTIVATIONS_DICT[depth]
    self.initial_conv = hk.Conv2D(**initial_conv_config)

    self.block_groups = []
    strides = (1, 2, 2, 2)
    for i in range(4):
      self.block_groups.append(BlockGroup(
          channels=channels_per_group[i],
          num_blocks=blocks_per_group[i],
          stride=strides[i],
          bottleneck=bottleneck,
          activation=self.activation,
          w_init=w_init,
          name=f"block_group_{i}",
      ))

    self.logits = hk.Linear(num_classes, **logits_config)

  def __call__(
      self,
      inputs: chex.Array,
      is_training: bool,
      **kwargs: Any
  ) -> chex.Array:
    out = inputs
    out = self.initial_conv(out)
    out = hk.max_pool(
        out, window_shape=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding="SAME")

    for block_group in self.block_groups:
      out = block_group(out, is_training=is_training, **kwargs)

    out = self.activation(out)
    out = jnp.mean(out, axis=(1, 2))

    if self.dropout_rate > 0.0 and is_training:
      out = hk.dropout(hk.next_rng_key(), self.dropout_rate, out)

    return self.logits(out)


def lrelunet(
    num_classes: int = 1000,
    depth: int = 101,
    **kwargs: Any,
) -> hk.Transformed:
  """Constructs a Haiku transformed object of the LReLUNet101 network."""
  def func(
      batch: Union[chex.Array, Mapping[str, chex.Array]],
      is_training: bool
  ) -> chex.Array:
    """Evaluates the network."""
    if isinstance(batch, dict):
      batch = batch["images"]
    model = LReLUNet(num_classes=num_classes, depth=depth, **kwargs)
    return model(batch, is_training=is_training)
  return hk.transform(func)


def lrelunet_loss(
    params: hk.Params,
    rng: chex.PRNGKey,
    batch: Mapping[str, chex.Array],
    is_training: bool,
    l2_reg: chex.Numeric,
    label_smoothing: float = 0.1,
    average_loss: bool = True,
    num_classes: int = 1000,
    depth: int = 101,
    **kwargs: Any,
) -> Tuple[
    chex.Array,
    Union[Dict[str, chex.Array], Tuple[hk.State, Dict[str, chex.Array]]]
]:
  """Evaluates the loss of the LReLUNet model."""
  logits = lrelunet(num_classes=num_classes, depth=depth, **kwargs).apply(
      params, rng, batch["images"], is_training)

  return losses.classifier_loss_and_stats(
      logits=logits,
      labels_as_int=batch["labels"],
      params=params,
      l2_reg=l2_reg,
      haiku_exclude_batch_norm=True,
      haiku_exclude_biases=True,
      label_smoothing=label_smoothing,
      average_loss=average_loss,
  )


class LReLUNetImageNetExperiment(training.ImageNetExperiment):
  """Jaxline experiment class for running the LReLUNet on ImageNet."""

  def __init__(
      self,
      mode: str,
      init_rng: chex.PRNGKey,
      config: config_dict.ConfigDict
  ):
    """Initializes the network instance."""
    super().__init__(
        mode=mode,
        init_rng=init_rng,
        config=config,
        init_parameters_func=functools.partial(
            lrelunet(num_classes=1000, **config.model_kwargs).init,
            is_training=True,
        ),
        model_loss_func=functools.partial(
            lrelunet_loss,
            l2_reg=config.l2_reg,
            num_classes=1000,
            **config.model_kwargs,
            **config.loss_kwargs,
        ),
        has_aux=True,
        has_rng=True,
        has_func_state=False,
    )
