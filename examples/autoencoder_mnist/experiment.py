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
"""Haiku implementation of the standard MNIST Autoencoder."""
import functools
from typing import Mapping, Tuple, Union, Dict

import haiku as hk
import jax
from jax import nn
import jax.numpy as jnp

import kfac_jax
from examples import losses
from examples import training


Array = kfac_jax.utils.Array
Numeric = kfac_jax.utils.Numeric
PRNGKey = kfac_jax.utils.PRNGKey


def autoencoder() -> hk.Transformed:
  """Constructs a Haiku transformed object of the autoencoder."""
  def func(batch: Union[Array, Mapping[str, Array]]) -> Array:
    """Evaluates the autoencoder."""
    if isinstance(batch, Mapping):
      batch = batch["images"]
    batch = batch.reshape([batch.shape[0], -1])
    model = hk.Sequential([
        hk.Linear(1000),
        jax.nn.tanh,
        hk.Linear(500),
        jax.nn.tanh,
        hk.Linear(250),
        jax.nn.tanh,
        hk.Linear(30),
        hk.Linear(250),
        jax.nn.tanh,
        hk.Linear(500),
        jax.nn.tanh,
        hk.Linear(1000),
        jax.nn.tanh,
        hk.Linear(batch.shape[-1]),
    ])
    return model(batch)
  return hk.without_apply_rng(hk.transform(func))


def autoencoder_loss(
    params: hk.Params,
    batch: Union[Array, Mapping[str, Array]],
    l2_reg: Numeric,
    is_training: bool,
    average_loss: bool = True,
) -> Tuple[Array, Dict[str, Array]]:
  """Evaluates the loss of the autoencoder."""

  if isinstance(batch, Mapping):
    batch = batch["images"]

  logits = autoencoder().apply(params, batch)

  cross_entropy = jnp.sum(losses.sigmoid_cross_entropy(logits, batch), axis=-1)
  averaged_cross_entropy = jnp.mean(cross_entropy)

  loss: Array = averaged_cross_entropy if average_loss else cross_entropy

  l2_reg_val = losses.l2_regularizer(params, False, False)
  if is_training:
    loss = loss + l2_reg * l2_reg_val

  error = nn.sigmoid(logits) - batch.reshape([batch.shape[0], -1])
  mean_squared_error = jnp.mean(jnp.sum(error * error, axis=1), axis=0)

  return loss, dict(
      cross_entropy=averaged_cross_entropy,
      l2_reg_val=l2_reg_val,
      mean_squared_error=mean_squared_error,
  )


class AutoencoderMnistExperiment(training.MnistExperiment):
  """Jaxline experiment class for running the MNIST Autoencoder."""

  def __init__(self, mode: str, init_rng: PRNGKey, config):
    super().__init__(
        supervised=False,
        flatten_images=True,
        mode=mode,
        init_rng=init_rng,
        config=config,
        init_parameters_func=autoencoder().init,
        model_loss_func=functools.partial(
            autoencoder_loss, l2_reg=config.l2_reg),
        has_aux=True,
        has_rng=False,
        has_func_state=False,
    )
