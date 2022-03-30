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
"""Haiku implementation of a small convolutional classifier for MNIST."""
import functools
from typing import Dict, Mapping, Tuple, Union

import chex
import haiku as hk
import jax
import jax.numpy as jnp

from examples import losses
from examples import training


def convolutional_classifier() -> hk.Transformed:
  """Constructs a Haiku transformed object of the classifier network."""
  def func(batch: Union[chex.Array, Mapping[str, chex.Array]]) -> chex.Array:
    """Evaluates the classifier."""
    if isinstance(batch, Mapping):
      batch = batch["images"]
    if batch.ndim == 3:
      # Add extra channel dimension
      batch = jnp.expand_dims(batch, axis=-1)

    model = hk.Sequential([
        hk.Conv2D(2, kernel_shape=(5, 5)),
        jax.nn.relu,
        hk.MaxPool((2, 2), strides=(2, 2), padding="SAME"),
        hk.Conv2D(4, kernel_shape=(5, 5)),
        jax.nn.relu,
        hk.MaxPool((2, 2), strides=(2, 2), padding="SAME"),
        hk.Flatten(),
        hk.Linear(32),
        jax.nn.relu,
        hk.Linear(10)
    ])
    return model(batch)
  return hk.without_apply_rng(hk.transform(func))


def classifier_loss(
    params: hk.Params,
    batch: Mapping[str, chex.Array],
    l2_reg: chex.Numeric,
    is_training: bool,
    average_loss: bool = True,
) -> Tuple[chex.Array, Dict[str, chex.Array]]:
  """Evaluates the loss of the classifier network."""
  del is_training  # not used

  logits = convolutional_classifier().apply(params, batch["images"])

  cross_entropy = losses.softmax_cross_entropy(logits, batch["labels"])

  if average_loss:
    cross_entropy = jnp.mean(cross_entropy)

  params_l2 = losses.l2_regularizer(params, False, False)
  regularized_loss = cross_entropy + l2_reg * params_l2

  accuracy = losses.top_k_accuracy(logits, batch["labels"], 1)

  return regularized_loss, dict(accuracy=accuracy)


class ClassifierMnistExperiment(training.MnistExperiment):
  """Jaxline experiment class for running the MNIST classifier."""

  def __init__(self, mode: str, init_rng: jnp.ndarray, config):
    super().__init__(
        supervised=True,
        flatten_images=False,
        mode=mode,
        init_rng=init_rng,
        config=config,
        init_parameters_func=convolutional_classifier().init,
        model_loss_func=functools.partial(
            classifier_loss, l2_reg=config.l2_reg),
        has_aux=True,
        has_rng=False,
        has_func_state=False,
    )
