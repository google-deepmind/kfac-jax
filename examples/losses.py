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
"""Utility functions for computing and automatically registering losses."""
from typing import Dict, Sequence, Tuple

import chex
import haiku as hk
from jax import lax
import jax.numpy as jnp
from jax.scipy import special
import kfac_jax


def l2_regularizer(
    params: kfac_jax.utils.Params,
    haiku_exclude_batch_norm: bool,
    haiku_exclude_biases: bool,
) -> chex.Array:
  """Computes an L2 regularizer."""
  if haiku_exclude_batch_norm:
    params = hk.data_structures.filter(
        lambda m, n, p: "batchnorm" not in m, params)
  if haiku_exclude_biases:
    params = hk.data_structures.filter(
        lambda m, n, p: n != "b", params)
  return 0.5 * kfac_jax.utils.inner_product(params, params)


def sigmoid_cross_entropy(
    logits: chex.Array,
    labels: chex.Array,
    weight: float = 1.0,
    register_loss: bool = True,
) -> chex.Array:
  """Sigmoid cross-entropy loss."""
  if register_loss:
    kfac_jax.register_sigmoid_cross_entropy_loss(logits, labels, weight)
  # Code is copied from Tensorflow.
  zeros = jnp.zeros_like(logits)
  relu_logits = jnp.where(logits >= zeros, logits, zeros)
  neg_abs_logits = jnp.where(logits >= zeros, -logits, logits)
  log_1p = jnp.log1p(jnp.exp(neg_abs_logits))
  return weight * jnp.add(relu_logits - logits * labels, log_1p)


def softmax_cross_entropy(
    logits: chex.Array,
    labels: chex.Array,
    weight: float = 1.0,
    register_loss: bool = True,
) -> chex.Array:
  """Softmax cross entropy loss."""
  if register_loss:
    kfac_jax.register_softmax_cross_entropy_loss(logits, labels, weight)
  max_logits = jnp.max(logits, keepdims=True, axis=-1)
  logits = logits - max_logits
  log_z = special.logsumexp(logits, axis=-1)
  if logits.shape == labels.shape:
    # Dense labels
    return weight * (- jnp.sum(logits * labels, axis=-1) + log_z)
  elif logits.ndim == labels.ndim + 1:
    # One hot encoded labels
    idx = jnp.arange(labels.shape[0])
    return weight * (- logits[idx, labels] + log_z)
  else:
    raise ValueError(f"The provided labels must have the same rank as the "
                     f"logits - {logits.ndim}, or one less, but got "
                     f"{labels.ndim}.")


def squared_error(
    prediction: chex.Array,
    targets: chex.Array,
    weight: float = 1.0,
    register_loss: bool = True,
) -> chex.Array:
  """Squared error loss."""
  if prediction.shape != targets.shape:
    raise ValueError("prediction and targets should have the same shape.")
  if register_loss:
    kfac_jax.register_squared_error_loss(prediction, targets, weight)
  return weight * jnp.sum(jnp.square(prediction - targets), axis=-1)


def top_k_accuracy(
    logits_or_probs: chex.Array,
    labels: chex.Array,
    k: int = 1,
) -> chex.Array:
  """Top-k accuracy."""
  if labels.ndim == logits_or_probs.ndim:
    # One hot labels
    labels = jnp.argmax(labels, axis=-1)
  elif labels.ndim + 1 != logits_or_probs.ndim:
    raise ValueError(f"The provided labels must have the same rank as the "
                     f"logits_or_probs - {logits_or_probs.ndim}, or one less, "
                     f"{labels.ndim}.")
  if k == 1:
    indices = jnp.argmax(logits_or_probs, axis=-1)
    correct = jnp.equal(indices, labels)
  else:
    _, indices = lax.top_k(logits_or_probs, k=k)
    correct = jnp.equal(indices, labels[..., None])
    correct = jnp.sum(correct, axis=-1)
  return jnp.mean(correct.astype(logits_or_probs.dtype), axis=0)


def add_label_smoothing(
    labels: chex.Array,
    label_smoothing: float,
    num_classes: int,
    labels_are_one_hot: bool = False,
) -> chex.Array:
  """Adds label smoothing to the labels."""
  if label_smoothing < 0. or label_smoothing > 1.:
    raise ValueError(f"label_smoothing is {label_smoothing} and should be in "
                     f"[0, 1].")
  if label_smoothing > 0:
    if not labels_are_one_hot:
      labels = hk.one_hot(labels, num_classes)
    assert labels.shape[-1] == num_classes
    smooth_positives = 1. - label_smoothing
    smooth_negatives = label_smoothing / num_classes
    labels = smooth_positives * labels + smooth_negatives
  return labels


def classifier_loss_and_stats(
    logits: chex.Array,
    labels_as_int: chex.Array,
    params: kfac_jax.utils.Params,
    l2_reg: chex.Numeric,
    haiku_exclude_batch_norm: bool,
    haiku_exclude_biases: bool,
    label_smoothing: float = 0.0,
    top_k_stats: Sequence[int] = (1, 5),
    average_loss: bool = True,
    register_loss: bool = True,
) -> Tuple[chex.Array, Dict[str, chex.Array]]:
  """Softmax cross-entropy with regularizer and accuracy statistics."""

  labels = add_label_smoothing(labels_as_int, label_smoothing, logits.shape[-1])

  softmax_loss = softmax_cross_entropy(logits, labels,
                                       register_loss=register_loss)
  averaged_raw_loss = jnp.mean(softmax_loss, axis=0)
  loss = averaged_raw_loss if average_loss else softmax_loss

  regularizer = l2_regularizer(
      params, haiku_exclude_batch_norm, haiku_exclude_biases)

  regularized_loss = loss + l2_reg * regularizer

  stats = dict(
      raw_loss=averaged_raw_loss,
      regularizer=regularizer,
  )
  for k in top_k_stats:
    stats[f"top_{k}_accuracy"] = top_k_accuracy(logits, labels_as_int, k)

  return regularized_loss, stats
