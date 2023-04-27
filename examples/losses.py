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
from typing import Optional, Sequence, Tuple, Dict

import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
from jax.scipy import special
import kfac_jax

Array = kfac_jax.utils.Array
Numeric = kfac_jax.utils.Numeric
Params = kfac_jax.utils.Params


def l2_regularizer(
    params: Params,
    haiku_exclude_batch_norm: bool,
    haiku_exclude_biases: bool,
) -> Array:
  """Computes an L2 regularizer."""

  if haiku_exclude_batch_norm:
    params = hk.data_structures.filter(  # pytype: disable=wrong-arg-types
        lambda m, n, p: "batchnorm" not in m, params)

  if haiku_exclude_biases:
    params = hk.data_structures.filter(  # pytype: disable=wrong-arg-types
        lambda m, n, p: n != "b", params
    )

  return 0.5 * kfac_jax.utils.inner_product(params, params)


def sigmoid_cross_entropy(
    logits: Array,
    labels: Array,
    weight: float = 1.0,
    register_loss: bool = True,
) -> Array:
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
    logits: Array,
    labels: Array,
    weight: Numeric = 1.0,
    register_loss: bool = True,
    mask: Optional[Array] = None,
) -> Array:
  """Softmax cross entropy loss."""

  if register_loss:
    if not isinstance(weight, float):
      raise NotImplementedError("Non-constant loss weights are not currently "
                                "supported.")

    # Currently the registration functions only support 2D array inputs values
    # for `logits`, and so we need the reshapes below.
    kfac_jax.register_softmax_cross_entropy_loss(
        logits.reshape([-1, logits.shape[-1]]),
        targets=labels.reshape([-1]),
        mask=mask.reshape([-1]) if mask is not None else None,
        weight=weight)

  max_logits = jnp.max(logits, keepdims=True, axis=-1)

  # It's unclear whether this stop_gradient is a good idea.
  # See https://github.com/google/jax/issues/13529
  max_logits = lax.stop_gradient(max_logits)

  logits = logits - max_logits

  log_z = special.logsumexp(logits, axis=-1)

  if logits.shape == labels.shape:
    # Labels are encoded as 1-hot vectors
    loss = -jnp.sum(logits * labels, axis=-1) + log_z

  elif logits.ndim == labels.ndim + 1:
    # Labels are encoded as integers

    # Taken from Optax's softmax_cross_entropy_with_integer_labels:
    label_logits = jnp.take_along_axis(
        logits, labels[..., None], axis=-1)[..., 0]

    loss = -label_logits + log_z

  else:
    raise ValueError(f"The provided labels must have the same rank as the "
                     f"logits - {logits.ndim}, or one less, but got "
                     f"{labels.ndim}.")

  if mask is not None:
    loss = loss * mask

  loss = weight * loss

  # sum over all but the batch dimension
  loss = jnp.sum(loss, axis=range(1, loss.ndim))

  return loss


def squared_error(
    prediction: Array,
    targets: Array,
    weight: float = 1.0,
    register_loss: bool = True,
) -> Array:
  """Squared error loss."""

  if prediction.shape != targets.shape:
    raise ValueError("prediction and targets should have the same shape.")

  if register_loss:
    kfac_jax.register_squared_error_loss(prediction, targets, weight)

  return weight * jnp.sum(jnp.square(prediction - targets), axis=-1)


def top_k_accuracy(
    logits_or_probs: Array,
    labels: Array,
    k: int = 1,
) -> Array:
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

  return jnp.mean(correct.astype(logits_or_probs.dtype))


def add_label_smoothing(
    labels: Array,
    label_smoothing: float,
    num_classes: int,
    labels_are_one_hot: bool = False,
) -> Array:
  """Adds label smoothing to the labels."""

  if label_smoothing < 0. or label_smoothing > 1.:
    raise ValueError(f"label_smoothing is {label_smoothing} but should be in "
                     f"[0, 1].")

  if label_smoothing > 0:

    if not labels_are_one_hot:
      labels = jax.nn.one_hot(labels, num_classes)

    assert labels.shape[-1] == num_classes

    smooth_positives = 1. - label_smoothing
    smooth_negatives = label_smoothing / num_classes
    labels = smooth_positives * labels + smooth_negatives

  return labels


def classifier_loss_and_stats(
    logits: Array,
    labels_as_int: Array,
    params: Params,
    l2_reg: Numeric,
    haiku_exclude_batch_norm: bool,
    haiku_exclude_biases: bool,
    label_smoothing: float = 0.0,
    top_k_stats: Sequence[int] = (1, 5),
    average_loss: bool = True,
    register_loss: bool = True,
    mask: Optional[Array] = None,
    normalization_mode: str = "batch_size_only",
) -> Tuple[Array, Dict[str, Array]]:
  """Softmax cross-entropy with regularizer and accuracy statistics."""

  batch_size = logits.shape[0]

  if labels_as_int.shape[0] != batch_size:
    raise ValueError(f"Size of first dimension of logits ({batch_size}) "
                     f"(i.e. batch size) doesn't match that of labels "
                     f"({labels_as_int.shape[0]})")

  if mask is not None and mask.shape[0] != batch_size:
    raise ValueError(f"Size of first dimension of logits ({batch_size}) "
                     f"(i.e. batch size) doesn't match that of mask "
                     f"({mask.shape[0]})")

  if normalization_mode == "batch_size_only":
    weight = 1.0

  elif normalization_mode == "all_dims":
    weight = 1.0 / kfac_jax.utils.product(logits.shape[1:-1])

  elif normalization_mode == "all_dims_nonmasked":
    assert mask is not None
    weight = batch_size / jnp.sum(mask)

  else:
    raise ValueError(f"Unrecognized value for normalization_mode: "
                     f"{normalization_mode}")

  labels = add_label_smoothing(labels_as_int, label_smoothing, logits.shape[-1])

  softmax_loss = softmax_cross_entropy(
      logits, labels, weight=weight, register_loss=register_loss, mask=mask)

  averaged_raw_loss = jnp.sum(softmax_loss, axis=0) / batch_size

  loss = averaged_raw_loss if average_loss else softmax_loss

  l2_reg_val = l2_regularizer(
      params, haiku_exclude_batch_norm, haiku_exclude_biases)

  regularized_loss = loss + l2_reg * l2_reg_val

  stats = dict(
      raw_loss=averaged_raw_loss,
      l2_reg_val=l2_reg_val,
  )
  for k in top_k_stats:
    stats[f"top_{k}_accuracy"] = top_k_accuracy(logits, labels_as_int, k)

  return regularized_loss, stats
