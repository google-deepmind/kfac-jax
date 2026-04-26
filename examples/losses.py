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
import types

from typing import Any, Sequence, Mapping

import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
from jax.scipy import special
import kfac_jax

Array = kfac_jax.utils.Array
Numeric = kfac_jax.utils.Numeric
Scalar = kfac_jax.utils.Scalar
Params = kfac_jax.utils.Params


def l2_regularizer(
    params: Params,
    haiku_exclude_batch_norm: bool = False,
    haiku_exclude_biases: bool = False,
) -> Array:
  """Computes an L2 regularizer.

  Computes 0.5 * ||params||^2, optionally excluding batch norm parameters
  and/or biases (assuming Haiku model conventions).

  Args:
    params: The model parameters. Must be a Mapping (e.g. a Haiku parameter
      dict).
    haiku_exclude_batch_norm: If True, excludes parameters from modules whose
      names contain "batchnorm" from the regularizer.
    haiku_exclude_biases: If True, excludes parameters named "b" from the
      regularizer.

  Returns:
    The L2 regularizer value: 0.5 * sum of squared parameter values.
  """
  assert isinstance(params, Mapping)

  if haiku_exclude_batch_norm:
    params = hk.data_structures.filter(
        lambda m, _, __: "batchnorm" not in m, params)

  if haiku_exclude_biases:
    params = hk.data_structures.filter(lambda _, n, __: n != "b", params)

  return 0.5 * kfac_jax.utils.inner_product(params, params)


def sigmoid_cross_entropy(
    logits: Array,
    labels: Array,
    weight: Numeric = 1.0,
    register_loss: bool = True,
    mask: Array | None = None,
    extra_registration_kwargs: dict[str, Any] | None = None,
    registration_module: types.ModuleType = kfac_jax,
) -> Array:
  """Sigmoid cross-entropy loss.

  Computes the sigmoid cross-entropy loss for multi-label binary classification,
  averaged over the batch dimension (axis 0), and summed over the other
  dimensions. Each logit is treated as an independent binary prediction.
  Optionally registers the loss with kfac_jax.

  The loss is computed as the mean over the batch of the per-example sum of
  per-label sigmoid cross-entropy values (optionally masked), scaled by
  ``weight``.

  Args:
    logits: The unnormalized predictions, with shape
      ``(batch_size, ..., num_labels)``.
    labels: The binary target labels with the same shape as ``logits``, with
      values in {0, 1}.
    weight: A scalar multiplier for the loss.
    register_loss: If True, registers the loss with kfac_jax.
    mask: An optional binary mask with the same shape as ``logits``. If
      provided, only entries where the mask is 1 contribute to the loss.
    extra_registration_kwargs: Optional extra keyword arguments to pass to the
      kfac_jax loss registration function.
    registration_module: The module containing the kfac_jax loss registration
      function ``register_sigmoid_cross_entropy_loss``.

  Returns:
    The weighted mean sigmoid cross-entropy loss.
  """

  extra_registration_kwargs = extra_registration_kwargs or {}

  if register_loss:
    registration_module.register_sigmoid_cross_entropy_loss(
        logits,
        targets=labels,
        mask=mask,
        weight=weight,
        **extra_registration_kwargs)

  # Code below is copied from Tensorflow:

  zeros = jnp.zeros_like(logits)

  relu_logits = jnp.where(logits >= zeros, logits, zeros)
  neg_abs_logits = jnp.where(logits >= zeros, -logits, logits)

  log_1p = jnp.log1p(jnp.exp(neg_abs_logits))

  loss = jnp.add(relu_logits - logits * labels, log_1p)

  if mask is not None:
    loss = loss * mask

  return weight * jnp.mean(jnp.sum(loss, axis=range(1, loss.ndim)))


def softmax_cross_entropy(
    logits: Array,
    labels: Array,
    weight: Numeric = 1.0,
    register_loss: bool = True,
    mask: Array | None = None,
    extra_registration_kwargs: dict[str, Any] | None = None,
    registration_module: types.ModuleType = kfac_jax,
) -> Array:
  """Softmax cross-entropy loss.

  Computes the softmax cross-entropy loss for multi-class classification,
  averaged over the batch dimension (axis 0), and summed over the other
  dimensions. Optionally registers the loss with kfac_jax.

  Supports both one-hot encoded labels (same shape as ``logits``) and integer
  labels (rank one less than ``logits``). The loss is computed as the mean over
  the batch of the per-example sum of spatial cross-entropy values (optionally
  masked), scaled by ``weight``.

  Args:
    logits: The unnormalized log-probabilities, with shape
      ``(batch_size, ..., num_classes)``.
    labels: The target labels. Either one-hot encoded with the same shape as
      ``logits``, or integer-encoded with shape ``logits.shape[:-1]``.
    weight: A scalar multiplier for the loss.
    register_loss: If True, registers the loss with kfac_jax.
    mask: An optional binary mask with shape ``logits.shape[:-1]``. If
      provided, only entries where the mask is nonzero contribute to the loss.
    extra_registration_kwargs: Optional extra keyword arguments to pass to the
      kfac_jax loss registration function.
    registration_module: The module containing kfac_jax registration function
      ``register_softmax_cross_entropy_loss``.

  Returns:
    The weighted mean softmax cross-entropy loss.
  """

  extra_registration_kwargs = extra_registration_kwargs or {}

  if register_loss:

    registration_module.register_softmax_cross_entropy_loss(
        logits,
        targets=labels,
        mask=mask,
        weight=weight,
        **extra_registration_kwargs)

  max_logits = jnp.max(logits, keepdims=True, axis=-1)

  # It's unclear whether this stop_gradient is a good idea.
  # See https://github.com/jax-ml/jax/issues/13529
  max_logits = lax.stop_gradient(max_logits)

  logits = logits - max_logits

  log_z = special.logsumexp(logits, axis=-1)

  if logits.shape == labels.shape:
    # Labels are encoded as (possibly smoothed) 1-hot vectors
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

  return weight * jnp.mean(jnp.sum(loss, axis=range(1, loss.ndim)))


def squared_error(
    prediction: Array,
    targets: Array,
    weight: Numeric = 1.0,
    register_loss: bool = True,
    extra_registration_kwargs: dict[str, Any] | None = None,
    registration_module: types.ModuleType = kfac_jax,
) -> Array:
  """Squared error loss.

  Computes the squared error loss between predictions and targets, averaged over
  the batch dimension (axis 0), and summed over the other dimensions. Optionally
  registers the loss with kfac_jax.

  The loss is computed as the mean over the batch of the per-example sum of
  squared differences, scaled by ``weight``.

  Args:
    prediction: The model predictions, with shape
      ``(batch_size, ..., num_outputs)``.
    targets: The target values, with the same shape as ``prediction``.
    weight: A scalar multiplier for the loss.
    register_loss: If True, registers the loss with kfac_jax.
    extra_registration_kwargs: Optional extra keyword arguments to pass to the
      kfac_jax loss registration function.
    registration_module: The module containing the kfac_jax loss registration
      function ``register_squared_error_loss``.

  Returns:
    The weighted mean squared error loss.

  Raises:
    ValueError: If ``prediction`` and ``targets`` have different shapes.
  """

  extra_registration_kwargs = extra_registration_kwargs or {}

  if prediction.shape != targets.shape:
    raise ValueError("prediction and targets should have the same shape.")

  if register_loss:
    registration_module.register_squared_error_loss(
        prediction,
        targets=targets,
        weight=weight,
        **extra_registration_kwargs)

  return weight * jnp.mean(jnp.sum(jnp.square(prediction - targets),
                                   axis=range(1, prediction.ndim)))


def top_k_accuracy(
    logits_or_probs: Array,
    labels: Array,
    mask: Array | None = None,
    k: int = 1,
    pmap_axis_name: str | None = None,
) -> Array:
  """Top-k accuracy.

  Computes the fraction of examples for which the true label is among the top-k
  predictions. Supports both one-hot encoded and integer-encoded labels.

  Args:
    logits_or_probs: The unnormalized logits or probabilities, with shape
      ``(batch_size, ..., num_classes)``.
    labels: The target labels. Either one-hot encoded with the same shape as
      ``logits_or_probs``, or integer-encoded with shape
      ``logits_or_probs.shape[:-1]``.
    mask: An optional binary mask with shape ``logits_or_probs.shape[:-1]``. If
      provided, accuracy is computed only over entries where the mask is
      nonzero, using the "all_dims_nonmasked" normalization convention.
    k: The number of top predictions to consider. Defaults to 1.
    pmap_axis_name: The name of the pmap axis to use for pmean. Only used if
      mask is not None for computing the normalization. If None, will compute
      the normalization per device instead, which is arguably suboptimal.

  Returns:
    The top-k accuracy as a scalar.

  Raises:
    ValueError: If the rank of ``labels`` is incompatible with
      ``logits_or_probs``.
  """

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

  correct = correct.astype(logits_or_probs.dtype)

  # Normalization is always "all_dims_nonmasked".
  # TODO(jamesmartens): make this more general?
  if mask is not None:
    sum_mask = jnp.sum(mask)
    sum_mask = kfac_jax.utils.pmean_if_pmap(
        sum_mask, axis_name=pmap_axis_name)
    return jnp.sum(correct * mask) / jnp.where(sum_mask == 0, 1.0, sum_mask)

  return jnp.mean(correct)


def add_label_smoothing(
    labels: Array,
    label_smoothing: float,
    num_classes: int,
    labels_are_one_hot: bool = False,
) -> Array:
  """Adds label smoothing to the labels.

  Applies label smoothing for multi-class classification (for use with softmax
  losses). Integer labels are first converted to one-hot vectors, and then
  smoothed by mixing with a uniform distribution over all classes:
  ``labels = (1 - label_smoothing) * one_hot + label_smoothing / num_classes``.

  Args:
    labels: The target labels. Either integer-encoded or one-hot encoded (set
      ``labels_are_one_hot=True`` for the latter).
    label_smoothing: The label smoothing coefficient. Must be in [0, 1]. A
      value of 0 means no smoothing.
    num_classes: The total number of classes.
    labels_are_one_hot: If True, treats ``labels`` as already one-hot encoded
      and skips the conversion step.

  Returns:
    The smoothed labels as one-hot-like vectors of shape
    ``(..., num_classes)``.

  Raises:
    ValueError: If ``label_smoothing`` is not in [0, 1].
  """

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


def add_binary_label_smoothing(
    labels: Array,
    label_smoothing: float,
) -> Array:
  """Adds label smoothing to binary labels (for use with sigmoid losses).

  Unlike `add_label_smoothing`, which distributes probability mass across
  `num_classes` categories (for use with softmax losses), this smooths each
  binary label independently toward 0.5.

  Args:
    labels: Binary labels with values in {0, 1} (and possibly NaN for missing
      labels).
    label_smoothing: The label smoothing coefficient. Must be in [0, 1]. A value
      of 0 means no smoothing, and a value of 1 means all labels become 0.5.

  Returns:
    The smoothed labels.
  """

  if label_smoothing < 0. or label_smoothing > 1.:
    raise ValueError(f"label_smoothing is {label_smoothing} but should be in "
                     f"[0, 1].")

  if label_smoothing > 0:
    labels = labels * (1. - label_smoothing) + 0.5 * label_smoothing

  return labels


def classifier_loss_and_stats(
    predictions: Array,
    labels_as_int: Array,
    params: Params,
    l2_reg: Scalar,
    haiku_exclude_batch_norm: bool,
    haiku_exclude_biases: bool,
    label_smoothing: float = 0.0,
    top_k_stats: Sequence[int] = (1, 5),
    register_loss: bool = True,
    mask: Array | None = None,
    normalization_mode: str = "batch_size_only",
    extra_registration_kwargs: dict[str, Any] | None = None,
    registration_module: types.ModuleType = kfac_jax,
    loss_type: str = "cross_entropy",
    pmap_axis_name: str | None = None,
) -> tuple[Array, dict[str, Array]]:
  """Classification loss with regularizer and accuracy statistics.

  Args:
    predictions: The output of the model. Logits for loss_type="cross_entropy",
      or quantity to be compared to (possibly smoothed) one-hot targets for
      loss_type="squared_error". Predictions will have shape (batch_size, ...,
      num_classes).
    labels_as_int: The labels to be used in the loss regarded as integers. Must
      be of shape predictions.shape[:-1].
    params: The parameters of the model.
    l2_reg: The L2 regularization coefficient.
    haiku_exclude_batch_norm: Whether to exclude batch norm parameters from the
      L2 regularization (assumes models are Haiku models).
    haiku_exclude_biases: Whether to exclude biases from the L2 regularization
      (assumes models are Haiku models).
    label_smoothing: The label smoothing coefficient.
    top_k_stats: The top-k accuracies to compute.
    register_loss: Whether to register the loss.
    mask: If not None, a binary mask of shape predictions.shape[:-1]. Its
      nonzero values determine which predictions are used in the loss
      computation.
    normalization_mode: The normalization mode to use for the returned loss, one
      of "batch_size_only", "all_dims", or "all_dims_nonmasked".
      "batch_size_only" means the loss is normalized by the batch size.
      "all_dims" means the loss is normalized by the product of the dimensions
      of the predictions array, excluding the last dimension.
      "all_dims_nonmasked" means the loss is normalized by the number of
      nonzero entries of the mask if it is not None.
    extra_registration_kwargs: Extra kwargs to pass to the registration
      functions.
    registration_module: The module containing the loss registration functions
      that will be used. These are 'register_softmax_cross_entropy_loss' and
      'register_squared_error_loss".
    loss_type: The type of loss to use ("cross_entropy" or "squared_error").
    pmap_axis_name: The name of the pmap axis to use for pmean. Only used if
      normalization_mode is "all_dims_nonmasked" for computing the
      normalization. If None, will compute the normalization per device instead,
      which is arguably suboptimal.

  Returns:
    The regularized loss and a dictionary of statistics.
  """

  stats = {}

  batch_size = predictions.shape[0]  # this is per-device

  if labels_as_int.shape != predictions.shape[:-1]:
    raise ValueError(
        f"Shape mismatch: labels_as_int shape ({labels_as_int.shape}) "
        f"not compatible with predictions shape {predictions.shape}"
    )

  if mask is not None and mask.shape != predictions.shape[:-1]:
    raise ValueError(
        f"Shape mismatch: mask shape ({mask.shape}) "
        f"not compatible with predictions shape {predictions.shape}"
    )

  if normalization_mode == "batch_size_only":
    weight = 1.0

  elif normalization_mode == "all_dims":
    weight = 1.0 / kfac_jax.utils.product(predictions.shape[1:-1])

  elif normalization_mode == "all_dims_nonmasked":
    assert mask is not None
    normalizer = jnp.sum(mask)
    normalizer = kfac_jax.utils.pmean_if_pmap(
        normalizer, axis_name=pmap_axis_name)
    normalizer = jnp.where(normalizer == 0, 1.0, normalizer)
    weight = batch_size / normalizer

  else:
    raise ValueError(f"Unrecognized value for normalization_mode: "
                     f"{normalization_mode}")

  labels = add_label_smoothing(
      labels_as_int, label_smoothing, predictions.shape[-1]
  )

  if loss_type == "cross_entropy":

    loss = softmax_cross_entropy(
        predictions,
        labels,
        weight=weight,
        register_loss=register_loss,
        mask=mask,
        extra_registration_kwargs=extra_registration_kwargs,
        registration_module=registration_module,
    )

  elif loss_type == "squared_error":

    if predictions.ndim == labels.ndim + 1:
      labels = jax.nn.one_hot(labels, predictions.shape[-1])

    assert mask is None  # not supported for squared error

    loss = squared_error(
        predictions,
        labels,
        weight=weight,
        register_loss=register_loss,
        extra_registration_kwargs=extra_registration_kwargs,
        registration_module=registration_module,
    )

  else:
    raise ValueError(f"Unknown loss type: {loss_type}")

  if l2_reg > 0.0:

    l2_reg_val = l2_regularizer(
        params, haiku_exclude_batch_norm, haiku_exclude_biases)

    stats["raw_loss"] = loss
    stats["l2_reg_val"] = l2_reg_val

    loss = loss + l2_reg * l2_reg_val

  for k in top_k_stats:
    stats[f"top_{k}_accuracy"] = top_k_accuracy(
        predictions, labels_as_int, mask=mask, k=k)

  return loss, stats
