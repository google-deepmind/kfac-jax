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
"""Module containing the ImplicitExactCurvature class."""
from typing import Callable, Sequence
import jax
import jax.numpy as jnp
from kfac_jax._src import loss_functions
from kfac_jax._src import tracer
from kfac_jax._src import utils

# Types for annotation
Array = utils.Array
Shape = utils.Shape
LossFunction = loss_functions.LossFunction
LossFunctionsTuple = tuple[loss_functions.LossFunction, ...]
LossFunctionsSequence = Sequence[loss_functions.LossFunction]
LossFunctionInputs = loss_functions.LossFunctionInputs
LossFunctionInputsSequence = Sequence[loss_functions.LossFunctionInputs]
LossFunctionInputsTuple = tuple[loss_functions.LossFunctionInputs, ...]


class ImplicitExactCurvature:
  """Represents all exact curvature matrices never constructed explicitly."""

  def __init__(
      self,
      func: utils.Func,
      params_index: int = 0,
      batch_size_extractor: Callable[[utils.Batch], int] =
      utils.default_batch_size_extractor,
  ):
    """Initializes the ImplicitExactCurvature instance.

    Args:
      func: The model function, which should have at least one registered loss.
      params_index: The index of the parameters argument in arguments list of
        ``func``.
      batch_size_extractor: A function that takes as input the function
        arguments and returns the batch size for a single device.
        (Default: ``kfac.utils.default_batch_size_extractor``)
    """
    self.compute_losses = tracer.compute_all_losses(
        func=func,
        params_index=params_index
    )
    self._loss_tags_vjp, _ = tracer.loss_tags_vjp(
        func=func,
        params_index=params_index
    )
    self._loss_tags_jvp, _ = tracer.loss_tags_jvp(
        func=func,
        params_index=params_index,
    )
    self._loss_tags_hvp, _ = tracer.loss_tags_hvp(
        func=func,
        params_index=params_index,
    )
    self._batch_size_extractor = batch_size_extractor

  def batch_size(self, func_args: utils.FuncArgs) -> int:
    """The expected batch size given a list of loss instances."""
    return self._batch_size_extractor(func_args[-1])

  @classmethod
  def _multiply_loss_fisher(
      cls,
      losses: Sequence[loss_functions.NegativeLogProbLoss],
      loss_vectors: LossFunctionInputsSequence,
  ) -> LossFunctionInputsTuple:
    """Multiplies ``loss_vectors`` by the Fisher of the total loss."""
    assert len(losses) == len(loss_vectors)
    return tuple(loss.multiply_fisher(vec)
                 for loss, vec in zip(losses, loss_vectors))

  @classmethod
  def _multiply_loss_ggn(
      cls,
      losses: LossFunctionsSequence,
      loss_vectors: LossFunctionInputsSequence,
  ) -> LossFunctionInputsTuple:
    """Multiplies ``loss_vectors`` by the GGN of the total loss."""
    return tuple(loss.multiply_ggn(vec)
                 for loss, vec in zip(losses, loss_vectors))

  @classmethod
  def _multiply_loss_fisher_factor(
      cls,
      losses: Sequence[loss_functions.NegativeLogProbLoss],
      loss_inner_vectors: Sequence[Array],
  ) -> LossFunctionInputsTuple:
    """Multiplies the vectors with the Fisher factors of each loss.

    Args:
      losses: A sequence of loss instances.
      loss_inner_vectors: A sequence of vectors, each corresponding to one
        instance of a loss in losses.

    Returns:
      The product of all vectors with the factors of the Fisher of each the
      losses.
    """
    assert len(losses) == len(loss_inner_vectors)
    return tuple(loss.multiply_fisher_factor(vec)
                 for loss, vec in zip(losses, loss_inner_vectors))

  @classmethod
  def _multiply_loss_ggn_factor(
      cls,
      losses: Sequence[loss_functions.LossFunction],
      loss_inner_vectors: Sequence[Array],
  ) -> LossFunctionInputsTuple:
    """Multiplies the vectors with the GGN factors of each loss.

    Args:
      losses: A sequence of loss instances.
      loss_inner_vectors: A sequence of vectors, each corresponding to one
        instance of a loss in losses.

    Returns:
      The product of all vectors with the factors of the GGN of each the
      losses.
    """
    return tuple(loss.multiply_ggn_factor(vec)
                 for loss, vec in zip(losses, loss_inner_vectors))

  @classmethod
  def _multiply_loss_fisher_factor_transpose(
      cls,
      losses: Sequence[loss_functions.NegativeLogProbLoss],
      loss_vectors: LossFunctionInputsSequence,
  ) -> tuple[Array, ...]:
    """Multiplies the vectors with the transposed Fisher factors of each loss.

    Args:
      losses: A sequence of loss instances.
      loss_vectors: A sequence of vectors, each corresponding to one instance of
        a loss in losses.

    Returns:
      The product of all vectors with the factors of the Fisher of each the
      losses.
    """
    assert len(losses) == len(loss_vectors)
    return tuple(loss.multiply_fisher_factor_transpose(vec)
                 for loss, vec in zip(losses, loss_vectors))

  @classmethod
  def _multiply_loss_ggn_factor_transpose(
      cls,
      losses: LossFunctionsSequence,
      loss_vectors: LossFunctionInputsSequence,
  ) -> tuple[Array, ...]:
    """Multiplies the vectors with the transposed GGN factors of each loss.

    Args:
      losses: A sequence of loss instances.
      loss_vectors: A sequence of vectors, each corresponding to one instance of
        a loss in losses.

    Returns:
      The product of all vectors with the factors of the GGN of each the
      losses.
    """
    return tuple(loss.multiply_ggn_factor_transpose(vec)
                 for loss, vec in zip(losses, loss_vectors))

  @classmethod
  def _assert_losses_same(
      cls,
      losses1: Sequence[loss_functions.LossFunction],
      losses2: Sequence[loss_functions.LossFunction],
  ) -> None:
    """Asserts that the two losses sequence are equivalent."""
    assert len(losses1) == len(losses2)
    for loss1, loss2 in zip(losses1, losses2):
      assert isinstance(loss1, type(loss2))
      inputs1 = jax.tree_util.tree_leaves(loss1.parameter_dependants)
      inputs2 = jax.tree_util.tree_leaves(loss2.parameter_dependants)
      for in1, in2 in zip(inputs1, inputs2):
        assert in1.shape == in2.shape
        assert in1.dtype == in2.dtype

  @utils.auto_scope_method
  def multiply_hessian(
      self,
      func_args: utils.FuncArgs,
      parameter_structured_vector: utils.Params,
  ) -> utils.Params:
    """Multiplies the vector with the Hessian matrix of the total loss.

    Args:
      func_args: The inputs to the model function, on which to evaluate the
        Hessian matrix.
      parameter_structured_vector: The vector which to multiply with the Hessian
        matrix.

    Returns:
      The product ``Hv``.
    """
    vector, _ = self._loss_tags_hvp(func_args, parameter_structured_vector)
    batch_size = self.batch_size(func_args)

    assert utils.abstract_objects_equal(parameter_structured_vector, vector)

    return utils.scalar_div(vector, batch_size)

  @utils.auto_scope_method
  def multiply_jacobian(
      self,
      func_args: utils.FuncArgs,
      parameter_structured_vector: utils.Params,
      return_loss_objects: bool = False,
  ) -> (
      LossFunctionInputsTuple |
      tuple[LossFunctionInputsTuple, LossFunctionsTuple]
  ):
    """Multiplies a vector by the model's Jacobian.

    Args:
      func_args: The inputs to the model function.
      parameter_structured_vector: A vector in the same structure as the
        parameters of the model.
      return_loss_objects: If set to `True` will return as an additional output
        the loss objects evaluated at the provided function arguments.

    Returns:
      The product ``J v``, where ``J`` is the model's Jacobian and ``v`` is
      given by ``parameter_structured_vector``.
    """
    losses, jacobian_vectors = self._loss_tags_jvp(
        func_args, parameter_structured_vector)
    if return_loss_objects:
      return jacobian_vectors, losses
    return jacobian_vectors

  @utils.auto_scope_method
  def multiply_jacobian_transpose(
      self,
      func_args: utils.FuncArgs,
      loss_input_vectors: LossFunctionInputsSequence,
      return_loss_objects: bool = False,
  ) -> utils.Params | tuple[utils.Params, LossFunctionsTuple]:
    """Multiplies a vector by the model's transposed Jacobian.

    Args:
      func_args: The inputs to the model function.
      loss_input_vectors: A sequence over losses of sequences of arrays that
        are the size of the loss's inputs. This represents the vector to be
        multiplied.
      return_loss_objects: If set to `True` will return as an additional output
        the loss objects evaluated at the provided function arguments.

    Returns:
      The product ``J^T v``, where ``J`` is the model's Jacobian and ``v`` is
      given by ``loss_inner_vectors``.
    """
    losses, vjp = self._loss_tags_vjp(func_args)
    vector = vjp(loss_input_vectors)
    if return_loss_objects:
      return vector, losses
    return vector

  @utils.auto_scope_method
  def multiply_fisher(
      self,
      func_args: utils.FuncArgs,
      parameter_structured_vector: utils.Params,
  ) -> utils.Params:
    """Multiplies the vector with the Fisher matrix of the total loss.

    Args:
      func_args: The inputs to the model function, on which to evaluate the
        Fisher matrix.
      parameter_structured_vector: The vector which to multiply with the Fisher
        matrix.

    Returns:
      The product ``Fv``.
    """
    losses: Sequence[loss_functions.NegativeLogProbLoss]
    jacobian_vectors, losses = self.multiply_jacobian(
        func_args, parameter_structured_vector, True)  # pytype: disable=annotation-type-mismatch

    if any(not isinstance(l, loss_functions.NegativeLogProbLoss)
           for l in losses):
      raise ValueError("To use `multiply_fisher` all registered losses must "
                       "be a subclass of `NegativeLogProbLoss`.")

    loss_fisher_jacobian_vectors = self._multiply_loss_fisher(
        losses, jacobian_vectors)  # pytype: disable=wrong-arg-types

    vector = self.multiply_jacobian_transpose(
        func_args, loss_fisher_jacobian_vectors)
    batch_size = self.batch_size(func_args)

    assert utils.abstract_objects_equal(parameter_structured_vector, vector)

    return utils.scalar_div(vector, batch_size)

  @utils.auto_scope_method
  def multiply_ggn(
      self,
      func_args: utils.FuncArgs,
      parameter_structured_vector: utils.Params,
  ) -> utils.Params:
    """Multiplies the vector with the GGN matrix of the total loss.

    Args:
      func_args: The inputs to the model function, on which to evaluate the GGN
        matrix.
      parameter_structured_vector: The vector which to multiply with the GGN
        matrix.

    Returns:
      The product ``Gv``.
    """
    jacobian_vectors, losses = self.multiply_jacobian(
        func_args, parameter_structured_vector, True)

    loss_ggn_jacobian_vectors = self._multiply_loss_ggn(
        losses, jacobian_vectors)

    vector = self.multiply_jacobian_transpose(
        func_args, loss_ggn_jacobian_vectors)
    batch_size = self.batch_size(func_args)

    assert utils.abstract_objects_equal(parameter_structured_vector, vector)

    return utils.scalar_div(vector, batch_size)

  @utils.auto_scope_method
  def multiply_fisher_factor_transpose(
      self,
      func_args: utils.FuncArgs,
      parameter_structured_vector: utils.Params,
  ) -> tuple[Array, ...]:
    """Multiplies the vector with the transposed factor of the Fisher matrix.

    Args:
      func_args: The inputs to the model function, on which to evaluate the
        Fisher matrix.
      parameter_structured_vector: The vector which to multiply with the Fisher
        matrix.

    Returns:
      The product ``B^T v``, where ``F = BB^T``.
    """
    jacobian_vectors, losses = self.multiply_jacobian(
        func_args, parameter_structured_vector, True)

    if any(not isinstance(l, loss_functions.NegativeLogProbLoss)
           for l in losses):
      raise ValueError("To use `multiply_fisher` all registered losses must "
                       "be a subclass of `NegativeLogProbLoss`.")

    loss_vectors = self._multiply_loss_fisher_factor_transpose(
        losses, jacobian_vectors)  # pytype: disable=wrong-arg-types
    batch_size = self.batch_size(func_args)

    return utils.scalar_div(loss_vectors, jnp.sqrt(batch_size))

  @utils.auto_scope_method
  def multiply_ggn_factor_transpose(
      self,
      func_args: utils.FuncArgs,
      parameter_structured_vector: utils.Params,
  ) -> tuple[Array, ...]:
    """Multiplies the vector with the transposed factor of the GGN matrix.

    Args:
      func_args: The inputs to the model function, on which to evaluate the GGN
        matrix.
      parameter_structured_vector: The vector which to multiply with the GGN
        matrix.

    Returns:
      The product ``B^T v``, where ``G = BB^T``.
    """
    jacobian_vectors, losses = self.multiply_jacobian(
        func_args, parameter_structured_vector, True)

    vectors = self._multiply_loss_ggn_factor_transpose(losses, jacobian_vectors)
    batch_size = self.batch_size(func_args)

    return utils.scalar_div(vectors, jnp.sqrt(batch_size))

  @utils.auto_scope_method
  def multiply_fisher_factor(
      self,
      func_args: utils.FuncArgs,
      loss_inner_vectors: Sequence[Array],
  ) -> utils.Params:
    """Multiplies the vector with the factor of the Fisher matrix.

    Args:
      func_args: The inputs to the model function, on which to evaluate the
        Fisher matrix.
      loss_inner_vectors: The vector which to multiply with the Fisher factor
        matrix.

    Returns:
      The product ``Bv``, where ``F = BB^T``.
    """
    losses: Sequence[loss_functions.NegativeLogProbLoss]
    losses, vjp = self._loss_tags_vjp(func_args)  # pytype: disable=annotation-type-mismatch

    if any(not isinstance(l, loss_functions.NegativeLogProbLoss)
           for l in losses):
      raise ValueError("To use `multiply_fisher` all registered losses must "
                       "be a subclass of `NegativeLogProbLoss`.")

    fisher_factor_vectors = self._multiply_loss_fisher_factor(
        losses, loss_inner_vectors)  # pytype: disable=wrong-arg-types

    vectors = vjp(fisher_factor_vectors)
    batch_size = self.batch_size(func_args)

    return utils.scalar_div(vectors, jnp.sqrt(batch_size))

  @utils.auto_scope_method
  def multiply_ggn_factor(
      self,
      func_args: utils.FuncArgs,
      loss_inner_vectors: Sequence[Array],
  ) -> utils.Params:
    """Multiplies the vector with the factor of the GGN matrix.

    Args:
      func_args: The inputs to the model function, on which to evaluate the GGN
        matrix.
      loss_inner_vectors: The vector which to multiply with the GGN factor
        matrix.

    Returns:
      The product ``Bv``, where ``G = BB^T``.
    """
    losses, vjp = self._loss_tags_vjp(func_args)

    ggn_factor_vectors = self._multiply_loss_ggn_factor(
        losses, loss_inner_vectors)

    vectors = vjp(ggn_factor_vectors)

    batch_size = self.batch_size(func_args)

    return utils.scalar_div(vectors, jnp.sqrt(batch_size))

  def get_loss_inner_vector_shapes_and_batch_size(
      self,
      func_args: utils.FuncArgs,
      mode: str
  ) -> tuple[tuple[Shape, ...], int]:
    """Get shapes of loss inner vectors, and the batch size.

    Args:
      func_args: The inputs to the model function.
      mode: A string representing the type of curvature matrix for the loss
       inner vectors. Can be "fisher" or "ggn".

    Returns:
      Shapes of loss inner vectors in a tuple, and the batch size as an int.
    """
    losses, _ = self._loss_tags_vjp(func_args)
    shapes = []

    for loss in losses:
      if mode == "fisher":
        if not isinstance(loss, loss_functions.NegativeLogProbLoss):
          raise ValueError(f"To use {mode=}, each loss must be a subclass of "
                           "`NegativeLogProbLoss`.")
        shapes.append(loss.fisher_factor_inner_shape)
      elif mode == "ggn":
        shapes.append(loss.ggn_factor_inner_shape)

      else:
        raise ValueError(f"Unrecognized mode: {mode}")

    return tuple(shapes), self.batch_size(func_args)

  def get_loss_input_shapes_and_batch_size(
      self,
      func_args: utils.FuncArgs,
  ) -> tuple[tuple[tuple[Shape, ...], ...], int]:
    """Get shapes of loss input vectors, and the batch size.

    Args:
      func_args: The inputs to the model function.

    Returns:
      A tuple over losses of tuples containing the shapes of their different
      inputs, and the batch size (as an int).
    """
    losses, _ = self._loss_tags_vjp(func_args)  # pytype: disable=attribute-error  # always-use-return-annotations
    batch_size = self.batch_size(func_args)

    return (tuple(tuple(x.shape for x in loss.parameter_dependants)
                  for loss in losses),
            batch_size)

