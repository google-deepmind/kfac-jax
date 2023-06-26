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
"""K-FAC curvature explicit and implicit estimators.

Curvature matrices are always defined in terms of some single differentiable
function of the parameters and inputs. In all cases in this module this quantity
is not the output from the model function (usually provided as argument to the
constructor of each curvature matrix), but is the sum of all losses
(weighted accordingly) which have been registered with a loss tag in the
computation graph of the model function. This quantity is referred to as the
``total_loss``.

In this module there are three curvature matrices considered:
  ``H`` - the Hessian matrix
  ``F`` - the Fisher matrix
  ``G`` - The Generalized Gauss-Newton(GGN) matrix
Vectors that are multiplied by a curvature matrix (or any of its matrix powers)
are always represented as a PyTree structure, equivalent to the parameters of
the model function. In all functions such vector is named
``parameter_structured_vector`` in the argument list.

Factors of a matrix ``M`` are defined as matrices ``B`` such that ``BB^T = M``.
If we have to left-multiply ``B`` with a vector ``v``, than ``v`` has the same
format as if we have to multiply the whole curvature matrix ``M``. However the
second size of ``B`` is not clearly defined (and can be different for the
different curvature matrices). In all methods working with factors, e.g. if we
need to right multiply ``B`` with a vector ``v`` or the result of left
multiplying ``B`` by a parameter structured vector, then the provided vector
``v`` should be a list of lists of arrays. Each element of ``v`` corresponds to
a single loss registered in the model function, and its elements should have the
shapes as the corresponding ``loss.XXX_inner_shapes`` (XXX=Hessian, Fisher or
GGN). In all function such vector is named ``loss_vectors`` in the argument
list.

See for example: www.cs.utoronto.ca/~jmartens/docs/HF_book_chapter.pdf and
https://arxiv.org/abs/1412.1193 for more information about the Hessian, Fisher
and GGN matrices and how to compute matrix-vector products.
"""
import abc
import functools
from typing import Any, Callable, Optional, Sequence, Mapping, Generic, TypeVar, Tuple, Union, Dict

import jax
from jax import scipy
import jax.numpy as jnp
from kfac_jax._src import curvature_blocks
from kfac_jax._src import layers_and_loss_tags as tags
from kfac_jax._src import loss_functions
from kfac_jax._src import tracer
from kfac_jax._src import utils
import numpy as np

# Types for annotation
Array = utils.Array
PRNGKey = utils.PRNGKey
Numeric = utils.Numeric
Scalar = utils.Scalar
Shape = utils.Shape
CurvatureBlockCtor = Callable[
    [tags.LayerTagEqn, str],
    curvature_blocks.CurvatureBlock
]
StateType = TypeVar("StateType")

# Special global variables
_ESTIMATION_MODES = ("fisher_gradients", "fisher_empirical", "fisher_exact",
                     "fisher_curvature_prop", "ggn_exact", "ggn_curvature_prop")

_DEFAULT_TAG_TO_BLOCK_CTOR: Dict[str, CurvatureBlockCtor] = dict(
    dense_tag=curvature_blocks.DenseTwoKroneckerFactored,
    conv2d_tag=curvature_blocks.Conv2DTwoKroneckerFactored,
    generic_tag=curvature_blocks.NaiveDiagonal,
    scale_and_shift_tag=curvature_blocks.ScaleAndShiftDiagonal,
)


def get_default_tag_to_block_ctor(
    tag_name: str
) -> Optional[CurvatureBlockCtor]:
  """Returns the default curvature block constructor for the give tag name."""
  return _DEFAULT_TAG_TO_BLOCK_CTOR.get(tag_name)


def set_default_tag_to_block_ctor(
    tag_name: str,
    block_ctor: CurvatureBlockCtor
) -> None:
  """Sets the default curvature block constructor for the given tag."""
  _DEFAULT_TAG_TO_BLOCK_CTOR[tag_name] = block_ctor


def set_multi_default_tag_to_block_ctor(
    tags_to_block_ctor: Mapping[str, CurvatureBlockCtor]
):
  _DEFAULT_TAG_TO_BLOCK_CTOR.update(tags_to_block_ctor)


class ImplicitExactCurvature:
  """Represents all exact curvature matrices never constructed explicitly."""

  def __init__(
      self,
      func: utils.Func,
      params_index: int = 0,
      batch_size_extractor: Callable[[utils.Batch], Numeric] =
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
    self._loss_tags_vjp = tracer.loss_tags_vjp(
        func=func,
        params_index=params_index
    )
    self._loss_tags_jvp = tracer.loss_tags_jvp(
        func=func,
        params_index=params_index,
    )
    self._loss_tags_hvp = tracer.loss_tags_hvp(
        func=func,
        params_index=params_index,
    )
    self._batch_size_extractor = batch_size_extractor

  def batch_size(self, func_args: utils.FuncArgs) -> Numeric:
    """The expected batch size given a list of loss instances."""
    return self._batch_size_extractor(func_args[-1])

  @classmethod
  def _multiply_loss_fisher(
      cls,
      losses: Sequence[loss_functions.NegativeLogProbLoss],
      loss_vectors: Sequence[Sequence[Array]]
  ) -> Tuple[Tuple[Array, ...], ...]:
    """Multiplies ``loss_vectors`` by the Fisher of the total loss."""
    assert len(losses) == len(loss_vectors)
    return tuple(loss.multiply_fisher(vec)
                 for loss, vec in zip(losses, loss_vectors))

  @classmethod
  def _multiply_loss_ggn(
      cls,
      losses: Sequence[loss_functions.LossFunction],
      loss_vectors: Sequence[Sequence[Array]]
  ) -> Tuple[Tuple[Array, ...], ...]:
    """Multiplies ``loss_vectors`` by the GGN of the total loss."""
    return tuple(loss.multiply_ggn(vec)
                 for loss, vec in zip(losses, loss_vectors))

  @classmethod
  def _multiply_loss_fisher_factor(
      cls,
      losses: Sequence[loss_functions.NegativeLogProbLoss],
      loss_inner_vectors: Sequence[Array],
  ) -> Tuple[Tuple[Array, ...], ...]:
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
  ) -> Tuple[Tuple[Array, ...], ...]:
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
      loss_vectors: Sequence[Sequence[Array]]
  ) -> Tuple[Array, ...]:
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
      losses: Sequence[loss_functions.LossFunction],
      loss_vectors: Sequence[Sequence[Array]]
  ) -> Tuple[Array, ...]:
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
    losses, jacobian_vectors = self._loss_tags_jvp(
        func_args, parameter_structured_vector)
    if any(not isinstance(l, loss_functions.NegativeLogProbLoss)
           for l in losses):
      raise ValueError("To use `multiply_fisher` all registered losses must "
                       "be a subclass of `NegativeLogProbLoss`.")
    _, vjp = self._loss_tags_vjp(func_args)
    self._assert_losses_same(losses, _)

    loss_fisher_jacobian_vectors = self._multiply_loss_fisher(
        losses, jacobian_vectors)
    vector = vjp(loss_fisher_jacobian_vectors)
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
    losses, jacobian_vectors = self._loss_tags_jvp(
        func_args, parameter_structured_vector)
    _, vjp = self._loss_tags_vjp(func_args)
    self._assert_losses_same(losses, _)

    loss_ggn_jacobian_vectors = self._multiply_loss_ggn(
        losses, jacobian_vectors)
    vector = vjp(loss_ggn_jacobian_vectors)
    batch_size = self.batch_size(func_args)

    assert utils.abstract_objects_equal(parameter_structured_vector, vector)

    return utils.scalar_div(vector, batch_size)

  @utils.auto_scope_method
  def multiply_fisher_factor_transpose(
      self,
      func_args: utils.FuncArgs,
      parameter_structured_vector: utils.Params,
  ) -> Tuple[Array, ...]:
    """Multiplies the vector with the transposed factor of the Fisher matrix.

    Args:
      func_args: The inputs to the model function, on which to evaluate the
        Fisher matrix.
      parameter_structured_vector: The vector which to multiply with the Fisher
        matrix.

    Returns:
      The product ``B^T v``, where ``F = BB^T``.
    """
    losses: Sequence[loss_functions.NegativeLogProbLoss]
    losses, jacobian_vectors = self._loss_tags_jvp(
        func_args, parameter_structured_vector)
    if any(not isinstance(l, loss_functions.NegativeLogProbLoss)
           for l in losses):
      raise ValueError("To use `multiply_fisher` all registered losses must "
                       "be a subclass of `NegativeLogProbLoss`.")
    loss_vectors = self._multiply_loss_fisher_factor_transpose(
        losses, jacobian_vectors)
    batch_size = self.batch_size(func_args)
    return utils.scalar_div(loss_vectors, jnp.sqrt(batch_size))

  @utils.auto_scope_method
  def multiply_ggn_factor_transpose(
      self,
      func_args: utils.FuncArgs,
      parameter_structured_vector: utils.Params,
  ) -> Tuple[Array, ...]:
    """Multiplies the vector with the transposed factor of the GGN matrix.

    Args:
      func_args: The inputs to the model function, on which to evaluate the GGN
        matrix.
      parameter_structured_vector: The vector which to multiply with the GGN
        matrix.

    Returns:
      The product ``B^T v``, where ``G = BB^T``.
    """
    losses, jacobian_vectors = self._loss_tags_jvp(
        func_args, parameter_structured_vector)
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
    losses, vjp = self._loss_tags_vjp(func_args)

    if any(not isinstance(l, loss_functions.NegativeLogProbLoss)
           for l in losses):
      raise ValueError("To use `multiply_fisher` all registered losses must "
                       "be a subclass of `NegativeLogProbLoss`.")

    fisher_factor_vectors = self._multiply_loss_fisher_factor(
        losses, loss_inner_vectors)

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

  @utils.auto_scope_method
  def multiply_jacobian_transpose(
      self,
      func_args: utils.FuncArgs,
      loss_input_vectors: Sequence[Sequence[Array]],
  ) -> utils.Params:
    """Multiplies a vector by the model's transposed Jacobian.

    Args:
      func_args: The inputs to the model function.
      loss_input_vectors: A sequence over losses of sequences of arrays that
        are the size of the loss's inputs. This represents the vector to be
        multiplied.

    Returns:
      The product ``J^T v``, where ``J`` is the model's Jacobian and ``v`` is
      is given by ``loss_inner_vectors``.
    """
    _, vjp = self._loss_tags_vjp(func_args)
    return vjp(loss_input_vectors)

  def get_loss_inner_vector_shapes_and_batch_size(
      self,
      func_args: utils.FuncArgs,
      mode: str
  ) -> Tuple[Tuple[Shape, ...], int]:
    """Get shapes of loss inner vectors, and the batch size.

    Args:
      func_args: The inputs to the model function.
      mode: A string representing the type of curvature matrix for the loss
       inner vectors. Can be "fisher" or "ggn".

    Returns:
      Shapes of loss inner vectors in a tuple, and the batch size as an int.
    """
    losses, _ = self._loss_tags_vjp(func_args)  # pytype: disable=attribute-error  # always-use-return-annotations
    batch_size = self.batch_size(func_args)

    if mode == "fisher":
      return (tuple(loss.fisher_factor_inner_shape for loss in losses),  # pytype: disable=bad-return-type  # numpy-scalars
              batch_size)
    elif mode == "ggn":
      return tuple(loss.ggn_factor_inner_shape for loss in losses), batch_size  # pytype: disable=bad-return-type  # numpy-scalars
    else:
      raise ValueError(f"Unrecognized mode: {mode}")

  def get_loss_input_shapes_and_batch_size(
      self,
      func_args: utils.FuncArgs
  ) -> Tuple[Tuple[Tuple[Shape, ...], ...], int]:
    """Get shapes of loss input vectors, and the batch size.

    Args:
      func_args: The inputs to the model function.

    Returns:
      A tuple over losses of tuples containing the shapes of their different
      inputs, and the batch size (as an int).
    """
    losses, _ = self._loss_tags_vjp(func_args)  # pytype: disable=attribute-error  # always-use-return-annotations
    batch_size = self.batch_size(func_args)

    return (tuple(tuple(x.shape for x in loss.parameter_dependants)  # pytype: disable=bad-return-type  # numpy-scalars
                  for loss in losses),
            batch_size)


class CurvatureEstimator(Generic[StateType], utils.Finalizable):
  """An abstract curvature estimator class.

  This is a class that abstracts away the process of estimating a curvature
  matrix and provides many useful functionalities for interacting with it.
  The state of the estimator contains two parts: the estimated curvature
  internal representation, as well as potential cached values of different
  expression involving the curvature matrix (for example matrix powers).
  The cached values are only updated once you call the method
  :func:`~CurvatureEstimator.update_cache`. Multiple methods contain the keyword
  argument ``use_cached`` which specify whether you want to compute the
  corresponding expression using the current curvature estimate or used a cached
  version.

  Attributes:
    func: The model evaluation function.
    params_index: The index of the parameters argument in arguments list of
      ``func``.
    default_estimation_mode: The estimation mode which to use by default when
      calling :func:`~CurvatureEstimator.update_curvature_matrix_estimate`.
  """

  def __init__(
      self,
      func: utils.Func,
      params_index: int = 0,
      default_estimation_mode: str = "fisher_gradients",
  ):
    """Initializes the CurvatureEstimator instance.

    Args:
      func: The model function, which should have at least one registered loss.
      params_index: The index of the parameters argument in arguments list of
        ``func``.
      default_estimation_mode: The estimation mode which to use by default when
        calling :func:`~CurvatureEstimator.update_curvature_matrix_estimate`.
    """
    if default_estimation_mode not in _ESTIMATION_MODES:
      raise ValueError("Unrecognised default_estimation_mode "
                       f"{default_estimation_mode}.")
    super().__init__()
    self.func = func
    self.params_index = params_index
    self.default_estimation_mode = default_estimation_mode

  @property
  def default_mat_type(self) -> str:
    """The type of matrix that this estimator is approximating."""
    idx = self.default_estimation_mode.index("_")
    return self.default_estimation_mode[:idx]

  @property
  @abc.abstractmethod
  def dim(self) -> int:
    """The number of elements of all parameter variables together."""

  @abc.abstractmethod
  def init(
      self,
      rng: PRNGKey,
      func_args: utils.FuncArgs,
      exact_powers_to_cache: Optional[curvature_blocks.ScalarOrSequence],
      approx_powers_to_cache: Optional[curvature_blocks.ScalarOrSequence],
      cache_eigenvalues: bool = False,
  ) -> StateType:
    """Initializes the state for the estimator.

    Args:
      rng: The PRNGKey which to be used for any randomness of the
          initialization.
      func_args: Example function arguments, which to be used to trace the model
        function and initialize the state.
      exact_powers_to_cache: A single value, or multiple values in a list, which
        specify which exact matrix powers that each block should be caching.
        Matrix powers for which you intend to call
        ``self.multiply_matrix_power``, ``self.multiply_inverse`` or
        ``self.multiply`` with ``exact_power=True`` and ``use_cached=True`` must
        be provided here.
      approx_powers_to_cache: A single value, or multiple values in a list,
        which specify approximate matrix powers that each block should be
        caching. Matrix powers for which you intend to call
        ``self.multiply_matrix_power``, ``self.multiply_inverse`` or
        ``self.multiply`` with ``exact_power=False`` and ``use_cached=True``
        must be provided here.
      cache_eigenvalues: Specifies whether each block should be caching the
          eigenvalues of its approximate curvature.
    Returns:
      The initialized state of the estimator.
    """

  @abc.abstractmethod
  def sync(
      self,
      state: StateType,
      pmap_axis_name: Optional[str],
  ) -> StateType:
    """Synchronizes across devices the state of the estimator."""

  @abc.abstractmethod
  def multiply_matpower(
      self,
      state: StateType,
      parameter_structured_vector: utils.Params,
      identity_weight: Numeric,
      power: Scalar,
      exact_power: bool,
      use_cached: bool,
      pmap_axis_name: Optional[str],
  ) -> utils.Params:
    """Computes ``(CurvatureMatrix + identity_weight I)**power`` times ``vector``.

    Args:
      state: The state of the estimator.
      parameter_structured_vector: A vector in the same structure as the
          parameters of the model.
      identity_weight: Specifies the weight of the identity element that is
          added to the curvature matrix. This can be either a scalar value or a
          list/tuple of scalar in which case each value specifies the weight
          individually for each block.
      power: The power to which you want to raise the matrix
          ``(EstimateCurvature + identity_weight I)``.
      exact_power: When set to ``True`` the matrix power of
          ``EstimateCurvature + identity_weight I`` is computed exactly.
          Otherwise this method might use a cheaper approximation, which *may*
          vary across different blocks.
      use_cached: Whether to use a cached (and possibly stale) version of the
          curvature matrix estimate.
      pmap_axis_name: The name of any pmap axis, which will be used for
          aggregating any computed values over multiple devices, as well as
          parallelizing the computation over devices in a block-wise fashion.

    Returns:
      A parameter structured vector containing the product.
    """

  def multiply(
      self,
      state: StateType,
      parameter_structured_vector: utils.Params,
      identity_weight: Numeric,
      exact_power: bool,
      use_cached: bool,
      pmap_axis_name: Optional[str],
  ) -> utils.Params:
    """Computes ``(CurvatureMatrix + identity_weight I)`` times ``vector``."""

    return self.multiply_matpower(
        state=state,
        parameter_structured_vector=parameter_structured_vector,
        identity_weight=identity_weight,
        power=1,
        exact_power=exact_power,
        use_cached=use_cached,
        pmap_axis_name=pmap_axis_name
    )

  def multiply_inverse(
      self,
      state: StateType,
      parameter_structured_vector: utils.Params,
      identity_weight: Numeric,
      exact_power: bool,
      use_cached: bool,
      pmap_axis_name: Optional[str],
  ) -> utils.Params:
    """Computes ``(CurvatureMatrix + identity_weight I)^-1`` times ``vector``."""

    return self.multiply_matpower(
        state=state,
        parameter_structured_vector=parameter_structured_vector,
        identity_weight=identity_weight,
        power=-1,
        exact_power=exact_power,
        use_cached=use_cached,
        pmap_axis_name=pmap_axis_name
    )

  @abc.abstractmethod
  def eigenvalues(
      self,
      state: StateType,
      use_cached: bool,
  ) -> Array:
    """Computes the eigenvalues of the curvature matrix.

    Args:
      state: The state of the estimator.
      use_cached: Whether to use a cached versions of the eigenvalues or to use
        the most recent curvature estimates to compute them. The cached version
        are going to be *at least* as fresh as the last time you called
        :func:`~CurvatureEstimator.update_cache`  with ``eigenvalues=True``.

    Returns:
      A single array containing the eigenvalues of the curvature matrix.
    """

  @abc.abstractmethod
  def update_curvature_matrix_estimate(
      self,
      state: StateType,
      ema_old: Numeric,
      ema_new: Numeric,
      batch_size: Numeric,
      rng: PRNGKey,
      func_args: utils.FuncArgs,
      estimation_mode: Optional[str] = None,
  ) -> StateType:
    """Updates the estimator's curvature estimates.

    Args:
      state: The state of the estimator to update.
      ema_old: Specifies the weight of the old value when computing the updated
        estimate in the moving average.
      ema_new: Specifies the weight of the new value when computing the updated
        estimate in the moving average.
      batch_size: The batch size.
      rng: A PRNGKey to be used for any potential sampling in the estimation
        process.
      func_args: A structure with the values of the inputs to the traced
        function (the ``tagged_func`` passed into the constructor) which to be
        used for the estimation process. Should have the same structure as the
        argument ``func_args`` passed in the constructor.
      estimation_mode: The type of curvature estimator to use. By default
        (e.g. if ``None``) will use ``self.default_estimation_mode``. One of:

        * fisher_gradients - the basic estimation approach from the original
          K-FAC paper.

        * fisher_curvature_prop - method which estimates the Fisher using
          self-products of random 1/-1 vectors times "half-factors" of the
          Fisher, as described `here <https://arxiv.org/abs/1206.6464>`__.

        * fisher_exact - is the obvious generalization of Curvature
          Propagation to compute the exact Fisher (modulo any additional
          diagonal or Kronecker approximations) by looping over one-hot vectors
          for each coordinate of the output instead of using 1/-1 vectors. It is
          more expensive to compute than the other three options by a factor
          equal to the output dimension, roughly speaking.

        * fisher_empirical - computes the 'empirical' Fisher information
          matrix (which uses the data's distribution for the targets, as
          opposed to the true Fisher which uses the model's distribution) and
          requires that each registered loss have specified targets.

        * ggn_curvature_prop - Analogous to fisher_curvature_prop, but
          estimates the Generalized Gauss-Newton matrix (GGN).

        * ggn_exact - Analogous to fisher_exact, but estimates the Generalized
          Gauss-Newton matrix (GGN).

    Returns:
      The updated state.
    """

  @abc.abstractmethod
  def update_cache(
      self,
      state: StateType,
      identity_weight: Numeric,
      exact_powers: Optional[curvature_blocks.ScalarOrSequence],
      approx_powers: Optional[curvature_blocks.ScalarOrSequence],
      eigenvalues: bool,
      pmap_axis_name: Optional[str],
  ) -> StateType:
    """Updates the estimator cached values.

    Args:
      state: The state of the estimator to update.
      identity_weight: Specified the weight of the identity element that is
          added to the curvature matrix. This can be either a scalar value or a
          list/tuple of scalar in which case each value specifies the weight
          individually for each block.
      exact_powers: Specifies which exact matrix powers in the cache should be
          updated.
      approx_powers: Specifies which approximate matrix powers in the cache
          should be updated.
      eigenvalues: Specifies whether to update the cached eigenvalues
          of each block. If they have not been cached before, this will create
          an entry with them in the block's cache.
      pmap_axis_name: The name of any pmap axis, which will be used for
          aggregating any computed values over multiple devices, as well as
          parallelizing the computation over devices in a block-wise fashion.

    Returns:
      The updated state.
    """

  @abc.abstractmethod
  def to_dense_matrix(self, state: StateType) -> Array:
    """Returns an explicit dense array representing the curvature matrix."""


class BlockDiagonalCurvature(
    CurvatureEstimator["BlockDiagonalCurvature.State"]):
  """Block diagonal curvature estimator class."""

  @utils.register_state_class
  class State(utils.State):
    """Persistent state of the estimator.

    Attributes:
      synced: A Jax boolean, specifying if the state has been synced across
        devices (this does not include the cache, which is never explicitly
        synced).
      blocks_states: A tuple of the state of the estimator corresponding to each
        block.
    """
    synced: Array
    blocks_states: Tuple[curvature_blocks.CurvatureBlock.State, ...]

  def __init__(
      self,
      func: utils.Func,
      params_index: int = 0,
      default_estimation_mode: str = "fisher_gradients",
      layer_tag_to_block_ctor:
      Optional[Mapping[str, CurvatureBlockCtor]] = None,
      index_to_block_ctor:
      Optional[Mapping[Tuple[int, ...], CurvatureBlockCtor]] = None,
      auto_register_tags: bool = True,
      distributed_multiplies: bool = True,
      distributed_cache_updates: bool = True,
      **auto_register_kwargs: Any,
  ):
    """Initializes the curvature instance.

    Args:
      func: The model function, which should have at least one registered loss.
      params_index: The index of the parameters argument in arguments list of
        ``func``.
      default_estimation_mode: The estimation mode which to use by default when
        calling ``self.update_curvature_matrix_estimate``.
      layer_tag_to_block_ctor: An optional dict mapping tags to specific classes
        of block approximations, which to override the default ones.
      index_to_block_ctor: An optional dict mapping a specific block parameter
        indices to specific classes of block approximation, which to override
        the default ones. To get the correct indices check
        ``estimator.indices_to_block_map``.
      auto_register_tags: Whether to automatically register layer tags for
        parameters that have not been manually registered. For further details
        see ``tag_graph_matcher.auto_register_tags``.
      distributed_multiplies: Whether to distribute the curvature matrix
        multiplication operations across the different devices in a block-wise
        fashion. If False, each device will (redundantly) perform the operations
        for all of the blocks.
      distributed_cache_updates: Whether to distribute the cache
        update multiplication operations across the different devices in a
        block-wise fashion. If False, each device will (redundantly) perform
        the operations for all of the blocks.
      **auto_register_kwargs: Any keyword arguments to pass to into the auto
        registration function.
    """
    super().__init__(func, params_index, default_estimation_mode)
    self._index_to_block_ctor = index_to_block_ctor or dict()
    self._layer_tag_to_block_ctor = layer_tag_to_block_ctor or dict()
    self._auto_register_tags = auto_register_tags
    self._auto_register_kwargs = auto_register_kwargs
    self._vjp = tracer.layer_tags_vjp(
        func=func,
        params_index=params_index,
        auto_register_tags=auto_register_tags,
        **auto_register_kwargs
    )
    # Initialized during finalization
    self._jaxpr: Optional[tracer.ProcessedJaxpr] = None
    self._blocks: Optional[Tuple[curvature_blocks.CurvatureBlock]] = None

    self._distributed_multiplies = distributed_multiplies
    self._distributed_cache_updates = distributed_cache_updates

  def _check_finalized(self):
    if not self.finalized:
      raise ValueError("The estimator has not been finalized. Call `init` or "
                       "`finalize` first.")

  def _create_blocks(self):
    """Creates all the curvature blocks instances in ``self._blocks``."""
    assert self._jaxpr is not None

    blocks_list = []
    counters = dict()

    for tag_eqn, idx in zip(self._jaxpr.layer_tags, self._jaxpr.layer_indices):  # pytype: disable=attribute-error  # always-use-return-annotations

      # Correctly get the block class
      if idx in self._index_to_block_ctor:
        cls = self._index_to_block_ctor[idx]

      elif tag_eqn.primitive.name in self._layer_tag_to_block_ctor:
        cls = self._layer_tag_to_block_ctor[tag_eqn.primitive.name]

      elif tag_eqn.primitive.name in _DEFAULT_TAG_TO_BLOCK_CTOR:
        cls = _DEFAULT_TAG_TO_BLOCK_CTOR[tag_eqn.primitive.name]

      else:
        raise ValueError(f"Did not find anywhere a block class for tag "
                         f"{tag_eqn.primitive.name}.")

      if "name" in tag_eqn.params:

        block_name = tag_eqn.params["name"]
        assert block_name not in counters
        counters[block_name] = 1

      else:
        if isinstance(cls, functools.partial):
          block_name = cls.func.__name__
        else:
          block_name = cls.__name__

        c = counters.get(block_name, 0)
        counters[block_name] = c + 1
        block_name += "__" + str(c)

      blocks_list.append(cls(tag_eqn, block_name))

    self._blocks = tuple(blocks_list)

  @property
  def blocks(self) -> Optional[Tuple[curvature_blocks.CurvatureBlock]]:
    """The tuple of :class:`~CurvatureBlock` instances used for each layer."""
    self._check_finalized()
    return self._blocks

  @property
  def num_blocks(self) -> int:
    """The number of separate blocks that this estimator has."""
    return len(self.blocks)

  @property
  def block_dims(self) -> Shape:
    """The number of elements of all parameter variables for each block."""
    return tuple(block.dim for block in self.blocks)

  @property
  def dim(self) -> int:
    """The number of elements of all parameter variables together."""
    return sum(self.block_dims)

  @property
  def jaxpr(self) -> tracer.ProcessedJaxpr:
    self._check_finalized()
    return self._jaxpr  # pytype: disable=bad-return-type  # always-use-return-annotations

  @property
  def params_structure_vector_of_indices(self) -> utils.Params:
    """A tree structure with parameters replaced by their indices."""
    return jax.tree_util.tree_unflatten(
        self.jaxpr.params_tree, range(len(self.jaxpr.params_vars_flat))
    )

  @property
  def indices_to_block_map(
      self
  ) -> Mapping[Tuple[int, ...], curvature_blocks.CurvatureBlock]:
    """A mapping of parameter indices to their associated blocks."""
    return dict(zip(self.jaxpr.layer_indices, self.blocks))

  @property
  def params_block_index(self) -> utils.Params:
    """A structure, which shows each parameter to which block it corresponds.

    Returns:
      A parameter-like structure, where each parameter is replaced by an integer
      index. This index specifies the block (found by ``self.blocks[index]``)
      which approximates the part of the curvature matrix associated with the
      parameter.
    """
    params_block_index: list[Optional[int]] = [None] * self.num_params_variables

    for i, block_indices in enumerate(self.jaxpr.layer_indices):
      for index in block_indices:
        params_block_index[index] = i

    assert all(x is not None for x in params_block_index)

    return jax.tree_util.tree_unflatten(
        self.jaxpr.params_tree, params_block_index)

  @property
  def num_params_variables(self) -> int:
    """The number of separate parameter variables of the model."""
    return len(self.jaxpr.params_vars_flat)

  @utils.auto_scope_method
  def _compute_losses_vjp(self, func_args: utils.FuncArgs):
    """Computes all model statistics needed for estimating the curvature."""
    return self._vjp(func_args)

  def params_vector_to_blocks_vectors(
      self,
      parameter_structured_vector: utils.Params,
  ) -> Tuple[Tuple[Array, ...]]:
    """Splits the parameters to values for each corresponding block."""

    params_values_flat = jax.tree_util.tree_leaves(parameter_structured_vector)
    blocks_vectors: list[Tuple[Array, ...]] = []

    for indices in self.jaxpr.layer_indices:
      blocks_vectors.append(tuple(params_values_flat[i] for i in indices))

    return tuple(blocks_vectors)

  def blocks_vectors_to_params_vector(
      self,
      blocks_vectors: Sequence[Sequence[Array]],
  ) -> utils.Params:
    """Reverses the effect of ``self.vectors_to_blocks``."""

    if len(blocks_vectors) != self.num_blocks:
      raise ValueError("Incorrect number of block vectors. Expected "
                       f"{self.num_blocks}, but got {len(blocks_vectors)}.")

    values_flat: list[Optional[Array]] = [None] * self.num_params_variables

    for idx, (indices, vectors) in enumerate(
        zip(self.jaxpr.layer_indices, blocks_vectors)):

      if len(indices) != len(vectors):
        raise ValueError(f"Expected len(block_vectors[{idx}])=={len(indices)}, "
                         f"not {len(vectors)}.")

      for i, v in zip(indices, vectors):
        assert values_flat[i] is None
        values_flat[i] = v

    assert not any(v is None for v in values_flat)

    return jax.tree_util.tree_unflatten(self.jaxpr.params_tree, values_flat)

  def _finalize(self, func_args: utils.FuncArgs):
    self._jaxpr = self._vjp(func_args, return_only_jaxpr=True)  # pytype: disable=annotation-type-mismatch  # always-use-return-annotations
    self._create_blocks()

  @utils.auto_scope_method
  def init(
      self,
      rng: PRNGKey,
      func_args: utils.FuncArgs,
      exact_powers_to_cache: Optional[curvature_blocks.ScalarOrSequence],
      approx_powers_to_cache: Optional[curvature_blocks.ScalarOrSequence],
      cache_eigenvalues: bool = False,
  ) -> "BlockDiagonalCurvature.State":
    if not self.finalized:
      self.finalize(func_args)

    blocks_init = []
    blocks_rng = jax.random.split(rng, self.num_blocks)

    for block, block_rng in zip(self.blocks, blocks_rng):

      block_init = block.init(
          rng=block_rng,
          exact_powers_to_cache=exact_powers_to_cache,
          approx_powers_to_cache=approx_powers_to_cache,
          cache_eigenvalues=cache_eigenvalues)

      blocks_init.append(block_init)

    return BlockDiagonalCurvature.State(
        synced=jnp.asarray(True),
        blocks_states=tuple(blocks_init),
    )

  def _sync_state(
      self,
      state: "BlockDiagonalCurvature.State",
      pmap_axis_name: Optional[str],
  ) -> "BlockDiagonalCurvature.State":
    block_states = []
    for block, block_state in zip(self.blocks, state.blocks_states):
      block_states.append(block.sync(block_state.copy(), pmap_axis_name))
    return BlockDiagonalCurvature.State(
        synced=jnp.asarray(True),
        blocks_states=tuple(block_states),
    )

  @utils.auto_scope_method
  def sync(
      self,
      state: "BlockDiagonalCurvature.State",
      pmap_axis_name: Optional[str],
  ) -> "BlockDiagonalCurvature.State":
    return jax.lax.cond(
        state.synced,
        lambda s: s,
        functools.partial(self._sync_state, pmap_axis_name=pmap_axis_name),
        state,
    )

  @utils.auto_scope_method
  def multiply_matpower(
      self,
      state: "BlockDiagonalCurvature.State",
      parameter_structured_vector: utils.Params,
      identity_weight: Union[Numeric, Sequence[Numeric]],
      power: Scalar,
      exact_power: bool,
      use_cached: bool,
      pmap_axis_name: Optional[str],
  ) -> utils.Params:

    blocks_vectors = self.params_vector_to_blocks_vectors(
        parameter_structured_vector)

    identity_weight = utils.to_tuple_or_repeat(identity_weight, self.num_blocks)

    thunks = []
    for block, block_state, block_vector, block_identity_weight in zip(
        self.blocks, state.blocks_states, blocks_vectors, identity_weight):

      thunks.append(
          functools.partial(
              block.multiply_matpower,
              state=block_state,
              vector=block_vector,
              identity_weight=block_identity_weight,
              power=power,
              exact_power=exact_power,
              use_cached=use_cached,
              )
          )

    if self._distributed_multiplies and pmap_axis_name is not None:

      result = utils.distribute_thunks(thunks, pmap_axis_name)

    else:
      result = tuple(thunk() for thunk in thunks)

    parameter_structured_result = self.blocks_vectors_to_params_vector(result)

    assert utils.abstract_objects_equal(
        parameter_structured_vector, parameter_structured_result)

    return parameter_structured_result

  @utils.auto_scope_method
  def block_eigenvalues(
      self,
      state: "BlockDiagonalCurvature.State",
      use_cached: bool,
  ) -> Tuple[Array, ...]:
    """Computes the eigenvalues for each block of the curvature estimator.

    Args:
      state: The state of the estimator.
      use_cached: Whether to use a cached versions of the eigenvalues or to use
        the most recent curvature estimates to compute them. The cached version
        are going to be *at least* as fresh as the last time you called
        :func:`~CurvatureEstimator.update_cache` with ``eigenvalues=True``.

    Returns:
      A tuple of arrays containing the eigenvalues for each block. The
      order of this tuple corresponds to the ordering of ``self.blocks``.
      To understand which parameters correspond to which block you can call
      ``self.parameters_block_index``.
    """
    return tuple(block.eigenvalues(b_state, use_cached=use_cached)
                 for block, b_state in zip(self.blocks, state.blocks_states))

  @utils.auto_scope_method
  def eigenvalues(
      self,
      state: "BlockDiagonalCurvature.State",
      use_cached: bool,
  ) -> Array:

    blocks_eigenvalues = self.block_eigenvalues(state, use_cached)
    return jnp.concatenate(blocks_eigenvalues, axis=0)

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: "BlockDiagonalCurvature.State",
      ema_old: Numeric,
      ema_new: Numeric,
      batch_size: Numeric,
      rng: PRNGKey,
      func_args: utils.FuncArgs,
      estimation_mode: Optional[str] = None,
  ) -> "BlockDiagonalCurvature.State":

    if not self.finalized:
      self.finalize(func_args)

    estimation_mode = estimation_mode or self.default_estimation_mode

    # Compute the losses and the VJP function from the function inputs
    losses, losses_vjp = self._compute_losses_vjp(func_args)

    if "fisher" in estimation_mode:
      if any(not isinstance(l, loss_functions.NegativeLogProbLoss)
             for l in losses):
        raise ValueError(
            f"One of the losses in the function is not an instance of "
            f"`loss_functions.NegativeLogProbLoss`, which is incompatible "
            f"with the estimation mode provided - {estimation_mode}.")

    # Helper function that updates the blocks given a vjp vector
    def update_blocks(vjp_vec_, state_, ema_old_, ema_new_):

      blocks_info_ = losses_vjp(vjp_vec_)
      assert len(blocks_info_) == self.num_blocks

      new_state = []
      for block_, block_state_, block_info_ in zip(
          self.blocks, state_.blocks_states, blocks_info_):

        new_state.append(block_.update_curvature_matrix_estimate(
            block_state_, block_info_, ema_old_, ema_new_,
            batch_size))

      return BlockDiagonalCurvature.State(
          synced=jnp.asarray(False),
          blocks_states=tuple(new_state),
      )

    if estimation_mode == "fisher_gradients":

      keys = jax.random.split(rng, len(losses)) if len(losses) > 1 else [rng]
      vjp_vec = tuple(
          loss.grad_of_evaluate_on_sample(key, coefficient_mode="sqrt")
          for loss, key in zip(losses, keys))

      return update_blocks(vjp_vec, state, ema_old, ema_new)

    elif estimation_mode == "fisher_empirical":

      vjp_vec = tuple(
          loss.grad_of_evaluate(None, coefficient_mode="regular")
          for loss in losses)

      return update_blocks(vjp_vec, state, ema_old, ema_new)

    elif estimation_mode in ("fisher_curvature_prop", "ggn_curvature_prop"):

      keys = jax.random.split(rng, len(losses)) if len(losses) > 1 else [rng]
      vjp_vec = []

      for loss, key in zip(losses, keys):

        if estimation_mode == "fisher_curvature_prop":
          shape = loss.fisher_factor_inner_shape
          random_b = jax.random.bernoulli(key, shape=shape)
          vjp_vec.append(loss.multiply_fisher_factor(random_b * 2.0 - 1.0))

        else:
          shape = loss.ggn_factor_inner_shape
          random_b = jax.random.bernoulli(key, shape=shape)
          vjp_vec.append(loss.multiply_ggn_factor(random_b * 2.0 - 1.0))

      return update_blocks(tuple(vjp_vec), state, ema_old, ema_new)

    elif estimation_mode in ("fisher_exact", "ggn_exact"):
      # We use the following trick to simulate summation. The equation is:
      #   estimate = ema_old * estimate + ema_new * (sum_i estimate_index_i^2)
      #   weight = ema_old * weight + ema_new
      # Instead we update the estimate n times with the following updates:
      #   for k = 1
      #     estimate_k = ema_old * estimate + (ema_new/n) * n*estimate_index_k^2
      #     weight_k = ema_old * weight + (ema_new/n)
      #   for k > 1:
      #     estimate_k = 1.0 * estimate_k-1 + (ema_new/n) * n*estimate_index_k^2
      #     weight_k = 1.0 * weight_k-1 + (ema_new/n)
      # Which is mathematically equivalent to the original version.

      zero_tangents = jax.tree_util.tree_map(
          jnp.zeros_like, list(loss.parameter_dependants for loss in losses))

      if estimation_mode == "fisher_exact":
        shapes = [l.fisher_factor_inner_shape[1:] for l in losses]
      else:
        shapes = [l.ggn_factor_inner_shape[1:] for l in losses]

      total_num_indices = sum(sum(s) for s in shapes)
      ema_new = ema_new / total_num_indices

      # For now we support only inner shapes of 1 dimension, hence below the
      # (loss_num_indices,).
      assert all(len(s) == 1 for s in shapes)

      for i, (loss, (loss_num_indices,)) in enumerate(zip(losses, shapes)):
        for index in range(loss_num_indices):

          vjp_vec = zero_tangents.copy()

          if estimation_mode == "fisher_exact":
            vjp_vec[i] = loss.multiply_fisher_factor_replicated_one_hot([index])
          else:
            vjp_vec[i] = loss.multiply_ggn_factor_replicated_one_hot([index])

          if isinstance(vjp_vec[i], Array):
            # In the special case of only one parameter, it still needs to be a
            # tuple for the tangents.
            vjp_vec[i] = (vjp_vec[i],)

          vjp_vec[i] = jax.tree_util.tree_map(
              lambda x: x * jnp.sqrt(total_num_indices), vjp_vec[i])

          state = update_blocks(tuple(vjp_vec), state, ema_old, ema_new)

          ema_old = 1.0

      return state

    else:
      raise ValueError(f"Unrecognised estimation_mode {estimation_mode}.")

  @utils.auto_scope_method
  def update_cache(
      self,
      state: "BlockDiagonalCurvature.State",
      identity_weight: Union[Numeric, Sequence[Numeric]],
      exact_powers: Optional[curvature_blocks.ScalarOrSequence],
      approx_powers: Optional[curvature_blocks.ScalarOrSequence],
      eigenvalues: bool,
      pmap_axis_name: Optional[str],
  ) -> "BlockDiagonalCurvature.State":
    identity_weight = utils.to_tuple_or_repeat(identity_weight, self.num_blocks)

    thunks = []
    for block, block_state, block_identity_weight in zip(self.blocks,
                                                         state.blocks_states,
                                                         identity_weight):
      thunks.append(
          functools.partial(
              block.update_cache,
              state=block_state,
              identity_weight=block_identity_weight,
              exact_powers=exact_powers,
              approx_powers=approx_powers,
              eigenvalues=eigenvalues,
              )
          )

    if self._distributed_cache_updates and pmap_axis_name is not None:

      assert utils.in_pmap(pmap_axis_name)

      def filter_outputs(thunk, vals):

        # We must precompute the matches outside of the thunk itself, as the
        # thunk will be traced separately from the current compiled context
        # (since it's called within a lax.switch statement).
        matches = jax.tree_util.tree_map(lambda o, v: o is v, thunk(), vals)

        def new_thunk():
          return jax.tree_util.tree_map(
              lambda o, m: None if m else o, thunk(), matches
          )
        return new_thunk

      # Create new thunks that only return the state arrays that they actually
      # modify. This should reduce the communication costs associated with the
      # syncs performed by utils.distribute_thunks.
      filtered_thunks = tuple(
          filter_outputs(thunk, block_state)
          for thunk, block_state in zip(thunks, state.blocks_states))

      new_states = utils.distribute_thunks(filtered_thunks, pmap_axis_name)

      # Restore all of the unmodified state arrays.
      new_states = jax.tree_util.tree_map(lambda s, n: s if n is None else n,
                                          state.blocks_states, new_states)

    else:
      new_states = tuple(thunk() for thunk in thunks)

    return BlockDiagonalCurvature.State(
        synced=state.synced,
        blocks_states=new_states,
    )

  @utils.auto_scope_method
  def to_diagonal_block_dense_matrix(
      self,
      state: "BlockDiagonalCurvature.State",
  ) -> Tuple[Array, ...]:
    """Returns a tuple of arrays with explicit dense matrices of each block."""
    return tuple(block.to_dense_matrix(block_state) for block, block_state in
                 zip(self.blocks, state.blocks_states))

  @utils.auto_scope_method
  def to_dense_matrix(
      self,
      state: "BlockDiagonalCurvature.State"
  ) -> Array:
    return scipy.linalg.block_diag(*self.to_diagonal_block_dense_matrix(state))


class ExplicitExactCurvature(BlockDiagonalCurvature):
  """Explicit exact full curvature estimator class.

  This class estimates the full curvature matrix by looping over the batch
  dimension of the input data and for each single example computes an estimate
  of the curvature matrix and then averages over all examples in the input data.
  This implies that the computation scales linearly (without parallelism) with
  the batch size. The class stores the estimated curvature as a dense matrix,
  hence its memory requirement is (number of parameters)^2. If
  ``estimation_mode`` is ``fisher_exact`` or ``ggn_exact`` than this would
  compute the exact curvature, but other modes are also supported. As a result
  of looping over the input data this class needs to know the index of the batch
  in the arguments to the model function and additionally, since the loop is
  achieved through indexing, each array leaf of that argument must have the same
  first dimension size, which will be interpreted as the batch size.
  """

  def __init__(
      self,
      func: utils.Func,
      params_index: int = 0,
      batch_index: int = 1,
      default_estimation_mode: str = "fisher_exact",
      layer_tag_to_block_ctor:
      Optional[Mapping[str, CurvatureBlockCtor]] = None,
      index_to_block_ctor:
      Optional[Mapping[Tuple[int, ...], CurvatureBlockCtor]] = None,
      auto_register_tags: bool = True,
      **auto_register_kwargs
  ):
    """Initializes the curvature instance.

    Args:
      func: The model function, which should have at least one registered loss.
      params_index: The index of the parameters argument in arguments list of
        ``func``.
      batch_index: Specifies at which index of the inputs to ``func`` is the
        batch, representing data over which we average the curvature.
      default_estimation_mode: The estimation mode which to use by default when
        calling ``self.update_curvature_matrix_estimate``.
      layer_tag_to_block_ctor: An optional dict mapping tags to specific classes
        of block approximations, which to override the default ones.
      index_to_block_ctor: An optional dict mapping a specific block parameter
        indices to specific classes of block approximation, which to override
        the default ones. To get the correct indices check
        ``estimator.indices_to_block_map``.
      auto_register_tags: Whether to automatically register layer tags for
        parameters that have not been manually registered. For further details
        see :func:``~auto_register_tags``.
      **auto_register_kwargs: Any keyword arguments to pass to into the auto
        registration function.
    """
    super().__init__(
        func=func,
        default_estimation_mode=default_estimation_mode,
        params_index=params_index,
        layer_tag_to_block_ctor=layer_tag_to_block_ctor,
        index_to_block_ctor=index_to_block_ctor,
        auto_register_tags=auto_register_tags,
        **auto_register_kwargs
    )
    self._batch_index = batch_index

  @property
  def batch_index(self) -> int:
    """The index in the inputs of the model function, which is the batch."""
    return self._batch_index

  def _create_blocks(self):
    # Here in order to be able to have a block together for all parameters, we
    # create a non-existing (in the original graph) generic layer tag equation.
    assert self._jaxpr is not None

    jax_version = (
        jax.__version_info__ if hasattr(jax, "__version_info__")
        else tuple(map(int, jax.__version__.split("."))))

    if jax_version > (0, 3, 4):
      self._blocks = (curvature_blocks.NaiveFull(
          layer_tag_eq=tags.LayerTagEqn(
              primitive=tags.generic,
              invars=list(self._jaxpr.params_vars_flat),
              outvars=list(self._jaxpr.params_vars_flat),
              params={},
              effects=jax.core.no_effects,
              source_info=jax.core.source_info_util.new_source_info()
          ),
          name="ExactCurvature"
      ),)

    else:
      self._blocks = (curvature_blocks.NaiveFull(
          layer_tag_eq=tags.LayerTagEqn(
              primitive=tags.generic,
              invars=list(self._jaxpr.params_vars_flat),
              outvars=list(self._jaxpr.params_vars_flat),
              params={},
              source_info=jax.core.source_info_util.new_source_info()  # pytype: disable=missing-parameter
          ),
          name="ExactCurvature"
      ),)

  def _compute_losses_vjp(self, func_args):

    # For some reason pytype can't detect that this attribute exists from the
    # super class.
    losses, losses_vjp = self._vjp(func_args)  # pytype: disable=attribute-error

    def modified_losses_jvp(vjp_vec):

      blocks_info = losses_vjp(vjp_vec)

      tangents = [block["params_tangent"] for block in blocks_info]
      tangents = jax.tree_util.tree_leaves(tangents)

      # Need to reorder all of the block information to follow the canonical
      # order of variables
      params_vars = BlockDiagonalCurvature.params_vector_to_blocks_vectors(
          self, self.jaxpr.params_vars)  # pytype: disable=wrong-arg-types
      order = np.argsort([p.count
                          for p in jax.tree_util.tree_leaves(params_vars)])

      return [dict(params_tangent=tuple(tangents[i] for i in order))]

    return losses, modified_losses_jvp

  def params_vector_to_blocks_vectors(
      self,
      parameter_structured_vector: utils.Params,
  ) -> Tuple[Tuple[Array, ...]]:

    return (tuple(jax.tree_util.tree_leaves(parameter_structured_vector)),)

  def blocks_vectors_to_params_vector(
      self,
      blocks_vectors: Sequence[Sequence[Array]],
  ) -> utils.Params:

    assert len(blocks_vectors) == self.num_blocks

    return jax.tree_util.tree_unflatten(
        self.jaxpr.params_tree, blocks_vectors[0])

  def update_curvature_matrix_estimate(
      self,
      state: BlockDiagonalCurvature.State,
      ema_old: Numeric,
      ema_new: Numeric,
      batch_size: Numeric,
      rng: PRNGKey,
      func_args: utils.FuncArgs,
      estimation_mode: Optional[str] = None,
  ) -> curvature_blocks.Full.State:

    rng = jax.random.split(rng, batch_size)

    def single_state_update(
        index: Numeric,
        state_: curvature_blocks.Full.State
    ) -> curvature_blocks.Full.State:

      is_first = index == 0
      args = list(func_args)

      # Index the batch for the `index` arguments.
      args[self._batch_index] = jax.tree_util.tree_map(
          lambda x: x[index][None], args[self._batch_index])

      return BlockDiagonalCurvature.update_curvature_matrix_estimate(
          self,
          state=state_,
          ema_old=is_first * ema_old + (1 - is_first) * 1.0,
          ema_new=ema_new / batch_size,
          batch_size=1,
          rng=rng[index],
          func_args=args,
          estimation_mode=estimation_mode,
      )

    return jax.lax.fori_loop(0, batch_size, single_state_update, state)

  def update_cache(
      self,
      state: BlockDiagonalCurvature.State,
      identity_weight: Numeric,
      exact_powers: Optional[curvature_blocks.ScalarOrSequence],
      approx_powers: Optional[curvature_blocks.ScalarOrSequence],
      eigenvalues: bool,
      pmap_axis_name: Optional[str],
  ) -> curvature_blocks.Full.State:

    block_state = self.blocks[0].update_cache(
        state=state.blocks_states[0],
        identity_weight=identity_weight,
        exact_powers=exact_powers,
        approx_powers=approx_powers,
        eigenvalues=eigenvalues,
    )

    return BlockDiagonalCurvature.State(blocks_states=(block_state,))
