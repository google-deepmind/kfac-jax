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
"""Module containing the abstract class for curvature estimators."""
import abc
from typing import Callable, Sequence, Generic, TypeVar
import jax.numpy as jnp
from kfac_jax._src import curvature_blocks
from kfac_jax._src import layers_and_loss_tags as tags
from kfac_jax._src import loss_functions
from kfac_jax._src import tracer
from kfac_jax._src import utils

# Types for annotation
Array = utils.Array
PRNGKey = utils.PRNGKey
Numeric = utils.Numeric
Scalar = utils.Scalar
Shape = utils.Shape
LossFunction = loss_functions.LossFunction
LossFunctionsTuple = tuple[loss_functions.LossFunction, ...]
LossFunctionsSequence = Sequence[loss_functions.LossFunction]
LossFunctionInputs = loss_functions.LossFunctionInputs
LossFunctionInputsSequence = Sequence[loss_functions.LossFunctionInputs]
LossFunctionInputsTuple = tuple[loss_functions.LossFunctionInputs, ...]
CurvatureBlockCtor = Callable[
    [tags.LayerTagEqn],
    curvature_blocks.CurvatureBlock
]
StateType = TypeVar("StateType")


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
  corresponding expression using the current curvature estimate or using a
  cached version.

  Attributes:
    func: The model evaluation function.
    params_index: The index of the parameters argument in arguments list of
      ``func``.
    batch_index: The index of the batch data argument in arguments list of
      ``func``.
    default_estimation_mode: The estimation mode which to use by default when
      calling :func:`~CurvatureEstimator.update_curvature_matrix_estimate`.
  """

  def __init__(
      self,
      func: utils.Func,
      params_index: int = 0,
      batch_index: int = 1,
      default_estimation_mode: str = "ggn_curvature_prop",
  ):
    """Initializes the CurvatureEstimator instance.

    Args:
      func: The model function, which should have at least one registered loss.
      params_index: The index of the parameters argument in arguments list of
        ``func``.
      batch_index: The index of the batch data argument in arguments list of
        ``func``.
      default_estimation_mode: The estimation mode which to use by default when
        calling :func:`~CurvatureEstimator.update_curvature_matrix_estimate`.
    """

    if default_estimation_mode not in self.valid_estimation_modes:
      raise ValueError(
          f"Unsupported estimation mode: {default_estimation_mode}. This class "
          f"currently only supports ones in {self.valid_estimation_modes}.")

    super().__init__()

    self.func = func
    self.params_index = params_index
    self.batch_index = batch_index
    self.default_estimation_mode = default_estimation_mode
    self.compute_losses, _ = tracer.compute_all_losses(
        func=func, params_index=params_index
    )

  @property
  def default_mat_type(self) -> str:
    """The type of matrix that this estimator is approximating."""
    idx = self.default_estimation_mode.index("_")
    return self.default_estimation_mode[:idx]

  @property
  @abc.abstractmethod
  def valid_estimation_modes(self) -> tuple[str, ...]:
    """The valid estimation modes for this estimator."""

  @property
  @abc.abstractmethod
  def dim(self) -> int:
    """The number of elements of all parameter variables together."""

  @abc.abstractmethod
  def init(
      self,
      rng: PRNGKey,
      func_args: utils.FuncArgs,
      exact_powers_to_cache: curvature_blocks.ScalarOrSequence | None,
      approx_powers_to_cache: curvature_blocks.ScalarOrSequence | None,
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
      pmap_axis_name: str | None,
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
      pmap_axis_name: str | None,
      norm_to_scale_identity_weight_per_block: str | None = None,
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
      norm_to_scale_identity_weight_per_block: The name of a norm to use to
          compute extra per-block scaling for identity_weight. See
          psd_matrix_norm() in utils/math.py for the definition of these.

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
      pmap_axis_name: str | None,
      norm_to_scale_identity_weight_per_block: str | None = None,
  ) -> utils.Params:
    """Computes ``(CurvatureMatrix + identity_weight I)`` times ``vector``."""

    return self.multiply_matpower(
        state=state,
        parameter_structured_vector=parameter_structured_vector,
        identity_weight=identity_weight,
        power=1,
        exact_power=exact_power,
        use_cached=use_cached,
        pmap_axis_name=pmap_axis_name,
        norm_to_scale_identity_weight_per_block=norm_to_scale_identity_weight_per_block,
    )

  def multiply_inverse(
      self,
      state: StateType,
      parameter_structured_vector: utils.Params,
      identity_weight: Numeric,
      exact_power: bool,
      use_cached: bool,
      pmap_axis_name: str | None,
      norm_to_scale_identity_weight_per_block: str | None = None,
  ) -> utils.Params:
    """Computes ``(CurvatureMatrix + identity_weight I)^-1`` times ``vector``."""

    return self.multiply_matpower(
        state=state,
        parameter_structured_vector=parameter_structured_vector,
        identity_weight=identity_weight,
        power=-1,
        exact_power=exact_power,
        use_cached=use_cached,
        pmap_axis_name=pmap_axis_name,
        norm_to_scale_identity_weight_per_block=norm_to_scale_identity_weight_per_block,
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
      identity_weight: Numeric,
      batch_size: Numeric,
      rng: PRNGKey,
      func_args: utils.FuncArgs,
      estimation_mode: str | None = None,
  ) -> StateType:
    """Updates the estimator's curvature estimates.

    Args:
      state: The state of the estimator to update.
      ema_old: Specifies the weight of the old value when computing the updated
        estimate in the moving average.
      ema_new: Specifies the weight of the new value when computing the updated
        estimate in the moving average.
      identity_weight: The weight of the identity added to the block's curvature
        matrix before computing the cached matrix power.
      batch_size: The batch size.
      rng: A PRNGKey to be used for any potential sampling in the estimation
        process.
      func_args: A structure with the values of the inputs to the traced
        function (the ``tagged_func`` passed into the constructor) which to be
        used for the estimation process. Should have the same structure as the
        argument ``func_args`` passed in the constructor.
      estimation_mode: The type of curvature estimator to use. By default
        (e.g. if ``None``) will use ``self.default_estimation_mode``. Must be
        one of ``self.valid_estimation_modes``.
    Returns:
      The updated state.
    """

  @abc.abstractmethod
  def update_cache(
      self,
      state: StateType,
      identity_weight: Numeric,
      exact_powers: curvature_blocks.ScalarOrSequence | None,
      approx_powers: curvature_blocks.ScalarOrSequence | None,
      eigenvalues: bool,
      pmap_axis_name: str | None,
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

  def compute_func_from_registered(self, func_args, batch_size) -> Array:

    losses = self.compute_losses(func_args)

    loss_values = tuple(
        jnp.sum(loss.evaluate(None, coefficient_mode="regular"))
        for loss in losses)

    return sum(loss_values) / batch_size
