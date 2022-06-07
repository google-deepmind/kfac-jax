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
"""K-FAC curvature approximation to single layer blocks."""
import abc
import collections
import functools
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from kfac_jax._src import layers_and_loss_tags as tags
from kfac_jax._src import patches_second_moment as psm
from kfac_jax._src import tag_graph_matcher as tgm
from kfac_jax._src import utils
import numpy as np

# Types for annotation
ScalarOrSequence = Union[chex.Scalar, Sequence[chex.Scalar]]

# Special global variables
# The default value that would be used for the argument
# ``max_elements_for_vmap``, when it is set to ``None`` in the
# ``Conv2DDiagonal`` and ``Conv2DFull` curvature blocks.
_MAX_PARALLEL_ELEMENTS: int = 2 ** 23
# The default value that would be used for the argument
# ``eigen_decomposition_threshold``, when it is set to ``None`` in any of the
# curvature blocks that inherit from ``Full`.
_DEFAULT_EIGEN_DECOMPOSITION_THRESHOLD = 5


def set_max_parallel_elements(value: int):
  """Sets the default value of maximum parallel elements in the module.

  This value is used to determine the parallel-to-memory tradeoff in the
  curvature estimation procedure of :class:`~Conv2DDiagonal` and
  :class:`~Conv2DFull`. See their corresponding docs for further details.

  Args:
    value: The default value for maximum number of parallel elements.
  """
  global _MAX_PARALLEL_ELEMENTS
  _MAX_PARALLEL_ELEMENTS = value


def get_max_parallel_elements() -> int:
  """Returns the default value of maximum parallel elements in the module.

  This value is used to determine the parallel-to-memory tradeoff in the
  curvature estimation procedure of :class:`~Conv2DDiagonal` and
  :class:`~Conv2DFull`. See their corresponding docs for further details.

  Returns:
    The default value for maximum number of parallel elements.
  """
  global _MAX_PARALLEL_ELEMENTS
  return _MAX_PARALLEL_ELEMENTS


def set_default_eigen_decomposition_threshold(value: int):
  """Sets the default value of the eigen decomposition threshold.

  This value is used in :class:`~Full` to determine when updating the cache,
  at what number of different powers to switch the implementation from a simple
  matrix power to an eigenvector decomposition.

  Args:
    value: The default value for eigen decomposition threshold.
  """
  global _DEFAULT_EIGEN_DECOMPOSITION_THRESHOLD
  _DEFAULT_EIGEN_DECOMPOSITION_THRESHOLD = value


def get_default_eigen_decomposition_threshold() -> int:
  """Returns the default value of the eigen decomposition threshold.

  This value is used in :class:`~Full` to determine when updating the cache,
  at what number of different powers to switch the implementation from a simple
  matrix power to an eigenvector decomposition.

  Returns:
    The default value of the eigen decomposition threshold.
  """
  global _DEFAULT_EIGEN_DECOMPOSITION_THRESHOLD
  return _DEFAULT_EIGEN_DECOMPOSITION_THRESHOLD


def _to_real_set(
    number_or_sequence: Optional[ScalarOrSequence]
) -> Set[chex.Scalar]:
  """Converts the optional number or sequence to a set."""
  if number_or_sequence is None:
    return set()
  elif isinstance(number_or_sequence, (float, int)):
    return {number_or_sequence}  # pytype: disable=bad-return-type
  elif (isinstance(number_or_sequence, collections.abc.Sequence) and
        all(isinstance(x, (int, float)) for x in number_or_sequence)):
    return set(number_or_sequence)
  else:
    raise ValueError(f"Expecting a real-number or a sequence of reals, but got "
                     f"{type(number_or_sequence)}.")


class CurvatureBlock(utils.Finalizable):
  """Abstract class for curvature approximation blocks.

  A CurvatureBlock defines a curvature matrix to be estimated, and gives methods
  to multiply powers of this with a vector. Powers can be computed exactly or
  with a class-determined approximation. Cached versions of the powers can be
  pre-computed to make repeated multiplications cheaper. During initialization
  you would have to explicitly specify all powers that you will need to cache.
  """

  @utils.pytree_dataclass
  class State:
    """Persistent state of the block.

    Any subclasses of :class:`~CurvatureBlock` should also internally extend
    this class, with any attributes needed for the curvature estimation.

    Attributes:
      cache: A dictionary, containing any state data that is updated on
        irregular intervals, such as inverses, eigenvalues, etc. Elements of
        this are updated via calls to :func:`~CurvatureBlock.update_cache`, and
        do not necessarily correspond to the the most up to date curvature
        estimate.
    """
    cache: Optional[Dict[str, Union[chex.Array, Dict[str, chex.Array]]]]

  def __init__(self, layer_tag_eq: tags.LayerTagEqn, name: str):
    """Initializes the block.

    Args:
      layer_tag_eq: The Jax equation corresponding to the layer tag that this
        block will approximate the curvature to.
      name: The name of this block.
    """
    super().__init__()

    self._layer_tag_eq = layer_tag_eq
    self._name = name

    self.finalize()

  @property
  def layer_tag_primitive(self) -> tags.LayerTag:
    """The :class:`jax.core.Primitive` corresponding to the block's tag equation."""
    primitive = self._layer_tag_eq.primitive
    assert isinstance(primitive, tgm.tags.LayerTag)
    return primitive

  @property
  def parameter_variables(self) -> Tuple[jax.core.Var, ...]:
    """The parameter variables of the underlying Jax equation."""
    param_vars = []
    for p in self.layer_tag_primitive.split_all_inputs(
        self._layer_tag_eq.invars)[2]:
      assert isinstance(p, jax.core.Var)
      param_vars.append(p)
    return tuple(param_vars)

  @property
  def outputs_shapes(self) -> Tuple[chex.Shape, ...]:
    """The shapes of the output variables of the block's tag equation."""
    output_vars = self.layer_tag_primitive.split_all_inputs(
        self._layer_tag_eq.invars)[0]
    return jax.tree_map(lambda x: x.aval.shape, output_vars)

  @property
  def inputs_shapes(self) -> Tuple[chex.Shape, ...]:
    """The shapes of the input variables of the block's tag equation."""
    input_vars = self.layer_tag_primitive.split_all_inputs(
        self._layer_tag_eq.invars)[1]
    return jax.tree_map(lambda x: x.aval.shape, input_vars)

  @property
  def parameters_shapes(self) -> Tuple[chex.Shape, ...]:
    """The shapes of the parameter variables of the block's tag equation."""
    return tuple(jax.tree_map(lambda x: tuple(x.aval.shape),
                              self.parameter_variables))

  @property
  def parameters_canonical_order(self) -> Tuple[int, ...]:
    """The canonical order of the parameter variables."""
    return tuple(np.argsort([p.count for p in self.parameter_variables]))

  @property
  def layer_tag_extra_params(self) -> Dict[str, Any]:
    """Any extra parameters of passed into the Jax primitive of this block."""
    return self._layer_tag_eq.params

  @property
  def number_of_parameters(self) -> int:
    """Number of parameter variables of this block."""
    return len(self.parameters_shapes)

  @property
  def dim(self) -> int:
    """The number of elements of all parameter variables together."""
    return sum(utils.product(shape) for shape in self.parameters_shapes)

  @property
  def scale(self) -> chex.Numeric:
    """Any additional scaling factor, not present in the Jax equation."""
    return 1.0

  def __str__(self):
    return f"{self._name!r}[{self.parameters_shapes!r}]"

  def init(
      self,
      rng: chex.PRNGKey,
      exact_powers_to_cache: Optional[ScalarOrSequence],
      approx_powers_to_cache: Optional[ScalarOrSequence],
      cache_eigenvalues: bool,
  ) -> "CurvatureBlock.State":
    """Initializes the state for this block.

    Args:
      rng: The PRNGKey which to be used for any randomness of the initialization
      exact_powers_to_cache: A single value, or multiple values in a list, which
          specify which exact matrix powers the block should be caching. Matrix
          powers, which are expected to be used in
          :func:`~CurvatureBlock.multiply_matrix_power`,
          :func:`~CurvatureBlock.multiply_inverse` or
          :func:`~CurvatureBlock.multiply`  with ``exact_power=True`` and
          ``use_cached=True`` must be provided here.
      approx_powers_to_cache: A single value, or multiple values in a list,
          which specify approximate matrix powers the block should be caching.
          Matrix powers, which are expected to be used in
          :func:`~CurvatureBlock.multiply_matrix_power`,
          :func:`~CurvatureBlock.multiply_inverse` or
          :func:`~CurvatureBlock.multiply`  with ``exact_power=False`` and
          ``use_cached=True`` must be provided here.
      cache_eigenvalues: Specifies whether the block should be caching the
          eigenvalues of its approximate curvature.
    Returns:
      A dictionary with the initialized state.
    """
    return self._init(
        rng=rng,
        exact_powers_to_cache=_to_real_set(exact_powers_to_cache),
        approx_powers_to_cache=_to_real_set(approx_powers_to_cache),
        cache_eigenvalues=cache_eigenvalues)

  @abc.abstractmethod
  def _init(
      self,
      rng: chex.PRNGKey,
      exact_powers_to_cache: Set[chex.Scalar],
      approx_powers_to_cache: Set[chex.Scalar],
      cache_eigenvalues: bool,
  ) -> "CurvatureBlock.State":
    """The non-public interface of ``init``."""

  def multiply_matpower(
      self,
      state: "CurvatureBlock.State",
      vector: Sequence[chex.Array],
      identity_weight: chex.Numeric,
      power: chex.Scalar,
      exact_power: bool,
      use_cached: bool,
  ) -> Tuple[chex.Array, ...]:
    """Computes ``(BlockMatrix + identity_weight I)**power`` times ``vector``.

    Args:
      state: The state for this block.
      vector: A tuple of arrays that should have the same shapes as the block's
        parameters_shapes, which represent the vector you want to multiply.
      identity_weight: A scalar specifying the weight on the identity matrix
        that is added to the block matrix before raising it to a power. If
        ``use_cached=False`` it is guaranteed that this argument will be used in
        the computation. When returning cached values, this argument *may* be
        ignored in favor whatever value was last passed to
        :func:`~CurvatureBlock.update_cached_estimate`. The precise semantics
        of this depend on the concrete subclass and its particular behavior with
        regards to caching.
      power: The power to which to raise the matrix.
      exact_power: Specifies whether to compute the exact matrix power of
        ``BlockMatrix + identity_weight I``. When this argument is ``False``
        the exact behaviour will depend on the concrete subclass and the
        result will *in general* be an approximation to
        ``(BlockMatrix + identity_weight I)^power``, although some subclasses
        may still compute the exact matrix power.
      use_cached: Whether to use a cached version for computing the product or
        to use the most recent curvature estimates. The cached version is
        going to be *at least* as fresh as the value provided to the last call
        to :func:`~CurvatureBlock.update_cached_estimate` with the same value of
        `power`.
    Returns:
      A tuple of arrays, representing the result of the matrix-vector product.
    """
    result = self._multiply_matpower_unscaled(
        state=state,
        vector=vector,
        identity_weight=(identity_weight if self.scale == 1.0 else
                         identity_weight / self.scale),
        power=power,
        exact_power=exact_power,
        use_cached=use_cached,
    )
    if self.scale != 1.0:
      return utils.scalar_mul(result, jnp.power(self.scale, power))
    return result

  @abc.abstractmethod
  def _multiply_matpower_unscaled(
      self,
      state: "CurvatureBlock.State",
      vector: Sequence[chex.Array],
      identity_weight: chex.Numeric,
      power: chex.Scalar,
      exact_power: bool,
      use_cached: bool,
  ) -> Tuple[chex.Array, ...]:
    """Performs matrix-vector multiplication, ignoring ``self.scale``."""

  def multiply(
      self,
      state: "CurvatureBlock.State",
      vector: Sequence[chex.Array],
      identity_weight: chex.Numeric,
      exact_power: bool,
      use_cached: bool,
  ) -> Tuple[chex.Array, ...]:
    """Computes ``(BlockMatrix + identity_weight I)`` times ``vector``."""
    return self.multiply_matpower(
        state=state,
        vector=vector,
        identity_weight=identity_weight,
        power=1,
        exact_power=exact_power,
        use_cached=use_cached,
    )

  def multiply_inverse(
      self,
      state: "CurvatureBlock.State",
      vector: Sequence[chex.Array],
      identity_weight: chex.Numeric,
      exact_power: bool,
      use_cached: bool,
  ) -> Tuple[chex.Array, ...]:
    """Computes ``(BlockMatrix + identity_weight I)^-1`` times ``vector``."""
    return self.multiply_matpower(
        state=state,
        vector=vector,
        identity_weight=identity_weight,
        power=-1,
        exact_power=exact_power,
        use_cached=use_cached,
    )

  def eigenvalues(
      self,
      state: "CurvatureBlock.State",
      use_cached: bool,
  ) -> chex.Array:
    """Computes the eigenvalues for this block approximation.

    Args:
      state: The state dict for this block.
      use_cached: Whether to use a cached versions of the eigenvalues or to use
        the most recent curvature estimates to compute them. The cached version
        are going to be *at least* as fresh as the last time you called
        :func:`~CurvatureBlock.update_cached_estimate` with ``eigenvalues=True``
        .

    Returns:
      An array containing the eigenvalues of the block.
    """
    eigenvalues = self._eigenvalues_unscaled(state, use_cached) * self.scale
    assert eigenvalues.size == self.dim
    return eigenvalues

  @abc.abstractmethod
  def _eigenvalues_unscaled(
      self,
      state: "CurvatureBlock.State",
      use_cached: bool,
  ) -> chex.Array:
    """Computes the eigenvalues for this block, ignoring `self.scale`."""

  @abc.abstractmethod
  def update_curvature_matrix_estimate(
      self,
      state: "CurvatureBlock.State",
      estimation_data: Mapping[str, Sequence[chex.Array]],
      ema_old: chex.Numeric,
      ema_new: chex.Numeric,
      batch_size: int,
      pmap_axis_name: Optional[str],
  ) -> "CurvatureBlock.State":
    """Updates the block's curvature estimates using the ``info`` provided.

    Each block *in general* estimates a moving average of its associated
    curvature matrix. If you don't want a moving average you can set
    ``ema_old=0`` and ``ema_new=1``.

    Args:
      state: The state dict for this block to update.
      estimation_data: A map containing data used for updating the curvature
          matrix estimate for this block. This can be computed by calling the
          function returned from :func:`~layer_tags_vjp`. Please see its
          implementation for more details on the name of the fields and how they
          are constructed.
      ema_old: Specifies the weight of the old value when computing the updated
          estimate in the moving average.
      ema_new: Specifies the weight of the new value when computing the updated
          estimate in the moving average.
      batch_size: The batch size used in computing the values in ``info``.
      pmap_axis_name: The name of any pmap axis, which might be needed for
          computing the updates.
    """

  def update_cache(
      self,
      state: "CurvatureBlock.State",
      identity_weight: chex.Numeric,
      exact_powers: Optional[ScalarOrSequence],
      approx_powers: Optional[ScalarOrSequence],
      eigenvalues: bool,
      pmap_axis_name: Optional[str],
  ) -> "CurvatureBlock.State":
    """Updates the cached estimates of the different powers specified.

    Args:
      state: The state dict for this block to update.
      identity_weight: The weight of the identity added to the block's curvature
          matrix before computing the cached matrix power.
      exact_powers: Specifies any cached exact matrix powers to be updated.
      approx_powers: Specifies any cached approximate matrix powers to be
          updated.
      eigenvalues: Specifies whether to update the cached eigenvalues
          of the block. If they have not been cached before, this will create an
          entry with them in the block's cache.
      pmap_axis_name: The name of any pmap axis, which wil be used for
          aggregating any computed values over multiple devices.
    Returns:
      None
    """
    return self._update_cache(
        state=state,
        identity_weight=(identity_weight if self.scale == 1.0 else
                         identity_weight / self.scale),
        exact_powers=_to_real_set(exact_powers),
        approx_powers=_to_real_set(approx_powers),
        eigenvalues=eigenvalues,
        pmap_axis_name=pmap_axis_name,
    )

  @abc.abstractmethod
  def _update_cache(
      self,
      state: "CurvatureBlock.State",
      identity_weight: chex.Numeric,
      exact_powers: Set[chex.Scalar],
      approx_powers: Set[chex.Scalar],
      eigenvalues: bool,
      pmap_axis_name: Optional[str],
  ) -> "CurvatureBlock.State":
    """The cache updating function, ignoring ``self.scale``."""

  def to_dense_matrix(self, state: "CurvatureBlock.State") -> chex.Array:
    """Returns a dense representation of the approximate curvature matrix."""
    return self.scale * self._to_dense_unscaled(state)

  @abc.abstractmethod
  def _to_dense_unscaled(self, state: "CurvatureBlock.State") -> chex.Array:
    """A dense representation of the curvature, ignoring ``self.scale``."""


class ScaledIdentity(CurvatureBlock):
  """A block that assumes that the curvature is a scaled identity matrix."""

  def __init__(
      self,
      layer_tag_eq: tags.LayerTagEqn,
      name: str,
      scale: chex.Numeric = 1.0,
  ):
    """Initializes the block.

    Args:
      layer_tag_eq: The Jax equation corresponding to the layer tag, that this
        block will approximate the curvature to.
      name: The name of this block.
      scale: The extra scale of the identity matrix.
    """
    self._scale = scale
    super().__init__(layer_tag_eq, name)

  @property
  def scale(self) -> chex.Numeric:
    return self._scale

  def _init(
      self,
      rng: chex.PRNGKey,
      exact_powers_to_cache: Set[chex.Scalar],
      approx_powers_to_cache: Set[chex.Scalar],
      cache_eigenvalues: bool,
  ) -> CurvatureBlock.State:
    del rng, exact_powers_to_cache, approx_powers_to_cache  # Unused
    return CurvatureBlock.State(
        cache=None,
    )

  def _multiply_matpower_unscaled(
      self,
      state: CurvatureBlock.State,
      vector: Sequence[chex.Array],
      identity_weight: chex.Numeric,
      power: chex.Scalar,
      exact_power: bool,
      use_cached: bool,
  ) -> Tuple[chex.Array, ...]:
    del exact_power, use_cached  # Unused
    identity_weight = identity_weight + 1.0
    if power == 1:
      return jax.tree_map(lambda x: identity_weight * x, vector)
    elif power == -1:
      return jax.tree_map(lambda x: x / identity_weight, vector)
    else:
      identity_weight = jnp.power(identity_weight, power)
      return jax.tree_map(lambda x: identity_weight * x, vector)

  def _eigenvalues_unscaled(
      self,
      state: "CurvatureBlock.State",
      use_cached: bool,
  ) -> chex.Array:
    return jnp.ones([self.dim])

  def update_curvature_matrix_estimate(
      self,
      state: CurvatureBlock.State,
      estimation_data: Mapping[str, Sequence[chex.Array]],
      ema_old: chex.Numeric,
      ema_new: chex.Numeric,
      batch_size: int,
      pmap_axis_name: Optional[str],
  ) -> CurvatureBlock.State:
    return state

  def _update_cache(
      self,
      state: CurvatureBlock.State,
      identity_weight: chex.Numeric,
      exact_powers: Set[chex.Scalar],
      approx_powers: Set[chex.Scalar],
      eigenvalues: bool,
      pmap_axis_name: Optional[str],
  ) -> CurvatureBlock.State:
    return state

  def _to_dense_unscaled(self, state: CurvatureBlock.State) -> chex.Array:
    del state  # not used
    return jnp.eye(self.dim)


class Diagonal(CurvatureBlock, abc.ABC):
  """An abstract class for approximating only the diagonal of curvature."""

  @utils.pytree_dataclass
  class State(CurvatureBlock.State):
    """Persistent state of the block.

    Attributes:
      diagonal_factors: A tuple of the moving averages of the estimated
        diagonals of the curvature for each parameter that is part of the
        associated layer.
    """
    diagonal_factors: Tuple[utils.WeightedMovingAverage]

  def _init(
      self,
      rng: chex.PRNGKey,
      exact_powers_to_cache: Set[chex.Scalar],
      approx_powers_to_cache: Set[chex.Scalar],
      cache_eigenvalues: bool,
  ) -> "Diagonal.State":
    del rng
    return Diagonal.State(
        cache=None,
        diagonal_factors=tuple(utils.WeightedMovingAverage.zero(s)
                               for s in self.parameters_shapes),
    )

  def _multiply_matpower_unscaled(
      self,
      state: "Diagonal.State",
      vector: Sequence[chex.Array],
      identity_weight: chex.Numeric,
      power: chex.Scalar,
      exact_power: bool,
      use_cached: bool,
  ) -> Tuple[chex.Array, ...]:
    factors = tuple(f.value + identity_weight for f in state.diagonal_factors)
    assert len(factors) == len(vector)
    if power == 1:
      return tuple(f * v for f, v in zip(factors, vector))
    elif power == -1:
      return tuple(v / f for f, v in zip(factors, vector))
    else:
      return tuple(jnp.power(f, power) * v for f, v in zip(factors, vector))

  def _eigenvalues_unscaled(
      self,
      state: "Diagonal.State",
      use_cached: bool,
  ) -> chex.Array:
    return jnp.concatenate([f.value.flatten() for f in state.diagonal_factors],
                           axis=0)

  def update_curvature_matrix_estimate(
      self,
      state: "Diagonal.State",
      estimation_data: Mapping[str, Sequence[chex.Array]],
      ema_old: chex.Numeric,
      ema_new: chex.Numeric,
      batch_size: int,
      pmap_axis_name: Optional[str],
  ) -> "Diagonal.State":
    state = self._update_curvature_matrix_estimate(
        state, estimation_data, ema_old, ema_new, batch_size, pmap_axis_name)
    for factor in state.diagonal_factors:
      factor.sync(pmap_axis_name)
    return state

  @abc.abstractmethod
  def _update_curvature_matrix_estimate(
      self,
      state: "Diagonal.State",
      estimation_data: Mapping[str, Sequence[chex.Array]],
      ema_old: chex.Numeric,
      ema_new: chex.Numeric,
      batch_size: int,
      pmap_axis_name: Optional[str],
  ) -> "Diagonal.State":
    pass

  def _update_cache(
      self,
      state: "Diagonal.State",
      identity_weight: chex.Numeric,
      exact_powers: chex.Numeric,
      approx_powers: chex.Numeric,
      eigenvalues: bool,
      pmap_axis_name: Optional[str],
  ) -> "Diagonal.State":
    return state

  def _to_dense_unscaled(self, state: "Diagonal.State") -> chex.Array:
    # Extract factors in canonical order
    factors = [state.diagonal_factors[i].value.flatten()
               for i in self.parameters_canonical_order]
    # Construct diagonal matrix
    return jnp.diag(jnp.concatenate(factors, axis=0))


class Full(CurvatureBlock, abc.ABC):
  """An abstract class for approximating the block matrix with a full matrix."""

  @utils.pytree_dataclass
  class State(CurvatureBlock.State):
    """Persistent state of the block.

    Attributes:
      matrix: A moving average of the estimated curvature matrix for all
        parameters that are part of the associated layer.
    """
    matrix: utils.WeightedMovingAverage

  def __init__(
      self,
      layer_tag_eq: tags.LayerTagEqn,
      name: str,
      eigen_decomposition_threshold: Optional[int] = None,
  ):
    """Initializes the block.

    Args:
      layer_tag_eq: The Jax equation corresponding to the layer tag that this
        block will approximate the curvature to.
      name: The name of this block.
      eigen_decomposition_threshold: During calls to ``init`` and
       ``update_cache`` if higher number of matrix powers than this threshold
       are requested,  instead of computing individual approximate powers, will
       directly compute the eigen-decomposition instead (which provide access to
       any matrix power). If this is ``None`` will use the value returned from
       :func:`~get_default_eigen_decomposition_threshold()`.
    """
    if eigen_decomposition_threshold is None:
      threshold = get_default_eigen_decomposition_threshold()
      self._eigen_decomposition_threshold = threshold
    else:
      self._eigen_decomposition_threshold = eigen_decomposition_threshold
    super().__init__(layer_tag_eq, name)

  def parameters_list_to_single_vector(
      self,
      parameters_shaped_list: Sequence[chex.Array],
  ) -> chex.Array:
    """Converts values corresponding to parameters of the block to vector."""
    if len(parameters_shaped_list) != self.number_of_parameters:
      raise ValueError(f"Expected a list of {self.number_of_parameters} values,"
                       f" but got {len(parameters_shaped_list)} instead.")
    for array, shape in zip(parameters_shaped_list, self.parameters_shapes):
      if array.shape != shape:
        raise ValueError(f"Expected a value of shape {shape}, but got "
                         f"{array.shape} instead.")
    return jnp.concatenate([v.flatten() for v in parameters_shaped_list])

  def single_vector_to_parameters_list(
      self,
      vector: chex.Array,
  ) -> Tuple[chex.Array, ...]:
    """Reverses the transformation ``self.parameters_list_to_single_vector``."""
    if vector.ndim != 1:
      raise ValueError(f"Expecting a vector, got {vector.ndim}-tensor.")
    if vector.size != self.dim:
      raise ValueError(f"Expected a vector of size {self.dim}, but got "
                       f"{vector.size} instead.")
    parameters_shaped_list = []
    index = 0
    for shape in self.parameters_shapes:
      size = utils.product(shape)
      parameters_shaped_list.append(vector[index: index + size].reshape(shape))
      index += size
    assert index == self.dim
    return tuple(parameters_shaped_list)

  def _init(
      self,
      rng: chex.PRNGKey,
      exact_powers_to_cache: Set[chex.Scalar],
      approx_powers_to_cache: Set[chex.Scalar],
      cache_eigenvalues: bool,
  ) -> "Full.State":
    del rng
    # This block does not have any notion of "approximate" powers
    exact_powers_to_cache = exact_powers_to_cache | approx_powers_to_cache
    cache = {}
    if len(exact_powers_to_cache) > self._eigen_decomposition_threshold:
      cache["eigenvalues"] = jnp.zeros([self.dim])
      cache["eigen_vectors"] = jnp.zeros([self.dim, self.dim])
    elif cache_eigenvalues:
      cache["eigenvalues"] = jnp.zeros([self.dim])
    if len(exact_powers_to_cache) <= self._eigen_decomposition_threshold:
      for power in exact_powers_to_cache:
        cache[str(power)] = jnp.zeros([self.dim, self.dim])
    return Full.State(
        cache=cache,
        matrix=utils.WeightedMovingAverage.zero((self.dim, self.dim)),
    )

  def _multiply_matpower_unscaled(
      self,
      state: "Full.State",
      vector: Sequence[chex.Array],
      identity_weight: chex.Numeric,
      power: chex.Scalar,
      exact_power: bool,
      use_cached: bool,
  ) -> Tuple[chex.Array, ...]:
    vector = self.parameters_list_to_single_vector(vector)
    if power == 1:
      result = jnp.matmul(state.matrix.value, vector) + identity_weight * vector
    elif not use_cached:
      matrix = state.matrix.value + identity_weight * jnp.eye(self.dim)
      if power == -1:
        result = jnp.linalg.solve(matrix, vector)
      else:
        result = jnp.matmul(jnp.linalg.matrix_power(matrix, power), vector)
    else:
      if str(power) in state.cache:
        result = jnp.matmul(state.cache[str(power)], vector)
      else:
        s = state.cache["eigenvalues"]
        q = state.cache["eigen_vectors"]
        result = jnp.matmul(jnp.transpose(q), vector)
        result = jnp.power(s + identity_weight, power) * result
        result = jnp.matmul(q, result)
    return self.single_vector_to_parameters_list(result)

  def _eigenvalues_unscaled(
      self,
      state: "Full.State",
      use_cached: bool,
  ) -> chex.Array:
    if not use_cached:
      return utils.safe_psd_eigh(state.matrix.value)[0]
    else:
      return state.cache["eigenvalues"]

  @abc.abstractmethod
  def update_curvature_matrix_estimate(
      self,
      state: "Full.State",
      estimation_data: Mapping[str, Sequence[chex.Array]],
      ema_old: chex.Numeric,
      ema_new: chex.Numeric,
      batch_size: int,
      pmap_axis_name: Optional[str],
  ) -> "Full.State":
    pass

  def _update_cache(
      self,
      state: "Full.State",
      identity_weight: chex.Numeric,
      exact_powers: Set[chex.Scalar],
      approx_powers: Set[chex.Scalar],
      eigenvalues: bool,
      pmap_axis_name: Optional[str],
  ) -> "Full.State":
    # This block does not have any notion of "approximate" powers
    exact_powers = exact_powers | approx_powers
    state.matrix.sync(pmap_axis_name)
    if len(exact_powers) > self._eigen_decomposition_threshold:
      s, q = utils.safe_psd_eigh(state.matrix.value)
      state.cache = dict(eigenvalues=s, eigen_vectors=q)
    else:
      if eigenvalues:
        state.cache["eigenvalues"] = utils.safe_psd_eigh(state.matrix.value)[0]
      for power in exact_powers:
        if power == -1:
          state.cache[str(power)] = utils.psd_inv_cholesky(
              state.matrix.value, identity_weight)
        else:
          matrix = state.matrix.value + identity_weight * jnp.eye(self.dim)
          state.cache[str(power)] = jnp.linalg.matrix_power(matrix, power)
    return state

  def _to_dense_unscaled(self, state: "Full.State") -> chex.Array:
    # Permute the matrix according to the parameters canonical order
    return utils.block_permuted(
        state.matrix.value,
        block_sizes=[utils.product(shape) for shape in self.parameters_shapes],
        block_order=self.parameters_canonical_order
    )


class TwoKroneckerFactored(CurvatureBlock, abc.ABC):
  """An abstract class for approximating the block with a Kronecker product."""

  @utils.pytree_dataclass
  class State(CurvatureBlock.State):
    """Persistent state of the block.

    Attributes:
      inputs_factor: A moving average of the estimated second moment matrix of
        the inputs to the associated layer.
      outputs_factor: A moving average of the estimated second moment matrix of
        the gradients of w.r.t. the outputs of the associated layer.
    """
    inputs_factor: utils.WeightedMovingAverage
    outputs_factor: utils.WeightedMovingAverage

  @property
  def has_bias(self) -> bool:
    """Whether this layer's equation has a bias."""
    return len(self._layer_tag_eq.invars) == 4

  @abc.abstractmethod
  def input_size(self) -> int:
    """Number of inputs to the layer to which this block corresponds."""

  @abc.abstractmethod
  def output_size(self) -> int:
    """Number of outputs to the layer to which this block corresponds."""

  def parameters_shaped_list_to_single_matrix(
      self,
      parameters_shaped_list: Sequence[chex.Array],
  ) -> chex.Array:
    """Converts the values of parameters into a single matrix."""
    for p, s in zip(parameters_shaped_list, self.parameters_shapes):
      assert p.shape == s
    if self.has_bias:
      w, b = parameters_shaped_list
      return jnp.concatenate([w.reshape([-1, w.shape[-1]]), b[None]], axis=0)
    else:
      # This correctly reshapes the parameters of both dense and conv2d blocks
      w, = parameters_shaped_list
      return w.reshape([-1, w.shape[-1]])

  def single_matrix_to_parameters_shaped_list(
      self,
      matrix: chex.Array,
  ) -> Tuple[chex.Array, ...]:
    """Inverts the transformation of ``self.parameters_list_to_single_matrix``."""
    if self.has_bias:
      w, b = matrix[:-1], matrix[-1]
      return w.reshape(self.parameters_shapes[0]), b
    else:
      return matrix.reshape(self.parameters_shapes[0]),

  def _init(
      self,
      rng: chex.PRNGKey,
      exact_powers_to_cache: Set[chex.Scalar],
      approx_powers_to_cache: Set[chex.Scalar],
      cache_eigenvalues: bool,
  ) -> "TwoKroneckerFactored.State":
    # The extra scale is technically a constant, but in general it could be
    # useful for anyone examining the state to know it explicitly,
    # hence we actually keep it as part of the state.
    d_in = self.input_size()
    d_out = self.output_size()
    cache = {}
    if cache_eigenvalues or exact_powers_to_cache:
      cache["inputs_factor_eigenvalues"] = jnp.zeros([d_in])
      cache["outputs_factor_eigenvalues"] = jnp.zeros([d_out])
    if exact_powers_to_cache:
      cache["inputs_factor_eigen_vectors"] = jnp.zeros([d_in, d_in])
      cache["outputs_factor_eigen_vectors"] = jnp.zeros([d_out, d_out])
    for power in approx_powers_to_cache:
      if power != -1:
        raise NotImplementedError(f"Approximations for power {power} is not "
                                  f"yet implemented.")
      cache[str(power)] = dict(
          inputs_factor=jnp.zeros([d_in, d_in]),
          outputs_factor=jnp.zeros([d_out, d_out]),
      )
    return TwoKroneckerFactored.State(
        cache=cache,
        inputs_factor=utils.WeightedMovingAverage.zero((d_in, d_in)),
        outputs_factor=utils.WeightedMovingAverage.zero((d_out, d_out)),
    )

  def _multiply_matpower_unscaled(
      self,
      state: "TwoKroneckerFactored.State",
      vector: Sequence[chex.Array],
      identity_weight: chex.Numeric,
      power: chex.Scalar,
      exact_power: bool,
      use_cached: bool,
  ) -> Tuple[chex.Array, ...]:
    vector = self.parameters_shaped_list_to_single_matrix(vector)
    if power == 1:
      result = utils.kronecker_product_mul_v(
          state.outputs_factor.value,
          state.inputs_factor.value,
          vector,
          a_is_symmetric=True)
      result = result + identity_weight * vector
    elif exact_power:
      if not use_cached:
        s_i, q_i = utils.safe_psd_eigh(state.inputs_factor.value)
        s_o, q_o = utils.safe_psd_eigh(state.outputs_factor.value)
        eigenvalues = jnp.outer(s_i, s_o)
      else:
        s_i = state.cache["inputs_factor_eigenvalues"]
        q_i = state.cache["inputs_factor_eigen_vectors"]
        s_o = state.cache["outputs_factor_eigenvalues"]
        q_o = state.cache["outputs_factor_eigen_vectors"]
        eigenvalues = jnp.outer(s_i, s_o)
      eigenvalues = eigenvalues + identity_weight
      eigenvalues = jnp.power(eigenvalues, power)
      result = utils.kronecker_eigen_basis_mul_v(q_o, q_i, eigenvalues, vector)
    else:
      if not use_cached:
        raise NotImplementedError()
      else:
        result = utils.kronecker_product_mul_v(
            state.cache[str(power)]["outputs_factor"],
            state.cache[str(power)]["inputs_factor"],
            vector,
            a_is_symmetric=True)
    return self.single_matrix_to_parameters_shaped_list(result)

  def _eigenvalues_unscaled(
      self,
      state: "TwoKroneckerFactored.State",
      use_cached: bool,
  ) -> chex.Array:
    if use_cached:
      s_i = state.cache["inputs_factor_eigenvalues"]
      s_o = state.cache["outputs_factor_eigenvalues"]
    else:
      s_i, _ = utils.safe_psd_eigh(state.inputs_factor.value)
      s_o, _ = utils.safe_psd_eigh(state.outputs_factor.value)
    return jnp.outer(s_o, s_i)

  def _update_cache(
      self,
      state: "TwoKroneckerFactored.State",
      identity_weight: chex.Numeric,
      exact_powers: chex.Numeric,
      approx_powers: chex.Numeric,
      eigenvalues: bool,
      pmap_axis_name: Optional[str],
  ) -> "TwoKroneckerFactored.State":
    state.inputs_factor.sync(pmap_axis_name)
    state.outputs_factor.sync(pmap_axis_name)
    if eigenvalues or exact_powers:
      s_i, q_i = utils.safe_psd_eigh(state.inputs_factor.value)
      s_o, q_o = utils.safe_psd_eigh(state.outputs_factor.value)
      state.cache["inputs_factor_eigenvalues"] = s_i
      state.cache["outputs_factor_eigenvalues"] = s_o
      if exact_powers:
        state.cache["inputs_factor_eigen_vectors"] = q_i
        state.cache["outputs_factor_eigen_vectors"] = q_o
    for power in approx_powers:
      if power != -1:
        raise NotImplementedError(f"Approximations for power {power} is not "
                                  f"yet implemented.")
      cache = state.cache[str(power)]
      # This computes the approximate inverse factor using the pi-adjusted
      # inversion from the original KFAC paper.
      (cache["inputs_factor"],
       cache["outputs_factor"]) = utils.pi_adjusted_inverse(
           a=state.inputs_factor.value,
           b=state.outputs_factor.value,
           damping=identity_weight,
           pmap_axis_name=pmap_axis_name)
    return state

  def _to_dense_unscaled(
      self,
      state: "TwoKroneckerFactored.State"
  ) -> chex.Array:
    assert 0 < self.number_of_parameters <= 2
    inputs_factor = state.inputs_factor.value
    if self.has_bias:
      # Permute the matrix according to the parameters canonical order
      if self.parameters_canonical_order[0] != 0:
        inputs_factor = utils.block_permuted(
            state.inputs_factor.value,
            block_sizes=[state.inputs_factor.raw_value.shape[0] - 1, 1],
            block_order=(1, 0),
        )
      else:
        inputs_factor = state.inputs_factor.value
    return jnp.kron(inputs_factor, state.outputs_factor.value)


class NaiveDiagonal(Diagonal):
  """Approximates the diagonal of the curvature with in the most obvious way.

  The update to the curvature estimate is computed by ``(sum_i g_i) ** 2 / N``.
  where `g_i` is the gradient of each individual data point, and ``N`` is the
  batch size.
  """

  def _update_curvature_matrix_estimate(
      self,
      state: "NaiveDiagonal.State",
      estimation_data: Mapping[str, Sequence[chex.Array]],
      ema_old: chex.Numeric,
      ema_new: chex.Numeric,
      batch_size: int,
      pmap_axis_name: str,
  ) -> "NaiveDiagonal.State":
    for factor, dw in zip(
        state.diagonal_factors, estimation_data["params_tangent"]):
      factor.update(dw * dw / batch_size, ema_old, ema_new)
    return state


class NaiveFull(Full):
  """Approximates the full curvature with in the most obvious way.

  The update to the curvature estimate is computed by
  ``(sum_i g_i) (sum_i g_i)^T / N``, where ``g_i`` is the gradient of each
  individual data point, and ``N`` is the batch size.
  """

  def update_curvature_matrix_estimate(
      self,
      state: Full.State,
      estimation_data: Mapping[str, Sequence[chex.Array]],
      ema_old: chex.Numeric,
      ema_new: chex.Numeric,
      batch_size: int,
      pmap_axis_name: Optional[str],
  ) -> Full.State:
    params_grads = jax.tree_leaves(estimation_data["params_tangent"])
    params_grads = jax.tree_map(lambda x: x.flatten(), params_grads)
    grads = jnp.concatenate(params_grads, axis=0)
    state.matrix.update(jnp.outer(grads, grads) / batch_size, ema_old, ema_new)
    return state


#  _____
# |  __ \
# | |  | | ___ _ __  ___  ___
# | |  | |/ _ \ '_ \/ __|/ _ \
# | |__| |  __/ | | \__ \  __/
# |_____/ \___|_| |_|___/\___|
#


class DenseDiagonal(Diagonal):
  """A `Diagonal` block specifically for dense layers."""

  @property
  def has_bias(self) -> bool:
    """Whether the layer has a bias parameter."""
    return len(self.parameters_shapes) == 2

  def _update_curvature_matrix_estimate(
      self,
      state: "Diagonal.State",
      estimation_data: Mapping[str, Sequence[chex.Array]],
      ema_old: chex.Numeric,
      ema_new: chex.Numeric,
      batch_size: int,
      pmap_axis_name: Optional[str],
  ) -> "Diagonal.State":
    x, = estimation_data["inputs"]
    dy, = estimation_data["outputs_tangent"]
    assert utils.first_dim_is_size(batch_size, x, dy)

    diagonals = (jnp.matmul((x * x).T, dy * dy) / batch_size,)
    if self.has_bias:
      diagonals += (jnp.mean(dy * dy, axis=0),)

    assert len(diagonals) == self.number_of_parameters
    for diagonal_factor, diagonal in zip(state.diagonal_factors, diagonals):
      diagonal_factor.update(diagonal, ema_old, ema_new)
    return state


class DenseFull(Full):
  """A `Full` block specifically for dense layers."""

  def update_curvature_matrix_estimate(
      self,
      state: "Full.State",
      estimation_data: Mapping[str, Sequence[chex.Array]],
      ema_old: chex.Numeric,
      ema_new: chex.Numeric,
      batch_size: int,
      pmap_axis_name: Optional[str],
  ) -> "Full.State":
    x, = estimation_data["inputs"]
    dy, = estimation_data["outputs_tangent"]
    assert utils.first_dim_is_size(batch_size, x, dy)

    params_tangents = x[:, :, None] * dy[:, None, :]
    if self.number_of_parameters == 2:
      params_tangents = jnp.concatenate([params_tangents, dy[:, None]], axis=1)
    params_tangents = jnp.reshape(params_tangents, [batch_size, -1])
    matrix_update = jnp.matmul(params_tangents.T, params_tangents) / batch_size
    state.matrix.update(matrix_update, ema_old, ema_new)
    return state


class DenseTwoKroneckerFactored(TwoKroneckerFactored):
  """A :class:`~TwoKroneckerFactored` block specifically for dense layers."""

  def input_size(self) -> int:
    """The size of the Kronecker-factor corresponding to inputs."""
    if self.has_bias:
      return self.parameters_shapes[0][0] + 1
    else:
      return self.parameters_shapes[0][0]

  def output_size(self) -> int:
    """The size of the Kronecker-factor corresponding to outputs."""
    return self.parameters_shapes[0][1]

  def update_curvature_matrix_estimate(
      self,
      state: TwoKroneckerFactored.State,
      estimation_data: Mapping[str, Sequence[chex.Array]],
      ema_old: chex.Numeric,
      ema_new: chex.Numeric,
      batch_size: int,
      pmap_axis_name: Optional[str],
  ) -> TwoKroneckerFactored.State:
    del pmap_axis_name
    x, = estimation_data["inputs"]
    dy, = estimation_data["outputs_tangent"]
    assert utils.first_dim_is_size(batch_size, x, dy)

    if self.has_bias:
      x_one = jnp.ones_like(x[:, :1])
      x = jnp.concatenate([x, x_one], axis=1)
    input_stats = jnp.matmul(x.T, x) / batch_size
    output_stats = jnp.matmul(dy.T, dy) / batch_size
    state.inputs_factor.update(input_stats, ema_old, ema_new)
    state.outputs_factor.update(output_stats, ema_old, ema_new)
    return state


#   _____                ___  _____
#  / ____|              |__ \|  __ \
# | |     ___  _ ____   __ ) | |  | |
# | |    / _ \| '_ \ \ / // /| |  | |
# | |___| (_) | | | \ V // /_| |__| |
#  \_____\___/|_| |_|\_/|____|_____/
#


class Conv2DDiagonal(Diagonal):
  """A :class:`~Diagonal` block specifically for 2D convolution layers."""

  def __init__(
      self,
      layer_tag_eq: tags.LayerTagEqn,
      name: str,
      max_elements_for_vmap: Optional[int] = None,
  ):
    """Initializes the block.

    Since there is no 'nice' formula for computing the average of the
    tangents for a 2D convolution, what we do is that we have a function -
    ``self.conv2d_tangent_squared`` - that computes for a single feature map the
    square of the tangents for the kernel of the convolution. To average over
    the batch we have two choices - vmap or loop over the batch sequentially
    using scan. This utility function provides a trade-off by being able to
    specify the maximum number of batch size that we can vmap over. This means
    that the maximum memory usage will be ``max_batch_size_for_vmap`` times the
    memory needed when calling ``self.conv2d_tangent_squared``. And the actual
    ``vmap`` will be called ``ceil(total_batch_size / max_batch_size_for_vmap)``
    number of times in a loop to find the final average.

    Args:
      layer_tag_eq: The Jax equation corresponding to the layer tag, that this
        block will approximate the curvature to.
      name: The name of this block.
      max_elements_for_vmap: The threshold used for determining how much
        computation to the in parallel and how much in serial manner. If
        ``None`` will use the value returned by
        :func:`~get_max_parallel_elements`.
    """
    self._averaged_kernel_squared_tangents = utils.loop_and_parallelize_average(
        func=self.conv2d_tangent_squared,
        max_parallel_size=max_elements_for_vmap or get_max_parallel_elements(),
    )
    super().__init__(layer_tag_eq, name)

  @property
  def has_bias(self) -> bool:
    return len(self.parameters_shapes) == 2

  def conv2d_tangent_squared(
      self,
      image_features_map: chex.Array,
      output_tangent: chex.Array,
  ) -> chex.Array:
    """Computes the elementwise square of a tangent for a single feature map."""
    extra_params = {k: v for k, v in self.layer_tag_extra_params.items()
                    if k not in ("lhs_shape", "rhs_shape")}
    _, vjp = jax.vjp(
        functools.partial(
            jax.lax.conv_general_dilated,
            **extra_params
        ),
        image_features_map[None], jnp.zeros(self.parameters_shapes[0])
    )
    # Note the first tangent is w.r.t. inputs
    return jnp.square(vjp(output_tangent[None])[1])

  def _update_curvature_matrix_estimate(
      self,
      state: Diagonal.State,
      estimation_data: Mapping[str, Sequence[chex.Array]],
      ema_old: chex.Numeric,
      ema_new: chex.Numeric,
      batch_size: int,
      pmap_axis_name: Optional[str],
  ) -> Diagonal.State:
    x, = estimation_data["inputs"]
    dy, = estimation_data["outputs_tangent"]
    assert utils.first_dim_is_size(batch_size, x, dy)

    diagonals = (self._averaged_kernel_squared_tangents(x, dy),)
    if self.has_bias:
      sum_axis = tuple(range(1, dy.ndim - len(self.parameters_shapes[1])))
      bias_dy = jnp.sum(dy, axis=sum_axis)
      diagonals += (jnp.mean(bias_dy * bias_dy, axis=0),)

    assert len(diagonals) == self.number_of_parameters
    for diagonal_factor, diagonal in zip(state.diagonal_factors, diagonals):
      diagonal_factor.update(diagonal, ema_old, ema_new)
    return state


class Conv2DFull(Full):
  """A :class:`~Full` block specifically for 2D convolution layers."""

  def __init__(
      self,
      layer_tag_eq: tags.LayerTagEqn,
      name: str,
      max_elements_for_vmap: Optional[int] = None,
  ):
    """Initializes the block.

    Since there is no 'nice' formula for computing the average of the
    tangents for a 2D convolution, what we do is that we have a function -
    ``self.conv2d_tangent_squared`` - that computes for a single feature map the
    square of the tangents for the kernel of the convolution. To average over
    the batch we have two choices - vmap or loop over the batch sequentially
    using scan. This utility function provides a trade-off by being able to
    specify the maximum number of batch size that we can vmap over. This means
    that the maximum memory usage will be ``max_batch_size_for_vmap`` times the
    memory needed when calling ``self.conv2d_tangent_squared``. And the actual
    ``vmap`` will be called ``ceil(total_batch_size / max_batch_size_for_vmap)``
    number of times in a loop to find the final average.

    Args:
      layer_tag_eq: The Jax equation corresponding to the layer tag, that this
        block will approximate the curvature to.
      name: The name of this block.
      max_elements_for_vmap: The threshold used for determining how much
        computation to the in parallel and how much in serial manner. If
        ``None`` will use the value returned by
        :func:`~get_max_parallel_elements`.
    """
    self._averaged_tangents_outer_product = utils.loop_and_parallelize_average(
        func=self.conv2d_tangent_outer_product,
        max_parallel_size=max_elements_for_vmap or get_max_parallel_elements(),
    )
    super().__init__(layer_tag_eq, name)

  def conv2d_tangent_outer_product(
      self,
      inputs: chex.Array,
      tangent_of_outputs: chex.Array,
  ) -> chex.Array:
    """Computes the outer product of a tangent for a single feature map."""
    extra_params = {k: v for k, v in self.layer_tag_extra_params.items()
                    if k not in ("lhs_shape", "rhs_shape")}
    _, vjp = jax.vjp(
        functools.partial(
            jax.lax.conv_general_dilated,
            **extra_params
        ),
        inputs[None], jnp.zeros(self.parameters_shapes[0])
    )
    # Note the first tangent is w.r.t. inputs
    tangents = (vjp(tangent_of_outputs[None])[1],)
    if self.number_of_parameters == 2:
      num_axis = tangent_of_outputs.ndim - len(self.parameters_shapes[1])
      sum_axis = tuple(range(num_axis))
      tangents += (jnp.sum(tangent_of_outputs, axis=sum_axis),)
    flat_tangents = self.parameters_list_to_single_vector(tangents)
    return jnp.outer(flat_tangents, flat_tangents)

  def update_curvature_matrix_estimate(
      self,
      state: Full.State,
      estimation_data: Mapping[str, Sequence[chex.Array]],
      ema_old: chex.Numeric,
      ema_new: chex.Numeric,
      batch_size: int,
      pmap_axis_name: Optional[str],
  ) -> Full.State:
    x, = estimation_data["inputs"]
    dy, = estimation_data["outputs_tangent"]
    assert utils.first_dim_is_size(batch_size, x, dy)

    matrix_update = self._averaged_tangents_outer_product(x, dy)
    state.matrix.update(matrix_update, ema_old, ema_new)
    return state


class Conv2DTwoKroneckerFactored(TwoKroneckerFactored):
  """A :class:`~TwoKroneckerFactored` block specifically for 2D convolution layers."""

  @property
  def outputs_channel_index(self) -> int:
    """The ``channels`` index in the outputs of the layer."""
    return self._layer_tag_eq.params["dimension_numbers"].out_spec[1]

  @property
  def inputs_channel_index(self) -> int:
    """The ``channels`` index in the inputs of the layer."""
    return self._layer_tag_eq.params["dimension_numbers"].lhs_spec[1]

  @property
  def weights_output_channel_index(self) -> int:
    """The ``channels`` index in weights of the layer."""
    return self._layer_tag_eq.params["dimension_numbers"].rhs_spec[0]

  @property
  def weights_spatial_size(self) -> int:
    """The spatial filter size of the weights."""
    weights_shape = self._layer_tag_eq.params["rhs_shape"]
    indices = self._layer_tag_eq.params["dimension_numbers"].rhs_spec[2:4]
    return weights_shape[indices[0]] * weights_shape[indices[1]]

  def input_size(self) -> int:
    if self.has_bias:
      return self.num_inputs_channels * self.weights_spatial_size + 1
    else:
      return self.num_inputs_channels * self.weights_spatial_size

  def output_size(self) -> int:
    return self.num_outputs_channels

  @property
  def num_inputs_channels(self) -> int:
    """The number of channels in the inputs to the layer."""
    return self._layer_tag_eq.params["lhs_shape"][self.inputs_channel_index]

  @property
  def num_outputs_channels(self) -> int:
    """The number of channels in the outputs to the layer."""
    return self._layer_tag_eq.params["rhs_shape"][
        self.weights_output_channel_index]

  def num_locations(
      self,
      inputs: chex.Array,
      params: Sequence[chex.Array],
  ) -> int:
    """The number of spatial locations that each filter is applied to."""
    spatial_index = self._layer_tag_eq.params["dimension_numbers"].lhs_spec[2:]
    inputs_spatial_shape = tuple(inputs.shape[i] for i in spatial_index)
    kernel_index = self._layer_tag_eq.params["dimension_numbers"].rhs_spec[2:]
    kernel_spatial_shape = tuple(params[0].shape[i] for i in kernel_index)
    return psm.num_conv_locations(
        inputs_spatial_shape, kernel_spatial_shape,
        self._layer_tag_eq.params["window_strides"],
        self._layer_tag_eq.params["padding"])

  def compute_inputs_stats(
      self,
      inputs: chex.Array,
      params: Sequence[chex.Array],
  ) -> chex.Array:
    """Computes the statistics for the inputs factor."""
    input_cov_m, input_cov_v = psm.patches_moments(
        inputs,
        kernel_shape=params[0].shape,
        strides=self._layer_tag_eq.params["window_strides"],
        padding=self._layer_tag_eq.params["padding"],
        data_format=None,
        dim_numbers=self._layer_tag_eq.params["dimension_numbers"],
        precision=self._layer_tag_eq.params.get("precision"),
    )
    # Flatten the kernel and channels dimensions
    k, h, c = input_cov_v.shape
    input_cov_v = jnp.reshape(input_cov_v, (k * h * c,))
    input_cov_m = jnp.reshape(input_cov_m, (k * h * c, k * h * c))
    # Normalize by the batch size
    input_cov_m = input_cov_m / inputs.shape[0]
    input_cov_v = input_cov_v / inputs.shape[0]
    if not self.has_bias:
      return input_cov_m
    num_locations = jnp.full((1,), self.num_locations(inputs, params))
    input_cov = jnp.concatenate([input_cov_m, input_cov_v[None]], axis=0)
    input_cov_v = jnp.concatenate([input_cov_v, num_locations], axis=0)
    return jnp.concatenate([input_cov, input_cov_v[:, None]], axis=1)

  def compute_outputs_stats(
      self,
      tangent_of_output: chex.Array,
  ) -> chex.Array:
    """Computes the statistics for the outputs factor."""
    if self.outputs_channel_index != 3:
      index = list(range(4))
      index.remove(self.outputs_channel_index)
      index = index + [self.outputs_channel_index]
      tangent_of_output = jnp.transpose(tangent_of_output, index)
    tangent_of_output = jnp.reshape(tangent_of_output,
                                    (-1, tangent_of_output.shape[-1]))
    # Normalize by the batch size * number of locations
    return (jnp.matmul(tangent_of_output.T, tangent_of_output) /
            tangent_of_output.shape[0])

  def update_curvature_matrix_estimate(
      self,
      state: TwoKroneckerFactored.State,
      estimation_data: Mapping[str, Sequence[chex.Array]],
      ema_old: chex.Numeric,
      ema_new: chex.Numeric,
      batch_size: int,
      pmap_axis_name: Optional[str],
  ) -> TwoKroneckerFactored.State:
    del pmap_axis_name
    x, = estimation_data["inputs"]
    dy, = estimation_data["outputs_tangent"]
    assert utils.first_dim_is_size(batch_size, x, dy)

    input_cov = self.compute_inputs_stats(x, estimation_data["params"])
    output_cov = self.compute_outputs_stats(dy)
    state.inputs_factor.update(input_cov, ema_old, ema_new)
    state.outputs_factor.update(output_cov, ema_old, ema_new)
    return state


#   _____           _                         _  _____ _     _  __ _
#  / ____|         | |        /\             | |/ ____| |   (_)/ _| |
# | (___   ___ __ _| | ___   /  \   _ __   __| | (___ | |__  _| |_| |_
#  \___ \ / __/ _` | |/ _ \ / /\ \ | '_ \ / _` |\___ \| '_ \| |  _| __|
#  ____) | (_| (_| | |  __// ____ \| | | | (_| |____) | | | | | | | |_
# |_____/ \___\__,_|_|\___/_/    \_\_| |_|\__,_|_____/|_| |_|_|_|  \__|
#


class ScaleAndShiftDiagonal(Diagonal):
  """A diagonal approximation specifically for a scale and shift layers."""

  @property
  def has_scale(self) -> bool:
    """Whether this layer's equation has a scale."""
    return self._layer_tag_eq.params["has_scale"]

  @property
  def has_shift(self) -> bool:
    """Whether this layer's equation has a shift."""
    return self._layer_tag_eq.params["has_shift"]

  def _update_curvature_matrix_estimate(
      self,
      state: Diagonal.State,
      estimation_data: Mapping[str, Sequence[chex.Array]],
      ema_old: chex.Numeric,
      ema_new: chex.Numeric,
      batch_size: int,
      pmap_axis_name: Optional[str],
  ) -> Diagonal.State:
    x, = estimation_data["inputs"]
    dy, = estimation_data["outputs_tangent"]
    assert utils.first_dim_is_size(batch_size, x, dy)

    if self.has_scale:
      assert (state.diagonal_factors[0].raw_value.shape ==
              self.parameters_shapes[0])
      scale_shape = estimation_data["params"][0].shape
      axis = range(x.ndim)[1:(x.ndim - len(scale_shape))]
      d_scale = jnp.sum(x * dy, axis=tuple(axis))
      scale_diag_update = jnp.sum(d_scale * d_scale, axis=0) / batch_size
      state.diagonal_factors[0].update(scale_diag_update, ema_old, ema_new)

    if self.has_shift:
      assert (state.diagonal_factors[-1].raw_value.shape ==
              self.parameters_shapes[-1])
      shift_shape = estimation_data["params"][-1].shape
      axis = range(x.ndim)[1:(x.ndim - len(shift_shape))]
      d_shift = jnp.sum(dy, axis=tuple(axis))
      shift_diag_update = jnp.sum(d_shift * d_shift, axis=0) / batch_size
      state.diagonal_factors[-1].update(shift_diag_update, ema_old, ema_new)

    return state


class ScaleAndShiftFull(Full):
  """A full dense approximation specifically for a scale and shift layers."""

  @property
  def _has_scale(self) -> bool:
    """Whether this layer's equation has a scale."""
    return self._layer_tag_eq.params["has_scale"]

  @property
  def _has_shift(self) -> bool:
    """Whether this layer's equation has a shift."""
    return self._layer_tag_eq.params["has_shift"]

  def update_curvature_matrix_estimate(
      self,
      state: Full.State,
      estimation_data: Mapping[str, Sequence[chex.Array]],
      ema_old: chex.Numeric,
      ema_new: chex.Numeric,
      batch_size: int,
      pmap_axis_name: Optional[str],
  ) -> Full.State:
    del pmap_axis_name
    x, = estimation_data["inputs"]
    dy, = estimation_data["outputs_tangent"]
    assert utils.first_dim_is_size(batch_size, x, dy)

    tangents = []
    if self._has_scale:
      # Scale tangent
      scale_shape = estimation_data["params"][0].shape
      axis = range(x.ndim)[1:(x.ndim - len(scale_shape))]
      d_scale = jnp.sum(x * dy, axis=tuple(axis))
      d_scale = d_scale.reshape([batch_size, -1])
      tangents.append(d_scale)

    if self._has_shift:
      # Shift tangent
      shift_shape = estimation_data["params"][-1].shape
      axis = range(x.ndim)[1:(x.ndim - len(shift_shape))]
      d_shift = jnp.sum(dy, axis=tuple(axis))
      d_shift = d_shift.reshape([batch_size, -1])
      tangents.append(d_shift)

    tangents = jnp.concatenate(tangents, axis=1)
    matrix_update = jnp.matmul(tangents.T, tangents) / batch_size
    state.matrix.update(matrix_update, ema_old, ema_new)

    return state
