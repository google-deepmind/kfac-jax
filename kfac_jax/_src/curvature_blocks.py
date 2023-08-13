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
import string
from typing import Optional, Sequence, Any, Set, Tuple, Union, Dict, Mapping

import jax
import jax.numpy as jnp
from kfac_jax._src import layers_and_loss_tags as tags
from kfac_jax._src import patches_second_moment as psm
from kfac_jax._src import tag_graph_matcher as tgm
from kfac_jax._src import utils
import numpy as np

# Types for annotation
Array = utils.Array
Scalar = utils.Scalar
Numeric = utils.Numeric
PRNGKey = utils.PRNGKey
Shape = utils.Shape
DType = utils.DType
ScalarOrSequence = Union[Scalar, Sequence[Scalar]]
Cache = Dict[str, Union[Array, Dict[str, Array]]]

# Special global variables
# This is used for einsum strings
_ALPHABET = string.ascii_lowercase
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
  return _DEFAULT_EIGEN_DECOMPOSITION_THRESHOLD


def _to_real_set(
    number_or_sequence: Optional[ScalarOrSequence]
) -> Set[Scalar]:
  """Converts the optional number or sequence to a set."""
  if number_or_sequence is None:
    return set()
  elif isinstance(number_or_sequence, set):
    return number_or_sequence
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
  pre-computed to make repeated multiplications cheaper. During initialization,
  you would have to explicitly specify all powers that you will need to cache.
  """

  @utils.register_state_class
  class State(utils.State):
    """Persistent state of the block.

    Any subclasses of :class:`~CurvatureBlock` should also internally extend
    this class, with any attributes needed for the curvature estimation.

    Attributes:
      cache: A dictionary, containing any state data that is updated on
        irregular intervals, such as inverses, eigenvalues, etc. Elements of
        this are updated via calls to :func:`~CurvatureBlock.update_cache`, and
        do not necessarily correspond to the most up-to-date curvature estimate.
    """
    cache: Optional[Dict[str, Union[Array, Dict[str, Array]]]]

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
  def outputs_shapes(self) -> Tuple[Shape, ...]:
    """The shapes of the output variables of the block's tag equation."""

    output_vars = self.layer_tag_primitive.split_all_inputs(
        self._layer_tag_eq.invars)[0]

    return jax.tree_util.tree_map(lambda x: x.aval.shape, output_vars)

  @property
  def inputs_shapes(self) -> Tuple[Shape, ...]:
    """The shapes of the input variables of the block's tag equation."""

    input_vars = self.layer_tag_primitive.split_all_inputs(
        self._layer_tag_eq.invars)[1]

    return jax.tree_util.tree_map(lambda x: x.aval.shape, input_vars)

  @property
  def parameters_shapes(self) -> Tuple[Shape, ...]:
    """The shapes of the parameter variables of the block's tag equation."""
    return tuple(jax.tree_util.tree_map(
        lambda x: tuple(x.aval.shape), self.parameter_variables))

  @property
  def dtype(self) -> DType:
    dtypes = set(p.aval.dtype for p in self.parameter_variables)  # pytype: disable=attribute-error
    if len(dtypes) > 1:
      raise ValueError("Not all parameters are the same dtype.")
    return dtypes.pop()

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

  def scale(self, state: "CurvatureBlock.State", use_cache: bool) -> Numeric:
    """A scalar pre-factor of the curvature approximation.

    Importantly, all methods assume that whenever a user requests cached values,
    any state dependant scale is taken into account by the cache (e.g. either
    stored explicitly and used or mathematically added to values).

    Args:
      state: The state for this block.
      use_cache: Whether the method requesting this is using cached values or
        not.

    Returns:
      A scalar value to be multiplied with any unscaled block representation.
    """
    if use_cache:
      return self.fixed_scale()

    return self.fixed_scale() * self.state_dependent_scale(state)

  def fixed_scale(self) -> Numeric:
    """A fixed scalar pre-factor of the curvature (e.g. constant)."""
    return 1.0

  def state_dependent_scale(self, state: "CurvatureBlock.State") -> Numeric:
    """A scalar pre-factor of the curvature, computed from the most fresh curvature estimate."""
    del state  # Unused
    return 1.0

  def __str__(self):
    return f"{self._name!r}[{self.parameters_shapes!r}]"

  @utils.auto_scope_method
  def init(
      self,
      rng: PRNGKey,
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
          :func:`~CurvatureBlock.multiply_matpower`,
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
      rng: PRNGKey,
      exact_powers_to_cache: Set[Scalar],
      approx_powers_to_cache: Set[Scalar],
      cache_eigenvalues: bool,
  ) -> "CurvatureBlock.State":
    """The non-public interface of ``init``."""

  @abc.abstractmethod
  def sync(
      self,
      state: "CurvatureBlock.State",
      pmap_axis_name: str,
  ) -> "CurvatureBlock.State":
    """Syncs the state across different devices (does not sync the cache)."""

  @utils.auto_scope_method
  def multiply_matpower(
      self,
      state: "CurvatureBlock.State",
      vector: Sequence[Array],
      identity_weight: Numeric,
      power: Scalar,
      exact_power: bool,
      use_cached: bool,
  ) -> Tuple[Array, ...]:
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
        :func:`~CurvatureBlock.update_cache`. The precise semantics of this
        depend on the concrete subclass and its particular behavior in regard to
        caching.
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
        to :func:`~CurvatureBlock.update_cache` with the same value of ``power``

    Returns:
      A tuple of arrays, representing the result of the matrix-vector product.
    """
    scale = self.scale(state, use_cached)
    result = self._multiply_matpower_unscaled(
        state=state,
        vector=vector,
        identity_weight=identity_weight / scale,
        power=power,
        exact_power=exact_power,
        use_cached=use_cached,
    )

    return utils.scalar_mul(result, jnp.power(scale, power))

  @abc.abstractmethod
  def _multiply_matpower_unscaled(
      self,
      state: "CurvatureBlock.State",
      vector: Sequence[Array],
      identity_weight: Numeric,
      power: Scalar,
      exact_power: bool,
      use_cached: bool,
  ) -> Tuple[Array, ...]:
    """Performs matrix-vector multiplication, ignoring ``self.scale``."""

  def multiply(
      self,
      state: "CurvatureBlock.State",
      vector: Sequence[Array],
      identity_weight: Numeric,
      exact_power: bool,
      use_cached: bool,
  ) -> Tuple[Array, ...]:
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
      vector: Sequence[Array],
      identity_weight: Numeric,
      exact_power: bool,
      use_cached: bool,
  ) -> Tuple[Array, ...]:
    """Computes ``(BlockMatrix + identity_weight I)^-1`` times ``vector``."""

    return self.multiply_matpower(
        state=state,
        vector=vector,
        identity_weight=identity_weight,
        power=-1,
        exact_power=exact_power,
        use_cached=use_cached,
    )

  @utils.auto_scope_method
  def eigenvalues(
      self,
      state: "CurvatureBlock.State",
      use_cached: bool,
  ) -> Array:
    """Computes the eigenvalues for this block approximation.

    Args:
      state: The state dict for this block.
      use_cached: Whether to use a cached versions of the eigenvalues or to use
        the most recent curvature estimates to compute them. The cached version
        are going to be *at least* as fresh as the last time you called
        :func:`~CurvatureBlock.update_cache` with ``eigenvalues=True``.

    Returns:
      An array containing the eigenvalues of the block.
    """
    eigenvalues = self._eigenvalues_unscaled(state, use_cached)

    assert eigenvalues.size == self.dim

    return self.scale(state, use_cached) * eigenvalues

  @abc.abstractmethod
  def _eigenvalues_unscaled(
      self,
      state: "CurvatureBlock.State",
      use_cached: bool,
  ) -> Array:
    """Computes the eigenvalues for this block, ignoring `self.scale`."""

  @abc.abstractmethod
  def update_curvature_matrix_estimate(
      self,
      state: "CurvatureBlock.State",
      estimation_data: Dict[str, Sequence[Array]],
      ema_old: Numeric,
      ema_new: Numeric,
      batch_size: Numeric,
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
    """

  @utils.auto_scope_method
  def update_cache(
      self,
      state: "CurvatureBlock.State",
      identity_weight: Numeric,
      exact_powers: Optional[ScalarOrSequence],
      approx_powers: Optional[ScalarOrSequence],
      eigenvalues: bool,
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

    Returns:
        The updated state.
    """
    return self._update_cache(
        state=state,
        identity_weight=identity_weight / self.scale(state, False),
        exact_powers=_to_real_set(exact_powers),
        approx_powers=_to_real_set(approx_powers),
        eigenvalues=eigenvalues,
    )

  @abc.abstractmethod
  def _update_cache(
      self,
      state: "CurvatureBlock.State",
      identity_weight: Numeric,
      exact_powers: Set[Scalar],
      approx_powers: Set[Scalar],
      eigenvalues: bool,
  ) -> "CurvatureBlock.State":
    """The cache updating function, ignoring ``self.scale``."""

  @utils.auto_scope_method
  def to_dense_matrix(self, state: "CurvatureBlock.State") -> Array:
    """Returns a dense representation of the approximate curvature matrix."""
    return self.scale(state, False) * self._to_dense_unscaled(state)

  @abc.abstractmethod
  def _to_dense_unscaled(self, state: "CurvatureBlock.State") -> Array:
    """A dense representation of the curvature, ignoring ``self.scale``."""


class ScaledIdentity(CurvatureBlock):
  """A block that assumes that the curvature is a scaled identity matrix."""

  def __init__(
      self,
      layer_tag_eq: tags.LayerTagEqn,
      name: str,
      scale: Numeric = 1.0,
  ):
    """Initializes the block.

    Args:
      layer_tag_eq: The Jax equation corresponding to the layer tag, that this
        block will approximate the curvature to.
      name: The name of this block.
      scale: The scale of the identity matrix.
    """
    self._scale = scale
    super().__init__(layer_tag_eq, name)

  def fixed_scale(self) -> Numeric:
    return self._scale

  def _init(
      self,
      rng: PRNGKey,
      exact_powers_to_cache: Set[Scalar],
      approx_powers_to_cache: Set[Scalar],
      cache_eigenvalues: bool,
  ) -> CurvatureBlock.State:

    del rng, exact_powers_to_cache, approx_powers_to_cache  # Unused

    return CurvatureBlock.State(
        cache=None,
    )

  def sync(
      self,
      state: CurvatureBlock.State,
      pmap_axis_name: str,
  ) -> CurvatureBlock.State:
    return state

  def _multiply_matpower_unscaled(
      self,
      state: CurvatureBlock.State,
      vector: Sequence[Array],
      identity_weight: Numeric,
      power: Scalar,
      exact_power: bool,
      use_cached: bool,
  ) -> Tuple[Array, ...]:

    del exact_power, use_cached  # Unused

    identity_weight = identity_weight + 1.0

    if power == 1:
      return jax.tree_util.tree_map(lambda x: identity_weight * x, vector)

    elif power == -1:
      return jax.tree_util.tree_map(lambda x: x / identity_weight, vector)

    else:
      identity_weight = jnp.power(identity_weight, power)
      return jax.tree_util.tree_map(lambda x: identity_weight * x, vector)

  def _eigenvalues_unscaled(
      self,
      state: "CurvatureBlock.State",
      use_cached: bool,
  ) -> Array:
    return jnp.ones([self.dim])

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: CurvatureBlock.State,
      estimation_data: Dict[str, Sequence[Array]],
      ema_old: Numeric,
      ema_new: Numeric,
      batch_size: Numeric,
  ) -> CurvatureBlock.State:

    return state.copy()

  def _update_cache(
      self,
      state: CurvatureBlock.State,
      identity_weight: Numeric,
      exact_powers: Set[Scalar],
      approx_powers: Set[Scalar],
      eigenvalues: bool,
  ) -> CurvatureBlock.State:

    return state.copy()

  def _to_dense_unscaled(self, state: CurvatureBlock.State) -> Array:
    del state  # not used
    return jnp.eye(self.dim)


class Diagonal(CurvatureBlock, abc.ABC):
  """An abstract class for approximating only the diagonal of curvature."""

  @utils.register_state_class
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
      rng: PRNGKey,
      exact_powers_to_cache: Set[Scalar],
      approx_powers_to_cache: Set[Scalar],
      cache_eigenvalues: bool,
  ) -> "Diagonal.State":

    del rng

    return Diagonal.State(
        cache=None,
        diagonal_factors=tuple(
            utils.WeightedMovingAverage.zeros_array(shape, self.dtype)
            for shape in self.parameters_shapes
        ),
    )

  def sync(
      self,
      state: "Diagonal.State",
      pmap_axis_name: str,
  ) -> "Diagonal.State":

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    for factor in state.diagonal_factors:
      factor.sync(pmap_axis_name)

    return state

  def _multiply_matpower_unscaled(
      self,
      state: "Diagonal.State",
      vector: Sequence[Array],
      identity_weight: Numeric,
      power: Scalar,
      exact_power: bool,
      use_cached: bool,
  ) -> Tuple[Array, ...]:

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
  ) -> Array:
    return jnp.concatenate([f.value.flatten() for f in state.diagonal_factors],
                           axis=0)

  def _update_cache(
      self,
      state: "Diagonal.State",
      identity_weight: Numeric,
      exact_powers: Set[Scalar],
      approx_powers: Set[Scalar],
      eigenvalues: bool,
  ) -> "Diagonal.State":
    return state.copy()

  def _to_dense_unscaled(self, state: "Diagonal.State") -> Array:

    # Extract factors in canonical order
    factors = [state.diagonal_factors[i].value.flatten()
               for i in self.parameters_canonical_order]

    # Construct diagonal matrix
    return jnp.diag(jnp.concatenate(factors, axis=0))


class Full(CurvatureBlock, abc.ABC):
  """An abstract class for approximating the block matrix with a full matrix."""

  @utils.register_state_class
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
      parameters_shaped_list: Sequence[Array],
  ) -> Array:
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
      vector: Array,
  ) -> Tuple[Array, ...]:
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
      rng: PRNGKey,
      exact_powers_to_cache: Set[Scalar],
      approx_powers_to_cache: Set[Scalar],
      cache_eigenvalues: bool,
  ) -> "Full.State":

    del rng

    # This block does not have any notion of "approximate" powers
    exact_powers_to_cache = exact_powers_to_cache | approx_powers_to_cache
    cache = {}

    if len(exact_powers_to_cache) > self._eigen_decomposition_threshold:
      cache["eigenvalues"] = jnp.zeros([self.dim], self.dtype)
      cache["eigen_vectors"] = jnp.zeros([self.dim, self.dim], self.dtype)

    elif cache_eigenvalues:
      cache["eigenvalues"] = jnp.zeros([self.dim], self.dtype)

    if len(exact_powers_to_cache) <= self._eigen_decomposition_threshold:
      for power in exact_powers_to_cache:
        cache[str(power)] = jnp.zeros([self.dim, self.dim], self.dtype)

    return Full.State(
        cache=cache,
        matrix=utils.WeightedMovingAverage.zeros_array(
            [self.dim, self.dim], self.dtype),
    )

  def sync(
      self,
      state: "Full.State",
      pmap_axis_name: str,
  ) -> "Full.State":

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    state.matrix.sync(pmap_axis_name)

    return state

  def _multiply_matpower_unscaled(
      self,
      state: "Full.State",
      vector: Sequence[Array],
      identity_weight: Numeric,
      power: Scalar,
      exact_power: bool,
      use_cached: bool,
  ) -> Tuple[Array, ...]:

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
  ) -> Array:
    if not use_cached:
      return utils.safe_psd_eigh(state.matrix.value)[0]
    else:
      return state.cache["eigenvalues"]

  def _update_cache(
      self,
      state: "Full.State",
      identity_weight: Numeric,
      exact_powers: Set[Scalar],
      approx_powers: Set[Scalar],
      eigenvalues: bool,
  ) -> "Full.State":

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    scale = self.state_dependent_scale(state)

    # This block does not have any notion of "approximate" powers
    exact_powers = exact_powers | approx_powers

    if len(exact_powers) > self._eigen_decomposition_threshold:

      s, q = utils.safe_psd_eigh(state.matrix.value)
      state.cache = dict(eigenvalues=scale * s, eigen_vectors=q)

    else:

      if eigenvalues:
        state.cache["eigenvalues"] = scale * utils.safe_psd_eigh(
            state.matrix.value)[0]

      for power in exact_powers:

        if power == -1:
          state.cache[str(power)] = utils.psd_inv_cholesky(
              state.matrix.value + identity_weight * jnp.eye(self.dim)) / scale
        else:
          matrix = state.matrix.value + identity_weight * jnp.eye(self.dim)
          state.cache[str(power)] = (
              (scale ** power) * jnp.linalg.matrix_power(matrix, power))

    return state

  def _to_dense_unscaled(self, state: "Full.State") -> Array:
    # Permute the matrix according to the parameters canonical order
    return utils.block_permuted(
        state.matrix.value,
        block_sizes=[utils.product(shape) for shape in self.parameters_shapes],
        block_order=self.parameters_canonical_order
    )


class KroneckerFactored(CurvatureBlock, abc.ABC):
  """An abstract class for approximating the block with a Kronecker product."""

  @utils.register_state_class
  class State(CurvatureBlock.State):
    """Persistent state of the block.

    Attributes:
      factors: A tuple of the moving averages of the estimated factors of the
        curvature for each axis group.
    """

    factors: Tuple[utils.WeightedMovingAverage, ...]

    @classmethod
    def from_dict(cls, dict_rep: Dict[str, Any]) -> "KroneckerFactored.State":
      class_name = dict_rep.pop("__class__", cls.__name__)
      assert class_name == cls.__name__
      return cls(
          factors=tuple(
              utils.WeightedMovingAverage.from_dict(rep)
              for rep in dict_rep["factor"]
          )
      )

  def __init__(
      self,
      layer_tag_eq: tags.LayerTagEqn,
      name: str,
      axis_groups: Optional[Sequence[Sequence[int]]] = None,
  ):
    self._layer_tag_eq = layer_tag_eq

    if axis_groups is None:
      self.axis_groups = tuple((i,) for i in range(self.array_ndim))
    else:
      self.axis_groups = tuple(tuple(g) for g in axis_groups)

    all_axis = sum(self.axis_groups, ())

    # Make sure the axis groups are sorted
    if sorted(all_axis) != list(range(min(all_axis), max(all_axis) + 1)):
      # We currently don't support out of order axis groups
      raise NotImplementedError()

    super().__init__(layer_tag_eq, name)

  @abc.abstractmethod
  def parameters_shaped_list_to_array(
      self,
      parameters_shaped_list: Sequence[Array],
  ) -> Array:
    """Combines all parameters to a single non axis grouped array."""

  @abc.abstractmethod
  def array_to_parameters_shaped_list(self, array: Array) -> Tuple[Array, ...]:
    """An inverse transformation of ``self.parameters_shaped_list_to_array``."""

  @property
  def array_shape(self) -> Shape:
    """The shape of the single non axis grouped array."""
    avals = [jnp.zeros(v.aval.shape) for v in self.parameter_variables]
    return self.parameters_shaped_list_to_array(avals).shape

  @property
  def array_ndim(self) -> int:
    """The number of dimensions of the single non axis grouped array."""
    return len(self.array_shape)

  @property
  def grouped_array_shape(self) -> Shape:
    """The shape of the single axis grouped array."""
    return tuple(
        utils.product([self.array_shape[i] for i in group])
        for group in self.axis_groups
    )

  @property
  def grouped_array_ndim(self) -> int:
    """The number of dimensions of the grouped array."""
    return len(self.axis_groups)

  def parameter_shaped_list_to_grouped_array(
      self,
      parameters_shaped_list: Sequence[Array],
  ) -> Array:
    """Combines all parameters to a single grouped array."""
    array = self.parameters_shaped_list_to_array(parameters_shaped_list)
    return jnp.reshape(array, self.grouped_array_shape)

  def grouped_array_to_parameters_shaped_list(
      self,
      grouped_array: Array,
  ) -> Tuple[Array, ...]:
    """An inverse transformation of ``self.parameter_shaped_list_to_grouped_array``."""
    array = jnp.reshape(grouped_array, self.array_shape)
    return self.array_to_parameters_shaped_list(array)

  def _init(
      self,
      rng: PRNGKey,
      exact_powers_to_cache: Set[Scalar],
      approx_powers_to_cache: Set[Scalar],
      cache_eigenvalues: bool,
  ) -> "KroneckerFactored.State":
    cache = {}
    factors = []

    for i, d in enumerate(self.grouped_array_shape):
      factors.append(
          utils.WeightedMovingAverage.zeros_array((d, d), self.dtype)
      )

      if cache_eigenvalues or exact_powers_to_cache:
        cache[f"{i}_factor_eigenvalues"] = jnp.zeros((d,), dtype=self.dtype)

      if exact_powers_to_cache:
        cache[f"{i}_factor_eigen_vectors"] = jnp.zeros((d, d), dtype=self.dtype)

      for power in approx_powers_to_cache:
        if power != -1:
          raise NotImplementedError(
              f"Approximations for power {power} is not yet implemented."
          )
        if str(power) not in cache:
          cache[str(power)] = {}

        cache[str(power)][f"{i}_factor"] = jnp.zeros((d, d), dtype=self.dtype)

    return KroneckerFactored.State(
        cache=cache,
        factors=tuple(factors),
    )

  def sync(
      self,
      state: "KroneckerFactored.State",
      pmap_axis_name: str,
  ) -> "KroneckerFactored.State":

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    for factor in state.factors:
      factor.sync(pmap_axis_name)

    return state

  def _multiply_matpower_unscaled(
      self,
      state: "KroneckerFactored.State",
      vector: Sequence[Array],
      identity_weight: Numeric,
      power: Scalar,
      exact_power: bool,
      use_cached: bool,
  ) -> Tuple[Array, ...]:
    assert len(state.factors) == len(self.axis_groups)

    vector = self.parameter_shaped_list_to_grouped_array(vector)

    if power == 1:

      factors = [f.value for f in state.factors]

      if exact_power:
        result = utils.kronecker_product_axis_mul_v(factors, vector)
        result = result + identity_weight * vector

      else:
        # If compute pi_adjusted_kronecker_factors used a more expensive matrix
        # norm in its computation, it might make sense to cache it. But we
        # currently don't do that.

        result = utils.kronecker_product_axis_mul_v(
            utils.pi_adjusted_kronecker_factors(*factors,
                                                damping=identity_weight),
            vector)

    elif exact_power:

      if use_cached:
        s = [
            state.cache[f"{i}_factor_eigenvalues"]
            for i in range(len(state.factors))
        ]
        q = [
            state.cache[f"{i}_factor_eigen_vectors"]
            for i in range(len(state.factors))
        ]

      else:
        s, q = zip(
            *[utils.safe_psd_eigh(factor.value) for factor in state.factors]
        )

      eigenvalues = utils.outer_product(*s) + identity_weight
      eigenvalues = jnp.power(eigenvalues, power)

      result = utils.kronecker_eigen_basis_axis_mul_v(q, eigenvalues, vector)

    else:

      if power != -1:
        raise NotImplementedError(
            f"Approximations for power {power} is not yet implemented."
        )

      if use_cached:
        factors = [
            state.cache[str(power)][f"{i}_factor"]
            for i in range(len(state.factors))
        ]

      else:
        factors = utils.pi_adjusted_kronecker_inverse(
            *[factor.value for factor in state.factors],
            damping=identity_weight,
        )

      result = utils.kronecker_product_axis_mul_v(factors, vector)

    return self.grouped_array_to_parameters_shaped_list(result)

  def _eigenvalues_unscaled(
      self,
      state: "KroneckerFactored.State",
      use_cached: bool,
  ) -> Array:
    assert len(state.factors) == len(self.axis_groups)

    if use_cached:
      s = [
          state.cache[f"{i}_factor_eigenvalues"]
          for i in range(len(state.factors))
      ]
    else:
      s_q = [utils.safe_psd_eigh(factor.value) for factor in state.factors]
      s, _ = zip(*s_q)

    return utils.outer_product(*s)

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: "KroneckerFactored.State",
      estimation_data: Mapping[str, Sequence[Array]],
      ema_old: Numeric,
      ema_new: Numeric,
      batch_size: Numeric,
  ) -> "KroneckerFactored.State":
    assert len(state.factors) == len(self.axis_groups)

    # This function call will return a copy of state:
    return self._update_curvature_matrix_estimate(
        state, estimation_data, ema_old, ema_new, batch_size
    )

  def _update_cache(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      state: "KroneckerFactored.State",
      identity_weight: Numeric,
      exact_powers: Numeric,
      approx_powers: Numeric,
      eigenvalues: bool,
  ) -> "KroneckerFactored.State":
    assert len(state.factors) == len(self.axis_groups)

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    scale = self.state_dependent_scale(state)
    factor_scale = jnp.power(scale, 1.0 / len(self.axis_groups))

    if eigenvalues or exact_powers:
      s_q = [utils.safe_psd_eigh(factor.value) for factor in state.factors]
      s, q = zip(*s_q)
      for i in range(len(state.factors)):
        state.cache[f"{i}_factor_eigenvalues"] = factor_scale * s[i]

        if exact_powers:
          state.cache[f"{i}_factor_eigen_vectors"] = q[i]

    for power in approx_powers:
      if power != -1:
        raise NotImplementedError(
            f"Approximations for power {power} is not yet implemented."
        )

      cache = state.cache[str(power)]
      # This computes the approximate inverse factors using the generalization
      # of the pi-adjusted inversion from the original KFAC paper.

      inv_factors = utils.pi_adjusted_kronecker_inverse(
          *[factor.value for factor in state.factors],
          damping=identity_weight,
      )
      for i in range(len(state.factors)):
        cache[f"{i}_factor"] = inv_factors[i] / factor_scale

    return state


class TwoKroneckerFactored(KroneckerFactored):
  """A Kronecker factored block for layers with weights and an optional bias."""

  def __init__(
      self,
      layer_tag_eq: tags.LayerTagEqn,
      name: str,
  ):
    super().__init__(layer_tag_eq, name, ((0,), (1,)))

  @property
  def has_bias(self) -> bool:
    """Whether this layer's equation has a bias."""
    return len(self._layer_tag_eq.invars) == 4

  def parameters_shaped_list_to_array(
      self,
      parameters_shaped_list: Sequence[Array],
  ) -> Array:
    for p, s in zip(parameters_shaped_list, self.parameters_shapes):
      assert p.shape == s

    if self.has_bias:
      w, b = parameters_shaped_list
      return jnp.concatenate([w.reshape([-1, w.shape[-1]]), b[None]], axis=0)
    else:
      # This correctly reshapes the parameters of both dense and conv2d blocks
      [w] = parameters_shaped_list
      return w.reshape([-1, w.shape[-1]])

  def array_to_parameters_shaped_list(self, array: Array) -> Tuple[Array, ...]:
    if self.has_bias:
      w, b = array[:-1], array[-1]
      return w.reshape(self.parameters_shapes[0]), b
    else:
      return tuple([array.reshape(self.parameters_shapes[0])])

  def _to_dense_unscaled(self, state: "KroneckerFactored.State") -> Array:
    assert 0 < self.number_of_parameters <= 2
    inputs_factor = state.factors[0].value

    if self.has_bias and self.parameters_canonical_order[0] != 0:
      # Permute the matrix according to the parameters canonical order
      inputs_factor = utils.block_permuted(
          state.factors[0].value,
          block_sizes=[state.factors[0].raw_value.shape[0] - 1, 1],
          block_order=(1, 0),
      )

    return jnp.kron(inputs_factor, state.factors[1].value)


class NaiveDiagonal(Diagonal):
  """Approximates the diagonal of the curvature with in the most obvious way.

  The update to the curvature estimate is computed by ``(sum_i g_i) ** 2 / N``.
  where `g_i` is the gradient of each individual data point, and ``N`` is the
  batch size.
  """

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: "NaiveDiagonal.State",
      estimation_data: Dict[str, Sequence[Array]],
      ema_old: Numeric,
      ema_new: Numeric,
      batch_size: Numeric,
  ) -> "NaiveDiagonal.State":

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    for factor, dw in zip(state.diagonal_factors,
                          estimation_data["params_tangent"]):
      factor.update(dw * dw / batch_size, ema_old, ema_new)

    return state


class NaiveFull(Full):
  """Approximates the full curvature with in the most obvious way.

  The update to the curvature estimate is computed by
  ``(sum_i g_i) (sum_i g_i)^T / N``, where ``g_i`` is the gradient of each
  individual data point, and ``N`` is the batch size.
  """

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: Full.State,
      estimation_data: Dict[str, Sequence[Array]],
      ema_old: Numeric,
      ema_new: Numeric,
      batch_size: Numeric,
  ) -> Full.State:

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    params_grads = jax.tree_util.tree_leaves(estimation_data["params_tangent"])
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

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: "Diagonal.State",
      estimation_data: Dict[str, Sequence[Array]],
      ema_old: Numeric,
      ema_new: Numeric,
      batch_size: Numeric,
  ) -> "Diagonal.State":

    # Copy this first since we mutate it later in this function.
    state = state.copy()

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

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: "Full.State",
      estimation_data: Dict[str, Sequence[Array]],
      ema_old: Numeric,
      ema_new: Numeric,
      batch_size: Numeric,
  ) -> "Full.State":

    # Copy this first since we mutate it later in this function.
    state = state.copy()

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

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: KroneckerFactored.State,
      estimation_data: Mapping[str, Sequence[Array]],
      ema_old: Numeric,
      ema_new: Numeric,
      batch_size: Numeric,
  ) -> KroneckerFactored.State:
    # Copy this first since we mutate it later in this function.
    state = state.copy()

    [x] = estimation_data["inputs"]
    [dy] = estimation_data["outputs_tangent"]

    assert utils.first_dim_is_size(batch_size, x, dy)

    if self.has_bias:
      x_one = jnp.ones_like(x[:, :1])
      x = jnp.concatenate([x, x_one], axis=1)

    input_stats = jnp.einsum("ay,az->yz", x, x) / batch_size
    output_stats = jnp.einsum("ay,az->yz", dy, dy) / batch_size

    state.factors[0].update(input_stats, ema_old, ema_new)
    state.factors[1].update(output_stats, ema_old, ema_new)

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
      image_features_map: Array,
      output_tangent: Array,
  ) -> Array:
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

    return jnp.square(vjp(output_tangent[None])[1])

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: Diagonal.State,
      estimation_data: Dict[str, Sequence[Array]],
      ema_old: Numeric,
      ema_new: Numeric,
      batch_size: Numeric,
  ) -> Diagonal.State:

    # Copy this first since we mutate it later in this function.
    state = state.copy()

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
    specify the maximum batch that that will be handled in a single iteration
    of the loop. This means that the maximum memory usage will be
    ``max_batch_size_for_vmap`` times the memory needed when calling
    ``self.conv2d_tangent_squared``. And the actual ``vmap`` will be
    called ``ceil(total_batch_size / max_batch_size_for_vmap)`` number of times
    in a loop to find the final average.

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
      inputs: Array,
      tangent_of_outputs: Array,
  ) -> Array:
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

    tangents = (vjp(tangent_of_outputs[None])[1],)

    if self.number_of_parameters == 2:
      num_axis = tangent_of_outputs.ndim - len(self.parameters_shapes[1])
      sum_axis = tuple(range(num_axis))
      tangents += (jnp.sum(tangent_of_outputs, axis=sum_axis),)

    flat_tangents = self.parameters_list_to_single_vector(tangents)

    return jnp.outer(flat_tangents, flat_tangents)

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: Full.State,
      estimation_data: Dict[str, Sequence[Array]],
      ema_old: Numeric,
      ema_new: Numeric,
      batch_size: Numeric,
  ) -> Full.State:

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    x, = estimation_data["inputs"]
    dy, = estimation_data["outputs_tangent"]
    assert utils.first_dim_is_size(batch_size, x, dy)

    matrix_update = self._averaged_tangents_outer_product(x, dy)
    state.matrix.update(matrix_update, ema_old, ema_new)

    return state


class Conv2DTwoKroneckerFactored(TwoKroneckerFactored):
  """A :class:`~TwoKroneckerFactored` block specifically for 2D convolution layers."""

  def fixed_scale(self) -> Numeric:
    return float(self.num_locations)

  @property
  def kernel_output_axis(self) -> int:
    return self._layer_tag_eq.params["dimension_numbers"].rhs_spec[0]

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
  def weights_spatial_shape(self) -> Shape:
    spatial_index = self._layer_tag_eq.params["dimension_numbers"].rhs_spec[2:]
    return tuple(self.parameters_shapes[0][i] for i in spatial_index)

  @property
  def weights_spatial_size(self) -> int:
    """The spatial filter size of the weights."""
    return utils.product(self.weights_spatial_shape)  # pytype: disable=bad-return-type  # numpy-scalars

  @property
  def inputs_spatial_shape(self) -> Shape:
    spatial_index = self._layer_tag_eq.params["dimension_numbers"].lhs_spec[2:]
    return tuple(self.inputs_shapes[0][i] for i in spatial_index)

  @property
  def num_locations(self) -> int:
    """The number of spatial locations that each filter is applied to."""
    return psm.num_conv_locations(
        self.inputs_spatial_shape,
        self.weights_spatial_shape,
        self._layer_tag_eq.params["window_strides"],
        self._layer_tag_eq.params["padding"])

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
    return self._layer_tag_eq.invars[0].aval.shape[  # pytype: disable=attribute-error
        self.inputs_channel_index]

  @property
  def num_outputs_channels(self) -> int:
    """The number of channels in the outputs to the layer."""
    return self._layer_tag_eq.invars[1].aval.shape[  # pytype: disable=attribute-error
        self.weights_output_channel_index]

  def compute_inputs_stats(
      self,
      inputs: Array,
      weighting_array: Optional[Array] = None,
  ) -> Array:
    """Computes the statistics for the inputs factor."""
    batch_size = inputs.shape[0]

    input_cov_m, input_cov_v = psm.patches_moments(
        inputs,
        kernel_spatial_shape=self.weights_spatial_shape,
        strides=self._layer_tag_eq.params["window_strides"],
        padding=self._layer_tag_eq.params["padding"],
        data_format=None,
        dim_numbers=self._layer_tag_eq.params["dimension_numbers"],
        precision=self._layer_tag_eq.params.get("precision"),
        weighting_array=weighting_array,
    )

    # Flatten the kernel and channels dimensions
    k, h, c = input_cov_v.shape
    input_cov_v = jnp.reshape(input_cov_v, (k * h * c,))
    input_cov_m = jnp.reshape(input_cov_m, (k * h * c, k * h * c))

    # Normalize by the `batch size` * `num_locations`
    normalizer = batch_size * self.num_locations
    input_cov_m = input_cov_m / normalizer
    input_cov_v = input_cov_v / normalizer

    if not self.has_bias:
      return input_cov_m

    if weighting_array is None:
      corner = jnp.ones([1], dtype=input_cov_m.dtype)
    else:
      corner = jnp.mean(weighting_array).reshape([1])

    input_cov = jnp.concatenate([input_cov_m, input_cov_v[None]], axis=0)
    input_cov_v = jnp.concatenate([input_cov_v, corner], axis=0)

    return jnp.concatenate([input_cov, input_cov_v[:, None]], axis=1)

  def compute_outputs_stats(self, tangent_of_output: Array) -> Array:
    """Computes the statistics for the outputs factor."""
    lhs_str = utils.replace_char(_ALPHABET[:4], "y", self.outputs_channel_index)
    rhs_str = utils.replace_char(_ALPHABET[:4], "z", self.outputs_channel_index)
    ein_str = f"{lhs_str},{rhs_str}->yz"
    stats = jnp.einsum(ein_str, tangent_of_output, tangent_of_output)

    # Normalize by the `batch size` * `num_locations`
    normalizer = tangent_of_output.shape[0] * self.num_locations
    return stats / normalizer

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: TwoKroneckerFactored.State,
      estimation_data: Mapping[str, Sequence[Array]],
      ema_old: Numeric,
      ema_new: Numeric,
      batch_size: Numeric,
  ) -> TwoKroneckerFactored.State:

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    [x] = estimation_data["inputs"]
    [dy] = estimation_data["outputs_tangent"]

    assert utils.first_dim_is_size(batch_size, x, dy)

    input_stats = self.compute_inputs_stats(x)
    output_stats = self.compute_outputs_stats(dy)

    state.factors[0].update(input_stats, ema_old, ema_new)
    state.factors[1].update(output_stats, ema_old, ema_new)

    return state


#   _____           _                         _  _____ _     _  __ _
#  / ____|         | |        /\             | |/ ____| |   (_)/ _| |
# | (___   ___ __ _| | ___   /  \   _ __   __| | (___ | |__  _| |_| |_
#  \___ \ / __/ _` | |/ _ \ / /\ \ | '_ \ / _` |\___ \| '_ \| |  _| __|
#  ____) | (_| (_| | |  __// ____ \| | | | (_| |____) | | | | | | | |_
# |_____/ \___\__,_|_|\___/_/    \_\_| |_|\__,_|_____/|_| |_|_|_|  \__|
#


def compatible_shapes(ref_shape, target_shape):

  if len(target_shape) > len(ref_shape):
    raise ValueError("Target shape should be smaller.")

  for ref_d, target_d in zip(reversed(ref_shape), reversed(target_shape)):
    if ref_d != target_d and target_d != 1:
      raise ValueError(f"{target_shape} is incompatible with {ref_shape}.")


def compatible_sum(tensor, target_shape, skip_axes):
  """Compute sum over ``tensor`` to achieve shape given by ``target_shape``."""

  compatible_shapes(tensor.shape, target_shape)

  n = tensor.ndim - len(target_shape)

  axis = [i + n for i, t in enumerate(target_shape)
          if t == 1 and i + n not in skip_axes]

  tensor = jnp.sum(tensor, axis=axis, keepdims=True)

  axis = [i for i in range(tensor.ndim - len(target_shape))
          if i not in skip_axes]

  return jnp.sum(tensor, axis=axis)


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

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: Diagonal.State,
      estimation_data: Dict[str, Sequence[Array]],
      ema_old: Numeric,
      ema_new: Numeric,
      batch_size: Numeric,
  ) -> Diagonal.State:

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    x, = estimation_data["inputs"]
    dy, = estimation_data["outputs_tangent"]

    assert utils.first_dim_is_size(batch_size, x, dy)

    if self.has_scale:

      assert (state.diagonal_factors[0].raw_value.shape ==
              self.parameters_shapes[0])

      scale_shape = estimation_data["params"][0].shape

      d_scale = compatible_sum(x * dy, scale_shape, skip_axes=[0])

      scale_diag_update = jnp.sum(
          d_scale * d_scale,
          axis=0, keepdims=d_scale.ndim == len(scale_shape)
      ) / batch_size

      state.diagonal_factors[0].update(scale_diag_update, ema_old, ema_new)

    if self.has_shift:

      shift_shape = estimation_data["params"][-1].shape
      d_shift = compatible_sum(dy, shift_shape, skip_axes=[0])

      shift_diag_update = jnp.sum(
          d_shift * d_shift,
          axis=0, keepdims=d_shift.ndim == len(shift_shape)
      ) / batch_size

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

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: Full.State,
      estimation_data: Dict[str, Sequence[Array]],
      ema_old: Numeric,
      ema_new: Numeric,
      batch_size: Numeric,
  ) -> Full.State:

    # Copy this first since we mutate it later in this function.
    state = state.copy()

    x, = estimation_data["inputs"]
    dy, = estimation_data["outputs_tangent"]
    assert utils.first_dim_is_size(batch_size, x, dy)

    tangents = []

    if self._has_scale:
      # Scale tangent
      scale_shape = estimation_data["params"][0].shape

      d_scale = compatible_sum(x * dy, scale_shape, skip_axes=[0])
      d_scale = d_scale.reshape([batch_size, -1])

      tangents.append(d_scale)

    if self._has_shift:
      # Shift tangent

      shift_shape = estimation_data["params"][-1].shape

      d_shift = compatible_sum(dy, shift_shape, skip_axes=[0])
      d_shift = d_shift.reshape([batch_size, -1])

      tangents.append(d_shift)

    tangents = jnp.concatenate(tangents, axis=1)
    matrix_update = jnp.matmul(tangents.T, tangents) / batch_size

    state.matrix.update(matrix_update, ema_old, ema_new)

    return state
