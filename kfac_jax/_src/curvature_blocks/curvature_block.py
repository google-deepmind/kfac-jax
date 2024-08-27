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
"""Module containing the abstract base class for curvature blocks."""

import abc
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import jax.scipy
from kfac_jax._src import layers_and_loss_tags as tags
from kfac_jax._src import tag_graph_matcher as tgm
from kfac_jax._src import tracer
from kfac_jax._src import utils
from kfac_jax._src.curvature_blocks import utils as cb_utils
import numpy as np


# Types for annotation
Array = utils.Array
Scalar = utils.Scalar
Numeric = utils.Numeric
PRNGKey = utils.PRNGKey
Shape = utils.Shape
DType = utils.DType
ScalarOrSequence = Scalar | Sequence[Scalar]


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
    cache: dict[str, Array | dict[str, Array]] | None

  def __init__(self, layer_tag_eq: tags.LayerTagEqn):
    """Initializes the block.

    Args:
      layer_tag_eq: The Jax equation corresponding to the layer tag that this
        block will approximate the curvature to.
    """
    super().__init__()

    self._layer_tag_eq = layer_tag_eq

    self.finalize()

  @property
  def name(self) -> str:
    return tags.layer_eqn_name(self._layer_tag_eq)

  @property
  def layer_tag_primitive(self) -> tags.LayerTag:
    """The :class:`jax.core.Primitive` corresponding to the block's tag equation."""

    primitive = self._layer_tag_eq.primitive
    assert isinstance(primitive, tgm.tags.LayerTag)

    return primitive

  @property
  def parameter_variables(self) -> tuple[jax.core.Var, ...]:
    """The parameter variables of the underlying Jax equation."""

    param_vars = []

    for p in tags.layer_eqn_data(self._layer_tag_eq).params:

      assert isinstance(p, jax.core.Var)
      param_vars.append(p)

    return tuple(param_vars)

  @property
  def outputs_shapes(self) -> tuple[Shape, ...]:
    """The shapes of the output variables of the block's tag equation."""

    output_vars = tags.layer_eqn_data(self._layer_tag_eq).outputs

    return jax.tree.map(lambda x: x.aval.shape, output_vars)

  @property
  def inputs_shapes(self) -> tuple[Shape, ...]:
    """The shapes of the input variables of the block's tag equation."""

    input_vars = tags.layer_eqn_data(self._layer_tag_eq).inputs

    return jax.tree.map(lambda x: x.aval.shape, input_vars)

  @property
  def parameters_shapes(self) -> tuple[Shape, ...]:
    """The shapes of the parameter variables of the block's tag equation."""
    return tuple(jax.tree.map(
        lambda x: tuple(x.aval.shape), self.parameter_variables))

  @property
  def dtype(self) -> DType:
    dtypes = set(p.aval.dtype for p in self.parameter_variables)  # pytype: disable=attribute-error
    if len(dtypes) > 1:
      raise ValueError("Not all parameters are the same dtype.")
    return dtypes.pop()

  @property
  def parameters_canonical_order(self) -> tuple[int, ...]:
    """The canonical order of the parameter variables."""

    return tuple(np.argsort([p.count for p in self.parameter_variables]))

  @property
  def layer_tag_extra_params(self) -> dict[str, Any]:
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

  def scale(self, state: State, use_cache: bool) -> Numeric:
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

    # TODO(jamesmartens,botev): This way of handling state dependent scale is
    # a bit hacky and leads to complexity in other parts of the code that must
    # be aware of how this part works. Should try to replace this with something
    # better.

    if use_cache:
      return self.fixed_scale()

    return self.fixed_scale() * self.state_dependent_scale(state)

  def fixed_scale(self) -> Numeric:
    """A fixed scalar pre-factor of the curvature (e.g. constant)."""
    return 1.0

  def state_dependent_scale(self, state: State) -> Numeric:
    """A scalar pre-factor of the curvature, computed from the most fresh curvature estimate."""
    del state  # Unused
    return 1.0

  def __str__(self):
    return (f"{self.__class__.__name__}, tag name: {self.name}, "
            f"params shapes: {self.parameters_shapes!r}")

  @utils.auto_scope_method
  def init(
      self,
      rng: PRNGKey,
      exact_powers_to_cache: ScalarOrSequence | None,
      approx_powers_to_cache: ScalarOrSequence | None,
      cache_eigenvalues: bool,
  ) -> State:
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
        exact_powers_to_cache=cb_utils.to_real_set(exact_powers_to_cache),
        approx_powers_to_cache=cb_utils.to_real_set(approx_powers_to_cache),
        cache_eigenvalues=cache_eigenvalues)

  @abc.abstractmethod
  def _init(
      self,
      rng: PRNGKey,
      exact_powers_to_cache: set[Scalar],
      approx_powers_to_cache: set[Scalar],
      cache_eigenvalues: bool,
  ) -> State:
    """The non-public interface of ``init``."""

  @abc.abstractmethod
  def sync(
      self,
      state: State,
      pmap_axis_name: str,
  ) -> State:
    """Syncs the state across different devices (does not sync the cache)."""

  @utils.auto_scope_method
  def multiply_matpower(
      self,
      state: State,
      vector: Sequence[Array],
      identity_weight: Numeric,
      power: Scalar,
      exact_power: bool,
      use_cached: bool,
  ) -> tuple[Array, ...]:
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
      state: State,
      vector: Sequence[Array],
      identity_weight: Numeric,
      power: Scalar,
      exact_power: bool,
      use_cached: bool,
  ) -> tuple[Array, ...]:
    """Performs matrix-vector multiplication, ignoring ``self.scale``."""

  def multiply(
      self,
      state: State,
      vector: Sequence[Array],
      identity_weight: Numeric,
      exact_power: bool,
      use_cached: bool,
  ) -> tuple[Array, ...]:
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
      state: State,
      vector: Sequence[Array],
      identity_weight: Numeric,
      exact_power: bool,
      use_cached: bool,
  ) -> tuple[Array, ...]:
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
      state: State,
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
      state: State,
      use_cached: bool,
  ) -> Array:
    """Computes the eigenvalues for this block, ignoring `self.scale`."""

  @abc.abstractmethod
  def update_curvature_matrix_estimate(
      self,
      state: State,
      estimation_data: tracer.LayerVjpData[Array],
      ema_old: Numeric,
      ema_new: Numeric,
      identity_weight: Numeric,
      batch_size: Numeric,
  ) -> State:
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
      identity_weight: The weight of the identity added to the block's curvature
          matrix before computing the cached matrix power.
      batch_size: The batch size used in computing the values in ``info``.
    """

  @utils.auto_scope_method
  def update_cache(
      self,
      state: State,
      identity_weight: Numeric,
      exact_powers: ScalarOrSequence | None,
      approx_powers: ScalarOrSequence | None,
      eigenvalues: bool,
  ) -> State:
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
        exact_powers=cb_utils.to_real_set(exact_powers),
        approx_powers=cb_utils.to_real_set(approx_powers),
        eigenvalues=eigenvalues,
    )

  @abc.abstractmethod
  def _update_cache(
      self,
      state: State,
      identity_weight: Numeric,
      exact_powers: set[Scalar],
      approx_powers: set[Scalar],
      eigenvalues: bool,
  ) -> State:
    """The cache updating function, ignoring ``self.scale``."""

  @utils.auto_scope_method
  def to_dense_matrix(self, state: State) -> Array:
    """Returns a dense representation of the curvature matrix."""
    return self.scale(state, False) * self._to_dense_unscaled(state)

  @abc.abstractmethod
  def _to_dense_unscaled(self, state: State) -> Array:
    """A dense representation of the curvature, ignoring ``self.scale``."""

  def undamped_diagonal(self, state: State) -> tuple[Array, ...]:
    """Returns the diagonal of the undamped curvature."""
    return utils.scalar_mul(self._undamped_diagonal_unscaled(state),
                            self.scale(state, False))

  def _undamped_diagonal_unscaled(self, state: State) -> tuple[Array, ...]:
    """Returns the diagonal of the undamped curvature, ignoring ``self.scale``."""
    raise NotImplementedError()

  def norm(self, state: State, norm_type: str) -> Numeric:
    """Computes the norm of the curvature block, according to ``norm_type``."""

    return self.scale(state, False) * self._norm_unscaled(state, norm_type)

  @abc.abstractmethod
  def _norm_unscaled(
      self,
      state: State,
      norm_type: str
  ) -> Numeric:
    """Like ``norm`` but with ``self.scale`` not included."""


class ScaledIdentity(CurvatureBlock):
  """A block that assumes that the curvature is a scaled identity matrix."""

  def __init__(
      self,
      layer_tag_eq: tags.LayerTagEqn,
      scale: Numeric = 1.0,
  ):
    """Initializes the block.

    Args:
      layer_tag_eq: The Jax equation corresponding to the layer tag, that this
        block will approximate the curvature to.
      scale: The scale of the identity matrix.
    """
    self._scale = scale
    super().__init__(layer_tag_eq)

  def fixed_scale(self) -> Numeric:
    return self._scale

  def _init(
      self,
      rng: PRNGKey,
      exact_powers_to_cache: set[Scalar],
      approx_powers_to_cache: set[Scalar],
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
  ) -> tuple[Array, ...]:

    del exact_power  # Unused

    # state_dependent_scale needs to be included because it won't be by the
    # caller of this function (multiply_matpower) when use_cached=True
    scale = self.state_dependent_scale(state) if use_cached else 1.0

    identity_weight = identity_weight + scale

    if power == 1:
      return jax.tree.map(lambda x: identity_weight * x, vector)

    elif power == -1:
      return jax.tree.map(lambda x: x / identity_weight, vector)

    else:
      identity_weight = jnp.power(identity_weight, power)
      return jax.tree.map(lambda x: identity_weight * x, vector)

  def _eigenvalues_unscaled(
      self,
      state: CurvatureBlock.State,
      use_cached: bool,
  ) -> Array:
    return jnp.ones([self.dim])

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: CurvatureBlock.State,
      estimation_data: tracer.LayerVjpData[Array],
      ema_old: Numeric,
      ema_new: Numeric,
      identity_weight: Numeric,
      batch_size: Numeric,
  ) -> CurvatureBlock.State:

    return state.copy()

  def _update_cache(
      self,
      state: CurvatureBlock.State,
      identity_weight: Numeric,
      exact_powers: set[Scalar],
      approx_powers: set[Scalar],
      eigenvalues: bool,
  ) -> CurvatureBlock.State:

    return state.copy()

  def _to_dense_unscaled(self, state: CurvatureBlock.State) -> Array:
    del state  # not used
    return jnp.eye(self.dim)

  def _norm_unscaled(
      self,
      state: CurvatureBlock.State,
      norm_type: str
  ) -> Numeric:

    return utils.psd_matrix_norm(jnp.ones([self.dim]), norm_type=norm_type)
