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
"""Module containing the BlockDiagonalCurvature class."""
import functools
from typing import Any, Callable, Sequence, Mapping
from absl import logging
import jax
from jax import scipy
import jax.numpy as jnp
from kfac_jax._src import curvature_blocks
from kfac_jax._src import layers_and_loss_tags as tags
from kfac_jax._src import loss_functions
from kfac_jax._src import tracer
from kfac_jax._src import utils
from kfac_jax._src.curvature_estimator import curvature_estimator
import numpy as np

# Types for annotation
Array = utils.Array
PRNGKey = utils.PRNGKey
Numeric = utils.Numeric
Scalar = utils.Scalar
Shape = utils.Shape
CurvatureBlockCtor = Callable[
    [tags.LayerTagEqn],
    curvature_blocks.CurvatureBlock
]

_DEFAULT_TAG_TO_BLOCK_CTOR: dict[str, CurvatureBlockCtor] = dict(
    dense=curvature_blocks.DenseTwoKroneckerFactored,
    conv2d=curvature_blocks.Conv2DTwoKroneckerFactored,
    generic=curvature_blocks.NaiveDiagonal,
    scale_and_shift=curvature_blocks.ScaleAndShiftDiagonal,
    repeated_dense=curvature_blocks.RepeatedDenseKroneckerFactored,
)


def get_default_tag_to_block_ctor(
    tag_name: str
) -> CurvatureBlockCtor | None:
  """Returns the default curvature block constructor for the give tag name."""
  if tag_name.endswith("_tag"):
    raise ValueError(
        "You are using the old style of tag names. Remove the '_tag' suffix."
    )
  return _DEFAULT_TAG_TO_BLOCK_CTOR.get(tag_name)


def set_default_tag_to_block_ctor(
    tag_name: str,
    block_ctor: CurvatureBlockCtor
) -> None:
  """Sets the default curvature block constructor for the given tag."""
  if tag_name.endswith("_tag"):
    raise ValueError(
        "You are using the old style of tag names. Remove the '_tag' suffix."
    )
  _DEFAULT_TAG_TO_BLOCK_CTOR[tag_name] = block_ctor


def set_multi_default_tag_to_block_ctor(
    tags_to_block_ctor: Mapping[str, CurvatureBlockCtor]
):
  _DEFAULT_TAG_TO_BLOCK_CTOR.update(tags_to_block_ctor)


class BlockDiagonalCurvature(
    curvature_estimator.CurvatureEstimator["BlockDiagonalCurvature.State"]):
  """Block diagonal curvature estimator class.

    Supports for the following estimation modes:
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
  """

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
    blocks_states: tuple[curvature_blocks.CurvatureBlock.State, ...]

  def __init__(
      self,
      func: utils.Func,
      default_estimation_mode: str | None = None,
      layer_tag_to_block_ctor:
      Mapping[str, CurvatureBlockCtor] | None = None,
      index_to_block_ctor:
      Mapping[tuple[int, ...], CurvatureBlockCtor] | None = None,
      auto_register_tags: bool = True,
      distributed_multiplies: bool = True,
      distributed_cache_updates: bool = True,
      num_samples: int = 1,
      should_vmap_samples: bool = False,
      auto_register_kwargs: dict[str, Any] | None = None,
      **kwargs: Any,
  ):
    """Initializes the BlockDiagonalCurvature instance.

    Args:
      func: The model function, which should have at least one registered loss.
      default_estimation_mode: The estimation mode which to use by default when
        calling ``self.update_curvature_matrix_estimate``. If ``None`` this will
        be ``'ggn_curvature_prop'``.
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
      num_samples: Number of samples (per case) to use when computing stochastic
        curvature matrix estimates. This option is only used when
        ``estimation_mode == 'fisher_gradients'`` or ``estimation_mode ==
        '[fisher,ggn]_curvature_prop'``.
      should_vmap_samples: Whether to use ``jax.vmap`` to compute samples
        when ``num_samples > 1``.
      auto_register_kwargs: Keyword arguments to pass to into the
        layer auto-registration function.
      **kwargs: Addiional keyword arguments passed to the superclass
        ``CurvatureEstimator``.
    """

    super().__init__(
        func=func,
        default_estimation_mode=default_estimation_mode or "ggn_curvature_prop",
        **kwargs,
    )

    self._index_to_block_ctor = index_to_block_ctor or dict()
    self._layer_tag_to_block_ctor = layer_tag_to_block_ctor or dict()
    self._auto_register_tags = auto_register_tags
    self._auto_register_kwargs = auto_register_kwargs or {}
    self._vjp, self._jaxpr_extractor = tracer.layer_tags_vjp(
        func=func,
        params_index=self.params_index,
        auto_register_tags=auto_register_tags,
        **self._auto_register_kwargs
    )

    # Initialized during finalization
    self._jaxpr: tracer.ProcessedJaxpr | None = None
    self._blocks: tuple[curvature_blocks.CurvatureBlock, ...] | None = None

    self._distributed_multiplies = distributed_multiplies
    self._distributed_cache_updates = distributed_cache_updates

    self._num_samples = num_samples
    self._should_vmap_samples = should_vmap_samples

  @property
  def valid_estimation_modes(self) -> tuple[str, ...]:
    """The valid estimation modes for this estimator."""
    return ("fisher_gradients", "fisher_empirical", "fisher_exact",
            "fisher_curvature_prop", "ggn_exact", "ggn_curvature_prop")

  def _check_finalized(self):
    if not self.finalized:
      raise ValueError("The estimator has not been finalized. Call `init` or "
                       "`finalize` first.")

  def _create_blocks(self):
    """Creates all the curvature blocks instances in ``self._blocks``."""

    assert self._jaxpr is not None

    blocks_list = []

    for tag_eqn, idx in zip(self._jaxpr.layer_tags, self._jaxpr.layer_indices):
      meta = tag_eqn.params.get("meta")
      assert meta is not None and isinstance(meta, tags.LayerMetaData)
      assert not meta.nesting

      # Correctly get the block class
      if idx in self._index_to_block_ctor:
        cls = self._index_to_block_ctor[idx]

      elif meta.variant in self._layer_tag_to_block_ctor:
        cls = self._layer_tag_to_block_ctor[meta.variant]

      else:
        cls = get_default_tag_to_block_ctor(meta.variant)
        if cls is None:
          raise ValueError(
              "Did not find anywhere a block class for layer tag variant "
              f"{meta.variant}."
          )

      blocks_list.append(cls(tag_eqn))

    self._blocks = tuple(blocks_list)

  @property
  def blocks(self) -> tuple[curvature_blocks.CurvatureBlock, ...] | None:
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
    assert self._jaxpr is not None
    return self._jaxpr

  @property
  def params_structure_vector_of_indices(self) -> utils.Params:
    """A tree structure with parameters replaced by their indices."""
    return jax.tree_util.tree_unflatten(
        self.jaxpr.params_tree, range(len(self.jaxpr.params_vars_flat))
    )

  @property
  def indices_to_block_map(
      self
  ) -> Mapping[tuple[int, ...], curvature_blocks.CurvatureBlock]:
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
    params_block_index: list[int | None] = [None] * self.num_params_variables

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

  @property
  def param_order(self):
    # is there a nicer way to do this?
    params_vars = self.params_vector_to_blocks_vectors(self.jaxpr.params_vars)
    return np.argsort([p.count for p in jax.tree_util.tree_leaves(params_vars)])

  def log_registrations(self):
    if self._blocks is None:
      raise ValueError(
          "You must initialize the estimator before calling this method."
      )

    logging.info("BlockDiagonalCurvature blocks:")
    for block in self._blocks:
      logging.info(str(block))
    logging.info("=" * 50)

  def params_vector_to_blocks_vectors(
      self,
      parameter_structured_vector: utils.Params,
  ) -> tuple[tuple[Array, ...], ...]:
    """Splits the parameters to values for each corresponding block."""

    params_values_flat = jax.tree_util.tree_leaves(parameter_structured_vector)
    blocks_vectors: list[tuple[Array, ...]] = []

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

    values_flat: list[Array | None] = [None] * self.num_params_variables

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
    self._jaxpr = self._jaxpr_extractor(func_args)
    self._create_blocks()
    self.log_registrations()

  @utils.auto_scope_method
  def init(
      self,
      rng: PRNGKey,
      func_args: utils.FuncArgs,
      exact_powers_to_cache: curvature_blocks.ScalarOrSequence | None,
      approx_powers_to_cache: curvature_blocks.ScalarOrSequence | None,
      cache_eigenvalues: bool = False,
  ) -> State:

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
      state: State,
      pmap_axis_name: str | None,
  ) -> State:

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
      state: State,
      pmap_axis_name: str | None,
  ) -> State:

    return jax.lax.cond(
        state.synced,
        lambda s: s,
        functools.partial(self._sync_state, pmap_axis_name=pmap_axis_name),
        state,
    )

  @utils.auto_scope_method
  def multiply_matpower(
      self,
      state: State,
      parameter_structured_vector: utils.Params,
      identity_weight: Numeric | Sequence[Numeric],
      power: Scalar,
      exact_power: bool,
      use_cached: bool,
      pmap_axis_name: str | None,
      norm_to_scale_identity_weight_per_block: str | None = None,
  ) -> utils.Params:

    blocks_vectors = self.params_vector_to_blocks_vectors(
        parameter_structured_vector)

    identity_weight = utils.to_tuple_or_repeat(identity_weight, self.num_blocks)

    def make_thunk(block, block_state, block_vector, block_identity_weight):

      def thunk():

        weight = block_identity_weight

        if (norm_to_scale_identity_weight_per_block is not None
            and norm_to_scale_identity_weight_per_block != "none"):

          weight *= block.norm(
              block_state, norm_to_scale_identity_weight_per_block)

        return block.multiply_matpower(
            state=block_state,
            vector=block_vector,
            identity_weight=weight,
            power=power,
            exact_power=exact_power,
            use_cached=use_cached,
        )

      return thunk

    thunks = []
    for block, block_state, block_vector, block_identity_weight in zip(
        self.blocks, state.blocks_states, blocks_vectors, identity_weight):

      thunks.append(
          make_thunk(block, block_state, block_vector, block_identity_weight))

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
      state: State,
      use_cached: bool,
  ) -> tuple[Array, ...]:
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
      state: State,
      use_cached: bool,
  ) -> Array:

    blocks_eigenvalues = self.block_eigenvalues(state, use_cached)
    return jnp.concatenate(blocks_eigenvalues, axis=0)

  # Helper function that updates the blocks given a vjp vector
  def _update_blocks(
      self,
      blocks_info,
      state,
      ema_old: Numeric,
      ema_new: Numeric,
      identity_weight: Numeric,
      batch_size: int,
  ):

    assert len(blocks_info) == self.num_blocks

    new_state = []
    for block, block_state, block_info in zip(
        self.blocks, state.blocks_states, blocks_info):

      new_state.append(
          block.update_curvature_matrix_estimate(
              block_state,
              block_info,
              ema_old=ema_old,
              ema_new=ema_new,
              identity_weight=identity_weight,
              batch_size=batch_size,
          )
      )

    return BlockDiagonalCurvature.State(
        synced=jnp.asarray(False),
        blocks_states=tuple(new_state),
    )

  def _maybe_do_multiple_updates(self, update_func, state, rng, ema_old):

    if self._num_samples > 1 and self._should_vmap_samples:

      def f(rng_i):
        return update_func(state, rng_i, ema_old)

      states = jax.vmap(f)(jax.random.split(rng, self._num_samples))

      # This implementation is quick and hacky and might break in the future.
      # It works by averaging the states only for their floating point leaves,
      # which are assumed to be statistics tensors.
      return jax.tree_util.tree_map(
          lambda x: (  # pylint: disable=g-long-lambda
              jnp.mean(x, axis=0) if jnp.issubdtype(x.dtype, jnp.floating)
              else x[0]),
          states)

    elif self._num_samples > 1:

      def f(carry, rng_i):

        state_i, ema_old_i = carry
        new_state_i = update_func(state_i, rng_i, ema_old_i)

        return (new_state_i, jnp.ones_like(ema_old_i)), None

      (new_state, _), _ = jax.lax.scan(
          f,
          init=(state, jnp.asarray(ema_old)),
          xs=jax.random.split(rng, self._num_samples)
      )
      return new_state

    elif self._num_samples == 1:
      return update_func(state, rng, ema_old)

    else:
      # Don't update the preconditioner at all.
      return state

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: State,
      ema_old: Numeric,
      ema_new: Numeric,
      identity_weight: Numeric,
      batch_size: Numeric,
      rng: PRNGKey,
      func_args: utils.FuncArgs,
      estimation_mode: str | None = None,
  ) -> State:

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

    if estimation_mode == "fisher_gradients":

      def update_func(state_i, rng_i, ema_old_i):

        keys = jax.random.split(
            rng_i, len(losses)) if len(losses) > 1 else [rng_i]

        vjp_vec = tuple(
            loss.grad_of_evaluate_on_sample(key, coefficient_mode="sqrt")
            for loss, key in zip(losses, keys))

        return self._update_blocks(
            losses_vjp(vjp_vec),
            state=state_i,
            ema_old=ema_old_i,
            ema_new=ema_new,
            identity_weight=identity_weight,
            batch_size=batch_size,
        )

      return self._maybe_do_multiple_updates(update_func, state, rng, ema_old)

    elif estimation_mode == "fisher_empirical":

      vjp_vec = tuple(
          loss.grad_of_evaluate(None, coefficient_mode="regular")
          for loss in losses)

      return self._update_blocks(
          losses_vjp(vjp_vec),
          state=state,
          ema_old=ema_old,
          ema_new=ema_new,
          identity_weight=identity_weight,
          batch_size=batch_size,
      )

    elif estimation_mode in ("fisher_curvature_prop", "ggn_curvature_prop"):

      def update_func(state_i, rng_i, ema_old_i):

        keys = jax.random.split(
            rng_i, len(losses)) if len(losses) > 1 else [rng_i]

        vjp_vec = []

        for loss, key in zip(losses, keys):

          if estimation_mode == "fisher_curvature_prop":
            shape = loss.fisher_factor_inner_shape
            random_sign = jax.random.rademacher(key, shape=shape)
            vjp_vec.append(loss.multiply_fisher_factor(random_sign))

          else:
            shape = loss.ggn_factor_inner_shape
            random_sign = jax.random.rademacher(key, shape=shape)
            vjp_vec.append(loss.multiply_ggn_factor(random_sign))

        return self._update_blocks(
            losses_vjp(tuple(vjp_vec)),
            state=state_i,
            ema_old=ema_old_i,
            ema_new=ema_new,
            identity_weight=identity_weight,
            batch_size=batch_size,
        )

      return self._maybe_do_multiple_updates(update_func, state, rng, ema_old)

    elif estimation_mode in ("fisher_exact", "ggn_exact"):

      zero_tangents = jax.tree_util.tree_map(
          jnp.zeros_like, list(loss.parameter_dependants for loss in losses))

      if estimation_mode == "fisher_exact":
        shapes = [l.fisher_factor_inner_shape[1:] for l in losses]
      else:
        shapes = [l.ggn_factor_inner_shape[1:] for l in losses]

      # For now we support only inner shapes of 1 dimension, hence below the
      # (loss_num_indices,).
      assert all(len(s) == 1 for s in shapes)

      total_num_indices = sum(sum(s) for s in shapes)

      # This doesn't affect how the averaging is done except how the new stats
      # are weighted vs the old stats (which if they also used this correction
      # will just be 1:1).
      ema_new = ema_new / total_num_indices

      # This loop should probably be converted into a JAX loop for the sake of
      # efficient compilation.
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

          # This will affect the tangent stat but not the activation stats. We
          # do it because we want an average over loss indices for the
          # activation stats, but a sum for the tangent stats.
          vjp_vec[i] = jax.tree_util.tree_map(
              lambda x: x * jnp.sqrt(total_num_indices), vjp_vec[i])

          state = self._update_blocks(
              losses_vjp(tuple(vjp_vec)),
              state=state,
              ema_old=ema_old,
              ema_new=ema_new,
              identity_weight=identity_weight,
              batch_size=batch_size,
          )

          ema_old = 1.0

      return state

    else:
      raise ValueError(f"Unrecognised estimation_mode {estimation_mode}.")

  @utils.auto_scope_method
  def update_cache(
      self,
      state: State,
      identity_weight: Numeric | Sequence[Numeric],
      exact_powers: curvature_blocks.ScalarOrSequence | None,
      approx_powers: curvature_blocks.ScalarOrSequence | None,
      eigenvalues: bool,
      pmap_axis_name: str | None,
      norm_to_scale_identity_weight_per_block: str | None = None,
  ) -> State:

    identity_weight = utils.to_tuple_or_repeat(identity_weight, self.num_blocks)

    def make_thunk(block, block_state, block_identity_weight):

      def thunk():

        weight = block_identity_weight

        if (norm_to_scale_identity_weight_per_block is not None
            and norm_to_scale_identity_weight_per_block != "none"):

          weight *= block.norm(
              block_state, norm_to_scale_identity_weight_per_block)

        return block.update_cache(
            state=block_state,
            identity_weight=block_identity_weight,
            exact_powers=exact_powers,
            approx_powers=approx_powers,
            eigenvalues=eigenvalues,
        )

      return thunk

    thunks = []
    for block, block_state, block_identity_weight in zip(self.blocks,
                                                         state.blocks_states,
                                                         identity_weight):

      thunks.append(make_thunk(block, block_state, block_identity_weight))

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

  def undamped_diagonal(self, state: State) -> utils.Params:
    result = tuple(
        block.undamped_diagonal(block_state)
        for block, block_state in zip(self.blocks, state.blocks_states))

    return self.blocks_vectors_to_params_vector(result)

  @utils.auto_scope_method
  def to_diagonal_block_dense_matrix(self, state: State) -> tuple[Array, ...]:
    """Returns a tuple of arrays with explicit dense matrices of each block."""
    return tuple(block.to_dense_matrix(block_state) for block, block_state in
                 zip(self.blocks, state.blocks_states))

  @utils.auto_scope_method
  def to_dense_matrix(self, state: State) -> Array:
    return scipy.linalg.block_diag(*self.to_diagonal_block_dense_matrix(state))
