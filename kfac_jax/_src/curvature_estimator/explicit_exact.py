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
"""Module containing the ExplicitExactCurvature class."""
from typing import Any, Callable, Mapping
import jax
from kfac_jax._src import curvature_blocks
from kfac_jax._src import layers_and_loss_tags as tags
from kfac_jax._src import utils
from kfac_jax._src.curvature_estimator import block_diagonal

# Types for annotation
PRNGKey = utils.PRNGKey
Numeric = utils.Numeric
CurvatureBlockCtor = Callable[
    [tags.LayerTagEqn],
    curvature_blocks.CurvatureBlock
]
BlockDiagonalState = block_diagonal.BlockDiagonalCurvature.State


class ExplicitExactCurvature(block_diagonal.BlockDiagonalCurvature):
  """Explicit exact full curvature estimator class.

  This class estimates the full curvature matrix by looping over the batch
  dimension of the input data and for each single example computes an estimate
  of the curvature matrix and then averages over all examples in the input data.
  This implies that the computation scales linearly (without parallelism) with
  the batch size. The class stores the estimated curvature as a dense matrix,
  hence its memory requirement is (number of parameters)^2. If
  ``estimation_mode`` is ``fisher_exact`` or ``ggn_exact`` then this would
  compute the exact curvature, but other modes are also supported. As a result
  of looping over the input data this class needs to know the index of the batch
  in the arguments to the model function and additionally, since the loop is
  achieved through indexing, each array leaf of that argument must have the same
  first dimension size, which will be interpreted as the batch size.
  """

  def __init__(
      self,
      func: utils.Func,
      default_estimation_mode: str | None = None,
      layer_tag_to_block_ctor:
      Mapping[str, CurvatureBlockCtor] | None = None,
      auto_register_tags: bool = False,
      param_order: tuple[int, ...] | None = None,
      **kwargs: Any,
  ):
    """Initializes the curvature instance.

    Args:
      func: The model function, which should have at least one registered loss.
      default_estimation_mode: The estimation mode which to use by default when
        calling ``self.update_curvature_matrix_estimate``. If ``None`` this will
        be ``'fisher_exact'``.
      layer_tag_to_block_ctor: An optional dict mapping tags to specific classes
        of block approximations, which to override the default ones.
      auto_register_tags: This argument will be ignored since this subclass
        doesn't use automatic registration.
      param_order: An optional tuple of ints specifying the order of parameters
        (with the reference order being the one used by ``func``). If not
        specified, the reference order is used. The parameter order will
        determine the order of blocks returned by
        ``to_diagonal_block_dense_matrix``, and the order of the rows and
        columns of ``to_dense_matrix``.
      **kwargs: Addiional keyword arguments passed to the superclass
        ``BlockDiagonalCurvature``.
    """

    if layer_tag_to_block_ctor is None:
      layer_tag_to_block_ctor = dict(generic=curvature_blocks.NaiveFull)

    def retagged_func(params, *args):

      params_flat, params_treedef = jax.tree_util.tree_flatten(params)

      if param_order is not None:
        params_flat_canonical_order = [params_flat[i] for i in param_order]
        params_flat[param_order[0]] = tags.register_generic(
            *params_flat_canonical_order)

      else:
        params_flat[0] = tags.register_generic(*params_flat)

      params = jax.tree_util.tree_unflatten(params_treedef, params_flat)

      return func(params, *args)

    super().__init__(
        func=retagged_func,
        default_estimation_mode=default_estimation_mode or "ggn_curvature_prop",
        layer_tag_to_block_ctor=layer_tag_to_block_ctor,
        auto_register_tags=False,
        **kwargs,
    )

  @utils.auto_scope_method
  def update_curvature_matrix_estimate(
      self,
      state: BlockDiagonalState,
      ema_old: Numeric,
      ema_new: Numeric,
      identity_weight: Numeric,
      batch_size: Numeric,
      rng: PRNGKey,
      func_args: utils.FuncArgs,
      estimation_mode: str | None = None,
  ) -> BlockDiagonalState:

    rng = jax.random.split(rng, batch_size)

    super_ = super()

    def single_state_update(
        index: Numeric,
        state_: BlockDiagonalState
    ) -> BlockDiagonalState:

      is_first = index == 0
      args = list(func_args)

      # Index the batch for the `index` arguments.
      args[self.batch_index] = jax.tree_util.tree_map(
          lambda x: x[index][None], args[self.batch_index])

      return super_.update_curvature_matrix_estimate(
          state=state_,
          ema_old=is_first * ema_old + (1 - is_first) * 1.0,
          ema_new=ema_new / batch_size,
          identity_weight=identity_weight,
          batch_size=1,
          rng=rng[index],
          func_args=args,
          estimation_mode=estimation_mode,
      )

    return jax.lax.fori_loop(0, batch_size, single_state_update, state)
