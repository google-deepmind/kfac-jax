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
"""Testing functionalities of the curvature estimation."""

import functools

import jax
import kfac_jax
from tests import models


Array = kfac_jax.utils.Array
PRNGKey = kfac_jax.utils.PRNGKey
Shape = kfac_jax.utils.Shape
StateType = kfac_jax.curvature_estimator.StateType


NON_LINEAR_MODELS_AND_CURVATURE_TYPE = [
    model + ("ggn",) for model in models.NON_LINEAR_MODELS
] + [
    model + ("fisher",) for model in models.NON_LINEAR_MODELS
]


LINEAR_MODELS_AND_CURVATURE_TYPE = [
    model + ("ggn",) for model in models.LINEAR_MODELS
] + [
    model + ("fisher",) for model in models.LINEAR_MODELS
]


PIECEWISE_LINEAR_MODELS_AND_CURVATURE = [
    model + ("ggn",) for model in models.PIECEWISE_LINEAR_MODELS
] + [
    model + ("fisher",) for model in models.PIECEWISE_LINEAR_MODELS
]


CONV_SIZES_AND_ESTIMATION_MODES = [
    [
        dict(images=(16, 16, 3), labels=(10,)),
        1230971,
        "ggn",
    ],
    [
        dict(images=(16, 16, 3), labels=(10,)),
        1230971,
        "fisher",
    ],
]

LAYER_CHANNELS = [4, 8, 16]


@functools.partial(jax.jit, static_argnums=(0, 3, 4))
def compute_exact_approx_curvature(
    estimator: kfac_jax.CurvatureEstimator[StateType],
    rng: PRNGKey,
    func_args: kfac_jax.utils.FuncArgs,
    batch_size: int,
    curvature_type: str,
) -> StateType:
  """Computes the full Fisher matrix approximation for the estimator."""
  state = estimator.init(
      rng=rng,
      func_args=func_args,
      exact_powers_to_cache=None,
      approx_powers_to_cache=None,
      cache_eigenvalues=False,
  )
  state = estimator.update_curvature_matrix_estimate(
      state=state,
      ema_old=0.0,
      ema_new=1.0,
      identity_weight=0.0,  # This doesn't matter here.
      batch_size=batch_size,
      rng=rng,
      func_args=func_args,
      estimation_mode=f"{curvature_type}_exact",
  )
  estimator.sync(state, pmap_axis_name="i")
  return state
