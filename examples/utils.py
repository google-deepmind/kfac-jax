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
"""Utilities for kfac_jax examples."""

import jax
import kfac_jax

ArrayTree = kfac_jax.utils.ArrayTree
Mask = kfac_jax.utils.Mask
MaskOrFn = kfac_jax.utils.MaskOrFn


def haiku_params_mask(
    params: ArrayTree,
    exclude_batch_norm: bool = True,
    exclude_biases: bool = True,
) -> Mask:
  """Returns a boolean mask PyTree for Haiku parameters.

  Args:
    params: The Haiku model parameters.
    exclude_batch_norm: If True, excludes parameters from modules whose names
      contain "batchnorm" from regularization.
    exclude_biases: If True, excludes parameters named "b" or "bias" from
      regularization.

  Returns:
    A PyTree of booleans matching the structure of `params`.
  """
  def is_regularized(path, _):

    keys = [
        str(getattr(k, "key", getattr(k, "name", getattr(k, "idx", str(k)))))
        for k in path
    ]

    module_name = "/".join(keys[:-1]).lower() if len(keys) > 1 else ""
    param_name = keys[-1].lower() if keys else ""

    if exclude_batch_norm and "batchnorm" in module_name:
      return False

    if exclude_biases and param_name in ("b", "bias"):
      return False

    return True

  return jax.tree.map_with_path(is_regularized, params)


def haiku_exclude_batch_norm_and_biases(params: ArrayTree) -> Mask:
  """A mask function for Haiku parameters that excludes batch norm and biases."""
  return haiku_params_mask(params, exclude_batch_norm=True, exclude_biases=True)
