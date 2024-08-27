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
from kfac_jax._src.curvature_blocks import curvature_block
from kfac_jax._src.curvature_blocks import diagonal
from kfac_jax._src.curvature_blocks import full
from kfac_jax._src.curvature_blocks import kronecker_factored
from kfac_jax._src.curvature_blocks import tnt
from kfac_jax._src.curvature_blocks import utils

CurvatureBlock = curvature_block.CurvatureBlock
ScaledIdentity = curvature_block.ScaledIdentity
ScalarOrSequence = curvature_block.ScalarOrSequence

Diagonal = diagonal.Diagonal
Full = full.Full
KroneckerFactored = kronecker_factored.KroneckerFactored
NaiveDiagonal = diagonal.NaiveDiagonal
NaiveFull = full.NaiveFull
NaiveTNT = tnt.NaiveTNT
DenseDiagonal = diagonal.DenseDiagonal
DenseFull = full.DenseFull
DenseTwoKroneckerFactored = kronecker_factored.DenseTwoKroneckerFactored
RepeatedDenseKroneckerFactored = (
    kronecker_factored.RepeatedDenseKroneckerFactored)
DenseTNT = tnt.DenseTNT
Conv2DDiagonal = diagonal.Conv2DDiagonal
Conv2DFull = full.Conv2DFull
Conv2DTwoKroneckerFactored = kronecker_factored.Conv2DTwoKroneckerFactored
Conv2DTNT = tnt.Conv2DTNT
ScaleAndShiftDiagonal = diagonal.ScaleAndShiftDiagonal
ScaleAndShiftFull = full.ScaleAndShiftFull

set_max_parallel_elements = utils.set_max_parallel_elements
get_max_parallel_elements = utils.get_max_parallel_elements
set_default_eigen_decomposition_threshold = (
    utils.set_default_eigen_decomposition_threshold)
get_default_eigen_decomposition_threshold = (
    utils.get_default_eigen_decomposition_threshold)

