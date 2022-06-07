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
"""kfac-jax public APIs."""

from kfac_jax._src import curvature_blocks
from kfac_jax._src import curvature_estimator
from kfac_jax._src import layers_and_loss_tags
from kfac_jax._src import loss_functions
from kfac_jax._src import optimizer
from kfac_jax._src import patches_second_moment
from kfac_jax._src import tag_graph_matcher
from kfac_jax._src import tracer
from kfac_jax._src import utils


__version__ = "0.0.2"

# Patches Second Moments
patches_moments = patches_second_moment.patches_moments
patches_moments_explicit = patches_second_moment.patches_moments_explicit

# Layers and loss tags
LossTag = layers_and_loss_tags.LossTag
LayerTag = layers_and_loss_tags.LayerTag
register_generic = layers_and_loss_tags.register_generic
register_dense = layers_and_loss_tags.register_dense
register_conv2d = layers_and_loss_tags.register_conv2d
register_scale_and_shift = layers_and_loss_tags.register_scale_and_shift

# Tag graph matcher
auto_register_tags = tag_graph_matcher.auto_register_tags

# Tracer
ProcessedJaxpr = tracer.ProcessedJaxpr
loss_tags_vjp = tracer.loss_tags_vjp
loss_tags_jvp = tracer.loss_tags_jvp
loss_tags_hvp = tracer.loss_tags_hvp
layer_tags_vjp = tracer.layer_tags_vjp

# Loss functions
LossFunction = loss_functions.LossFunction
NegativeLogProbLoss = loss_functions.NegativeLogProbLoss
DistributionNegativeLogProbLoss = loss_functions.DistributionNegativeLogProbLoss
NormalMeanNegativeLogProbLoss = loss_functions.NormalMeanNegativeLogProbLoss
NormalMeanVarianceNegativeLogProbLoss = loss_functions.NormalMeanVarianceNegativeLogProbLoss
MultiBernoulliNegativeLogProbLoss = loss_functions.MultiBernoulliNegativeLogProbLoss
CategoricalLogitsNegativeLogProbLoss = loss_functions.CategoricalLogitsNegativeLogProbLoss
OneHotCategoricalLogitsNegativeLogProbLoss = loss_functions.OneHotCategoricalLogitsNegativeLogProbLoss
register_sigmoid_cross_entropy_loss = loss_functions.register_sigmoid_cross_entropy_loss
register_multi_bernoulli_predictive_distribution = loss_functions.register_multi_bernoulli_predictive_distribution
register_softmax_cross_entropy_loss = loss_functions.register_softmax_cross_entropy_loss
register_categorical_predictive_distribution = loss_functions.register_categorical_predictive_distribution
register_squared_error_loss = loss_functions.register_squared_error_loss
register_normal_predictive_distribution = loss_functions.register_normal_predictive_distribution

# Curvature blocks
CurvatureBlock = curvature_blocks.CurvatureBlock
ScaledIdentity = curvature_blocks.ScaledIdentity
Diagonal = curvature_blocks.Diagonal
Full = curvature_blocks.Full
TwoKroneckerFactored = curvature_blocks.TwoKroneckerFactored
NaiveDiagonal = curvature_blocks.NaiveDiagonal
NaiveFull = curvature_blocks.NaiveFull
DenseDiagonal = curvature_blocks.DenseDiagonal
DenseFull = curvature_blocks.DenseFull
DenseTwoKroneckerFactored = curvature_blocks.DenseTwoKroneckerFactored
Conv2DDiagonal = curvature_blocks.Conv2DDiagonal
Conv2DFull = curvature_blocks.Conv2DFull
Conv2DTwoKroneckerFactored = curvature_blocks.Conv2DTwoKroneckerFactored
ScaleAndShiftDiagonal = curvature_blocks.ScaleAndShiftDiagonal
ScaleAndShiftFull = curvature_blocks.ScaleAndShiftFull
set_max_parallel_elements = curvature_blocks.set_max_parallel_elements
get_max_parallel_elements = curvature_blocks.get_max_parallel_elements
set_default_eigen_decomposition_threshold = curvature_blocks.set_default_eigen_decomposition_threshold
get_default_eigen_decomposition_threshold = curvature_blocks.get_default_eigen_decomposition_threshold

# Curvature estimators
CurvatureEstimator = curvature_estimator.CurvatureEstimator
BlockDiagonalCurvature = curvature_estimator.BlockDiagonalCurvature
ExplicitExactCurvature = curvature_estimator.ExplicitExactCurvature
ImplicitExactCurvature = curvature_estimator.ImplicitExactCurvature
set_default_tag_to_block_ctor = curvature_estimator.set_default_tag_to_block_ctor
get_default_tag_to_block_ctor = curvature_estimator.get_default_tag_to_block_ctor

# Optimizers
Optimizer = optimizer.Optimizer


__all__ = (
    # Modules
    "utils",
    "patches_second_moment",
    "layers_and_loss_tags",
    "loss_functions",
    "tag_graph_matcher",
    "tracer",
    "curvature_blocks",
    "curvature_estimator",
    "optimizer",
    # Patches second moments
    "patches_moments",
    "patches_moments_explicit",
    # Layer and loss tags
    "LossTag",
    "LayerTag",
    "register_generic",
    "register_dense",
    "register_conv2d",
    "register_scale_and_shift",
    # Tag graph matcher
    "auto_register_tags",
    # Tracer
    "ProcessedJaxpr",
    "loss_tags_vjp",
    "loss_tags_jvp",
    "loss_tags_hvp",
    "layer_tags_vjp",
    # Loss functions
    "LossFunction",
    "NegativeLogProbLoss",
    "DistributionNegativeLogProbLoss",
    "NormalMeanNegativeLogProbLoss",
    "NormalMeanVarianceNegativeLogProbLoss",
    "MultiBernoulliNegativeLogProbLoss",
    "CategoricalLogitsNegativeLogProbLoss",
    "OneHotCategoricalLogitsNegativeLogProbLoss",
    "register_sigmoid_cross_entropy_loss",
    "register_multi_bernoulli_predictive_distribution",
    "register_softmax_cross_entropy_loss",
    "register_categorical_predictive_distribution",
    "register_squared_error_loss",
    "register_normal_predictive_distribution",
    # Curvature blocks
    "CurvatureBlock",
    "ScaledIdentity",
    "Diagonal",
    "Full",
    "TwoKroneckerFactored",
    "NaiveDiagonal",
    "NaiveFull",
    "DenseDiagonal",
    "DenseFull",
    "DenseTwoKroneckerFactored",
    "Conv2DDiagonal",
    "Conv2DFull",
    "Conv2DTwoKroneckerFactored",
    "ScaleAndShiftDiagonal",
    "ScaleAndShiftFull",
    "set_max_parallel_elements",
    "get_max_parallel_elements",
    "set_default_eigen_decomposition_threshold",
    "get_default_eigen_decomposition_threshold",
    # Estimators
    "CurvatureEstimator",
    "BlockDiagonalCurvature",
    "ExplicitExactCurvature",
    "ImplicitExactCurvature",
    "set_default_tag_to_block_ctor",
    "get_default_tag_to_block_ctor",
    # Optimizers
    "Optimizer",
)

#  _________________________________________
# / Please don't use symbols in `_src` they \
# \ are not part of the KFAC Jax public API./
#  -----------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#
try:
  del _src  # pylint: disable=undefined-variable
except NameError:
  pass
