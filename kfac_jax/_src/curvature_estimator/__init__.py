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
"""K-FAC curvature explicit and implicit estimators.

Curvature matrices are always defined in terms of some single differentiable
function of the parameters and inputs. In all cases in this module this quantity
is not the output from the model function (usually provided as argument to the
constructor of each curvature matrix), but is the sum of all losses
(weighted accordingly) which have been registered with a loss tag in the
computation graph of the model function. This quantity is referred to as the
``total_loss``.

In this module there are three curvature matrices considered:
  ``H`` - the Hessian matrix
  ``F`` - the Fisher matrix
  ``G`` - The Generalized Gauss-Newton(GGN) matrix
Vectors that are multiplied by a curvature matrix (or any of its matrix powers)
are always represented as a PyTree structure, equivalent to the parameters of
the model function. In all functions such vector is named
``parameter_structured_vector`` in the argument list.

Factors of a matrix ``M`` are defined as matrices ``B`` such that ``BB^T = M``.
If we have to left-multiply ``B`` with a vector ``v``, than ``v`` has the same
format as if we have to multiply the whole curvature matrix ``M``. However the
second size of ``B`` is not clearly defined (and can be different for the
different curvature matrices). In all methods working with factors, e.g. if we
need to right multiply ``B`` with a vector ``v`` or the result of left
multiplying ``B`` by a parameter structured vector, then the provided vector
``v`` should be a list of lists of arrays. Each element of ``v`` corresponds to
a single loss registered in the model function, and its elements should have the
shapes as the corresponding ``loss.XXX_inner_shapes`` (XXX=Hessian, Fisher or
GGN). In all function such vector is named ``loss_vectors`` in the argument
list.

See for example: www.cs.utoronto.ca/~jmartens/docs/HF_book_chapter.pdf and
https://arxiv.org/abs/1412.1193 for more information about the Hessian, Fisher
and GGN matrices and how to compute matrix-vector products.
"""

from kfac_jax._src.curvature_estimator import block_diagonal
from kfac_jax._src.curvature_estimator import curvature_estimator
from kfac_jax._src.curvature_estimator import explicit_exact
from kfac_jax._src.curvature_estimator import implicit_exact
from kfac_jax._src.curvature_estimator import optax_interface


BlockDiagonalCurvature = block_diagonal.BlockDiagonalCurvature
set_default_tag_to_block_ctor = (
    block_diagonal.set_default_tag_to_block_ctor)
get_default_tag_to_block_ctor = (
    block_diagonal.get_default_tag_to_block_ctor)
set_multi_default_tag_to_block_ctor = (
    block_diagonal.set_multi_default_tag_to_block_ctor)

StateType = curvature_estimator.StateType
CurvatureBlockCtor = curvature_estimator.CurvatureBlockCtor
CurvatureEstimator = curvature_estimator.CurvatureEstimator

ExplicitExactCurvature = explicit_exact.ExplicitExactCurvature

ImplicitExactCurvature = implicit_exact.ImplicitExactCurvature
LossFunction = implicit_exact.LossFunction
LossFunctionsTuple = implicit_exact.LossFunctionsTuple
LossFunctionsSequence = implicit_exact.LossFunctionsSequence
LossFunctionInputs = implicit_exact.LossFunctionInputs
LossFunctionInputsSequence = implicit_exact.LossFunctionInputsSequence
LossFunctionInputsTuple = implicit_exact.LossFunctionInputsTuple

OptaxPreconditioner = optax_interface.OptaxPreconditioner
OptaxPreconditionState = optax_interface.OptaxPreconditionState
