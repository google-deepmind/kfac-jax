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
"""K-FAC related utility classes and functions."""

from kfac_jax._src.utils import accumulators
from kfac_jax._src.utils import math
from kfac_jax._src.utils import misc
from kfac_jax._src.utils import parallel
from kfac_jax._src.utils import staging
from kfac_jax._src.utils import types

# types
Array = types.Array
PRNGKey = types.PRNGKey
Scalar = types.Scalar
Numeric = types.Numeric
Shape = types.Shape
DType = types.DType
PyTree = types.PyTree
ArrayTree = types.ArrayTree
TArrayTree = types.TArrayTree
Params = types.Params
Batch = types.Batch
FuncState = types.FuncState
FuncAux = types.FuncAux
PyTreeDef = types.PyTreeDef
FuncArgs = types.FuncArgs
FuncOuts = types.FuncOuts
Func = types.Func
ValueFunc = types.ValueFunc
ValueAndGradFunc = types.ValueAndGradFunc
AssumedFuncOutput = types.AssumedFuncOutput
tree_is_empty = types.tree_is_empty
abstract_objects_equal = types.abstract_objects_equal
get_float_dtype_and_check_consistency = (
    types.get_float_dtype_and_check_consistency)
del types

# misc
deserialize_state_tree = misc.deserialize_state_tree
serialize_state_tree = misc.serialize_state_tree
to_tuple_or_repeat = misc.to_tuple_or_repeat
first_dim_is_size = misc.first_dim_is_size
fake_element_from_iterator = misc.fake_element_from_iterator
default_batch_size_extractor = misc.default_batch_size_extractor
auto_scope_function = misc.auto_scope_function
auto_scope_method = misc.auto_scope_method
register_state_class = misc.register_state_class
replace_char = misc.replace_char
call_func_with_conditional_kwargs = misc.call_func_with_conditional_kwargs
Finalizable = misc.Finalizable
State = misc.State
del misc

# parallel
in_pmap = parallel.in_pmap
wrap_if_pmap = parallel.wrap_if_pmap
pmean_if_pmap = parallel.pmean_if_pmap
psum_if_pmap = parallel.psum_if_pmap
pmap_mean = parallel.pmap_mean
pmap_sum = parallel.pmap_sum
index_if_not_scalar = parallel.index_if_not_scalar
get_first = parallel.get_first
get_mean = parallel.get_mean
get_sum = parallel.get_sum
broadcast_all_local_devices = parallel.broadcast_all_local_devices
pmap_zeros_like = parallel.pmap_zeros_like
jit_zeros_like = parallel.jit_zeros_like
replicate_all_local_devices = parallel.replicate_all_local_devices
make_different_rng_key_on_all_devices = (
    parallel.make_different_rng_key_on_all_devices)
p_split = parallel.p_split
p_split_num = parallel.p_split_num
host_sync = parallel.host_sync
host_all_gather = parallel.host_all_gather
host_mean = parallel.host_mean
pmap_sync_and_divide_value = parallel.pmap_sync_and_divide_value
jit_sync_and_divide_value = parallel.jit_sync_and_divide_value
copy_array = parallel.copy_array
copy_obj = parallel.copy_obj
pmap_copy_obj = parallel.pmap_copy_obj
distribute_thunks = parallel.distribute_thunks
del parallel

# math
set_special_case_zero_inv = math.set_special_case_zero_inv
get_special_case_zero_inv = math.get_special_case_zero_inv
set_use_cholesky_inversion = math.set_use_cholesky_inversion
get_use_cholesky_inversion = math.get_use_cholesky_inversion
product = math.product
outer_product = math.outer_product
scalar_mul = math.scalar_mul
scalar_div = math.scalar_div
weighted_sum_of_objects = math.weighted_sum_of_objects
sum_of_objects = math.sum_objects
inner_product = math.inner_product
symmetric_matrix_inner_products = math.symmetric_matrix_inner_products
matrix_of_inner_products = math.matrix_of_inner_products
vector_of_inner_products = math.vector_of_inner_products
block_permuted = math.block_permuted
norm = math.norm
squared_norm = math.squared_norm
per_parameter_norm = math.per_parameter_norm
psd_inv = math.psd_inv
psd_solve = math.psd_solve
psd_solve_maybe_zero_last_idx = math.psd_solve_maybe_zero_last_idx
pi_adjusted_kronecker_factors = math.pi_adjusted_kronecker_factors
pi_adjusted_kronecker_inverse = math.pi_adjusted_kronecker_inverse
kronecker_product_axis_mul_v = math.kronecker_product_axis_mul_v
kronecker_eigen_basis_axis_mul_v = math.kronecker_eigen_basis_axis_mul_v
kronecker_product_mul_v = math.kronecker_product_mul_v
kronecker_eigen_basis_mul_v = math.kronecker_eigen_basis_mul_v
safe_psd_eigh = math.safe_psd_eigh
loop_and_parallelize_average = math.loop_and_parallelize_average
psd_matrix_norm = math.psd_matrix_norm
invert_psd_matrices = math.invert_psd_matrices
inverse_sqrt_psd_matrices = math.inverse_sqrt_psd_matrices

del math

# accumulators
WeightedMovingAverage = accumulators.WeightedMovingAverage
MultiChunkAccumulator = accumulators.MultiChunkAccumulator
del accumulators

# staged
staged = staging.staged
WithStagedMethods = staging.WithStagedMethods
del staging
