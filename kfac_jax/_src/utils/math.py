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
"""K-FAC utilities for various mathematical operations."""
import functools
import string
from typing import Callable, Optional, Sequence, Iterable, TypeVar, Tuple, Union

import jax
from jax import lax
from jax.experimental.sparse import linalg as experimental_splinalg
import jax.numpy as jnp
from jax.scipy import linalg

from kfac_jax._src.utils import types

import numpy as np
import optax
import tree


Array = types.Array
Numeric = types.Numeric
PRNGKey = types.PRNGKey
ArrayTree = types.ArrayTree
TArrayTree = types.TArrayTree
TNumeric = TypeVar("TNumeric", bound=Numeric)

_ALPHABET = string.ascii_lowercase

# If true we use a special case formula for when a block has one or more zero
# factors.
_SPECIAL_CASE_ZERO_INV: bool = True

# Cholesky inverses are deterministic on GPUs and somewhat faster to compute,
# but tend to perform a bit worse and not tolerate very low damping values.
# If disabling Cholesky inverses and using GPUs, one must make sure to set
# distributed_inverses=True, since otherwise the devices can potentially become
# out of sync, which is a silent and very serious failure
_USE_CHOLESKY_INVERSION: bool = False


def set_special_case_zero_inv(value: bool):
  """Sets whether `pi_adjusted_inverse` handles zero and nan matrices."""
  global _SPECIAL_CASE_ZERO_INV
  _SPECIAL_CASE_ZERO_INV = value


def get_special_case_zero_inv() -> bool:
  """Returns whether `pi_adjusted_inverse` handles zero and nan matrices."""
  return _SPECIAL_CASE_ZERO_INV


def set_use_cholesky_inversion(value: bool):
  """Sets whether `pi_adjusted_inverse` handles zero and nan matrices."""
  global _USE_CHOLESKY_INVERSION
  _USE_CHOLESKY_INVERSION = value


def get_use_cholesky_inversion() -> bool:
  """Returns whether `pi_adjusted_inverse` handles zero and nan matrices."""
  return _USE_CHOLESKY_INVERSION


def product(iterable_object: Iterable[TNumeric]) -> TNumeric:
  """Computes the product of all elements in the iterable."""
  x = 1

  for element in iterable_object:
    x = x * element

  return x


def outer_product(*arrays: Array) -> Array:
  """Computes the outer product of an arbitrary number of vectors."""
  if not all(a.ndim == 1 for a in arrays):
    raise ValueError("All arrays must be vectors.")
  in_str = ",".join(_ALPHABET[:len(arrays)])
  out_str = _ALPHABET[:len(arrays)]
  return jnp.einsum(f"{in_str}->{out_str}", *arrays)


def scalar_mul(obj: TArrayTree, scalar: Numeric) -> TArrayTree:
  """Multiplies all PyTree leaves of the object by the provided scalar."""
  # The check below is in its current form because of how `jax.jit` tracing
  # mechanism work. If we use `scalar == 1` and `scalar` is an array,  inside a
  # `jit` context, jax will raise an error, since you are not allowed to use
  # abstract values in concrete boolean statements, like native python
  # if/while/for constructs.
  if isinstance(scalar, types.SCALAR_TYPES) and scalar == 1.0:
    return obj

  return jax.tree_util.tree_map(lambda x: x * scalar, obj)


def scalar_div(obj: TArrayTree, scalar: Numeric) -> TArrayTree:
  """Divides all PyTree leaves of the object by the provided scalar."""
  # The check below is in its current form because of how `jax.jit` tracing
  # mechanism work. If we use `scalar == 1` and `scalar` is an array,  inside a
  # `jit` context, jax will raise an error, since you are not allowed to use
  # abstract values in concrete boolean statements, like native python
  # if/while/for constructs.
  if isinstance(scalar, types.SCALAR_TYPES) and scalar == 1.0:
    return obj

  return jax.tree_util.tree_map(lambda x: x / scalar, obj)


def weighted_sum_of_objects(
    objects: Sequence[TArrayTree],
    coefficients: Sequence[Numeric],
) -> TArrayTree:
  """Computes a weighted sum of the objects'.

  The function computes `sum_i coefficients[i] * objects[i]`. All objects must
  have the same PyTree structure, and PyTree leaves in equivalent positions must
  have the same shape.

  Args:
    objects: The sequence of objects to be summed together.
    coefficients: The coefficients corresponding to each object instance.

  Returns:
    An object, representing the weighted sum, of the same type as the inputs.
  """
  if len(objects) != len(coefficients):
    raise ValueError("The number of coefficients must equal the number of "
                     "objects.")
  if not objects:
    raise ValueError("The objects' sequences can not be empty.")

  accumulator = scalar_mul(objects[0], coefficients[0])

  for o_i, c_i in zip(objects[1:], coefficients[1:]):
    if not types.abstract_objects_equal(accumulator, o_i):
      raise ValueError("One or more objects do not have equivalent abstract "
                       "structure.")
    accumulator = jax.tree_util.tree_map(
        jnp.add, accumulator, scalar_mul(o_i, c_i))

  return accumulator


def sum_objects(objects: Sequence[TArrayTree]) -> TArrayTree:
  return weighted_sum_of_objects(objects, [1] * len(objects))


def _inner_product_float64(obj1: ArrayTree, obj2: ArrayTree) -> Array:
  """Computes inner product explicitly in float64 precision."""

  raise NotImplementedError()

  # This function isn't currently working due to a break in
  # jax.experimental.enable_x64.

  # def array_ip(x, y):
  #   x = jnp.array(jnp.reshape(x, [-1]), dtype=jnp.float64)
  #   y = jnp.array(jnp.reshape(y, [-1]), dtype=jnp.float64)
  #   return jnp.dot(x, y, precision=lax.Precision.HIGHEST)

  # original_dtype = types.get_float_dtype_and_check_consistency((obj1, obj2))

  # with jax.experimental.enable_x64():

  #   elements_inner_products = jax.tree_util.tree_map(array_ip, obj1, obj2)

  #   flat_list = jax.tree_util.tree_leaves(elements_inner_products)
  #   result = flat_List[0]

  #   for element_ip in flat_List[1:]:
  #     result = result + element_ip

  # return jnp.array(result, dtype=original_dtype)


def inner_product(
    obj1: ArrayTree,
    obj2: ArrayTree,
    in_float64: bool = False
) -> Array:
  """Computes the inner product `<vec(obj1), vec(obj2)>`.

  To compute the inner product, each of the two input objects is assumed to
  represent a vector by flattening and concatenating all of their PyTree leaves.
  Objects `obj1` and `obj2` must have the same PyTree structure, and PyTree
  leaves in equivalent positions must have the same shape.

  Args:
    obj1: The first object representing a vector.
    obj2: The second object representing a vector.
    in_float64: Whether to compute the inner product explicitly in `float64`
      precision. If this is set to `True` the computation will be in double
      precision regardless of whether `float64` has been enabled in Jax.

  Returns:
    The scalar value of the inner product.
  """
  if not types.abstract_objects_equal(obj1, obj2, check_dtype=False):
    raise ValueError("The objects do not have identical abstract structure.")

  if in_float64:
    return _inner_product_float64(obj1, obj2)

  elements_product = jax.tree_util.tree_map(
      lambda x, y: jnp.sum(x * y), obj1, obj2)

  return sum(jax.tree_util.tree_leaves(elements_product))


def symmetric_matrix_inner_products(
    vectors1: Sequence[ArrayTree],
    vectors2: Sequence[ArrayTree],
    ip_function: Callable[[ArrayTree, ArrayTree], Array] = inner_product,
) -> Array:
  """Computes a matrix of the inner products between the two sequences.

  Args:
    vectors1: A sequence of identically structured PyTrees, each one
      representing a single vector.
    vectors2: A sequence of identically structured PyTrees, each one
      representing a single vector.
    ip_function: A callable which computes the inner product between PyTrees.
      Defaults to the standard dot-product.

  Returns:
    A symmetric matrix `m` with elements `m[i, j] = <vectors[i], vectors2[j]>`
    for `i >= j`.
  """
  if len(vectors1) != len(vectors2):
    raise ValueError("The two sequences should have the same length.")

  m = [[] for _ in vectors1]
  for i, v_i in enumerate(vectors1):
    for j, v_j in enumerate(vectors2):
      if j < i:
        m[i].append(m[j][i])
      else:
        m[i].append(ip_function(v_i, v_j))

  return jnp.asarray(m)


def matrix_of_inner_products(
    vectors: Sequence[ArrayTree],
    ip_function: Callable[[ArrayTree, ArrayTree], Array] = inner_product,
) -> Array:
  """Computes the matrix of inner products of the sequence of vectors.

  Args:
    vectors: A sequence of identically structured PyTrees, each one representing
      a single vector.
    ip_function: A callable which computes the inner product between PyTrees.
      Defaults to the standard dot-product.

  Returns:
    A matrix `m` with elements `m[i, j] = <vectors[i], vectors[j]>`.
  """
  return symmetric_matrix_inner_products(vectors, vectors,
                                         ip_function=ip_function)


def vector_of_inner_products(
    base: ArrayTree,
    vectors: Sequence[ArrayTree],
    ip_function: Callable[[ArrayTree, ArrayTree], Array] = inner_product,
) -> Array:
  """Computes a vector of inner products with base.

  Args:
    base: A PyTree representing the base vector.
    vectors: A sequence of identically structured PyTrees, each one representing
      a single vector.
    ip_function: A callable which computes the inner product between PyTrees.
      Defaults to the standard dot-product.

  Returns:
    A vector `v` with elements `v[i] = <base, vectors[i]>`.
  """
  v = []
  for v_i in vectors:
    v.append(ip_function(v_i, base))

  return jnp.asarray(v)


def block_permuted(
    matrix: Array,
    block_sizes: Sequence[int],
    block_order: Sequence[int],
) -> Array:
  """Permutes whole blocks of the input matrix.

  Given a square matrix, this function splits it into blocks, each one having
  a size defined in `block_sizes` and permutes them, both in rows and
  columns. The permutation sends to the `i` slot the `block_order[i]` block of
  the input matrix. Example:
      matrix = [[A_0, B_0, C_0], [A_1, B_1, C_1], [A_2, B_2, C_2]]
      block_order = [2, 0, 1]
      => [[C_2, A_2, B_2], [C_0, A_0, B_0], [C_1, A_1, B_1]]

  Args:
    matrix: The matrix, whose blocks will be permuted.
    block_sizes: A sequences of each block's size.
    block_order: A sequence of the order of the blocks.

  Returns:
    The resulting matrix after permuting the blocks.
  """
  if len(block_sizes) != len(block_order):
    raise ValueError(
        f"The length of `block_sizes` (=={len(block_sizes)} "
        f"and `block_order` (=={len(block_order)}) must be "
        "the same.")

  if all(i == j for i, j in enumerate(block_order)):
    return matrix

  indices = np.cumsum(block_sizes)[:-1]
  blocks = [jnp.split(row, indices, 1) for row in jnp.split(matrix, indices, 0)]
  reordered_blocks = [[blocks[i][j] for j in block_order] for i in block_order]

  return jnp.block(reordered_blocks)


def squared_norm(obj: ArrayTree) -> Array:
  """Computes the squared Euclidean norm of the provided PyTree object."""
  elements_squared_norm = jax.tree_util.tree_map(
      lambda x: jnp.sum(jnp.square(x)), obj)

  return sum(jax.tree_util.tree_leaves(elements_squared_norm))


def norm(obj: ArrayTree) -> Array:
  """Computes the Euclidean norm of the provided PyTree object."""
  elements_squared_norm = jax.tree_util.tree_map(
      lambda x: jnp.sum(jnp.square(x)), obj)

  return jnp.sqrt(sum(jax.tree_util.tree_leaves(elements_squared_norm)))


def per_parameter_norm(obj: ArrayTree, key_prefix: str) -> ArrayTree:

  per_param_norm = jax.tree_util.tree_map(jnp.linalg.norm, obj)
  per_param_norm = tree.flatten_with_path(per_param_norm)

  return {
      key_prefix + "(" + "/".join(k) + ")": v for k, v in per_param_norm
  }


def psd_inv(matrix: Array) -> Array:
  """Computes the inverse of `matrix`, which is assumed PSD."""

  if matrix.shape[:1] != matrix.shape[1:]:
    raise ValueError(f"Expected square matrix, but got shape {matrix.shape}.")

  if get_use_cholesky_inversion():
    identity = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    return linalg.solve(matrix, identity, assume_a="pos")
  else:
    return linalg.inv(matrix)


def psd_solve(matrix: Array, vector: Array) -> Array:
  """Computes the solution of `matrix * x = vector`, for a PSD `matrix`."""

  if matrix.shape[:1] != matrix.shape[1:]:
    raise ValueError(f"Expected square matrix, but got shape {matrix.shape}.")

  if get_use_cholesky_inversion():
    return linalg.solve(matrix, vector, assume_a="pos")

  else:
    return linalg.solve(matrix, vector)


def psd_solve_without_last_idx(a: Array, b: Array) -> Array:
  sub_a = a[..., :-1, :-1]
  sub_b = b[..., :-1]
  sub_x = psd_solve(sub_a, sub_b)
  return jnp.concatenate([sub_x, jnp.zeros_like(b[..., :1])], axis=-1)


def psd_solve_maybe_zero_last_idx(a: Array, b: Array) -> Array:
  # Check the last column and row are zero.
  check = jnp.logical_and(jnp.all(a[..., -1] == 0), jnp.all(a[..., -1, :] == 0))
  return jax.lax.cond(check, psd_solve_without_last_idx, psd_solve, a, b)


def psd_matrix_norm(
    matrix: Array,
    norm_type: str = "avg_diag",
    method_2norm: str = "lobpcg",
    rng_key: Optional[PRNGKey] = None
) -> Numeric:
  """Computes one of several different matrix norms for PSD matrices.

  NOTE: not all the functions options provided here are actually norms, but most
  are.

  Args:
    matrix: a square matrix represented as a 2D array, a 1D vector giving the
      diagonal, or a 0D scalar (which gets interpreted as a 1x1 matrix). Must be
      positive semi-definite (PSD).
    norm_type: a string specifying the type of matrix norm. Can be "2_norm" for
      the matrix 2-norm aka the spectral norm, "avg_diag" for the average of
      diagonal entries, "1_norm" for the matrix 1-norm, or "avg_fro" for the
      Frobenius norm divided by the square root of the number of rows.
    method_2norm: a string specifying the method used to compute 2-norms. Can
      be "lobpcg" (recommended) or "power_iteration".
    rng_key: an optional JAX PRNGKey key to used initialize the lobpcg method
      for computing the 2-norm.

  Returns:
    A 0D scalar giving the requested norm.
  """

  if norm_type == "2_norm":

    if matrix.ndim == 0:
      return matrix

    elif matrix.ndim == 1:
      return jnp.max(matrix)

    elif matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:

      if method_2norm == "lobpcg":

        if rng_key is None:
          rng_key = jax.random.PRNGKey(123)

        v = jax.random.normal(rng_key, shape=[matrix.shape[0], 1])

        return experimental_splinalg.lobpcg_standard(
            matrix, v, m=300, tol=1e-8)[0][0]

      elif method_2norm == "power_iteration":

        return optax.power_iteration(
            matrix, num_iters=300, error_tolerance=1e-7)[1]

      else:
        raise ValueError(f"Unrecognized method string: '{norm_type}'")

    else:
      raise ValueError(f"Unsupported shape for factor array: {matrix.shape}")

  elif norm_type == "avg_diag":

    if matrix.ndim == 0:
      return matrix

    elif matrix.ndim == 1:
      return jnp.sum(matrix) / matrix.shape[0]

    elif matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
      return jnp.trace(matrix) / matrix.shape[0]

    else:
      raise ValueError(f"Unsupported shape for factor array: {matrix.shape}")

  elif norm_type == "median_diag":

    if matrix.ndim == 0:
      return matrix

    elif matrix.ndim == 1:
      return jnp.median(matrix)

    elif matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
      return jnp.median(jnp.diag(matrix))

    else:
      raise ValueError(f"Unsupported shape for factor array: {matrix.shape}")

  elif norm_type == "trace":

    if matrix.ndim == 0:
      return matrix

    elif matrix.ndim == 1:
      return jnp.sum(matrix)

    elif matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
      return jnp.trace(matrix)

    else:
      raise ValueError(f"Unsupported shape for factor array: {matrix.shape}")

  elif norm_type == "median_eig":

    if matrix.ndim == 0:
      return matrix

    elif matrix.ndim == 1:
      return jnp.median(matrix)

    elif matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
      # call safe_psd_eigh instead?
      s, _ = jnp.linalg.eigh(matrix)
      return jnp.median(s)

    else:
      raise ValueError(f"Unsupported shape for factor array: {matrix.shape}")

  elif norm_type == "one_over_dim":  # this isn't a norm

    if matrix.ndim == 0:
      return 1.0

    elif matrix.ndim == 1:
      return 1.0 / matrix.shape[0]

    elif matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
      return 1.0 / matrix.shape[0]

    else:
      raise ValueError(f"Unsupported shape for factor array: {matrix.shape}")

  elif norm_type == "1_norm":  # equiv to inf norm for symmetric matrices

    if matrix.ndim == 0:
      return matrix

    elif matrix.ndim == 1:
      return jnp.max(matrix)

    elif matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
      return jnp.linalg.norm(matrix, ord=1)

    else:
      raise ValueError(f"Unsupported shape for factor array: {matrix.shape}")

  elif norm_type == "avg_fro":

    if matrix.ndim == 0:
      return matrix

    elif matrix.ndim == 1:
      return jnp.linalg.norm(matrix) / jnp.sqrt(matrix.shape[0])

    elif matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
      return jnp.linalg.norm(matrix) / jnp.sqrt(matrix.shape[0])

    else:
      raise ValueError(f"Unsupported shape for factor array: {matrix.shape}")

  elif norm_type == "fro":

    if matrix.ndim == 0:
      return matrix

    elif matrix.ndim == 1:
      return jnp.linalg.norm(matrix)

    elif matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
      return jnp.linalg.norm(matrix)

    else:
      raise ValueError(f"Unsupported shape for factor array: {matrix.shape}")

  raise ValueError(f"Unrecognized norm type: '{norm_type}'")


def pi_adjusted_kronecker_factors(
    *factors: Array,
    damping: Numeric
) -> Tuple[Array, ...]:
  """Computes Kronecker factors with pi-adjusted factored damping.

  The `f1 kron f2 kron ... kron fn + damping * I` is not a Kronecker product
  in general, because of the added identity. [1] proposed a pi-adjusted factored
  damping approach to approximate it as a Kronecker product. [2] generalized
  this approach from two to tree factors, and [3] generalized it to arbitrary
  numbers of factors. This function implements the generalized approach.

  [1] - https://arxiv.org/abs/1503.05671
  [2] - https://openreview.net/forum?id=SkkTMpjex
  [3] - https://ui.adsabs.harvard.edu/abs/2021arXiv210602925R/abstract

  Args:
    *factors: A list of factors represented as 2D arrays, vectors (which are
      interpreted as representing the diagonal of a matrix) or scalars (which
      are interpreted as being a 1x1 matrix). All factors must be PSD.
    damping: The weight of the identity added to the Kronecker product.

  Returns:
    A list of factors with the same length as `factors`, and with the same
    corresponding representations, whose Kronecker product approximates
    `(f1 kron f2 kron ... kron fn) + damping * I` according to the
    pi-adjusted factored-damping approach.
  """

  # The implementation writes each single factor as `c_i u_i`, where the matrix
  # `u_i` is such that `trace(u_i) / dim(u_i) = 1`. We then factor out all the
  # scalar factors `c_i` into a single overall scaling coefficient and
  # distribute the damping to each single non-scalar factor `u_i` equally.

  norm_type = "avg_diag"

  norms = jnp.array([psd_matrix_norm(f, norm_type=norm_type) for f in factors])

  # Compute the normalized factors `u_i`, such that Trace(u_i) / dim(u_i) = 1
  us = [fi / ni for fi, ni in zip(factors, norms)]

  k = len(factors)

  # TODO(jamesmartens,botev): consider making the use of special behavior for
  # scalar factors a module-level configurable option. One can argue that scalar
  # factors should behave the same as non-scalar factors for the sake of
  # consistent behavior as the layer widths shrink to 1.

  def regular_case() -> Tuple[Array, ...]:

    num_non_scalars = sum(1 if f.size != 1 else 0 for f in factors)

    if num_non_scalars != 0:

      # Distribute c and damping/c among k factors, where c = jnp.prod(norms),
      # satisfying kron(factors) = c * kron(us).

      # NOTE: c_k (geometric mean of norms) can also be calculated by
      # c ** (1/k) = jnp.prod(norms) ** (1 / len(norms)), but this alternative
      # can make the result zero due to the multiplication of (potentially)
      # small values, i.e. jnp.prod(norms).
      c_k = jnp.exp(jnp.mean(jnp.log(norms)))

      d_k = jnp.power(damping, 1.0 / k) / c_k

      if k > num_non_scalars:

        c_non_scalar = c_k ** (float(k) / num_non_scalars)

        # We distribute the damping only inside the non-scalar factors
        d_hat = jnp.power(damping, 1.0 / num_non_scalars) / c_non_scalar

      else:
        d_hat = d_k

    else:

      # This could cause under/overflow, but it's unavoidable here.
      c = jnp.prod(jnp.array(norms))

      # In the case where all factors are scalar we need to add the damping and
      # then take the k-th root
      c_k = jnp.power(c + damping, 1.0 / k)

    u_hats = []

    for u in us:

      if u.size == 1:  # scalar case
        u_hat = jnp.ones_like(u)  # damping not used in the scalar factors

      elif u.ndim == 2:
        u_hat = u + d_hat * jnp.eye(u.shape[0], dtype=u.dtype)

      else:  # diagonal case
        assert u.ndim == 1
        u_hat = u + d_hat

      u_hats.append(u_hat * c_k)

    return tuple(u_hats)

  def zero_case() -> Tuple[Array, ...]:

    # In the special case where for some reason one of the factors is zero, then
    # the we write each factor as `damping^(1/k) * I`.

    c_k = jnp.power(damping, 1.0 / k)

    u_hats = []

    for u in us:

      if u.ndim == 2:
        u_hat = jnp.eye(u.shape[0], dtype=u.dtype)

      else:
        u_hat = jnp.ones_like(u)

      u_hats.append(u_hat * c_k)

    return tuple(u_hats)

  if get_special_case_zero_inv():
    return lax.cond(jnp.greater(jnp.min(norms), 0.0), regular_case, zero_case)

  else:
    return regular_case()


def invert_psd_matrices(
    matrices: ArrayTree
) -> ArrayTree:
  """Inverts a PyTree of matrices.

  Args:
    matrices: A PyTree of 2D arrays, vectors (which are interpreted as
      representing the diagonal of a matrix) or scalars (which are interpreted
      as being a 1x1 matrix) representing the matrices to be inverted. All
      matrices must be PSD.

  Returns:
    A PyTree of matrices giving the inverses of the corresponding matrices
    passed as arguments (with the same respective representations).
  """

  def invert_psd_matrix(m):

    if m.ndim == 2:
      return psd_inv(m)

    assert m.ndim <= 1
    return 1.0 / m

  return jax.tree_map(invert_psd_matrix, matrices)


def inverse_sqrt_psd_matrices(matrices: ArrayTree) -> ArrayTree:

  def inverse_sqrt_psd_matrix(m):

    if m.ndim == 2:
      # Check copy.bara.sky before changing the next line:
      return qr_pth_inv_root.qr_pth_inv_root(4, m, cholesky_qr=True)

    assert m.ndim <= 1
    return 1.0 / jnp.sqrt(m)

  return jax.tree_map(inverse_sqrt_psd_matrix, matrices)


def pi_adjusted_kronecker_inverse(
    *factors: Array,
    damping: Numeric,
) -> Tuple[Array, ...]:
  """Computes pi-adjusted factored damping inverses.

  The inverse of `(f1 kron f2 kron ... kron fn) + damping * I` is not Kronecker
  factored in general, because of the added identity. [1] proposed a pi-adjusted
  factored damping approach to approximate the inverse as a Kronecker product.
  [2] generalized this approach from two to tree factors, and [3] generalized it
  to arbitrary numbers of factors. This function implements the generalized
  approach.

  [1] - https://arxiv.org/abs/1503.05671
  [2] - https://openreview.net/forum?id=SkkTMpjex
  [3] - https://ui.adsabs.harvard.edu/abs/2021arXiv210602925R/abstract

  Args:
    *factors: A list of factors represented as 2D arrays, vectors (which are
      interpreted as representing the diagonal of a matrix) or scalars (which
      are interpreted as being a 1x1 matrix). All factors must be PSD.
    damping: The weight of the identity added to the Kronecker product.

  Returns:
    A list of factors with the same length as `factors`, and with the same
    corresponding representations, whose Kronecker product approximates the
    inverse of `(f1 kron f2 kron ... kron fn) + damping * I` according to the
    pi-adjusted factored-damping approach.
  """

  return invert_psd_matrices(
      pi_adjusted_kronecker_factors(*factors, damping=damping))  # pytype: disable=bad-return-type


def kronecker_product_axis_mul_v(
    factors: Sequence[Array],
    v: Array,
    axis_groups: Optional[Sequence[Sequence[int]]] = None,
    transpose: Union[bool, Sequence[bool]] = False,
):
  """Computes ``kron(*factors) rvec(v)`` where ``rvec`` is row-wise vectorization.

  Args:
    factors: The sequence of factors forming the Kronecker product. Must be
      square 2D arrays.
    v: A tensor whose vectorization will be multiplied by the Kronecker product.
    axis_groups: A list whose i-th element is a sequence of consecutive integers
      specifying the axes of the input tensor ``v`` that correspond to the i-th
      Kronecker factor. Passing ``None`` is equivalent to passing
      ``[[0],[1],[2],...]``.
    transpose: A single boolean or a sequence of booleans. If it is a sequence,
      each element specifies if the corresponding factor should be transposed.
      If it is a single boolean, specifies if all factors should be transposed.

  Returns:
    The result, shaped as a tensor, of multiplying the vectorization of the
    input tensor by the Kronecker-factored matrix.
  """
  if axis_groups is None:
    axis_groups = tuple((i,) for i in range(v.ndim))
  else:
    axis_groups = tuple(tuple(group) for group in axis_groups)

  # Sanity checks
  if sum(axis_groups, ()) != tuple(range(v.ndim)):
    raise ValueError(f"The `axis_groups={axis_groups}` are either not in "
                     f"consecutive order or do not cover exactly the axis of "
                     f"the input `v`..")
  if len(factors) != len(axis_groups):
    raise ValueError("The number of factors provided must be equal to the "
                     "number of axis groups provided.")

  if isinstance(transpose, bool):
    transpose = [transpose] * len(factors)

  elif len(transpose) != len(factors):
    raise ValueError("The length of the transpose sequence must match the "
                     "number of factors.")

  factor_strs = ["yz" if t else "zy" for t in transpose]
  general_str = _ALPHABET[:v.ndim]

  result = v
  for group, factor, f_str in zip(axis_groups, factors, factor_strs):

    # This flattens all axis in `group` of `result` into a single one.
    shape = v.shape[:min(group)] + (-1,) + v.shape[max(group) + 1:]
    vector = result.reshape(shape)

    # This contracts `result` with `factor` along the single axis.
    vector_str = general_str[:min(group)] + "y" + general_str[max(group) + 1:]
    result_str = vector_str.replace("y", "z")
    einsum_str = f"{f_str},{vector_str}->{result_str}"
    r_next = jnp.einsum(einsum_str, factor, vector)

    # This reshapes back to the original shape.
    result = r_next.reshape(v.shape)

  return result


def kronecker_eigen_basis_axis_mul_v(
    q_factors: Sequence[Array],
    eigenvalues: Array,
    v: Array,
    axis_groups: Optional[Sequence[Sequence[int]]] = None,
):
  """Computes a matrix-vector product in a Kronecker product eigen-basis.

  The function computes:
    ``kron(*q_factors) diag(eigenvalues) kron(*q_factors)^T rvec(v)``

  where all variables are appropriately sized matrices and ``rvec`` is
  row-wise vectorization. The computation is related to the usual Kronecker
  product ``kron(*factors) rvec(v)``, if ``factors`` are all symmetric PSD
  matrices and ``q_factors`` are the matrices of eigenvectors of ``factors`` and
  ``eigenvalues`` is the kronecker product of the eigenvalues of ``factors``.
  However, the function does not assume that its inputs are of this form.

  Args:
    q_factors: A sequence of the orthonormal basis of eigenvectors of each
      Kronecker factor.
    eigenvalues: A tensor containing the eigenvalues (e.g. the Kronecker product
      of eigenvalues of all factors).
    v: The input vector as a tensor.
    axis_groups: A list whose i-th element is a sequence of consecutive integers
      specifying the axes of the input tensor ``v`` that correspond to the i-th
      Kronecker factor. Passing ``None`` is equivalent to passing
      ``[[0],[1],[2],...]``.

  Returns:
    The result of multiplying the input vector by the Kronecker product of the
    factors, shaped as a tensor.
  """
  q_proj_v = kronecker_product_axis_mul_v(q_factors, v, axis_groups, True)

  if eigenvalues.shape != q_proj_v.shape:
    raise ValueError("The eigenvalues array should have the same shape as the "
                     "projection of `v` onto `kron(*factors)`.")

  eig_weighted_v = eigenvalues * q_proj_v

  return kronecker_product_axis_mul_v(q_factors, eig_weighted_v, axis_groups)


def kronecker_product_mul_v(
    a: Array,
    b: Array,
    v: Array,
    a_is_symmetric: bool,
) -> Array:
  """Computes `unvec[(a kron b) vec(v)]` for correctly sized input matrices."""
  del a_is_symmetric  # not used
  return kronecker_product_axis_mul_v([b, a], v)


def kronecker_eigen_basis_mul_v(
    q_a: Array,
    q_b: Array,
    eigenvalues: Array,
    v: Array,
) -> Array:
  """Computes a matrix-vector product in a Kronecker product eigen-basis.

  The function computes:
    `(q_a kron q_b) diagonal(eigenvalues) (q_a kron q_b)^T vec(v)`

  where all variables are appropriately sized matrices. The computation is
  related to the usual Kronecker product `(a kron b) vec(v)`, if `a` and `b` are
  symmetric matrices and `q_a` and `q_b` are the matrices of eigenvectors of `a`
  and `b` and `eigenvalues` is the outer product of the eigenvalues of `a` and
  `b`. However, the function does not assume anything about the `eigenvalues`
  and allows for any dense matrix.

  Args:
    q_a: An orthonormal basis for eigenvectors of the first Kronecker factor.
    q_b: An orthonormal basis for eigenvectors of the second Kronecker factor.
    eigenvalues: A matrix containing the eigenvalues (e.g. the product of
      eigenvalues of both factors).
    v: The input vector as a matrix.

  Returns:
    The result of the matrix-vector product.
  """
  return kronecker_eigen_basis_axis_mul_v([q_b, q_a], eigenvalues, v)


def _host_eigh(x: Array, *_) -> Tuple[Array, Array]:
  """This calls the CPU numpy function for eigh."""

  shape_s = jax.ShapeDtypeStruct(x.shape[:-1], x.dtype)
  shape_q = jax.ShapeDtypeStruct(x.shape, x.dtype)

  return jax.pure_callback(np.linalg.eigh, (shape_s, shape_q), x)


def _eigh(
    x: Array,
    force_on_host: bool = False,
) -> Tuple[Array, Array]:
  """Computes eigenvectors and eigenvalues, with optionally offloading to cpu."""

  if force_on_host:
    return _host_eigh(x)

  s, q = jnp.linalg.eigh(x)

  # Recently with CUDA 11.7 there is a bug in cuSOLVER which makes the eigh
  # implementation unstable sometimes on GPUs.
  return jax.lax.cond(
      jnp.any(jnp.isnan(s)),
      _host_eigh,
      lambda *args: args[1:],
      x, s, q
  )


def safe_psd_eigh(
    x: Array,
    force_on_host: bool = False,
) -> Tuple[Array, Array]:
  """Computes the eigenvalue decomposition for a PSD matrix.

  The function is similar to `jax.numpy.linalg.eigh`, but it clips the returned
  eigenvalues to always be non-negative, which we know mathematically holds for
  PSD matrices, but due to numerical errors `jax.numpy.linalg.eigh` could return
  negative values.

  Args:
    x: The input matrix, assumed to be PSD.
    force_on_host: If `True` will perform the computation on the host CPU.

  Returns:
    A pair of (eigenvalues, eigenvectors) arrays.
  """

  d = x.shape[0]

  # Here we are handling the case of NaNs separately, because in some versions
  # of cuda and cudablas they can cause a runtime error.
  s, q = lax.cond(
      jnp.any(jnp.isnan(x)),
      lambda _: (jnp.full([d], jnp.nan, dtype=x.dtype),  # pylint: disable=g-long-lambda
                 jnp.full([d, d], jnp.nan, dtype=x.dtype)),
      functools.partial(_eigh, force_on_host=force_on_host),
      x,
  )

  # The matrix is PSD by construction, but numerical inaccuracies can produce
  # slightly negative eigenvalues. Hence, clip at zero.
  return jnp.clip(s, a_min=0.0), q


def loop_and_parallelize_average(
    func: Callable[..., ArrayTree],
    max_parallel_size: int,
) -> Callable[..., ArrayTree]:
  """Returns a function that computes the average of `func` over any arguments.

  The returned function is mathematically equivalent to
    jnp.mean(jax.vmap(func)(*args), axis=0).
  However, naively using the above code could lead to prohibitively large memory
  usage, as it scales linearly with the leading axis size of `args`, because of
  `jax.vmap`. To amortize the memory cost, if the leading axis has size larger
  than `max_parallel_size`, we call multiple times `vmap` in a loop via `scan`
  by splitting the arguments to multiple chunks. This allows to trade off memory
  usage for the cost of compute time.

  Args:
    func: A function that computes a singleton output.
    max_parallel_size: The maximum number of elements that are allowed to be
      part of a single call to `jax.vmap`.

  Returns:
    A function that computes the averaged output of `func` over the leading
    axis of its arguments.
  """
  vmap_fn = jax.vmap(func)

  @functools.wraps(func)
  def average_func(*args) -> ArrayTree:

    lead_axis_sizes = set(x.shape[0] for x in jax.tree_util.tree_leaves(args))

    if not lead_axis_sizes:
      raise ValueError("You must pass in at least one argument with a PyTree "
                       "leaf node.")

    elif len(lead_axis_sizes) != 1:
      raise ValueError(f"Inconsistent leading axis sizes seen: "
                       f"{lead_axis_sizes!r}.")

    leading_size = next(iter(lead_axis_sizes))

    singleton_args = jax.tree_util.tree_map(lambda _x: _x[0], args)
    _, output_tree = jax.make_jaxpr(func, return_shape=True)(*singleton_args)

    singleton_size = sum(x.size for x in jax.tree_util.tree_leaves(output_tree))
    output_size = singleton_size * leading_size

    # Compute the loop size and any remainder size
    if max_parallel_size is None or output_size <= max_parallel_size:

      parallel_size = leading_size

    else:
      parallel_size = max(
          min(max_parallel_size // singleton_size, leading_size), 1)

    # The arguments have to be split into chunks along their leading axis,
    # however since `jax.scan` does not support inputs with different size,
    # if the leading axis is not divisible by the parallel_size, we need to
    # separately compute the values for the last remaining arguments chunks.
    num_parallel_chunks = leading_size // parallel_size
    remainder_size = leading_size % parallel_size
    all_chunks_size = leading_size - remainder_size

    # Index to get the loop arguments
    loop_args = jax.tree_util.tree_map(lambda x: x[:all_chunks_size], args)

    if num_parallel_chunks == 1:
      averaged_value = jnp.mean(vmap_fn(*loop_args), axis=0)

    else:

      def scan_fn(accumulator, args_):

        vmap_value = vmap_fn(*args_)

        avg_value = jax.tree_util.tree_map(
            lambda x: jnp.mean(x, axis=0), vmap_value)

        return jax.tree_util.tree_map(jnp.add, accumulator, avg_value), None

      loop_shape = (num_parallel_chunks, parallel_size)

      loop_args = jax.tree_util.tree_map(
          lambda x: x.reshape(loop_shape + x.shape[1:]),
          loop_args)

      summed_value, _ = jax.lax.scan(
          scan_fn,
          init=jax.tree_util.tree_map(
              jnp.zeros_like, output_tree),
          xs=loop_args)

      averaged_value = scalar_div(summed_value, num_parallel_chunks)

    if remainder_size == 0:
      return averaged_value

    # Index to get the remainder arguments
    remainder_args = jax.tree_util.tree_map(lambda x: x[all_chunks_size:], args)
    remainder_value = jnp.mean(vmap_fn(*remainder_args), axis=0)

    avg_weight = all_chunks_size / leading_size
    remainder_weight = remainder_size / leading_size

    return weighted_sum_of_objects(
        [averaged_value, remainder_value], [avg_weight, remainder_weight])

  return average_func
