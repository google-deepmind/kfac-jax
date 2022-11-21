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
import abc
import dataclasses
import functools
import numbers
import operator
import string
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence, Tuple, Type, TypeVar, Union

import chex
import jax
from jax import core
from jax import lax
from jax import tree_util
import jax.numpy as jnp
from jax.scipy import linalg
import numpy as np
import tree

_ALPHABET = string.ascii_lowercase
_CHEX_SCALAR_TYPES = (float, int)


# Types for annotation
T = TypeVar("T")
Params = chex.ArrayTree
Batch = chex.ArrayTree
FuncState = chex.ArrayTree
FuncAux = chex.ArrayTree
PyTreeDef = chex.PyTreeDef
PyTreeType = Any
PyTree = chex.ArrayTree
TPyTree = TypeVar("TPyTree", bound=PyTree)
FuncArgs = Sequence[PyTree]
Func = Callable[..., Union[chex.Array, Tuple[chex.Array, FuncAux]]]
ValueFunc = Callable[..., chex.Array]
ValueAndGradFunc = Callable[..., Tuple[chex.Array, Params]]

AssumedFuncOutput = Union[
    chex.Array,
    Tuple[chex.Array, FuncAux],
    Tuple[chex.Array, Tuple[FuncState, FuncAux]],
]

# Special global state
# If true we use a special case formula for when a block has one or more zero
# factors.
_SPECIAL_CASE_ZERO_INV: bool = True


def set_special_case_zero_inv(value: bool):
  """Sets whether `pi_adjusted_inverse` handles zero and nan matrices."""
  global _SPECIAL_CASE_ZERO_INV
  _SPECIAL_CASE_ZERO_INV = value


def get_special_case_zero_inv() -> bool:
  """Returns whether `pi_adjusted_inverse` handles zero and nan matrices."""
  return _SPECIAL_CASE_ZERO_INV


def fake_element_from_iterator(
    iterator: Iterator[PyTree],
) -> Tuple[PyTree, Iterator[PyTree]]:
  """Returns a zeroed-out initial element of the iterator "non-destructively".

  This function mutates the input iterator, hence after calling this function
  it will be advanced by one. An equivalent to the original iterator (e.g. not
  advanced by one) is returned as the second element of the returned pair. The
  advised usage of the function is:
    `fake_element, iterator = fake_element_from_iterator(iterator)`

  Args:
    iterator: A PyTree iterator. Must yield at least one element.

  Returns:
    A pair `(element, output_iterator)` where `element` is a zeroed-out version
    of the first element of the iterator, and `output_iterator` is an
    equivalent iterator to the input one.
  """
  init_element = next(iterator)
  fake_element = jax.tree_util.tree_map(np.zeros_like, init_element)
  def equivalent_iterator() -> Iterator[PyTree]:
    yield init_element
    # For some reason unknown to us, "yield from" can fail in certain
    # circumstances
    while True:
      yield next(iterator)
  return fake_element, equivalent_iterator()


def loop_and_parallelize_average(
    func: Callable[..., PyTree],
    max_parallel_size: int,
) -> Callable[..., PyTree]:
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
  def average_func(*args) -> PyTree:

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
              lambda x: jnp.zeros(x.shape), output_tree),
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


#  _____  __  __          _____
# |  __ \|  \/  |   /\   |  __ \
# | |__) | \  / |  /  \  | |__) |
# |  ___/| |\/| | / /\ \ |  ___/
# | |    | |  | |/ ____ \| |
# |_|    |_|  |_/_/    \_\_|
#


def in_pmap(axis_name) -> bool:
  """Returns whether we are in a pmap with the given axis name."""

  if axis_name is None:
    return False

  try:
    # The only way to know if we are under `jax.pmap` is to check if the
    # function call below raises a `NameError` or not.
    core.axis_frame(axis_name)

    return True

  except NameError:
    return False


def wrap_if_pmap(
    p_func: Callable[[PyTree, str], PyTree],
) -> Callable[[PyTree, Optional[str]], PyTree]:
  """Wraps `p_func` to be executed only when inside a `jax.pmap` context."""

  @functools.wraps(p_func)
  def p_func_if_pmap(obj: PyTree, axis_name: Optional[str]) -> PyTree:

    return p_func(obj, axis_name) if in_pmap(axis_name) else obj

  return p_func_if_pmap


pmean_if_pmap = wrap_if_pmap(lax.pmean)
psum_if_pmap = wrap_if_pmap(lax.psum)

compute_mean = jax.pmap(lambda x: lax.pmean(x, "i"), axis_name="i")
compute_sum = jax.pmap(lambda x: lax.psum(x, "i"), axis_name="i")


def index_if_not_scalar(value: chex.Numeric, index: int = 0) -> chex.Numeric:
  """Index `value` at axis 0 if it is not a scalar, otherwise return it."""

  if isinstance(value, chex.Array):

    if value.ndim > 0:
      return value[index]
    else:
      return value

  elif isinstance(value, _CHEX_SCALAR_TYPES):
    return value

  else:
    raise ValueError("The input should be an instance of `chex.Numeric`.")


@jax.jit
def get_first(obj: PyTree) -> PyTree:
  """Index the PyTree leaves `x` of `obj` by `x[0]` if they are not scalars."""
  return jax.tree_util.tree_map(index_if_not_scalar, obj)


def get_mean(obj: PyTree) -> PyTree:
  """Returns the average of `obj` over different devices."""
  return get_first(compute_mean(obj))


def get_sum(obj: PyTree) -> PyTree:
  """Returns the sum of `obj` over different devices."""
  return get_first(compute_sum(obj))


broadcast_all_local_devices = jax.pmap(lambda x: x)
pmap_zeros_like = jax.pmap(lambda x: jax.tree_util.tree_map(jnp.zeros_like, x))
jit_zeros_like = jax.jit(lambda x: jax.tree_util.tree_map(jnp.zeros_like, x))


def replicate_all_local_devices(obj: PyTree) -> PyTree:
  """Replicates `obj` to all local Jax devices."""

  n = jax.local_device_count()
  obj_stacked = jax.tree_util.tree_map(
      lambda x: jnp.stack([x] * n, axis=0), obj)

  return broadcast_all_local_devices(obj_stacked)


def make_different_rng_key_on_all_devices(rng: chex.PRNGKey) -> chex.PRNGKey:
  """Makes a different PRNG for all Jax devices and processes."""

  rng = jax.random.fold_in(rng, jax.process_index())
  rng = jax.random.split(rng, jax.local_device_count())

  return broadcast_all_local_devices(rng)


p_split = jax.pmap(lambda key: tuple(jax.random.split(key)))
p_split_num = jax.pmap(lambda key, num: tuple(jax.random.split(key, num)),
                       static_broadcasted_argnums=1)


def check_and_fix_format_for_pmap(obj: PyTree) -> PyTree:
  """Checks shape[0]==device_count and broadcasts scalars to [device_count]."""
  device_count = jax.local_device_count()

  def check_and_fix(x: chex.Numeric) -> chex.Array:

    # broadcast any 0D scalars
    if isinstance(x, numbers.Number) or not x.shape:
      return jnp.stack([x] * device_count, axis=0)

    # otherwise ensure that arrays have the right shape
    assert x.shape[0] == device_count

    return x

  return jax.tree_util.tree_map(check_and_fix, obj)


default_device_sync = None


def host_sync(
    obj: PyTree,
    sync_op: Callable[[PyTree, str], PyTree],
) -> PyTree:
  """Syncs `obj` across multiple hosts with the operation `sync_op`."""

  # The implementation here is to use the pmap syncing mechanisms but with only
  # the default device of each host. Technically we could do this with all
  # the devices on each host, but that would possibly be wasteful.

  if jax.process_count() > 1:

    # We set default_device_sync here because calling jax.local_devices during
    # the library import stage will break JAX.

    global default_device_sync

    if default_device_sync is None:

      default_devices = [jax.local_devices(process_index=p_idx)[0]
                         for p_idx in range(jax.process_count())]

      default_device_sync = jax.pmap(lambda x, sync_op: sync_op(x, "i"),
                                     devices=default_devices,
                                     axis_name="i",
                                     static_broadcasted_argnums=1)

    obj = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), obj)

    return get_first(default_device_sync(obj, sync_op))

  return obj


def host_all_gather(x: PyTree) -> PyTree:
  """Gathers on every host the values of the PyTree leaves `x`."""
  return host_sync(x, lax.all_gather)


def host_mean(x: PyTree) -> PyTree:
  """Computes the mean of the PyTree leaves of `x` over multiple hosts."""
  return host_sync(x, lax.pmean)


def sync_and_divide_value(
    value: PyTree,
    counter: chex.Numeric,
    axis_name: Optional[str] = None,
) -> PyTree:
  """Computes the mean of `value` over all hosts and divides it by `counter`."""
  value = jax.tree_util.tree_map(lambda x: x / counter, value)
  return pmean_if_pmap(value, axis_name)


jit_sync_and_divide_value = jax.jit(sync_and_divide_value, donate_argnums=0)
pmap_sync_and_divide_value = jax.pmap(
    functools.partial(sync_and_divide_value, axis_name="i"),
    axis_name="i",
    donate_argnums=0,
)


# We might be able to change this to "return jnp.array(x)" in newer JAX versions
def copy_array(x: chex.Array) -> chex.Array:
  """Copies a Jax array so that it can be donated freely."""
  return x + jnp.zeros_like(x)


copy_obj = jax.jit(lambda x: jax.tree_util.tree_map(copy_array, x))
pmap_copy_obj = jax.pmap(copy_obj)


def distribute_thunks(
    thunks: Sequence[Callable[[], PyTree]],
    pmap_axis_name: str,
    ) -> PyTree:
  """Distributes the computation of a list of thunks over the pmapped devices.

  Given a list of thunks, this function distributes their computation over the
  devices of the current pmap in a round-robin fashion, syncronizes the results
  across devices, and then returns them as a sequence of PyTrees.

  Note that this function is meant to be used in a compiled context, and may
  call ``thunk[i]()`` several times for each i, with all but one call getting
  "optimized away" by XLA.

  Args:
    thunks: A sequence of callables performing the desired computations. Each
      callable must take zero arguments and return a PyTree of JAX arrays. As
      with callables passed to (most) standard JAX API functions, these need to
      be stateless and free of side effects. The output of each callable must be
      the same regardless of the device it is executed on.
    pmap_axis_name: The name of the pmap axis to use.

  Returns:
    A sequence of PyTrees that are the output of the corresponding element of
    ``thunks``.
  """

  # The strategy here is to make a callable for each device which executes only
  # the thunks i such that i % total_devices == device_index, and returns a tree
  # of zeros for the remaining thunks. We then do a lax.switch over these based
  # on device_index, and return psum over these. Note that the more obvious way
  # of doing this, which is to perform a psum over the output of a sequence of
  # lax.cond calls (with one for each thunk), won't work in general. This is
  # because in order to save memory, XLA will sometimes elect to execute these
  # conds sequentially instead of in parallel.

  if not in_pmap(pmap_axis_name):
    raise ValueError(f"Provided pmap_axis_name {pmap_axis_name} is not a valid "
                     "pmap axis in current pmap (or this function was not "
                     "called in a pmap).")

  assert pmap_axis_name is not None

  total_devices = lax.psum(1, axis_name=pmap_axis_name)  # returns a constant
  current_device_index = lax.axis_index(pmap_axis_name)

  # This should get optimized away by XLA since we don't use the values:
  dummy_output_trees = tuple(thunk() for thunk in thunks)

  def make_branch(device_index):

    def branch():
      """Execute only thunks i such that i % total_devices == device_index."""

      outs = []
      for i in range(len(thunks)):

        if i % total_devices == device_index:
          outs.append(thunks[i]())
        else:
          outs.append(
              jax.tree_util.tree_map(jnp.zeros_like, dummy_output_trees[i]))

      return tuple(outs)

    return branch

  branches = tuple(make_branch(device_index)
                   for device_index in range(total_devices))

  output_trees = jax.lax.switch(current_device_index, branches)

  return jax.lax.psum(output_trees, axis_name=pmap_axis_name)


#  __  __       _______ _    _
# |  \/  |   /\|__   __| |  | |
# | \  / |  /  \  | |  | |__| |
# | |\/| | / /\ \ | |  |  __  |
# | |  | |/ ____ \| |  | |  | |
# |_|  |_/_/    \_\_|  |_|  |_|
#
def product(iterable_object: Iterable[chex.Numeric]) -> chex.Numeric:
  """Computes the product of all elements in the iterable."""
  x = 1

  for element in iterable_object:
    x = x * element

  return x


def outer_product(*arrays: chex.Array) -> chex.Array:
  """Computes the outer product of an arbitrary number of vectors."""
  if not all(a.ndim == 1 for a in arrays):
    raise ValueError("All arrays must be vectors.")
  in_str = ",".join(_ALPHABET[:len(arrays)])
  out_str = _ALPHABET[:len(arrays)]
  return jnp.einsum(f"{in_str}->{out_str}", *arrays)


def scalar_mul(obj: TPyTree, scalar: chex.Numeric) -> TPyTree:
  """Multiplies all PyTree leaves of the object by the provided scalar."""
  # The check below is in its current form because of how `jax.jit` tracing
  # mechanism work. If we use `scalar == 1` and `scalar` is an array,  inside a
  # `jit` context, jax will raise an error, since you are not allowed to use
  # abstract values in concrete boolean statements, like native python
  # if/while/for constructs.
  if isinstance(scalar, _CHEX_SCALAR_TYPES) and scalar == 1.0:
    return obj

  return jax.tree_util.tree_map(lambda x: x * scalar, obj)


def scalar_div(obj: TPyTree, scalar: chex.Numeric) -> TPyTree:
  """Divides all PyTree leaves of the object by the provided scalar."""
  # The check below is in its current form because of how `jax.jit` tracing
  # mechanism work. If we use `scalar == 1` and `scalar` is an array,  inside a
  # `jit` context, jax will raise an error, since you are not allowed to use
  # abstract values in concrete boolean statements, like native python
  # if/while/for constructs.
  if isinstance(scalar, _CHEX_SCALAR_TYPES) and scalar == 1.0:
    return obj

  return jax.tree_util.tree_map(lambda x: x / scalar, obj)


def weighted_sum_of_objects(
    objects: Sequence[TPyTree],
    coefficients: Sequence[chex.Numeric],
) -> TPyTree:
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
    if not abstract_objects_equal(accumulator, o_i):
      raise ValueError("One or more objects do not have equivalent abstract "
                       "structure.")
    accumulator = jax.tree_util.tree_map(
        jnp.add, accumulator, scalar_mul(o_i, c_i))

  return accumulator


def _inner_product_float64(obj1: PyTree, obj2: PyTree) -> chex.Array:
  """Computes inner product explicitly in float64 precision."""

  def array_ip(x, y):
    x = jnp.array(jnp.reshape(x, [-1]), dtype=jnp.float64)
    y = jnp.array(jnp.reshape(y, [-1]), dtype=jnp.float64)
    return jnp.dot(x, y, precision=lax.Precision.HIGHEST)

  with jax.experimental.enable_x64():

    elements_inner_products = jax.tree_util.tree_map(array_ip, obj1, obj2)

    flat_list = jax.tree_util.tree_leaves(elements_inner_products)
    result = flat_list[0]

    for element_ip in flat_list[1:]:
      result = result + element_ip

  # Convert back to default Jax dtype (usually float32)
  return jnp.array(result)


def inner_product(
    obj1: PyTree,
    obj2: PyTree,
    in_float64: bool = False
) -> chex.Array:
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
  if not abstract_objects_equal(obj1, obj2, check_dtype=False):
    raise ValueError("The objects do not have identical abstract structure.")

  if in_float64:
    return _inner_product_float64(obj1, obj2)

  elements_product = jax.tree_util.tree_map(
      lambda x, y: jnp.sum(x * y), obj1, obj2)

  return sum(jax.tree_util.tree_leaves(elements_product))


def symmetric_matrix_inner_products(
    vectors1: Sequence[PyTree],
    vectors2: Sequence[PyTree],
    ip_function: Callable[[PyTree, PyTree], chex.Array] = inner_product,
) -> chex.Array:
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
    vectors: Sequence[PyTree],
    ip_function: Callable[[PyTree, PyTree], chex.Array] = inner_product,
) -> chex.Array:
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
    base: PyTree,
    vectors: Sequence[PyTree],
    ip_function: Callable[[PyTree, PyTree], chex.Array] = inner_product,
) -> chex.Array:
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
    matrix: chex.Array,
    block_sizes: Sequence[int],
    block_order: Sequence[int],
) -> chex.Array:
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


def norm(obj: PyTree) -> chex.Array:
  """Computes the Euclidean norm of the provided PyTree object."""
  elements_squared_norm = jax.tree_util.tree_map(
      lambda x: jnp.sum(jnp.square(x)), obj)

  return jnp.sqrt(sum(jax.tree_util.tree_leaves(elements_squared_norm)))


def per_parameter_norm(obj: PyTree, key_prefix: str) -> PyTree:

  per_param_norm = jax.tree_util.tree_map(jnp.linalg.norm, obj)
  per_param_norm = tree.flatten_with_path(per_param_norm)

  return {
      key_prefix + "(" + "/".join(k) + ")": v for k, v in per_param_norm
  }


def psd_inv_cholesky(matrix: chex.Array, damping: chex.Array) -> chex.Array:
  """Computes the inverse of `matrix + damping*I`, with matrix assumed PSD."""

  if matrix.shape[:1] != matrix.shape[1:]:
    raise ValueError(f"Expected square matrix, but got shape {matrix.shape}.")

  identity = jnp.eye(matrix.shape[0])

  return linalg.solve(matrix + damping * identity, identity, assume_a="pos")


def pi_adjusted_kronecker_inverse(
    *arrays: chex.Array,
    damping: chex.Numeric,
    pmap_axis_name: Optional[str],
) -> Tuple[chex.Array, ...]:
  """Computes pi-adjusted factored damping inverses.

  The inverse of `a_1 kron a_2 kron ... kron a_n + damping * I` is not Kronecker
  factored in general, because of the added identity. [1] proposed a pi-adjusted
  factored damping approach to approximate the inverse as a Kronecker product.
  [2] generalized this approach from two to tree factors, and [3] generalized it
  to arbitrary numbers of factors. This function implements the generalized
  approach.

  [1] - https://arxiv.org/abs/1503.05671
  [2] - https://openreview.net/forum?id=SkkTMpjex
  [3] - https://ui.adsabs.harvard.edu/abs/2021arXiv210602925R/abstract

  Args:
    *arrays: A list of matrices, vectors (which are interpreted
      as representing the diagonal of a matrix) or scalars (which are
      interpreted as being a 1x1 matrix). All matrices must be PSD.
    damping: The weight of the identity added to the Kronecker product.
    pmap_axis_name: A `jax.pmap` axis name to use for synchronization.

  Returns:
    A list of factors with the same length as the input `arrays` whose Kronecker
    product approximates the inverse of `a_1 kron ... kron a_n + damping * I`.
  """
  # The implementation writes each single factor as `c_i u_i`, where the matrix
  # `u_i` is such that `trace(u_i) / dim(u_i) = 1`. We then factor out all the
  # scalar factors `c_i` into a single overall scaling coefficient and
  # distribute the damping to each single non-scalar factor `u_i` equally before
  # inverting them.

  # Need the a[None] in order to support a scalar input, with shape ().
  norms = [jnp.sum(a[None]) if a.ndim < 2 else jnp.trace(a) for a in arrays]

  # We need to sync the norms here, because reduction can be non-deterministic.
  # They specifically are on GPUs by default for better performance.
  norms = pmean_if_pmap(norms, pmap_axis_name)

  # Compute the normalized factors `u_i`, such that Trace(u_i) / dim(u_i) = 1
  dims = [1 if a.size == 1 else a.shape[0] for a in arrays]
  us = [ai * di / ni for ai, di, ni in zip(arrays, dims, norms)]

  # kron(arrays) = c * kron(us)
  c = jnp.exp(jnp.sum(jnp.log(jnp.stack(norms)) - jnp.log(jnp.stack(dims))))

  def regular_inverse() -> Tuple[chex.Array, ...]:

    # We distribute the damping only inside the non-scalar factors
    non_scalars = sum(1 if di != 1 else 0 for di in dims)
    d_hat = jnp.power(damping / c, 1.0 / non_scalars)

    # We distribute the overall scale over each factor, including scalars
    if non_scalars == 0:
      # In the case where all factors are scalar we need to add the damping
      c_k = jnp.power(c + damping, 1.0 / len(arrays))
    else:
      c_k = jnp.power(c, 1.0 / len(arrays))

    a_hats_inv = []
    for a in us:

      if a.size == 1:
        inv = jnp.ones_like(a)

      elif a.ndim == 2:
        inv = psd_inv_cholesky(a, d_hat)

      else:
        inv = 1.0 / (a + d_hat)

      a_hats_inv.append(inv / c_k)

    return tuple(a_hats_inv)

  def zero_inverse() -> Tuple[chex.Array, ...]:
    # In the special case where for some reason one of the factors is zero, then
    # the inverse is just `damping^-1 * I`, hence we write each factor as
    # `damping^(1/k) * I`.
    c_k = jnp.power(damping, 1.0 / len(arrays))

    a_hats_inv = []
    for a in us:

      if a.ndim == 2:
        inv = jnp.eye(a.shape[0])

      else:
        inv = jnp.ones_like(a)

      a_hats_inv.append(inv / c_k)

    return tuple(a_hats_inv)

  if get_special_case_zero_inv():
    return lax.cond(
        jnp.greater(c, 0.0),
        regular_inverse,
        zero_inverse)
  else:
    return regular_inverse()


def kronecker_product_axis_mul_v(
    factors: Sequence[chex.Array],
    v: chex.Array,
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
    q_factors: Sequence[chex.Array],
    eigenvalues: chex.Array,
    v: chex.Array,
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
    a: chex.Array,
    b: chex.Array,
    v: chex.Array,
    a_is_symmetric: bool,
) -> chex.Array:
  """Computes `unvec[(a kron b) vec(v)]` for correctly sized input matrices."""
  del a_is_symmetric  # not used
  return kronecker_product_axis_mul_v([b, a], v)


def kronecker_eigen_basis_mul_v(
    q_a: chex.Array,
    q_b: chex.Array,
    eigenvalues: chex.Array,
    v: chex.Array,
) -> chex.Array:
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


def _host_eigh(x: chex.Array, *_) -> Tuple[chex.Array, chex.Array]:
  """This calls the CPU numpy function for eigh."""

  shape_s = jax.ShapeDtypeStruct(x.shape[:-1], x.dtype)
  shape_q = jax.ShapeDtypeStruct(x.shape, x.dtype)

  return jax.pure_callback(np.linalg.eigh, (shape_s, shape_q), x)


def _eigh(
    x: chex.Array,
    force_on_host: bool = False,
) -> Tuple[chex.Array, chex.Array]:
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
    x: chex.Array,
    force_on_host: bool = False,
) -> Tuple[chex.Array, chex.Array]:
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
      lambda _: (jnp.full([d], jnp.nan), jnp.full([d, d], jnp.nan)),
      functools.partial(_eigh, force_on_host=force_on_host),
      x,
  )

  # The matrix is PSD by construction, but numerical inaccuracies can produce
  # slightly negative eigenvalues. Hence, clip at zero.
  return jnp.clip(s, a_min=0.0), q


#  __  __ _____  _____  _____
# |  \/  |_   _|/ ____|/ ____|
# | \  / | | | | (___ | |
# | |\/| | | |  \___ \| |
# | |  | |_| |_ ____) | |____
# |_|  |_|_____|_____/ \_____|


def tree_is_empty(obj: PyTree) -> bool:
  """Returns whether the given PyTree is empty."""
  return not jax.tree_util.tree_leaves(obj)


def abstract_objects_equal(
    obj1: PyTree,
    obj2: PyTree,
    check_dtype: bool = True
) -> bool:
  """`True` if the objects have the same PyTree structure, shapes and dtypes."""
  return (jax.tree_util.tree_structure(obj1) ==
          jax.tree_util.tree_structure(obj2) and
          all(e1.shape == e2.shape and (e1.dtype == e2.dtype or not check_dtype)
              for e1, e2 in zip(jax.tree_util.tree_leaves(obj1),
                                jax.tree_util.tree_leaves(obj2))))


def to_tuple_or_repeat(
    x: Union[chex.Numeric, Sequence[chex.Numeric]],
    length: int,
) -> Tuple[chex.Numeric, ...]:
  """Converts `x` to a tuple of fixed length.

  If `x` is an array, it is split along its last axis to a tuple (assumed to
  have `x.shape[-1] == length`). If it is a scalar, the scalar is repeated
  `length` times into a tuple, and if it is a list or a tuple it is just
  verified that its length is the same.

  Args:
    x: The input array, scalar, list or tuple.
    length: The length of the returned tuple.

  Returns:
    A tuple constructed by either replicating or splitting `x`.
  """
  if isinstance(x, jnp.ndarray) and x.size > 1:  # pytype: disable=attribute-error
    assert x.shape[-1] == length  # pytype: disable=attribute-error
    return tuple(x[..., i] for i in range(length))
  elif isinstance(x, (list, tuple)):
    assert len(x) == length
    return tuple(x)
  elif isinstance(x, (int, float, jnp.ndarray)):
    return (x,) * length
  else:
    raise ValueError(f"Unrecognized type for `x` - {type(x)}.")


def first_dim_is_size(size: int, *args: chex.Array) -> bool:
  """Checks that each element of `args` has first axis size equal to `size`."""
  return all(arg.shape[0] == size for arg in args)


def pytree_dataclass(class_type: Type[Any]) -> Type[Any]:
  """Extended dataclass decorator, which also registers the class as a PyTree.

  The function is equivalent to `dataclasses.dataclass`, but additionally
  registers the `class_type` as a PyTree. This is done done by setting the
  PyTree nodes to all of the `dataclasses.fields` of the class.

  Args:
    class_type: The class type to transform.

  Returns:
    The transformed `class_type` which is now a dataclass and also registered as
    a PyTree.
  """
  class_type = dataclasses.dataclass(class_type)
  fields_names = tuple(field.name for field in dataclasses.fields(class_type))

  def serialize_state(instance) -> Tuple[Tuple[Any, ...], Any]:
    return tuple(getattr(instance, name) for name in fields_names), None

  def deserialize_state(_: Any, args: Sequence[Any]) -> Any:
    return class_type(*args)

  tree_util.register_pytree_node(class_type, serialize_state, deserialize_state)

  return class_type


@pytree_dataclass
class State(object):

  def copy(self):
    """Returns a copy of the PyTree structure (but not the JAX arrays)."""
    (flattened, structure) = jax.tree_util.tree_flatten(self)
    return jax.tree_util.tree_unflatten(structure, flattened)


@pytree_dataclass
class WeightedMovingAverage:
  """A wrapped class for an arbitrary weighted moving average."""
  weight: chex.Array
  raw_value: PyTree

  @property
  def value(self) -> PyTree:
    """The value of the underlying arrays data structure."""
    return jax.tree_util.tree_map(lambda x: x / self.weight, self.raw_value)

  def update(
      self,
      value: PyTree,
      old_weight_multiplier: chex.Numeric,
      new_weight: chex.Numeric,
  ) -> None:
    """Updates the underlying array and weight accordingly."""
    self.weight = self.weight * old_weight_multiplier + new_weight
    self.raw_value = jax.tree_util.tree_map(
        lambda x, y: x * old_weight_multiplier + y * new_weight,
        self.raw_value,
        value,
    )

  def sync(self, pmap_axis_name: Optional[str]) -> None:
    """Syncs the underlying array across devices."""
    self.raw_value = pmean_if_pmap(self.raw_value, pmap_axis_name)

  @classmethod
  def zero(cls, shape: chex.Shape) -> "WeightedMovingAverage":
    """Initializes a `WeightedMovingAverage` with a single array of zeros."""
    return WeightedMovingAverage(
        weight=jnp.zeros([]), raw_value=jnp.zeros(shape))

  @classmethod
  def zeros_like(cls, value: PyTree) -> "WeightedMovingAverage":
    """Initializes a `WeightedMovingAverage` with zeros structure like `value`."""
    return WeightedMovingAverage(
        weight=jnp.zeros([]),
        raw_value=jax.tree_util.tree_map(jnp.zeros_like, value)
    )

  def copy(self):
    """Returns a copy of the PyTree structure (but not the JAX arrays)."""
    (flattened, structure) = jax.tree_util.tree_flatten(self)
    return jax.tree_util.tree_unflatten(structure, flattened)


class MultiChunkAccumulator:
  """Statistics accumulation, abstracted over multiple chunks."""

  def __init__(
      self,
      init_obj_value: Optional[PyTree],
      weight: chex.Numeric,
      multi_device: bool,
  ):
    """Initializes an accumulator instance with the provided object and counter.

    Args:
      init_obj_value: The initial value of the accumulator.
      weight: The initial weight, which specifies how many samples are assumed
        to have been already counted in the initial value of the accumulator.
      multi_device: Whether the objects that are accumulated are outputs of a
        multi-device computation (e.g. `jax.pmap`).
    """
    self._accumulator = init_obj_value
    self._weight = weight
    self._multi_device = multi_device

  @property
  def accumulator(self) -> PyTree:
    """The current value of the underlying not-normalized accumulator."""
    return self._accumulator

  @property
  def weight(self) -> chex.Numeric:
    """The current normalization weight of the underlying accumulator."""
    return self._weight

  @property
  def multi_device(self) -> bool:
    """Whether the accumulator is the output of a multi-device computation."""
    return self._multi_device

  @property
  def value(self) -> PyTree:
    """The current normalized value of the accumulator."""

    if tree_is_empty(self.accumulator):
      return self.accumulator

    if self._multi_device:
      return pmap_sync_and_divide_value(self.accumulator, self.weight)
    else:
      return jit_sync_and_divide_value(self.accumulator, self.weight)

  def clear(self) -> None:
    """Sets the underlying accumulator and weight to `None`."""
    self._accumulator = None
    self._weight = None

  def value_and_clear(self) -> PyTree:
    """Retrieves the normalized value of the accumulator and clears it."""
    value = self.value
    self.clear()
    return value

  def add(self, value_obj: PyTree, weight: chex.Numeric = 1) -> None:
    """Adds an element to the moving average and the max.

    The exact update equation for the statistics are:
      raw_value_t = raw_value_{t-1} + value_obj * weight
      weight_t = weight_{t-1} + weight

    Args:
      value_obj: The value of the object, which scaled by `weight` will be added
        to the accumulator.
      weight: The relative weight of the `value_obj`.
    """
    value_obj = jax.tree_util.tree_map(lambda x: x * weight, value_obj)

    if self._accumulator is None:
      self._accumulator = value_obj
      if isinstance(weight, _CHEX_SCALAR_TYPES):
        self._weight = jnp.full_like(self._weight, weight)
      elif not isinstance(weight, jnp.ndarray):
        raise ValueError("`weight` should be an instance of float, int or "
                         "jnp.ndarray.")
      elif self._weight.shape != weight.shape:
        raise ValueError("If `weight` is an `jnp.ndarray` then should have the "
                         "same shape as the weight of the accumulator.")
      else:
        self._weight = weight
      return

    if not tree_is_empty(self._accumulator):
      if tree_is_empty(value_obj):
        raise ValueError("The provided `value_obj` has an empty PyTree "
                         "structure, but the accumulator has been initialized "
                         "with a non-empty PyTree object.")
      self._accumulator = jax.tree_util.tree_map(
          jnp.add, self._accumulator, value_obj)
    elif not tree_is_empty(value_obj):
      raise ValueError("The provided `value_obj` has a non-empty PyTree "
                       "structure, but the accumulator has been initialized "
                       "with an empty PyTree object.")
    self._weight = self._weight + weight

  @classmethod
  def zeros_like(
      cls,
      obj: PyTree,
      multi_device: bool
  ) -> "MultiChunkAccumulator":
    """Creates a zero initialized accumulator as `obj`."""

    if multi_device:
      value_obj = pmap_zeros_like(obj) if not tree_is_empty(obj) else obj
      weight = replicate_all_local_devices(jnp.zeros([], dtype=jnp.int32))
    else:
      value_obj = jit_zeros_like(obj) if not tree_is_empty(obj) else obj
      weight = jnp.zeros([], dtype=jnp.int32)

    return cls(value_obj, weight, multi_device)

  @classmethod
  def empty(cls, multi_device: bool) -> "MultiChunkAccumulator":
    """Creates an empty accumulator."""

    weight = jnp.zeros([], dtype=jnp.int32)

    if multi_device:
      weight = replicate_all_local_devices(weight)

    return cls(None, weight, multi_device)

  def __repr__(self):
    return (f"{self.__class__.__name__}({self._accumulator!r}, "
            f"{self._weight!r}, {self._multi_device})")

  def copy(self):
    """Returns a copy of the PyTree structure (but not the JAX arrays)."""
    (flattened, structure) = jax.tree_util.tree_flatten(self)
    return jax.tree_util.tree_unflatten(structure, flattened)


tree_util.register_pytree_node(
    MultiChunkAccumulator,
    lambda x: ((x.accumulator, x.weight), (x.multi_device,)),
    lambda fixed, arrays: MultiChunkAccumulator(*arrays, *fixed)
)


class Finalizable(abc.ABC):
  """A mixin for classes that can "finalize" their attributes.

  The class provides the function `finalize` which freezes all attributes of the
  instance after its call. Any attributes assignment thereafter will raise an
  error. All subclasses must always call `super().__init__()` for the mixin to
  function properly, and they must set any attributes before any call to
  `finalize` has happened.
  """

  def __init__(
      self,
      forbid_setting_attributes_after_finalize: bool = True,
      excluded_attribute_names: Sequence[str] = (),
      **parent_kwargs: Any,
  ):
    """Initializes the instance.

    Args:
      forbid_setting_attributes_after_finalize: If `True`, trying to set
        attributes (via direct obj.attr = ...) after `finalize` was called on
        the instance will raise an error. If `False`, this is not checked.
      excluded_attribute_names: When `forbid_setting_attributes_after_finalize`
        is set to `True` this specifies any attributes names that can still be
        set.
      **parent_kwargs: Any keyword arguments to be passed to any parent class.
    """
    self._finalized = False
    self._forbid_setting_attributes = forbid_setting_attributes_after_finalize
    self._excluded_attribute_names = frozenset(excluded_attribute_names)
    super().__init__(**parent_kwargs)

  @property
  def finalized(self) -> bool:
    """Whether the object has already been finalized."""
    return self._finalized  # pytype: disable=attribute-error

  def finalize(self, *args: Any, **kwargs: Any):
    """Finalizes the object, after which no attributes can be set."""

    if self.finalized:
      raise ValueError("Object has already been finalized.")

    self._finalize(*args, **kwargs)
    self._finalized = True

  def _finalize(self, *args: Any, **kwargs: Any):
    """Any logic that a child class needs to do during the finalization."""

  def __setattr__(self, name: str, value: Any):

    if (not getattr(self, "_finalized", False) or
        not getattr(self, "_forbid_setting_attributes", True) or
        name in getattr(self, "_excluded_attribute_names", ())):

      super().__setattr__(name, value)

    else:
      raise AttributeError("Can't set attributes after finalization.")


class WithStagedMethods(Finalizable):
  """An mixin for classes which can have staged/compiled methods."""

  class StagingContext:
    """A context manager for handling methods that are staged/compiled."""

    def __init__(self, wsm_instance: "WithStagedMethods"):
      """Initializes the context manager.

      Args:
        wsm_instance: The corresponding `WithStagedMethods` instance.
      """
      self._wsm_instance = wsm_instance

    def __enter__(self):
      """Enters the staging context."""
      if self._wsm_instance._in_staging:
        raise RuntimeError("Cannot enter staging context while already in "
                           "staging context.")
      self._wsm_instance._in_staging = True

    def __exit__(self, *_):
      """Exits the staging context."""
      assert self._wsm_instance._in_staging, "Exiting while not in staging."
      self._wsm_instance._in_staging = False

  def __init__(
      self,
      multi_device: bool = False,
      pmap_axis_name: Optional[str] = None,
      debug: bool = False,
      **parent_kwargs: Any,
  ):
    """Initializes the instance.

    Args:
      multi_device: Whether any of decorated staged methods are to be run on a
        single or multiple devices. If this is set to `True` than any call
        would internally be delegated to `jax.pmap` and otherwise to  `jax.jit`.
      pmap_axis_name: The name of the pmap axis to use when running on
        multiple devices. This is required if `multi_device=True`.
      debug: If this is set `True` than any call to a stage method would
        directly call the method and would not stage/compile it.
      **parent_kwargs: Any additional keyword arguments for the parent class.
    """
    if "excluded_attribute_names" in parent_kwargs:
      parent_kwargs["excluded_attribute_names"] = (
          ("_in_staging",) + tuple(parent_kwargs["excluded_attribute_names"]))
    else:
      parent_kwargs["excluded_attribute_names"] = ("_in_staging",)

    super().__init__(**parent_kwargs)

    if multi_device and not isinstance(pmap_axis_name, str):
      raise ValueError("When `multi_device=True` you must pass in a string for "
                       "`pmap_axis_name`.")

    self._multi_device = multi_device
    self._pmap_axis_name = pmap_axis_name
    self._debug = debug
    self._in_staging = False

  @property
  def multi_device(self) -> bool:
    """Indicates whether staged method will be run across multiple devices."""
    return self._multi_device

  @property
  def pmap_axis_name(self) -> Optional[str]:
    """The name of the `jax.pmap` axis to use for staged methods."""
    return self._pmap_axis_name

  @property
  def debug(self) -> bool:
    """Whether staged methods would be run in 'debug' mode."""
    return self._debug

  @property
  def in_staging(self) -> bool:
    """Whether we are in a staging context while compiling staged methods."""
    return self._in_staging

  def staging_context(self) -> "StagingContext":
    """Returns a staging context manager, linked to this instance."""
    return self.StagingContext(self)

  def get_first(self, obj: PyTree) -> PyTree:
    """Indexes the `obj` PyTree leaves over leading axis if `multi_device`."""
    return get_first(obj) if self.multi_device else obj

  def copy_obj(self, obj: PyTree) -> PyTree:
    """Copies the object."""
    return pmap_copy_obj(obj) if self.multi_device else copy_obj(obj)

  def replicate(self, obj: PyTree) -> PyTree:
    """Replicates the object to all local devices if `multi_device`."""
    return replicate_all_local_devices(obj) if self.multi_device else obj


def staged(
    method: Callable[..., PyTree],
    static_argnums: Optional[Union[int, Sequence[int]]] = None,
    donate_argnums: Optional[Union[int, Sequence[int]]] = None,
) -> Callable[..., PyTree]:
  """Makes the instance method staged.

  This decorator **should** only be applied to instance methods of classes that
  inherit from the `WithStagedMethods` class. The decorator makes the decorated
  method staged, which is equivalent to `jax.jit` if `instance.multi_device` is
  `False` and to `jax.pmap` otherwise. When specifying static and donated
  argunms, the `self` reference **must not** be counted. Example:

    @functools.partial(staged, donate_argunms=0)
    def try(self, x):
      ...

    then `instance.try(x)` is equivalent to
    `jax.jit(instance.try, donate_argnums=0)(x)` if `instance.multi_device` is
    `False` and to `jax.pmap(instance.try, donate_argnums=0)(x)` otherwise.

  Args:
    method: The method to be transformed into a staged method.
    static_argnums: The static argument numbers, as defined in `jax.jit/pmap`.
    donate_argnums: The donated argument numbers, as defined in
      `jax.jit/pmap`.

  Returns:
    The transformed method, which will now be a staged function.
  """

  if isinstance(static_argnums, int):
    static_argnums = (static_argnums,)

  # This is needed because of b/147015762
  if donate_argnums is None:
    donate_argnums = ()
  if isinstance(donate_argnums, int):
    donate_argnums = (donate_argnums,)
  else:
    donate_argnums: Tuple[int, ...] = tuple(donate_argnums)

  bcast_argnums = static_argnums

  # shift static_argnums by 1 and include instance (self)
  static_argnums = (0,) + tuple(i + 1 for i in (static_argnums or ()))
  # shift donate_argnums by 1 and include state
  donate_argnums = tuple(i + 1 for i in donate_argnums)

  pmap_funcs = {}
  jitted_func = jax.jit(method,
                        static_argnums=static_argnums,
                        donate_argnums=donate_argnums)

  @functools.wraps(method)
  def decorated(instance: "WithStagedMethods", *args: Any) -> PyTree:

    if instance.in_staging:
      return method(instance, *args)

    with instance.staging_context():
      if instance.multi_device and instance.debug:
        # In this case we want to call `method` once for each device index.
        # Note that this might not always produce sensible behavior, and will
        # depend on the details of the method and if it has side effects on the
        # state of the class.

        outs = []
        non_bcast_args = [args[i] if i not in bcast_argnums else None
                          for i in range(len(args))]

        for i in range(jax.local_device_count()):

          non_bcast_args_i = jax.tree_util.tree_map(
              operator.itemgetter(i), non_bcast_args)

          args_i = [
              non_bcast_args_i[j] if j not in bcast_argnums else args[j]
              for j in range(len(args))
          ]

          outs.append(method(instance, *args_i))

        outs = jax.tree_util.tree_map(jnp.stack, *outs)

      elif instance.debug:
        outs = method(instance, *args)

      elif instance.multi_device:

        new_args = list(args)

        for i in range(len(args)):
          if i + 1 not in static_argnums:
            new_args[i] = check_and_fix_format_for_pmap(args[i])

        func = pmap_funcs.get(instance.pmap_axis_name)

        if func is None:
          func = jax.pmap(
              method,
              static_broadcasted_argnums=static_argnums,
              donate_argnums=donate_argnums,
              axis_name=instance.pmap_axis_name,
          )
          pmap_funcs[instance.pmap_axis_name] = func

        outs = func(instance, *new_args)

      else:
        outs = jitted_func(instance, *args)

    return outs

  return decorated


def auto_scope_method(method):
  """Wraps the method call to have automatically generated Jax name scope."""
  @functools.wraps(method)
  def wrapped(instance, *args, **kwargs):
    class_name = type(instance).__name__
    method_name = method.__name__
    if method_name.startswith("_"):
      method_name = method_name[1:]
    with jax.named_scope(f"{class_name}_{method_name}"):
      return method(instance, *args, **kwargs)

  return wrapped


def auto_scope_function(function):
  """Wraps the function call to have automatically generated Jax name scope."""
  @functools.wraps(function)
  def wrapped(*args, **kwargs):
    with jax.named_scope(function.__name__):
      return function(*args, **kwargs)

  return wrapped


def default_batch_size_extractor(
    batch: Batch,
) -> chex.Numeric:

  return jax.tree_util.tree_leaves(batch)[0].shape[0]
