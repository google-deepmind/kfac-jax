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
""""K-FAC loss functions objects, tags and registration functions."""
import abc
from typing import  Optional, Sequence, Tuple

import chex
import distrax
import jax
import jax.numpy as jnp

from kfac_jax._src import layers_and_loss_tags as tags
from kfac_jax._src import utils


class LossFunction(utils.Finalizable):
  """Abstract base class for loss functions.

  Note that unlike typical loss functions used in neural networks these are
  neither summed nor averaged over the batch and the output of evaluate() will
  not be a scalar. It is up to the user to then to correctly manipulate them as
  needed.
  """

  def __init__(self, weight: float):
    """Initializes the loss instance.

    Args:
      weight: The relative weight attributed to the loss.
    """
    super().__init__()
    self._weight = weight
    self.finalize()

  @property
  def weight(self) -> float:
    """The relative weight of the loss."""
    return self._weight

  @property
  @abc.abstractmethod
  def targets(self) -> Optional[chex.Array]:
    """The targets being predicted by the model.

    Returns:
      None or Tensor of appropriate shape for calling self._evaluate() on.
    """

  @property
  @abc.abstractmethod
  def inputs(self) -> Tuple[chex.Array, ...]:
    """The inputs to the loss function (excluding the targets)."""

  @abc.abstractmethod
  def copy_with_different_inputs(
      self,
      inputs: Sequence[chex.Array],
  ) -> "LossFunction":
    """Creates the same :class:`~LossFunction` object, but with different inputs."""

  def evaluate(
      self,
      targets: Optional[chex.Array] = None,
      coefficient_mode: str = "regular",
  ) -> chex.Array:
    """Evaluates the loss function on the targets.

    Args:
      targets: The targets, on which to evaluate the loss. If this is set to
        ``None`` will use ``self.targets`` instead.
      coefficient_mode: Specifies how to use the relative weight of the loss in
        the returned value. There are three options:

        1. 'regular' - returns ``self.weight * loss(targets)``

        2. 'sqrt' - returns ``sqrt(self.weight) * loss(targets)``

        3. 'off' - returns ``loss(targets)``

    Returns:
      The value of the loss scaled appropriately by ``self.weight`` according to
      the coefficient mode.
    Raises:
      ValueError if both ``targets`` and ``self.targets`` are ``None``.
    """
    if targets is None and self.targets is None:
      raise ValueError("Cannot evaluate losses with unspecified targets.")
    elif targets is None:
      targets = self.targets
    if coefficient_mode == "regular":
      multiplier = self.weight
    elif coefficient_mode == "sqrt":
      multiplier = jnp.sqrt(self.weight)
    elif coefficient_mode == "off":
      multiplier = 1.0
    else:
      raise ValueError(f"Unrecognized coefficient_mode={coefficient_mode}.")
    return self._evaluate(targets) * multiplier

  @abc.abstractmethod
  def _evaluate(self, targets: chex.Array) -> chex.Array:
    """Evaluates the value of the loss, disregarding the relative weight."""

  def grad_of_evaluate(
      self,
      targets: Optional[chex.Array],
      coefficient_mode: str,
  ) -> Tuple[chex.Array, ...]:
    """Evaluates the gradient of the loss function, w.r.t. its inputs.

    Args:
      targets: The targets at which to evaluate the loss. If this is ``None``
        will use ``self.targets`` instead.
      coefficient_mode: The coefficient mode to use for evaluation. See
        ``self.evaluate`` for more details.

    Returns:
      The gradient of the loss function w.r.t. its inputs, at the provided
      targets.
    """
    targets = self.targets if targets is None else targets
    def evaluate_sum(inputs: Sequence[chex.Array]) -> chex.Array:
      """Evaluates the loss summed over all axis, including batch etc."""
      instance = self.copy_with_different_inputs(inputs)
      return jnp.sum(instance.evaluate(targets, coefficient_mode))
    return jax.grad(evaluate_sum)(self.inputs)

  def multiply_ggn(
      self,
      vector: Sequence[chex.Array],
  ) -> Tuple[chex.Array, ...]:
    """Right-multiplies a vector by the GGN of the loss function.

    Here the GGN is the Generalized Gauss-Newton matrix (whose definition is
    somewhat flexible) of the loss function with respect to its inputs.

    Args:
      vector: The vector to multiply. Must have the same shape(s) as
        ``self.inputs``.

    Returns:
      The vector right-multiplied by the GGN. Will have the same shape(s) as
      ``self.inputs``.
    """
    return utils.scalar_mul(self.multiply_ggn_unweighted(vector), self.weight)

  @abc.abstractmethod
  def multiply_ggn_unweighted(
      self,
      vector: Sequence[chex.Array],
  ) -> Tuple[chex.Array, ...]:
    """Same as :func:`~LossFunction.multiply_ggn`, disregarding the relative weight."""

  def multiply_ggn_factor(
      self,
      vector: chex.Array,
  ) -> Tuple[chex.Array, ...]:
    """Right-multiplies a vector by a factor B of the GGN.

    Here the GGN is the Generalized Gauss-Newton matrix (whose definition is
    somewhat flexible) of the loss function with respect to its inputs.
    Typically this will be block-diagonal across different cases in the batch,
    since the loss function is typically summed across cases.

    Note that B can be any matrix satisfying ``B * B^T = G`` where ``G`` is the
    GGN, but will agree with the one used in the other methods of this class.

    Args:
      vector: The vector to multiply. Must be of the shape(s) given by
        'self.ggn_factor_inner_shape'.

    Returns:
      The vector right-multiplied by B. Will be of the same shape(s) as
      ``self.inputs``.
    """
    return utils.scalar_mul(
        self.multiply_ggn_factor_unweighted(vector), jnp.sqrt(self.weight))

  @abc.abstractmethod
  def multiply_ggn_factor_unweighted(
      self, vector: chex.Array
  ) -> Tuple[chex.Array, ...]:
    """Same as :func:`~LossFunction.multiply_ggn_factor`, disregarding the relative weight."""

  def multiply_ggn_factor_transpose(
      self,
      vector: Sequence[chex.Array],
  ) -> chex.Array:
    """Right-multiplies a vector by the transpose of a factor B of the GGN.

    Here the GGN is the Generalized Gauss-Newton matrix (whose definition is
    somewhat flexible) of the loss function with respect to its inputs.
    Typically this will be block-diagonal across different cases in the batch,
    since the loss function is typically summed across cases.

    Note that B can be any matrix satisfying ``B * B^T = G`` where G is the GGN,
    but will agree with the one used in the other methods of this class.

    Args:
      vector: The vector to multiply. Must have the same shape(s) as
        ``self.inputs``.

    Returns:
      The vector right-multiplied by B^T. Will be of the shape(s) given by
      ``self.ggn_factor_inner_shape``.
    """
    return utils.scalar_mul(
        self.multiply_ggn_factor_transpose_unweighted(vector),
        jnp.sqrt(self.weight))

  @abc.abstractmethod
  def multiply_ggn_factor_transpose_unweighted(
      self,
      vector: Sequence[chex.Array],
  ) -> chex.Array:
    """Same as :func:`~LossFunction.multiply_ggn_factor_transpose`, disregarding the relative weight."""

  def multiply_ggn_factor_replicated_one_hot(
      self,
      index: Sequence[int],
  ) -> Tuple[chex.Array, ...]:
    """Right-multiplies a replicated-one-hot vector by a factor B of the GGN.

    Here the GGN is the Generalized Gauss-Newton matrix (whose definition is
    somewhat flexible) of the loss function with respect to its inputs.
    Typically this will be block-diagonal across different cases in the batch,
    since the loss function is typically summed across cases.

    A replicated-one-hot vector means a tensor which, for each slice along the
    batch dimension (assumed to be dimension 0), is 1.0 in the entry
    corresponding to the given index and 0 elsewhere.

    Note that B can be any matrix satisfying ``B * B^T = G`` where G is the GGN,
    but will agree with the one used in the other methods of this class.

    Args:
      index: A tuple representing in the index of the entry in each slice that
        is 1.0. Note that len(index) must be equal to the number of elements of
        the ``ggn_factor_inner_shape`` tensor minus one.

    Returns:
      The vector right-multiplied by B^T. Will be of the same shape(s) as the
      ``inputs`` property.
    """
    return utils.scalar_mul(
        self.multiply_ggn_factor_replicated_one_hot_unweighted(index),
        jnp.sqrt(self.weight))

  @abc.abstractmethod
  def multiply_ggn_factor_replicated_one_hot_unweighted(
      self,
      index: Sequence[int],
  ) -> Tuple[chex.Array, ...]:
    """Same as :func:`~LossFunction.multiply_ggn_factor_replicated_one_hot`, disregarding the relative weight."""

  @property
  @abc.abstractmethod
  def ggn_factor_inner_shape(self) -> chex.Shape:
    """The shape of the array returned by `self.multiply_ggn_factor`."""


class NegativeLogProbLoss(LossFunction):
  """Base class for loss functions that represent negative log-probability."""

  @property
  def inputs(self) -> Tuple[chex.Array, ...]:
    return self.params

  @property
  @abc.abstractmethod
  def params(self) -> Tuple[chex.Array, ...]:
    """Parameters to the underlying distribution."""

  def multiply_fisher(
      self,
      vector: Sequence[chex.Array],
  ) -> Tuple[chex.Array, ...]:
    """Right-multiplies a vector by the Fisher.

    Args:
      vector: The vector to multiply. Must have the same shape(s) as
        ``self.inputs``.

    Returns:
      The vector right-multiplied by the Fisher. Will have of the same shape(s)
      as ``self.inputs``.
    """
    return utils.scalar_mul(
        self.multiply_fisher_unweighted(vector), self.weight)

  @abc.abstractmethod
  def multiply_fisher_unweighted(
      self,
      vector: Sequence[chex.Array],
  ) -> Tuple[chex.Array, ...]:
    """Same as :func:`~LossFunction.multiply_fisher`, disregarding the relative weight."""

  def multiply_fisher_factor(
      self,
      vector: chex.Array,
  ) -> Tuple[chex.Array, ...]:
    """Right-multiplies a vector by a factor B of the Fisher.

    Here the Fisher is the Fisher information matrix (i.e. expected outer-
    product of gradients) with respect to the parameters of the underlying
    probability distribution (whose log-prob defines the loss). Typically this
    will be block-diagonal across different cases in the batch, since the
    distribution is usually (but not always) conditionally iid across different
    cases.

    Note that B can be any matrix satisfying ``B * B^T = F`` where F is the
    Fisher, but will agree with the one used in the other methods of this class.

    Args:
      vector: The vector to multiply. Must have the same shape(s) as
        ``self.fisher_factor_inner_shape``.

    Returns:
      The vector right-multiplied by B. Will have the same shape(s) as
      ``self.inputs``.
    """
    return utils.scalar_mul(
        self.multiply_fisher_factor_unweighted(vector), jnp.sqrt(self.weight))

  @abc.abstractmethod
  def multiply_fisher_factor_unweighted(
      self,
      vector: chex.Array,
  ) -> Tuple[chex.Array, ...]:
    """Same as :func:`~LossFunction.multiply_fisher_factor`, disregarding the relative weight."""

  def multiply_fisher_factor_transpose(
      self,
      vector: Sequence[chex.Array],
  ) -> chex.Array:
    """Right-multiplies a vector by the transpose of a factor B of the Fisher.

    Here the Fisher is the Fisher information matrix (i.e. expected outer-
    product of gradients) with respect to the parameters of the underlying
    probability distribution (whose log-prob defines the loss). Typically this
    will be block-diagonal across different cases in the batch, since the
    distribution is usually (but not always) conditionally iid across different
    cases.

    Note that B can be any matrix satisfying ``B * B^T = F`` where F is the
    Fisher, but will agree with the one used in the other methods of this class.

    Args:
      vector: The vector to multiply. Must have the same shape(s) as
        ``self.inputs``.

    Returns:
      The vector right-multiplied by B^T.  Will have the shape given by
      ``self.fisher_factor_inner_shape``.
    """
    return utils.scalar_mul(
        self.multiply_fisher_factor_transpose_unweighted(vector),
        jnp.sqrt(self.weight))

  @abc.abstractmethod
  def multiply_fisher_factor_transpose_unweighted(
      self,
      vector: Sequence[chex.Array],
  ) -> chex.Array:
    """Same as :func:`~LossFunction.multiply_fisher_factor_transpose`, disregarding the relative weight."""

  def multiply_fisher_factor_replicated_one_hot(
      self,
      index: Sequence[int],
  ) -> Tuple[chex.Array, ...]:
    """Right-multiplies a replicated-one-hot vector by a factor B of the Fisher.

    Here the Fisher is the Fisher information matrix (i.e. expected outer-
    product of gradients) with respect to the parameters of the underlying
    probability distribution (whose log-prob defines the loss). Typically this
    will be block-diagonal across different cases in the batch, since the
    distribution is usually (but not always) conditionally iid across different
    cases.

    A replicated-one-hot vector means a tensor which, for each slice along the
    batch dimension (assumed to be dimension 0), is 1.0 in the entry
    corresponding to the given index and 0 elsewhere.

    Note that B can be any matrix satisfying ``B * B^T = H`` where H is the
    Fisher, but will agree with the one used in the other methods of this class.

    Args:
      index: A tuple representing in the index of the entry in each slice that
        is 1.0. Note that len(index) must be equal to the number of elements of
        the ``fisher_factor_inner_shape`` tensor minus one.

    Returns:
      The vector right-multiplied by B. Will have the same shape(s) as
      ``self.inputs``.
    """
    return utils.scalar_mul(
        self.multiply_fisher_factor_replicated_one_hot_unweighted(index),
        jnp.sqrt(self.weight))

  @abc.abstractmethod
  def multiply_fisher_factor_replicated_one_hot_unweighted(
      self,
      index: Sequence[int],
  ) -> Tuple[chex.Array, ...]:
    """Same as :func:`~LossFunction.multiply_fisher_factor_replicated_one_hot`, disregarding the relative weight."""

  @property
  @abc.abstractmethod
  def fisher_factor_inner_shape(self) -> chex.Shape:
    """The shape of the array returned by :func:`~LossFunction.multiply_fisher_factor`."""

  @abc.abstractmethod
  def sample(self, rng: chex.PRNGKey) -> chex.Array:
    """Sample ``targets`` from the underlying distribution."""

  def grad_of_evaluate_on_sample(
      self,
      rng: chex.Array,
      coefficient_mode: str,
  ) -> Tuple[chex.Array, ...]:
    """Evaluates the gradient of the log probability on a random sample.

    Args:
      rng: Jax PRNG key for sampling.
      coefficient_mode: The coefficient mode to use for evaluation.

    Returns:
      The gradient of the log probability of targets sampled from the
      distribution.
    """
    return self.grad_of_evaluate(self.sample(rng), coefficient_mode)


class NaturalParamsNegativeLogProbLoss(NegativeLogProbLoss, abc.ABC):
  """Negative log-probability loss, whose inputs are natural parameters.

  We will take the GGN of the loss to be the Fisher associated with the
  distribution, which also happens to be equal to the Hessian for this class
  of loss functions.  See here: https://arxiv.org/abs/1412.1193

  Natural parameters are defined for exponential-family models. See for
  example `wikipedia <https://en.wikipedia.org/wiki/Exponential_family>`__.
  """

  def multiply_ggn_unweighted(
      self,
      vector: Sequence[chex.Array],
  ) -> Tuple[chex.Array, ...]:
    return self.multiply_fisher_unweighted(vector)

  def multiply_ggn_factor_unweighted(
      self,
      vector: chex.Array,
  ) -> Tuple[chex.Array, ...]:
    return self.multiply_fisher_factor_unweighted(vector)

  def multiply_ggn_factor_transpose_unweighted(
      self,
      vector: Sequence[chex.Array],
  ) -> chex.Array:
    return self.multiply_fisher_factor_transpose_unweighted(vector)

  def multiply_ggn_factor_replicated_one_hot_unweighted(
      self,
      index: Sequence[int],
  ) -> Tuple[chex.Array, ...]:
    return self.multiply_fisher_factor_replicated_one_hot_unweighted(index)

  @property
  def ggn_factor_inner_shape(self) -> chex.Shape:
    return self.fisher_factor_inner_shape


class DistributionNegativeLogProbLoss(NegativeLogProbLoss):
  """Negative log-probability loss that uses a Distrax distribution."""

  @property
  @abc.abstractmethod
  def dist(self) -> distrax.Distribution:
    """The underlying Distrax distribution."""

  def _evaluate(self, targets: chex.Array) -> chex.Array:
    return - self.dist.log_prob(targets)

  def sample(self, rng: chex.PRNGKey) -> chex.Array:
    return self.dist.sample(seed=rng)

  @property
  def fisher_factor_inner_shape(self) -> chex.Shape:
    return jax.eval_shape(
        lambda: self.sample(rng=jax.random.PRNGKey(0))).shape


class NormalMeanNegativeLogProbLoss(DistributionNegativeLogProbLoss,
                                    NaturalParamsNegativeLogProbLoss):
  """Loss log prob loss for a normal distribution parameterized by a mean vector.

  Note that the covariance is treated as the identity divided by 2.
  Also note that the Fisher for such a normal distribution with respect the mean
  parameter is given by:

     F = (1 / variance) * I

  See for example https://www.ii.pwr.edu.pl/~tomczak/PDF/[JMT]Fisher_inf.pdf.
  """

  def __init__(
      self,
      mean: chex.Array,
      targets: Optional[chex.Array] = None,
      variance: float = 0.5,
      weight: float = 1.0,
  ):
    """Initializes the loss instance.

    Args:
      mean: The mean of the normal distribution.
      targets: Optional targets to use for evaluation.
      variance: The scalar variance of the normal distribution.
      weight: The relative weight of the loss.
    """
    if not isinstance(variance, (int, float)):
      raise ValueError("The `variance` argument should be python scalar.")
    self._mean = mean
    self._targets = targets
    self._variance = float(variance)
    super().__init__(weight=weight)

  @property
  def targets(self) -> Optional[chex.Array]:
    return self._targets

  @property
  def dist(self) -> distrax.MultivariateNormalDiag:
    scale_diag = jnp.full_like(self._mean, jnp.sqrt(self._variance))
    return distrax.MultivariateNormalDiag(loc=self._mean, scale_diag=scale_diag)

  @property
  def params(self) -> Tuple[chex.Array]:
    return (self._mean,)

  def copy_with_different_inputs(
      self,
      inputs: Sequence[chex.Array]
  ) -> "NormalMeanNegativeLogProbLoss":
    """Creates the same :class:`~LossFunction` object, but with different inputs.

    Args:
      inputs: The inputs to use to the constructor of a class instance. This
        must be a sequence of length 1.

    Returns:
      An instance of :class:`~NormalMeanNegativeLogPorLoss` with the provided
        inputs.
    Raises:
      A ValueError if the ``inputs`` is a sequence of different length than 1.
    """
    [mean] = inputs
    return NormalMeanNegativeLogProbLoss(
        mean=mean,
        targets=self.targets,
        variance=self._variance,
        weight=self.weight,
    )

  def multiply_fisher_unweighted(
      self,
      vector: Sequence[chex.Array]
  ) -> Tuple[chex.Array]:
    return (vector[0] / self._variance,)

  def multiply_fisher_factor_unweighted(
      self,
      vector: chex.Array,
  ) -> Tuple[chex.Array]:
    return (vector / jnp.sqrt(self._variance),)

  def multiply_fisher_factor_transpose_unweighted(
      self,
      vector: Sequence[chex.Array],
  )  -> chex.Array:
    # it's symmetric
    return self.multiply_fisher_factor_unweighted(vector[0])[0]

  def multiply_fisher_factor_replicated_one_hot_unweighted(
      self,
      index: Sequence[int],
  ) -> Tuple[chex.Array]:
    index = index[0]
    ones_slice = jnp.ones([self._mean.shape[0]])[..., None]
    output_slice = ones_slice / jnp.sqrt(self._variance)
    return (insert_slice_in_zeros(output_slice, 1, self._mean.shape[1], index),)


class NormalMeanVarianceNegativeLogProbLoss(DistributionNegativeLogProbLoss):
  """Negative log prob loss for a normal distribution with mean and variance.

  This class parameterizes a multivariate normal distribution with n independent
  dimensions. Unlike :class:`~NormalMeanNegativeLogProbLoss`, this class does
  not assume the variance is held constant. The Fisher Information for n = 1 is
  given by:

  F = [[1 / variance,                0],
       [           0, 0.5 / variance^2]]

  where the parameters of the distribution are concatenated into a single
  vector as ``[mean, variance]``. For n > 1, the mean parameter vector is
  concatenated with the variance parameter vector. For further details checkout
  the Wikipedia `page
  <https://en.wikipedia.org/wiki/Fisher_information#Multivariate_normal_distribution>`__.
  """

  def __init__(
      self,
      mean: chex.Array,
      variance: chex.Array,
      targets: Optional[chex.Array] = None,
      weight: float = 1.0,
  ):
    """Initializes the loss instance.

    Args:
      mean: The mean of the normal distribution.
      variance: The variance of the normal distribution.
      targets: Optional targets to use for evaluation.
      weight: The relative weight of the loss.
    """
    if mean.ndim != 2:
      raise ValueError("Only 2D mean array is supported.")
    if variance.ndim != 2:
      raise ValueError("Only 2D variance array is supported.")
    self._mean = mean
    self._variance = variance
    self._targets = targets
    super().__init__(weight=weight)

  @property
  def targets(self) -> Optional[chex.Array]:
    return self._targets

  @property
  def dist(self) -> distrax.MultivariateNormalDiag:
    return distrax.MultivariateNormalDiag(
        loc=self._mean, scale_diag=jnp.sqrt(self._variance))

  @property
  def params(self) -> Tuple[chex.Array, chex.Array]:
    return self._mean, self._variance

  def copy_with_different_inputs(
      self,
      inputs: Sequence[chex.Array]
  ) -> "NormalMeanVarianceNegativeLogProbLoss":
    """Creates the same :class:`~LossFunction` object, but with different inputs.

    Args:
      inputs: The inputs to use to the constructor of a class instance. This
        must be a sequence of length 2.

    Returns:
      An instance of :class:`~NormalMeanVarianceNegativeLogProbLoss` with the
      provided inputs.
    Raises:
      A ValueError if the ``inputs`` is a sequence of different length than 2.
    """
    [mean, variance] = inputs
    return NormalMeanVarianceNegativeLogProbLoss(
        mean, variance, self.targets, self.weight)

  @property
  def _fisher_mean(self) -> chex.Array:
    """The Fisher w.r.t. to the mean parameters."""
    return 1. / self._variance

  @property
  def _fisher_mean_factor(self) -> chex.Array:
    """The Fisher factor w.r.t. to the mean parameters."""
    return jnp.sqrt(self._fisher_mean)

  @property
  def _fisher_var(self) -> chex.Array:
    """The Fisher w.r.t. to the variance parameters."""
    return 1. / (2 * jnp.square(self._variance))

  @property
  def _fisher_var_factor(self) -> chex.Array:
    """The Fisher factor w.r.t. to the variance parameters."""
    return 1. / (jnp.sqrt(2.) * self._variance)

  def multiply_fisher_unweighted(
      self,
      vectors: Sequence[chex.Array],
  ) -> Tuple[chex.Array, chex.Array]:
    mean_vec, var_vec = vectors
    return self._fisher_mean * mean_vec, self._fisher_var * var_vec

  def multiply_fisher_factor_unweighted(
      self,
      vector: chex.Array,
  ) -> Tuple[chex.Array, chex.Array]:
    mean_vec, var_vec = jnp.split(vector, 2, axis=-1)
    result_mean_vec = self._fisher_mean_factor * mean_vec
    result_var_vec = self._fisher_var_factor * var_vec
    return result_mean_vec, result_var_vec

  def multiply_fisher_factor_transpose_unweighted(
      self,
      vectors: Sequence[chex.Array],
  ) -> chex.Array:
    mean_vec, var_vec = vectors
    result_mean_vec = self._fisher_mean_factor * mean_vec
    result_var_vec = self._fisher_var_factor * var_vec
    return jnp.concatenate([result_mean_vec, result_var_vec], axis=-1)

  def multiply_fisher_factor_replicated_one_hot_unweighted(
      self,
      index: Sequence[int],
  ) -> Tuple[chex.Array, chex.Array]:
    [index] = index

    if index < int(self._mean.shape[-1]):
      # Index corresponds to mean parameter.
      mean_slice = self._fisher_mean_factor[:, index][..., None]
      mean_output = insert_slice_in_zeros(mean_slice, 1, int(
          self._mean.shape[1]), index)
      var_output = jnp.zeros_like(mean_output)
    else:
      index -= int(self._mean.shape[-1])
      # Index corresponds to variance parameter.
      var_slice = self._fisher_var_factor[:, index][..., None]
      var_output = insert_slice_in_zeros(var_slice, 1,
                                         int(self._variance.shape[1]), index)
      mean_output = jnp.zeros_like(var_output)

    return mean_output, var_output

  @property
  def fisher_factor_inner_shape(self) -> chex.Shape:
    return self._mean.shape[:-1] + self._mean.shape[-1:] * 2

  def multiply_ggn_unweighted(
      self,
      vector: Sequence[chex.Array],
  ) -> Tuple[chex.Array, ...]:
    raise NotImplementedError()

  def multiply_ggn_factor_unweighted(
      self, vector: chex.Array
  ) -> Tuple[chex.Array, ...]:
    raise NotImplementedError()

  def multiply_ggn_factor_transpose_unweighted(
      self,
      vector: Sequence[chex.Array],
  ) -> chex.Array:
    raise NotImplementedError()

  def multiply_ggn_factor_replicated_one_hot_unweighted(
      self,
      index: Sequence[int],
  ) -> Tuple[chex.Array, ...]:
    raise NotImplementedError()

  @property
  def ggn_factor_inner_shape(self) -> chex.Shape:
    raise NotImplementedError()


class MultiBernoulliNegativeLogProbLoss(DistributionNegativeLogProbLoss,
                                        NaturalParamsNegativeLogProbLoss):
  """Negative log prob loss for multiple Bernoulli distributions parametrized by logits.

  Represents N independent Bernoulli distributions where N = len(logits). Its
  Fisher Information matrix is given by ``F = diag(p * (1-p))``, where
  ``p = sigmoid(logits)``.

  As F is diagonal with positive entries, its factor B is
  ``B = diag(sqrt(p * (1-p)))``.
  """

  def __init__(
      self,
      logits: chex.Array,
      targets: Optional[chex.Array] = None,
      weight: float = 1.0,
  ):
    """Initializes the loss instance.

    Args:
      logits: The logits of the Bernoulli distribution.
      targets: Optional targets to use for evaluation.
      weight: The relative weight of the loss.
    """
    self._logits = logits
    self._targets = targets
    super().__init__(weight=weight)

  @property
  def targets(self) -> Optional[chex.Array]:
    return self._targets

  @property
  def dist(self) -> distrax.Bernoulli:
    return distrax.Bernoulli(logits=self._logits, dtype=jnp.int32)

  @property
  def _probs(self) -> chex.Array:
    """The probabilities of the underlying Bernoulli distribution."""
    return self.dist.probs

  @property
  def params(self) -> Tuple[chex.Array]:
    return self._logits,

  def copy_with_different_inputs(
      self,
      inputs: Sequence[chex.Array]
  ) -> "MultiBernoulliNegativeLogProbLoss":
    [logits] = inputs
    return MultiBernoulliNegativeLogProbLoss(
        logits, self.targets, self.weight)

  def multiply_fisher_unweighted(
      self,
      vector: Sequence[chex.Array]
  ) -> Tuple[chex.Array]:
    return (self._probs * (1 - self._probs) * vector[0],)

  def multiply_fisher_factor_unweighted(
      self,
      vector: chex.Array
  ) -> Tuple[chex.Array]:
    return (jnp.sqrt(self._probs * (1 - self._probs)) * vector,)

  def multiply_fisher_factor_transpose_unweighted(
      self,
      vector: Sequence[chex.Array]
  ) -> chex.Array:
    # it's symmetric in this case
    return self.multiply_fisher_factor_unweighted(vector[0])[0]

  def multiply_fisher_factor_replicated_one_hot_unweighted(
      self,
      index: Sequence[int],
  ) -> Tuple[chex.Array]:
    [index] = index
    probs_slice = self._probs[:, index][..., None]
    output_slice = jnp.sqrt(probs_slice * (1 - probs_slice))
    return (insert_slice_in_zeros(
        output_slice, 1, self._logits.shape[1], index),)


class CategoricalLogitsNegativeLogProbLoss(DistributionNegativeLogProbLoss,
                                           NaturalParamsNegativeLogProbLoss):
  """Negative log prob loss for a categorical distribution parameterized by logits.


  Note that the Fisher (for a single case) of a categorical distribution, with
  respect to the natural parameters (i.e. the logits), is given by
  ``F = diag(p) - p*p^T``, where ``p = softmax(logits)``. F can be factorized as
  ``F = B * B^T``, where ``B = diag(q) - p*q^T`` and ``q`` is the entry-wise
  square root of ``p``. This is easy to verify using the fact that ``q^T*q = 1``
  .
  """

  def __init__(
      self,
      logits: chex.Array,
      targets: Optional[chex.Array] = None,
      weight: float = 1.0,
  ):
    """Initializes the loss instance.

    Args:
      logits: The logits of the Categorical distribution of shape
        ``(batch_size, output_size)``.
      targets: Optional targets to use for evaluation, which specify an integer
        index of the correct class. Must be of shape ``(batch_size,)``.
      weight: The relative weight of the loss.
    """
    self._logits = logits
    self._targets = targets
    super().__init__(weight=weight)

  @property
  def targets(self) -> Optional[chex.Array]:
    return self._targets

  @property
  def dist(self) -> distrax.Categorical:
    return distrax.Categorical(logits=self._logits, dtype=jnp.int32)

  @property
  def _probs(self) -> chex.Array:
    """The probabilities of the underlying Bernoulli distribution."""
    return self.dist.probs

  @property
  def _sqrt_probs(self) -> chex.Array:
    """The square root of ``self.probs``."""
    return jnp.sqrt(self._probs)

  @property
  def params(self) -> Tuple[chex.Array]:
    return self._logits,

  @property
  def fisher_factor_inner_shape(self) -> chex.Shape:
    return self._logits.shape

  def copy_with_different_inputs(
      self,
      inputs: Sequence[chex.Array]
  ) -> "CategoricalLogitsNegativeLogProbLoss":
    [logits] = inputs
    return CategoricalLogitsNegativeLogProbLoss(
        logits, self.targets, self.weight)

  def multiply_fisher_unweighted(
      self,
      vector: Sequence[chex.Array]
  ) -> Tuple[chex.Array]:
    probs = self._probs
    fisher_product = vector[0] * probs - probs * jnp.sum(
        vector[0] * probs, axis=-1, keepdims=True)
    return (fisher_product,)

  def multiply_fisher_factor_unweighted(
      self,
      vector: chex.Array
  ) -> Tuple[chex.Array]:
    probs = self._probs
    sqrt_probs = self._sqrt_probs
    return (sqrt_probs * vector - probs * jnp.sum(
        sqrt_probs * vector, axis=-1, keepdims=True),)

  def multiply_fisher_factor_transpose_unweighted(
      self,
      vector: Sequence[chex.Array]
  ) -> chex.Array:
    probs = self._probs
    sqrt_probs = self._sqrt_probs
    return sqrt_probs * vector[0] - sqrt_probs * jnp.sum(
        probs * vector[0], axis=-1, keepdims=True)

  def multiply_fisher_factor_replicated_one_hot_unweighted(
      self,
      index: Sequence[int],
  ) -> Tuple[chex.Array]:
    [index] = index
    probs = self._probs
    sqrt_probs_slice = self._sqrt_probs[:, index][..., None]
    padded_slice = insert_slice_in_zeros(sqrt_probs_slice, 1, probs.shape[1],
                                         index)
    return (padded_slice - probs * sqrt_probs_slice,)


class OneHotCategoricalLogitsNegativeLogProbLoss(
    CategoricalLogitsNegativeLogProbLoss):
  """Neg log prob loss for a categorical distribution with onehot targets.

  Identical to CategoricalLogitsNegativeLogProbLoss except that the underlying
  distribution is OneHotCategorical as opposed to Categorical.
  """

  @property
  def dist(self) -> distrax.OneHotCategorical:
    return distrax.OneHotCategorical(logits=self._logits, dtype=jnp.int32)

  def copy_with_different_inputs(
      self,
      inputs: Sequence[chex.Array]
  ) -> "OneHotCategoricalLogitsNegativeLogProbLoss":
    [logits] = inputs
    return OneHotCategoricalLogitsNegativeLogProbLoss(
        logits, self.targets, self.weight)


def insert_slice_in_zeros(
    slice_to_insert: chex.Array,
    dim: int,
    dim_size: int,
    position: int,
) -> chex.Array:
  """Inserts slice into a larger array of zeros.

  Forms a new array which is the same shape as slice_to_insert, except that
  the dimension given by ``dim`` is expanded to the size given by ``dim_size``.
  ``position`` determines the position (index) at which to insert the slice
  within that dimension.

  Assumes slice_to_insert.shape[dim] = 1.

  Args:
    slice_to_insert: The slice to insert.
    dim: The dimension which to expand with zeros.
    dim_size: The new size of the ``dim`` dimension.
    position: The position of ``slice_to_insert`` in the new tensor.

  Returns:
    The new array.

  Raises:
    ValueError: If the slice's shape at the given dim is not 1.
  """
  slice_shape = slice_to_insert.shape
  if slice_shape[dim] != 1:
    raise ValueError(f"Expected slice_to_insert.shape to have {dim} dim of 1,"
                     f" but was {slice_to_insert.shape[dim]}.")

  before = [0] * len(slice_shape)
  after = before[:]
  before[dim] = position
  after[dim] = dim_size - position - 1
  return jnp.pad(slice_to_insert, list(zip(before, after)))


#  _______            _____            _     _             _   _
# |__   __|          |  __ \          (_)   | |           | | (_)
#    | | __ _  __ _  | |__) |___  __ _ _ ___| |_ _ __ __ _| |_ _  ___  _ __
#    | |/ _` |/ _` | |  _  // _ \/ _` | / __| __| '__/ _` | __| |/ _ \| '_ \
#    | | (_| | (_| | | | \ \  __/ (_| | \__ \ |_| | | (_| | |_| | (_) | | | |
#    |_|\__,_|\__, | |_|  \_\___|\__, |_|___/\__|_|  \__,_|\__|_|\___/|_| |_|
#              __/ |              __/ |
#             |___/              |___/


NormalMeanNegativeLogProbLoss_tag = tags.LossTag(
    NormalMeanNegativeLogProbLoss, num_inputs=1)

NormalMeanVarianceNegativeLogProbLoss_tag = tags.LossTag(
    NormalMeanVarianceNegativeLogProbLoss, num_inputs=2)

CategoricalLogitsNegativeLogProbLoss_tag = tags.LossTag(
    CategoricalLogitsNegativeLogProbLoss, num_inputs=1)

MultiBernoulliNegativeLogProbLoss_tag = tags.LossTag(
    MultiBernoulliNegativeLogProbLoss, num_inputs=1)

OneHotCategoricalLogitsNegativeLogProbLoss_tag = tags.LossTag(
    OneHotCategoricalLogitsNegativeLogProbLoss, num_inputs=1)


def register_normal_predictive_distribution(
    mean: chex.Array,
    targets: Optional[chex.Array] = None,
    variance: float = 0.5,
    weight: float = 1.0,
) -> Tuple[chex.Array, ...]:
  """Registers a normal predictive distribution.

  This corresponds to a squared error loss of the form
     ``weight/(2*var) * ||target - mean||^2``

  Args:
    mean: A tensor defining the mean vector of the distribution. The first
      dimension must be the batch size.
    targets: (OPTIONAL) The targets for the loss function.  Only required if one
      wants to use the "empirical Fisher" instead of the true Fisher (which is
      controlled by the ``estimation_mode`` to the optimizer).
      (Default: None)
    variance: float. The variance of the distribution. Note that the default
      value of 0.5 corresponds to a standard squared error loss weight *
      ||target - prediction||^2. If you want your squared error loss to be of
      the form ``0.5*coeff*||target - prediction||^2`` you should use
      variance=1.0.
      (Default: 0.5)
    weight: A scalar coefficient to multiply the log prob loss associated with
      this distribution. The Fisher will be multiplied by the corresponding
      factor. In general this is NOT equivalent to changing the temperature of
      the distribution, but in the ase of normal distributions it may be.
      (Default: 1.0)

  Returns:
    The mean and targets as dependable on the tag.
  """
  if targets is None:
    targets = jnp.zeros_like(mean)
  return NormalMeanNegativeLogProbLoss_tag.bind(
      mean, targets, variance=variance, weight=weight)


def register_squared_error_loss(
    prediction: chex.Array,
    targets: Optional[chex.Array] = None,
    weight: float = 1.0,
) -> Tuple[chex.Array, ...]:
  """Registers a squared error loss function.

  This assumes the squared error loss of the form ``||target - prediction||^2``,
  averaged across the mini-batch. If your loss uses a coefficient of 0.5
  you need to set the "weight" argument to reflect this.

  Args:
    prediction: The prediction made by the network (i.e. its output). The first
      dimension must be the batch size.
    targets: (OPTIONAL) The targets for the loss function.  Only required if one
      wants to use the "empirical Fisher" instead of the true Fisher (which is
      controlled by the ``estimation_mode`` to the optimizer).
      (Default: None)
    weight: A float coefficient to multiply the loss function by.
      (Default: 1.0)
  Returns:
    The mean and targets as dependable on the tag.
  """
  return register_normal_predictive_distribution(
      prediction, targets, variance=0.5, weight=weight)


def register_multi_bernoulli_predictive_distribution(
    logits: chex.Array,
    targets: Optional[chex.Array] = None,
    weight: float = 1.0,
) -> Tuple[chex.Array, ...]:
  """Registers a multi-Bernoulli predictive distribution.

  Note that this is distinct from
  :func:`~register_categorical_predictive_distribution` and should not be
  confused with it.

  Args:
    logits: The logits of the distribution (i.e. its parameters). The first
      dimension must be the batch size.
    targets: (OPTIONAL) The targets for the loss function.  Only required if
      one wants to use the "empirical Fisher" instead of the true Fisher
      (which is controlled by the ``estimation_mode`` to the optimizer).
      (Default: None)
    weight: (OPTIONAL) a scalar. A coefficient to multiply the log prob loss
      associated with this distribution. The Fisher will be multiplied by the
      corresponding factor. This is NOT equivalent to changing the temperature
      of the distribution since we don't renormalize the log prob in the
      objective function. (Default: 1.0)
  Returns:
    The logits and targets as dependable on the tag.
  """
  if targets is None:
    targets = jnp.zeros_like(logits)
  return MultiBernoulliNegativeLogProbLoss_tag.bind(
      logits, targets, weight=weight)


def register_sigmoid_cross_entropy_loss(
    logits: chex.Array,
    targets: Optional[chex.Array] = None,
    weight: float = 1.0,
) -> Tuple[chex.Array, ...]:
  """Registers a sigmoid cross-entropy loss function.

  Note that this is distinct from :func:`~register_softmax_cross_entropy_loss`
  and should not be confused with it. It is similar to
  :func:`~register_multi_bernoulli_predictive_distribution` but without the
  explicit probabilistic interpretation. It behaves identically for now.

  Args:
    logits: The logits tensor. The first dimension must be the batch size.
    targets: (OPTIONAL) The targets for the loss function.  Only required if
      one wants to use the "empirical Fisher" instead of the true Fisher
      (which is controlled by the ``estimation_mode`` to the optimizer).
      (Default: None)
    weight: (OPTIONAL) a scalar. A coefficient to multiply the loss function by.
      (Default: 1.0)
  Returns:
    The logits and targets as dependable on the tag.
  """
  return register_multi_bernoulli_predictive_distribution(
      logits, targets, weight=weight)


def register_categorical_predictive_distribution(
    logits: chex.Array,
    targets: Optional[chex.Array] = None,
    weight: float = 1.0,
) -> Tuple[chex.Array, ...]:
  """Registers a categorical predictive distribution.

  Note that this is distinct from
  :func:`~register_multi_bernoulli_predictive_distribution` and should not be
  confused with it.

  Args:
    logits: The logits of the distribution (i.e. its parameters). The first
      dimension must be the batch size.
    targets: (OPTIONAL) The targets for the loss function.  Only required if
      one wants to use the "empirical Fisher" instead of the true Fisher
      (which is controlled by the ``estimation_mode`` to the optimizer).
      (Default: None)
    weight: (OPTIONAL) a scalar. A coefficient to multiply the
      log prob loss associated with this distribution. The Fisher will be
      multiplied by the corresponding factor. This is NOT equivalent to
      changing the temperature of the distribution since we don't renormalize
      the log prob in the objective function. (Default: 1.0)
  Returns:
    The logits and targets as dependable on the tag.
  """
  if targets is None:
    targets = jnp.zeros_like(logits[..., 0])
  if targets.ndim == logits.ndim:
    return OneHotCategoricalLogitsNegativeLogProbLoss_tag.bind(
        logits, targets, weight=weight)
  elif targets.ndim == logits.ndim - 1:
    return CategoricalLogitsNegativeLogProbLoss_tag.bind(
        logits, targets, weight=weight)
  else:
    raise ValueError(f"The logits rank is {logits.ndim} and the targets rank "
                     f"must be either equal or one less than it, but is "
                     f"{targets.ndim}.")


def register_softmax_cross_entropy_loss(
    logits: chex.Array,
    targets: Optional[chex.Array] = None,
    weight: float = 1.0,
) -> Tuple[chex.Array, ...]:
  """Registers a softmax cross-entropy loss function.

  Note that this is distinct from :func:`~register_sigmoid_cross_entropy_loss`
  and should not be confused with it. It is similar to
  :func:`~register_categorical_predictive_distribution` but without the explicit
  probabilistic interpretation. It behaves identically for now.

  Args:
    logits: The logits of the distribution (i.e. its parameters). The first
      dimension must be the batch size.
    targets: (OPTIONAL) The targets for the loss function.  Only required if
      one wants to use the "empirical Fisher" instead of the true Fisher
      (which is controlled by the ``estimation_mode`` to the optimizer).
      (Default: None)
    weight: (OPTIONAL) a scalar. A coefficient to multiply the loss function by.
      (Default: 1.0)
  Returns:
    The logits and targets as dependable on the tag.
  """
  return register_categorical_predictive_distribution(logits, targets, weight)
