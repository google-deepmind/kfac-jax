Curvature Estimation
====================

KFAC-JAX supports curvature estimation for neural networks.
Below we list the main components of the library used for this.

Curvature estimator subclasses
------------------------------

The :class:`kfac_jax.CurvatureEstimator` class is
the main abstract class for different estimators of the curvature matrix of a
neural network.
The library currently implements the following three concrete subclasses:

1. :class:`kfac_jax.BlockDiagonalCurvature` - this estimator approximates the
curvature as block diagonal, with blocks corresponding to layers.
This estimator is used by the main optimizer class :class:`kfac_jax.Optimizer`
for computing the preconditioner.

2. :class:`kfac_jax.ExplicitExactCurvature` -  this estimator approximates the
curvature as an explicit dense array.
Note its state is very large (square in the number of parameters) and
infeasible for even moderately sizes models.
By default it will compute the **exact** curvature matrix, which can also be
computationally expensive when the model output or batch size is large.
It is possible to compute stochastic approximations by using different values
of the `estimation_mode` argument.
This class is mainly targeted for small scale toy and demonstrative problems.


3. :class:`kfac_jax.ImplicitExactCurvature` - a special estimator which provides
matrix-vector products functionality with the full curvature matrix, without
materializing it in memory.
This can be very useful in some circumstances, but the class does not implement
product with the inverse of the matrix, or any power other than 1.
This estimator is used during optimization for automatic selection of learning
rate and momentum via the quadratic model.

Block approximations
--------------------

The block diagonal curvature approximation is a very general class, which allows
users to choose different approximations for different layers.
There are three major types of curvature blocks (with corresponding subclasses):

1. :class:`kfac_jax.Diagonal` - approximates the curvature block for the given
layer using only a diagonal matrix.
This type of approximation only considers the curvature for each entry of each
parameter, and ignores interactions between different entries/parameters.
It is used by default for "scale and shift" layers, such as those that commonly
appear in batch norm and layer normalization layers.

2. :class:`kfac_jax.TwoKroneckerFactored` - approximates the curvature block for
the given layer using the Kronecker product of two matrices as in the
`K-FAC paper <https://arxiv.org/abs/1503.05671>`_.
Unlike diagonal approximations, this captures interactions between different
parameter entries within a layer. However, it is an approximation, and
essentially factorizes this interaction as a product of interactions between
input vector entries, and output vector entries. It is practical for most
layers, as long as they aren't very wide, and is used by default for
dense/fully-connected and convolutional layers.

3. :class:`kfac_jax.Full` - explicitly computes the full curvature matrix for
the block parameters.
When the number of parameters is large, using this block can be too memory
intensive, as its state size is square in the number of parameters.
As a result, it is most suitable for layers with small parameterizations.
By default it is not used for any layer type.

Supported layers
----------------

Currently, the library only includes support for the three most common types of
layers used in practice:

1. Dense layers, corresponding to ``y = Wx + b``.
2. 2D convolution layers, corresponding to ``y = W * x + b``.
3. Scale and shift layers, corresponding to ``y = w . x + b``.

Here ``*`` corresponds to convolution and ``.`` to elementwise product.
Parameter reuse, such as in recurrent networks and attention layers, is
not currently supported.

If you want to extend the library with your own layers refer to the
relevant section in :doc:`advanced<advanced>` for how to do this.


Supported losses
----------------

Currently, the library only includes support for the three most common types of
loss functions used in practice:

1. :func:`kfac_jax.register_sigmoid_cross_entropy_loss` specifies that the model
outputs a vector of logits and is trained using the standard sigmoid cross
entropy loss.
This can be interpreted as predicting a factorized Bernoulli distribution over
output labels.
An alias to this is
:func:`kfac_jax.register_multi_bernoulli_predictive_distribution`.

2. :func:`kfac_jax.register_softmax_cross_entropy_loss` specifies
that the model outputs a vector of logits and is trained using the standard
softmax cross entropy loss.
This can be interpreted as predicting a Categorical distribution over output
labels.
An alias to this is
:func:`kfac_jax.register_categorical_predictive_distribution`.

3. :func:`kfac_jax.register_squared_error_loss` specifies
that the model outputs a vector and is trained using the standard squared loss.
This can be interpreted as predicting a Gaussian with a variance of `0.5`.
An alias to this is :func:`kfac_jax.register_normal_predictive_distribution`.

If you want to create and extend the library with your own loss functions
checkout the relevant section in :doc:`advanced<advanced>` on how to do this.

Optimizer
=========

The optimization algorithm implement in :class:`kfac_jax.Optimizer` follows the
`K-FAC paper <https://arxiv.org/abs/1503.05671>`_.
Throughout optimization the optimizer instance keeps the following state::

    C - the curvature estimator state.
    velocity - velocity vectors of the parameters.
    damping - weight of the additional damping added for inverting C.
    counter - a step counter.

If we denote the current minibatch of data by ``x``, the current parameters by
``theta`` and  the function that computes the value and gradient of the loss
by ``f``, a high level pseudocode for a single step of the optimizer
is::

    1 loss, gradient = f(theta, x)
    2 C = update_curvature_estimate(C, theta, x)
    3 preconditioned_gradient = compute_inverse(C) @ gradient
    4 c1, c2 = compute_update_coefficients(theta, x, preconditioned_gradient, velocity)
    5 velocity_new = c1 * preconditioned_gradient + c2 * velocity
    6 theta_new = theta + velocity_new
    7 damping = update_damping(loss, theta_new, C)


Amortizing expensive computations
---------------------------------

When running the optimizer, several of the steps involved can have
a somewhat significant computational overhead.
For this reason, the optimizer class allows these to be performed every `K`
steps, and to cache these values across iterations.
This has been found to work well in practice without significant drawbacks in
training performance.
Specifically, this is applied to computing the inverse of the estimated
approximate curvature (step 3), and to the updates to the damping (step 7).

Computing the update coefficients
---------------------------------

The update coefficients ``c1`` and ``c2`` in step 4 can either be provided
manually by the user at each step, or can be computed automatically using the
procedure described in Section 7 of the original
`K-FAC paper <https://arxiv.org/abs/1503.05671>`_.
This is controlled by the optimizer arguments ``use_adaptive_learning_rate``
and ``use_adaptive_momentum``.
Note that these features don't currently work very well unless you use a very
large batch size, and/or increase the batch size dynamically during training
(as was done in the original K-FAC paper).

Updating the damping
--------------------

The damping update is done via the Levenberg-Marquardt heuristic.
This is done by computing the reduction ratio
``(f(theta_new) - f(theta)) / (q(theta_new) - q_theta)``, where ``q`` is the
quadratic model value induced by either the exact or approximate curvature
matrix.
If the optimizer uses either learning rate or momentum adaptation, or
``always_use_exact_qmodel_for_damping_adjustment`` is set to ``True``, the
optimizer will use the exact curvature matrix; otherwise it will use the
approximate curvature.
If this value deviates too much from ``1`` we either increase or decrease the
damping as described in Section 6.5 from the original
`K-FAC paper <https://arxiv.org/abs/1503.05671>`_.
Whether the damping is adapted, or provided by the user at each single step, is
controlled by the optimizer argument ``use_adaptive_damping``.
