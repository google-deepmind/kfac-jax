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

1. Dense layers, corresponding to :math:`y = Wx + b`.
2. 2D convolution layers, corresponding to :math:`y = W \star x + b`.
3. Scale and shift layers, corresponding to :math:`y = w \odot x + b`.

Here :math:`\star` corresponds to convolution and :math:`\odot` to elementwise
product.
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

The optimization algorithm implemented in :class:`kfac_jax.Optimizer` follows
the `K-FAC paper <https://arxiv.org/abs/1503.05671>`_.
Throughout optimization the Optimizer instance keeps the following persistent
state:

.. math::
    \begin{aligned}
        & \bm{v}_t - \text{velocity vector, representing the last parameter update.
        } \\
        & \bm{C}_t - \text{The state of the curvature estimator on step } t .\\
        & \lambda_t - \text{ weight of the additional damping added for
        inverting } \bm{C}. \\
        & t - \text{the step counter.}
    \end{aligned}


If we denote the current minibatch of data by :math:`\bm{x}_t`, the current
parameters by :math:`\bm{\theta}_t`, the L2 regularizer by :math:`\gamma` and the
loss function (which includes the L2 regularizer) by :math:`\mathcal{L}`, a
high level pseudocode for a single step of the optimizer is:

.. math::
    \begin{aligned}
        &(1) \quad l_t, \bm{g}_t  = \mathcal{L}(\bm{\theta}_t, \bm{x}_t),
        \nabla_\theta \mathcal{L}(\bm{\theta}_t, \bm{x}_t)
        \\
        &(2) \quad \bm{C}_{t+1} = \text{update curvature}(\bm{C}_t,
        \bm{\theta}_t, \bm{x}_t) \\
        &(3) \quad \hat{\bm{g}}_t = (\bm{C}_{t+1} + (\lambda_t + \gamma) \bm{I}
        )^{-1} \bm{g}_t \\
        &(4) \quad \alpha_t, \beta_t = \text{update coefficients}(
        \hat{\bm{g}}_t, \bm{x}_t, \bm{\theta}_t, \bm{v}_t) \\
        &(5) \quad \bm{v}_{t+1} = \alpha_t \hat{\bm{g}}_t + \beta_t \bm{v}_t \\
        &(6) \quad \bm{\theta}_{t+1} = \bm{\theta}_t + \bm{v}_{t+1} \\
        &(7) \quad \lambda_{t+1} = \text{update damping}(l_t, \bm{\theta}_{t+1},
        \bm{C}_{t+1})
    \end{aligned}

Steps 1, 2, 3, 5 and 6 are standard for any second order optimization algorithm.
Step 4 and 7 are described in more details below.


Computing the update coefficients (4)
-------------------------------------

The update coefficients :math:`\alpha_t` and :math:`\beta_t` in step 4 can
either be provided manually by the user at each step, or can be computed
automatically from the local quadratic model.
This is controlled by the optimizer arguments ``use_adaptive_learning_rate``
and ``use_adaptive_momentum``.
Note that these features don't currently work very well unless you use a very
large batch size, and/or increase the batch size dynamically during training
(as was done in the original K-FAC paper).

Automatic selection of update coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The procedure to automatically select the update coefficients uses the local
quadratic model defined as:

.. math::
    q(\bm{\delta}) = l_t + \bm{g}_t^T \bm{\delta} + \frac{1}{2} \bm{\delta}^T
    (\bm{C} + (\lambda_t + \gamma) \bm{I}) \bm{\delta},

where :math:`\bm{C}` is usually the exact curvature matrix.
To compute :math:`\alpha_t` and :math:`\beta_t`, we minimize
:math:`q(\alpha_t \hat{\bm{g}}_t + \beta_t \bm{v}_t)` with respect to the two
scalars, treating :math:`\hat{\bm{g}}_t` and :math:`\bm{v}_t` as fixed vectors.
Since this is a simple two dimensional quadratic problem, and it requires only
matrix-vector products with :math:`\bm{C}`, it can be solved efficiently.
For further details see Section 7 of the original
`K-FAC paper <https://arxiv.org/abs/1503.05671>`_.


Updating the damping (7)
------------------------

The damping update is done via the Levenberg-Marquardt heuristic.
This is done by computing the reduction ratio:

.. math::
    \rho = \frac{\mathcal{L}(\bm{\theta}_{t+1}) - \mathcal{L}(\bm{\theta}_{t})}
    {q(\bm{v}_{t+1}) - q(\bm{0})}

where :math:`q` is the quadratic model value induced by either the exact or
approximate curvature matrix.
If the optimizer uses either learning rate or momentum adaptation, or
``always_use_exact_qmodel_for_damping_adjustment`` is set to ``True``, the
optimizer will use the exact curvature matrix; otherwise it will use the
approximate curvature.
If the value of :math:`\rho` deviates too much from 1 we either increase or
decrease the damping :math:`\lambda` as described in Section 6.5 of the original
`K-FAC paper <https://arxiv.org/abs/1503.05671>`_.
Whether the damping is adapted, or provided by the user at each single step, is
controlled by the optimizer argument ``use_adaptive_damping``.


Amortizing expensive computations
---------------------------------

When running the optimizer, several of the steps involved can have
a noticeable computational overhead.
For this reason, the optimizer class allows these to be performed every `K`
steps, and to cache the values across iterations.
This has been found to work well in practice without significant drawbacks in
training performance.
This is applied to computing the inverse of the estimated approximate curvature
(step 3), and to the updates of the damping (step 7).
