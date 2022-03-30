.. currentmodule:: kfac_jax

Standard API
============


Optimizers
----------

.. autosummary::

    Optimizer

Optimizer
~~~~~~~~~

.. autoclass:: Optimizer
   :members:


Curvature Estimators
--------------------

.. autosummary::

    CurvatureEstimator
    BlockDiagonalCurvature
    ExplicitExactCurvature
    ImplicitExactCurvature
    set_default_tag_to_block_ctor
    get_default_tag_to_block_ctor

CurvatureEstimator
~~~~~~~~~~~~~~~~~~

.. autoclass:: CurvatureEstimator
   :members:

BlockDiagonalCurvature
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BlockDiagonalCurvature
   :members:

ExplicitExactCurvature
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ExplicitExactCurvature
   :members:

ImplicitExactCurvature
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ImplicitExactCurvature
   :members:

set_default_tag_to_block_ctor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: set_default_tag_to_block_ctor

get_default_tag_to_block_ctor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: get_default_tag_to_block_ctor


Loss Functions
--------------

.. autosummary::

    LossFunction
    NegativeLogProbLoss
    DistributionNegativeLogProbLoss
    NormalMeanNegativeLogProbLoss
    NormalMeanVarianceNegativeLogProbLoss
    MultiBernoulliNegativeLogProbLoss
    CategoricalLogitsNegativeLogProbLoss
    OneHotCategoricalLogitsNegativeLogProbLoss
    register_sigmoid_cross_entropy_loss
    register_multi_bernoulli_predictive_distribution
    register_softmax_cross_entropy_loss
    register_categorical_predictive_distribution
    register_squared_error_loss
    register_normal_predictive_distribution

LossFunction
~~~~~~~~~~~~

.. autoclass:: LossFunction
   :members:

NegativeLogProbLoss
~~~~~~~~~~~~~~~~~~~

.. autoclass:: NegativeLogProbLoss
   :members:

DistributionNegativeLogProbLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DistributionNegativeLogProbLoss
   :members:

NormalMeanNegativeLogProbLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NormalMeanNegativeLogProbLoss
   :members:

NormalMeanVarianceNegativeLogProbLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NormalMeanVarianceNegativeLogProbLoss
   :members:

MultiBernoulliNegativeLogProbLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MultiBernoulliNegativeLogProbLoss
   :members:

CategoricalLogitsNegativeLogProbLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CategoricalLogitsNegativeLogProbLoss
   :members:

OneHotCategoricalLogitsNegativeLogProbLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: OneHotCategoricalLogitsNegativeLogProbLoss
   :members:

register_sigmoid_cross_entropy_loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: register_sigmoid_cross_entropy_loss

register_multi_bernoulli_predictive_distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: register_multi_bernoulli_predictive_distribution

register_softmax_cross_entropy_loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: register_softmax_cross_entropy_loss

register_categorical_predictive_distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: register_categorical_predictive_distribution

register_squared_error_loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: register_squared_error_loss

register_normal_predictive_distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: register_normal_predictive_distribution


Curvature Blocks
----------------

.. autosummary::

    CurvatureBlock
    ScaledIdentity
    Diagonal
    Full
    TwoKroneckerFactored
    NaiveDiagonal
    NaiveFull
    DenseDiagonal
    DenseFull
    DenseTwoKroneckerFactored
    Conv2DDiagonal
    Conv2DFull
    Conv2DTwoKroneckerFactored
    ScaleAndShiftDiagonal
    ScaleAndShiftFull
    set_max_parallel_elements
    get_max_parallel_elements
    set_default_eigen_decomposition_threshold
    get_default_eigen_decomposition_threshold


CurvatureBlock
~~~~~~~~~~~~~~

.. autoclass:: CurvatureBlock
   :members:

ScaledIdentity
~~~~~~~~~~~~~~

.. autoclass:: ScaledIdentity
   :members:

Diagonal
~~~~~~~~

.. autoclass:: Diagonal
   :members:

Full
~~~~

.. autoclass:: Full
   :members:

TwoKroneckerFactored
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TwoKroneckerFactored
   :members:

NaiveDiagonal
~~~~~~~~~~~~~

.. autoclass:: NaiveDiagonal
   :members:

NaiveFull
~~~~~~~~~

.. autoclass:: NaiveFull
   :members:

DenseDiagonal
~~~~~~~~~~~~~

.. autoclass:: DenseDiagonal
   :members:

DenseFull
~~~~~~~~~

.. autoclass:: DenseFull
   :members:

DenseTwoKroneckerFactored
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DenseTwoKroneckerFactored
   :members:

Conv2DDiagonal
~~~~~~~~~~~~~~

.. autoclass:: Conv2DDiagonal
   :members:

Conv2DFull
~~~~~~~~~~

.. autoclass:: Conv2DFull
   :members:

Conv2DTwoKroneckerFactored
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Conv2DTwoKroneckerFactored
   :members:

ScaleAndShiftDiagonal
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ScaleAndShiftDiagonal
   :members:

ScaleAndShiftFull
~~~~~~~~~~~~~~~~~

.. autoclass:: ScaleAndShiftFull
   :members:

set_max_parallel_elements
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: set_max_parallel_elements

get_max_parallel_elements
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: get_max_parallel_elements

set_default_eigen_decomposition_threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: set_default_eigen_decomposition_threshold

get_default_eigen_decomposition_threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: get_default_eigen_decomposition_threshold


Advanced Features API
=====================


Layer and loss tags
-------------------

.. autosummary::

    LossTag
    LayerTag
    register_generic
    register_dense
    register_conv2d
    register_scale_and_shift

LossTag
~~~~~~~

.. autoclass:: LossTag
   :members:

LayerTag
~~~~~~~~

.. autoclass:: LayerTag
   :members:

register_generic
~~~~~~~~~~~~~~~~

.. autofunction:: register_generic

register_dense
~~~~~~~~~~~~~~

.. autofunction:: register_dense

register_conv2d
~~~~~~~~~~~~~~~

.. autofunction:: register_conv2d

register_scale_and_shift
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: register_scale_and_shift


Automatic Tags Registration
---------------------------

.. autosummary::

    auto_register_tags

auto_register_tags
~~~~~~~~~~~~~~~~~~

.. autofunction:: auto_register_tags


Function tracing and Jacobian computation
-----------------------------------------

.. autosummary::

    ProcessedJaxpr
    loss_tags_vjp
    loss_tags_jvp
    loss_tags_hvp
    layer_tags_vjp


ProcessedJaxpr
~~~~~~~~~~~~~~

.. autoclass:: ProcessedJaxpr
   :members:

loss_tags_vjp
~~~~~~~~~~~~~

.. autofunction:: loss_tags_vjp

loss_tags_jvp
~~~~~~~~~~~~~

.. autofunction:: loss_tags_jvp

loss_tags_hvp
~~~~~~~~~~~~~

.. autofunction:: loss_tags_hvp

layer_tags_vjp
~~~~~~~~~~~~~~

.. autofunction:: layer_tags_vjp


Patches Second Moments
----------------------


.. autosummary::

    patches_moments
    patches_moments_explicit

patches_moments
~~~~~~~~~~~~~~~

.. autofunction:: patches_moments

patches_moments_explicit
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: patches_moments_explicit
