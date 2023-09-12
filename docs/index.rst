:github_url: https://github.com/google-deepmind/kfac-jax/tree/main/docs

KFAC-JAX Documentation
======================

KFAC-JAX is a library built on top of `JAX <https://github.com/google/jax>`_ for
second-order optimization of neural networks and for computing scalable
curvature approximations.
The main goal of the library is to provide researchers with an easy-to-use
implementation of the `K-FAC paper <https://arxiv.org/abs/1503.05671>`_
optimizer and curvature estimator.


Installation
------------

KFAC-JAX is written in pure Python, but depends on C++ code via JAX.

First, follow `these instructions <https://github.com/google/jax#installation>`_
to install JAX with the relevant accelerator support.

Then, install KFAC-JAX using pip::

    $ pip install git+https://github.com/google-deepmind/kfac-jax

Alternatively, you can install via PyPI::

    $ pip install -U kfac-jax

Our examples rely on additional libraries, all of which you can install using::

    $ pip install -r requirements_examples.txt

.. toctree::
   :caption: Guides
   :maxdepth: 1

   guides

.. toctree::
   :caption: High Level Overview
   :maxdepth: 1

   overview

.. toctree::
   :caption: API Documentation
   :maxdepth: 1

   api

.. toctree::
   :caption: Advanced Topics
   :maxdepth: 1

   advanced

Contribute
----------

We are not taking code contributions from Github at this time. All PRs from
Github will be rejected.
If you find a bug, or for any feature requests and ideas, please email us at
[kfac-jax-dev@google.com](mailto:kfac-jax-dev@google.com), or raise an issue on
the Github issue tracker.

- Issue tracker: https://github.com/google-deepmind/kfac-jax/issues
- Source code: https://github.com/google-deepmind/kfac-jax/tree/main

Support
-------

If you are having issues, please let us know by filing an issue on our
`issue tracker <https://github.com/google-deepmind/kfac-jax/issues>`_.

License
-------

KFAC-JAX is licensed under the Apache 2.0 License.

Indices and tables
==================

* :ref:`genindex`
