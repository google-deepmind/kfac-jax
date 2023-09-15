# KFAC-JAX - Second Order Optimization with Approximate Curvature in JAX

[**Installation**](#installation)
| [**Quickstart**](#quickstart)
| [**Documentation**](https://kfac-jax.readthedocs.io/)
| [**Examples**](https://github.com/google-deepmind/kfac-jax/tree/main/examples/)
| [**Citing KFAC-JAX**](#citing-kfac-jax)

![CI status](https://github.com/google-deepmind/kfac-jax/workflows/ci/badge.svg)
![docs](https://readthedocs.org/projects/kfac-jax/badge/?version=latest)
![pypi](https://img.shields.io/pypi/v/kfac-jax)

KFAC-JAX is a library built on top of [JAX] for second-order optimization of
neural networks and for computing scalable curvature approximations.
The main goal of the library is to provide researchers with an easy-to-use
implementation of the [K-FAC] optimizer and curvature estimator.

## Installation<a id="installation"></a>

KFAC-JAX is written in pure Python, but depends on C++ code via JAX.

First, follow [these instructions](https://github.com/google/jax#installation)
to install JAX with the relevant accelerator support.

Then, install KFAC-JAX using pip:

```bash
$ pip install git+https://github.com/google-deepmind/kfac-jax
```

Alternatively, you can install via PyPI:

```bash
$ pip install -U kfac-jax
```

Our examples rely on additional libraries, all of which you can install using:

```bash
$ pip install -r requirements_examples.txt
```

## Quickstart<a id="quickstart"></a>

Let's take a look at a simple example of training a neural network, defined
using [Haiku], with the K-FAC optimizer:

```python
import haiku as hk
import jax
import jax.numpy as jnp
import kfac_jax

# Hyper parameters
NUM_CLASSES = 10
L2_REG = 1e-3
NUM_BATCHES = 100


def make_dataset_iterator(batch_size):
  # Dummy dataset, in practice this should be your dataset pipeline
  for _ in range(NUM_BATCHES):
    yield jnp.zeros([batch_size, 100]), jnp.ones([batch_size], dtype="int32")


def softmax_cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray):
  """Softmax cross entropy loss."""
  # We assume integer labels
  assert logits.ndim == targets.ndim + 1

  # Tell KFAC-JAX this model represents a classifier
  # See https://kfac-jax.readthedocs.io/en/latest/overview.html#supported-losses
  kfac_jax.register_softmax_cross_entropy_loss(logits, targets)
  log_p = jax.nn.log_softmax(logits, axis=-1)
  return - jax.vmap(lambda x, y: x[y])(log_p, targets)


def model_fn(x):
  """A Haiku MLP model function - three hidden layer network with tanh."""
  return hk.nets.MLP(
    output_sizes=(50, 50, 50, NUM_CLASSES),
    with_bias=True,
    activation=jax.nn.tanh,
  )(x)


# The Haiku transformed model
hk_model = hk.without_apply_rng(hk.transform(model_fn))


def loss_fn(model_params, model_batch):
  """The loss function to optimize."""
  x, y = model_batch
  logits = hk_model.apply(model_params, x)
  loss = jnp.mean(softmax_cross_entropy(logits, y))

  # The optimizer assumes that the function you provide has already added
  # the L2 regularizer to its gradients.
  return loss + L2_REG * kfac_jax.utils.inner_product(params, params) / 2.0


# Create the optimizer
optimizer = kfac_jax.Optimizer(
  value_and_grad_func=jax.value_and_grad(loss_fn),
  l2_reg=L2_REG,
  value_func_has_aux=False,
  value_func_has_state=False,
  value_func_has_rng=False,
  use_adaptive_learning_rate=True,
  use_adaptive_momentum=True,
  use_adaptive_damping=True,
  initial_damping=1.0,
  multi_device=False,
)

input_dataset = make_dataset_iterator(128)
rng = jax.random.PRNGKey(42)
dummy_images, dummy_labels = next(input_dataset)
rng, key = jax.random.split(rng)
params = hk_model.init(key, dummy_images)
rng, key = jax.random.split(rng)
opt_state = optimizer.init(params, key, (dummy_images, dummy_labels))

# Training loop
for i, batch in enumerate(input_dataset):
  rng, key = jax.random.split(rng)
  params, opt_state, stats = optimizer.step(
      params, opt_state, key, batch=batch, global_step_int=i)
  print(i, stats)
```

### Do not stage (``jit`` or ``pmap``) the optimizer

You should not apply `jax.jit` or `jax.pmap` to the call to `Optimizer.step`.
This is already done for you automatically by the optimizer class.
To control the staging behaviour of the optimizer set the flag ``multi_device``
to ``True`` for ``pmap`` and to ``False`` for ``jit``.

### Do not stage (``jit`` or ``pmap``) the loss function

The ``value_and_grad_func`` argument provided to the optimizer should compute
the loss function value and its gradients. Since the optimizer already stages
its step function internally, applying ``jax.jit`` to ``value_and_grad_func`` is
**NOT** recommended.
Importantly, applying ``jax.pmap`` is **WRONG** and most likely will lead to
errors.

### Registering the model loss function

In order for KFAC-JAX to be able to correctly approximate the curvature matrix
of the model it needs to know the precise loss function that you want to
optimize.
This is done via registration with certain functions provided by the library.
For instance, in the example above this is done via the call to
``kfac_jax.register_softmax_cross_entropy_loss``, which tells the optimizer that
the loss is the standard softmax cross-entropy.
If you don't do this you will get an error when you try to call the optimizer.
For all supported loss functions please read the [documentation].

``Important:`` The optimizer assumes that the loss is averaged over examples in
the minibatch. It is crucial that you follow this convention.

### Other model function options

Oftentimes, one will want to output some auxiliary statistics or metrics in
addition to the loss value.
This can already be done in the ``value_and_grad_func``, in which case we follow
the same conventions as JAX and expect the output to be ``(loss, aux), grads``.
Similarly, the loss function can take an additional function state (batch norm
layers usually have this) or an PRNG key (used in stochastic layers). All of
these, however, need to be explicitly told to the optimizer via its arguments
``value_func_has_aux``, ``value_func_has_state`` and ``value_func_has_rng``.

### Verify optimizer registrations

We strongly encourage the user to pay attention to the logging messages produced
by the automatic registration system, in order to ensure that it has correctly
understood your model.
For the example above this looks like this:

```python
==================================================
Graph parameter registrations:
{'mlp/~/linear_0': {'b': 'Auto[dense_with_bias_3]',
                    'w': 'Auto[dense_with_bias_3]'},
 'mlp/~/linear_1': {'b': 'Auto[dense_with_bias_2]',
                    'w': 'Auto[dense_with_bias_2]'},
 'mlp/~/linear_2': {'b': 'Auto[dense_with_bias_1]',
                    'w': 'Auto[dense_with_bias_1]'},
 'mlp/~/linear_3': {'b': 'Auto[dense_with_bias_0]',
                    'w': 'Auto[dense_with_bias_0]'}}
==================================================
```

As can be seen from this message, the library has correctly detected all
parameters of the model to be part of dense layers.

### Further reading
For a high level overview of the optimizer, the different curvature
approximations, and the supported layers, please see the [documentation].

## Citing KFAC-JAX<a id="citing-kfac-jax"></a>

To cite this repository:

```
@software{kfac-jax2022github,
  author = {Aleksandar Botev and James Martens},
  title = {{KFAC-JAX}},
  url = {https://github.com/google-deepmind/kfac-jax},
  version = {0.0.2},
  year = {2022},
}
```

In this bibtex entry, the version number is intended to be from
[`kfac_jax/__init__.py`](https://github.com/google-deepmind/kfac-jax/blob/main/kfac_jax/__init__.py),
and the year corresponds to the project's open-source release.


[K-FAC]: https://arxiv.org/abs/1503.05671
[JAX]: https://github.com/google/jax
[Haiku]: https://github.com/google-deepmind/dm-haiku
[documentation]: https://kfac-jax.readthedocs.io/
