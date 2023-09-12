KFAC-JAX internals
==================

Tracing and detecting layers
----------------------------

Compared to black-box optimizers, K-FAC is not fully
agnostic to the model and loss function being optimized.
The algorithm needs to know what type of loss function it is
optimizing, as this determines the exact form of the curvature matrix.
Additionally, it needs to know which parameters are grouped together into the
same block, and what type of layer that is, in order to select
appropriate block approximation.
As demonstrated in the quickstart section in :doc:`guides <guides>` one must
explicitly register their loss function via library calls.
Layers are detected and registered automatically by default, however there is a
mechanism to do it manually for particular layers.

Computing relative statistics for curvature estimation
------------------------------------------------------

In contrast to standard first-order optimizers, K-FAC requires gradients
with respect to the outputs of each layer in order to compute the curvature
estimate.
This is particularly difficult in JAX, as it does not allow computing of
gradients with respect to intermediate expressions inside a function.
To deal with this, the library traces the provided ``value_and_grad_func`` up to
the registered loss function, discovers all of the layers and creates a new
function, which has additional auxiliary inputs, set to zero, which are added
to the outputs of each layer.
This is what :func:`kfac_jax:layer_tags_vjp` performs and is at the core of the
curvature matrix estimator.

Extending KFAC-JAX
==================

Below we provide a short guide on how to extend the library to new types of
layers, and to implement new curvature blocks.

Creating new layer tags
-----------------------

1. Create a new instance :class:`kfac_jax.LayerTag`, specifying the number
of inputs and outputs the corresponding computation has.

2. Write a function that takes as inputs the parameters, inputs, and outputs of
the computation of your new layer, and binds it with the layer tag you have
defined in the previous step.

3. If you want your tag to be automatically detectable by the library, you will
need to create a new "graph pattern", which tells the library what a computation
for this layer looks like and how to interpret it.
This process can be broken down to the following smaller steps:

    a. Create a function which performs the computation done by your new layer
    in isolation.

    b. Create a function that extracts additional parameters that can be
    relevant for computing the curvature block approximations. (E.g. for
    convolutional layers we have to capture things like dilation and other
    parameters which identify whether this is standard or separable
    convolution.) Note that the input to this function is going the be a
    sequence of JAX equations that are returned by calling
    :func:`jax.make_jaxpr` on the function performing the computation.

    c. Using the functions created in the above steps, create an instance of
    :class:`kfac_jax.GraphPattern`.

4. Provide your graph pattern to the optimizer as part of the
``auto_register_kwargs`` argument.

Below is an example of how one might add support for dense/fully-connected
layers (which are of course already supported by KFAC-JAX) using the above
steps:

.. code-block:: python

    from typing import Sequence
    import chex
    import jax
    import jax.numpy as jnp
    import kfac_jax

    # Step 1
    dense = LayerTag(name="dense_tag", num_inputs=1, num_outputs=1)

    # Step 2
    def register_dense(
        y: chex.Array,
        x: chex.Array,
        w: chex.Array,
        b: Optional[chex.Array] = None,
        **kwargs,
    ) -> chex.Array:
      """Registers a dense layer: ``y = matmul(x, w) + b``."""
      if b is None:
        return dense.bind(y, x, w, **kwargs)
      return dense.bind(y, x, w, b, **kwargs)

    # Step 3.a
    def _dense(x: chex.Array, params: Sequence[chex.Array]) -> chex.Array:
      """Example of a dense layer function."""
      w, *opt_b = params
      y = jnp.matmul(x, w)
      return y if not opt_b else y + opt_b[0]

    # Step 3.b
    def _dense_parameter_extractor(
        eqns: Sequence[core.JaxprEqn],
    ) -> Mapping[str, Any]:
      """Extracts all parameters from the conv_general_dilated operator."""
      for eqn in eqns:
        if eqn.primitive.name == "dot_general":
          return dict(**eqn.params)
      assert False

    # Step 3.c
    dense_with_bias_pattern = GraphPattern(
        name="dense_with_bias",
        tag_primitive=tags.dense,
        precedence=0,
        compute_func=_dense,
        parameters_extractor_func=_dense_parameter_extractor,
        example_args=[np.zeros([11, 13]), [np.zeros([13, 7]), np.zeros([7])]],
    )

    # Step 4
    optimizer = kfac_jax.Optimizer(
        ...
        auto_register_kwargs=dict(
            graph_patterns=((dense_with_bias_pattern,) +
                            kfac_jax.tag_graph_matcher.DEFAULT_GRAPH_PATTERNS),
        ),
        ...
    )

See the `FermiNet project
<https://github.com/google-deepmind/ferminet/blob/jax/ferminet/curvature_tags_and_blocks.py>`_
for another example of how to add a new layer tag using the above steps.

Creating new curvature blocks
-----------------------------

1. Create a new curvature block class by extending
:class:`kfac_jax.CurvatureBlock`.

2. Tell the optimizer which tags should use the new curvature block by providing
a mapping between the name of the tags and the class you created in the previous
step through the ``layer_tag_to_block_ctor`` argument of
:class:`kfac_jax.Optimizer`.

Below is an example of how one might add a standard Kronecker-factored block
approximation of dense layers (which is of course already supported by
KFAC-JAX):

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import kfac_jax

    # Step 1
    class DenseTwoKroneckerFactored(TwoKroneckerFactored):
      """A :class:`~TwoKroneckerFactored` block specifically for dense layers."""

      def input_size(self) -> int:
        """The size of the Kronecker-factor corresponding to inputs."""
        if self.has_bias:
          return self.parameters_shapes[0][0] + 1
        else:
          return self.parameters_shapes[0][0]

      def output_size(self) -> int:
        """The size of the Kronecker-factor corresponding to outputs."""
        return self.parameters_shapes[0][1]

      def update_curvature_matrix_estimate(
          self,
          state: TwoKroneckerFactored.State,
          estimation_data: Mapping[str, Sequence[chex.Array]],
          ema_old: chex.Numeric,
          ema_new: chex.Numeric,
          batch_size: int,
          pmap_axis_name: Optional[str],
      ) -> TwoKroneckerFactored.State:
        del pmap_axis_name
        x, = estimation_data["inputs"]
        dy, = estimation_data["outputs_tangent"]
        assert utils.first_dim_is_size(batch_size, x, dy)

        if self.has_bias:
          x_one = jnp.ones_like(x[:, :1])
          x = jnp.concatenate([x, x_one], axis=1)
        input_stats = jnp.matmul(x.T, x) / batch_size
        output_stats = jnp.matmul(dy.T, dy) / batch_size
        state.inputs_factor.update(input_stats, ema_old, ema_new)
        state.outputs_factor.update(output_stats, ema_old, ema_new)
        return state

    # Step 2
    optimizer = kfac_jax.Optimizer(
        ...
        layer_tag_to_block_ctor=dict(dense_tag=DenseTwoKroneckerFactored),
        ...
    )

See the `FermiNet project
<https://github.com/google-deepmind/ferminet/blob/jax/ferminet/curvature_tags_and_blocks.py>`_
for another example of how to add curvature block using the above steps.
