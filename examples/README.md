# KFAC-JAX Examples

This folder contains code with common functionality used in all examples, and
the examples subfolders as well.
Each example follows the following structure:
* `experiment.py` has the model definition, loss definition, and pipeline
experiment class.
* `pipeline.py` has the hyper-parameter configuration.


To run the examples you will need to install additional dependencies via:

```shell
$ pip install -r examples/requirements.txt
```

To run an example simply do:

```shell
$ python example_name/pipeline.py
```

## Autoencoder on MNIST

This example uses the K-FAC optimizer to perform deterministic (i.e. full batch)
training of a deep autoencoder on MNIST.
The default configuration uses the automatic learning rate, momentum, and
damping adaptation techniques from the original K-FAC paper. NOTE: we do *not*
recommend using automatic learning rate and momentum adaptation for stochastic
optimization. Automatic damping adaptation is also of questionable effectiveness
in the stochastic setting.

## Classifier on MNIST

This example uses the K-FAC optimizer to perform deterministic (i.e. full batch)
training of a very small convolutional network for MNIST classification.
The default configuration uses the automatic learning rate, momentum, and
damping adaptation techniques from the original K-FAC paper. NOTE: we do not
recommend using automatic learning rate and momentum adaptation for stochastic
optimization. Automatic damping adaptation is also of questionable effectiveness
in the stochastic setting.

## Resnet50 on ImageNet

This example uses the K-FAC optimizer to perform stochastic training (with
fixed batch size) of a Resnet50 network for ImageNet classification.
The default configuration uses the automatic damping adaptation technique from
the original K-FAC paper.
The momentum is fixed at `0.9` and the learning rate follows an ad-hoc schedule.


## Resnet101 with TAT on ImageNet

This example uses the K-FAC optimizer to perform stochastic training (with
fixed batch size) of a Resnet101 network for ImageNet classification,
with no residual connections or normalization layers as in the
[TAT paper].
The default configuration uses a fixed damping of `0.001`.
The momentum is fixed at `0.9` and the learning rate follows a cosine decay
schedule.

[TAT paper]: https://arxiv.org/abs/2203.08120
