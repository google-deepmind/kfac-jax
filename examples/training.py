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
"""Jaxline experiment classes and utilities."""
import abc
import copy
import functools
import os
import time
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

from absl import logging
import chex
import jax
import jax.numpy as jnp
from jaxline import experiment
from jaxline import utils as pipe_utils
import kfac_jax
from examples import datasets
from examples import optimizers
from ml_collections import config_dict
import numpy as np


# Types for annotation
InitFunc = Callable[[chex.PRNGKey, kfac_jax.utils.Batch], kfac_jax.utils.Params]


class SupervisedExperiment(experiment.AbstractExperiment):
  """Abstract supervised experiment.

  Attributes:
    mode: Either 'train' or 'eval' specifying whether to run training or
      evaluation of the experiment.
    init_rng: The Jax PRNG key that is used to seed any randomness of the
      experiment.
    config: The experiment config.
    has_aux: Whether the model function returns any auxiliary data.
    has_rng: Whether the model function needs an PRNG key.
    has_func_state: Whether the model function has a state.
    init_parameters_func: A function that initializes the parameters and
      optionally the state of the model if it has one.
    model_loss_func: A function that computes the loss for the model.
    train_model_func: The `model_loss_func` with `is_training` set to `True`.
    eval_model_func: The `model_loss_func` with `is_training` set to `False`.
    eval_batch: A pmapped version of `self._evaluate_single_batch`.
    optimizer: The optimizer instance used for training.
  """

  CHECKPOINT_ATTRS = {
      "_params": "params",
      "_state": "state",
      "_opt_state": "opt_state",
  }

  NON_BROADCAST_CHECKPOINT_ATTRS = {
      "_python_step": "python_step"
  }

  def __init__(
      self,
      mode: str,
      init_rng: chex.PRNGKey,
      config: config_dict.ConfigDict,
      init_parameters_func: InitFunc,
      model_loss_func: kfac_jax.optimizer.ValueFunc,
      has_aux: bool,
      has_rng: bool,
      has_func_state: bool,
  ):
    """Initializes experiment.

    Args:
      mode: Either 'train' or 'eval' specifying whether to run training or
        evaluation of the experiment.
      init_rng: The Jax PRNG key that is used to seed any randomness of the
        experiment.
      config: The experiment config.
      init_parameters_func: A function that initializes the parameters and
        optionally the state of the model if it has one.
      model_loss_func: A function that computes the loss for the model.
      has_aux: Whether the model function returns auxiliary data.
      has_rng: Whether the model function requires an RNG.
      has_func_state: Whether the model function has a state.
    """
    super().__init__(mode=mode, init_rng=init_rng)
    self.mode = mode
    self.init_rng = init_rng
    self.config = config
    self.has_aux = has_aux
    self.has_rng = has_rng
    self.has_func_state = has_func_state
    self.verify_batch_size_config()

    self.init_parameters_func = init_parameters_func
    self.model_loss_func = model_loss_func
    self.train_model_func = functools.partial(
        self.model_loss_func, is_training=True)
    self.eval_model_func = functools.partial(
        self.model_loss_func, is_training=False)
    self.eval_batch = jax.pmap(
        self._evaluate_single_batch, axis_name="eval_axis")

    # Log some useful information
    self.log_machines_setup()

    # Create the optimizer
    self.optimizer = self.create_optimizer()

    # Initialize the state
    self._train_input, self._eval_input = None, None

    self._params, self._state, self._opt_state = None, None, None
    self._python_step = 0
    self.initialize_state()

  def log_machines_setup(self):
    logging.info("Worker with mode %s", self.mode)
    logging.info("Number of hosts[%d]: %d", jax.process_index(),
                 jax.process_count())
    logging.info("Number of devices[%d]: %d/%d", jax.process_index(),
                 jax.local_device_count(), jax.device_count())
    if self.mode == "train":
      logging.info("Training device batch size[%d]: (%d x %d)/%d",
                   jax.process_index(),
                   jax.local_device_count(),
                   self.train_per_device_batch_size,
                   self.train_total_batch_size)
    else:
      logging.info("Evaluation device batch size[%d]: %d/%d",
                   jax.process_index(),
                   self.eval_per_device_batch_size,
                   self.eval_total_batch_size)

  def verify_batch_size_config(self):
    """Verifies that the provided batch size config is valid."""
    if self.config.batch_size.train.total == -1:
      self.config.batch_size.train.total = None
    if self.config.batch_size.train.per_device == -1:
      self.config.batch_size.train.per_device = None
    if self.config.batch_size.eval.total == -1:
      self.config.batch_size.eval.total = None
    if self.config.batch_size.eval.per_device == -1:
      self.config.batch_size.eval.per_device = None
    if (self.config.batch_size.train.total is None ==
        self.config.batch_size.train.per_device is not None):
      raise ValueError("Exactly one of the ``batch_size.train.total`` and "
                       "``batch_size.train.per_device`` config arguments must "
                       "be set to a value and the other one must be ``None``.")
    if (self.config.batch_size.eval.total is None ==
        self.config.batch_size.eval.per_device is None):
      raise ValueError("Exactly one of the ``batch_size.eval.total`` and "
                       "``batch_size.eval.per_device`` config arguments must "
                       "be set to a value and the other one must be ``None``.")

  @property
  @abc.abstractmethod
  def dataset_size(self) -> int:
    """The number of data points in the training set."""

  @property
  @functools.lru_cache(maxsize=1)
  def train_per_device_batch_size(self) -> int:
    """The training per-device batch size."""
    if self.config.batch_size.train.per_device is None:
      if self.config.batch_size.train.total % jax.device_count() != 0:
        raise ValueError("The total batch size must be divisible by the number "
                         "of devices.")
      return self.config.batch_size.train.total // jax.device_count()
    else:
      return self.config.batch_size.train.per_device

  @property
  @functools.lru_cache(maxsize=1)
  def train_host_batch_size(self) -> int:
    """The training per-host batch size."""
    assert self.mode == "train"
    return self.train_per_device_batch_size * jax.local_device_count()

  @property
  @functools.lru_cache(maxsize=1)
  def train_total_batch_size(self) -> int:
    """The training total batch size."""
    return self.train_per_device_batch_size * jax.device_count()

  @property
  @functools.lru_cache(maxsize=1)
  def eval_per_device_batch_size(self) -> int:
    """The evaluator per-device batch size."""
    if self.config.batch_size.eval.per_device is None:
      if self.config.batch_size.eval.total % jax.device_count() != 0:
        raise ValueError("The total batch size must be divisible by the number "
                         "of devices.")
      return self.config.batch_size.eval.total // jax.device_count()
    else:
      return self.config.batch_size.eval.per_device

  @property
  @functools.lru_cache(maxsize=1)
  def eval_host_batch_size(self) -> int:
    """The evaluator per-host batch size."""
    assert self.mode == "eval"
    return self.eval_per_device_batch_size * jax.local_device_count()

  @property
  @functools.lru_cache(maxsize=1)
  def eval_total_batch_size(self) -> int:
    """The evaluator total batch size."""
    return self.eval_per_device_batch_size * jax.device_count()

  @property
  @functools.lru_cache(maxsize=1)
  def train_inputs(self) -> Union[
      Iterator[kfac_jax.utils.Batch],
      Tuple[Iterator[kfac_jax.utils.Batch], Iterator[kfac_jax.utils.Batch]],
  ]:
    """The training data iterator."""
    return self._train_input

  def progress(
      self,
      global_step: chex.Numeric,
      opt_state: Optional[kfac_jax.Optimizer.State] = None,
  ) -> chex.Numeric:
    """Computes the current progress of the optimization."""
    # del opt_state  # not used
    if self.config.training.steps is not None:
      return global_step / self.config.training.steps
    else:
      data_seen = self.train_total_batch_size * global_step
      total_data = self.dataset_size * self.config.training.epochs
      return data_seen / total_data

  def should_run_step(
      self,
      global_step: int,
      config: config_dict.ConfigDict,
  ) -> bool:
    del config  # not used
    return self.progress(global_step, self._opt_state) < 1

  def create_optimizer(self) -> Union[
      optimizers.OptaxWrapper,
      kfac_jax.Optimizer,
  ]:
    """Creates the optimizer specified in the experiment's config."""
    optimizer_config = copy.deepcopy(self.config.optimizer)
    return optimizers.create_optimizer(
        name=self.config.optimizer.name,
        config=optimizer_config,
        train_model_func=self.train_model_func,
        l2_reg=self.config.l2_reg,
        has_aux=self.has_aux,
        has_func_state=self.has_func_state,
        has_rng=self.has_rng,
        dataset_size=self.dataset_size,
        train_total_batch_size=self.train_total_batch_size,
        steps=self.config.training.steps,
        epochs=self.config.training.epochs,
    )

  def initialize_state(self):
    """Initializes all of the experiment's state variables."""
    init_rng, preprocess_rng = jax.random.split(self.init_rng)
    init_rng = kfac_jax.utils.replicate_all_local_devices(init_rng)
    preprocess_rng = jax.random.fold_in(preprocess_rng, jax.process_index())

    # Initialize and load dataset
    if self.mode == "train":
      self._train_input = pipe_utils.py_prefetch(
          datasets.dataset_as_generator(
              self._build_train_input,
              split="train",
              seed=int(preprocess_rng[0]),
              device_batch_size=self.train_per_device_batch_size,
          )
      )
      # Need an example batch for initialization
      init_batch, self._train_input = kfac_jax.utils.fake_element_from_iterator(
          self._train_input)

    elif self.mode == "eval":
      self._eval_input = dict(
          train=self._build_eval_input(
              split="train",
              seed=int(preprocess_rng[0]),
              device_batch_size=self.eval_per_device_batch_size
          ),
          test=self._build_eval_input(
              split="test",
              seed=int(preprocess_rng[0]),
              device_batch_size=self.eval_per_device_batch_size
          ),
      )
      init_batch = next(iter(datasets.tensorflow_datasets.as_numpy(
          self._eval_input["train"])))

    else:
      raise NotImplementedError()

    # Initialize parameters and optional state
    init_func = jax.pmap(self.init_parameters_func)
    params_rng, optimizer_rng = kfac_jax.utils.p_split(init_rng)
    if self.has_func_state:
      self._params, self._state = init_func(params_rng, init_batch)
    else:
      self._params = init_func(params_rng, init_batch)

    # Initialize optimizer state
    self._opt_state = self.optimizer.init(
        self._params, optimizer_rng, init_batch, self._state)

    if not self.has_func_state:
      # Needed for checkpointing
      self._state = kfac_jax.utils.replicate_all_local_devices(
          jax.numpy.zeros([]))

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| "__/ _` | | "_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #
  #
  @abc.abstractmethod
  def _build_train_input(
      self,
      split: str,
      seed: int,
      device_batch_size: int,
      **_: Any,
  ) -> datasets.tf.data.Dataset:
    """Constructs the training dataset."""

  def step(
      self,
      global_step: jnp.ndarray,
      rng: jnp.ndarray,
      **unused_args: Any
  ) -> Dict[str, jnp.ndarray]:
    del global_step  # Instead we use the self._python_step

    # Perform optimizer step
    result = self.optimizer.step(
        params=self._params,
        state=self._opt_state,
        rng=rng,
        data_iterator=self.train_inputs,
        func_state=self._state if self.has_func_state else None,
        global_step_int=self._python_step
    )

    # Unpack result
    if self.has_func_state:
      self._params, self._opt_state, self._state, stats = result
    else:
      self._params, self._opt_state, stats = result

    if "aux" in stats:
      # Average everything in aux and then put it in stats
      stats.update(kfac_jax.utils.compute_mean(stats.pop("aux")))

    stats["progress"] = self.progress(self._python_step, self._opt_state)

    self._python_step += 1
    return kfac_jax.utils.get_first(stats)

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #
  @abc.abstractmethod
  def _build_eval_input(
      self,
      split: str,
      seed: int,
      device_batch_size: int,
      **_: Any,
  ) -> datasets.tf.data.Dataset:
    """Constructs the evaluation dataset."""

  def _evaluate_single_batch(
      self,
      global_step: jnp.ndarray,
      params: kfac_jax.utils.Params,
      func_state: kfac_jax.utils.FuncState,
      opt_state: Union[kfac_jax.Optimizer.State, optimizers.OptaxState],
      rng: chex.PRNGKey,
      batch: kfac_jax.utils.Batch,
  ) -> Dict[str, chex.Array]:
    """Evaluates a single batch."""
    func_args = kfac_jax.optimizer.make_func_args(
        params=params,
        func_state=func_state,
        rng=rng,
        batch=batch,
        has_state=self.has_func_state,
        has_rng=self.has_rng
    )
    _, stats = self.eval_model_func(*func_args)
    stats["progress"] = self.progress(global_step, opt_state)
    return kfac_jax.utils.pmean_if_pmap(stats, "eval_axis")

  def evaluate(
      self,
      global_step: chex.Array,
      rng: chex.PRNGKey,
      writer: Optional[pipe_utils.Writer],
  ) -> Dict[str, chex.Array]:
    del writer  # not used

    # Evaluates both the train and eval split metrics
    all_stats = dict()
    for name, dataset in self._eval_input.items():
      logging.info("Running evaluation for %s", name)

      averaged_stats = kfac_jax.utils.MultiChunkAccumulator.empty(True)
      for batch in datasets.tensorflow_datasets.as_numpy(dataset):
        key, rng = kfac_jax.utils.p_split(rng)
        stats = self.eval_batch(
            global_step, self._params, self._state, self._opt_state, key, batch)
        averaged_stats.add(stats, 1)

      # Extract all stats
      for k, v in averaged_stats.value.items():
        all_stats[f"{name}_{k}"] = kfac_jax.utils.get_first(v)

      logging.info("Evaluation for %s is completed with %d number of batches.",
                   name, int(averaged_stats.weight[0]))

    return jax.tree_map(np.array, all_stats)


def train_standalone_supervised(
    random_seed: int,
    full_config: config_dict.ConfigDict,
    experiment_ctor:
    Callable[[str, chex.PRNGKey, config_dict.ConfigDict], SupervisedExperiment],
    storage_folder: Optional[str],
) -> Dict[str, chex.Array]:
  """Run an experiment without the Jaxline runtime."""
  rng = jax.random.PRNGKey(random_seed)
  rng, init_rng = jax.random.split(rng)
  experiment_instance = experiment_ctor(
      "train", init_rng, full_config.experiment_kwargs.config,
  )
  if storage_folder is not None:
    os.makedirs(storage_folder, exist_ok=True)

  rng = jax.random.fold_in(rng, jax.process_index())
  rng = jax.random.split(rng, jax.local_device_count())
  rng = kfac_jax.utils.broadcast_all_local_devices(rng)
  global_step = jnp.zeros([], dtype=jnp.int32)
  global_step = kfac_jax.utils.replicate_all_local_devices(global_step)
  stats = {}
  start_time = time.time()
  i = 0
  while experiment_instance.should_run_step(i, full_config):
    if (i % full_config.save_checkpoint_interval == 0 and
        storage_folder is not None):
      # Optional save to file
      jnp.savez(f"{storage_folder}/snapshot_{i}.npz",
                *jax.tree_leaves(experiment_instance.snapshot_state()))
    # Run a step
    rng, step_rng = kfac_jax.utils.p_split(rng)
    scalars = experiment_instance.step(global_step, step_rng)
    elapsed_time = jnp.asarray(time.time() - start_time)
    stats["time"] = stats.get("time", []) + [elapsed_time]
    for k in sorted(scalars):
      stats.setdefault(k, []).append(scalars[k])

    # Logging
    if i % full_config.log_tensors_interval == 0:
      for k, v in stats.items():
        if jnp.issubdtype(v[-1].dtype, jnp.integer):
          logging.info("%s: %d", k, v[-1])
        else:
          logging.info("%s: %.3f", k, v[-1])
      logging.info("-" * 20)
    i += 1

  if storage_folder is not None:
    jnp.savez(f"{storage_folder}/snapshot_final.npz",
              *jax.tree_leaves(experiment_instance.snapshot_state()))
    jnp.savez(f"{storage_folder}/stats.npz", **stats)
  return stats


class MnistExperiment(SupervisedExperiment):
  """An experiment using the MNIST dataset."""

  def __init__(
      self,
      supervised: bool,
      flatten_images: bool,
      mode: str,
      init_rng: jnp.ndarray,
      config: config_dict.ConfigDict,
      init_parameters_func: InitFunc,
      model_loss_func: kfac_jax.optimizer.ValueFunc,
      has_aux: bool,
      has_rng: bool,
      has_func_state: bool,
      **kwargs,
  ):
    self._supervised = supervised
    self._flatten_images = flatten_images
    super().__init__(
        mode=mode,
        init_rng=init_rng,
        config=config,
        has_aux=has_aux,
        has_rng=has_rng,
        has_func_state=has_func_state,
        init_parameters_func=init_parameters_func,
        model_loss_func=model_loss_func,
    )

  @property
  @functools.lru_cache(maxsize=1)
  def dataset_size(self) -> int:
    return 60_000

  def _build_train_input(
      self,
      split: str,
      seed: int,
      device_batch_size: int,
      **_: Any,
  ) -> datasets.tf.data.Dataset:
    assert split == "train"
    return datasets.mnist_dataset(
        split=split,
        has_labels=self._supervised,
        flatten_images=self._flatten_images,
        device_batch_size=device_batch_size,
        repeat=True,
        shuffle=True,
        drop_reminder=True,
        seed=seed,
        reshuffle_each_iteration=True,
    )

  def _build_eval_input(
      self,
      split: str,
      seed: int,
      device_batch_size: int,
      **_: Any,
  ) -> datasets.tf.data.Dataset:
    assert split in ("train", "test")
    return datasets.mnist_dataset(
        split=split,
        has_labels=self._supervised,
        flatten_images=self._flatten_images,
        device_batch_size=device_batch_size,
        repeat=False,
        shuffle=False,
        drop_reminder=False,
        seed=seed
    )


class ImageNetExperiment(SupervisedExperiment):
  """An experiment using the ImageNet dataset."""

  def __init__(
      self,
      mode: str,
      init_rng: chex.PRNGKey,
      config: config_dict.ConfigDict,
      init_parameters_func: InitFunc,
      model_loss_func: kfac_jax.optimizer.ValueFunc,
      has_aux: bool,
      has_rng: bool,
      has_func_state: bool,
  ):
    super().__init__(
        mode=mode,
        init_rng=init_rng,
        config=config,
        init_parameters_func=init_parameters_func,
        model_loss_func=model_loss_func,
        has_aux=has_aux,
        has_rng=has_rng,
        has_func_state=has_func_state,
    )

  @property
  @functools.lru_cache(maxsize=1)
  def dataset_size(self) -> int:
    return datasets.imagenet_num_examples_and_split("train_and_valid")[0]

  def _build_train_input(
      self,
      split: str,
      seed: int,
      device_batch_size: int,
      **_: Any,
  ) -> datasets.tf.data.Dataset:
    assert split == "train"
    return datasets.imagenet_dataset(
        split="train_and_valid",
        seed=seed,
        is_training=True,
        batch_dims=(jax.local_device_count(), device_batch_size),
        data_dir=None,
    )

  def _build_eval_input(
      self,
      split: str,
      seed: int,
      device_batch_size: int,
      **_: Any,
  ) -> datasets.tf.data.Dataset:
    assert split in ("train", "test")
    return datasets.imagenet_dataset(
        split="train_eval" if split == "train" else "test",
        seed=seed,
        is_training=False,
        batch_dims=(jax.local_device_count(), device_batch_size),
    )
