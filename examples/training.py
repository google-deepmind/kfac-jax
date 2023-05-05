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
from typing import Any, Callable, Iterator, Optional, Tuple, Union, Dict

from absl import logging
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jaxline import experiment
from jaxline import utils as pipe_utils
import kfac_jax
from examples import datasets
from examples import optimizers
import ml_collections


# Types for annotation
Array = kfac_jax.utils.Array
Numeric = kfac_jax.utils.Numeric
PRNGKey = kfac_jax.utils.PRNGKey
Params = kfac_jax.utils.Params
Batch = kfac_jax.utils.Batch
FuncState = kfac_jax.utils.FuncState

InitFunc = Callable[[PRNGKey, Batch], Params]


class SupervisedExperiment(abc.ABC):
  """Abstract supervised experiment.

  Attributes:
    mode: Either 'train' or 'eval' specifying whether to run training or
      evaluation of the experiment.
    init_rng: The Jax PRNG key that is used to seed the initialization of the
      model parameters.
    seed_rng: An RNG used fo seeding the dataset iterators.
    config: The experiment config.
    has_aux: Whether the model function returns any auxiliary data.
    has_rng: Whether the model function needs an PRNG key.
    has_func_state: Whether the model function has a state.
    eval_splits: Evaluation splits of the evaluation dataset loader.
    init_parameters_func: A function that initializes the parameters and
      optionally the state of the model if it has one.
    params_init: A function that initializes the model parameters.
    model_loss_func: A function that computes the loss for the model.
    train_model_func: The `model_loss_func` with `is_training` set to `True`.
    eval_model_func: The `model_loss_func` with `is_training` set to `False`.
    eval_batch: A pmapped version of `self._evaluate_single_batch`.
    optimizer: The optimizer instance used for training.
  """

  def __init__(
      self,
      mode: str,
      init_rng: PRNGKey,
      config: ml_collections.ConfigDict,
      init_parameters_func: InitFunc,
      model_loss_func: kfac_jax.optimizer.ValueFunc,
      has_aux: bool,
      has_rng: bool,
      has_func_state: bool,
      eval_splits: Tuple[str, ...] = ("train", "test"),
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
      eval_splits: Evaluation splits of the evaluation dataset loader.
    """
    self.mode = mode
    self.init_rng, self.seed_rng = jax.random.split(init_rng)
    self.seed_rng = jax.random.fold_in(self.seed_rng, jax.process_index())
    self.config = config
    self.has_aux = has_aux
    self.has_rng = has_rng
    self.has_func_state = has_func_state
    self.eval_splits = eval_splits
    self.verify_batch_size_config()

    self.params_init = jax.jit(init_parameters_func,
                               out_shardings=self.model_sharding)
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
    self._train_input, self._eval_input, self._init_batch = None, None, None

    self._params, self._state, self._opt_state = None, None, None
    self._python_step = 0

  def log_machines_setup(self):
    """Logs the machine setup for the experiment."""
    logging.info("Worker with mode %s", self.mode)
    logging.info("Number of hosts[%d]: %d", jax.process_index(),
                 jax.process_count())
    logging.info("Number of devices[%d]: %d/%d", jax.process_index(),
                 jax.local_device_count(), jax.device_count())
    if self.mode == "train":
      logging.info("Training device batch size[%d]: (%d x %d)/%d",
                   jax.process_index(),
                   self.train_num_devices,
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

  @functools.cached_property
  def sharding_config(self) -> ml_collections.ConfigDict:
    """The sharding config."""
    default_config = ml_collections.ConfigDict(dict(
        mesh_shape=(jax.device_count(),),
        mesh_axis=("batch",),
        dataset_axis=("batch",),
        model_axis=(),
        optimizer_axis=(),
    ))
    config = self.config.get("sharding", default_config)
    config.update(default_config)
    return config

  @functools.cached_property
  def jit_mesh(self) -> jax.sharding.Mesh:
    """The device mesh used when calling `jax.jit`."""
    devices = mesh_utils.create_device_mesh(self.sharding_config.mesh_shape)
    return jax.sharding.Mesh(devices, self.sharding_config.mesh_axis)

  @functools.cached_property
  def dataset_sharding_spec(self) -> jax.sharding.PartitionSpec:
    """The sharding specification for the dataset."""
    axis = [(None if name not in self.sharding_config.dataset_axis else name)
            for name in self.sharding_config.mesh_axis]
    return jax.sharding.PartitionSpec(*axis)

  @functools.cached_property
  def dataset_sharding(self) -> jax.sharding.NamedSharding:
    """The NamedSharding for the dataset."""
    return jax.sharding.NamedSharding(self.jit_mesh, self.dataset_sharding_spec)

  @functools.cached_property
  def model_sharding_spec(self) -> jax.sharding.PartitionSpec:
    """The sharding specification for the model."""
    axis = [(None if name not in self.sharding_config.model_axis else name)
            for name in self.sharding_config.mesh_axis]
    return jax.sharding.PartitionSpec(*axis)

  @functools.cached_property
  def model_sharding(self) -> jax.sharding.NamedSharding:
    """The NamedSharding for the model."""
    return jax.sharding.NamedSharding(self.jit_mesh, self.model_sharding_spec)

  @functools.cached_property
  def optimizer_sharding_spec(self) -> jax.sharding.PartitionSpec:
    """The sharding specification for the optimizer."""
    axis = [
        (None if name not in self.sharding_config.optimizer_axis else name)
        for name in self.sharding_config.mesh_axis
    ]
    return jax.sharding.PartitionSpec(*axis)

  @functools.cached_property
  def optimizer_state_sharding(self) -> jax.sharding.NamedSharding:
    """The NamedSharding for the optimizer state."""
    return self.optimizer.state_sharding

  @property
  @abc.abstractmethod
  def dataset_size(self) -> int:
    """The number of data points in the training set."""

  @property
  @functools.lru_cache(maxsize=1)
  def train_num_local_devices(self) -> int:
    """The number of training local devices."""
    return jax.local_device_count()

  @property
  @functools.lru_cache(maxsize=1)
  def train_num_devices(self) -> int:
    """The number of training devices."""
    return jax.device_count()

  @property
  @functools.lru_cache(maxsize=1)
  def train_per_device_batch_size(self) -> int:
    """The training per-device batch size."""
    if self.config.batch_size.train.per_device is None:
      if self.config.batch_size.train.total % self.train_num_devices != 0:
        raise ValueError("The total batch size must be divisible by the number "
                         "of devices.")
      return self.config.batch_size.train.total // self.train_num_devices
    else:
      return self.config.batch_size.train.per_device

  @property
  @functools.lru_cache(maxsize=1)
  def train_host_batch_size(self) -> int:
    """The training per-host batch size."""
    assert self.mode == "train"
    return self.train_per_device_batch_size * self.train_num_local_devices

  @property
  @functools.lru_cache(maxsize=1)
  def train_total_batch_size(self) -> int:
    """The training total batch size."""
    return self.train_per_device_batch_size * self.train_num_devices

  @property
  @functools.lru_cache(maxsize=1)
  def eval_num_local_devices(self) -> int:
    """The evaluator number of local devices."""
    return jax.local_device_count()

  @property
  @functools.lru_cache(maxsize=1)
  def eval_num_devices(self) -> int:
    """The evaluator number of devices."""
    return jax.device_count()

  @property
  @functools.lru_cache(maxsize=1)
  def eval_per_device_batch_size(self) -> int:
    """The evaluator per-device batch size."""
    if self.config.batch_size.eval.per_device is None:
      if self.config.batch_size.eval.total % jax.device_count() != 0:
        raise ValueError("The total batch size must be divisible by the number "
                         "of devices.")
      return self.config.batch_size.eval.total // self.num_eval_devices
    else:
      return self.config.batch_size.eval.per_device

  @property
  @functools.lru_cache(maxsize=1)
  def eval_host_batch_size(self) -> int:
    """The evaluator per-host batch size."""
    assert self.mode == "eval"
    return self.eval_per_device_batch_size * self.eval_num_local_devices

  @property
  @functools.lru_cache(maxsize=1)
  def eval_total_batch_size(self) -> int:
    """The evaluator total batch size."""
    return self.eval_per_device_batch_size * self.num_eval_devices

  @property
  @functools.lru_cache(maxsize=1)
  def train_input(self) -> Iterator[Batch]:
    """Returns the current training iterator."""
    if self._train_input is None:
      logging.info("Initializing data iterators.")
      seed_rng = jax.random.fold_in(self.seed_rng, self._python_step)
      self._train_input = pipe_utils.py_prefetch(
          functools.partial(
              self._build_train_input,
              split="train",
              seed=int(seed_rng[0]),
              device_batch_size=self.train_per_device_batch_size,
          )
      )
    return self._train_input

  @property
  @functools.lru_cache(maxsize=1)
  def train_inputs(self) -> Union[Iterator[Batch],
                                  Tuple[Iterator[Batch], Iterator[Batch]]]:
    """The training data iterator."""
    return self.train_input

  @property
  @functools.lru_cache(maxsize=1)
  def eval_input(self) -> Dict[str, Callable[[], Iterator[Batch]]]:
    """"Returns all evaluation iterators constructors."""
    if self._eval_input is None:
      seed_rng = jax.random.fold_in(self.seed_rng, self._python_step)
      self._eval_input = {}
      for split in self.eval_splits:
        self._eval_input[split] = functools.partial(
            self._build_eval_input,
            split="train",
            seed=int(seed_rng[1]),
            device_batch_size=self.eval_per_device_batch_size,
        )
    return self._eval_input

  @property
  @functools.lru_cache(maxsize=1)
  def init_batch(self) -> Batch:
    """A fake batch size used to initialize the model parameters and state."""
    if self._init_batch is None:
      if self.mode == "train":
        self._init_batch, iterator = kfac_jax.utils.fake_element_from_iterator(
            self.train_input)
        self._train_input = iterator
      else:
        self._init_batch = next(self.eval_input["train"]())
    return self._init_batch

  def progress(
      self,
      global_step: Numeric,
  ) -> Numeric:
    """Computes the current progress of the training as a number in [0,1]."""

    if self.config.training.steps is not None:
      return global_step / self.config.training.steps

    else:
      data_seen = self.train_total_batch_size * global_step
      total_data = self.dataset_size * self.config.training.epochs

      return data_seen / total_data

  def terminate_training(
      self,
      global_step: int,
      config: ml_collections.ConfigDict,
  ) -> bool:

    del config  # not used

    return int(self.progress(global_step)) >= 1

  def create_optimizer(self) -> Union[optimizers.OptaxWrapper,
                                      kfac_jax.Optimizer]:
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

  def maybe_initialize_state(self):
    """Initializes all the experiment's state variables."""
    if self._params is not None:
      logging.info("Loaded from checkpoint, not initializing parameters.")
      return

    # Initialize parameters and optional state
    params_rng, optimizer_rng = jax.random.split(self.init_rng)
    if self.has_func_state:
      self._params, self._state = self.params_init(params_rng, self.init_batch)
    else:
      self._params = self.params_init(params_rng, self.init_batch)

    # Initialize optimizer state
    self._opt_state = self.optimizer.init(
        self._params, optimizer_rng, self.init_batch, self._state)

    if not self.has_func_state:
      # Needed for checkpointing
      self._state = ()

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

  def train_step(self, global_step: Array, rng: PRNGKey) -> Dict[str, Numeric]:
    """Performs a single training step."""
    del global_step  # Unused

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
      stats.update(stats.pop("aux", {}))

    stats["progress"] = self.progress(self._python_step)

    self._python_step += 1

    for name in self.config.get("per_device_stats_to_log", []):

      gathered_stat = jnp.reshape(
          kfac_jax.utils.host_all_gather(stats[name]), [-1])

      for i in range(gathered_stat.shape[0]):
        stats[f"{name}_{i}"] = jnp.array([gathered_stat[i]])

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
      global_step: Array,
      params: Params,
      func_state: FuncState,
      opt_state: Union[kfac_jax.Optimizer.State, optimizers.OptaxState],
      rng: PRNGKey,
      batch: Batch,
  ) -> Dict[str, Array]:
    """Evaluates a single batch."""

    del global_step  # This might be used in subclasses

    func_args = kfac_jax.optimizer.make_func_args(
        params=params,
        func_state=func_state,
        rng=rng,
        batch=batch,
        has_state=self.has_func_state,
        has_rng=self.has_rng
    )

    loss, stats = self.eval_model_func(*func_args)

    stats["loss"] = loss

    if hasattr(opt_state, "data_seen"):
      stats["data_seen"] = opt_state.data_seen

    return stats

  def run_evaluation(
      self,
      global_step: Array,
      rng: PRNGKey,
  ) -> Dict[str, Numeric]:
    """Runs the evaluation of the currently loaded model parameters."""
    all_stats = dict()

    # Evaluates both the train and eval split metrics
    for name, dataset_iter_thunk in self.eval_input.items():  # pytype: disable=attribute-error

      logging.info("Running evaluation for %s", name)

      averaged_stats = kfac_jax.utils.MultiChunkAccumulator.empty(True)

      for batch in dataset_iter_thunk():

        key, rng = kfac_jax.utils.p_split(rng)

        stats = self.eval_batch(
            global_step, self._params, self._state, self._opt_state, key, batch)

        averaged_stats.add(stats, 1)

      # Extract all stats
      for k, v in averaged_stats.value.items():  # pytype: disable=attribute-error
        all_stats[f"{name}_{k}"] = kfac_jax.utils.get_first(v)

      logging.info("Evaluation for %s is completed with %d number of batches.",
                   name, int(averaged_stats.weight[0]))

    all_stats["progress"] = self.progress(self._python_step)

    return all_stats


class JaxlineExperiment(SupervisedExperiment, experiment.AbstractExperiment):
  """A Jaxline supervised experiment."""

  CHECKPOINT_ATTRS = {
      "_params": "params",
      "_state": "state",
      "_opt_state": "opt_state",
  }

  NON_BROADCAST_CHECKPOINT_ATTRS = {
      "_python_step": "python_step"
  }

  def should_run_step(
      self,
      global_step: int,
      config: ml_collections.ConfigDict,
  ) -> bool:
    return not self.terminate_training(global_step, config)

  def step(  # pytype: disable=signature-mismatch
      self,
      global_step: Array,
      rng: PRNGKey,
      **unused_kwargs,
  ) -> Dict[str, Numeric]:
    self.maybe_initialize_state()
    return self.train_step(global_step, rng)

  def evaluate(  # pytype: disable=signature-mismatch
      self,
      global_step: Array,
      rng: PRNGKey,
      **unused_kwargs,
  ) -> Dict[str, Numeric]:
    return self.run_evaluation(global_step, rng)


def train_standalone_supervised(
    random_seed: int,
    full_config: ml_collections.ConfigDict,
    experiment_ctor:
    Callable[[str, PRNGKey, ml_collections.ConfigDict], JaxlineExperiment],
    storage_folder: Optional[str],
) -> Dict[str, Array]:
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
      jnp.savez(
          f"{storage_folder}/snapshot_{i}.npz",
          *jax.tree_util.tree_leaves(experiment_instance.snapshot_state())
      )

    rng, step_rng = kfac_jax.utils.p_split(rng)

    # Run a step
    scalars = experiment_instance.step(global_step, step_rng)

    elapsed_time = jnp.asarray(time.time() - start_time)
    stats["time"] = stats.get("time", []) + [elapsed_time]

    for k in sorted(scalars):
      stats.setdefault(k, []).append(jnp.asarray(scalars[k]))

    # Logging
    if i % full_config.log_tensors_interval == 0:
      for k, v in stats.items():
        if jnp.issubdtype(v[-1].dtype, jnp.integer):
          logging.info("%s: %d", k, v[-1])
        else:
          logging.info("%s: %.3f", k, v[-1])
      logging.info("-" * 20)
    i += 1

  stats = {k: jnp.stack(v) for k, v in stats.items()}
  if storage_folder is not None:
    jnp.savez(f"{storage_folder}/snapshot_final.npz",
              *jax.tree_util.tree_leaves(experiment_instance.snapshot_state()))
    jnp.savez(f"{storage_folder}/stats.npz", **stats)
  return stats


class MnistExperiment(JaxlineExperiment):
  """An experiment using the MNIST dataset."""

  def __init__(
      self,
      supervised: bool,
      flatten_images: bool,
      mode: str,
      init_rng: PRNGKey,
      config: ml_collections.ConfigDict,
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

  @functools.cached_property
  def dataset_size(self) -> int:
    return 60_000

  def _build_train_input(
      self,
      split: str,
      seed: int,
      device_batch_size: int,
      **_: Any,
  ) -> Iterator[Batch]:
    assert split == "train"
    return datasets.mnist_dataset(
        split=split,
        has_labels=self._supervised,
        flatten_images=self._flatten_images,
        device_batch_size=device_batch_size,
        repeat=True,
        shuffle=True,
        drop_remainder=True,
        sharding=self.dataset_sharding,
        seed=seed,
        reshuffle_each_iteration=True,
    )

  def _build_eval_input(
      self,
      split: str,
      seed: int,
      device_batch_size: int,
      **_: Any,
  ) -> Iterator[Batch]:
    assert split in ("train", "test")

    return datasets.mnist_dataset(
        split=split,
        has_labels=self._supervised,
        flatten_images=self._flatten_images,
        device_batch_size=device_batch_size,
        repeat=False,
        shuffle=False,
        drop_remainder=False,
        sharding=self.dataset_sharding,
        seed=seed,
    )


class ImageNetExperiment(JaxlineExperiment):
  """An experiment using the ImageNet dataset."""

  def __init__(
      self,
      mode: str,
      init_rng: PRNGKey,
      config: ml_collections.ConfigDict,
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
