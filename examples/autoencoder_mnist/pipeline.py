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
"""Training the MNIST Autoencoder with Jaxline."""
import functools

from absl import app
from absl import flags
from jaxline import base_config
from jaxline import platform
from examples.autoencoder_mnist import experiment
from ml_collections import config_dict

Experiment = experiment.AutoencoderMnistExperiment


def get_config() -> config_dict.ConfigDict:
  """Creates the config for the experiment."""
  config = base_config.get_base_config()
  config.random_seed = 123109801
  config.training_steps = None
  config.interval_type = None
  config.logging_interval_type = "steps"
  config.log_train_data_interval = 10
  config.log_tensors_interval = 1
  config.checkpoint_interval_type = "steps"
  config.save_checkpoint_interval = 100
  config.checkpoint_dir = "/tmp/kfac_jax_jaxline/"
  config.train_checkpoint_all_hosts = False

  # Experiment config.
  config.experiment_kwargs = config_dict.ConfigDict(
      dict(
          config=dict(
              l2_reg=1e-5,
              training=dict(
                  steps=5_000,
                  epochs=None,
              ),
              batch_size=dict(
                  train=dict(
                      total=60_000,
                      per_device=-1,
                  ),
                  eval=dict(
                      total=10_000,
                      per_device=-1,
                  ),
              ),
              optimizer=dict(
                  name="kfac",
                  kfac=dict(
                      inverse_update_period=5,
                      damping_adaptation_interval=5,
                      num_burnin_steps=5,
                      curvature_ema=0.95,
                      use_adaptive_damping=True,
                      use_adaptive_learning_rate=True,
                      use_adaptive_momentum=True,
                      damping_adaptation_decay=0.95,
                      initial_damping=150.0,
                      min_damping=1e-5,
                      max_damping=1000.0,
                  ),
              )
          )
      )
  )

  config.lock()
  return config


if __name__ == "__main__":
  flags.mark_flag_as_required("config")
  app.run(functools.partial(platform.main, Experiment))
