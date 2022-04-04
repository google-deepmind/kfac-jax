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
"""Training the Resnet50 v2 on ImageNet with Jaxline."""
import functools

from absl import app
from absl import flags
from jaxline import base_config
from jaxline import platform
from examples.resnet50_imagenet import experiment
from ml_collections import config_dict

Experiment = experiment.Resnet50ImageNetExperiment


def get_config() -> config_dict.ConfigDict:
  """Creates the config for the experiment."""
  config = base_config.get_base_config()
  config.random_seed = 21387715
  config.training_steps = None
  config.interval_type = None
  config.logging_interval_type = "steps"
  config.log_train_data_interval = 100
  config.log_tensors_interval = 100
  config.checkpoint_interval_type = "steps"
  config.save_checkpoint_interval = 1000
  config.checkpoint_dir = "/tmp/kfac_jax_jaxline/"
  config.train_checkpoint_all_hosts = False

  # Experiment config.
  config.experiment_kwargs = config_dict.ConfigDict(
      dict(
          config=dict(
              l2_reg=1e-5,
              training=dict(
                  steps=200_000,
                  epochs=None,
              ),
              model_kwargs=dict(
                  zero_init=False,
                  bn_decay_rate=0.9,
                  batch_norm_synced=False,
              ),
              loss_kwargs=dict(
                  label_smoothing=0.0,
              ),
              batch_size=dict(
                  train=dict(
                      total=1024,
                      per_device=-1,
                  ),
                  eval=dict(
                      total=4000,
                      per_device=-1,
                  ),
              ),
              optimizer=dict(
                  name="kfac",
                  kfac=dict(
                      damping_adaptation_interval=50,
                      damping_adaptation_decay=0.99963,
                      inverse_update_period=50,
                      norm_constraint=0.01,
                      use_adaptive_learning_rate=False,
                      use_adaptive_momentum=False,
                      initial_damping=0.001,
                      learning_rate_schedule=dict(name="kfac_resnet50"),
                      momentum_schedule=dict(name="fixed", value=0.9),
                      use_adaptive_damping=True,
                      min_damping=1e-6,
                      max_damping=1000.0,
                      num_burnin_steps=5,
                      curvature_ema=0.99,
                      batch_norm_registration="diag",
                  ),
                  sgd=dict(
                      decay=0.9,
                      nesterov=True,
                      learning_rate_schedule=dict(name="imagenet_sgd")
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
