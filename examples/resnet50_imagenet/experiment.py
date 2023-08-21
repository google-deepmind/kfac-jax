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
"""Using Haiku's Resnet50 v2 implementation for ImageNet."""
import functools
from typing import Any, Dict, Mapping, Tuple, Union

import chex
import haiku as hk
from examples import losses
from examples import training
from ml_collections import config_dict


def resnet50(
    bn_decay_rate: float,
    batch_norm_synced: bool = False,
    zero_init: bool = True,
    num_classes: int = 1000,
    **kwargs: Any,
) -> hk.TransformedWithState:
  """Constructs a Haiku transformed object of the ResNet50 v2 network."""
  bn_config = dict(decay_rate=bn_decay_rate)
  if batch_norm_synced:
    bn_config["cross_replica_axis"] = "kfac_axis"

  def func(
      batch: Union[chex.Array, Mapping[str, chex.Array]],
      is_training: bool
  ) -> chex.Array:
    """Evaluates the network."""
    if isinstance(batch, dict):
      batch = batch["images"]
    model = hk.nets.ResNet50(
        num_classes,
        resnet_v2=True,
        bn_config=bn_config,
        logits_config=dict() if zero_init else dict(w_init=None),
        **kwargs,
    )
    return model(batch, is_training=is_training)
  return hk.without_apply_rng(hk.transform_with_state(func))


def resnet50_loss(
    params: hk.Params,
    state: hk.State,
    batch: Mapping[str, chex.Array],
    is_training: bool,
    l2_reg: chex.Numeric,
    label_smoothing: float = 0.1,
    average_loss: bool = True,
    num_classes: int = 1000,
    bn_decay_rate: float = 0.9,
    batch_norm_synced: bool = False,
    **kwargs: Any,
) -> Tuple[
    chex.Array,
    Union[Dict[str, chex.Array], Tuple[hk.State, Dict[str, chex.Array]]]
]:
  """Evaluates the loss of the Resnet50 model."""

  logits, state = resnet50(
      bn_decay_rate=bn_decay_rate,
      batch_norm_synced=batch_norm_synced,
      num_classes=num_classes,
      **kwargs,
  ).apply(params, state, batch["images"], is_training=is_training)

  loss, stats = losses.classifier_loss_and_stats(
      logits=logits,
      labels_as_int=batch["labels"],
      params=params,
      l2_reg=l2_reg if is_training else 0.0,
      haiku_exclude_batch_norm=True,
      haiku_exclude_biases=True,
      label_smoothing=label_smoothing if is_training else 0.0,
      average_loss=average_loss,
  )

  if is_training:
    return loss, (state, stats)
  else:
    return loss, stats


class Resnet50ImageNetExperiment(training.ImageNetExperiment):
  """Jaxline experiment class for running the Resnet50 v2 on ImageNet."""

  def __init__(
      self,
      mode: str,
      init_rng: chex.PRNGKey,
      config: config_dict.ConfigDict
  ):
    """Initializes the experiment."""
    super().__init__(
        mode=mode,
        init_rng=init_rng,
        config=config,
        init_parameters_func=functools.partial(
            resnet50(num_classes=1000, **config.model_kwargs).init,
            is_training=True,
        ),
        model_loss_func=functools.partial(
            resnet50_loss,
            l2_reg=config.l2_reg,
            num_classes=1000,
            **config.model_kwargs,
            **config.loss_kwargs,
        ),
        has_aux=True,
        has_rng=False,
        has_func_state=True,
    )
