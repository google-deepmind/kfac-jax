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
"""Utilities and loading pipelines for datasets used in the examples.

"""
import types
from typing import Callable, Dict, Iterator, Optional, Tuple

import jax
import jax.numpy as jnp
import kfac_jax
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets
tfds = tensorflow_datasets

# Types for annotation
Array = jax.Array
Shape = Tuple[int, ...]
Batch = Dict[str, Array]
TfBatch = Dict[str, tf.Tensor]

# Special global variables
_IMAGENET_MEAN_RGB = (0.485, 0.456, 0.406)
_IMAGENET_STDDEV_RGB = (0.229, 0.224, 0.225)


def iterator_on_device(iterator: Iterator[Batch]) -> Iterator[Batch]:
  for batch in iterator:
    yield kfac_jax.utils.broadcast_all_local_devices(batch)


def mnist_dataset(
    split: str,
    has_labels: bool,
    flatten_images: bool,
    device_batch_size: int,
    repeat: bool,
    shuffle: bool,
    drop_remainder: bool,
    seed: Optional[int] = None,
    multi_device: bool = True,
    reshuffle_each_iteration: bool = True,
    dtype: str = "float32",
) -> Iterator[Batch]:
  """Standard MNIST dataset pipeline.

  Args:
    split: Which data split to load.
    has_labels: Whether to return the labels or only the images.
    flatten_images: Whether to flatten the images to a vector.
    device_batch_size: The per-device batch size to use.
    repeat: Whether to repeat the dataset.
    shuffle: Whether to shuffle the dataset.
    drop_remainder: Whether to drop the remainder of the dataset if the number
      of data points is not divisible by the total batch size.
    seed: Any seed to use for random pre-processing.
    multi_device: If the returned batch should take into account the number of
      devices present, in which case it will return an array with shape
      `(num_device, device_batch_size, ...)`.
    reshuffle_each_iteration: Whether to reshuffle the dataset in a new order
      after each iteration.
    dtype: The returned data type of the images.

  Returns:
    The MNIST dataset as a tensorflow dataset.
  """

  # Set for multi devices vs single device
  num_devices = jax.device_count() if multi_device else 1
  num_local_devices = jax.local_device_count() if multi_device else 1

  if multi_device:
    host_batch_shape = [num_local_devices, device_batch_size]
  else:
    host_batch_shape = [device_batch_size]

  host_batch_size = num_local_devices * device_batch_size

  num_examples = tfds.builder("mnist").info.splits[split].num_examples

  if num_examples % num_devices != 0:
    raise ValueError("The number of examples should be divisible by the number "
                     "of devices.")

  def preprocess_batch(
      images: tf.Tensor,
      labels: tf.Tensor
  ) -> Dict[str, tf.Tensor]:
    """Standard reshaping of the images to (28, 28)."""
    images = tf.image.convert_image_dtype(images, dtype)
    single_example_shape = [784] if flatten_images else [28, 28]
    images = tf.reshape(images, host_batch_shape + single_example_shape)
    labels = tf.reshape(labels, host_batch_shape)
    if has_labels:
      return dict(images=images, labels=labels)
    else:
      return dict(images=images)

  ds = tfds.load(name="mnist", split=split, as_supervised=True)

  ds = ds.shard(jax.process_count(), jax.process_index())

  ds = ds.cache()

  if host_batch_size < num_examples and shuffle:

    ds = ds.shuffle(buffer_size=(num_examples // jax.process_count()),
                    seed=seed,
                    reshuffle_each_iteration=reshuffle_each_iteration)
  if repeat:
    ds = ds.repeat()

  ds = ds.batch(host_batch_size, drop_remainder=drop_remainder)

  ds = ds.map(preprocess_batch,
              num_parallel_calls=tf.data.experimental.AUTOTUNE)

  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

  return iterator_on_device(iter(tensorflow_datasets.as_numpy(ds)))


def imagenet_num_examples_and_split(
    split: str
) -> Tuple[int, tensorflow_datasets.Split]:
  """Returns the number of examples in the given split of Imagenet."""

  if split == "train":
    return 1271167, tensorflow_datasets.Split.TRAIN
  elif split == "valid":
    return 10000, tensorflow_datasets.Split.TRAIN
  elif split == "train_and_valid":
    return 1281167, tensorflow_datasets.Split.TRAIN
  elif split == "train_eval":
    return 200000, tensorflow_datasets.Split.TRAIN
  elif split == "test":
    return 50000, tensorflow_datasets.Split.VALIDATION
  else:
    raise NotImplementedError()


def imagenet_dataset(
    split: str,
    is_training: bool,
    batch_dims: Shape,
    seed: int = 123,
    shuffle_files: bool = True,
    buffer_size_factor: int = 10,
    shuffle: bool = False,
    cache: bool = False,
    dtype: jnp.dtype = jnp.float32,
    image_size: Shape = (224, 224),
    data_dir: Optional[str] = None,
    extra_preprocessing_func: Optional[
        Callable[[Array, Array],
                 Tuple[Array, Array]]] = None,
) -> Iterator[Batch]:
  """Standard ImageNet dataset pipeline.

  Args:
    split: Which data split to load.
    is_training: Whether this is on the training or evaluator worker.
    batch_dims: The shape of the batch dimensions.
    seed: Any seed to use for random pre-processing, shuffling, and file
      shuffling.
    shuffle_files: Whether to shuffle the ImageNet files.
    buffer_size_factor: Batch size factor for computing cache size.
    shuffle: Whether to shuffle the cache.
    cache: Whether to cache the whole dataset.
    dtype: The returned data type of the images.
    image_size: The image sizes.
    data_dir: If specified, will use this directory to load the dataset from.
    extra_preprocessing_func: A callable to perform addition data preprocessing
      if desired. Should take arguments `image` and `label` consisting of the
      image and its label (without batch dimension), and return a tuple
      consisting of the processed version of these two.

  Returns:
    The ImageNet dataset as a tensorflow dataset.
  """

  preprocess_seed = seed
  shuffle_seed = seed + 1
  file_shuffle_seed = seed + 2
  del seed

  num_examples, tfds_split = imagenet_num_examples_and_split(split)

  shard_range = np.array_split(np.arange(num_examples),
                               jax.process_count())[jax.process_index()]
  start, end = shard_range[0], shard_range[-1] + 1
  if split == "train":
    # Note that our TRAIN=TFDS_TRAIN[10000:] and VALID=TFDS_TRAIN[:10000].
    offset = 10000
    start += offset
    end += offset

  total_batch_size = int(np.prod(batch_dims))

  tfds_split = tfds.core.ReadInstruction(
      tfds_split, from_=start, to=end, unit="abs")

  read_config = tfds.ReadConfig(shuffle_seed=file_shuffle_seed)

  read_config.options.threading.private_threadpool_size = 48
  read_config.options.threading.max_intra_op_parallelism = 1
  read_config.options.deterministic = True

  ds = tfds.load(
      name="imagenet2012:5.*.*",
      shuffle_files=shuffle_files,
      split=tfds_split,
      decoders={"image": tfds.decode.SkipDecoding()},
      data_dir=data_dir,
      read_config=read_config,
  )

  if is_training:

    if cache:
      ds = ds.cache()

    ds = ds.repeat()

    if shuffle:
      ds = ds.shuffle(buffer_size=buffer_size_factor * total_batch_size,
                      seed=shuffle_seed)

  elif num_examples % total_batch_size != 0:
    # If the dataset is not divisible by the batch size then just randomize
    if shuffle:
      ds = ds.shuffle(buffer_size=buffer_size_factor * total_batch_size,
                      seed=shuffle_seed)

  if is_training:
    rng = jax.random.PRNGKey(preprocess_seed)
    tf_seed = tf.convert_to_tensor(rng, dtype=tf.int32)

    # When training we generate a stateless pipeline, at test we don't need it
    def scan_fn(
        seed_: tf.Tensor,
        data: TfBatch,
    ) -> Tuple[tf.Tensor, Tuple[TfBatch, tf.Tensor]]:
      new_seeds = tf.random.experimental.stateless_split(seed_, num=2)
      return new_seeds[0], (data, new_seeds[1])

    # create a sequence of seeds across cases by repeated splitting
    ds = ds.scan(tf_seed, scan_fn)

  def preprocess(
      example: Dict[str, tf.Tensor],
      seed_: Optional[tf.Tensor] = None
  ) -> Dict[str, tf.Tensor]:

    image = _imagenet_preprocess_image(
        image_bytes=example["image"],
        seed=seed_,
        is_training=is_training,
        image_size=image_size
    )
    label = tf.cast(example["label"], tf.int32)

    if extra_preprocessing_func is not None:
      image, label = extra_preprocessing_func(image, label)

    return {"images": image, "labels": label}

  ds = ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def cast_fn(batch_):

    tf_dtype = (tf.bfloat16 if dtype == jnp.bfloat16
                else tf.dtypes.as_dtype(dtype))

    batch_ = dict(**batch_)

    batch_["images"] = tf.cast(batch_["images"], tf_dtype)

    return batch_

  for i, batch_size in enumerate(reversed(batch_dims)):

    ds = ds.batch(batch_size, drop_remainder=not is_training)

    if i == 0:
      # NOTE: You may be tempted to move the casting earlier on in the pipeline,
      # but for bf16 some operations will end up silently placed on the TPU and
      # this causes stalls while TF and JAX battle for the accelerator.
      ds = ds.map(cast_fn)

  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

  return iterator_on_device(iter(tensorflow_datasets.as_numpy(ds)))


def _imagenet_preprocess_image(
    image_bytes: tf.Tensor,
    seed: tf.Tensor,
    is_training: bool,
    image_size: Shape,
) -> tf.Tensor:
  """Returns processed and resized images for Imagenet."""

  if is_training:
    seeds = tf.random.experimental.stateless_split(seed, num=2)

    # Random cropping of the image
    image = _decode_and_random_crop(
        image_bytes, seed=seeds[0], image_size=image_size)

    # Random left-right flipping
    image = tf.image.stateless_random_flip_left_right(image, seed=seeds[1])

  else:
    image = _decode_and_center_crop(image_bytes, image_size=image_size)

  assert image.dtype == tf.uint8

  # NOTE: Bicubic resize (1) casts uint8 to float32 and (2) resizes without
  # clamping overshoots. This means values returned will be outside the range
  # [0.0, 255.0] (e.g. we have observed outputs in the range [-51.1, 336.6]).
  image = tf.image.resize(image, image_size, tf.image.ResizeMethod.BICUBIC)

  # Normalize image
  mean = tf.constant(_IMAGENET_MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
  std = tf.constant(_IMAGENET_STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)

  return (image - mean * 255) / (std * 255)


def _distorted_bounding_box_crop(
    image_bytes: tf.Tensor,
    jpeg_shape: tf.Tensor,
    bbox: tf.Tensor,
    seed: tf.Tensor,
    min_object_covered: float,
    aspect_ratio_range: Tuple[float, float],
    area_range: Tuple[float, float],
    max_attempts: int,
) -> tf.Tensor:
  """Generates cropped_image using one of the bboxes randomly distorted for Imagenet."""

  bbox_begin, bbox_size, _ = tf.image.stateless_sample_distorted_bounding_box(
      image_size=jpeg_shape,
      bounding_boxes=bbox,
      seed=seed,
      min_object_covered=min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=max_attempts,
      use_image_if_no_bounding_boxes=True
  )

  # Crop the image to the specified bounding box.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

  return tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)


def _decode_and_random_crop(
    image_bytes: tf.Tensor,
    seed: tf.Tensor,
    image_size: Shape = (224, 224),
) -> tf.Tensor:
  """Make a random crop of 224 for Imagenet."""
  jpeg_shape = tf.image.extract_jpeg_shape(image_bytes)
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = _distorted_bounding_box_crop(
      image_bytes=image_bytes,
      jpeg_shape=jpeg_shape,
      bbox=bbox,
      seed=seed,
      min_object_covered=0.1,
      aspect_ratio_range=(3 / 4, 4 / 3),
      area_range=(0.08, 1.0),
      max_attempts=10)
  if tf.reduce_all(tf.equal(jpeg_shape, tf.shape(image))):
    # If the random crop failed fall back to center crop.
    image = _decode_and_center_crop(image_bytes, jpeg_shape, image_size)
  return image


def _decode_and_center_crop(
    image_bytes: tf.Tensor,
    jpeg_shape: Optional[tf.Tensor] = None,
    image_size: Shape = (224, 224),
) -> tf.Tensor:
  """Crops to center of image with padding then scales for Imagenet."""

  if jpeg_shape is None:
    jpeg_shape = tf.image.extract_jpeg_shape(image_bytes)

  image_height = jpeg_shape[0]
  image_width = jpeg_shape[1]

  # Pad the image with at least 32px on the short edge and take a
  # crop that maintains aspect ratio.
  scale = tf.minimum(tf.cast(image_height, tf.float32) / (image_size[0] + 32),
                     tf.cast(image_width, tf.float32) / (image_size[1] + 32))

  padded_center_crop_height = tf.cast(scale * image_size[0], tf.int32)
  padded_center_crop_width = tf.cast(scale * image_size[1], tf.int32)

  offset_height = ((image_height - padded_center_crop_height) + 1) // 2
  offset_width = ((image_width - padded_center_crop_width) + 1) // 2

  crop_window = tf.stack([offset_height, offset_width,
                          padded_center_crop_height, padded_center_crop_width])

  return tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)


def _imagenet_distort_color(
    image: tf.Tensor,
    seed: tf.Tensor,
    color_ordering: int = 0,
) -> tf.Tensor:
  """Randomly distorts colors for Imagenet."""

  seeds = tf.random.experimental.stateless_split(seed, num=4)

  if color_ordering == 0:
    image = tf.image.stateless_random_brightness(
        image, max_delta=32. / 255., seed=seeds[0])
    image = tf.image.stateless_random_saturation(
        image, lower=0.5, upper=1.5, seed=seeds[1])
    image = tf.image.stateless_random_hue(
        image, max_delta=0.2, seed=seeds[2])
    image = tf.image.stateless_random_contrast(
        image, lower=0.5, upper=1.5, seed=seeds[3])

  elif color_ordering == 1:
    image = tf.image.stateless_random_brightness(
        image, max_delta=32. / 255., seed=seeds[0])
    image = tf.image.stateless_random_contrast(
        image, lower=0.5, upper=1.5, seed=seeds[1])
    image = tf.image.stateless_random_saturation(
        image, lower=0.5, upper=1.5, seed=seeds[2])
    image = tf.image.stateless_random_hue(
        image, max_delta=0.2, seed=seeds[3])

  else:
    raise ValueError("color_ordering must be in {0, 1}")

  # The random_* ops do not necessarily clamp.
  return tf.clip_by_value(image, 0.0, 1.0)
