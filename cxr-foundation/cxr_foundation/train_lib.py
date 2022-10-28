#!/usr/bin/python
#
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Library functions for training small networks on CXR embeddings."""
from cxr_foundation import constants

import functools
import glob

import tensorflow.compat.v2 as tf
import tensorflow_models as tfm

from typing import Dict, List, Optional, Tuple


_DEFAULT_EMBEDDINGS_SIZE = 1376


def parse_fn(
    serialized_example: bytes,
    embeddings_size: int = _DEFAULT_EMBEDDINGS_SIZE,
    example_feature_key: str = constants.IMAGE_ID_KEY,
    embeddings_feature_key: str = constants.EMBEDDING_KEY
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Parses a single tf.Example to return embeddings + label/weights."""
  features = {
      embeddings_feature_key:
          tf.io.FixedLenFeature([embeddings_size],
                                tf.float32,
                                default_value=tf.constant(
                                    0.0, shape=[embeddings_size]))
  }
  features[example_feature_key] = tf.io.FixedLenSequenceFeature(
      [], tf.string, allow_missing=True, default_value='')
  parsed_tensors = tf.io.parse_example(serialized_example, features=features)
  return (parsed_tensors[embeddings_feature_key],
          parsed_tensors[example_feature_key])


def process_tfrecord_shard(
    filename: str,
    embeddings_size: int = _DEFAULT_EMBEDDINGS_SIZE,
    example_feature_key: str = constants.IMAGE_ID_KEY,
    embeddings_feature_key: str = constants.EMBEDDING_KEY) -> tf.data.Dataset:
  """Process a single shard of a TFRecord."""
  return tf.data.TFRecordDataset(filename).map(
      functools.partial(
          parse_fn,
          embeddings_size=embeddings_size,
          example_feature_key=example_feature_key,
          embeddings_feature_key=embeddings_feature_key))


def get_dataset(
    filenames: List[str],
    labels: Dict[str, int],
    embeddings_size: int = _DEFAULT_EMBEDDINGS_SIZE,
    weights: Optional[Dict[str, float]] = None,
    example_feature_key: str = constants.IMAGE_ID_KEY,
    embeddings_feature_key: str = constants.EMBEDDING_KEY) -> tf.data.Dataset:
  """Create tf.data.Dataset from the TFRecords."""
  assert labels, 'Must pass non-empty labels dict.'
  dataset = tf.data.Dataset.from_tensor_slices(filenames).interleave(
      functools.partial(
          process_tfrecord_shard,
          embeddings_size=embeddings_size,
          example_feature_key=example_feature_key,
          embeddings_feature_key=embeddings_feature_key))

  labels = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
          tf.constant(list(labels.keys())),
          tf.constant(list(labels.values())),
          key_dtype=tf.string,
          value_dtype=tf.int32),
      default_value=-1)

  if weights is not None:
    weights = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(list(weights.keys())),
            tf.constant(list(weights.values())),
            key_dtype=tf.string,
            value_dtype=tf.float32),
        default_value=0.)

  def filter_zero_weights(embedding, label, weight):
    del embedding, label
    return tf.math.reduce_all(weight > 0)

  def filter_negative_labels(embedding, label, weight):
    return tf.math.reduce_all(label > -1)

  def _lookup_label(features, image_id):
    return features, labels.lookup(image_id), image_id

  def _lookup_weight(features, label, image_id):
    weight = 1.0
    if weights:
      weight = weights.lookup(image_id)
    return features, label, weight, image_id

  # Look up labels and weights.
  dataset = dataset.map(_lookup_label).map(_lookup_weight)

  # Remove the IDs and filter out zero weights and negative labels.
  return dataset.map(lambda features, label, weight, image_id:
                     (features, label, weight)).filter(
                         filter_negative_labels).filter(filter_zero_weights)


def create_model(heads,
                 embeddings_size = _DEFAULT_EMBEDDINGS_SIZE,
                 learning_rate=0.1,
                 end_lr_factor=1.0,
                 dropout=0.0,
                 decay_steps=1000,
                 loss_weights=None,
                 hidden_layer_sizes=[512, 256],
                 weight_decay=0.0,
                 seed=None):
  """Creates linear probe or multilayer perceptron using LARS + cosine decay."""
  inputs = tf.keras.Input(shape=(embeddings_size,))
  hidden = inputs
  # If no hidden_layer_sizes are provided, model will be a linear probe.
  for size in hidden_layer_sizes:
    hidden = tf.keras.layers.Dense(
        size,
        activation='relu',
        kernel_initializer=tf.keras.initializers.HeUniform(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(l2=weight_decay),
        bias_regularizer=tf.keras.regularizers.l2(l2=weight_decay))(
            hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Dropout(dropout, seed=seed)(hidden)
  output = tf.keras.layers.Dense(
      units=len(heads),
      activation='sigmoid',
      kernel_initializer=tf.keras.initializers.HeUniform(seed=seed))(
          hidden)
  outputs = {}
  for i, head in enumerate(heads):
    outputs[head] = tf.keras.layers.Lambda(
        lambda x: x[..., i:i + 1], name=head.lower())(
            output)
  model = tf.keras.Model(inputs, outputs)
  learning_rate_fn = tf.keras.experimental.CosineDecay(
      tf.cast(learning_rate, tf.float32),
      tf.cast(decay_steps, tf.float32),
      alpha=tf.cast(end_lr_factor, tf.float32))
  model.compile(
      optimizer=tfm.optimization.lars_optimizer.LARS(
          learning_rate=learning_rate_fn),
      loss=dict([(head, 'binary_crossentropy') for head in heads]),
      loss_weights=loss_weights or dict([(head, 1.) for head in heads]),
      weighted_metrics=['AUC'])
  return model
