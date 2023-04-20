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
"""Script to train small networks on top of CXR embeddings.

Lightly wraps the `train_model` function. See the `train_model` docstring
for the expected flag value formats.

Example command:

`python3 -m train --data_dir ./data/ \
   --labels_csv labels.csv \
   --head_name AIRSPACE_OPACITY \
   --train_label train \
   --validate_label validate \
   --save_model_name model \
   --batch_size 512 \
   --num_epochs 30`
"""

import glob
import os
from typing import Iterable

from absl import app
from absl import flags
import pandas as pd
import tensorflow as tf

from cxr_foundation import train_lib

flags.DEFINE_string(
    'data_dir',
    '',
    (
        'The absolute or relative path containing the training embeddings .tfrecord files.'
    )
)

flags.DEFINE_string('labels_csv', '', 'CSV file containing splits and labels')

flags.DEFINE_string(
    'head_name',
    None,
    (
        'The name of the head/column to train on, from `labels_csv`.'
    )
)

flags.DEFINE_string(
    'train_label',
    '',
    'The value of the "split" column of `labels_csv` that indicates a training sample.'
)

flags.DEFINE_string(
    'validate_label',
    '',
    'The value of the "split" column of `labels_csv` that indicates a validation sample.'
)

flags.DEFINE_string('save_model_name', '', 'The absolute or relative file name to save the model to.')

flags.DEFINE_integer('batch_size', 512, 'The batch size for model training.')
flags.DEFINE_integer('num_epochs', 300, 'The number of epochs to train.')

flags.DEFINE_string('best_metrics', 'val_auc', 'The metrics used for saving the best model ckpt.')
flags.DEFINE_string('best_metrics_mode', 'max',
                    'The decision to overwrite the current save file is made based on either the '
                    'maximization or the minimization of the monitored quantity.')

FLAGS = flags.FLAGS


def _main(_):
  """
  Simple command line wrapper to call train_model
  """
  file_names = glob.glob(os.path.join(FLAGS.data_dir, '*.tfrecord'))
  
  with open(FLAGS.labels_csv) as f:
    df_labels = pd.read_csv(f)
  df_labels = df_labels[~df_labels[FLAGS.head_name].isna()]

  model = train_model(
      file_names=file_names,
      df_labels=df_labels,
      head_name=FLAGS.head_name,
      train_label=FLAGS.train_label,
      validate_label=FLAGS.validate_label,
      batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.num_epochs,
      save_model_name=FLAGS.save_model_name)

  model.save(FLAGS.save_model_name, include_optimizer=False)
  print(f"Saved trained model to file: {FLAGS.save_model_name}")
  
  return


def train_model(
    file_names: Iterable[str],
    df_labels: pd.DataFrame,
    head_name: str,
    train_label: str,
    validate_label: str,
    model: tf.keras.Model = None,
    batch_size: int = 512,
    num_epochs: int = 300,
    save_model_name: str = None,
) -> tf.keras.Model:
  """Train a classification model from a set of .tfrecord image embeddings and their labels.

  Args:
    file_names: The set of .tfrecord image embedding file names.
    df_labels: Data frame containing labels and splits. See below for required column names.
    head_name: The name of the head/column to train on, from `df_labels`.
    train_label: The value of the "split" column of `df_labels` that indicates a training sample.
    validate_label: The value of the "split" column of `df_labels` that indicates a validation sample.
    model: The model to train. Defaults to the model from `train_lib.create_model` if none is specified. 
    batch_size: Batch size for training.
    num_epochs: Number of epochs to train.
    save_model_name: Name for the model to save.

  The `df_labels` DataFrame must contain the follow columns with the specified headings:
  - "{head_name}" (equal to the `head_name` param): The label/outcome to train on.
  - "image_id": Corresponds to the image/id feature key within the TFRecord.
  - "split": Indicates the dataset split. ie. train/test/validation

  Example `df_labels` contents:

  image_id,split,AIRSPACE_OPACITY
  004.png,train,0.0
  001.png,train,0.0
  000.png,validate,1.0

  Returns:
    The trained model
  """

  # Create training Dataset
  training_df = df_labels[df_labels['split'] == train_label]
  training_labels = dict(
      zip(training_df['image_id'], training_df[head_name].astype(int))
  )
  training_data = train_lib.get_dataset(file_names, labels=training_labels)

  # Create validation Dataset
  validation_data = None
  if validate_label:
    validate_df = df_labels[df_labels['split'] == validate_label]
    validate_labels = dict(zip(validate_df['image_id'], validate_df[head_name].astype(int)))
    validation_data = (
        train_lib.get_dataset(file_names, labels=validate_labels).batch(1).cache()
    )

  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=save_model_name,
      save_weights_only=True,
      monitor=FLAGS.best_metrics,
      mode=FLAGS.best_metrics_mode,
      save_best_only=True,
      verbose=1)

  # Get default model if none was specified
  model = model or train_lib.create_model([head_name])
  model.fit(
      x=training_data.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache(),
      validation_data=validation_data,
      epochs=num_epochs,
      callbacks=[model_checkpoint_callback],
  )
  model.load_weights(save_model_name)
  return model


if __name__ == '__main__':
  app.run(_main)
