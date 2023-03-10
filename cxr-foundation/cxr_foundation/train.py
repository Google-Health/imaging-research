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

`data_dir` is a path (absolute or relative) to a directory (can be a
Google Cloud bucket) containing TFRecord output from run_inference.py,
the generated embeddings for input CXRs. `labels_csv` is a CSV file with
format like:

image_id,split,AIRSPACE_OPACITY
004.png,train,0.0
001.png,train,0.0
000.png,validate,1.0

`image_id` corresponds to the image/id feature key within the TFRecord.
`split` is the name of the image set, e.g. 'training' or 'validation.'
This `split` is passed to one of `--train_label` or
`--tune_split_name`.

Example command:

`python3 -m train --train_label train --tune_split_name tune \
 --labels_csv labels.csv --head_name AIRSPACE_OPACITY \
 --data_dir ./data/ --num_epochs 30`
"""
from cxr_foundation import train_lib

import glob
import os
from absl import app
from absl import flags

import pandas as pd
import tensorflow as tf


# Keep default values in sync with `train_model` params
flags.DEFINE_string('labels_csv', '', 'CSV file containing splits and labels', required=True)

flags.DEFINE_string(
    'data_dir',
    '',
    (
        'The absolute or relative path to directory containing training data.'
    ),
    required=True
)
flags.DEFINE_string(
    'train_label',
    '',
    'The value of the "split" column of `labels_csv` that indicates a training sample.', required=True,
)

flags.DEFINE_string(
    'validate_label',
    '',
    'The value of the "split" column of `labels_csv` that indicates a validation sample.', required=True
)

flags.DEFINE_string(
    'head_name',
    None,
    (
        'The name of the column to train, '
        'e.g. the label class like AIRSPACE_OPACITY in the example above.'
    ),
    required=True
)
flags.DEFINE_integer('batch_size', 512, 'The batch size for model training.')
flags.DEFINE_integer('num_epochs', 300, 'The number of epochs to train.')

FLAGS = flags.FLAGS


def _main():
  model = train_model(
      labels_csv=FLAGS.labels_csv,
      data_dir=FLAGS.data_dir,
      train_label=FLAGS.train_label,
      validate_label=FLAGS.validate_label,
      head_name=FLAGS.head_name,
      batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.num_epochs,
  )



def train_model(
    labels_csv: str,
    data_dir: str,
    train_label: str,
    validate_label: str,
    head_name: str,
    batch_size: int = 512,
    num_epochs: int = 300,
    save_model_name: str = '',
) -> tf.keras.Model:
  """Train a small supervised classification network from a set of .tfrecord image embeddings.

  Args:
    labels_csv: CSV file containing splits and labels
    data_dir: Absolute or relative path to directory (can be Google Cloud bucket
      starting with gs://)  containing training data
    train_label: Name of training image set (column `split`) in
      `labels_csv`.
    validate_label: Name of tune image set (column `split`) in `labels_csv`.
    head_name: Name of the head/column to train on, from the labels CSV file.
    batch_size: Batch size.
    num_epochs: Number of epochs to train.

  Returns:
    The trained model
  """
  with open(labels_csv) as f:
    df = pd.read_csv(f)
  df = df[~df[head_name].isna()]


  filenames = glob.glob(os.path.join(data_dir, '*.tfrecord'))

  # Create training Dataset
  training_df = df[df['split'] == train_label]
  training_labels = dict(
      zip(training_df['image_id'], training_df[head_name].astype(int))
  )
  training_data = train_lib.get_dataset(filenames, labels=training_labels)

  # Create validation Dataset
  validation_data = None
  if validate_label:
    tune_df = df[df['split'] == validate_label]
    tune_labels = dict(zip(tune_df['image_id'], tune_df[head_name].astype(int)))
    validation_data = (
        train_lib.get_dataset(filenames, labels=tune_labels).batch(1).cache()
    )

  model = train_lib.create_model([head_name])
  model.fit(
      x=training_data.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache(),
      validation_data=validation_data,
      epochs=num_epochs,
  )
  if save_model_name:
    model.save(os.path.join(data_dir, 'model'), include_optimizer=False)

  return model



if __name__ == '__main__':
  app.run(_main)
