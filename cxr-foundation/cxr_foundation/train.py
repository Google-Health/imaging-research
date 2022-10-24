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
000.png,tune,1.0

`image_id` corresponds to the image/id feature key within the TFRecord.
`split` is the name of the image set, e.g. 'training' or 'validation.'
This `split` is passed to one of `--train_split_name` or
`--tune_split_name`.

Example command:

`python3 -m train --train_split_name train --tune_split_name tune \
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

flags.DEFINE_string('labels_csv', '', 'CSV file containing splits and labels')
flags.DEFINE_string(
    'data_dir', '', 'Absolute or relative path to directory '
    '(can be Google Cloud bucket starting with gs://) '
    'containing training data.')
flags.DEFINE_string(
    'train_split_name', '', 'Name of training image set '
    '(column `split`) in `labels_csv`.')
flags.DEFINE_string(
    'tune_split_name', '', 'Name of tune image set (column `split`) '
    'in `labels_csv`.')
flags.DEFINE_string(
    'head_name', None, 'Name of the head to train, '
    'e.g. the label class like AIRSPACE_OPACITY in the example above.')
flags.DEFINE_integer('batch_size', 512, 'Batch size.')
flags.DEFINE_integer('num_epochs', 300, 'Number of epochs to train.')

FLAGS = flags.FLAGS


def main(argv):
  with open(FLAGS.labels_csv) as f:
    df = pd.read_csv(f)
  df = df[~df[FLAGS.head_name].isna()]
  model = train_lib.create_model([FLAGS.head_name])
  training_df = df[df['split'] == FLAGS.train_split_name]
  training_labels = dict(
      zip(training_df['image_id'], training_df[FLAGS.head_name].astype(int)))
  filenames = glob.glob(os.path.join(FLAGS.data_dir, '*.tfrecord'))
  training_data = train_lib.get_dataset(filenames, labels=training_labels)
  tune_data = None
  if FLAGS.tune_split_name:
    tune_df = df[df['split'] == FLAGS.tune_split_name]
    tune_labels = dict(
        zip(tune_df['image_id'], tune_df[FLAGS.head_name].astype(int)))
    tune_data = train_lib.get_dataset(
        filenames, labels=tune_labels).batch(1).cache()
  model.fit(
      x=training_data.batch(FLAGS.batch_size).prefetch(
          tf.data.AUTOTUNE).cache(),
      validation_data=tune_data,
      epochs=FLAGS.num_epochs)
  model.save(
      os.path.join(FLAGS.data_dir, 'model'), include_optimizer=False)


if __name__ == '__main__':
  app.run(main)
