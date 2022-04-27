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
import train_lib

import unittest

import glob
import numpy as np
import tensorflow.compat.v1 as tf

from typing import Any, List, Tuple


def _flatten(dataset: List[Tuple[tf.Tensor]]) -> List[Any]:
  flattened = []
  for datum in dataset:
    row = []
    for i in datum:
      try:
        row.append(list(i))
      except TypeError:
        row.append(i)
    flattened.append(row)
  return flattened


class TrainLibTestCase(unittest.TestCase):

  def test_empty_labels_fails(self):
    with self.assertRaises(AssertionError):
      train_lib.get_dataset(sorted(glob.glob('./testdata/*.tfrecord')), {})

  def test_get_dataset_single_label(self):
    dataset = _flatten(
        train_lib.get_dataset(
            sorted(glob.glob('./testdata/*.tfrecord')),
            {'gs://superrad/inputs/cxr14/00000001_000.png': 1},
            embeddings_size=5))
    actual_embeddings = np.array([d[0] for d in dataset])
    actual_labels = np.array([d[1] for d in dataset])
    actual_weights = np.array([d[2] for d in dataset])
    np.testing.assert_array_almost_equal(
        np.array([[3., 0., 4., 0., 3.]]), actual_embeddings)
    np.testing.assert_array_almost_equal(
        np.array([[1]]), actual_labels)
    np.testing.assert_array_almost_equal(np.ones(1), actual_weights)

  def test_get_dataset_weights(self):
    dataset = _flatten(
        train_lib.get_dataset(
            sorted(glob.glob('./testdata/*.tfrecord')),
            {'gs://superrad/inputs/cxr14/00000001_000.png': 1},
            embeddings_size=5,
            weights={'gs://superrad/inputs/cxr14/00000001_000.png': 0.5}))
    actual_embeddings = np.array([d[0] for d in dataset])
    actual_labels = np.array([d[1] for d in dataset])
    actual_weights = np.array([d[2] for d in dataset])
    np.testing.assert_array_almost_equal(
        np.array([[3., 0., 4., 0., 3.]]), actual_embeddings)
    np.testing.assert_array_almost_equal(np.array([[1]]), actual_labels)
    np.testing.assert_array_almost_equal(np.array([[0.5]]), actual_weights)
