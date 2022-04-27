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
import constants
import example_generator_lib
import unittest
from unittest import mock

import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing import util as test_util
import functools
import numpy as np
import os
import six
import tempfile
import tensorflow.compat.v1 as tf

import inference_beam_lib


class InferenceBeamTestCase(unittest.TestCase):

  def assertMetricsEqual(self, metrics, metric_name, expected):
    actual = metrics.query(beam.metrics.MetricsFilter().with_name(
        metric_name))['counters'][0].committed
    self.assertEqual(expected, actual)


class CreateExampleTest(InferenceBeamTestCase):

  def setUp(self):
    self._output_path = tempfile.TemporaryDirectory()
    self._input_file = './testdata/random.png'
    self._expected = tf.train.Example()
    with open('./testdata/expected.png', 'rb') as f:
      self._expected.features.feature[constants.IMAGE_KEY].bytes_list.value[:] = [f.read()]
    self._expected.features.feature[constants.IMAGE_ID_KEY].bytes_list.value[:] = [
        six.ensure_binary(self._input_file)
    ]
    self._expected.features.feature[constants.IMAGE_FORMAT_KEY].bytes_list.value[:] = [
      b'png'
    ]

  def tearDown(self):
    self._output_path.cleanup()

  def test_success(self):
    with test_pipeline.TestPipeline() as p:
      input = p | beam.Create([self._input_file])
      output = input | beam.ParDo(
          inference_beam_lib.CreateExampleDoFn(self._output_path.name))
      result = p.run()
      test_util.assert_that(output, test_util.equal_to([self._expected]))

  def test_already_existing(self):
    filebase = inference_beam_lib._image_id_to_filebase(self._input_file)
    with open(
        os.path.join(self._output_path.name, f'{filebase}.tfrecord'), 'w') as f:
      f.write('Output file already exists.')
    with test_pipeline.TestPipeline() as p:
      input = p | beam.Create([self._input_file])
      output = input | beam.ParDo(
          inference_beam_lib.CreateExampleDoFn(self._output_path.name))
      result = p.run()
      test_util.assert_that(output, test_util.equal_to([]))
      self.assertMetricsEqual(result.metrics(), 'output-file-exists', 1)


class GenerateEmbeddingsTest(InferenceBeamTestCase):

  def setUp(self):
    self._input_example = tf.train.Example()
    self._input_example.features.feature[
        constants.IMAGE_KEY].bytes_list.value[:] = [b'PNGblahblahblah']

  @mock.patch('google.cloud.aiplatform.gapic.PredictionServiceClient')
  def test_success(self, mock_api_client):
    mock_api_client.return_value.predict.return_value.predictions = [
        1.0, 0.0, 3.0
    ]
    with test_pipeline.TestPipeline() as p:
      input = p | beam.Create([self._input_example])
      output = input | beam.ParDo(
          inference_beam_lib.GenerateEmbeddingsDoFn(
              'project',
              'endpoint')) | beam.FlatMap(lambda x: x.features.feature[
                  constants.EMBEDDING_KEY].float_list.value)
      result = p.run()
      test_util.assert_that(output, test_util.equal_to([1.0, 0.0, 3.0]))
      self.assertMetricsEqual(result.metrics(), 'successful-inference', 1)

  def test_missing_image_feature(self):
    input_example = tf.train.Example()
    input_example.features.feature['unused'].bytes_list.value[:] = [b'unused']
    with test_pipeline.TestPipeline() as p:
      input = p | beam.Create([input_example])
      output = input | beam.ParDo(
          inference_beam_lib.GenerateEmbeddingsDoFn('project', 'endpoint'))
      result = p.run()
      test_util.assert_that(output, test_util.equal_to([]))
      self.assertMetricsEqual(result.metrics(), 'skipped-inference', 1)

  @mock.patch('google.cloud.aiplatform.gapic.PredictionServiceClient')
  def test_raise_error(self, mock_api_client):

    def failed_inference(*args, **kwargs):
      raise ValueError

    mock_api_client.return_value.predict.side_effect = failed_inference
    with self.assertRaises(RuntimeError):
      with test_pipeline.TestPipeline() as p:
        input = p | beam.Create([self._input_example])
        output = input | beam.ParDo(
            inference_beam_lib.GenerateEmbeddingsDoFn(
                'project', 'endpoint', skip_errors=False))
        p.run()

  @mock.patch('google.cloud.aiplatform.gapic.PredictionServiceClient')
  def test_skip_error(self, mock_api_client):

    def failed_inference(*args, **kwargs):
      raise ValueError

    mock_api_client.return_value.predict.side_effect = failed_inference
    with test_pipeline.TestPipeline() as p:
      input = p | beam.Create([self._input_example])
      output = input | beam.ParDo(
          inference_beam_lib.GenerateEmbeddingsDoFn(
              'project', 'endpoint', skip_errors=True))
      result = p.run()
      test_util.assert_that(output, test_util.equal_to([None]))
      self.assertMetricsEqual(result.metrics(), 'failed-inference', 1)


class ProcessPredictionTest(InferenceBeamTestCase):

  def setUp(self):
    self._output_path = tempfile.TemporaryDirectory()

  def tearDown(self):
    self._output_path.cleanup()

  def test_missing_element(self):
    with test_pipeline.TestPipeline() as p:
      input = p | beam.Create([None])
      output = input | beam.ParDo(inference_beam_lib.ProcessPredictionDoFn())
      result = p.run()
      test_util.assert_that(output, test_util.equal_to([]))
      self.assertMetricsEqual(result.metrics(), 'missing-representation', 1)

  def test_missing_representation(self):
    input_example = tf.train.Example()
    with test_pipeline.TestPipeline() as p:
      input = p | beam.Create([input_example])
      output = input | beam.ParDo(inference_beam_lib.ProcessPredictionDoFn())
      result = p.run()
      test_util.assert_that(output, test_util.equal_to([]))
      self.assertMetricsEqual(result.metrics(), 'missing-representation', 1)

  def test_no_output_path(self):
    input_example = tf.train.Example()
    input_example.features.feature[
        constants.EMBEDDING_KEY].float_list.value[:] = [1.0, 2.0, 3.0]
    with test_pipeline.TestPipeline() as p:
      input = p | beam.Create([input_example])
      output = input | beam.ParDo(inference_beam_lib.ProcessPredictionDoFn())
      result = p.run()
      test_util.assert_that(
          output,
          functools.partial(
              np.testing.assert_array_almost_equal, y=[[1.0, 2.0, 3.0]]))

  def test_output_path(self):
    input_example = tf.train.Example()
    input_example.features.feature[
        constants.IMAGE_ID_KEY].bytes_list.value[:] = [b'image1234.png']
    input_example.features.feature[
        constants.EMBEDDING_KEY].float_list.value[:] = [1.0, 2.0, 3.0]
    with test_pipeline.TestPipeline() as p:
      input = p | beam.Create([input_example])
      output = input | beam.ParDo(
          inference_beam_lib.ProcessPredictionDoFn(self._output_path.name))
      result = p.run()
      test_util.assert_that(
          output,
          functools.partial(
              np.testing.assert_array_almost_equal, y=[[1.0, 2.0, 3.0]]))
      actual = tf.train.Example()
      filename = os.path.join(self._output_path.name, 'image1234.tfrecord')
      for example in tf.python_io.tf_record_iterator(filename):
        actual = tf.train.Example.FromString(example)
      self.assertEqual(input_example, actual)


if __name__ == '__main__':
  unittest.main()
