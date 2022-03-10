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
import unittest
from unittest import mock

import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing import util as test_util
import tensorflow as tf

import inference_beam_lib


class GenerateEmbeddingsTest(unittest.TestCase):

  def setUp(self):
    self._input_example = tf.train.Example()
    self._input_example.features.feature['image/encoded'].bytes_list.value[:] = [b'PNGblahblahblah']

  def assertMetricsEqual(self, metrics, metric_name, expected):
    actual = metrics.query(beam.metrics.MetricsFilter().with_name(metric_name))['counters'][0].committed
    self.assertEqual(expected, actual)

  @mock.patch('google.cloud.aiplatform.gapic.PredictionServiceClient')
  def test_success(self, mock_api_client):
    mock_api_client.return_value.predict.return_value.predictions = [1.0, 0.0, 3.0]
    with test_pipeline.TestPipeline() as p:
      input = p | beam.Create([self._input_example])
      output = input | beam.ParDo(
          inference_beam_lib.GenerateEmbeddingsDoFn('project', 'endpoint')) | beam.FlatMap(
              lambda x: x.features.feature['representation'].float_list.value)
      result = p.run()
      test_util.assert_that(
          output,
          test_util.equal_to([1.0, 0.0, 3.0]))
      self.assertMetricsEqual(result.metrics(), 'successful-inference', 1)

  def test_missing_image_feature(self):
    input_example = tf.train.Example()
    input_example.features.feature['unused'].bytes_list.value[:] = [b'unused']
    with test_pipeline.TestPipeline() as p:
      input = p | beam.Create([input_example])
      output = input | beam.ParDo(
          inference_beam_lib.GenerateEmbeddingsDoFn('project', 'endpoint'))
      result = p.run()
      test_util.assert_that(
          output,
          test_util.equal_to([]))
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
            inference_beam_lib.GenerateEmbeddingsDoFn('project', 'endpoint', skip_errors=False))
        p.run()

  @mock.patch('google.cloud.aiplatform.gapic.PredictionServiceClient')
  def test_skip_error(self, mock_api_client):

    def failed_inference(*args, **kwargs):
      raise ValueError

    mock_api_client.return_value.predict.side_effect = failed_inference
    with test_pipeline.TestPipeline() as p:
      input = p | beam.Create([self._input_example])
      output = input | beam.ParDo(
          inference_beam_lib.GenerateEmbeddingsDoFn('project', 'endpoint', skip_errors=True))
      result = p.run()
      test_util.assert_that(
          output,
          test_util.equal_to([None]))
      self.assertMetricsEqual(result.metrics(), 'failed-inference', 1)


if __name__ == '__main__':
    unittest.main()
