#!/usr/bin/python
#
# Copyright (c) 2021, Google LLC
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. All advertising materials mentioning features or use of this software
#    must display the following acknowledgement:
#    This product includes software developed by Google LLC.
# 4. Neither the name of the Google LLC nor the
#    names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY Google LLC ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL Google LLC BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
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
          test_util.equal_to([input_example]))
      self.assertMetricsEqual(result.metrics(), 'skipped-inference', 1)

  @mock.patch('google.cloud.aiplatform.gapic.PredictionServiceClient')
  def test_raise_error(self, mock_api_client):

    def failed_inference(*args, **kwargs):
      raise ValueError

    mock_api_client.return_value.predict.side_effect = failed_inference
    do_fn = inference_beam_lib.GenerateEmbeddingsDoFn('project', 'endpoint', skip_errors=False)
    with self.assertRaises(ValueError):
      do_fn.process(self._input_example)

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
          test_util.equal_to([self._input_example]))
      self.assertMetricsEqual(result.metrics(), 'failed-inference', 1)


if __name__ == '__main__':
    unittest.main()
