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
import base64
import contextlib
import googleapiclient
import http
from apache_beam.utils import retry

import io
import os
import re
import six
import tempfile
import numpy as np

import apache_beam as beam

from apache_beam.options import pipeline_options
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.runners import DataflowRunner
import google.auth

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

from google.api_core import exceptions
from google.api_core.client_options import ClientOptions
from google.api_core.retry import Retry
from google.cloud import aiplatform
from google.cloud import aiplatform_v1
from google.cloud import storage

from googleapiclient import discovery
from googleapiclient import http

from google.protobuf.struct_pb2 import Value

from typing import Any, Iterable, List, Mapping, Optional, Sequence, Text, Tuple, TypeVar, Union

GCS_PREFIX = 'gs://'
LOCATION = 'us-central1'

_RETRIABLE_TYPES = (
    exceptions.TooManyRequests,  # 429
    exceptions.InternalServerError,  # 500
    exceptions.BadGateway,  # 502
    exceptions.ServiceUnavailable,  # 503
)

_METRICS_NAMESPACE = 'cxr-embeddings'


def _is_retryable(exc):
  return isinstance(exc, _RETRIABLE_TYPES)


def _image_id_to_filebase(image_id: str) -> str:
  filebase, _ = os.path.splitext(os.path.basename(image_id))
  return filebase


@contextlib.contextmanager
def _open(path, *args, **kwargs):
  if path.startswith(GCS_PREFIX):
    f = beam.io.gcp.gcsio.GcsIO().open(path, *args, **kwargs)
  else:
    f = os.open(path, *args, **kwargs)
  yield f
  f.close()


@beam.typehints.with_input_types(str)
@beam.typehints.with_output_types(Optional[tf.train.Example])
class CreateExampleDoFn(beam.DoFn):

  def __init__(self, output_path: Optional[str] = None):
    self._output_path = output_path

  def process(self, uri: str):
    if self._output_path:
      exists = False
      filebase = _image_id_to_filebase(uri)
      filename = f'{self._output_path}/{filebase}.tfrecord'
      if self._output_path.startswith(GCS_PREFIX):
        exists = beam.io.gcp.gcsio.GcsIO().exists(filename)
      else:
        exists = os.path.exists(filename)
      if exists:
        beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                     'output-file-exists').inc()
        return
    example = tf.train.Example()
    with _open(uri, 'r') as f:
      example.features.feature['image/id'].bytes_list.value[:] = [
          six.ensure_binary(uri)
      ]
      example.features.feature['image/encoded'].bytes_list.value[:] = [
          f.read()
      ]
    yield example


@beam.typehints.with_input_types(Optional[tf.train.Example])
@beam.typehints.with_output_types(Optional[tf.train.Example])
class GenerateEmbeddingsDoFn(beam.DoFn):
  """A DoFn that generates embeddings from a cloud-hosted TensorFlow model."""

  def __init__(self, project, endpoint_id, skip_errors: bool = False):
    self._project = project
    self._endpoint_id = endpoint_id
    self._skip_errors = skip_errors

  def _make_instances(self,
                      serialized_example: bytes) -> List[Mapping[Text, Any]]:
    return [{'b64': base64.b64encode(serialized_example).decode()}]

  def _run_inference(self, serialized_example: bytes) -> Sequence[float]:
    instances = self._make_instances(serialized_example)
    api_client = aiplatform.gapic.PredictionServiceClient(
        client_options=ClientOptions(
            api_endpoint='us-central1-aiplatform.googleapis.com'))
    endpoint = api_client.endpoint_path(
        project=self._project, location=LOCATION, endpoint=self._endpoint_id)
    retry_policy = Retry(predicate=_is_retryable)
    try:
      response = api_client.predict(
          endpoint=endpoint, instances=instances, retry=retry_policy)
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                   'successful-inference').inc()
    except Exception as e:
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'failed-inference').inc()
      if not self._skip_errors:
        raise e
      return []
    return response.predictions

  def process(self, example: Optional[tf.train.Example]):
    if not example or 'image/encoded' not in example.features.feature:
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                   'skipped-inference').inc()
      return
    serialized_example = example.SerializeToString()
    outputs = self._run_inference(serialized_example)
    yield self._post_process(serialized_example, outputs)

  def _post_process(self, serialized_example: bytes,
                    outputs: Sequence[float]) -> Optional[tf.train.Example]:
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    if not outputs:
      return
    values = np.array(outputs)
    example.features.feature[
        'representation'].float_list.value[:] = values.flatten()
    return example


@beam.typehints.with_input_types(Optional[tf.train.Example])
@beam.typehints.with_output_types(np.ndarray)
class ProcessPredictionDoFn(beam.DoFn):

  def __init__(self, output_path: Optional[str] = None):
    self._output_path = output_path

  def process(self, element: Optional[tf.train.Example]):
    if not element or 'representation' not in element.features.feature:
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                   'missing-representation').inc()
      return np.array([])
    del element.features.feature['image/encoded']
    if self._output_path:
      filebase = _image_id_to_filebase(
          six.ensure_str(
              element.features.feature['image/id'].bytes_list.value[0]))
      filename = f'{self._output_path}/{filebase}.tfrecord'
      with tempfile.NamedTemporaryFile(delete=False) as f:
        with tf.io.TFRecordWriter(f.name) as w:
          w.write(element.SerializeToString())
        with _open(filename, 'w') as o:
          o.write(f.read())
    yield np.array(
        element.features.feature['representation'].float_list.value[:])
