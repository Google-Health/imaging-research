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
"""Collection of Beam DoFns to generate embeddings."""
from cxr_foundation import constants
from cxr_foundation import example_generator_lib

import base64
import contextlib
from enum import Enum
import io
from apache_beam.utils import retry

import os
import six
import tempfile
import numpy as np
from PIL import Image
import pydicom

import apache_beam as beam

from apache_beam.options import pipeline_options
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.runners import DataflowRunner
import google.auth

import tensorflow.compat.v1 as tf

from google.api_core import exceptions
from google.api_core.client_options import ClientOptions
from google.api_core.retry import Retry
from google.cloud import aiplatform

from typing import Any, List, Mapping, Optional, Sequence, Text

_RETRIABLE_TYPES = (
    exceptions.TooManyRequests,  # HTTP 429
    exceptions.InternalServerError,  # HTTP 500
    exceptions.BadGateway,  # HTTP 502
    exceptions.ServiceUnavailable,  # HTTP 503
    exceptions.DeadlineExceeded,  # HTTP 504
)

_METRICS_NAMESPACE = 'cxr-embeddings'
_API_ENDPOINT = 'us-central1-aiplatform.googleapis.com'
_VIEW_POSITION = 'ViewPosition'
_FRONTAL_VIEW_POSITIONS = ('AP', 'PA')


class InputFileType(Enum):
  PNG = 'png'
  DICOM = 'dicom'

  def __str__(self):
      return self.value


def _is_retryable(exc):
  return isinstance(exc, _RETRIABLE_TYPES)


def _image_id_to_filebase(image_id: str) -> str:
  filebase, _ = os.path.splitext(os.path.basename(image_id))
  return filebase


@contextlib.contextmanager
def _open(path, *args, **kwargs):
  if path.startswith(constants.GCS_PREFIX):
    f = beam.io.gcp.gcsio.GcsIO().open(path, *args, **kwargs)
  else:
    f = open(path, *args, **kwargs)
  yield f
  f.close()


@beam.typehints.with_input_types(str)
@beam.typehints.with_output_types(Optional[tf.train.Example])
class CreateExampleDoFn(beam.DoFn):

  def __init__(self,
               output_path: Optional[str] = None,
               input_file_type: InputFileType = InputFileType.PNG,
               skip_errors: bool = False):
    self._output_path = output_path
    self._input_file_type = input_file_type
    self._skip_errors = skip_errors

  def process(self, uri: str):
    if self._output_path:
      exists = False
      filebase = _image_id_to_filebase(uri)
      filename = f'{self._output_path}/{filebase}.tfrecord'
      if self._output_path.startswith(constants.GCS_PREFIX):
        exists = beam.io.gcp.gcsio.GcsIO().exists(filename)
      else:
        exists = os.path.exists(filename)
      if exists:
        beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                     'output-file-exists').inc()
        return
    try:
      with _open(uri, 'rb') as f:
        if self._input_file_type == InputFileType.PNG:
          img = np.asarray(Image.open(io.BytesIO(f.read())).convert('L'))
          example = example_generator_lib.png_to_tfexample(img)
        elif self._input_file_type == InputFileType.DICOM:
          dicom = pydicom.dcmread(io.BytesIO(f.read()))
          if _VIEW_POSITION in dicom and dicom.ViewPosition not in _FRONTAL_VIEW_POSITIONS:
            beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'non-frontal-cxr').inc()
            return
          example = example_generator_lib.dicom_to_tfexample(dicom)
        else:
          raise ValueError('Unknown file type.')
        example.features.feature[constants.IMAGE_ID_KEY].bytes_list.value[:] = [
            six.ensure_binary(uri)
        ]
      yield example
    except Exception as e:
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'failed-example-creation').inc()
      if not self._skip_errors:
        raise e


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
        client_options=ClientOptions(api_endpoint=_API_ENDPOINT))
    endpoint = api_client.endpoint_path(
        project=self._project,
        location=constants.LOCATION,
        endpoint=self._endpoint_id)
    retry_policy = Retry(predicate=_is_retryable)
    try:
      response = api_client.predict(
          endpoint=endpoint, instances=instances, retry=retry_policy, timeout=60)
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                   'successful-inference').inc()
    except Exception as e:
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'failed-inference').inc()
      if not self._skip_errors:
        raise e
      return []
    return response.predictions

  def process(self, example: Optional[tf.train.Example]):
    if not example or constants.IMAGE_KEY not in example.features.feature:
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
        constants.EMBEDDING_KEY].float_list.value[:] = values.flatten()
    return example


@beam.typehints.with_input_types(Optional[tf.train.Example])
@beam.typehints.with_output_types(np.ndarray)
class ProcessPredictionDoFn(beam.DoFn):

  def __init__(self, output_path: Optional[str] = None):
    self._output_path = output_path

  def process(self, element: Optional[tf.train.Example]):
    if not element or constants.EMBEDDING_KEY not in element.features.feature:
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                   'missing-representation').inc()
      return np.array([])
    if constants.IMAGE_KEY in element.features.feature:
      del element.features.feature[constants.IMAGE_KEY]
    if self._output_path:
      filebase = _image_id_to_filebase(
          six.ensure_str(element.features.feature[
              constants.IMAGE_ID_KEY].bytes_list.value[0]))
      filename = f'{self._output_path}/{filebase}.tfrecord'
      with tempfile.NamedTemporaryFile('wb+', delete=False) as f:
        with tf.io.TFRecordWriter(f.name) as w:
          w.write(element.SerializeToString())
        with _open(filename, 'wb') as o:
          o.write(f.read())
    yield np.array(
        element.features.feature[constants.EMBEDDING_KEY].float_list.value[:])
