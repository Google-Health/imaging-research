import base64
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


def _parse_gcs_uri(gcs_uri: str):
  match = re.match('gs://([^\/]+)/(.*)', gcs_uri)
  bucket_name = match.group(1)
  filename = match.group(2)
  return bucket_name, filename


class CreateExampleDoFn(beam.DoFn):

    def __init__(self, project=None, output_path: Optional[str] = None):
        self._project = project
        self._output_bucket = None
        self._output_path = None
        if output_path:
          self._output_bucket, _ = _parse_gcs_uri(output_path)
          self._output_path = output_path

    def process(self, element):
        uri, split, label = element
        if label is not None:
            label = int(label)
        example = tf.train.Example()
        # TODO(asellerg): take in user project.
        client = storage.Client(project=self._project)
        if self._output_path:
          image_id, _ = os.path.splitext(os.path.basename(uri))
          folder = self._output_path[len(f'gs://{self._output_bucket}/'):]
          filename = f'{folder}/{split}/{image_id}.tfrecord'
          blob = client.bucket(self._output_bucket).blob(filename)
          if blob.exists():
            example = tf.train.Example()
            with tempfile.NamedTemporaryFile(delete=False) as f:
              blob.download_to_file(f)
            for serialized_example in tf.python_io.tf_record_iterator(f.name):
              example.ParseFromString(serialized_example)
            if 'representation' in example.features.feature and example.features.feature['representation'].float_list.value:
              beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'output-file-exists').inc()
              return tf.train.Example()
            else:
              blob.delete()
        bucket_name, filename = _parse_gcs_uri(uri)
        bucket = client.bucket(bucket_name, user_project=self._project)
        blob = bucket.blob(filename)
        example.features.feature['label'].int64_list.value[:] = [label]
        example.features.feature['split'].bytes_list.value[:] = [six.ensure_binary(split)]
        example.features.feature['image/id'].bytes_list.value[:] = [six.ensure_binary(uri)]
        example.features.feature['image/encoded'].bytes_list.value[:] = [blob.download_as_bytes()]
        yield example


class GenerateEmbeddingsDoFn(beam.DoFn):
  """A DoFn that generates embeddings from a cloud-hosted TensorFlow model."""

  def __init__(self, project, endpoint_id):
    self._project = project
    self._endpoint_id = endpoint_id

  def _make_instances(
      self,
      serialized_example: bytes
      )-> List[Mapping[Text, Any]]:
    return [{'b64': base64.b64encode(serialized_example).decode()}]

  def _run_inference(
      self, serialized_example: bytes
      ) -> Sequence[Any]:
    instances = self._make_instances(serialized_example)
    api_client = aiplatform.gapic.PredictionServiceClient(client_options=ClientOptions(api_endpoint='us-central1-aiplatform.googleapis.com'))
    endpoint = api_client.endpoint_path(
        project=self._project, location=LOCATION, endpoint=self._endpoint_id
    )
    retry_policy = Retry(predicate=_is_retryable)
    try:
      response = api_client.predict(endpoint=endpoint, instances=instances, retry=retry_policy)
    except Exception:
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'failed-request').inc()
      return []
    return response.predictions

  def process(
      self,
      example: tf.train.Example
      ) -> Iterable[tf.train.Example]:
    if 'image/encoded' not in example.features.feature:
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'skipped-inference').inc()
      return [example]
    serialized_example = example.SerializeToString()
    outputs = self._run_inference(serialized_example)
    return [self._post_process(serialized_example, outputs)]

  def _post_process(
      self,
      serialized_example: bytes,
      outputs: Sequence[Value]
      ) -> tf.train.Example:
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    if not outputs:
      return example
    values = np.array(outputs)
    example.features.feature['representation'].float_list.value[:] = values.flatten()
    return example


class ProcessPredictionDoFn(beam.DoFn):

  def __init__(self, project: str, output_path: Optional[str] = None):
    self._project = project
    self._output_bucket = None
    if output_path:
      self._output_bucket, _ = _parse_gcs_uri(output_path)
    self._output_path = output_path

  def process(self, element: tf.train.Example):
    if 'representation' not in element.features.feature:
        beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'missing-representation').inc()
        return np.array([])
    del element.features.feature['image/encoded']
    if self._output_bucket:
        client = storage.Client(project=self._project)
        bucket = client.bucket(self._output_bucket)
        folder = self._output_path[len(f'gs://{self._output_bucket}/'):]
        split = six.ensure_str(element.features.feature['split'].bytes_list.value[0])
        filename, _ = os.path.splitext(six.ensure_str(os.path.basename(element.features.feature['image/id'].bytes_list.value[0])))
        with tempfile.NamedTemporaryFile(delete=False) as f:
          with tf.io.TFRecordWriter(f.name) as w:
            w.write(element.SerializeToString())
        blob = bucket.blob(f'{folder}/{split}/{filename}.tfrecord')
        blob.upload_from_filename(f.name)
    yield np.array(element.features.feature['representation'].float_list.value[:])
