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
"""Collection of functions to generate embeddings."""
import base64
from enum import Enum
import io
import os
from typing import Optional, Sequence

from google.api_core import exceptions
from google.api_core.client_options import ClientOptions
from google.api_core.retry import Retry
from google.cloud import aiplatform
import numpy as np
from PIL import Image
import pydicom
import tensorflow as tf

from cxr_foundation import constants
from cxr_foundation import example_generator_lib


_RETRIABLE_TYPES = (
    exceptions.TooManyRequests,  # HTTP 429
    exceptions.InternalServerError,  # HTTP 500
    exceptions.BadGateway,  # HTTP 502
    exceptions.ServiceUnavailable,  # HTTP 503
    exceptions.DeadlineExceeded,  # HTTP 504
)

_API_ENDPOINT = 'us-central1-aiplatform.googleapis.com'
_VIEW_POSITION = 'ViewPosition'
_FRONTAL_VIEW_POSITIONS = ('AP', 'PA')


class InputFileType(Enum):
  PNG = 'png'
  DICOM = 'dicom'

  def __str__(self):
      return self.value

class OutputFileType(Enum):
  TFRECORD = 'tfrecord'
  NPZ = 'npz'

  def __str__(self):
    return self.value


def _image_id_to_filebase(image_id: str) -> str:
  filebase, _ = os.path.splitext(os.path.basename(image_id))
  return filebase

def _output_file_name(input_file: str, output_path: str, format:OutputFileType) -> str:
    filebase = _image_id_to_filebase(input_file)
    if format == "tfrecord":
      return os.path.join(output_path, f"{filebase}.tfrecord")
    else:
      return os.path.join(output_path, f"{filebase}.npz")


def generate_embeddings(input_files: str, output_dir: str, input_type: InputFileType, output_type: OutputFileType, overwrite_existing: bool = False):
  """
  Generate embedding files from a set of input image files.

  """
  for file in input_files:
    output_file = _output_file_name(file, output_dir, input_type, output_type)

    if not overwrite_existing and os.path.exists(output_file):
      print(f"Found existing file. Skipping: {output_file}")
      continue

    example = create_example_from_image()
    embeddings = generate_embeddings(example)
    save_embeddings(embeddings)



def create_example_from_image(image_file: str, input_type: InputFileType) -> tf.train.Example:
    """
    Create a tf.train.Example from an image file.

    """
    with open(image_file, 'rb') as f:
      if input_type == InputFileType.PNG:
        img = np.asarray(Image.open(io.BytesIO(f.read())).convert('L'))
        example = example_generator_lib.png_to_tfexample(img)
      elif input_type == InputFileType.DICOM:
        dicom = pydicom.dcmread(io.BytesIO(f.read()))
        if _VIEW_POSITION in dicom and dicom.ViewPosition not in _FRONTAL_VIEW_POSITIONS:
          raise RuntimeError("DICOM file view position is not in accepted set: ", _FRONTAL_VIEW_POSITIONS)
        example = example_generator_lib.dicom_to_tfexample(dicom)
      else:
        raise ValueError('Unknown file type.')

    return example


def _is_retryable(exc):
  return isinstance(exc, _RETRIABLE_TYPES)


def generate_embedding_from_service(serialized_example: bytes, project:str, endpoint_id: str)-> Sequence[float]:
  """
  Generates embeddings from a hosted Vertex/AIPlatform model prediction endpoint.

  The API requires the input image to be in the form of a serialized tf.Example.
  """
  instances = [{'b64': base64.b64encode(serialized_example).decode()}]

  api_client = aiplatform.gapic.PredictionServiceClient(
      client_options=ClientOptions(api_endpoint=_API_ENDPOINT))

  endpoint = api_client.endpoint_path(
      project=project,
      location=constants.LOCATION,
      endpoint=endpoint_id)
  retry_policy = Retry(predicate=_is_retryable)

  response = api_client.predict(
      endpoint=endpoint, instances=instances, retry=retry_policy, timeout=60)
  return response.predictions


def save_embeddings(embeddings: Sequence[float], output_name: str, format: str, example: tf.train.Example = None):
    """
    Save the embeddings values to a numpy or tfrecord file.

    """
    if format == "numpy":
      arr = np.array(embeddings)
      # Keyed by "embedding"
      np.savez(output_name, embedding=arr)
      return

    # tfrecord format
    # What's going on here?

    with open(output_name, 'wb') as f:
      f.write(element.SerializeToString())

    # yield np.array(
    #     element.features.feature[constants.EMBEDDING_KEY].float_list.value[:])


  def _post_process(self, serialized_example: bytes,
                    outputs: Sequence[float]) -> Optional[tf.train.Example]:

    # Adds embedding key to existing tf.Example
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    if not outputs:
      return
    values = np.array(outputs)
    example.features.feature[
        constants.EMBEDDING_KEY].float_list.value[:] = values.flatten()
    return example
