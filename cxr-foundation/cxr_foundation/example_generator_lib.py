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
"""Methods to create tf.examples for model inference via pydicom."""

from cxr_foundation import constants

import io
from typing import Iterable, Union

import numpy as np
import png
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import tensorflow.compat.v1 as tf

_BITS_PER_BYTE = 8
_WINDOWWIDTH = 'WindowWidth'
_WINDOWCENTER = 'WindowCenter'


def _encode_png(array: np.ndarray) -> bytes:
  """Converts an unsigned integer 2-D NumPy array to a PNG-encoded string.

  Unsigned 8-bit and 16-bit images are supported.

  Args:
    array: Array to be encoded.

  Returns:
    PNG-encoded string.

  Raises:
    ValueError: If any of the following occurs:
      - `array` is not 2-D.
      - `array` data type is unsupported.
  """
  supported_types = frozenset([np.uint8, np.uint16])
  # Sanity checks.
  if array.ndim != 2:
    raise ValueError(f'Array must be 2-D. Actual dimensions: {array.ndim}')
  if array.dtype.type not in supported_types:
    raise ValueError('Pixels must be either `uint8` or `uint16`. '
                     f'Actual type: {array.dtype.name!r}')

  # Actual conversion.
  writer = png.Writer(
      width=array.shape[1],
      height=array.shape[0],
      greyscale=True,
      bitdepth=_BITS_PER_BYTE * array.dtype.itemsize)
  output_data = io.BytesIO()
  writer.write(output_data, array.tolist())
  return output_data.getvalue()


def _rescale_dynamic_range(image: np.ndarray) -> np.ndarray:
  """Rescales the dynamic range in an integer image to use the full bit range.

  Args:
    image: An image containing unsigned integer pixels.

  Returns:
    Rescaled copy of `image` that uses all the available bits per pixel.

  Raises:
    ValueError: If pixels are not of an integer type.
  """
  if not np.issubdtype(image.dtype, np.integer):
    raise ValueError('Image pixels must be an integer type. '
                     f'Actual type: {image.dtype.name!r}')
  iinfo = np.iinfo(image.dtype)
  return np.interp(image, (image.min(), image.max()),
                   (iinfo.min, iinfo.max)).astype(iinfo)


def _shift_to_unsigned(image: np.ndarray) -> np.ndarray:
  """Shifts values by the minimum value to an unsigned array suitible for PNG.

  This works with signed images and converts them to unsigned versions. It
  involves an inefficient step to convert to a larger data structure for
  shifting all values by the minimum value in the array. It also support float
  data by converting them into uint16.

  Args:
    image: An image containing signed integer pixels.

  Returns:
    Copy of `image` in an unsigned format. Note that the exact same image is
      returned when given an unsigned version.

  Raises:
    ValueError: If pixels are not of an integer type or float.
  """
  if image.dtype == np.uint16 or image.dtype == np.uint8:
    return image
  elif image.dtype == np.int16:
    image = image.astype(np.int32)
    return (image - np.min(image)).astype(np.uint16)
  elif image.dtype == np.int8:
    image = image.astype(np.int16)
    return (image - np.min(image)).astype(np.uint8)
  elif image.dtype == np.float:
    uint16_max = np.iinfo(np.uint16).max
    image = image - np.min(image)
    if np.max(image) > uint16_max:
      image = image * (uint16_max / np.max(image))
      image[image > uint16_max] = uint16_max
    return image.astype(np.uint16)
  raise ValueError('Image pixels must be an 8, 16 bit integer or float type. '
                   f'Actual type: {image.dtype.name!r}')


def _apply_pydicom_prep(ds: pydicom.Dataset) -> np.ndarray:
  """Prepares pixel data after applying data handling from pydicom."""

  def window_u16(image: np.ndarray, window_center: int,
                 window_width: int) -> np.ndarray:
    max_window = np.iinfo(np.uint16).max
    top_clip = window_center - 1 + window_width / 2
    bottom_clip = window_center - window_width / 2
    return np.interp(
        image.clip(bottom_clip, top_clip), (bottom_clip, top_clip),
        (0, max_window))

  arr = ds.pixel_array
  pixel_array = apply_modality_lut(arr, ds)
  if _WINDOWWIDTH in ds and _WINDOWCENTER in ds:
    window_center = ds.WindowCenter
    window_width = ds.WindowWidth
    if isinstance(ds.WindowCenter, pydicom.multival.MultiValue):
      window_center = int(ds.WindowCenter[0])
    if isinstance(ds.WindowWidth, pydicom.multival.MultiValue):
      window_width = int(ds.WindowWidth[0])
    pixel_array = window_u16(pixel_array, window_center, window_width)
  if ds.PhotometricInterpretation == 'MONOCHROME1':
    pixel_array = np.max(pixel_array) - pixel_array
  pixel_array = _shift_to_unsigned(pixel_array)
  # Don't rescale dynamic range for 8-bit images like CXR14.
  if pixel_array.dtype != np.uint8:
    pixel_array = _rescale_dynamic_range(pixel_array)
  return pixel_array


def _assign_bytes_feature(feature: tf.train.Feature,
                          value: Union[bytes, Iterable[bytes]]) -> None:
  """Assigns a bytes float value into feature."""
  if isinstance(value, bytes):
    feature.bytes_list.value[:] = [value]
  else:
    assert not isinstance(value, str)
    feature.bytes_list.value[:] = list(value)


def png_to_tfexample(image_array: np.ndarray) -> tf.train.Example:
  """Create a tf.example for inference.

  The image will be spread to the full bit-depth of 16-bit images.

  Args:
    image_array: An image to use to create the example.

  Returns:
    example: A tf.example for inference.
  """
  pixel_array = _shift_to_unsigned(image_array)
  # Don't rescale dynamic range for 8-bit images like CXR14.
  if pixel_array.dtype != np.uint8:
    pixel_array = _rescale_dynamic_range(pixel_array)
  png_bytes = _encode_png(pixel_array)
  example = tf.train.Example()
  features = example.features.feature
  _assign_bytes_feature(features[constants.IMAGE_KEY], png_bytes)
  _assign_bytes_feature(features[constants.IMAGE_FORMAT_KEY], b'png')
  return example


def dicom_to_tfexample(single_dicom: pydicom.Dataset) -> tf.train.Example:
  """Create a tf.example for inference.

  Resulting images are spread to the full bit-depth of 16-bit images.
  Applies apply_modality_lut first followed by window/level if prresent.

  Args:
    single_dicom: A pydicom dataset used to create the example.

  Returns:
    example: A tf.example for inference.
  """
  image_array = _apply_pydicom_prep(single_dicom)
  png_bytes = _encode_png(image_array)
  example = tf.train.Example()
  features = example.features.feature
  _assign_bytes_feature(features[constants.IMAGE_KEY], png_bytes)
  _assign_bytes_feature(features[constants.IMAGE_FORMAT_KEY], b'png')
  return example
