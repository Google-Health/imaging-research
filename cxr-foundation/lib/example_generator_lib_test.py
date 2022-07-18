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
"""Tests for example_generator_lib."""
import example_generator_lib

import numpy as np
import pydicom
import unittest


class ExampleGeneratorLibTest(unittest.TestCase):

  def test_create_example(self):
    """Test the creation of examples."""
    # This is a DICOM with a grayscale fake image.
    dicom_path = './testdata/fake.dcm'
    dicom = pydicom.dcmread(dicom_path)

    test_example = example_generator_lib.dicom_to_tfexample(dicom)
    f_dict = test_example.features.feature
    self.assertEqual(f_dict['image/format'].bytes_list.value[:], [b'png'])
    self.assertEqual(len(f_dict['image/encoded'].bytes_list.value[0]), 23287)


if __name__ == '__main__':
  unittest.main()