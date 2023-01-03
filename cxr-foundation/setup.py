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
import setuptools

setuptools.setup(
  name='cxr-foundation',
  version='0.0.13',
  description='CXR Foundation: chest x-ray embeddings generation.',
  install_requires=[
    'google-api-python-client',
    'google-apitools',
    'google-cloud-aiplatform',
    'google-cloud-storage',
    'apache_beam',
    'pandas',
    'tensorflow >= 2.10.0',
    'pillow',
    'pypng',
    'pydicom',
    'tf-models-official >= 2.10.0',
    'protobuf < 3.20',
    'typing-extensions',
    'shapely < 2.0.0'
  ],
  packages=setuptools.find_packages())
