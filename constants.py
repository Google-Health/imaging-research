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

GCS_PREFIX = 'gs://'
# Vertex is only in us-central1 for now.
LOCATION = 'us-central1'

# tf.Example feature keys.
IMAGE_KEY = 'image/encoded'
IMAGE_FORMAT_KEY = 'image/format'
IMAGE_ID_KEY = 'image/id'
EMBEDDING_KEY = 'embedding'
