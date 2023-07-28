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
"""A Beam pipeline to generate embeddings from a list of PNG or DICOM files."""
from cxr_foundation import constants
from cxr_foundation import inference_beam_lib

import argparse

import os



def generate_embeddings(input_files: str, output_dir: str, input_file_type: str, skip_errors:bool = False):
  """


  """

  validate()

  for file in input_files:
    create_example_from_image()



def main(argv=None, save_main_session=True):
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input_path',
      dest='input_path',
      required=True,
      help='Folder or GCS bucket containing input PNG or DICOM files.')
  parser.add_argument(
      '--input_file_type',
      dest='input_file_type',
      required=True,
      default=inference_beam_lib.InputFileType.PNG,
      type=inference_beam_lib.InputFileType,
      choices=list(inference_beam_lib.InputFileType),
      help=f'The type of input files, either DICOM or PNG.')
  parser.add_argument(
      '--output_path',
      dest='output_path',
      required=True,
      help='Folder or GCS bucket to write output tf.Examples.')

  parser.add_argument(
      '--endpoint_id',
      dest='endpoint_id',
      type=int,
      default=6695981832690728960,
      help='Numerical ID for embeddings API.')

  parser.add_argument(
      '--embeddings_project',
      dest='embeddings_project',
      default='gh-rad-validation-cxrembd-deid',
      help='GCP project ID that hosts embeddings API.')

  parser.add_argument(
      '--skip_errors',
      dest='skip_errors',
      default=False,
      help='Whether to suppress exceptions thrown by GenerateEmbeddingsDoFn.')


    input = [
        os.path.join(known_args.input_path, f)
        for f in os.listdir(known_args.input_path)
    ]

  if (not known_args.output_path.startswith(constants.GCS_PREFIX) and
      not os.path.exists(known_args.output_path)):
    os.mkdir(known_args.output_path)



            inference_beam_lib.CreateExampleDoFn(
                output_path=known_args.output_path,
                input_file_type=known_args.input_file_type,
                skip_errors=known_args.skip_errors)

            inference_beam_lib.GenerateEmbeddingsDoFn(
                known_args.embeddings_project, known_args.endpoint_id,
                skip_errors=known_args.skip_errors)

            inference_beam_lib.ProcessPredictionDoFn(
                output_path=known_args.output_path)


if __name__ == '__main__':
  main()
