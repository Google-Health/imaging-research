import inference_beam_lib

import argparse
import io
import logging
import os
import pandas as pd
import re
import tempfile

import apache_beam as beam

from apache_beam.options import pipeline_options
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.runners import DataflowRunner

from google.api_core.client_options import ClientOptions
from google.cloud import aiplatform
from google.cloud import aiplatform_v1
from google.cloud import storage


_REQUIREMENTS_FILE = 'requirements.txt'
_SETUP_FILE = './setup.py'


def main(argv=None, save_main_session=True):
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_file',
                      dest='input_file',
                      required=True)
  parser.add_argument('--output_path',
                      dest='output_path',
                      required=True)
  parser.add_argument('--limit',
                      dest='limit',
                      default=100,
                      type=int)
  parser.add_argument('--endpoint_id',
                      dest='endpoint_id',
                      required=True,
                      type=int)
  parser.add_argument('--embeddings_project',
                      dest='embeddings_project',
                      required=True)
  known_args, pipeline_args = parser.parse_known_args(argv)
  options = pipeline_options.PipelineOptions(pipeline_args)
  project = options.get_all_options()['project']

  match = re.match('gs://(\w+)/(.*)', known_args.input_file)
  bucket = match.group(1)
  filename = match.group(2)
  with tempfile.NamedTemporaryFile(delete=False) as f:
    client = storage.Client()
    bucket = client.bucket(bucket)
    blob = bucket.blob(filename)
    blob.download_to_file(f)
    f.seek(0)
    input_df = pd.read_csv(f)
  input = [(r['gcs_uri'], r['split'], r['label']) for _, r in input_df.iterrows()]

  if known_args.limit > 0:
    input = input[:known_args.limit]

  options.view_as(pipeline_options.SetupOptions).save_main_session = save_main_session
  options.view_as(pipeline_options.SetupOptions).requirements_file = _REQUIREMENTS_FILE
  options.view_as(pipeline_options.SetupOptions).setup_file = _SETUP_FILE

  with beam.Pipeline(options=options) as p:
    _ = (
        p | beam.Create(input) | beam.ParDo(
            inference_beam_lib.CreateExampleDoFn(
                project=project, output_path=known_args.output_path))
        | beam.ParDo(
            inference_beam_lib.GenerateEmbeddingsDoFn(
                known_args.embeddings_project, known_args.endpoint_id))
        | beam.ParDo(
            inference_beam_lib.ProcessPredictionDoFn(
                project, output_path=known_args.output_path))
        | beam.Map(print))

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  main()
