"""
Helper module for Google Cloud Store (GCS)
"""

from google.cloud.storage import Bucket


def download_blob(bucket: Bucket, source_blob_name: str, destination_file_name: str, print_name : str = None):
    """
    Downloads a blob from the bucket.

    https://cloud.google.com/storage/docs/downloading-objects

    Params:
    print_name : Print the file name when downloaded. Options: "source" or "dest" or None.
    """
    blob = bucket.blob(source_blob_name)
    try:
      blob.download_to_filename(destination_file_name)
    except Exception as e:
      print('Error during download - do you have the right permissions?')
      print(e)
      return

    if print_name == "source":
      print(f"Downloaded: {source_blob_name}")
    elif print_name == "dest":
       print(f"Downloaded: {destination_file_name}")
