"""
Module for managing/parsing MIMIC data files
"""

import re


# Example: 'files/p19/p19692222/s59566639/965b6053-a2c70d67-c0467ca6-02372346-fb7c6224.tfrecord'
FILE_PATTERN = re.compile(r"files/(?:\w+)/p(?P<subject_id>\w+)/s(?P<study_id>\w+)/(?P<dicom_id>[\w-]+)\.tfrecord")


def parse_embedding_file_pattern(file_path: str):
    """
    Extracts the subject_id, study_id, and dicom_id
    from the full file path string of a MIMIC CXR Embedding file:

    https://physionet.org/content/image-embeddings-mimic-cxr/

    Example input: files/p19/p19692222/s59566639/965b6053-a2c70d67-c0467ca6-02372346-fb7c6224.tfrecord

    """
    match = FILE_PATTERN.fullmatch(file_path)
    if not match:
        raise Exception(f"Failed to match file path: {file_path}")
    return (int(match[1]), int(match[2]), match[3])

