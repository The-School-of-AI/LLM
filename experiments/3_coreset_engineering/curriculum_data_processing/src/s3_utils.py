import boto3
import logging
from typing import List, Dict, Set
from pyarrow.fs import S3FileSystem

logger = logging.getLogger("data_pipeline")


def list_s3_subfolders(bucket: str, prefix: str) -> List[str]:
    """Lists immediate subfolders under a given S3 prefix."""
    s3 = boto3.client("s3")
    if not prefix.endswith("/"):
        prefix += "/"

    paginator = s3.get_paginator("list_objects_v2")
    subfolders = set()

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                # Extract the last folder name
                folder = cp.get("Prefix").rstrip("/").split("/")[-1]
                subfolders.add(folder)
    except Exception as e:
        logger.error(f"Error listing S3 folders for {bucket}/{prefix}: {e}")
        raise

    return sorted(list(subfolders))


def discover_dataset_structure(bucket: str, base_prefix: str) -> Dict[str, List[str]]:
    """
    Discovers sources and their associated bands.
    Returns: { 'source_name': ['B0', 'B1', ...] }
    """
    structure = {}
    sources = list_s3_subfolders(bucket, base_prefix)

    for source_folder in sources:
        # source_folder is usually 'source=books'
        source_name = (
            source_folder.split("=")[-1] if "=" in source_folder else source_folder
        )
        bands_prefix = f"{base_prefix}/{source_folder}/bands/"

        try:
            # Check if 'bands' exists
            band_folders = list_s3_subfolders(bucket, bands_prefix)
            bands = []
            for bf in band_folders:
                band_val = bf.split("=")[-1] if "=" in bf else bf
                bands.append(band_val)
            structure[source_name] = bands
        except Exception:
            logger.warning(
                f"Could not find bands for source {source_name} at {bands_prefix}"
            )
            continue

    return structure


def get_pyarrow_filesystem(region: str = "us-east-1") -> S3FileSystem:
    """Returns a PyArrow S3FileSystem instance."""
    return S3FileSystem(region=region)
