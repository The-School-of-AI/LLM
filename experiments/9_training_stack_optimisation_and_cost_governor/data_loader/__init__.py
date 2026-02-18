"""
Streaming data pipeline for large-scale training.

Three-layer pipeline: S3 → NVMe instance store → mmap dataset → async GPU prefetch.
Designed for 5 TB+ datasets on P5en.48xlarge with 8× H200 GPUs.
"""

from .instance_store import InstanceStoreManager
from .prefetch_loader import PrefetchDataLoader
from .s3_stager import S3Stager
from .streaming_dataset import StreamingTokenDataset

__all__ = [
    "InstanceStoreManager",
    "S3Stager",
    "StreamingTokenDataset",
    "PrefetchDataLoader",
]
