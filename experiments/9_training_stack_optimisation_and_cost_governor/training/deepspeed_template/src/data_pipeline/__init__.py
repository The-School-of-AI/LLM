"""
Data Pipeline for large-scale streaming training.

This package provides a shard-aware, fault-tolerant data pipeline that:
- Stages pre-tokenized shards from S3 → NVMe instance store
- Memory-maps staged files (zero-copy, constant memory)
- Tracks shard-level progress for exact checkpoint/resume
- Prefetches batches asynchronously so GPUs never wait for data

Components:
- InstanceStoreManager: NVMe RAID-0 setup and shard lifecycle management
- S3Stager: Deterministic shard discovery, download, and background staging
- StreamingTokenDataset: Memory-mapped, shard-aware torch Dataset
- PrefetchDataLoader: Async GPU transfer wrapper for DataLoader
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
