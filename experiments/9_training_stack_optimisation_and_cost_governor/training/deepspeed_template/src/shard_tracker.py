"""
Shard Tracker — track which data files have been fully processed.

Maintains a ``consumed_shards.json`` manifest alongside training artefacts so
that on resume (or when new data arrives) already-processed shards are
excluded at the file-discovery stage. This replaces the expensive brute-force
"re-read everything and skip N micro-batches" approach for the SPDL pipeline.

Usage::

    tracker = ShardTracker("/mnt/nvme/training/consumed_shards.json")

    # Filter files before building the data pipeline
    remaining = tracker.exclude(all_bin_files)

    # After fully consuming a shard
    tracker.mark_processed("shard_003.bin", rank=0)
    tracker.save()

    # On next run, those shards are automatically excluded
    remaining = tracker.exclude(all_bin_files)
    # -> shard_003.bin is not in remaining

Manifest format (v1)::

    {
        "version": 1,
        "consumed_shards": {
            "shard_003.bin": {
                "completed_at": "2026-02-20T14:35:12.345678",
                "rank": 0,
                "source_dir": "/mnt/nvme/training/train"
            }
        }
    }

Keys are **basenames** (not full paths) so the manifest is portable across
machines, NVMe mounts, and S3 staging paths.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union


class ShardTracker:
    """Track which data shards/files have been fully processed.

    Parameters
    ----------
    manifest_path : str or Path
        Absolute path to the ``consumed_shards.json`` file.  Created
        automatically on the first :meth:`save` if it doesn't exist.
    auto_save : bool, optional
        When *True* (default), every call to :meth:`mark_processed`
        immediately persists the manifest to disk.  Set to *False* for
        batch workflows where you call :meth:`save` explicitly.
    """

    _VERSION = 1

    def __init__(
        self,
        manifest_path: Union[str, Path],
        auto_save: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.auto_save = auto_save
        self._lock = threading.Lock()

        # Internal state: basename -> metadata dict
        self._consumed: Dict[str, Dict[str, Any]] = {}

        # Load existing manifest if present
        if self.manifest_path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mark_processed(
        self,
        file_name: str,
        *,
        rank: Optional[int] = None,
        source_dir: Optional[str] = None,
    ) -> None:
        """Mark a single shard file as fully processed.

        Parameters
        ----------
        file_name : str
            The shard filename (or full path — only the basename is stored).
        rank : int, optional
            The distributed rank that consumed this shard.
        source_dir : str, optional
            Directory the shard was loaded from (informational).
        """
        basename = os.path.basename(file_name)
        meta: Dict[str, Any] = {
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        if rank is not None:
            meta["rank"] = rank
        if source_dir is not None:
            meta["source_dir"] = source_dir

        with self._lock:
            self._consumed[basename] = meta

        if self.auto_save:
            self.save()

    def mark_many_processed(
        self,
        file_names: Iterable[str],
        *,
        rank: Optional[int] = None,
        source_dir: Optional[str] = None,
    ) -> None:
        """Mark multiple shard files as processed in one call.

        More efficient than calling :meth:`mark_processed` in a loop when
        ``auto_save=True`` because it writes the manifest only once.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            for f in file_names:
                basename = os.path.basename(f)
                meta: Dict[str, Any] = {"completed_at": now}
                if rank is not None:
                    meta["rank"] = rank
                if source_dir is not None:
                    meta["source_dir"] = source_dir
                self._consumed[basename] = meta

        if self.auto_save:
            self.save()

    def is_processed(self, file_name: str) -> bool:
        """Return *True* if ``file_name`` (basename) was already processed."""
        basename = os.path.basename(file_name)
        with self._lock:
            return basename in self._consumed

    def get_processed_files(self) -> Set[str]:
        """Return the set of processed basenames."""
        with self._lock:
            return set(self._consumed.keys())

    def exclude(self, file_list: List[str]) -> List[str]:
        """Filter out already-processed files.

        Parameters
        ----------
        file_list : list of str
            List of filenames (basenames or full paths).

        Returns
        -------
        list of str
            Only the files that have **not** been processed, preserving
            the original order and format (basename or full path).
        """
        processed = self.get_processed_files()
        return [f for f in file_list if os.path.basename(f) not in processed]

    def get_manifest(self) -> Dict[str, Any]:
        """Return a copy of the full manifest dict."""
        with self._lock:
            return {
                "version": self._VERSION,
                "consumed_shards": dict(self._consumed),
            }

    @property
    def num_processed(self) -> int:
        """Number of shards marked as processed."""
        with self._lock:
            return len(self._consumed)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the manifest to disk (atomic write via temp file + rename)."""
        manifest = self.get_manifest()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to a temp file in the same directory, then rename.
        # This prevents corrupt manifests if the process is killed mid-write.
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.manifest_path.parent),
            prefix=".consumed_shards_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, sort_keys=True)
                f.write("\n")
            # os.replace is atomic on POSIX; on Windows it's atomic if
            # the destination is on the same volume (which it is here).
            os.replace(tmp_path, str(self.manifest_path))
        except BaseException:
            # Clean up the temp file on any error
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load(self) -> None:
        """Load the manifest from disk, replacing in-memory state."""
        if not self.manifest_path.exists():
            return

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        version = data.get("version", 1)
        if version != self._VERSION:
            raise ValueError(
                f"Unsupported consumed_shards.json version: {version} "
                f"(expected {self._VERSION})"
            )

        with self._lock:
            self._consumed = data.get("consumed_shards", {})

    def reset(self) -> None:
        """Clear all tracked shards (in memory and on disk)."""
        with self._lock:
            self._consumed.clear()
        if self.auto_save:
            self.save()

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ShardTracker(manifest_path={str(self.manifest_path)!r}, "
            f"num_processed={self.num_processed})"
        )

    def __contains__(self, file_name: str) -> bool:
        return self.is_processed(file_name)

    def __len__(self) -> int:
        return self.num_processed
