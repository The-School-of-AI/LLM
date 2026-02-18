"""
NVMe instance store manager for P5en.48xlarge.

Handles RAID-0 setup across 8× NVMe SSDs, shard listing,
free-space reporting, and cleanup of consumed shards.
"""

import glob
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class InstanceStoreManager:
    """Manage NVMe instance-store for staging training data.

    On P5en.48xlarge, there are 8× 3840 GB NVMe SSDs.
    This class provides utilities to:
    - RAID-0 stripe them into a single mount (called from launch script)
    - List staged shards on the store
    - Report free space
    - Evict consumed shards to reclaim space
    """

    DEFAULT_DEVICES = [
        "/dev/nvme1n1",
        "/dev/nvme2n1",
        "/dev/nvme3n1",
        "/dev/nvme4n1",
        "/dev/nvme5n1",
        "/dev/nvme6n1",
        "/dev/nvme7n1",
        "/dev/nvme8n1",
    ]
    DEFAULT_MOUNT_POINT = "/data"

    def __init__(
        self,
        mount_point: str = DEFAULT_MOUNT_POINT,
        devices: Optional[List[str]] = None,
    ):
        self.mount_point = mount_point
        self.devices = devices or self.DEFAULT_DEVICES

    # ------------------------------------------------------------------
    # RAID-0 setup (idempotent — safe to call multiple times)
    # ------------------------------------------------------------------

    def setup_raid0(self) -> None:
        """Create RAID-0 array across NVMe SSDs and mount to ``self.mount_point``.

        This is designed to be called once at instance boot (e.g. from
        ``scripts/launch_p5en.sh``).  It is **idempotent**: if the mount
        point is already mounted, it skips.

        Must be run as root / with sudo.
        """
        if self._is_mounted():
            logger.info(
                "RAID-0 already mounted at %s — skipping setup", self.mount_point
            )
            return

        md_device = "/dev/md0"
        num = len(self.devices)
        logger.info("Creating RAID-0 with %d devices → %s", num, md_device)

        # Stop any existing md array (ignore errors)
        subprocess.run(
            ["mdadm", "--stop", md_device],
            check=False,
            capture_output=True,
        )

        # Create RAID-0
        subprocess.run(
            [
                "mdadm",
                "--create",
                md_device,
                "--level=0",
                f"--raid-devices={num}",
                *self.devices,
                "--force",
                "--run",
            ],
            check=True,
        )

        # Format with XFS (best for large sequential I/O)
        subprocess.run(["mkfs.xfs", "-f", md_device], check=True)

        # Mount
        os.makedirs(self.mount_point, exist_ok=True)
        subprocess.run(["mount", md_device, self.mount_point], check=True)

        logger.info("RAID-0 mounted at %s", self.mount_point)

    # ------------------------------------------------------------------
    # Shard management
    # ------------------------------------------------------------------

    def get_staged_shards(self, data_dir: str) -> List[str]:
        """Return sorted list of ``.npy`` shard paths under *data_dir*."""
        pattern = os.path.join(data_dir, "*.npy")
        shards = sorted(glob.glob(pattern))
        return shards

    def get_free_space(self, path: Optional[str] = None) -> int:
        """Return free space in bytes at *path* (defaults to mount point)."""
        target = path or self.mount_point
        usage = shutil.disk_usage(target)
        return usage.free

    def cleanup_consumed_shards(
        self, data_dir: str, current_shard_idx: int
    ) -> int:
        """Delete shards with index < *current_shard_idx*.

        Returns the number of shards removed.
        """
        shards = self.get_staged_shards(data_dir)
        removed = 0
        for shard_path in shards:
            # Extract shard index from filename (e.g. shard-00042.npy → 42)
            basename = Path(shard_path).stem
            try:
                idx = int(basename.split("-")[-1])
            except (ValueError, IndexError):
                continue

            if idx < current_shard_idx:
                os.remove(shard_path)
                logger.debug("Removed consumed shard: %s", shard_path)
                removed += 1

        if removed:
            logger.info(
                "Cleaned up %d consumed shards (idx < %d)", removed, current_shard_idx
            )
        return removed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_mounted(self) -> bool:
        """Check whether ``self.mount_point`` is already a mount point."""
        try:
            result = subprocess.run(
                ["mountpoint", "-q", self.mount_point],
                check=False,
                capture_output=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            # mountpoint command not available (e.g. on Windows dev boxes)
            return os.path.ismount(self.mount_point)
