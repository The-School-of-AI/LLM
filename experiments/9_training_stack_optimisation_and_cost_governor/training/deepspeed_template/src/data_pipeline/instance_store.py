"""
Instance Store Manager for NVMe SSD management.

Manages the ephemeral NVMe instance store on P5en.48xlarge:
- RAID-0 striping across 8× NVMe SSDs → /data
- Tracking which shards are staged on local disk
- Free space monitoring
- Eviction of consumed shards when disk space is limited

Note:
    The RAID-0 setup (setup_raid0) requires root privileges and is
    intended to be called once from the launch script at instance boot,
    NOT during training.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class InstanceStoreManager:
    """
    Manages the local NVMe instance store for data staging.

    On P5en.48xlarge, there are 8× 3840 GB NVMe SSDs that can be
    RAID-0 striped to give ~30 TB of fast local storage. This class
    manages the lifecycle of data shards on that store.

    Attributes:
        data_dir: Root directory for staged data (typically /data/dolmo).
        nvme_devices: List of NVMe device paths for RAID-0 setup.
        raid_device: Path to the RAID-0 md device.
        mount_point: Mount point for the RAID array.
    """

    # Default NVMe devices on P5en.48xlarge
    DEFAULT_NVME_DEVICES = [f"/dev/nvme{i}n1" for i in range(1, 9)]
    DEFAULT_RAID_DEVICE = "/dev/md0"
    DEFAULT_MOUNT_POINT = "/data"

    def __init__(
        self,
        data_dir: str = "/data/dolmo",
        nvme_devices: Optional[List[str]] = None,
        raid_device: str = DEFAULT_RAID_DEVICE,
        mount_point: str = DEFAULT_MOUNT_POINT,
    ):
        """
        Initialize the instance store manager.

        Args:
            data_dir: Directory where shard files are staged.
            nvme_devices: List of NVMe block device paths.
            raid_device: Path to the md RAID device.
            mount_point: Where the RAID array is mounted.
        """
        self.data_dir = data_dir
        self.nvme_devices = nvme_devices or self.DEFAULT_NVME_DEVICES
        self.raid_device = raid_device
        self.mount_point = mount_point

    # ------------------------------------------------------------------
    # RAID-0 Setup (called once at instance boot, requires root)
    # ------------------------------------------------------------------

    def setup_raid0(self) -> None:
        """
        Create a RAID-0 array across NVMe SSDs and mount it.

        This stripes the 8× NVMe SSDs into a single ~30 TB volume.
        Must be run as root. Intended to be called from the launch
        script at instance boot time.

        Raises:
            RuntimeError: If any setup step fails.
            PermissionError: If not running as root.
        """
        if os.geteuid() != 0:
            raise PermissionError(
                "setup_raid0() requires root privileges. "
                "Run from the launch script with sudo."
            )

        # Filter to devices that actually exist
        available_devices = [d for d in self.nvme_devices if os.path.exists(d)]
        if not available_devices:
            raise RuntimeError(
                f"No NVMe devices found. Expected: {self.nvme_devices}"
            )

        num_devices = len(available_devices)
        logger.info(
            "Setting up RAID-0 across %d NVMe devices: %s",
            num_devices,
            available_devices,
        )

        try:
            # 1. Create RAID-0 array
            subprocess.run(
                [
                    "mdadm",
                    "--create",
                    self.raid_device,
                    "--level=0",
                    f"--raid-devices={num_devices}",
                    *available_devices,
                    "--force",
                    "--run",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("Created RAID-0 array at %s", self.raid_device)

            # 2. Create ext4 filesystem (fast format)
            subprocess.run(
                ["mkfs.ext4", "-F", "-E", "lazy_itable_init=0", self.raid_device],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("Created ext4 filesystem on %s", self.raid_device)

            # 3. Mount
            os.makedirs(self.mount_point, exist_ok=True)
            subprocess.run(
                ["mount", "-o", "noatime,nodiratime", self.raid_device, self.mount_point],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("Mounted RAID-0 at %s", self.mount_point)

            # 4. Create data directory
            os.makedirs(self.data_dir, exist_ok=True)
            logger.info("Data directory ready: %s", self.data_dir)

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"RAID-0 setup failed at step: {e.cmd}\n"
                f"  stdout: {e.stdout}\n"
                f"  stderr: {e.stderr}"
            ) from e

    # ------------------------------------------------------------------
    # Shard Inventory
    # ------------------------------------------------------------------

    def get_staged_shards(self, extension: str = ".npy") -> List[str]:
        """
        List shard files already staged on the instance store.

        Returns them in **deterministic sorted order** (by filename)
        so that the shard index is consistent across restarts.

        Args:
            extension: File extension to look for (default: .npy).

        Returns:
            Sorted list of absolute paths to staged shard files.
        """
        if not os.path.isdir(self.data_dir):
            return []

        shards = sorted(
            str(p)
            for p in Path(self.data_dir).iterdir()
            if p.is_file() and p.suffix == extension
        )
        return shards

    def shard_exists(self, shard_filename: str) -> bool:
        """
        Check whether a specific shard is already staged locally.

        Args:
            shard_filename: Filename of the shard (e.g., 'shard-00042.npy').

        Returns:
            True if the shard file exists on the instance store.
        """
        return os.path.isfile(os.path.join(self.data_dir, shard_filename))

    # ------------------------------------------------------------------
    # Free Space
    # ------------------------------------------------------------------

    def get_free_space_bytes(self) -> int:
        """
        Get available free space on the instance store in bytes.

        Returns:
            Free space in bytes. Returns 0 if the path does not exist.
        """
        if not os.path.isdir(self.data_dir):
            return 0

        stat = os.statvfs(self.data_dir)
        return stat.f_bavail * stat.f_frsize

    def get_free_space_gb(self) -> float:
        """
        Get available free space in gigabytes.

        Returns:
            Free space in GB.
        """
        return self.get_free_space_bytes() / (1024 ** 3)

    # ------------------------------------------------------------------
    # Shard Eviction
    # ------------------------------------------------------------------

    def cleanup_consumed_shards(
        self,
        current_shard_idx: int,
        shard_paths: List[str],
        keep_behind: int = 1,
    ) -> int:
        """
        Delete shards that have already been fully consumed by training.

        This is only needed when the dataset is larger than the instance
        store (e.g., >30 TB). For typical 5 TB datasets, all shards fit
        and this is a no-op.

        Args:
            current_shard_idx: Index of the shard currently being read.
            shard_paths: Ordered list of all shard paths (matching the
                         deterministic order used by StreamingTokenDataset).
            keep_behind: Number of past shards to keep (safety margin).

        Returns:
            Number of shards deleted.
        """
        evict_up_to = max(0, current_shard_idx - keep_behind)
        deleted = 0

        for idx in range(evict_up_to):
            path = shard_paths[idx]
            if os.path.isfile(path):
                try:
                    os.remove(path)
                    deleted += 1
                    logger.debug("Evicted consumed shard: %s", path)
                except OSError as e:
                    logger.warning("Failed to evict shard %s: %s", path, e)

        if deleted > 0:
            logger.info(
                "Evicted %d consumed shard(s) (up to index %d)",
                deleted,
                evict_up_to - 1,
            )

        return deleted

    def cleanup_all(self) -> None:
        """
        Remove all staged data from the instance store.

        Use with caution — this deletes everything under ``data_dir``.
        """
        if os.path.isdir(self.data_dir):
            shutil.rmtree(self.data_dir)
            os.makedirs(self.data_dir, exist_ok=True)
            logger.info("Cleaned up all data in %s", self.data_dir)
