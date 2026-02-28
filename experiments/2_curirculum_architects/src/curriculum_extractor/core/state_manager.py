"""State management for incremental and fault-tolerant processing."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class FileProcessingState:
    """State for a single file being processed."""

    file_path: str
    status: str  # "pending", "in_progress", "completed", "failed"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    rows_processed: int = 0
    rows_rejected: int = 0
    error_message: Optional[str] = None
    checksum: Optional[str] = None  # For detecting file changes


@dataclass
class PipelineState:
    """Overall pipeline state."""

    version: str = "0.2.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    total_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    files: Dict[str, FileProcessingState] = field(default_factory=dict)


class StateManager:
    """Manages processing state for incremental and fault-tolerant extraction.

    Features:
    - Tracks which files have been processed
    - Supports resumption after failures
    - Stores state in Parquet for efficiency
    - Optional S3 support via s3fs
    """

    STATE_FILE_NAME = "_extraction_state.json"
    STATE_PARQUET_NAME = "_extraction_state.parquet"

    def __init__(
        self,
        state_path: str | Path,
        filesystem: Optional[Any] = None,
        auto_save: bool = True,
    ):
        """Initialize state manager.

        Args:
            state_path: Directory to store state files
            filesystem: Optional s3fs filesystem for S3 support
            auto_save: Automatically save state after updates
        """
        self.state_path = str(state_path)
        self.fs = filesystem
        self.auto_save = auto_save
        self._state: Optional[PipelineState] = None

    @property
    def state(self) -> PipelineState:
        """Get current state, loading if necessary."""
        if self._state is None:
            self._state = self._load_state()
        return self._state

    def _get_state_file_path(self) -> str:
        """Get full path to state file."""
        if self.state_path.startswith("s3://"):
            return f"{self.state_path.rstrip('/')}/{self.STATE_FILE_NAME}"
        return str(Path(self.state_path) / self.STATE_FILE_NAME)

    def _load_state(self) -> PipelineState:
        """Load state from disk/S3."""
        state_file = self._get_state_file_path()

        try:
            if self.fs:
                if self.fs.exists(state_file):
                    with self.fs.open(state_file, "r") as f:
                        data = json.load(f)
                else:
                    return PipelineState()
            else:
                path = Path(state_file)
                if path.exists():
                    with open(path) as f:
                        data = json.load(f)
                else:
                    return PipelineState()

            # Reconstruct state from dict
            files = {
                k: FileProcessingState(**v) for k, v in data.get("files", {}).items()
            }
            return PipelineState(
                version=data.get("version", "0.2.0"),
                created_at=data.get("created_at", datetime.utcnow().isoformat()),
                updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
                total_files=data.get("total_files", 0),
                completed_files=data.get("completed_files", 0),
                failed_files=data.get("failed_files", 0),
                files=files,
            )
        except Exception as e:
            print(f"Warning: Could not load state: {e}. Starting fresh.")
            return PipelineState()

    def save_state(self) -> None:
        """Save current state to disk/S3."""
        state_file = self._get_state_file_path()
        self.state.updated_at = datetime.utcnow().isoformat()

        # Convert to dict
        data = {
            "version": self.state.version,
            "created_at": self.state.created_at,
            "updated_at": self.state.updated_at,
            "total_files": self.state.total_files,
            "completed_files": self.state.completed_files,
            "failed_files": self.state.failed_files,
            "files": {k: asdict(v) for k, v in self.state.files.items()},
        }

        if self.fs:
            with self.fs.open(state_file, "w") as f:
                json.dump(data, f, indent=2)
        else:
            path = Path(state_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

    def register_files(self, file_paths: List[str]) -> List[str]:
        """Register files for processing, returning those not yet processed.

        Args:
            file_paths: List of file paths to register

        Returns:
            List of file paths that need processing (not completed)
        """
        pending = []

        for fp in file_paths:
            if fp not in self.state.files:
                self.state.files[fp] = FileProcessingState(
                    file_path=fp, status="pending"
                )
                self.state.total_files += 1
                pending.append(fp)
            elif self.state.files[fp].status not in ("completed",):
                # Re-process failed or in_progress files
                pending.append(fp)

        if self.auto_save:
            self.save_state()

        return pending

    def mark_in_progress(self, file_path: str) -> None:
        """Mark file as currently being processed."""
        if file_path in self.state.files:
            self.state.files[file_path].status = "in_progress"
            self.state.files[file_path].started_at = datetime.utcnow().isoformat()
        else:
            self.state.files[file_path] = FileProcessingState(
                file_path=file_path,
                status="in_progress",
                started_at=datetime.utcnow().isoformat(),
            )
            self.state.total_files += 1

        if self.auto_save:
            self.save_state()

    def mark_completed(
        self, file_path: str, rows_processed: int = 0, rows_rejected: int = 0
    ) -> None:
        """Mark file as successfully processed."""
        if file_path in self.state.files:
            file_state = self.state.files[file_path]
            if file_state.status != "completed":
                self.state.completed_files += 1
            file_state.status = "completed"
            file_state.completed_at = datetime.utcnow().isoformat()
            file_state.rows_processed = rows_processed
            file_state.rows_rejected = rows_rejected

        if self.auto_save:
            self.save_state()

    def mark_failed(self, file_path: str, error_message: str) -> None:
        """Mark file as failed."""
        if file_path in self.state.files:
            file_state = self.state.files[file_path]
            if file_state.status != "failed":
                self.state.failed_files += 1
            file_state.status = "failed"
            file_state.error_message = error_message

        if self.auto_save:
            self.save_state()

    def is_completed(self, file_path: str) -> bool:
        """Check if file has been successfully processed."""
        return (
            file_path in self.state.files
            and self.state.files[file_path].status == "completed"
        )

    def get_pending_files(self) -> List[str]:
        """Get list of files pending processing."""
        return [
            fp
            for fp, state in self.state.files.items()
            if state.status in ("pending", "in_progress", "failed")
        ]

    def get_completed_files(self) -> List[str]:
        """Get list of successfully processed files."""
        return [
            fp for fp, state in self.state.files.items() if state.status == "completed"
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "total_files": self.state.total_files,
            "completed_files": self.state.completed_files,
            "failed_files": self.state.failed_files,
            "pending_files": self.state.total_files
            - self.state.completed_files
            - self.state.failed_files,
            "total_rows_processed": sum(
                s.rows_processed for s in self.state.files.values()
            ),
            "total_rows_rejected": sum(
                s.rows_rejected for s in self.state.files.values()
            ),
        }

    def reset(self) -> None:
        """Reset all state (for full refresh)."""
        self._state = PipelineState()
        if self.auto_save:
            self.save_state()

    def reset_file(self, file_path: str) -> None:
        """Reset state for a specific file (for reprocessing)."""
        if file_path in self.state.files:
            if self.state.files[file_path].status == "completed":
                self.state.completed_files -= 1
            elif self.state.files[file_path].status == "failed":
                self.state.failed_files -= 1
            del self.state.files[file_path]
            self.state.total_files -= 1

        if self.auto_save:
            self.save_state()
