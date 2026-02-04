"""Distributed processing with Ray for large-scale extraction."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

import s3fs

from .extractor import CurriculumExtractor
from .state_manager import StateManager


class DistributedExtractor:
    """Distributed extraction using Ray for parallel processing.

    Designed for processing ~1TB of parquet data from S3 efficiently:
    - Parallel file processing across workers
    - State management for fault tolerance
    - Incremental processing (skip already processed files)
    """

    def __init__(
        self,
        curriculum_path: str | Path,
        metadata_output_path: str,
        rejection_output_path: str,
        state_path: str,
        s3_bucket: Optional[str] = None,
        num_workers: int = 4,
        metrics_config_path: Optional[str | Path] = None,
    ):
        """Initialize distributed extractor.

        Args:
            curriculum_path: Path to curriculum YAML
            metadata_output_path: S3/local path for metadata output
            rejection_output_path: S3/local path for rejection output
            state_path: S3/local path for state management
            s3_bucket: Optional S3 bucket name (for S3 mode)
            num_workers: Number of Ray workers
            metrics_config_path: Optional path to metrics config
        """
        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray is required for distributed processing. Install with: pip install ray[default]"
            )

        self.curriculum_path = str(curriculum_path)
        self.metadata_output_path = metadata_output_path
        self.rejection_output_path = rejection_output_path
        self.state_path = state_path
        self.s3_bucket = s3_bucket
        self.num_workers = num_workers
        self.metrics_config_path = (
            str(metrics_config_path) if metrics_config_path else None
        )

        # Initialize filesystem
        self.fs = s3fs.S3FileSystem() if s3_bucket else None

        # Initialize state manager
        self.state_manager = StateManager(state_path, self.fs)

    def list_input_files(
        self, input_path: str, pattern: str = "*.parquet"
    ) -> List[str]:
        """List parquet files in input path.

        Args:
            input_path: S3 or local path to input directory
            pattern: Glob pattern for files

        Returns:
            List of file paths
        """
        if self.fs:
            # S3 path
            files = self.fs.glob(f"{input_path.rstrip('/')}/{pattern}")
            return [f"s3://{f}" for f in files]
        else:
            # Local path
            return [str(p) for p in Path(input_path).glob(pattern)]

    def process_files(
        self,
        input_files: List[str],
        batch_size: int = 10000,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> Dict[str, Any]:
        """Process files in parallel with Ray.

        Args:
            input_files: List of input file paths
            batch_size: Rows per batch for processing
            progress_callback: Optional callback(file_path, rows_processed)

        Returns:
            Processing statistics
        """
        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,  # Disable dashboard to reduce overhead
                num_cpus=self.num_workers,  # Limit to requested workers
                object_store_memory=2 * 1024 * 1024 * 1024,  # 2GB object store
            )

        # Filter to pending files
        pending_files = self.state_manager.register_files(input_files)

        if not pending_files:
            return {
                "status": "completed",
                "message": "All files already processed",
                "total_files": len(input_files),
                "processed_files": 0,
            }

        # Create Ray references
        curriculum_ref = ray.put(self.curriculum_path)
        metrics_config_ref = ray.put(self.metrics_config_path)
        metadata_path_ref = ray.put(self.metadata_output_path)
        rejection_path_ref = ray.put(self.rejection_output_path)

        # Process files in parallel
        @ray.remote
        def process_single_file(
            file_path: str,
            curriculum_path: str,
            metrics_config_path: Optional[str],
            metadata_output: str,
            rejection_output: str,
            batch_size: int,
            use_s3: bool,
        ) -> Dict[str, Any]:
            """Process a single file (runs on worker)."""
            fs = s3fs.S3FileSystem() if use_s3 else None

            extractor = CurriculumExtractor(
                curriculum_path=curriculum_path,
                metrics_config_path=metrics_config_path,
                metadata_output_path=metadata_output,
                rejection_output_path=rejection_output,
                filesystem=fs,
            )

            try:
                if use_s3:
                    result = extractor.process_parquet_s3(
                        input_path=file_path,
                        batch_size=batch_size,
                    )
                else:
                    result = extractor.process_parquet(
                        input_path=file_path,
                        batch_size=batch_size,
                    )
                return {"file": file_path, **result}
            except Exception as e:
                return {
                    "file": file_path,
                    "status": "failed",
                    "error": str(e),
                }

        # Submit tasks
        futures = []
        for file_path in pending_files:
            future = process_single_file.remote(
                file_path,
                curriculum_ref,
                metrics_config_ref,
                metadata_path_ref,
                rejection_path_ref,
                batch_size,
                self.s3_bucket is not None,
            )
            futures.append(future)

        # Collect results
        results = []
        completed = 0
        failed = 0
        total_rows = 0
        total_rejected = 0

        for future in futures:
            result = ray.get(future)
            results.append(result)

            file_path = result["file"]

            if result.get("status") == "completed":
                completed += 1
                total_rows += result.get("processed_rows", 0)
                total_rejected += result.get("rejected_rows", 0)
                self.state_manager.mark_completed(
                    file_path,
                    rows_processed=result.get("processed_rows", 0),
                    rows_rejected=result.get("rejected_rows", 0),
                )
            elif result.get("status") == "failed":
                failed += 1
                self.state_manager.mark_failed(
                    file_path, result.get("error", "Unknown")
                )

            if progress_callback:
                progress_callback(file_path, result.get("processed_rows", 0))

        return {
            "status": "completed",
            "total_files": len(pending_files),
            "completed_files": completed,
            "failed_files": failed,
            "total_rows_processed": total_rows,
            "total_rows_rejected": total_rejected,
            "results": results,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.state_manager.get_stats()

    def reset_state(self) -> None:
        """Reset all state for full refresh."""
        self.state_manager.reset()


def run_distributed_extraction(
    input_path: str,
    curriculum_path: str,
    output_base_path: str,
    s3_mode: bool = False,
    num_workers: int = 4,
    batch_size: int = 10000,
) -> Dict[str, Any]:
    """Convenience function to run distributed extraction.

    Args:
        input_path: Path to input parquet files (directory)
        curriculum_path: Path to curriculum YAML
        output_base_path: Base path for outputs
        s3_mode: Whether to use S3
        num_workers: Number of parallel workers
        batch_size: Batch size for processing

    Returns:
        Processing statistics
    """
    metadata_path = f"{output_base_path}/metadata"
    rejection_path = f"{output_base_path}/rejections"
    state_path = f"{output_base_path}/state"

    extractor = DistributedExtractor(
        curriculum_path=curriculum_path,
        metadata_output_path=metadata_path,
        rejection_output_path=rejection_path,
        state_path=state_path,
        s3_bucket="dummy" if s3_mode else None,
        num_workers=num_workers,
    )

    input_files = extractor.list_input_files(input_path)
    print(f"Found {len(input_files)} input files")

    def progress(file: str, rows: int):
        print(f"Processed {file}: {rows} rows")

    return extractor.process_files(
        input_files=input_files,
        batch_size=batch_size,
        progress_callback=progress,
    )
