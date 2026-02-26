"""
Memory Profiler for PyTorch Training
=====================================

Provides comprehensive memory profiling capabilities using PyTorch Profiler
and Memory Snapshot API.

Features:
- **Memory Snapshot API** (NEW): Peak memory tracking, allocation timeline
- Memory profiling (CPU & CUDA)
- Performance profiling
- TensorBoard integration
- Chrome trace export
- Configurable scheduling
- Summary statistics

Deliverables:
- Memory allocation timeline
- Peak memory breakdown (pie chart via visualization)
- Memory bandwidth utilization over time
- Opportunities for memory reduction

"""

import pickle
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)


@dataclass
class ProfilerConfig:
    """Configuration for memory profiler."""

    # Output settings
    output_dir: str = "./profiler_logs"
    tensorboard_dir: Optional[str] = None  # If None, uses output_dir/tensorboard
    chrome_trace_file: str = "memory_profile.json"

    # Profiling activities
    profile_cpu: bool = True
    profile_cuda: bool = True

    # Memory profiling
    profile_memory: bool = True
    record_shapes: bool = True
    with_stack: bool = True

    # Memory Snapshot (NEW - for peak memory tracking)
    enable_memory_snapshot: bool = True  # Enable memory snapshot API
    snapshot_max_entries: int = 100000  # Max history entries for snapshot
    snapshot_file: str = "memory_snapshot.pickle"

    # Scheduling (when to profile)
    wait_steps: int = 5  # Steps to skip before profiling
    warmup_steps: int = 5  # Warmup steps
    active_steps: int = 10  # Steps to actively profile
    repeat: int = 1  # Number of times to repeat the cycle

    # Summary settings
    sort_by: str = (
        "cuda_time_total"  # cuda_time_total, cuda_memory_usage, cpu_time_total
    )
    row_limit: int = 20

    # Additional options
    with_flops: bool = False  # Estimate FLOPs (experimental)
    with_modules: bool = False  # Profile at module level


@dataclass
class MemoryStats:
    """Peak memory statistics at a point in time."""

    step: int
    timestamp: float

    # Current allocation
    allocated_bytes: int
    reserved_bytes: int

    # Peak values
    peak_allocated_bytes: int
    peak_reserved_bytes: int

    # Active allocations
    num_allocs: int

    # Derived metrics
    @property
    def allocated_gb(self) -> float:
        return self.allocated_bytes / (1024**3)

    @property
    def reserved_gb(self) -> float:
        return self.reserved_bytes / (1024**3)

    @property
    def peak_allocated_gb(self) -> float:
        return self.peak_allocated_bytes / (1024**3)

    @property
    def peak_reserved_gb(self) -> float:
        return self.peak_reserved_bytes / (1024**3)

    @property
    def fragmentation_ratio(self) -> float:
        """Memory fragmentation: reserved but not allocated."""
        if self.reserved_bytes == 0:
            return 0.0
        return 1.0 - (self.allocated_bytes / self.reserved_bytes)


class MemoryProfiler:
    """
    Memory profiler wrapper for PyTorch training.

    Provides two profiling modes:
    1. **Traditional Profiler** (cumulative metrics, operation-level detail)
    2. **Memory Snapshot API** (peak memory, allocation timeline, instant snapshots)

    The Memory Snapshot API gives you ACTUAL memory usage at any point in time,
    not cumulative allocations.
    """

    def __init__(self, config: ProfilerConfig):
        """
        Initialize profiler.

        Args:
            config: Profiler configuration
        """
        self.config = config
        self.profiler: Optional[profile] = None
        self._is_active = False
        self._step_count = 0
        self._memory_recording_active = False

        # Memory tracking
        self.memory_timeline: List[MemoryStats] = []
        self._start_time: Optional[float] = None
        self._peak_memory_bytes: int = 0
        self._peak_memory_step: int = 0

        # Setup directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if config.tensorboard_dir is None:
            self.tensorboard_dir = self.output_dir / "tensorboard"
        else:
            self.tensorboard_dir = Path(config.tensorboard_dir)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        # Build profiler
        self._build_profiler()

    def _build_profiler(self):
        """Build PyTorch profiler with configuration."""
        # Determine activities
        activities = []
        if self.config.profile_cpu:
            activities.append(ProfilerActivity.CPU)
        if self.config.profile_cuda and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        # Build schedule
        prof_schedule = schedule(
            wait=self.config.wait_steps,
            warmup=self.config.warmup_steps,
            active=self.config.active_steps,
            repeat=self.config.repeat,
        )

        # Create profiler
        self.profiler = profile(
            activities=activities,
            schedule=prof_schedule,
            on_trace_ready=tensorboard_trace_handler(str(self.tensorboard_dir)),
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            with_flops=self.config.with_flops,
            with_modules=self.config.with_modules,
        )

    def start(self):
        """Start profiling."""
        import time

        if self.profiler is None:
            raise RuntimeError("Profiler not initialized")

        # Start traditional profiler
        self.profiler.__enter__()
        self._is_active = True
        self._step_count = 0
        self._start_time = time.time()

        # Start memory snapshot recording if enabled
        if self.config.enable_memory_snapshot and torch.cuda.is_available():
            self._start_memory_recording()

        # Reset peak memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self._peak_memory_bytes = 0
            self._peak_memory_step = 0

        print("✓ Memory profiler started")
        print(f"  Output: {self.output_dir}")
        print(f"  TensorBoard: {self.tensorboard_dir}")
        if self.config.enable_memory_snapshot:
            print(
                f"  Memory Snapshot: ENABLED (max {self.config.snapshot_max_entries:,} entries)"
            )

    def _start_memory_recording(self):
        """Start memory history recording for snapshot API."""
        try:
            torch.cuda.memory._record_memory_history(
                max_entries=self.config.snapshot_max_entries
            )
            self._memory_recording_active = True
            print("  📊 Memory history recording started")
        except Exception as e:
            print(f"  ⚠️ Could not start memory recording: {e}")
            self._memory_recording_active = False

    def _stop_memory_recording(self):
        """Stop memory history recording."""
        if self._memory_recording_active:
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
                self._memory_recording_active = False
            except Exception as e:
                print(f"  ⚠️ Could not stop memory recording: {e}")

    def step(self):
        """
        Step the profiler.

        Call this after each training iteration.
        Also captures peak memory statistics.
        """

        if not self._is_active:
            return

        self.profiler.step()
        self._step_count += 1

        # Capture memory stats at this step
        if torch.cuda.is_available():
            stats = self._capture_memory_stats()
            self.memory_timeline.append(stats)

            # Track peak
            if stats.allocated_bytes > self._peak_memory_bytes:
                self._peak_memory_bytes = stats.allocated_bytes
                self._peak_memory_step = self._step_count

    def _capture_memory_stats(self) -> MemoryStats:
        """Capture current memory statistics."""
        import time

        return MemoryStats(
            step=self._step_count,
            timestamp=time.time() - (self._start_time or time.time()),
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            num_allocs=0,  # Would require more detailed tracking
        )

    def stop(self):
        """Stop profiling and export memory snapshot."""
        if not self._is_active:
            return

        self.profiler.__exit__(None, None, None)
        self._is_active = False
        print(f"✓ Memory profiler stopped after {self._step_count} steps")

        # Export memory snapshot if enabled
        if self.config.enable_memory_snapshot and self._memory_recording_active:
            self.export_memory_snapshot()
            self._stop_memory_recording()

        # Print peak memory summary
        self._print_peak_memory_summary()

    def _print_peak_memory_summary(self):
        """Print peak memory usage summary."""
        if not torch.cuda.is_available() or not self.memory_timeline:
            return

        print(f"\n{'='*60}")
        print("PEAK MEMORY SUMMARY")
        print(f"{'='*60}")

        # Get final stats
        final_stats = self.memory_timeline[-1] if self.memory_timeline else None

        if final_stats:
            print("\n📊 Memory Usage at Training End:")
            print(f"   Allocated: {final_stats.allocated_gb:.2f} GB")
            print(f"   Reserved:  {final_stats.reserved_gb:.2f} GB")
            print(f"   Fragmentation: {final_stats.fragmentation_ratio*100:.1f}%")

            print("\n🔺 Peak Memory (entire run):")
            print(f"   Peak Allocated: {final_stats.peak_allocated_gb:.2f} GB")
            print(f"   Peak Reserved:  {final_stats.peak_reserved_gb:.2f} GB")
            print(f"   Peak occurred at step: {self._peak_memory_step}")

            # Find step with minimum memory (potential optimization target)
            min_stats = min(self.memory_timeline, key=lambda s: s.allocated_bytes)
            print("\n📉 Minimum Memory:")
            print(
                f"   Min Allocated: {min_stats.allocated_gb:.2f} GB at step {min_stats.step}"
            )

            # Memory variance
            allocations = [s.allocated_bytes for s in self.memory_timeline]
            avg_alloc = sum(allocations) / len(allocations)
            variance = sum((a - avg_alloc) ** 2 for a in allocations) / len(allocations)
            std_dev = variance**0.5

            print("\n📈 Memory Variability:")
            print(f"   Average: {avg_alloc / (1024**3):.2f} GB")
            print(f"   Std Dev: {std_dev / (1024**3):.2f} GB")
            print(
                f"   Range: {(max(allocations) - min(allocations)) / (1024**3):.2f} GB"
            )

        print(f"{'='*60}\n")

    def should_stop(self, current_step: int) -> bool:
        """
        Check if profiling should stop based on schedule.

        Args:
            current_step: Current training step

        Returns:
            True if profiling window is complete
        """
        total_profile_steps = (
            self.config.wait_steps + self.config.warmup_steps + self.config.active_steps
        ) * self.config.repeat

        return current_step >= total_profile_steps

    def print_summary(
        self, sort_by: Optional[str] = None, row_limit: Optional[int] = None
    ):
        """
        Print profiling summary.

        Args:
            sort_by: Sort key (overrides config)
            row_limit: Number of rows to show (overrides config)
        """
        if self.profiler is None:
            print("No profiling data available")
            return

        sort_by = sort_by or self.config.sort_by
        row_limit = row_limit or self.config.row_limit

        print("\n" + "=" * 80)
        print("PROFILING SUMMARY")
        print("=" * 80)

        # Time summary
        if sort_by.startswith("cuda") and torch.cuda.is_available():
            print(f"\nTop {row_limit} operations by CUDA time:")
            print("-" * 80)
            print(
                self.profiler.key_averages().table(
                    sort_by="cuda_time_total", row_limit=row_limit
                )
            )

        if sort_by.startswith("cpu"):
            print(f"\nTop {row_limit} operations by CPU time:")
            print("-" * 80)
            print(
                self.profiler.key_averages().table(
                    sort_by="cpu_time_total", row_limit=row_limit
                )
            )

        # Memory summary
        if self.config.profile_memory and torch.cuda.is_available():
            print(f"\nTop {row_limit} operations by CUDA memory:")
            print("-" * 80)
            print(
                self.profiler.key_averages().table(
                    sort_by="cuda_memory_usage", row_limit=row_limit
                )
            )

        print("=" * 80 + "\n")

    def export_memory_snapshot(self, filename: Optional[str] = None):
        """
        Export memory snapshot for visualization.

        This creates a pickle file that can be visualized with:
        python -m torch.cuda._memory_viz trace_plot memory_snapshot.pickle

        Args:
            filename: Output filename (overrides config)
        """
        if not torch.cuda.is_available():
            print("CUDA not available, skipping memory snapshot")
            return

        filename = filename or self.config.snapshot_file
        output_path = self.output_dir / filename

        try:
            snapshot = torch.cuda.memory._snapshot()

            if snapshot is None or len(snapshot.get("segments", [])) == 0:
                print("⚠️ Memory snapshot is empty (no allocations recorded)")
                return

            with open(output_path, "wb") as f:
                pickle.dump(snapshot, f)

            print(f"✓ Memory snapshot exported to: {output_path}")
            print(
                f"  Visualize with: python -m torch.cuda._memory_viz trace_plot {output_path}"
            )

            # Also export a human-readable summary
            self._export_snapshot_summary(snapshot, output_path.with_suffix(".txt"))

        except Exception as e:
            print(f"⚠️ Could not export memory snapshot: {e}")

    def _export_snapshot_summary(self, snapshot: Dict, output_path: Path):
        """Export human-readable summary of memory snapshot."""
        try:
            with open(output_path, "w") as f:
                f.write("Memory Snapshot Summary\n")
                f.write("=" * 60 + "\n\n")

                segments = snapshot.get("segments", [])
                f.write(f"Total segments: {len(segments)}\n")

                # Calculate totals
                total_size = sum(seg.get("total_size", 0) for seg in segments)
                allocated_size = sum(seg.get("allocated_size", 0) for seg in segments)

                f.write(f"Total memory: {total_size / (1024**3):.2f} GB\n")
                f.write(f"Allocated memory: {allocated_size / (1024**3):.2f} GB\n")

                if total_size > 0:
                    f.write(f"Utilization: {allocated_size / total_size * 100:.1f}%\n")

                # Device info
                device_traces = snapshot.get("device_traces", [])
                if device_traces:
                    f.write(f"\nDevice traces: {len(device_traces)}\n")

            print(f"✓ Snapshot summary exported to: {output_path}")

        except Exception as e:
            print(f"⚠️ Could not export snapshot summary: {e}")

    def export_chrome_trace(self, filename: Optional[str] = None):
        """
        Export Chrome trace file for visualization.

        Args:
            filename: Output filename (overrides config)
        """
        if self.profiler is None:
            print("No profiling data available")
            return

        filename = filename or self.config.chrome_trace_file
        output_path = self.output_dir / filename

        try:
            self.profiler.export_chrome_trace(str(output_path))
            print(f"✓ Chrome trace exported to: {output_path}")
        except RuntimeError as e:
            if "Trace is already saved" in str(e):
                # Trace already exported by handler
                pass
            else:
                raise e

    def export_stacks(self, filename: str = "stack_trace.txt"):
        """
        Export stack traces to file.

        Args:
            filename: Output filename
        """
        if not self.config.with_stack:
            print("Stack tracing not enabled in config")
            return

        if self.profiler is None:
            print("No profiling data available")
            return

        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            f.write(
                self.profiler.key_averages(group_by_stack_n=5).table(
                    sort_by=self.config.sort_by, row_limit=50
                )
            )

        print(f"✓ Stack traces exported to: {output_path}")

    def export_memory_timeline(self, filename: str = "memory_timeline.json"):
        """
        Export memory timeline as JSON for custom visualization.

        Args:
            filename: Output filename
        """
        import json

        if not self.memory_timeline:
            print("No memory timeline data available")
            return

        output_path = self.output_dir / filename

        timeline_data = {
            "config": {
                "profiling_steps": self._step_count,
                "snapshot_enabled": self.config.enable_memory_snapshot,
            },
            "summary": {
                "peak_allocated_gb": self._peak_memory_bytes / (1024**3),
                "peak_step": self._peak_memory_step,
            },
            "timeline": [
                {
                    "step": s.step,
                    "timestamp": s.timestamp,
                    "allocated_gb": s.allocated_gb,
                    "reserved_gb": s.reserved_gb,
                    "peak_allocated_gb": s.peak_allocated_gb,
                    "fragmentation": s.fragmentation_ratio,
                }
                for s in self.memory_timeline
            ],
        }

        with open(output_path, "w") as f:
            json.dump(timeline_data, f, indent=2)

        print(f"✓ Memory timeline exported to: {output_path}")

    @contextmanager
    def profile_section(self, name: str):
        """
        Context manager for profiling a specific section.

        Args:
            name: Section name for labeling

        Example:
            with profiler.profile_section("forward_pass"):
                outputs = model(inputs)
        """
        if self._is_active:
            with torch.profiler.record_function(name):
                yield
        else:
            yield

    def get_total_steps(self) -> int:
        """Get total number of profiling steps based on schedule."""
        return (
            self.config.wait_steps + self.config.warmup_steps + self.config.active_steps
        ) * self.config.repeat

    def get_current_memory_gb(self) -> Dict[str, float]:
        """
        Get current memory usage in GB.

        Returns dict with:
        - allocated: Currently allocated memory
        - reserved: Currently reserved memory
        - peak_allocated: Peak allocated since last reset
        - peak_reserved: Peak reserved since last reset
        """
        if not torch.cuda.is_available():
            return {
                "allocated": 0,
                "reserved": 0,
                "peak_allocated": 0,
                "peak_reserved": 0,
            }

        return {
            "allocated": torch.cuda.memory_allocated() / (1024**3),
            "reserved": torch.cuda.memory_reserved() / (1024**3),
            "peak_allocated": torch.cuda.max_memory_allocated() / (1024**3),
            "peak_reserved": torch.cuda.max_memory_reserved() / (1024**3),
        }

    def get_memory_breakdown(self) -> Dict[str, Any]:
        """
        Get detailed memory breakdown for pie chart visualization.

        Returns dict with memory categories and their sizes.
        """
        if not torch.cuda.is_available():
            return {}

        stats = torch.cuda.memory_stats()

        return {
            "active": stats.get("active_bytes.all.current", 0) / (1024**3),
            "inactive": stats.get("inactive_split_bytes.all.current", 0) / (1024**3),
            "reserved_not_allocated": (
                torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            )
            / (1024**3),
            "num_allocs": stats.get("allocation.all.current", 0),
            "num_segments": stats.get("segment.all.current", 0),
        }


def create_default_profiler(
    output_dir: str = "./profiler_logs",
    profile_memory: bool = True,
    active_steps: int = 10,
    enable_memory_snapshot: bool = True,
) -> MemoryProfiler:
    """
    Create a profiler with sensible defaults.

    Args:
        output_dir: Directory for profiler output
        profile_memory: Whether to profile memory
        active_steps: Number of active profiling steps
        enable_memory_snapshot: Enable memory snapshot API for peak tracking

    Returns:
        Configured MemoryProfiler instance
    """
    config = ProfilerConfig(
        output_dir=output_dir,
        profile_memory=profile_memory,
        active_steps=active_steps,
        record_shapes=True,
        with_stack=True,
        enable_memory_snapshot=enable_memory_snapshot,
        repeat=1,  # Default to 1 to avoid OOM
    )

    return MemoryProfiler(config)


def visualize_memory_snapshot(snapshot_path: str):
    """
    Print instructions for visualizing memory snapshot.

    Args:
        snapshot_path: Path to the memory snapshot pickle file
    """
    print(
        f"""
╔══════════════════════════════════════════════════════════════════╗
║                   Memory Snapshot Visualization                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Option 1: Interactive trace plot (recommended)                  ║
║  ─────────────────────────────────────────────────────────────── ║
║  python -m torch.cuda._memory_viz trace_plot {snapshot_path}     ║
║                                                                  ║
║  Option 2: Segment visualization                                 ║
║  ─────────────────────────────────────────────────────────────── ║
║  python -m torch.cuda._memory_viz segment_plot {snapshot_path}   ║
║                                                                  ║
║  Option 3: Memory flamegraph                                     ║
║  ─────────────────────────────────────────────────────────────── ║
║  python -m torch.cuda._memory_viz trace_flamegraph {snapshot_path}║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    )
