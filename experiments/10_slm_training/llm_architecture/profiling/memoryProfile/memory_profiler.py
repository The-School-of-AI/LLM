"""
Memory Profiler for PyTorch Training
=====================================

Provides easy-to-use memory profiling capabilities using PyTorch Profiler.

Features:
- Memory profiling (CPU & CUDA)
- Performance profiling
- TensorBoard integration
- Chrome trace export
- Configurable scheduling
- Summary statistics

"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
from contextlib import contextmanager

import torch
from torch.profiler import (
    profile,
    ProfilerActivity,
    tensorboard_trace_handler,
    schedule
)


@dataclass
class ProfilerConfig:
    """Configuration for memory profiler."""
    
    # Output settings
    output_dir: str = './profiler_logs'
    tensorboard_dir: Optional[str] = None  # If None, uses output_dir/tensorboard
    chrome_trace_file: str = 'memory_profile.json'
    
    # Profiling activities
    profile_cpu: bool = True
    profile_cuda: bool = True
    
    # Memory profiling
    profile_memory: bool = True
    record_shapes: bool = True
    with_stack: bool = True
    
    # Scheduling (when to profile)
    wait_steps: int = 5      # Steps to skip before profiling
    warmup_steps: int = 5    # Warmup steps
    active_steps: int = 10   # Steps to actively profile
    repeat: int = 1          # Number of times to repeat the cycle
    
    # Summary settings
    sort_by: str = 'cuda_time_total'  # cuda_time_total, cuda_memory_usage, cpu_time_total
    row_limit: int = 20
    
    # Additional options
    with_flops: bool = False  # Estimate FLOPs (experimental)
    with_modules: bool = False  # Profile at module level


class MemoryProfiler:
    """
    Memory profiler wrapper for PyTorch training.
    
    Simplifies profiling setup and provides convenient methods for
    integration with training loops.
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
        
        # Setup directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if config.tensorboard_dir is None:
            self.tensorboard_dir = self.output_dir / 'tensorboard'
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
            repeat=self.config.repeat
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
            with_modules=self.config.with_modules
        )
    
    def start(self):
        """Start profiling."""
        if self.profiler is None:
            raise RuntimeError("Profiler not initialized")
        
        self.profiler.__enter__()
        self._is_active = True
        self._step_count = 0
        print(f"✓ Memory profiler started")
        print(f"  Output: {self.output_dir}")
        print(f"  TensorBoard: {self.tensorboard_dir}")
    
    def step(self):
        """
        Step the profiler.
        
        Call this after each training iteration.
        """
        if not self._is_active:
            return
        
        self.profiler.step()
        self._step_count += 1
    
    def stop(self):
        """Stop profiling."""
        if not self._is_active:
            return
        
        self.profiler.__exit__(None, None, None)
        self._is_active = False
        print(f"✓ Memory profiler stopped after {self._step_count} steps")
    
    def should_stop(self, current_step: int) -> bool:
        """
        Check if profiling should stop based on schedule.
        
        Args:
            current_step: Current training step
            
        Returns:
            True if profiling window is complete
        """
        total_profile_steps = (
            self.config.wait_steps +
            self.config.warmup_steps +
            self.config.active_steps
        ) * self.config.repeat
        
        return current_step >= total_profile_steps
    
    def print_summary(self, sort_by: Optional[str] = None, row_limit: Optional[int] = None):
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
        
        print("\n" + "="*80)
        print("PROFILING SUMMARY")
        print("="*80)
        
        # Time summary
        if sort_by.startswith('cuda') and torch.cuda.is_available():
            print(f"\nTop {row_limit} operations by CUDA time:")
            print("-"*80)
            print(self.profiler.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=row_limit
            ))
        
        if sort_by.startswith('cpu'):
            print(f"\nTop {row_limit} operations by CPU time:")
            print("-"*80)
            print(self.profiler.key_averages().table(
                sort_by="cpu_time_total",
                row_limit=row_limit
            ))
        
        # Memory summary
        if self.config.profile_memory and torch.cuda.is_available():
            print(f"\nTop {row_limit} operations by CUDA memory:")
            print("-"*80)
            print(self.profiler.key_averages().table(
                sort_by="cuda_memory_usage",
                row_limit=row_limit
            ))
        
        print("="*80 + "\n")
    
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
        
        with open(output_path, 'w') as f:
            f.write(self.profiler.key_averages(group_by_stack_n=5).table(
                sort_by=self.config.sort_by,
                row_limit=50
            ))
        
        print(f"✓ Stack traces exported to: {output_path}")
    
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
            self.config.wait_steps +
            self.config.warmup_steps +
            self.config.active_steps
        ) * self.config.repeat


def create_default_profiler(
    output_dir: str = './profiler_logs',
    profile_memory: bool = True,
    active_steps: int = 10
) -> MemoryProfiler:
    """
    Create a profiler with sensible defaults.
    
    Args:
        output_dir: Directory for profiler output
        profile_memory: Whether to profile memory
        active_steps: Number of active profiling steps
        
    Returns:
        Configured MemoryProfiler instance
    """
    config = ProfilerConfig(
        output_dir=output_dir,
        profile_memory=profile_memory,
        active_steps=active_steps,
        record_shapes=True,
        with_stack=True
    )
    
    return MemoryProfiler(config)