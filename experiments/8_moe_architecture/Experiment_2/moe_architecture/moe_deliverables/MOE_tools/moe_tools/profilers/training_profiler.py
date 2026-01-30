#!/usr/bin/env python3
"""
Training Stack Profiler Integration
===================================
Integration layer for profiling MoE training:
- GPU utilization monitoring
- Communication overhead analysis
- Memory bandwidth tracking
- Expert computation timing
- Routing latency breakdown

Integrates with:
- PyTorch Profiler
- NVIDIA Nsight
- Weights & Biases
- TensorBoard

Usage:
    from profilers.training_profiler import TrainingProfiler
    
    profiler = TrainingProfiler(config)
    
    with profiler.profile_step():
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    profiler.log_metrics()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from contextlib import contextmanager
from collections import defaultdict
import time
import json


@dataclass
class ProfilerConfig:
    """Configuration for training profiler."""
    
    # Profiling toggles
    profile_gpu: bool = True
    profile_memory: bool = True
    profile_communication: bool = True
    profile_experts: bool = True
    profile_router: bool = True
    
    # Sampling
    profile_every_n_steps: int = 10
    warmup_steps: int = 5
    
    # Integration backends
    use_torch_profiler: bool = True
    use_wandb: bool = False
    use_tensorboard: bool = True
    
    # Output
    output_dir: str = "./profiles"
    trace_memory: bool = False


@dataclass
class StepProfile:
    """Profile data for a single training step."""
    
    step: int = 0
    timestamp: float = 0.0
    
    # Timing (seconds)
    total_time: float = 0.0
    forward_time: float = 0.0
    backward_time: float = 0.0
    optimizer_time: float = 0.0
    
    # Detailed breakdown
    attention_time: float = 0.0
    router_time: float = 0.0
    expert_dispatch_time: float = 0.0
    expert_compute_time: float = 0.0
    expert_combine_time: float = 0.0
    
    # Communication
    allreduce_time: float = 0.0
    expert_alltoall_time: float = 0.0
    pipeline_comm_time: float = 0.0
    
    # Memory (bytes)
    peak_memory: int = 0
    allocated_memory: int = 0
    cached_memory: int = 0
    
    # GPU utilization (%)
    gpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    
    # Throughput
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0
    mfu: float = 0.0  # Model FLOPs Utilization


class Timer:
    """Simple timer context manager."""
    
    def __init__(self):
        self.start_time = 0.0
        self.elapsed = 0.0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time


class TrainingProfiler:
    """
    MoE Training Stack Profiler.
    
    Provides:
    1. Per-step timing breakdown
    2. GPU utilization monitoring
    3. Memory tracking
    4. Communication analysis
    5. Expert-level profiling
    6. Integration with logging backends
    """
    
    def __init__(self, config: ProfilerConfig):
        self.config = config
        self.current_step = 0
        self.profiles: List[StepProfile] = []
        
        # Active timers
        self._timers: Dict[str, Timer] = {}
        self._current_profile: Optional[StepProfile] = None
        
        # Hooks storage
        self._hooks: List[Any] = []
        
        # Aggregated stats
        self._stats = defaultdict(list)
        
        # Initialize backends
        self._init_backends()
    
    def _init_backends(self):
        """Initialize logging backends."""
        self._torch_profiler = None
        self._wandb_run = None
        self._tb_writer = None
        
        # TensorBoard
        if self.config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=f"{self.config.output_dir}/tensorboard")
            except ImportError:
                print("TensorBoard not available")
        
        # Weights & Biases
        if self.config.use_wandb:
            try:
                import wandb
                if wandb.run is None:
                    wandb.init(project="moe-training")
                self._wandb_run = wandb.run
            except ImportError:
                print("W&B not available")
    
    @contextmanager
    def profile_step(self, batch_size: int = 0, seq_length: int = 0):
        """
        Context manager for profiling a training step.
        
        Usage:
            with profiler.profile_step(batch_size=8, seq_length=2048):
                # training code
        """
        should_profile = (
            self.current_step >= self.config.warmup_steps and
            self.current_step % self.config.profile_every_n_steps == 0
        )
        
        if should_profile:
            self._current_profile = StepProfile(
                step=self.current_step,
                timestamp=time.time()
            )
            step_timer = Timer()
            step_timer.__enter__()
        
        try:
            yield self
        finally:
            if should_profile:
                step_timer.__exit__(None, None, None)
                self._current_profile.total_time = step_timer.elapsed
                
                # Calculate throughput
                if batch_size > 0 and seq_length > 0:
                    tokens = batch_size * seq_length
                    self._current_profile.tokens_per_second = tokens / max(step_timer.elapsed, 1e-6)
                    self._current_profile.samples_per_second = batch_size / max(step_timer.elapsed, 1e-6)
                
                # Collect GPU stats if available
                self._collect_gpu_stats()
                
                # Store profile
                self.profiles.append(self._current_profile)
                
                # Update aggregated stats
                self._update_stats()
            
            self.current_step += 1
    
    @contextmanager
    def time_region(self, name: str):
        """
        Time a specific region within a step.
        
        Usage:
            with profiler.time_region('attention'):
                # attention code
        """
        timer = Timer()
        timer.__enter__()
        try:
            yield
        finally:
            timer.__exit__(None, None, None)
            if self._current_profile:
                # Map name to profile attribute
                attr_map = {
                    'forward': 'forward_time',
                    'backward': 'backward_time',
                    'optimizer': 'optimizer_time',
                    'attention': 'attention_time',
                    'router': 'router_time',
                    'expert_dispatch': 'expert_dispatch_time',
                    'expert_compute': 'expert_compute_time',
                    'expert_combine': 'expert_combine_time',
                    'allreduce': 'allreduce_time',
                    'expert_alltoall': 'expert_alltoall_time',
                    'pipeline_comm': 'pipeline_comm_time',
                }
                if name in attr_map:
                    setattr(self._current_profile, attr_map[name], timer.elapsed)
    
    def _collect_gpu_stats(self):
        """Collect GPU memory and utilization stats."""
        if not self._current_profile:
            return
        
        try:
            import torch
            if torch.cuda.is_available():
                self._current_profile.peak_memory = torch.cuda.max_memory_allocated()
                self._current_profile.allocated_memory = torch.cuda.memory_allocated()
                self._current_profile.cached_memory = torch.cuda.memory_reserved()
                
                # Reset peak memory for next step
                torch.cuda.reset_peak_memory_stats()
        except:
            pass
        
        # GPU utilization (requires pynvml)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            self._current_profile.gpu_utilization = util.gpu
            self._current_profile.memory_utilization = util.memory
        except:
            pass
    
    def _update_stats(self):
        """Update aggregated statistics."""
        if not self._current_profile:
            return
        
        p = self._current_profile
        self._stats['total_time'].append(p.total_time)
        self._stats['tokens_per_second'].append(p.tokens_per_second)
        self._stats['peak_memory'].append(p.peak_memory)
        self._stats['gpu_utilization'].append(p.gpu_utilization)
    
    def log_metrics(self, extra_metrics: Optional[Dict] = None):
        """Log metrics to configured backends."""
        if not self.profiles:
            return
        
        latest = self.profiles[-1]
        
        metrics = {
            'step': latest.step,
            'time/total': latest.total_time,
            'time/forward': latest.forward_time,
            'time/backward': latest.backward_time,
            'time/attention': latest.attention_time,
            'time/router': latest.router_time,
            'time/expert_compute': latest.expert_compute_time,
            'time/allreduce': latest.allreduce_time,
            'time/expert_alltoall': latest.expert_alltoall_time,
            'throughput/tokens_per_sec': latest.tokens_per_second,
            'throughput/samples_per_sec': latest.samples_per_second,
            'memory/peak_gb': latest.peak_memory / 1e9,
            'memory/allocated_gb': latest.allocated_memory / 1e9,
            'gpu/utilization': latest.gpu_utilization,
            'gpu/memory_util': latest.memory_utilization,
        }
        
        if extra_metrics:
            metrics.update(extra_metrics)
        
        # TensorBoard
        if self._tb_writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(k, v, latest.step)
        
        # W&B
        if self._wandb_run:
            try:
                import wandb
                wandb.log(metrics)
            except:
                pass
    
    def get_summary(self) -> Dict:
        """Get profiling summary statistics."""
        if not self.profiles:
            return {'status': 'no_data'}
        
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0
        
        def percentile(lst, p):
            if not lst:
                return 0
            sorted_lst = sorted(lst)
            idx = int(len(sorted_lst) * p / 100)
            return sorted_lst[min(idx, len(sorted_lst) - 1)]
        
        latest = self.profiles[-1]
        
        return {
            'total_steps_profiled': len(self.profiles),
            'latest_step': latest.step,
            
            'timing': {
                'avg_step_time': f"{avg(self._stats['total_time'])*1000:.1f}ms",
                'p50_step_time': f"{percentile(self._stats['total_time'], 50)*1000:.1f}ms",
                'p99_step_time': f"{percentile(self._stats['total_time'], 99)*1000:.1f}ms",
            },
            
            'throughput': {
                'avg_tokens_per_sec': f"{avg(self._stats['tokens_per_second']):.0f}",
                'max_tokens_per_sec': f"{max(self._stats['tokens_per_second']) if self._stats['tokens_per_second'] else 0:.0f}",
            },
            
            'memory': {
                'avg_peak_gb': f"{avg(self._stats['peak_memory'])/1e9:.2f}",
                'max_peak_gb': f"{max(self._stats['peak_memory']) if self._stats['peak_memory'] else 0:.2f}",
            },
            
            'gpu': {
                'avg_utilization': f"{avg(self._stats['gpu_utilization']):.1f}%",
            },
            
            'breakdown_latest': {
                'attention': f"{latest.attention_time*1000:.1f}ms",
                'router': f"{latest.router_time*1000:.1f}ms",
                'expert_dispatch': f"{latest.expert_dispatch_time*1000:.1f}ms",
                'expert_compute': f"{latest.expert_compute_time*1000:.1f}ms",
                'allreduce': f"{latest.allreduce_time*1000:.1f}ms",
                'expert_alltoall': f"{latest.expert_alltoall_time*1000:.1f}ms",
            },
        }
    
    def export_profiles(self, filepath: str):
        """Export all profiles to JSON."""
        data = {
            'config': {
                'profile_every_n': self.config.profile_every_n_steps,
                'warmup_steps': self.config.warmup_steps,
            },
            'profiles': [
                {
                    'step': p.step,
                    'timestamp': p.timestamp,
                    'total_time': p.total_time,
                    'forward_time': p.forward_time,
                    'backward_time': p.backward_time,
                    'attention_time': p.attention_time,
                    'router_time': p.router_time,
                    'expert_compute_time': p.expert_compute_time,
                    'tokens_per_second': p.tokens_per_second,
                    'peak_memory': p.peak_memory,
                    'gpu_utilization': p.gpu_utilization,
                }
                for p in self.profiles
            ],
            'summary': self.get_summary(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def print_summary(self):
        """Print formatted profiling summary."""
        summary = self.get_summary()
        
        print("=" * 60)
        print("TRAINING PROFILER SUMMARY")
        print("=" * 60)
        
        print(f"\n📊 Steps Profiled: {summary['total_steps_profiled']}")
        
        print("\n⏱️ Timing:")
        for k, v in summary.get('timing', {}).items():
            print(f"  {k}: {v}")
        
        print("\n🚀 Throughput:")
        for k, v in summary.get('throughput', {}).items():
            print(f"  {k}: {v}")
        
        print("\n💾 Memory:")
        for k, v in summary.get('memory', {}).items():
            print(f"  {k}: {v}")
        
        print("\n🖥️ GPU:")
        for k, v in summary.get('gpu', {}).items():
            print(f"  {k}: {v}")
        
        print("\n📈 Latest Step Breakdown:")
        for k, v in summary.get('breakdown_latest', {}).items():
            print(f"  {k}: {v}")
        
        print("=" * 60)


class MoEProfilerHooks:
    """
    PyTorch hooks for detailed MoE profiling.
    
    Attaches to model modules to track per-component timing.
    """
    
    def __init__(self, profiler: TrainingProfiler):
        self.profiler = profiler
        self._hooks = []
        self._timings = defaultdict(float)
    
    def attach(self, model):
        """Attach profiling hooks to model."""
        try:
            import torch
            
            def make_hook(name):
                start_time = [0.0]
                
                def forward_pre(module, input):
                    start_time[0] = time.perf_counter()
                
                def forward_post(module, input, output):
                    elapsed = time.perf_counter() - start_time[0]
                    self._timings[name] += elapsed
                
                return forward_pre, forward_post
            
            # Find and hook key modules
            for name, module in model.named_modules():
                if 'attention' in name.lower():
                    pre, post = make_hook('attention')
                    self._hooks.append(module.register_forward_pre_hook(pre))
                    self._hooks.append(module.register_forward_hook(post))
                
                elif 'router' in name.lower():
                    pre, post = make_hook('router')
                    self._hooks.append(module.register_forward_pre_hook(pre))
                    self._hooks.append(module.register_forward_hook(post))
                
                elif 'expert' in name.lower() and 'moe' in name.lower():
                    pre, post = make_hook('expert_compute')
                    self._hooks.append(module.register_forward_pre_hook(pre))
                    self._hooks.append(module.register_forward_hook(post))
        except:
            pass
    
    def detach(self):
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def get_timings(self) -> Dict[str, float]:
        """Get accumulated timings and reset."""
        result = dict(self._timings)
        self._timings.clear()
        return result


if __name__ == "__main__":
    # Demo usage
    config = ProfilerConfig(
        profile_every_n_steps=1,
        warmup_steps=0,
    )
    profiler = TrainingProfiler(config)
    
    # Simulate training loop
    for step in range(20):
        with profiler.profile_step(batch_size=8, seq_length=2048):
            with profiler.time_region('forward'):
                time.sleep(0.01)  # Simulate forward
            
            with profiler.time_region('attention'):
                time.sleep(0.005)
            
            with profiler.time_region('router'):
                time.sleep(0.001)
            
            with profiler.time_region('expert_compute'):
                time.sleep(0.003)
            
            with profiler.time_region('backward'):
                time.sleep(0.02)  # Simulate backward
        
        profiler.log_metrics()
    
    profiler.print_summary()
