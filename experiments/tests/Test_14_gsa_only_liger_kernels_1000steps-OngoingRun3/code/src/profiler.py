"""
Step-level profiler for Recurrence Model 1B — Test 14 OngoingRun3.

Captures CUDA-accurate timing at three granularity levels:
  1. Step phases  : forward, backward, optimizer_step, allreduce, dataloader
  2. Layer level  : per LightningDecoderLayer + MTP block (forward + backward)
  3. Kernel level : indexer, sparse_attn, sinkhorn, deltanet_fla, mlp, rmsnorm, rope

Usage (activated from train.py):
    from .profiler import StepProfiler
    profiler = StepProfiler(rank=local_rank, profile_steps={10, 11, 12})
    profiler.activate()   # sets global so model hooks auto-register
    ...
    profiler.start_step(global_step)
    # --- train step ---
    profiler.end_step(tokens_this_step)
    ...
    profiler.deactivate()
    profiler.write_report("results/run/profile_report.txt")
    profiler.write_jsonl("results/run/profile.jsonl")

Inside the model, use the module-level helpers:
    from src.profiler import time_region
    with time_region("gsa.indexer"):
        var_t, k_t, top_indices = fused_indexer_topk(...)
    with time_region("gsa.sparse_attn"):
        o_sparse = triton_sparse_attention(...)

time_region() is a no-op when no profiler is active, so it adds zero overhead
during normal training.
"""

from __future__ import annotations

import json
import os
import threading
import time as _time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch

# ─── Global profiler singleton ──────────────────────────────────────────────
_ACTIVE_PROFILER: Optional["StepProfiler"] = None
_PROFILER_LOCK = threading.Lock()


def get_active_profiler() -> Optional["StepProfiler"]:
    return _ACTIVE_PROFILER


# ─── Low-level CUDA event timer ─────────────────────────────────────────────

class _CUDARegion:
    """Records one named region using a CUDA event pair."""

    __slots__ = ("name", "start_evt", "end_evt", "_committed", "_ms")

    def __init__(self, name: str):
        self.name = name
        self._committed = False
        self._ms: Optional[float] = None
        if torch.cuda.is_available():
            self.start_evt = torch.cuda.Event(enable_timing=True)
            self.end_evt = torch.cuda.Event(enable_timing=True)
        else:
            self.start_evt = None
            self.end_evt = None

    def record_start(self):
        if self.start_evt is not None:
            self.start_evt.record()
        else:
            self._wall_start = _time.perf_counter()

    def record_end(self):
        if self.end_evt is not None:
            self.end_evt.record()
        else:
            self._wall_end = _time.perf_counter()

    def elapsed_ms(self) -> float:
        """Synchronize and return elapsed milliseconds."""
        if self._ms is not None:
            return self._ms
        if self.start_evt is not None and self.end_evt is not None:
            torch.cuda.synchronize()
            self._ms = self.start_evt.elapsed_time(self.end_evt)
        else:
            self._ms = (self._wall_end - self._wall_start) * 1000.0
        return self._ms


# ─── Context-manager helper (used by model code) ────────────────────────────

@contextmanager
def time_region(name: str):
    """
    Thin, zero-overhead context manager.

    When no profiler is active this is a strict no-op (one global read + branch).
    When profiling is active, records a CUDA event pair for `name`.
    """
    profiler = _ACTIVE_PROFILER
    if profiler is None or not profiler._recording:
        yield
        return

    region = _CUDARegion(name)
    region.record_start()
    try:
        yield
    finally:
        region.record_end()
        profiler._record_region(region)


# ─── Module hook helpers ─────────────────────────────────────────────────────

def _make_forward_hooks(profiler: "StepProfiler", label: str):
    """Return (pre_hook, post_hook) that time the forward pass of a module."""
    region_key = f"{label}.fwd"

    def pre_hook(module, args):
        if not profiler._recording:
            return
        r = _CUDARegion(region_key)
        r.record_start()
        module._profiler_fwd_region = r

    def post_hook(module, args, output):
        if not profiler._recording:
            return
        r = getattr(module, "_profiler_fwd_region", None)
        if r is None:
            return
        r.record_end()
        profiler._record_region(r)
        del module._profiler_fwd_region

    return pre_hook, post_hook


def _make_backward_hook(profiler: "StepProfiler", label: str):
    """Return a full_backward_hook that times the backward pass of a module."""
    region_key = f"{label}.bwd"

    def bwd_hook(module, grad_input, grad_output):
        if not profiler._recording:
            return
        r = _CUDARegion(region_key)
        r.record_start()
        # Backward hooks fire after the backward, so end immediately
        r.record_end()
        profiler._record_region(r)

    return bwd_hook


# ─── Per-step data ───────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    step: int
    tokens: int = 0
    regions: Dict[str, float] = field(default_factory=dict)  # name → ms

    def add(self, name: str, ms: float):
        if name in self.regions:
            self.regions[name] += ms
        else:
            self.regions[name] = ms


# ─── Main profiler class ─────────────────────────────────────────────────────

class StepProfiler:
    """
    Collects per-step kernel and layer timing for the 1B recurrence model.

    Thread-safe for a single training process; each rank should create its
    own instance and only rank-0 writes reports.

    Args:
        rank           : This process's local rank (only rank-0 writes to disk).
        profile_steps  : Set of global step numbers to profile.
                         Pass an empty set to disable.
        output_dir     : Where to write profile.jsonl and profile_report.txt.
    """

    def __init__(
        self,
        rank: int = 0,
        profile_steps: Optional[Set[int]] = None,
        output_dir: str = "results/run",
    ):
        self.rank = rank
        self.profile_steps: Set[int] = profile_steps or set()
        self.output_dir = output_dir
        self._recording = False
        self._current: Optional[StepRecord] = None
        self._history: List[StepRecord] = []
        self._hook_handles: List = []

    # ── activation / deactivation ────────────────────────────────────────────

    def activate(self):
        """Register this profiler as the global singleton."""
        global _ACTIVE_PROFILER
        with _PROFILER_LOCK:
            _ACTIVE_PROFILER = self

    def deactivate(self):
        """Remove this profiler from the global singleton."""
        global _ACTIVE_PROFILER
        with _PROFILER_LOCK:
            if _ACTIVE_PROFILER is self:
                _ACTIVE_PROFILER = None
        self._remove_hooks()

    # ── module hook registration ─────────────────────────────────────────────

    def register_model(self, model):
        """
        Attach forward (and where possible backward) hooks to every
        named sub-module of interest.

        Call this once after DeepSpeed engine wraps the model:
            profiler.register_model(engine.module)
        """
        self._remove_hooks()  # clear stale handles

        # Top-level structure of Model1B:
        #   model.layers[i]                  → LightningDecoderLayer
        #   model.layers[i].attn_block       → MHCSublayer (attention path)
        #   model.layers[i].attn_block.sublayer → GatedDeltaNet | GatedSparseAttention
        #   model.layers[i].attn_block.coeffs   → MHCCoeffs (Sinkhorn)
        #   model.layers[i].mlp_block        → MHCSublayer (MLP path)
        #   model.layers[i].mlp_block.sublayer  → LightningMLP
        #   model.mtp_block                  → MTPTransformerBlock
        #   model.embed_norm, model.norm     → RMSNorm
        #   model.pf_to_model                → Linear (Kronecker projection)

        for i, layer in enumerate(model.layers):
            layer_label = f"layer{i}"
            attn_mod = layer.attn_block.sublayer
            layer_type = getattr(layer, "layer_type", "unknown")

            # Full decoder layer
            self._attach_fwd_hooks(layer, f"{layer_label}")

            # Attention sub-module (GatedDeltaNet or GatedSparseAttention)
            kernel_tag = "deltanet" if layer_type == "deltanet" else "gsa"
            self._attach_fwd_hooks(attn_mod, f"{layer_label}.{kernel_tag}")

            # MHCCoeffs (Sinkhorn routing) — lives inside both attn and mlp blocks
            self._attach_fwd_hooks(layer.attn_block.coeffs, f"{layer_label}.sinkhorn_attn")
            self._attach_fwd_hooks(layer.mlp_block.coeffs, f"{layer_label}.sinkhorn_mlp")

            # MLP sublayer
            mlp_mod = layer.mlp_block.sublayer
            self._attach_fwd_hooks(mlp_mod, f"{layer_label}.mlp")

        # MTP block
        if hasattr(model, "mtp_block") and model.mtp_block is not None:
            self._attach_fwd_hooks(model.mtp_block, "mtp_block")
            self._attach_fwd_hooks(model.mtp_block.attn_block.sublayer, "mtp_block.gsa")
            self._attach_fwd_hooks(model.mtp_block.mlp_block.sublayer, "mtp_block.mlp")

        # Embedding / output projections
        if hasattr(model, "pf_to_model") and model.pf_to_model is not None:
            self._attach_fwd_hooks(model.pf_to_model, "kronecker_proj")
        if hasattr(model, "embed_norm") and model.embed_norm is not None:
            self._attach_fwd_hooks(model.embed_norm, "embed_norm")
        self._attach_fwd_hooks(model.norm, "final_norm")

    def _attach_fwd_hooks(self, module, label: str):
        pre, post = _make_forward_hooks(self, label)
        h1 = module.register_forward_pre_hook(pre)
        h2 = module.register_forward_hook(post)
        self._hook_handles.extend([h1, h2])

    def _remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    # ── step boundary methods ─────────────────────────────────────────────────

    def start_step(self, global_step: int, tokens: int = 0):
        """Call just before the forward pass of a step."""
        if global_step not in self.profile_steps:
            self._recording = False
            return
        self._recording = True
        self._current = StepRecord(step=global_step, tokens=tokens)

    def end_step(self, tokens: int = 0):
        """Call after optimizer.step(). Finalizes the step record."""
        if not self._recording or self._current is None:
            return
        self._recording = False
        if tokens:
            self._current.tokens = tokens
        # Force CUDA sync so all event elapsed_time() calls are ready
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._history.append(self._current)
        self._current = None

    # ── explicit phase timers (called from train.py) ─────────────────────────

    @contextmanager
    def phase(self, name: str):
        """Time a coarse training phase (e.g. 'forward', 'backward', 'optim')."""
        if not self._recording:
            yield
            return
        r = _CUDARegion(name)
        r.record_start()
        try:
            yield
        finally:
            r.record_end()
            self._record_region(r)

    # ── internal accumulation ─────────────────────────────────────────────────

    def _record_region(self, region: _CUDARegion):
        if self._current is None:
            return
        try:
            ms = region.elapsed_ms()
            self._current.add(region.name, ms)
        except Exception:
            pass  # never crash training

    # ── reporting ─────────────────────────────────────────────────────────────

    def write_jsonl(self, path: Optional[str] = None):
        """Write one JSON line per profiled step to `path`."""
        if self.rank != 0 or not self._history:
            return
        if path is None:
            path = os.path.join(self.output_dir, "profile.jsonl")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for rec in self._history:
                row = {"step": rec.step, "tokens": rec.tokens, **rec.regions}
                f.write(json.dumps(row) + "\n")

    def write_report(self, path: Optional[str] = None):
        """Write a human-readable summary table to `path` and stdout."""
        if self.rank != 0 or not self._history:
            return
        if path is None:
            path = os.path.join(self.output_dir, "profile_report.txt")
        lines = self._build_report_lines()
        text = "\n".join(lines)
        print(text)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + "\n")

    def _build_report_lines(self) -> List[str]:
        if not self._history:
            return ["[profiler] No data collected."]

        # Average across all recorded steps
        from collections import defaultdict
        sums: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        total_tokens = 0
        for rec in self._history:
            total_tokens += rec.tokens
            for k, v in rec.regions.items():
                sums[k] += v
                counts[k] += 1

        n = len(self._history)
        avgs = {k: sums[k] / counts[k] for k in sums}
        avg_tokens = total_tokens / n

        lines = []
        lines.append("=" * 72)
        lines.append(f"  STEP PROFILER REPORT  ({n} step(s) averaged, {avg_tokens:.0f} tokens/step)")
        lines.append("=" * 72)

        # ── Phase summary ────────────────────────────────────────────────────
        phase_keys = [
            "dataloader",
            "forward",
            "gsa_leak_allreduce",
            "fused_ce",
            "fused_ce_mtp",
            "backward",
            "optim_step",
            "token_count_allreduce",
            "system_metrics",
            "log_write",
            "checkpoint_save",
        ]
        lines.append("\n── Step Phases ──────────────────────────────────────────────────")
        lines.append(f"  {'Region':<30}  {'ms':>8}  {'%step':>7}")
        lines.append(f"  {'-'*30}  {'-'*8}  {'-'*7}")
        step_total = avgs.get("step_total", sum(avgs.get(k, 0) for k in phase_keys) or 1)
        for k in phase_keys + ["step_total"]:
            if k in avgs:
                pct = 100.0 * avgs[k] / step_total
                lines.append(f"  {k:<30}  {avgs[k]:>8.1f}  {pct:>6.1f}%")

        # ── Per-layer summary ────────────────────────────────────────────────
        lines.append("\n── Per-Layer Forward (ms) ───────────────────────────────────────")
        lines.append(f"  {'Layer':<38}  {'fwd ms':>8}")
        lines.append(f"  {'-'*38}  {'-'*8}")
        layer_keys = sorted(k for k in avgs if k.startswith("layer") and k.endswith(".fwd"))
        for k in layer_keys:
            lines.append(f"  {k:<38}  {avgs[k]:>8.2f}")
        if "mtp_block.fwd" in avgs:
            lines.append(f"  {'mtp_block.fwd':<38}  {avgs['mtp_block.fwd']:>8.2f}")

        # ── Kernel-level breakdown ────────────────────────────────────────────
        # Aggregate kernel types across layers
        kernel_types = ["deltanet", "gsa", "sinkhorn_attn", "sinkhorn_mlp", "mlp",
                        "indexer", "sparse_attn", "sinkhorn_knopp", "rmsnorm", "rope",
                        "deltanet_fla"]
        kernel_totals: Dict[str, float] = {}
        for ktype in kernel_types:
            keys = [k for k in avgs if f".{ktype}.fwd" in k or k == f"{ktype}.fwd"]
            if keys:
                kernel_totals[ktype] = sum(avgs[k] for k in keys)
        # Also pick up per-call regions (time_region context managers)
        for k, v in avgs.items():
            if "." in k and not k.endswith(".fwd") and not k.endswith(".bwd"):
                short = k.split(".")[-1]
                if short not in kernel_totals:
                    kernel_totals[short] = 0.0
                kernel_totals[short] += v

        if kernel_totals:
            lines.append("\n── Kernel-Type Totals (all layers summed) ───────────────────────")
            lines.append(f"  {'Kernel':<30}  {'total ms':>10}")
            lines.append(f"  {'-'*30}  {'-'*10}")
            for ktype, total_ms in sorted(kernel_totals.items(), key=lambda x: -x[1]):
                lines.append(f"  {ktype:<30}  {total_ms:>10.2f}")

        # ── All raw regions (sorted by time) ─────────────────────────────────
        lines.append("\n── All Regions (sorted by avg ms) ───────────────────────────────")
        lines.append(f"  {'Region':<44}  {'avg ms':>8}  {'calls':>6}")
        lines.append(f"  {'-'*44}  {'-'*8}  {'-'*6}")
        for k, v in sorted(avgs.items(), key=lambda x: -x[1])[:60]:
            lines.append(f"  {k:<44}  {v:>8.2f}  {counts[k]:>6}")

        if avg_tokens > 0:
            step_ms = avgs.get("step_total", 1)
            tok_per_sec = avg_tokens / (step_ms / 1000.0)
            lines.append(f"\n  Estimated throughput: {tok_per_sec:,.0f} tok/sec")

        lines.append("=" * 72)
        return lines


# ============================================================================
# PipelineProfiler — always-on wall-clock timer for main.py pipeline stages
# ============================================================================

class PipelineProfiler:
    """
    Always-on wall-clock profiler for the full training pipeline in main.py.

    Unlike StepProfiler, this requires NO configuration and NO profile_steps.
    It runs on every execution, capturing seconds/minutes-scale timings for
    every pipeline stage from startup through final checkpoint save.

    The kernel-level StepProfiler is still gated by profile_steps. This profiler
    gives you the outer picture: where does total wall time go across setup,
    training, evaluation, and teardown?

    Usage (in main.py):
        from src.profiler import PipelineProfiler
        pipe = PipelineProfiler(rank=local_rank, output_dir="results/run")

        with pipe.stage("data_load"):
            train_loader, ... = get_dataloaders(...)

        with pipe.stage("epoch_0_train"):
            avg_loss, global_step = train_epoch(...)

        pipe.write_report()   # always writes, even with no profile_steps
        pipe.write_jsonl()
    """

    def __init__(self, rank: int = 0, output_dir: str = "results/run"):
        self.rank = rank
        self.output_dir = output_dir
        # Ordered list of (name, start_sec, end_sec, extra_meta)
        self._records: List[Tuple[str, float, float, dict]] = []
        self._open: Dict[str, float] = {}   # name → start time for open stages
        self._pipeline_start = _time.perf_counter()

    @contextmanager
    def stage(self, name: str, **meta):
        """
        Context manager that times a named pipeline stage using wall clock.

        Args:
            name : Unique label for this stage (e.g. 'data_load', 'epoch_0_train').
            **meta: Optional metadata to store in the JSONL record
                    (e.g. epoch=0, steps=1000, avg_loss=2.34).
        """
        t0 = _time.perf_counter()
        try:
            yield
        finally:
            t1 = _time.perf_counter()
            self._records.append((name, t0, t1, meta))

    def elapsed_sec(self, name: str) -> Optional[float]:
        """Return elapsed seconds for a named stage, or None if not found."""
        for n, t0, t1, _ in self._records:
            if n == name:
                return t1 - t0
        return None

    # ── reporting ─────────────────────────────────────────────────────────────

    def write_report(self, path: Optional[str] = None):
        """Write the pipeline timing report to disk and stdout (rank-0 only)."""
        if self.rank != 0:
            return
        if path is None:
            path = os.path.join(self.output_dir, "pipeline_report.txt")
        lines = self._build_report_lines()
        text = "\n".join(lines)
        print(text)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + "\n")

    def write_jsonl(self, path: Optional[str] = None):
        """Write one JSON line per pipeline stage (rank-0 only)."""
        if self.rank != 0 or not self._records:
            return
        if path is None:
            path = os.path.join(self.output_dir, "pipeline.jsonl")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for name, t0, t1, meta in self._records:
                row = {
                    "stage": name,
                    "start_sec": round(t0 - self._pipeline_start, 3),
                    "end_sec": round(t1 - self._pipeline_start, 3),
                    "elapsed_sec": round(t1 - t0, 3),
                    "elapsed_min": round((t1 - t0) / 60.0, 3),
                    **meta,
                }
                f.write(json.dumps(row) + "\n")

    def _build_report_lines(self) -> List[str]:
        if not self._records:
            return ["[pipeline profiler] No stages recorded."]

        total_elapsed = _time.perf_counter() - self._pipeline_start
        lines = []
        lines.append("=" * 76)
        lines.append(f"  PIPELINE PROFILER REPORT  (total wall time: {_fmt_duration(total_elapsed)})")
        lines.append("=" * 76)
        lines.append(f"  {'Stage':<35}  {'Time':>10}  {'%total':>7}  {'Cumulative':>11}")
        lines.append(f"  {'-'*35}  {'-'*10}  {'-'*7}  {'-'*11}")

        cumulative = 0.0
        for name, t0, t1, meta in self._records:
            elapsed = t1 - t0
            cumulative += elapsed
            pct = 100.0 * elapsed / total_elapsed if total_elapsed > 0 else 0.0
            cum_str = _fmt_duration(cumulative)
            extra = ""
            if meta:
                extra_parts = []
                for k, v in meta.items():
                    if isinstance(v, float):
                        extra_parts.append(f"{k}={v:.4f}")
                    else:
                        extra_parts.append(f"{k}={v}")
                extra = "  [" + ", ".join(extra_parts) + "]"
            lines.append(
                f"  {name:<35}  {_fmt_duration(elapsed):>10}  {pct:>6.1f}%  {cum_str:>11}{extra}"
            )

        # Unaccounted time (Python overhead, gaps between stages)
        accounted = sum(t1 - t0 for _, t0, t1, _ in self._records)
        unaccounted = total_elapsed - accounted
        if unaccounted > 0.5:
            pct = 100.0 * unaccounted / total_elapsed
            lines.append(f"  {'(unaccounted / gaps)':<35}  {_fmt_duration(unaccounted):>10}  {pct:>6.1f}%")

        lines.append(f"  {'-'*35}  {'-'*10}  {'-'*7}  {'-'*11}")
        lines.append(f"  {'TOTAL':<35}  {_fmt_duration(total_elapsed):>10}  {'100.0%':>7}")
        lines.append("=" * 76)
        return lines


def _fmt_duration(seconds: float) -> str:
    """Format seconds as human-readable string: ms / s / min / h."""
    if seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    if seconds < 3600.0:
        m = int(seconds // 60)
        s = seconds - m * 60
        return f"{m}m {s:.1f}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}h {m}m {s:.0f}s"
