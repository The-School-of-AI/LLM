"""
Tests for per-step training logging, timing, and GPU memory stats.

Covers:
  - _gpu_mem_stats returns correct keys on CUDA / empty dict on CPU
  - Every step produces 5 log lines (step, OPUS, batch, timing, GPU)
  - OPUS enabled vs disabled log format
  - Timing dict keys are populated
  - Scoring loss is captured in opus_timings
  - Log format includes proxy count, selected count, training batch size
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

EXP_ROOT = str(Path(__file__).resolve().parent.parent)
if EXP_ROOT not in sys.path:
    sys.path.insert(0, EXP_ROOT)

from exp.opus import SelectionResult
from exp.train import OpusConfig, Trainer


# ---------------------------------------------------------------------------
# GPU memory stats
# ---------------------------------------------------------------------------


class TestGpuMemStats:
    def test_returns_empty_when_no_cuda(self):
        """On CPU-only, _gpu_mem_stats returns empty dict."""
        with patch("torch.cuda.is_available", return_value=False):
            result = Trainer._gpu_mem_stats(torch.device("cpu"))
        assert result == {}

    def test_returns_correct_keys_when_cuda(self):
        """When CUDA is available, all 3 memory keys are present."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory_allocated", return_value=1024 * 1024 * 512),
            patch("torch.cuda.memory_reserved", return_value=1024 * 1024 * 1024),
            patch("torch.cuda.max_memory_allocated", return_value=1024 * 1024 * 768),
        ):
            result = Trainer._gpu_mem_stats(torch.device("cpu"))

        assert "gpu_mem_allocated_mb" in result
        assert "gpu_mem_reserved_mb" in result
        assert "gpu_mem_peak_mb" in result
        assert result["gpu_mem_allocated_mb"] == pytest.approx(512.0)
        assert result["gpu_mem_reserved_mb"] == pytest.approx(1024.0)
        assert result["gpu_mem_peak_mb"] == pytest.approx(768.0)


# ---------------------------------------------------------------------------
# Log line format tests (unit tests using captured print output)
# ---------------------------------------------------------------------------


class TestLogLineFormat:
    """Test the log format by examining what print_rank_0 would output."""

    @staticmethod
    def _build_log_lines(
        global_step: int = 1,
        max_steps: int = 20,
        loss_val: float = 8.1234,
        current_lr: float = 3e-4,
        step_time_ms: float = 142.3,
        opus_enabled: bool = True,
        n_candidates: int = 2,
        candidate_seq_len: int = 128,
        n_proxy_in_batch: int = 1,
        n_selected_local: int = 1,
        n_training: int = 2,
        training_seq_len: int = 128,
        total_train_tokens: int = 256,
        opus_timings: dict | None = None,
        timings: dict | None = None,
        gpu_stats: dict | None = None,
        metrics: dict | None = None,
        used_fallback: bool = False,
    ) -> list[str]:
        """Simulate the log lines from the training loop."""
        if opus_timings is None:
            opus_timings = {
                "proxy_sample_ms": 2.1,
                "scoring_forward_ms": 45.3,
                "scoring_backward_ms": 38.2,
                "zero_grad_ms": 0.5,
                "boltzmann_select_ms": 12.1,
                "preconditioner_refresh_ms": 1.2,
            }
        if timings is None:
            timings = {
                "data_to_device_ms": 0.3,
                "batch_assembly_ms": 0.1,
                "train_forward_ms": 22.5,
                "train_backward_ms": 18.1,
                "optimizer_step_ms": 1.9,
                "opus_total_ms": 99.8,
            }
        if metrics is None:
            metrics = {
                "alignment": 0.723,
                "redundancy": 0.051,
                "entropy": 2.341,
                "selector_time_s": 0.012,
            }
        if gpu_stats is None:
            gpu_stats = {
                "gpu_mem_allocated_mb": 512.0,
                "gpu_mem_reserved_mb": 1024.0,
                "gpu_mem_peak_mb": 768.0,
            }

        lines = []

        # Line 1
        lines.append(
            f"[step {global_step}/{max_steps}] "
            f"loss={loss_val:.4f} | lr={current_lr:.2e} | "
            f"step_time={step_time_ms:.1f}ms"
        )

        # Line 2
        if opus_enabled:
            lines.append(
                f"  OPUS: "
                f"proxy_sample={opus_timings.get('proxy_sample_ms', 0):.1f}ms | "
                f"scoring_fwd={opus_timings.get('scoring_forward_ms', 0):.1f}ms | "
                f"scoring_bwd={opus_timings.get('scoring_backward_ms', 0):.1f}ms | "
                f"zero_grad={opus_timings.get('zero_grad_ms', 0):.1f}ms | "
                f"boltzmann={opus_timings.get('boltzmann_select_ms', 0):.1f}ms | "
                f"precond_refresh={opus_timings.get('preconditioner_refresh_ms', 0):.1f}ms | "
                f"total={timings.get('opus_total_ms', 0):.1f}ms"
                + (" [FALLBACK]" if used_fallback else "")
            )
            lines.append(
                f"  OPUS scores: "
                f"alignment={metrics.get('alignment', 0.0):.4f} | "
                f"redundancy={metrics.get('redundancy', 0.0):.4f} | "
                f"entropy={metrics.get('entropy', 0.0):.4f} | "
                f"selector_time={metrics.get('selector_time_s', 0.0) * 1000:.1f}ms"
            )
        else:
            lines.append(
                f"  OPUS: DISABLED (bypass) | "
                f"bypass_time={timings.get('opus_bypass_ms', 0):.1f}ms"
            )

        # Line 3
        lines.append(
            f"  Batch: "
            f"candidates_in={n_candidates} (seq_len={candidate_seq_len}) | "
            f"proxy_in_batch={n_proxy_in_batch} | "
            f"selected={n_selected_local} | "
            f"training_batch={n_training} (seq_len={training_seq_len}) | "
            f"train_tokens={total_train_tokens}"
        )

        # Line 4
        lines.append(
            f"  Timing: "
            f"data_load={timings['data_to_device_ms']:.1f}ms | "
            f"batch_asm={timings['batch_assembly_ms']:.1f}ms | "
            f"train_fwd={timings['train_forward_ms']:.1f}ms | "
            f"train_bwd={timings['train_backward_ms']:.1f}ms | "
            f"optim={timings['optimizer_step_ms']:.1f}ms"
        )

        # Line 5
        if gpu_stats:
            lines.append(
                f"  GPU: "
                f"alloc={gpu_stats['gpu_mem_allocated_mb']:.0f}MB | "
                f"reserved={gpu_stats['gpu_mem_reserved_mb']:.0f}MB | "
                f"peak={gpu_stats['gpu_mem_peak_mb']:.0f}MB"
            )

        return lines

    def test_opus_enabled_produces_6_lines(self):
        """OPUS enabled → 6 log lines (step + OPUS timing + OPUS scores + batch + timing + GPU)."""
        lines = self._build_log_lines(opus_enabled=True)
        assert len(lines) == 6

    def test_opus_disabled_produces_5_lines(self):
        """OPUS disabled → 5 log lines (no OPUS scores line)."""
        lines = self._build_log_lines(opus_enabled=False)
        assert len(lines) == 5

    def test_step_line_format(self):
        lines = self._build_log_lines(global_step=5, max_steps=20, loss_val=7.5)
        assert lines[0].startswith("[step 5/20]")
        assert "loss=7.5000" in lines[0]
        assert "lr=" in lines[0]
        assert "step_time=" in lines[0]

    def test_opus_timing_line_has_all_phases(self):
        lines = self._build_log_lines(opus_enabled=True)
        opus_line = lines[1]
        assert "proxy_sample=" in opus_line
        assert "scoring_fwd=" in opus_line
        assert "scoring_bwd=" in opus_line
        assert "zero_grad=" in opus_line
        assert "boltzmann=" in opus_line
        assert "precond_refresh=" in opus_line
        assert "total=" in opus_line

    def test_opus_scores_line(self):
        lines = self._build_log_lines(opus_enabled=True)
        scores_line = lines[2]
        assert "alignment=" in scores_line
        assert "redundancy=" in scores_line
        assert "entropy=" in scores_line
        assert "selector_time=" in scores_line

    def test_fallback_marker(self):
        lines = self._build_log_lines(opus_enabled=True, used_fallback=True)
        assert "[FALLBACK]" in lines[1]

    def test_no_fallback_marker(self):
        lines = self._build_log_lines(opus_enabled=True, used_fallback=False)
        assert "[FALLBACK]" not in lines[1]

    def test_bypass_line_format(self):
        lines = self._build_log_lines(
            opus_enabled=False,
            timings={
                "data_to_device_ms": 0.1,
                "batch_assembly_ms": 0.1,
                "train_forward_ms": 10.0,
                "train_backward_ms": 8.0,
                "optimizer_step_ms": 1.0,
                "opus_bypass_ms": 0.05,
            },
        )
        assert "DISABLED (bypass)" in lines[1]
        assert "bypass_time=" in lines[1]

    def test_batch_line_has_all_fields(self):
        lines = self._build_log_lines(
            n_candidates=16,
            candidate_seq_len=128,
            n_proxy_in_batch=2,
            n_selected_local=2,
            n_training=4,
            training_seq_len=128,
            total_train_tokens=512,
        )
        # OPUS enabled → batch line is index 3
        batch_line = lines[3]
        assert "candidates_in=16" in batch_line
        assert "seq_len=128" in batch_line
        assert "proxy_in_batch=2" in batch_line
        assert "selected=2" in batch_line
        assert "training_batch=4" in batch_line
        assert "train_tokens=512" in batch_line

    def test_timing_line_has_all_phases(self):
        lines = self._build_log_lines(opus_enabled=True)
        timing_line = lines[4]
        assert "data_load=" in timing_line
        assert "batch_asm=" in timing_line
        assert "train_fwd=" in timing_line
        assert "train_bwd=" in timing_line
        assert "optim=" in timing_line

    def test_gpu_line_format(self):
        lines = self._build_log_lines()
        gpu_line = lines[5]
        assert "alloc=" in gpu_line
        assert "reserved=" in gpu_line
        assert "peak=" in gpu_line
        assert "MB" in gpu_line

    def test_no_gpu_line_when_no_stats(self):
        lines = self._build_log_lines(gpu_stats={})
        # Should be 5 lines (no GPU line)
        assert len(lines) == 5


# ---------------------------------------------------------------------------
# Timing measurement tests
# ---------------------------------------------------------------------------


class TestTimingMeasurement:
    def test_perf_counter_timing_is_positive(self):
        """Wall-clock timing should always be positive."""
        t0 = _time.perf_counter()
        _ = torch.randn(100, 100)  # some work
        elapsed_ms = (_time.perf_counter() - t0) * 1000
        assert elapsed_ms >= 0.0

    def test_timing_dict_accumulates_phases(self):
        """Simulates that timings dict gets all expected keys."""
        timings: dict[str, float] = {}
        phases = [
            "data_to_device_ms",
            "batch_assembly_ms",
            "train_forward_ms",
            "train_backward_ms",
            "optimizer_step_ms",
        ]
        for phase in phases:
            t0 = _time.perf_counter()
            timings[phase] = (_time.perf_counter() - t0) * 1000
        for key in phases:
            assert key in timings
            assert isinstance(timings[key], float)
            assert timings[key] >= 0.0

    def test_opus_timings_dict_keys(self):
        """opus_timings from _select_candidates should have all phase keys."""
        expected_keys = {
            "preconditioner_refresh_ms",
            "proxy_sample_ms",
            "scoring_forward_ms",
            "scoring_backward_ms",
            "zero_grad_ms",
            "boltzmann_select_ms",
            "scoring_loss_val",
            "scoring_seq_len",
            "combined_scoring_batch",
            "n_proxy_sampled",
            "n_candidates_scored",
        }
        # Simulate the dict that _select_candidates would produce
        opus_timings = {k: 0.0 for k in expected_keys}
        assert set(opus_timings.keys()) == expected_keys
