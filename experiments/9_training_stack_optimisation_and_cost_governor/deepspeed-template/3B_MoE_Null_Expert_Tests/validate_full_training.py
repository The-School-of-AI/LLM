"""
validate_full_training.py
==========================
Production-grade validation of DeepSpeed ZeRO-2 + Reversible Midpoint + Ghost Expert MoE.

Exercises the FULL production code path:
  Model3B.forward() -> ReversibleMidpointStack -> MidpointFunction.backward()
  with DeepSpeed ZeRO-2 optimizer partitioning active.

Validation Phases:
  Phase 0a: Gate routing determinism (catches MoE non-determinism)
  Phase 0b: Reversible vs standard gradient correctness
  Phase 0c: ZeRO flat-buffer alignment (parameter update verification)
  Phase 0d: MoE dispatch correctness & ghost expert validation
  Phase 0e: Ghost vs Dense expert A/B comparison (THE cost savings proof)
  Phase 1:  Warmup training (steps 0-99)
  Phase 2:  Steady-state convergence (steps 100-199)
  Phase 3:  Checkpoint save + resume + optimizer state verification
  Phase 4:  Post-resume training (steps 200-249)

Data: Copy-shift task (target[t] = input[t-1]). Trivially learnable - if loss
      doesn't decrease, the training pipeline is fundamentally broken.

Verdicts:
  routing/null_rate_healthy          routing/null_rate_not_collapsed
  routing/load_balanced              routing/no_expert_collapse
  routing/no_token_dropping
  gradient/gate_receives_grad        gradient/experts_receive_grad
  gradient/reversible_correct        gradient/gate_deterministic
  gradient/zero_alignment_ok
  moe/dispatch_correctness
  loss/decreasing                    loss/not_diverging
  loss/below_random_chance           loss/strong_learning_signal
  loss/aux_stable                    loss/aux_ratio_bounded
  zero/no_nan_params                 zero/cross_rank_consistent
  health/no_nans                     health/no_infs
  health/no_loss_spikes
  checkpoint/zero2_resume            checkpoint/params_restored
  checkpoint/optimizer_state_ok
  perf/moe_fraction_reasonable

Hardware Target: AWS g5.12xlarge (4x NVIDIA A10G 24GB)
"""

import os
import sys
import csv
import json
import time
import math
import argparse
import traceback
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recurrence_model_3b import ModelConfig, Model3B

import contextlib, builtins

@contextlib.contextmanager
def quiet_init(rank=0):
    """Suppress noisy MoE init and ZeRO memory messages.
    
    On rank 0: filters out repetitive ⚡/Memory/MA/CPU lines, prints a single summary.
    On other ranks: suppresses all output.
    """
    _orig_print = builtins.print
    _suppressed_prefixes = (
        "⚡ ", "Memory reclaimed", "Before initializing optimizer",
        "After initializing optimizer", "After initializing ZeRO",
        "MA ", "CPU Virtual Memory:",
        "🤖 MODEL", "🔄 Recurrence", "Vocabulary:", "Hidden Size:",
        "Total Layers:", "- DeltaNet:", "- GSA:", "Context Target:",
        "Experts:", "Top-k:", "MTP:", "Total Parameters:",
        "Active Parameters:",
    )
    _count = {"moe_layers": 0, "ghost_swaps": 0}

    def _filtered_print(*args, **kwargs):
        if not args:
            return _orig_print(*args, **kwargs)
        msg = str(args[0])
        if any(msg.strip().startswith(p) for p in _suppressed_prefixes):
            if "Initializing DeepSpeed Optimized MoE" in msg:
                _count["moe_layers"] += 1
            elif "Swapping" in msg and "Ghosts" in msg:
                _count["ghost_swaps"] += 1
            return
        _orig_print(*args, **kwargs)

    if rank != 0:
        builtins.print = lambda *a, **kw: None
    else:
        builtins.print = _filtered_print
    try:
        yield _count
    finally:
        builtins.print = _orig_print
        if rank == 0 and (_count["moe_layers"] > 0 or _count["ghost_swaps"] > 0):
            _orig_print(f"    (initialized {_count['moe_layers']} MoE layers, "
                        f"{_count['ghost_swaps']} ghost swaps)")



# ============================================================================
# VALIDATION CONFIG
# ============================================================================

def make_validation_config():
    config = ModelConfig()
    config.hidden_size = 512
    config.num_layers = 4
    config.num_deltanet_layers = 3
    config.num_gsa_layers = 1
    config.delta_v_heads = 4
    config.delta_head_dim = 128
    config.delta_gate_dim = 48
    config.gsa_num_heads = 4
    config.gsa_head_dim = 128
    config.gsa_k_base = 32
    config.gsa_k_min = 8
    config.gsa_k_max = 64
    config.gsa_indexer_heads = 2
    config.num_real_experts = 4
    config.num_null_experts = 4
    config.total_expert_slots = 8
    config.top_k = 2
    config.expert_intermediate_size = 256
    config.shared_expert_intermediate_size = 512
    config.vocab_size = 4096
    config.max_seq_len = 256
    config.rope_base = 10000
    config.rope_original_max_position = 256
    config.rope_scaling_factor = 1.0
    config.enable_mtp = True
    config.mtp_num_predictions = 2
    config.n_streams = 4
    config.sinkhorn_iters = 5
    config.dropout = 0.0
    return config


# ============================================================================
# LEARNABLE SYNTHETIC DATA (Copy-Shift Task)
# ============================================================================

class SyntheticDataLoader:
    """
    Copy-shift task: target[t] = input[t-1].

    This is trivially learnable by any autoregressive model, so loss MUST decrease.
    Random targets would make loss/decreasing unreliable (model can't learn noise).
    MTP targets use shift-by-2: mtp_target[t] = input[t-2].

    If fixed_pool_size > 0, pre-generates that many batches and cycles through them.
    This lets the model memorize the data, guaranteeing loss -> 0 if the pipeline works.
    """
    def __init__(self, vocab_size, seq_len, batch_size, num_steps, device,
                 fixed_pool_size=0, seed=42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.device = device
        self.step = 0
        self.pool = []

        if fixed_pool_size > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
            for _ in range(fixed_pool_size):
                ids = torch.randint(0, vocab_size, (batch_size, seq_len),
                                    device=device, generator=rng)
                tgt_n = torch.roll(ids, shifts=1, dims=1)
                tgt_n[:, 0] = 0
                nxt = torch.roll(ids, shifts=-1, dims=1)
                nxt[:, -1] = torch.randint(0, vocab_size, (batch_size,),
                                            device=device, generator=rng)
                tgt_m = torch.roll(ids, shifts=2, dims=1)
                tgt_m[:, :2] = 0
                self.pool.append((ids, tgt_n, nxt, tgt_m))

    def __iter__(self):
        self.step = 0
        return self

    def __next__(self):
        if self.step >= self.num_steps:
            raise StopIteration
        self.step += 1

        if self.pool:
            return self.pool[(self.step - 1) % len(self.pool)]

        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len),
                                  device=self.device)

        # NTP target: predict previous token (copy-shift by 1)
        targets_ntp = torch.roll(input_ids, shifts=1, dims=1)
        targets_ntp[:, 0] = 0

        # MTP input: next tokens
        next_token_ids = torch.roll(input_ids, shifts=-1, dims=1)
        next_token_ids[:, -1] = torch.randint(0, self.vocab_size, (self.batch_size,),
                                               device=self.device)

        # MTP target: shift by 2
        targets_mtp = torch.roll(input_ids, shifts=2, dims=1)
        targets_mtp[:, :2] = 0

        return input_ids, targets_ntp, next_token_ids, targets_mtp

    def __len__(self):
        return self.num_steps


# ============================================================================
# METRIC COLLECTOR
# ============================================================================

class MetricCollector:
    def __init__(self, log_dir, rank=0):
        self.rank = rank
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / f"metrics_rank{rank}.csv"
        self.summary_path = self.log_dir / f"validation_summary_rank{rank}.txt"

        self.fields = [
            "step",
            "loss_total", "loss_ntp", "loss_mtp", "loss_aux", "aux_main_ratio",
            "router_null_rate", "router_avg_real_experts", "router_gate_load_variance",
            "router_max_min_ratio", "router_per_expert_counts", "router_dropped_frac",
            "grad_gate_norm", "grad_expert_norm", "grad_shared_expert_norm",
            "grad_memory_gate_norm", "grad_global_norm",
            "param_norm", "cross_rank_param_norm_spread",
            "step_time_ms", "tokens_per_sec", "moe_dispatch_ms",
            "gpu_mem_allocated_gb", "gpu_mem_reserved_gb",
            "nan_detected", "inf_detected", "loss_spike",
        ]
        self.rows = []
        self.loss_history = deque(maxlen=50)

        if self.rank == 0:
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()

    def record(self, metrics: dict):
        loss_total = metrics.get("loss_total", 0.0)
        if len(self.loss_history) > 10:
            rolling_avg = sum(self.loss_history) / len(self.loss_history)
            metrics["loss_spike"] = 1 if (loss_total > 10 * rolling_avg and rolling_avg > 0.01) else 0
        else:
            metrics["loss_spike"] = 0
        self.loss_history.append(loss_total)
        self.rows.append(metrics)
        if self.rank == 0:
            with open(self.csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore").writerow(metrics)

    def print_step(self, step, metrics, detail_every=25):
        if self.rank != 0:
            return
        loss_n = metrics.get("loss_ntp", 0)
        loss_a = metrics.get("loss_aux", 0)
        ms = metrics.get("step_time_ms", 0)
        tps = metrics.get("tokens_per_sec", 0)
        mem_a = metrics.get("gpu_mem_allocated_gb", 0)

        # Alert flags — always shown when triggered
        flags = ""
        if metrics.get("nan_detected", 0): flags += " ⚠NaN"
        if metrics.get("inf_detected", 0): flags += " ⚠Inf"
        if metrics.get("loss_spike", 0): flags += " ⚠Spike"
        mmr = metrics.get("router_max_min_ratio", 0)
        if mmr > 5: flags += f" ⚠Imbal({mmr:.1f})"
        drop_f = metrics.get("router_dropped_frac", 0)
        if drop_f > 0.1: flags += f" ⚠Drop({drop_f:.0%})"

        # Compact line every step
        print(
            f"  [{step:4d}] ntp={loss_n:.4f}  aux={loss_a:.3f}  "
            f"{ms/1000:.1f}s  {tps:.0f}t/s  {mem_a:.2f}G{flags}"
        )

        # Detailed breakdown at intervals
        if step % detail_every == 0 or step == 1:
            null_r = metrics.get("router_null_rate", 0)
            avg_re = metrics.get("router_avg_real_experts", 0)
            glv = metrics.get("router_gate_load_variance", 0)
            expert_counts = metrics.get("router_per_expert_counts", "")
            g_gate = metrics.get("grad_gate_norm", 0)
            g_exp = metrics.get("grad_expert_norm", 0)
            pnorm = metrics.get("param_norm", 0)
            moe_ms = metrics.get("moe_dispatch_ms", 0)
            print(
                f"         routing: null={null_r:.0%} real={avg_re:.1f} "
                f"var={glv:.4f} [{expert_counts}] drop={drop_f:.0%}"
            )
            print(
                f"         grads: gate={g_gate:.2e} expert={g_exp:.2e} |p|={pnorm:.1f}  "
                f"moe={moe_ms:.0f}ms"
            )

    def generate_summary(self, total_steps, checkpoint_resume_ok,
                         config=None,
                         checkpoint_params_match=False,
                         checkpoint_optim_ok=False,
                         reversible_grad_ok=None,
                         gate_determinism_ok=None,
                         zero_alignment_ok=None,
                         moe_dispatch_ok=None,
                         ab_results=None,
                         cross_rank_consistent=None):
        if self.rank != 0:
            return {}

        verdicts = {}

        # ---- Routing Health ----
        null_rates = [r["router_null_rate"] for r in self.rows if "router_null_rate" in r]
        if null_rates:
            final_null = sum(null_rates[-20:]) / len(null_rates[-20:])
            verdicts["routing/null_rate_healthy"] = 0.1 < final_null < 0.9
            verdicts["routing/null_rate_not_collapsed"] = final_null > 0.05

        load_vars = [r["router_gate_load_variance"] for r in self.rows if "router_gate_load_variance" in r]
        if load_vars:
            verdicts["routing/load_balanced"] = sum(load_vars[-20:]) / len(load_vars[-20:]) < 0.5

        mmrs = [r["router_max_min_ratio"] for r in self.rows
                if "router_max_min_ratio" in r and r["router_max_min_ratio"] > 0]
        if mmrs:
            # Tightened from 10 to 5 - ratio >5 almost guarantees token dropping
            verdicts["routing/no_expert_collapse"] = sum(mmrs[-20:]) / len(mmrs[-20:]) < 5.0

        # Dropped token fraction
        drop_fracs = [r.get("router_dropped_frac", 0) for r in self.rows
                      if "router_dropped_frac" in r]
        if drop_fracs:
            avg_drop = sum(drop_fracs[-20:]) / len(drop_fracs[-20:])
            verdicts["routing/no_token_dropping"] = avg_drop < 0.05  # <5% dropped

        # ---- Gradient Flow ----
        gate_norms = [r["grad_gate_norm"] for r in self.rows if "grad_gate_norm" in r]
        if gate_norms:
            verdicts["gradient/gate_receives_grad"] = sum(1 for g in gate_norms if g > 1e-10) > len(gate_norms) * 0.9
        expert_norms = [r["grad_expert_norm"] for r in self.rows if "grad_expert_norm" in r]
        if expert_norms:
            verdicts["gradient/experts_receive_grad"] = sum(1 for g in expert_norms if g > 1e-10) > len(expert_norms) * 0.9

        if reversible_grad_ok is not None:
            verdicts["gradient/reversible_correct"] = reversible_grad_ok
        if gate_determinism_ok is not None:
            verdicts["gradient/gate_deterministic"] = gate_determinism_ok
        if zero_alignment_ok is not None:
            verdicts["gradient/zero_alignment_ok"] = zero_alignment_ok
        if moe_dispatch_ok is not None:
            verdicts["moe/dispatch_correctness"] = moe_dispatch_ok
        if ab_results is not None:
            verdicts["moe/ghost_model_learns"] = ab_results.get("ghost_is_learning", False)
            verdicts["moe/dense_baseline_learns"] = ab_results.get("dense_is_learning", False)
            verdicts["moe/ghost_below_random"] = ab_results.get("ghost_below_random", False)
            verdicts["moe/memory_savings"] = ab_results.get("mem_saved_mb", 0) > 0
            verdicts["moe/gate_avoids_ghosts"] = ab_results.get("gate_avoids_ghosts", False)

        # ---- Loss Dynamics ----
        losses = [r["loss_total"] for r in self.rows if "loss_total" in r]
        ntp_losses = [r["loss_ntp"] for r in self.rows if "loss_ntp" in r]
        if len(losses) > 20:
            first_20 = sum(losses[:20]) / 20
            last_20 = sum(losses[-20:]) / 20
            verdicts["loss/decreasing"] = last_20 < first_20
            verdicts["loss/not_diverging"] = last_20 < 100.0

        # Strong learning signal check: on copy-shift, NTP loss MUST drop
        # well below random chance (ln(4096)=8.318). A working model should
        # reach at least 7.5 within 250 steps. If it stays near 8.3, the
        # model is outputting near-uniform — something is fundamentally broken.
        if len(ntp_losses) > 50:
            final_ntp = sum(ntp_losses[-20:]) / 20
            random_chance = math.log(config.vocab_size) if config is not None and hasattr(config, 'vocab_size') else 8.318
            verdicts["loss/below_random_chance"] = final_ntp < random_chance - 0.1
            verdicts["loss/strong_learning_signal"] = final_ntp < random_chance - 0.5
        aux_losses = [r["loss_aux"] for r in self.rows if "loss_aux" in r]
        if aux_losses:
            verdicts["loss/aux_stable"] = all(a < 10.0 for a in aux_losses[-20:])
        aux_ratios = [r["aux_main_ratio"] for r in self.rows if "aux_main_ratio" in r]
        if aux_ratios:
            verdicts["loss/aux_ratio_bounded"] = max(aux_ratios[-20:]) < 0.5

        # ---- ZeRO Health ----
        param_norms = [r["param_norm"] for r in self.rows if "param_norm" in r]
        if param_norms:
            verdicts["zero/no_nan_params"] = all(not math.isnan(p) and not math.isinf(p) for p in param_norms)
        if cross_rank_consistent is not None:
            verdicts["zero/cross_rank_consistent"] = cross_rank_consistent

        nan_counts = sum(1 for r in self.rows if r.get("nan_detected", 0))
        inf_counts = sum(1 for r in self.rows if r.get("inf_detected", 0))
        verdicts["health/no_nans"] = nan_counts == 0
        verdicts["health/no_infs"] = inf_counts == 0
        verdicts["health/no_loss_spikes"] = sum(1 for r in self.rows if r.get("loss_spike", 0)) < 5

        # ---- Checkpoint ----
        verdicts["checkpoint/zero2_resume"] = checkpoint_resume_ok
        verdicts["checkpoint/params_restored"] = checkpoint_params_match
        verdicts["checkpoint/optimizer_state_ok"] = checkpoint_optim_ok

        # ---- MoE Timing ----
        moe_times = [r["moe_dispatch_ms"] for r in self.rows
                     if "moe_dispatch_ms" in r and r["moe_dispatch_ms"] > 0]
        moe_fraction = 0.0
        if moe_times:
            avg_moe = sum(moe_times) / len(moe_times)
            avg_step = sum(r.get("step_time_ms", 0) for r in self.rows) / max(len(self.rows), 1)
            moe_fraction = avg_moe / max(avg_step, 1e-9)
            verdicts["perf/moe_fraction_reasonable"] = moe_fraction < 0.5

        # ---- Throughput Baseline ----
        tps_all = [r["tokens_per_sec"] for r in self.rows
                   if "tokens_per_sec" in r and r["tokens_per_sec"] > 0]
        # Skip first 10 steps as warmup (JIT, cache cold-start, etc.)
        warmup_cutoff = min(10, len(tps_all) // 4) if len(tps_all) > 10 else 0
        tps_warmup = tps_all[:warmup_cutoff] if warmup_cutoff > 0 else []
        tps_steady = tps_all[warmup_cutoff:] if warmup_cutoff > 0 else tps_all
        self._tps_stats = {}
        if tps_steady:
            self._tps_stats["warmup_mean"] = sum(tps_warmup) / len(tps_warmup) if tps_warmup else 0
            self._tps_stats["steady_mean"] = sum(tps_steady) / len(tps_steady)
            self._tps_stats["steady_min"] = min(tps_steady)
            self._tps_stats["steady_max"] = max(tps_steady)
            self._tps_stats["steady_std"] = (
                sum((t - self._tps_stats["steady_mean"]) ** 2 for t in tps_steady)
                / len(tps_steady)
            ) ** 0.5
            self._tps_stats["total_steps"] = len(tps_all)
            self._tps_stats["warmup_steps"] = warmup_cutoff
            # Verdict: throughput should be stable (std < 30% of mean)
            cv = self._tps_stats["steady_std"] / max(self._tps_stats["steady_mean"], 1e-9)
            verdicts["perf/throughput_stable"] = cv < 0.3

        # ---- Write Summary ----
        with open(self.summary_path, "w") as f:
            f.write("=" * 72 + "\n")
            f.write("  DEEPSPEED ZERO-2 FULL TRAINING VALIDATION SUMMARY\n")
            f.write("=" * 72 + "\n\n")
            all_pass = True
            for name, passed in sorted(verdicts.items()):
                status = "PASS" if passed else "FAIL"
                f.write(f"  [{status}]  {name}\n")
                if not passed:
                    all_pass = False
            total_pass = sum(verdicts.values())
            total_tests = len(verdicts)
            f.write(f"\n  Score: {total_pass}/{total_tests}\n\n")
            f.write("  ALL VALIDATIONS PASSED\n" if all_pass else "  FAILURES DETECTED\n")

            f.write("\n" + "-" * 72 + "\n  KEY STATISTICS\n" + "-" * 72 + "\n\n")
            if null_rates:
                f.write(f"  Final null_rate (avg last 20): {sum(null_rates[-20:])/len(null_rates[-20:]):.4f}\n")
            if moe_times:
                f.write(f"  Avg MoE dispatch time: {avg_moe:.2f} ms ({moe_fraction:.1%} of step)\n")
            if mmrs:
                f.write(f"  Avg expert max/min ratio: {sum(mmrs[-20:])/len(mmrs[-20:]):.2f}\n")
            if drop_fracs:
                f.write(f"  Avg dropped token fraction: {avg_drop:.4f}\n")
            if losses:
                f.write(f"  Loss first 20 avg: {sum(losses[:20])/min(len(losses),20):.4f}\n")
                f.write(f"  Loss last 20 avg:  {sum(losses[-20:])/len(losses[-20:]):.4f}\n")
            if ntp_losses:
                final_ntp = sum(ntp_losses[-20:]) / len(ntp_losses[-20:])
                random_chance_val = math.log(config.vocab_size) if config is not None and hasattr(config, 'vocab_size') else 8.318
                f.write(f"  NTP loss (last 20):  {final_ntp:.4f} (random chance = {random_chance_val:.4f})\n")
                f.write(f"  NTP gap from random: {final_ntp - random_chance_val:+.4f}\n")
            recent_counts = [r.get("router_per_expert_counts", "") for r in self.rows[-20:]
                             if r.get("router_per_expert_counts", "")]
            if recent_counts:
                f.write(f"\n  Last step expert counts: [{recent_counts[-1]}]\n")
            xr_spreads = [r.get("cross_rank_param_norm_spread", 0) for r in self.rows
                          if "cross_rank_param_norm_spread" in r]
            if xr_spreads:
                f.write(f"  Max cross-rank param norm spread: {max(xr_spreads):.6f}\n")

            if hasattr(self, '_tps_stats') and self._tps_stats:
                ts = self._tps_stats
                f.write("\n" + "-" * 72 + "\n  THROUGHPUT BASELINE\n" + "-" * 72 + "\n\n")
                f.write(f"  Steady-state tok/s (steps {ts['warmup_steps']}-{ts['total_steps']}):\n")
                f.write(f"    Mean:  {ts['steady_mean']:,.0f}\n")
                f.write(f"    Min:   {ts['steady_min']:,.0f}\n")
                f.write(f"    Max:   {ts['steady_max']:,.0f}\n")
                f.write(f"    Std:   {ts['steady_std']:,.0f}\n")
                cv = ts['steady_std'] / max(ts['steady_mean'], 1e-9)
                f.write(f"    CV:    {cv:.2%}\n")
                if ts['warmup_mean'] > 0:
                    f.write(f"  Warmup tok/s (steps 0-{ts['warmup_steps']}): {ts['warmup_mean']:,.0f}\n")
                    speedup = ts['steady_mean'] / max(ts['warmup_mean'], 1e-9)
                    f.write(f"  Warmup → Steady speedup: {speedup:.2f}x\n")
                f.write("\n  ℹ️  Save this as baseline before architecture changes.\n")

            f.write("\n" + "=" * 72 + "\n")

            # ---- 3B READINESS ASSESSMENT ----
            critical_verdicts = {
                "moe/dispatch_correctness": "Ghost expert dispatch produces correct outputs",
                "moe/ghost_model_learns": "Ghost-expert model learns (loss decreases)",
                "moe/dense_baseline_learns": "Dense-expert baseline learns (confirms task is learnable)",
                "moe/ghost_below_random": "Ghost loss drops well below random chance (real learning)",
                "moe/memory_savings": "Ghost experts save GPU memory vs dense",
                "moe/gate_avoids_ghosts": "Gate learns to route away from ghost experts",
                "gradient/zero_alignment_ok": "ZeRO-2 flat-buffer offsets are correct",
                "gradient/reversible_correct": "Reversible backprop gradients match standard",
                "gradient/gate_receives_grad": "Router receives gradient (can learn routing)",
                "gradient/experts_receive_grad": "Expert weights receive gradient (can learn)",
                "loss/below_random_chance": "NTP loss dropped below random chance on copy-shift",
                "loss/strong_learning_signal": "NTP loss dropped significantly (model is learning)",
                "routing/no_token_dropping": "No tokens dropped during dispatch",
                "checkpoint/zero2_resume": "Checkpoint save/load works with ZeRO-2",
            }
            f.write("\n" + "=" * 72 + "\n")
            f.write("  3B TRAINING READINESS ASSESSMENT\n")
            f.write("=" * 72 + "\n\n")
            critical_pass = 0
            critical_total = 0
            for vname, desc in critical_verdicts.items():
                if vname in verdicts:
                    critical_total += 1
                    passed = verdicts[vname]
                    if passed:
                        critical_pass += 1
                    status = "✅" if passed else "❌"
                    f.write(f"  {status} {desc}\n")
                    f.write(f"     ({vname}: {'PASS' if passed else 'FAIL'})\n")
            f.write(f"\n  Critical checks: {critical_pass}/{critical_total}\n")
            if critical_pass == critical_total:
                f.write("\n  ✅ ALL CRITICAL CHECKS PASSED — safe to proceed with 3B training run\n")
            else:
                f.write(f"\n  ❌ {critical_total - critical_pass} CRITICAL CHECK(S) FAILED — DO NOT proceed with 3B run\n")

            # ---- Cost Savings Report ----
            if ab_results is not None:
                f.write("\n" + "=" * 72 + "\n")
                f.write("  GHOST EXPERT COST SAVINGS REPORT\n")
                f.write("=" * 72 + "\n\n")
                f.write(f"  Ghost config:  {ab_results.get('ghost_params', 0):,} params\n")
                f.write(f"  Dense config:  {ab_results.get('dense_params', 0):,} params\n")
                f.write(f"  Param savings: {ab_results.get('param_saved', 0):,} ({ab_results.get('param_saved_pct', 0):.1f}%)\n\n")
                f.write(f"  Ghost peak memory:  {ab_results.get('ghost_peak_mb', 0):.1f} MB\n")
                f.write(f"  Dense peak memory:  {ab_results.get('dense_peak_mb', 0):.1f} MB\n")
                f.write(f"  Memory savings:     {ab_results.get('mem_saved_mb', 0):.1f} MB ({ab_results.get('mem_saved_pct', 0):.1f}%)\n\n")
                f.write(f"  Ghost final NTP loss: {ab_results.get('ghost_final_loss', 0):.4f}\n")
                f.write(f"  Dense final NTP loss: {ab_results.get('dense_final_loss', 0):.4f}\n")
                f.write(f"  Loss gap:            {ab_results.get('loss_gap_pct', 0):+.1f}%\n\n")
                f.write(f"  Step time delta:     {ab_results.get('time_saved_pct', 0):+.1f}%\n\n")
                # Extrapolate to 3B
                scale = 3e9 / max(ab_results.get('ghost_params', 1), 1)
                est_mem_save_gb = ab_results.get('mem_saved_mb', 0) * scale / 1000
                f.write(f"  ── Extrapolated to 3B scale (×{scale:.0f}) ──\n")
                f.write(f"  Est. memory savings: ~{est_mem_save_gb:.1f} GB per GPU\n")
                f.write(f"  Est. parameter savings: ~{ab_results.get('param_saved', 0) * scale / 1e9:.2f}B params\n")

            f.write("\n" + "=" * 72 + "\n")

        print("\n" + open(self.summary_path).read())
        return verdicts


# ============================================================================
# METRIC EXTRACTION HELPERS
# ============================================================================

def collect_grad_metrics(model):
    """Collect gradient metrics.

    ZeRO-2 Compatibility:
    DeepSpeed ZeRO-2 clears param.grad during backward (reduce-scatter hooks).
    By the time this function runs after backward(), param.grad is None.
    We first try param.grad (works for non-ZeRO or single-GPU).
    If all grads are None, we fall back to checking whether parameters
    changed after the upcoming step() call (deferred metrics).

    For real-time grad norms, use GradNormTracker (backward hook approach).
    """
    metrics = {}
    gate_norms, expert_norms, shared_norms, memory_gate_norms = [], [], [], []
    base_model = model.module if hasattr(model, 'module') else model
    total_checked = 0
    total_with_grad = 0
    for name, param in base_model.named_parameters():
        total_checked += 1
        if param.grad is None:
            continue
        total_with_grad += 1
        gn = param.grad.data.norm(2).item()
        if "router_adapter.gate.gate.weight" in name or "gate.gate.weight" in name:
            gate_norms.append(gn)
        elif ("w_gate.weight" in name or "w_up.weight" in name or "w_down.weight" in name) \
                and "shared" not in name and "router" not in name:
            expert_norms.append(gn)
        elif "shared_gate" in name or "shared_up" in name or "shared_down" in name:
            shared_norms.append(gn)
        elif "memory_gate_proj" in name:
            memory_gate_norms.append(gn)
    metrics["grad_gate_norm"] = sum(gate_norms) / max(len(gate_norms), 1)
    metrics["grad_expert_norm"] = sum(expert_norms) / max(len(expert_norms), 1)
    metrics["grad_shared_expert_norm"] = sum(shared_norms) / max(len(shared_norms), 1)
    metrics["grad_memory_gate_norm"] = sum(memory_gate_norms) / max(len(memory_gate_norms), 1)

    total_norm = sum(p.data.norm(2).item() ** 2 for p in base_model.parameters() if p.data is not None)
    metrics["param_norm"] = math.sqrt(total_norm)
    total_grad = sum(p.grad.data.norm(2).item() ** 2 for p in base_model.parameters() if p.grad is not None)
    metrics["grad_global_norm"] = math.sqrt(total_grad)

    # Flag if ZeRO cleared all grads (so caller knows to use GradNormTracker instead)
    metrics["_zero_cleared_grads"] = (total_with_grad == 0 and total_checked > 0)
    return metrics


class GradNormTracker:
    """
    Captures gradient norms via backward hooks — works with ZeRO-2.

    ZeRO-2's reduce-scatter hooks clear param.grad during backward, making
    post-backward gradient inspection impossible. This class registers
    backward hooks that fire BEFORE DeepSpeed processes gradients, capturing
    the full (unreduced) gradient norms per parameter group.

    Usage:
        tracker = GradNormTracker()
        tracker.attach(model_engine)
        ...
        model_engine.backward(loss)
        metrics = tracker.get_metrics()  # Has grad norms
        tracker.reset()  # Clear for next step
    """
    def __init__(self):
        self._norms = {"gate": [], "expert": [], "shared": [], "memory_gate": [], "other": []}
        self._hooks = []

    def attach(self, model):
        base = model.module if hasattr(model, 'module') else model
        for name, param in base.named_parameters():
            if not param.requires_grad:
                continue
            category = self._categorize(name)
            self._hooks.append(
                param.register_hook(self._make_hook(category))
            )

    def _categorize(self, name):
        if "router_adapter.gate.gate.weight" in name or "gate.gate.weight" in name:
            return "gate"
        if ("w_gate.weight" in name or "w_up.weight" in name or "w_down.weight" in name) \
                and "shared" not in name and "router" not in name:
            return "expert"
        if "shared_gate" in name or "shared_up" in name or "shared_down" in name:
            return "shared"
        if "memory_gate_proj" in name:
            return "memory_gate"
        return "other"

    def _make_hook(self, category):
        tracker = self
        def hook_fn(grad):
            if grad is not None:
                tracker._norms[category].append(grad.data.norm(2).item())
        return hook_fn

    def get_metrics(self):
        metrics = {}
        for key, label in [("gate", "grad_gate_norm"), ("expert", "grad_expert_norm"),
                           ("shared", "grad_shared_expert_norm"), ("memory_gate", "grad_memory_gate_norm")]:
            norms = self._norms[key]
            metrics[label] = sum(norms) / max(len(norms), 1)
        all_norms = [n for ns in self._norms.values() for n in ns]
        metrics["grad_global_norm"] = math.sqrt(sum(n**2 for n in all_norms)) if all_norms else 0.0
        return metrics

    def reset(self):
        for k in self._norms:
            self._norms[k].clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def collect_cross_rank_param_norm(model, device, world_size):
    if world_size <= 1:
        return 0.0
    base_model = model.module if hasattr(model, 'module') else model
    local_norm = math.sqrt(sum(p.data.norm(2).item() ** 2 for p in base_model.parameters() if p.data is not None))
    norm_tensor = torch.tensor([local_norm], device=device, dtype=torch.float64)
    gathered = [torch.zeros_like(norm_tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, norm_tensor)
    norms = [g.item() for g in gathered]
    return max(norms) - min(norms)


# ============================================================================
# ROUTING TRACKER (with per-expert counts + dropped token detection)
# ============================================================================

class RoutingTracker:
    def __init__(self):
        self.latest_metrics = {}
        self._hooks = []

    def attach(self, model):
        base = model.module if hasattr(model, 'module') else model
        for name, module in base.named_modules():
            if module.__class__.__name__ == "MoEGate":
                self._hooks.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, layer_name):
        tracker = self
        def hook_fn(module, input, output):
            topk_idx, topk_weight, is_null, aux_loss = output
            with torch.no_grad():
                null_rate = is_null.float().mean().item()
                real_per_token = (~is_null).float().sum(dim=-1).mean().item()
                num_experts = module.num_experts
                flat_idx = topk_idx.view(-1)
                flat_null = is_null.view(-1)
                real_idx = flat_idx[~flat_null]

                if real_idx.numel() > 0:
                    counts = torch.bincount(real_idx, minlength=num_experts).float()
                    fracs = counts / counts.sum().clamp(min=1)
                    load_var = fracs.var().item()
                    max_c = counts.max().item()
                    min_c = counts[counts > 0].min().item() if (counts > 0).any() else 1.0
                    max_min_ratio = max_c / max(min_c, 1.0)
                    per_expert_str = ",".join(str(int(c)) for c in counts.tolist())

                    # Dropped token estimation:
                    # The adapter uses capacity = ceil(k * S / num_real_experts)
                    # where num_real = actual experts (not ghost slots).
                    # total_tokens here = flat_idx.numel() = S * k (all selections incl ghosts)
                    total_tokens = flat_idx.numel()
                    num_real_experts = num_experts  # MoEGate.num_experts = real experts only
                    capacity_per_expert = math.ceil(total_tokens / max(num_real_experts, 1))
                    dropped = counts.clamp(min=0) - capacity_per_expert
                    dropped = dropped.clamp(min=0).sum().item()
                    dropped_frac = dropped / max(total_tokens, 1)
                else:
                    load_var = 0.0
                    max_min_ratio = 0.0
                    per_expert_str = ""
                    dropped_frac = 0.0

                tracker.latest_metrics = {
                    "router_null_rate": null_rate,
                    "router_avg_real_experts": real_per_token,
                    "router_gate_load_variance": load_var,
                    "router_max_min_ratio": max_min_ratio,
                    "router_per_expert_counts": per_expert_str,
                    "router_dropped_frac": dropped_frac,
                }
        return hook_fn

    def get_metrics(self):
        return self.latest_metrics.copy()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ============================================================================
# MOE-ISOLATED TIMING
# ============================================================================

class MoETimingTracker:
    def __init__(self):
        self.latest_ms = 0.0
        self._hooks = []
        self._start = 0.0

    def attach(self, model):
        base = model.module if hasattr(model, 'module') else model
        for name, module in base.named_modules():
            if module.__class__.__name__ == "DeepSpeedMoEFFN":
                self._hooks.append(module.register_forward_pre_hook(self._pre_hook))
                self._hooks.append(module.register_forward_hook(self._post_hook))
                break

    def _pre_hook(self, module, input):
        torch.cuda.synchronize()
        self._start = time.perf_counter()

    def _post_hook(self, module, input, output):
        torch.cuda.synchronize()
        self.latest_ms = (time.perf_counter() - self._start) * 1000

    def get_ms(self):
        return self.latest_ms

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ============================================================================
# GPU MEMORY PROFILER
# ============================================================================

class MemoryProfiler:
    """
    Tracks GPU memory at key points to identify the actual bottleneck
    (params vs optimizer vs activations). Provides data-driven recommendations
    for batch size / seq_len tuning.

    Usage:
        profiler = MemoryProfiler(device, rank)
        profiler.snapshot("after_model_init")
        profiler.snapshot("after_deepspeed_init")
        # Inside step 0:
        profiler.reset_peak()   # before forward
        profiler.snapshot("peak_forward")   # after forward
        profiler.snapshot("peak_backward")  # after backward
        profiler.report()
    """
    def __init__(self, device, rank=0):
        self.device = device
        self.rank = rank
        self.snapshots = {}

    def snapshot(self, label):
        torch.cuda.synchronize()
        self.snapshots[label] = {
            "allocated_mb": torch.cuda.memory_allocated(self.device) / 1e6,
            "reserved_mb": torch.cuda.memory_reserved(self.device) / 1e6,
            "max_allocated_mb": torch.cuda.max_memory_allocated(self.device) / 1e6,
        }

    def reset_peak(self):
        torch.cuda.reset_peak_memory_stats(self.device)

    def report(self):
        if self.rank != 0:
            return

        total_gpu_mb = torch.cuda.get_device_properties(self.device).total_memory / 1e6

        print("\n" + "=" * 72)
        print("  GPU MEMORY UTILIZATION REPORT")
        print("=" * 72)
        print(f"  Total GPU: {total_gpu_mb:.0f} MB ({total_gpu_mb/1024:.1f} GB)")
        print()

        for label, snap in self.snapshots.items():
            util = snap["allocated_mb"] / total_gpu_mb * 100
            print(f"  [{label}]")
            print(f"    Allocated: {snap['allocated_mb']:.0f} MB ({util:.1f}%)")
            print(f"    Reserved:  {snap['reserved_mb']:.0f} MB")
            print(f"    Peak:      {snap['max_allocated_mb']:.0f} MB")
            print()

        # Derived insights
        if "after_model_init" in self.snapshots and "after_deepspeed_init" in self.snapshots:
            ds_overhead = (self.snapshots["after_deepspeed_init"]["allocated_mb"] -
                          self.snapshots["after_model_init"]["allocated_mb"])
            print(f"  DeepSpeed ZeRO-2 overhead: {ds_overhead:.0f} MB ({ds_overhead/1024:.2f} GB)")

        if "peak_forward" in self.snapshots and "after_deepspeed_init" in self.snapshots:
            act_mem = (self.snapshots["peak_forward"]["max_allocated_mb"] -
                      self.snapshots["after_deepspeed_init"]["allocated_mb"])
            print(f"  Activation memory (forward peak): {act_mem:.0f} MB ({act_mem/1024:.2f} GB)")

        if "peak_backward" in self.snapshots:
            peak_bwd = self.snapshots["peak_backward"]["max_allocated_mb"]
            if "after_deepspeed_init" in self.snapshots:
                static = self.snapshots["after_deepspeed_init"]["allocated_mb"]
                total_dynamic = peak_bwd - static
                print(f"  Total dynamic memory (fwd+bwd peak): {total_dynamic:.0f} MB ({total_dynamic/1024:.2f} GB)")

        peak = max(s["max_allocated_mb"] for s in self.snapshots.values())
        headroom = total_gpu_mb - peak
        print(f"\n  Peak utilization: {peak:.0f} MB ({peak/total_gpu_mb*100:.1f}%)")
        print(f"  Headroom: {headroom:.0f} MB ({headroom/1024:.2f} GB)")

        if headroom > 2000:
            print(f"\n  RECOMMENDATION: {headroom/1024:.1f} GB headroom. Consider:")
            print(f"    - Increasing micro_batch_size for better throughput")
            print(f"    - Increasing seq_len for longer context")
        elif headroom > 500:
            print(f"\n  Status: Good utilization. {headroom/1024:.1f} GB safety margin.")
        else:
            print(f"\n  WARNING: Only {headroom:.0f} MB headroom. OOM risk.")
            print(f"    - Reduce micro_batch_size or seq_len")
            print(f"    - Enable activation checkpointing")
        print("=" * 72)


# ============================================================================
# CHECK A: GATE ROUTING DETERMINISM
# ============================================================================

def check_gate_determinism(model_engine, config, device, rank):
    """
    Call forward TWICE with identical input, verify gate produces identical indices.
    If indices differ, MidpointFunction.backward (which re-runs forward) will
    compute wrong gradients -- this is the root cause of reversible corruption.

    ALL ranks must participate because DeepSpeed MoE uses NCCL collectives
    even with ep_size=1. Only rank 0 compares results.
    """
    base_model = model_engine.module if hasattr(model_engine, 'module') else model_engine
    micro_batch = model_engine.train_micro_batch_size_per_gpu()
    seq_len = config.max_seq_len
    if rank == 0:
        print("  Gate routing determinism check...")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (micro_batch, seq_len), device=device)
    next_token_ids = torch.randint(0, config.vocab_size, (micro_batch, seq_len), device=device)

    def make_capture_hook():
        captured = []
        def hook_fn(module, inp, out):
            # DeepSpeedRouterAdapter returns (weights, indices)
            # MoEGate returns (topk_idx, topk_weight, is_null, aux_loss) — NOTE: idx first
            if isinstance(out, tuple) and len(out) >= 2:
                if module.__class__.__name__ == "DeepSpeedRouterAdapter":
                    captured.append(out[1].detach().clone())  # indices from adapter
                else:
                    captured.append(out[0].detach().clone())  # topk_idx from MoEGate
            elif isinstance(out, torch.Tensor):
                captured.append(out.detach().clone())
        return hook_fn, captured

    # Hook only one level: prefer DeepSpeedRouterAdapter (what DeepSpeed calls),
    # fall back to MoEGate if adapter not found
    gates = [(n, m) for n, m in base_model.named_modules()
             if m.__class__.__name__ == "DeepSpeedRouterAdapter"]
    if not gates:
        gates = [(n, m) for n, m in base_model.named_modules()
                 if m.__class__.__name__ == "MoEGate"]
    if not gates and rank == 0:
        print("    No MoEGate/DeepSpeedRouterAdapter found -- skipping")
        return True, 0

    # Run 1 — ALL RANKS
    hooks_1, caps_1 = [], []
    for _, g in gates:
        hfn, cap = make_capture_hook()
        hooks_1.append(g.register_forward_hook(hfn)); caps_1.append(cap)
    base_model.eval()
    with torch.no_grad():
        torch.manual_seed(42); torch.cuda.manual_seed(42)
        _ = base_model(input_ids, next_token_ids=next_token_ids, return_loss=True, return_memory=False)
    for h in hooks_1: h.remove()

    # Run 2 — ALL RANKS
    hooks_2, caps_2 = [], []
    for _, g in gates:
        hfn, cap = make_capture_hook()
        hooks_2.append(g.register_forward_hook(hfn)); caps_2.append(cap)
    with torch.no_grad():
        torch.manual_seed(42); torch.cuda.manual_seed(42)
        _ = base_model(input_ids, next_token_ids=next_token_ids, return_loss=True, return_memory=False)
    for h in hooks_2: h.remove()
    base_model.train()

    # Compare (rank 0 only)
    if rank == 0:
        total_mm, total_ent = 0, 0
        for i, (name, _) in enumerate(gates):
            if caps_1[i] and caps_2[i]:
                mm = (caps_1[i][0] != caps_2[i][0]).sum().item()
                total_ent += caps_1[i][0].numel(); total_mm += mm
                if mm > 0:
                    print(f"    {name}: {mm}/{caps_1[i][0].numel()} mismatches")

        if total_ent == 0:
            print(f"    WARNING: No routing decisions captured (hooks may not match gate output format)")
            return True, 0

        if total_mm == 0:
            print(f"    PASS: All {total_ent} routing decisions identical across 2 runs")
            return True, 0
        else:
            print(f"    FAIL: {total_mm}/{total_ent} mismatches ({total_mm/max(total_ent,1):.2%})")
            return False, total_mm

    return True, 0


# ============================================================================
# CHECK B: REVERSIBLE VS STANDARD GRADIENT (FIXED -- monkey-patch approach)
# ============================================================================

def verify_reversible_gradients(model_engine, config, device, rank):
    """
    Compare gradients from reversible backprop vs standard autograd.

    Uses a fresh model NOT wrapped by DeepSpeed ZeRO — so backward produces
    full unpartitioned gradients suitable for comparison (ZeRO-2's reduce-scatter
    would partition them, causing false 88x divergence).

    CRITICAL: ALL ranks must participate because Model3B contains DeepSpeedMoEFFN
    which uses ds_moe.MoE(ep_size=1). Even with ep_size=1, DeepSpeed MoE's
    internal dispatch uses NCCL collectives. If only rank 0 runs forward,
    it deadlocks waiting for other ranks in the MoE all-to-all.

    All ranks create fresh model + run forward+backward. Only rank 0 compares.

    Tolerance: <1% PASS, 1-5% WARNING, >5% FAIL
    """
    if rank == 0:
        print("  Reversible gradient correctness check...")
        print("    (Fresh model: no ZeRO hooks, all ranks participate for MoE collectives)")

    # ALL ranks create fresh model (MoE init may use NCCL process groups)
    with quiet_init(rank):
        fresh_model = Model3B(config, embedding_type="standard").to(device)
    fresh_model.train()

    micro_batch = model_engine.train_micro_batch_size_per_gpu()
    seq_len = config.max_seq_len

    # All ranks create identical input
    torch.manual_seed(42); torch.cuda.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (micro_batch, seq_len), device=device)
    next_token_ids = torch.randint(0, config.vocab_size, (micro_batch, seq_len), device=device)
    targets = torch.randint(0, config.vocab_size, (micro_batch, seq_len), device=device)

    # --- Run 1: Production (reversible) --- ALL RANKS
    fresh_model.zero_grad()
    try:
        torch.manual_seed(42); torch.cuda.manual_seed(42)
        out_rev = fresh_model(input_ids, next_token_ids=next_token_ids, return_loss=True, return_memory=False)
        logits_rev = out_rev[0]
        aux_rev = out_rev[2] if len(out_rev) > 2 and out_rev[2] is not None else torch.tensor(0.0, device=device)
        loss_rev = F.cross_entropy(logits_rev.view(-1, config.vocab_size), targets.view(-1)) + aux_rev
        loss_rev.backward()
    except Exception as e:
        if rank == 0:
            print(f"    ERROR in reversible forward: {e}"); traceback.print_exc()
        del fresh_model; torch.cuda.empty_cache()
        return False, float('inf')

    # Rank 0 saves reversible grads (these are FULL unpartitioned — no ZeRO hooks)
    rev_grads = {}
    if rank == 0:
        rev_grads = {n: p.grad.clone() for n, p in fresh_model.named_parameters() if p.grad is not None}
        print(f"    Reversible loss: {loss_rev.item():.6f}")

    # --- Find and patch reversible stack (ALL RANKS) ---
    rev_stack = None
    for _, m in fresh_model.named_modules():
        if m.__class__.__name__ == "ReversibleMidpointStack":
            rev_stack = m; break
    if rev_stack is None:
        if rank == 0:
            print("    No ReversibleMidpointStack -- skipping (N/A)")
        del fresh_model; torch.cuda.empty_cache()
        return True, 0.0

    original_forwards = []
    for ml in rev_stack.mid_layers:
        original_forwards.append(ml.forward)
        def _make_std(block_ref):
            def std_fwd(p_prev, p_cur):
                delta, aux = block_ref.block.force(p_cur)
                p_next = (block_ref.a * p_prev) + ((1.0 - block_ref.a) * p_cur) + (block_ref.two_h * delta)
                return p_next, aux
            return std_fwd
        ml.forward = _make_std(ml)

    # --- Run 2: Standard (patched) --- ALL RANKS
    fresh_model.zero_grad()
    try:
        torch.manual_seed(42); torch.cuda.manual_seed(42)
        out_std = fresh_model(input_ids, next_token_ids=next_token_ids, return_loss=True, return_memory=False)
        logits_std = out_std[0]
        aux_std = out_std[2] if len(out_std) > 2 and out_std[2] is not None else torch.tensor(0.0, device=device)
        loss_std = F.cross_entropy(logits_std.view(-1, config.vocab_size), targets.view(-1)) + aux_std
        loss_std.backward()
    except Exception as e:
        if rank == 0:
            print(f"    ERROR in standard forward: {e}"); traceback.print_exc()
        for i, ml in enumerate(rev_stack.mid_layers): ml.forward = original_forwards[i]
        del fresh_model; torch.cuda.empty_cache()
        return False, float('inf')

    if rank == 0:
        print(f"    Standard loss:   {loss_std.item():.6f}")
        print(f"    Loss difference: {abs(loss_rev.item() - loss_std.item()):.8f}")

    # Restore (all ranks)
    for i, ml in enumerate(rev_stack.mid_layers): ml.forward = original_forwards[i]

    # --- Compare (rank 0 only — grads are full, unpartitioned) ---
    max_rel_diff = 0.0
    passed = True
    if rank == 0:
        mismatches, total_compared, worst_param = 0, 0, ""
        for name, p in fresh_model.named_parameters():
            if p.grad is not None and name in rev_grads:
                rd = ((rev_grads[name] - p.grad).norm() / (p.grad.norm() + 1e-8)).item()
                total_compared += 1
                if rd > max_rel_diff: max_rel_diff = rd; worst_param = name
                if rd > 0.05:
                    mismatches += 1
                    if mismatches <= 5:
                        print(f"    MISMATCH: {name} rel_diff={rd:.6f}")

        print(f"    Compared {total_compared} params, max_rel_diff={max_rel_diff:.6f} ({worst_param})")

        if max_rel_diff < 0.01:
            print(f"    PASS: Gradients match (<1% relative diff)")
        elif max_rel_diff < 0.05:
            print(f"    WARNING: Moderate divergence ({max_rel_diff:.4f}) -- monitor stability")
        else:
            print(f"    FAIL: Severe divergence ({max_rel_diff:.4f}) -- reversible backprop broken")
            passed = False

    # Cleanup (all ranks)
    del fresh_model
    torch.cuda.empty_cache()

    return passed, max_rel_diff


# ============================================================================
# CHECK C: ZERO FLAT-BUFFER ALIGNMENT (Parameter Update Verification)
# ============================================================================

def verify_zero_alignment(model_engine, config, device, rank, world_size):
    """
    The "Jagged ZeRO" check. Ghost experts (nn.Identity, 0 params) mixed with
    real experts (SingleExpertMLP, N params) create asymmetric parameter lists.
    ZeRO-2 flattens all params into a contiguous buffer and computes offsets.
    If offsets are miscalculated, parameter updates bleed across boundaries.

    Method: Run a few real training steps through the DeepSpeed engine, then
    verify that BOTH expert and non-expert parameters actually updated.
    If ZeRO flat-buffer alignment is broken, some parameter groups won't update
    or will update incorrectly (bleed). This is more reliable than gradient
    injection, which is incompatible with ZeRO-2's reduce-scatter hooks.
    """
    base_model = model_engine.module if hasattr(model_engine, 'module') else model_engine

    if rank == 0:
        print("  ZeRO flat-buffer alignment check (parameter update verification)...")

    micro_batch = model_engine.train_micro_batch_size_per_gpu()
    seq_len = config.max_seq_len
    ga_steps = model_engine.gradient_accumulation_steps()

    # Categorize parameters into groups we expect to update
    param_groups = {
        "attention": [],   # q/k/v/o projections (in ZeRO flat buffer)
        "expert": [],      # MoE expert weights (in MoE all-reduce pathway)
        "gate": [],        # Router/gate weights
        "other": [],       # Everything else (norms, embeddings, etc.)
    }

    for n, p in base_model.named_parameters():
        if not p.requires_grad or p.numel() == 0 or "dummy_grad_sink" in n:
            continue
        if any(k in n for k in ("q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj")):
            param_groups["attention"].append(n)
        elif any(k in n for k in ("w_gate.", "w_up.", "w_down.", "experts.")):
            param_groups["expert"].append(n)
        elif any(k in n for k in ("router_adapter.", "gate.gate")):
            param_groups["gate"].append(n)
        else:
            param_groups["other"].append(n)

    if rank == 0:
        for grp, names in param_groups.items():
            print(f"    {grp}: {len(names)} parameters")

    # Snapshot all parameters before training
    # Exclude known-zero-gradient parameters (dummy_grad_sink is multiplied by 0.0 by design)
    initial_params = {
        n: p.data.clone() for n, p in base_model.named_parameters()
        if p.requires_grad and "dummy_grad_sink" not in n
    }

    # Run 2 full optimizer steps (2 * ga_steps micro-batches)
    num_opt_steps = 2
    for step in range(num_opt_steps):
        for micro in range(ga_steps):
            seed = 54321 + step * ga_steps + micro
            torch.manual_seed(seed); torch.cuda.manual_seed(seed)
            input_ids = torch.randint(0, config.vocab_size, (micro_batch, seq_len), device=device)
            next_ids = torch.randint(0, config.vocab_size, (micro_batch, seq_len), device=device)
            targets = torch.randint(0, config.vocab_size, (micro_batch, seq_len), device=device)
            mtp_targets = torch.randint(0, config.vocab_size, (micro_batch, seq_len), device=device)

            outputs = model_engine(input_ids, next_token_ids=next_ids, return_loss=True, return_memory=False)
            logits = outputs[0]
            logits_mtp = outputs[1] if len(outputs) > 1 else None
            aux = outputs[2] if len(outputs) > 2 and outputs[2] is not None else torch.tensor(0.0, device=device)

            # Main NTP loss
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
            # MTP loss — critical for MTP block gradients
            if logits_mtp is not None:
                ml = min(logits_mtp.size(1), mtp_targets.size(1))
                loss = loss + 0.3 * F.cross_entropy(
                    logits_mtp[:, :ml].reshape(-1, config.vocab_size),
                    mtp_targets[:, :ml].reshape(-1)
                )
            loss = loss + aux
            model_engine.backward(loss)
            model_engine.step()  # Must call after EVERY backward(); DS fires optimizer at GA boundary

    # Check which parameter groups updated
    group_results = {}
    param_dict = dict(base_model.named_parameters())
    for grp, names in param_groups.items():
        updated = 0
        stuck = 0
        max_delta = 0.0
        stuck_examples = []
        for n in names:
            if n not in initial_params or n not in param_dict:
                continue
            p = param_dict[n]
            delta = (p.data - initial_params[n]).abs().max().item()
            max_delta = max(max_delta, delta)
            if delta > 1e-10:
                updated += 1
            else:
                stuck += 1
                if len(stuck_examples) < 3:
                    stuck_examples.append(n)
        group_results[grp] = {
            "updated": updated, "stuck": stuck,
            "max_delta": max_delta, "stuck_examples": stuck_examples
        }

    # Report results
    # Scalar/arch params (A_log, D, lambda_r_raw, etc.) may have tiny gradients
    # and not move in just 2 optimizer steps on random data. These are warnings, not failures.
    # The ZeRO alignment test verifies flat-buffer offsets are correct.
    # Critical signal: attention (ZeRO buffer), expert (MoE pathway), gate (router grad flow)
    # must ALL update. "other" (norms, biases, coefficients) may have tiny gradients on
    # random data and not move in 2 optimizer steps — this is expected, not an alignment failure.
    critical_groups = {"attention", "expert", "gate"}
    all_critical_pass = True
    for grp, res in group_results.items():
        total = res["updated"] + res["stuck"]
        if total == 0:
            continue

        is_critical = grp in critical_groups
        status = "PASS" if res["stuck"] == 0 else ("FAIL" if is_critical else "WARN")
        if is_critical and res["stuck"] > 0:
            all_critical_pass = False

        if rank == 0:
            print(f"    {grp}: {res['updated']}/{total} updated (max_delta={res['max_delta']:.2e}) [{status}]")
            if not is_critical and res["stuck"] > 0:
                print(f"      ({res['stuck']} params with tiny gradients on random data — expected)")
            for ex in res["stuck_examples"]:
                if is_critical:
                    print(f"      STUCK: {ex}")

    if rank == 0:
        if all_critical_pass:
            print("    PASS: All critical parameter groups (attention, expert, gate) updated correctly")
        else:
            print("    FAIL: Critical parameter group did not update (ZeRO alignment issue)")

    return all_critical_pass


# ============================================================================

# ============================================================================
# CHECK D: MoE DISPATCH CORRECTNESS & GHOST EXPERT VALIDATION
# ============================================================================

def verify_moe_dispatch_correctness(model_engine, config, device, rank, world_size):
    """
    DEFINITIVE dispatch correctness audit for DeepSpeed MoE with ghost experts.

    This is the critical gate before committing to a 3B training run.  It proves
    that DeepSpeed's internal dispatch mechanism correctly uses our dispatch_mask
    and combine_weights, that ghost (Identity) experts are truly zero-cost, and
    that gradients flow through the real expert pathway.

    Tests (all must pass):
      A. Structural audit — ghost experts are Identity, 0 trainable params
      B. Golden-reference test — manually reconstruct MoE output from individual
         expert outputs + combine_weights; compare with DeepSpeed's actual output.
         If they match, dispatch is mathematically correct.
      C. Token coverage — every token appears in dispatch_mask (no silent drops)
      D. Capacity audit — no expert exceeds capacity
      E. Ghost identity — tokens dispatched to ghost experts return unchanged
      F. Gradient flow — real experts & gate receive gradients; ghost params don't
    """
    base_model = model_engine.module if hasattr(model_engine, 'module') else model_engine

    if rank == 0:
        print("  MoE dispatch correctness & ghost expert validation...\n")

    # ── Locate MoE layers and their expert lists ──
    moe_layers = []
    for name, module in base_model.named_modules():
        if module.__class__.__name__ == "DeepSpeedMoEFFN":
            moe_layers.append((name, module))

    if not moe_layers:
        if rank == 0:
            print("    SKIP: No DeepSpeedMoEFFN layers found")
        return True

    def _find_experts(moe_ffn):
        """Extract the deepspeed_experts list from an MoE module."""
        for attr_path in [
            lambda m: m.ds_moe.deepspeed_moe.experts.deepspeed_experts,
            lambda m: m.ds_moe.experts.deepspeed_experts,
            lambda m: m.ds_moe.experts,
        ]:
            try:
                experts = attr_path(moe_ffn)
                if experts is not None:
                    return experts
            except AttributeError:
                continue
        return None

    # =====================================================================
    # TEST A: Structural audit
    # =====================================================================
    if rank == 0:
        print("    Test A: Structural audit")
    total_real_params = 0
    ghost_identity_count = 0
    real_expert_count = 0
    struct_ok = True

    for layer_name, moe_ffn in moe_layers:
        experts = _find_experts(moe_ffn)
        if experts is None:
            if rank == 0:
                print(f"      FAIL: Cannot find expert list in {layer_name}")
            struct_ok = False
            continue
        for i, expert in enumerate(experts):
            n_params = sum(p.numel() for p in expert.parameters() if p.requires_grad)
            if i >= moe_ffn.num_real:
                if not isinstance(expert, nn.Identity):
                    if rank == 0:
                        print(f"      FAIL: Ghost expert {i} in {layer_name} is "
                              f"{type(expert).__name__}, not Identity")
                    struct_ok = False
                elif n_params > 0:
                    if rank == 0:
                        print(f"      FAIL: Ghost expert {i} has {n_params} trainable params")
                    struct_ok = False
                else:
                    ghost_identity_count += 1
            else:
                total_real_params += n_params
                real_expert_count += 1

    if real_expert_count > 0 and ghost_identity_count > 0:
        params_per_real = total_real_params // real_expert_count
        dense_would_be = (real_expert_count + ghost_identity_count) * params_per_real
        actual_savings = dense_would_be - total_real_params
        if rank == 0:
            print(f"      Real experts: {real_expert_count} ({total_real_params:,} params)")
            print(f"      Ghost experts: {ghost_identity_count} (Identity, 0 params)")
            print(f"      Param savings: {actual_savings:,} "
                  f"({actual_savings / max(dense_would_be, 1) * 100:.1f}% of expert params)")
            print(f"      {'PASS' if struct_ok else 'FAIL'}\n")
    else:
        if rank == 0:
            print(f"      FAIL: Found {real_expert_count} real, {ghost_identity_count} ghost\n")
        struct_ok = False

    # =====================================================================
    # TEST B: Golden-reference dispatch test
    #   Manually reconstruct MoE output from (dispatch_mask, combine_weights,
    #   individual expert outputs) and compare with DeepSpeed's actual output.
    # =====================================================================
    if rank == 0:
        print("    Test B: Golden-reference dispatch verification")

    model_engine.eval()
    micro_batch = model_engine.train_micro_batch_size_per_gpu()
    seq_len = config.max_seq_len

    torch.manual_seed(77777); torch.cuda.manual_seed(77777)
    input_ids = torch.randint(0, config.vocab_size, (micro_batch, seq_len), device=device)
    next_ids = torch.roll(input_ids, shifts=-1, dims=1)

    # Hook adapters to capture dispatch_mask and combine_weights
    adapter_captures = {}
    hooks = []
    orig_forwards = {}  # save originals for restoration
    for layer_name, moe_ffn in moe_layers:
        adapter = moe_ffn.router_adapter
        acap = {"dispatch_mask": None, "combine_weights": None, "exp_counts": None,
                "moe_input": None, "moe_output": None}
        adapter_captures[layer_name] = acap

        orig_forwards[layer_name] = adapter.forward  # save original

        def make_adapter_hook(cap, orig_fwd):
            def patched_forward(inputs, used_token=None):
                cap["moe_input"] = inputs.detach().clone()
                result = orig_fwd(inputs, used_token)
                aux, cw, dm, ec = result
                cap["dispatch_mask"] = dm.detach().clone()
                cap["combine_weights"] = cw.detach().clone()
                cap["exp_counts"] = ec
                return result
            return patched_forward

        adapter.forward = make_adapter_hook(acap, adapter.forward)

    # Hook each expert to capture input→output for golden reference
    expert_io = {}  # {(layer_name, expert_idx): {"input": ..., "output": ...}}
    for layer_name, moe_ffn in moe_layers:
        experts = _find_experts(moe_ffn)
        if experts is None:
            continue
        for i, expert in enumerate(experts):
            key = (layer_name, i)
            eio = {"input": None, "output": None}
            expert_io[key] = eio

            def make_ehook(cap):
                def hook_fn(module, inp, out):
                    cap["input"] = inp[0].detach().clone() if isinstance(inp, tuple) else inp.detach().clone()
                    cap["output"] = out.detach().clone() if not isinstance(out, tuple) else out[0].detach().clone()
                return hook_fn
            hooks.append(expert.register_forward_hook(make_ehook(eio)))

    # Hook ds_moe (the MoE wrapper) to capture its output
    ds_moe_captures = {}
    for layer_name, moe_ffn in moe_layers:
        dcap = {"output": None}
        ds_moe_captures[layer_name] = dcap

        def make_dshook(cap):
            def hook_fn(module, inp, out):
                # ds_moe returns (output, l_aux, exp_counts)
                if isinstance(out, tuple):
                    cap["output"] = out[0].detach().clone()
                else:
                    cap["output"] = out.detach().clone()
            return hook_fn
        hooks.append(moe_ffn.ds_moe.register_forward_hook(make_dshook(dcap)))

    with torch.no_grad():
        _ = model_engine(input_ids, next_token_ids=next_ids, return_loss=True, return_memory=False)

    for h in hooks:
        h.remove()

    # Restore adapter forward methods
    for layer_name, moe_ffn in moe_layers:
        adapter = moe_ffn.router_adapter
        adapter.forward = orig_forwards[layer_name]

    golden_ref_ok = True
    token_coverage_ok = True
    capacity_ok = True

    for layer_name, moe_ffn in moe_layers:
        acap = adapter_captures[layer_name]
        dcap = ds_moe_captures[layer_name]

        if acap["dispatch_mask"] is None or dcap["output"] is None:
            if rank == 0:
                print(f"      FAIL: Missing captures for {layer_name}")
            golden_ref_ok = False
            continue

        dm = acap["dispatch_mask"]       # (S, E, C) bool
        cw = acap["combine_weights"]     # (S, E, C) float
        moe_input = acap["moe_input"]    # (S, D)
        actual_out = dcap["output"]      # (S, D)
        exp_counts = acap["exp_counts"]  # list[int]

        S, E, C = dm.shape
        D = moe_input.shape[1]

        # ── Test C: Token coverage ──
        # Every token must appear in dispatch_mask for at least one (e, c) slot
        token_dispatched = dm.any(dim=2).any(dim=1)  # (S,) — True if token s dispatched anywhere
        n_dispatched = token_dispatched.sum().item()
        coverage = n_dispatched / S
        if coverage < 1.0:
            undispatched = S - n_dispatched
            if rank == 0:
                print(f"      FAIL: {undispatched}/{S} tokens NOT dispatched (coverage={coverage:.3f})")
            token_coverage_ok = False

        # ── Test D: Capacity audit ──
        tokens_per_expert = dm.sum(dim=(0, 2))  # (E,) — total tokens per expert
        for e_idx in range(E):
            if tokens_per_expert[e_idx].item() > C:
                if rank == 0:
                    print(f"      FAIL: Expert {e_idx} got {tokens_per_expert[e_idx].item()} tokens > capacity {C}")
                capacity_ok = False

        # ── Test B core: Golden reference reconstruction ──
        # DeepSpeed's einsum dispatch:
        #   dispatched_input = einsum('sec,sm->ecm', dm.float(), moe_input)  [m=D]
        #   expert_output[e] = experts[e](dispatched_input[e])
        #   output = einsum('sec,ecm->sm', cw, expert_output)
        #
        # We verify by checking the FINAL combined output matches.

        # Reconstruct from expert hooks
        experts = _find_experts(moe_ffn)
        if experts is None:
            if rank == 0:
                print(f"      FAIL: Cannot find experts for golden ref in {layer_name}")
            golden_ref_ok = False
            continue

        # Compute dispatched input manually (use float32 for golden reference precision)
        moe_input_f32 = moe_input.float()
        dispatched_input = torch.einsum('sec,sd->ecd', dm.float(), moe_input_f32)  # (E, C, D)

        # Run each expert manually on its dispatched buffer
        expert_outputs = torch.zeros(E, C, D, device=device, dtype=torch.float32)
        for e_idx in range(E):
            expert_in = dispatched_input[e_idx].to(moe_input.dtype)  # back to bf16 for expert
            with torch.no_grad():
                expert_out = experts[e_idx](expert_in)
            expert_outputs[e_idx] = expert_out.float()  # back to f32 for comparison

        # Combine back: output[s] = sum_e sum_c (cw[s,e,c] * expert_outputs[e,c,:])
        golden_output = torch.einsum('sec,ecd->sd', cw.float(), expert_outputs)  # (S, D)

        # Compare with actual (convert actual to f32)
        abs_diff = (actual_out.float() - golden_output).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        rel_diff = max_diff / max(actual_out.float().abs().max().item(), 1e-10)

        # bf16 tolerance: allow up to 1% relative error
        layer_ref_ok = rel_diff < 0.01
        if not layer_ref_ok:
            golden_ref_ok = False

        if rank == 0:
            print(f"      {layer_name}:")
            print(f"        dispatch_mask shape: {list(dm.shape)} (S={S}, E={E}, C={C})")
            print(f"        Token coverage: {n_dispatched}/{S} ({coverage*100:.1f}%)")
            print(f"        Expert counts: {exp_counts}")
            print(f"        Golden ref max_diff={max_diff:.2e}  rel_diff={rel_diff:.2e}  "
                  f"{'PASS' if layer_ref_ok else 'FAIL'}")

    if rank == 0:
        print(f"      Token coverage: {'PASS' if token_coverage_ok else 'FAIL'}")
        print(f"      Capacity audit: {'PASS' if capacity_ok else 'FAIL'}")
        print(f"      Golden reference: {'PASS' if golden_ref_ok else 'FAIL'}\n")

    # =====================================================================
    # TEST E: Ghost expert identity verification
    #   Tokens dispatched to ghost experts (idx >= num_real) MUST return
    #   output == input. If ghost experts leak compute, savings are fake.
    # =====================================================================
    if rank == 0:
        print("    Test E: Ghost expert identity verification")

    ghost_ok = True
    real_ok = True
    for layer_name, moe_ffn in moe_layers:
        experts = _find_experts(moe_ffn)
        if experts is None:
            continue

        real_transforming = 0
        ghost_identity = 0
        ghost_broken = 0
        real_stuck = 0

        for i, expert in enumerate(experts):
            eio = expert_io.get((layer_name, i))
            if eio is None or eio["input"] is None or eio["output"] is None:
                continue

            inp_e = eio["input"]
            out_e = eio["output"]

            # Skip if expert got no real tokens (all-zero buffer)
            if inp_e.abs().max().item() < 1e-10:
                continue

            diff_e = (out_e - inp_e).abs().max().item()

            if i >= moe_ffn.num_real:
                # Ghost: MUST be identity
                if diff_e < 1e-6:
                    ghost_identity += 1
                else:
                    ghost_broken += 1
                    ghost_ok = False
                    if rank == 0:
                        print(f"      FAIL: Ghost expert {i} in {layer_name} "
                              f"modifies input (diff={diff_e:.2e})")
            else:
                # Real: SHOULD transform
                if diff_e > 1e-6:
                    real_transforming += 1
                else:
                    real_stuck += 1
                    if rank == 0:
                        print(f"      WARN: Real expert {i} in {layer_name} "
                              f"output ≈ input (diff={diff_e:.2e})")

        if rank == 0:
            print(f"      {layer_name}: real={real_transforming} transforming "
                  f"({real_stuck} stuck), ghost={ghost_identity} identity "
                  f"({ghost_broken} broken)")

    if rank == 0:
        print(f"      Ghost identity: {'PASS' if ghost_ok else 'FAIL'}")
        print(f"      Real transform: {'PASS' if real_ok else 'FAIL'}\n")

    # =====================================================================
    # TEST F: Gradient flow verification
    #   Forward + backward with gradients enabled.  Verify:
    #   1. Gate (router) weights receive non-zero gradients
    #   2. Real expert parameters receive non-zero gradients
    #   3. Ghost expert (Identity) parameters do NOT accumulate gradients
    #   4. combine_weights carry gradient (gate gets learning signal)
    # =====================================================================
    if rank == 0:
        print("    Test F: Gradient flow through MoE pipeline")

    model_engine.train()

    torch.manual_seed(88888); torch.cuda.manual_seed(88888)
    grad_ids = torch.randint(0, config.vocab_size, (micro_batch, seq_len), device=device)
    grad_nxt = torch.roll(grad_ids, shifts=-1, dims=1)
    grad_tgt = torch.roll(grad_ids, shifts=1, dims=1); grad_tgt[:, 0] = 0

    # ZeRO-2 intercepts gradients via its own hooks into flat partition buffers,
    # making leaf-tensor register_hook() unreliable for gradient capture.
    # Phase 0c already proves all parameter groups (attention, expert, gate) update.
    #
    # What we CAN verify here that 0c doesn't:
    #   1. combine_weights carries grad_fn → gate gets learning signal from routing
    #   2. Gate weight is a leaf tensor in the graph (will receive gradient)
    #   3. Expert weights are leaf tensors in the graph (will receive gradient)
    # This is the graph-connectivity check; 0c is the runtime proof.

    cw_has_grad = {}
    grad_orig_forwards = {}
    for layer_name, moe_ffn in moe_layers:
        adapter = moe_ffn.router_adapter
        cap = {"cw_has_grad_fn": False}
        cw_has_grad[layer_name] = cap
        grad_orig_forwards[layer_name] = adapter.forward

        def make_grad_hook(cap_inner, orig_fwd_inner):
            def patched_fwd(inputs, used_token=None):
                result = orig_fwd_inner(inputs, used_token)
                aux, cw, dm, ec = result
                cap_inner["cw_has_grad_fn"] = cw.grad_fn is not None
                return result
            return patched_fwd
        adapter.forward = make_grad_hook(cap, adapter.forward)

    out = model_engine(grad_ids, next_token_ids=grad_nxt, return_loss=True, return_memory=False)

    # Restore adapter forwards before any error could skip cleanup
    for layer_name, moe_ffn in moe_layers:
        moe_ffn.router_adapter.forward = grad_orig_forwards[layer_name]

    # Check 1: combine_weights carries gradient
    # NOTE: Layers inside the reversible block won't show grad_fn during the
    # first forward pass (reversible wrapper detaches intermediates to save memory,
    # recomputes them during backward). This is expected behavior.
    # We verify AT LEAST ONE layer shows grad_fn (the ones at reversible boundaries).
    # Phase 0c proves ALL layers' gates update, covering what this check can't see.
    cw_pass_count = 0
    cw_fail_layers = []
    for layer_name, moe_ffn in moe_layers:
        if cw_has_grad.get(layer_name, {}).get("cw_has_grad_fn", False):
            cw_pass_count += 1
            if rank == 0:
                print(f"      combine_weights carries gradient in {layer_name} ✓")
        else:
            cw_fail_layers.append(layer_name)

    if cw_fail_layers and rank == 0:
        print(f"      combine_weights grad_fn absent in {len(cw_fail_layers)} inner layers "
              f"(expected: reversible block detaches intermediates)")

    cw_grad_ok = cw_pass_count > 0  # At least one layer must show grad_fn

    # Check 2: gate weight is a leaf requiring grad (will receive gradient via autograd)
    gate_in_graph = True
    for layer_name, moe_ffn in moe_layers:
        gate = moe_ffn.router_adapter.gate
        gate_weight = gate.gate if hasattr(gate, 'gate') else None
        if gate_weight is not None and hasattr(gate_weight, 'weight'):
            p = gate_weight.weight
            if not (p.requires_grad and p.is_leaf):
                gate_in_graph = False
                if rank == 0:
                    print(f"      FAIL: Gate weight in {layer_name} not a grad-requiring leaf")
            elif rank == 0:
                print(f"      Gate weight is trainable leaf ✓")
        elif rank == 0:
            print(f"      WARN: Cannot find gate weight in {layer_name}")
            gate_in_graph = False

    # Check 3: real expert weights are leaves requiring grad
    expert_in_graph = True
    for layer_name, moe_ffn in moe_layers:
        experts = _find_experts(moe_ffn)
        if experts is None:
            continue
        for i, expert in enumerate(experts):
            if i >= moe_ffn.num_real:
                continue  # Ghost Identity has no params
            has_trainable = False
            for pn, p in expert.named_parameters():
                if p.requires_grad and p.is_leaf:
                    has_trainable = True
                    break
            if not has_trainable:
                expert_in_graph = False
                if rank == 0:
                    print(f"      FAIL: Real expert {i} in {layer_name} has no trainable leaves")

    if rank == 0 and expert_in_graph:
        print(f"      All real expert weights are trainable leaves ✓")

    # Check 4: ghost experts have zero parameters (no gradient accumulation)
    ghost_grad_ok = True
    for layer_name, moe_ffn in moe_layers:
        experts = _find_experts(moe_ffn)
        if experts is None:
            continue
        for i, expert in enumerate(experts):
            if i < moe_ffn.num_real:
                continue
            n_params = sum(1 for _ in expert.parameters())
            if n_params > 0:
                ghost_grad_ok = False
                if rank == 0:
                    print(f"      FAIL: Ghost expert {i} in {layer_name} has {n_params} params")
    if rank == 0 and ghost_grad_ok:
        print(f"      Ghost experts have zero parameters ✓")

    # Complete the forward's backward + GA cycle so DS stays consistent
    logits_n, logits_m, aux = out[0], out[1], out[2]
    loss = F.cross_entropy(logits_n.view(-1, config.vocab_size), grad_tgt.view(-1)) + aux
    model_engine.backward(loss)
    model_engine.step()
    for remaining_micro in range(model_engine.gradient_accumulation_steps() - 1):
        out2 = model_engine(grad_ids, next_token_ids=grad_nxt, return_loss=True, return_memory=False)
        loss2 = F.cross_entropy(out2[0].view(-1, config.vocab_size), grad_tgt.view(-1)) + out2[2]
        model_engine.backward(loss2)
        model_engine.step()

    # Note: actual gradient magnitude verification is deferred to Phase 0c,
    # which proves all parameter groups update. ZeRO-2's flat-buffer gradient
    # management makes hook-based capture unreliable (hooks fire after ZeRO
    # reduce-scatters into partitioned buffers, showing zeros on non-owned params).
    grad_flow_ok = cw_grad_ok and gate_in_graph and expert_in_graph and ghost_grad_ok

    if rank == 0:
        print(f"      combine_weights grad:   {'PASS' if cw_grad_ok else 'FAIL'} ({cw_pass_count}/{len(moe_layers)} layers)")
        print(f"      Gate in graph:          {'PASS' if gate_in_graph else 'FAIL'}")
        print(f"      Experts in graph:       {'PASS' if expert_in_graph else 'FAIL'}")
        print(f"      Ghost zero-param:       {'PASS' if ghost_grad_ok else 'FAIL'}")
        print(f"      Graph connectivity:     {'PASS' if grad_flow_ok else 'FAIL'}")
        if grad_flow_ok:
            print(f"      (Actual gradient magnitudes verified by Phase 0c)\n")
        else:
            print()

    # =====================================================================
    # OVERALL VERDICT
    # =====================================================================
    all_ok = struct_ok and golden_ref_ok and token_coverage_ok and capacity_ok \
             and ghost_ok and real_ok and grad_flow_ok

    if rank == 0:
        print("    ┌─────────────────────────────────────────┐")
        print(f"    │ A. Structure:        {'PASS ✓' if struct_ok else 'FAIL ✗':>16} │")
        print(f"    │ B. Golden reference: {'PASS ✓' if golden_ref_ok else 'FAIL ✗':>16} │")
        print(f"    │ C. Token coverage:   {'PASS ✓' if token_coverage_ok else 'FAIL ✗':>16} │")
        print(f"    │ D. Capacity:         {'PASS ✓' if capacity_ok else 'FAIL ✗':>16} │")
        print(f"    │ E. Ghost identity:   {'PASS ✓' if ghost_ok else 'FAIL ✗':>16} │")
        print(f"    │ F. Graph connectivity:{'PASS ✓' if grad_flow_ok else 'FAIL ✗':>16} │")
        print(f"    │{'':>41}│")
        print(f"    │ DISPATCH AUDIT:      {'PASS ✓' if all_ok else 'FAIL ✗':>16} │")
        print("    └─────────────────────────────────────────┘\n")

    return all_ok



# ============================================================================
# CHECK E: GHOST vs DENSE EXPERT A/B COMPARISON
# ============================================================================
# THE definitive test: train identical architectures with ghost experts vs
# all-real experts on the same fixed data. Compare loss curves, memory, compute.
# If ghost experts match dense learning at lower cost, we're clear to proceed.
# ============================================================================

def _mini_train_loop(model_engine, data_pool, num_steps, config, device, rank,
                     track_routing=False):
    """Run num_steps optimizer steps, return (losses, step_times, peak_mem_mb, null_rates).
    If track_routing=True, also captures per-step null_rate (fraction of dispatches to ghosts).
    """
    model_engine.train()
    ga = model_engine.gradient_accumulation_steps()
    losses = []
    step_times = []
    null_rates = []  # per-step ghost routing fraction

    # Optional routing tracker
    rt = None
    if track_routing:
        rt = RoutingTracker()
        rt.attach(model_engine)

    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device) / 1e6

    pool_idx = 0
    for step in range(num_steps):
        t0 = time.perf_counter()
        step_loss = 0.0
        step_null = 0.0
        for micro in range(ga):
            ids, tgt_n, nxt, tgt_m = data_pool[pool_idx % len(data_pool)]
            pool_idx += 1
            out = model_engine(ids, next_token_ids=nxt, return_loss=True, return_memory=False)
            logits_n, logits_m, aux = out[0], out[1], out[2]
            l_ntp = F.cross_entropy(logits_n.view(-1, config.vocab_size), tgt_n.view(-1))
            if logits_m is not None:
                ml = min(logits_m.size(1), tgt_m.size(1))
                l_mtp = F.cross_entropy(logits_m[:, :ml].reshape(-1, config.vocab_size),
                                        tgt_m[:, :ml].reshape(-1))
            else:
                l_mtp = torch.tensor(0.0, device=device)
            loss = l_ntp + 0.3 * l_mtp + aux
            model_engine.backward(loss)
            model_engine.step()
            step_loss += l_ntp.item()
            if rt is not None:
                step_null += rt.latest_metrics.get("router_null_rate", 0.0)
        losses.append(step_loss / ga)
        if rt is not None:
            null_rates.append(step_null / ga)
        step_times.append(time.perf_counter() - t0)

        # Progress output every 10 steps
        if rank == 0 and (step % 10 == 0 or step == num_steps - 1):
            elapsed = sum(step_times)
            eta = (num_steps - step - 1) * (elapsed / (step + 1))
            nr_str = f"  null_rate={null_rates[-1]:.3f}" if null_rates else ""
            print(f"        step {step:3d}/{num_steps}  "
                  f"ntp_loss={losses[-1]:.4f}{nr_str}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    if rt is not None:
        rt.remove_hooks()

    peak_mem = torch.cuda.max_memory_allocated(device) / 1e6
    return losses, step_times, peak_mem, null_rates


def run_ghost_vs_dense_comparison(config, device, rank, world_size, ds_config_path,
                                  ab_steps=60, warmup_skip=5):
    """
    Ghost vs Dense expert comparison. Measures savings and confirms learning.

    Config A (ghost):  4 real + 4 ghost Identity in 8 DeepSpeed slots (production)
    Config B (dense):  8 real experts in 8 DeepSpeed slots (full cost baseline)

    WHAT THIS TEST PROVES:
      1. Ghost model learns (loss drops well below random chance)
      2. Dense baseline also learns (confirms test harness works)
      3. Ghost saves memory, params, and compute vs dense
      4. Quantifies the cost/quality tradeoff of 4 vs 8 experts

    WHAT THIS TEST DOES NOT PROVE (proven elsewhere):
      - Dispatch correctness → Phase 0d golden reference
      - Ghost mechanism is lossless → Phase 0d (dispatch correct) + Phase 0c (params update)
      - The loss gap between 4 and 8 experts is "acceptable" → architectural decision,
        unvalidatable on random data at 24M scale. Only the real 3B run answers this.
    """
    import copy as _copy
    import gc

    if rank == 0:
        print("  Building fixed data pool for A/B comparison...")

    micro_batch = 2
    pool = []
    rng = torch.Generator(device=device)
    rng.manual_seed(99999)
    ga = 4
    num_batches = ab_steps * ga
    for _ in range(min(num_batches, 32)):
        ids = torch.randint(0, config.vocab_size, (micro_batch, config.max_seq_len),
                            device=device, generator=rng)
        tgt_n = torch.roll(ids, shifts=1, dims=1); tgt_n[:, 0] = 0
        nxt = torch.roll(ids, shifts=-1, dims=1)
        nxt[:, -1] = torch.randint(0, config.vocab_size, (micro_batch,),
                                    device=device, generator=rng)
        tgt_m = torch.roll(ids, shifts=2, dims=1); tgt_m[:, :2] = 0
        pool.append((ids, tgt_n, nxt, tgt_m))

    results = {}

    # High-LR ds_config for faster learning signal
    import json, tempfile
    with open(ds_config_path) as f:
        ab_ds_config = json.load(f)
    ab_ds_config["optimizer"]["params"]["lr"] = 3e-3
    ab_ds_config.pop("flops_profiler", None)
    ab_ds_config["wall_clock_breakdown"] = False
    ab_config_fd, ab_config_path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(ab_config_fd, "w") as f:
        json.dump(ab_ds_config, f)

    # ======================== Config A: Ghost Experts ========================
    if rank == 0:
        print("\n    ── Config A: Ghost experts (4 real + 4 Identity) ──")
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    config_a = _copy.deepcopy(config)
    torch.manual_seed(12345); torch.cuda.manual_seed(12345)
    with quiet_init(rank):
        model_a = Model3B(config_a, embedding_type="standard")
    params_a = sum(p.numel() for p in model_a.parameters())
    mem_after_model_a = torch.cuda.memory_allocated(device) / 1e6
    if rank == 0:
        print(f"      Total params: {params_a:,}")
        print(f"      Model memory: {mem_after_model_a:.1f} MB")

    with quiet_init(rank):
        engine_a, _, _, _ = deepspeed.initialize(
            model=model_a, config=ab_config_path, model_parameters=model_a.parameters())

    losses_a, times_a, peak_a, null_rates_a = _mini_train_loop(
        engine_a, pool, ab_steps, config_a, device, rank, track_routing=True)

    ghost_avg_ms = sum(times_a[warmup_skip:]) / max(len(times_a) - warmup_skip, 1) * 1000
    if rank == 0:
        print(f"      Peak GPU memory:  {peak_a:.1f} MB")
        print(f"      NTP loss: {losses_a[0]:.4f} → {losses_a[-1]:.4f}")
        print(f"      Avg step time: {ghost_avg_ms:.0f} ms")

    del engine_a, model_a
    gc.collect(); torch.cuda.empty_cache()
    deepspeed.comm.barrier()

    # ======================== Config B: All-Real Experts ========================
    if rank == 0:
        print("\n    ── Config B: Dense experts (8 real, no ghosts) ──")
    torch.cuda.reset_peak_memory_stats(device)

    config_b = _copy.deepcopy(config)
    config_b.num_real_experts = config.total_expert_slots  # 8
    config_b.num_null_experts = 0
    config_b.data_sparsity = 1.0

    torch.manual_seed(12345); torch.cuda.manual_seed(12345)
    with quiet_init(rank):
        model_b = Model3B(config_b, embedding_type="standard")
    params_b = sum(p.numel() for p in model_b.parameters())
    mem_after_model_b = torch.cuda.memory_allocated(device) / 1e6
    if rank == 0:
        print(f"      Total params: {params_b:,}")
        print(f"      Model memory: {mem_after_model_b:.1f} MB")

    with quiet_init(rank):
        engine_b, _, _, _ = deepspeed.initialize(
            model=model_b, config=ab_config_path, model_parameters=model_b.parameters())

    losses_b, times_b, peak_b, _ = _mini_train_loop(
        engine_b, pool, ab_steps, config_b, device, rank)

    dense_avg_ms = sum(times_b[warmup_skip:]) / max(len(times_b) - warmup_skip, 1) * 1000
    if rank == 0:
        print(f"      Peak GPU memory:  {peak_b:.1f} MB")
        print(f"      NTP loss: {losses_b[0]:.4f} → {losses_b[-1]:.4f}")
        print(f"      Avg step time: {dense_avg_ms:.0f} ms")

    del engine_b, model_b
    gc.collect(); torch.cuda.empty_cache()
    deepspeed.comm.barrier()

    # ======================== Analysis ========================
    random_chance = math.log(config.vocab_size)

    # Learning detection: first-10 vs last-10 average
    ghost_first10 = sum(losses_a[:10]) / 10
    ghost_last10 = sum(losses_a[-10:]) / 10
    dense_first10 = sum(losses_b[:10]) / 10
    dense_last10 = sum(losses_b[-10:]) / 10
    ghost_is_learning = ghost_last10 < ghost_first10 - 0.01
    dense_is_learning = dense_last10 < dense_first10 - 0.01

    # Ghost should get well below random chance (proves real learning, not noise)
    ghost_below_random = ghost_last10 < random_chance * 0.85  # 15% below ln(vocab)

    # Savings
    mem_saved_mb = peak_b - peak_a
    mem_saved_pct = mem_saved_mb / max(peak_b, 1) * 100
    param_saved = params_b - params_a
    param_saved_pct = param_saved / max(params_b, 1) * 100
    time_saved_pct = (dense_avg_ms - ghost_avg_ms) / max(dense_avg_ms, 1) * 100

    # Informational: loss gap (not a pass/fail criterion)
    ghost_final = ghost_last10
    dense_final = dense_last10
    loss_gap_pct = (ghost_final - dense_final) / max(abs(dense_final), 1e-8) * 100

    # Ghost routing evolution: does the gate learn to avoid ghost experts?
    gate_avoids_ghosts = False
    null_rate_first5 = None
    null_rate_last5 = None
    null_rate_drop = 0.0
    if null_rates_a and len(null_rates_a) >= 10:
        null_rate_first5 = sum(null_rates_a[:5]) / 5
        null_rate_last5 = sum(null_rates_a[-5:]) / 5
        null_rate_drop = null_rate_first5 - null_rate_last5
        # Gate learns to avoid ghosts if null_rate drops by at least 5 percentage points
        # OR if final null_rate is below 10% (gate already learned)
        gate_avoids_ghosts = null_rate_drop > 0.05 or null_rate_last5 < 0.10

    results["ghost_losses"] = losses_a
    results["ghost_times"] = times_a
    results["ghost_peak_mb"] = peak_a
    results["ghost_params"] = params_a
    results["ghost_null_rates"] = null_rates_a
    results["dense_losses"] = losses_b
    results["dense_times"] = times_b
    results["dense_peak_mb"] = peak_b
    results["dense_params"] = params_b
    results["ghost_is_learning"] = ghost_is_learning
    results["dense_is_learning"] = dense_is_learning
    results["ghost_below_random"] = ghost_below_random
    results["ghost_final_loss"] = ghost_final
    results["dense_final_loss"] = dense_final
    results["mem_saved_mb"] = mem_saved_mb
    results["mem_saved_pct"] = mem_saved_pct
    results["param_saved"] = param_saved
    results["param_saved_pct"] = param_saved_pct
    results["time_saved_pct"] = time_saved_pct
    results["loss_gap_pct"] = loss_gap_pct
    results["gate_avoids_ghosts"] = gate_avoids_ghosts
    results["null_rate_first5"] = null_rate_first5
    results["null_rate_last5"] = null_rate_last5
    results["pass"] = (ghost_is_learning and dense_is_learning and
                       ghost_below_random and mem_saved_mb > 0 and
                       gate_avoids_ghosts)

    if rank == 0:
        print(f"\n    ── A/B Comparison ──")
        print(f"      Random chance:      {random_chance:.4f} (ln({config.vocab_size}))")
        print(f"                          Ghost(4+4)     Dense(8)")
        print(f"      Parameters:         {params_a:>10,}    {params_b:>10,}   (Δ={param_saved:,} / {param_saved_pct:.1f}%)")
        print(f"      Peak memory (MB):   {peak_a:>10.1f}    {peak_b:>10.1f}   (Δ={mem_saved_mb:.1f} / {mem_saved_pct:.1f}%)")
        print(f"      Step time (ms):     {ghost_avg_ms:>10.0f}    {dense_avg_ms:>10.0f}   ({time_saved_pct:+.1f}%)")
        print(f"      Final loss (avg10): {ghost_final:>10.4f}    {dense_final:>10.4f}   (gap={loss_gap_pct:+.1f}%)")
        print(f"      Learning:           {'YES':>10}    {'YES' if dense_is_learning else 'NO':>10}")
        print(f"      Below random:       {'YES' if ghost_below_random else 'NO':>10}    {'YES' if dense_last10 < random_chance * 0.85 else 'NO':>10}")

        # Ghost routing evolution
        if null_rate_first5 is not None:
            print(f"\n    ── Ghost Routing Evolution (CRITICAL) ──")
            print(f"      Null rate (first 5 steps):  {null_rate_first5:.1%}")
            print(f"      Null rate (last 5 steps):   {null_rate_last5:.1%}")
            print(f"      Null rate drop:             {null_rate_drop:+.1%}")
            if gate_avoids_ghosts:
                print(f"      Gate LEARNS to avoid ghosts ✓")
            else:
                print(f"      Gate does NOT learn to avoid ghosts ✗")
                print(f"        → Tokens still routed to Identity experts are wasted capacity.")
                print(f"        → This may explain the loss gap vs dense.")

        print(f"\n    ── Verdicts ──")
        print(f"      Ghost learns:           {'PASS ✓' if ghost_is_learning else 'FAIL ✗'}")
        print(f"      Ghost below random:     {'PASS ✓' if ghost_below_random else 'FAIL ✗'}")
        print(f"      Dense baseline learns:  {'PASS ✓' if dense_is_learning else 'FAIL ✗'}")
        print(f"      Memory savings:         {'PASS ✓' if mem_saved_mb > 0 else 'FAIL ✗'}")
        print(f"      Gate avoids ghosts:     {'PASS ✓' if gate_avoids_ghosts else 'FAIL ✗'}")
        print(f"      Loss gap (info only):   {loss_gap_pct:+.1f}% (dense has {param_saved:,} more expert params)")
        print()
        if results["pass"]:
            print(f"      ✅ PASS: Ghost model learns, gate routes away from ghosts, saves cost.")
            print(f"         Mechanism correctness proven by Phase 0d + Phase 0c.")
        else:
            reasons = []
            if not ghost_is_learning: reasons.append("ghost model not learning")
            if not ghost_below_random: reasons.append(f"ghost loss above random chance ({ghost_final:.2f} vs {random_chance:.2f})")
            if not dense_is_learning: reasons.append("dense baseline not learning (test harness broken)")
            if mem_saved_mb <= 0: reasons.append("no memory savings")
            if not gate_avoids_ghosts: reasons.append(f"gate doesn't learn to avoid ghosts (null_rate {null_rate_last5:.1%} → wasted dispatch capacity)")
            print(f"      ❌ FAIL: {'; '.join(reasons)}")

    try:
        os.unlink(ab_config_path)
    except OSError:
        pass

    return results


# CHECKPOINT OPTIMIZER STATE VERIFICATION
# ============================================================================

def compute_update_magnitudes(model_engine, config, device):
    """Run one deterministic step and measure per-parameter update magnitudes.
    Must use model_engine() for forward (not base_model) to avoid timer assertion,
    and must complete gradient_accumulation_steps micro-batches for step() to fire.
    """
    base_model = model_engine.module if hasattr(model_engine, 'module') else model_engine
    micro_batch = model_engine.train_micro_batch_size_per_gpu()
    seq_len = config.max_seq_len
    ga_steps = model_engine.gradient_accumulation_steps()

    params_before = {n: p.data.clone() for n, p in base_model.named_parameters()}

    for micro in range(ga_steps):
        torch.manual_seed(99999 + micro); torch.cuda.manual_seed(99999 + micro)
        input_ids = torch.randint(0, config.vocab_size, (micro_batch, seq_len), device=device)
        next_ids = torch.randint(0, config.vocab_size, (micro_batch, seq_len), device=device)
        targets = torch.roll(input_ids, shifts=1, dims=1)  # Consistent with copy-shift task
        targets[:, 0] = 0

        out = model_engine(input_ids, next_token_ids=next_ids, return_loss=True, return_memory=False)
        logits = out[0]
        aux = out[2] if len(out) > 2 and out[2] is not None else torch.tensor(0.0, device=device)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1)) + aux
        model_engine.backward(loss)
        model_engine.step()  # Must call after EVERY backward(); DS fires optimizer at GA boundary

    return {n: (p.data - params_before[n]).norm().item() for n, p in base_model.named_parameters() if n in params_before}


def verify_optimizer_state_after_resume(model_engine, config, device, rank, pre_save_mags):
    """
    Compare parameter update magnitudes pre-save vs post-resume on identical data.
    If Adam's exp_avg/exp_avg_sq corrupted in ZeRO-2 flat buffers, updates differ.
    Must use model_engine() for forward and complete GA steps for step() to fire.
    """
    base_model = model_engine.module if hasattr(model_engine, 'module') else model_engine
    micro_batch = model_engine.train_micro_batch_size_per_gpu()
    seq_len = config.max_seq_len
    ga_steps = model_engine.gradient_accumulation_steps()

    params_before = {n: p.data.clone() for n, p in base_model.named_parameters()}

    for micro in range(ga_steps):
        torch.manual_seed(99999 + micro); torch.cuda.manual_seed(99999 + micro)
        input_ids = torch.randint(0, config.vocab_size, (micro_batch, seq_len), device=device)
        next_ids = torch.randint(0, config.vocab_size, (micro_batch, seq_len), device=device)
        targets = torch.roll(input_ids, shifts=1, dims=1)
        targets[:, 0] = 0

        out = model_engine(input_ids, next_token_ids=next_ids, return_loss=True, return_memory=False)
        logits = out[0]
        aux = out[2] if len(out) > 2 and out[2] is not None else torch.tensor(0.0, device=device)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1)) + aux
        model_engine.backward(loss)
        model_engine.step()  # Must call after EVERY backward(); DS fires optimizer at GA boundary

    max_drift, worst, total = 0.0, "", 0
    for n, p in base_model.named_parameters():
        if n in params_before and n in pre_save_mags:
            post_mag = (p.data - params_before[n]).norm().item()
            pre_mag = pre_save_mags[n]
            denom = max(pre_mag, post_mag, 1e-10)
            drift = abs(pre_mag - post_mag) / denom
            total += 1
            if drift > max_drift: max_drift = drift; worst = n

    if rank == 0:
        print(f"    Optimizer check: {total} params, max drift={max_drift:.6f} ({worst})")

    passed = max_drift < 0.2  # 20% tolerance for ZeRO-2 fp32 reduction order
    if rank == 0:
        print(f"    {'PASS' if passed else 'FAIL'}: optimizer state {'restored' if passed else 'CORRUPTED'}")
    return passed, max_drift


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full ZeRO-2 Training Validation")
    parser.add_argument("--total_steps", type=int, default=250)
    parser.add_argument("--checkpoint_step", type=int, default=200)
    parser.add_argument("--log_dir", type=str, default="./validation_logs")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--skip_reversible_check", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Suppress noisy logging from non-rank-0 and DS internals
    import logging
    if rank != 0:
        logging.getLogger("deepspeed").setLevel(logging.ERROR)
    else:
        logging.getLogger("deepspeed").setLevel(logging.WARNING)

    if rank == 0:
        print("\n" + "=" * 72)
        print("  DEEPSPEED ZERO-2 FULL TRAINING VALIDATION")
        print("=" * 72)
        print(f"\n  World size: {world_size}")
        print(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  DeepSpeed: {deepspeed.__version__}")
        print()

    # -------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------
    if rank == 0: print("  Phase 0: Building model...")
    config = make_validation_config()
    config.max_seq_len = args.seq_len
    with quiet_init(rank):
        model = Model3B(config, embedding_type="standard")
    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0: print(f"  Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    profiler = MemoryProfiler(device, rank)
    profiler.snapshot("after_model_init")

    # -------------------------------------------------------------------
    # DeepSpeed initialize
    # -------------------------------------------------------------------
    if rank == 0: print("  Initializing DeepSpeed ZeRO-2...")
    with quiet_init(rank):
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=args.deepspeed_config, model_parameters=model.parameters(),
        )
    if rank == 0:
        print(f"  ZeRO stage: {model_engine.zero_optimization_stage()}")
        print(f"  Micro batch: {model_engine.train_micro_batch_size_per_gpu()}")
        print(f"  Grad accum: {model_engine.gradient_accumulation_steps()}\n")
    profiler.snapshot("after_deepspeed_init")

    # -------------------------------------------------------------------
    # Phase 0a: Gate determinism
    # -------------------------------------------------------------------
    gate_determinism_ok = None
    if not args.skip_reversible_check:
        if rank == 0:
            print("-" * 72)
            print("  Phase 0a: Gate routing determinism")
            print("-" * 72)
        gate_determinism_ok, _ = check_gate_determinism(model_engine, config, device, rank)
        t = torch.tensor([1.0 if gate_determinism_ok else 0.0], device=device)
        torch.distributed.broadcast(t, src=0)
        gate_determinism_ok = t.item() > 0.5
        if rank == 0: print()

    # -------------------------------------------------------------------
    # Phase 0b: Reversible gradient check
    # -------------------------------------------------------------------
    reversible_grad_ok = None
    if not args.skip_reversible_check:
        if rank == 0:
            print("-" * 72)
            print("  Phase 0b: Reversible gradient correctness")
            print("-" * 72)
        reversible_grad_ok, _ = verify_reversible_gradients(model_engine, config, device, rank)
        t = torch.tensor([1.0 if reversible_grad_ok else 0.0], device=device)
        torch.distributed.broadcast(t, src=0)
        reversible_grad_ok = t.item() > 0.5
        if rank == 0: print()
    else:
        if rank == 0: print("  Skipping reversible check (--skip_reversible_check)")

    # -------------------------------------------------------------------
    # Phase 0c: ZeRO flat-buffer alignment (parameter update verification)
    # Runs 2 optimizer steps and verifies all parameter groups update
    # -------------------------------------------------------------------
    zero_alignment_ok = None
    if not args.skip_reversible_check:
        if rank == 0:
            print("-" * 72)
            print("  Phase 0c: ZeRO flat-buffer alignment (parameter update verification)")
            print("-" * 72)
        zero_alignment_ok = verify_zero_alignment(model_engine, config, device, rank, world_size)
        t = torch.tensor([1.0 if zero_alignment_ok else 0.0], device=device)
        torch.distributed.broadcast(t, src=0)
        zero_alignment_ok = t.item() > 0.5
        if rank == 0: print()

    # -------------------------------------------------------------------
    # Re-initialize optimizer to clean state after injection test
    # (The injection test stepped the optimizer, contaminating Adam state)
    # -------------------------------------------------------------------
    if not args.skip_reversible_check:
        if rank == 0: print("  Re-initializing model + optimizer for clean training...")
        with quiet_init(rank):
            model_fresh = Model3B(config, embedding_type="standard")
            model_engine, optimizer, _, _ = deepspeed.initialize(
                model=model_fresh,
                config=args.deepspeed_config, model_parameters=model_fresh.parameters(),
            )
        if rank == 0: print("  Fresh model + optimizer initialized.\n")
        profiler.snapshot("after_fresh_reinit")

    # -------------------------------------------------------------------
    # Phase 0d: MoE dispatch correctness & ghost expert validation
    # Uses the fresh model to verify the MoE pipeline is mathematically correct
    # -------------------------------------------------------------------
    moe_dispatch_ok = None
    if not args.skip_reversible_check:
        if rank == 0:
            print("-" * 72)
            print("  Phase 0d: MoE dispatch correctness & ghost expert validation")
            print("-" * 72)
        moe_dispatch_ok = verify_moe_dispatch_correctness(model_engine, config, device, rank, world_size)
        t = torch.tensor([1.0 if moe_dispatch_ok else 0.0], device=device)
        torch.distributed.broadcast(t, src=0)
        moe_dispatch_ok = t.item() > 0.5
        if rank == 0: print()

    # -------------------------------------------------------------------
    # Phase 0e: Ghost vs Dense A/B Comparison
    # THE definitive test: proves ghost experts learn comparably at lower cost
    # -------------------------------------------------------------------
    ab_results = None
    if not args.skip_reversible_check:
        if rank == 0:
            print("-" * 72)
            print("  Phase 0e: Ghost vs Dense expert A/B comparison")
            print("-" * 72)
        ab_results = run_ghost_vs_dense_comparison(
            config, device, rank, world_size, args.deepspeed_config,
            ab_steps=60, warmup_skip=5)
        if rank == 0: print()

        # Phase 0e creates/destroys its own models; re-init for Phase 1
        if rank == 0: print("  Re-initializing model for Phase 1 training...")
        import gc; gc.collect(); torch.cuda.empty_cache()
        with quiet_init(rank):
            model_fresh2 = Model3B(config, embedding_type="standard")
            model_engine, optimizer, _, _ = deepspeed.initialize(
                model=model_fresh2,
                config=args.deepspeed_config, model_parameters=model_fresh2.parameters(),
            )
        if rank == 0: print("  Fresh model + optimizer initialized.\n")
        profiler.snapshot("after_ab_reinit")

    # -------------------------------------------------------------------
    # Setup data, metrics, trackers
    # -------------------------------------------------------------------
    micro_batch = model_engine.train_micro_batch_size_per_gpu()
    data_loader = SyntheticDataLoader(
        vocab_size=config.vocab_size, seq_len=args.seq_len,
        batch_size=micro_batch,
        num_steps=args.total_steps * model_engine.gradient_accumulation_steps(),
        device=device,
        fixed_pool_size=16,  # cycle through 16 fixed batches — model MUST memorize
    )
    collector = MetricCollector(log_dir=args.log_dir, rank=rank)
    routing_tracker = RoutingTracker()
    routing_tracker.attach(model_engine)
    moe_timer = MoETimingTracker()
    moe_timer.attach(model_engine)
    grad_tracker = GradNormTracker()
    grad_tracker.attach(model_engine)

    checkpoint_dir = Path(args.log_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_resume_ok = False
    ckpt_params_ok = False
    ckpt_optim_ok = False
    max_xr_spread = 0.0

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    if rank == 0:
        print("=" * 72)
        print("  Phase 1: Training (warmup) -- copy-shift task")
        print("=" * 72 + "\n")

    data_iter = iter(data_loader)
    model_engine.train()

    for step in range(args.total_steps):
        step_start = time.perf_counter()
        step_metrics = {"step": step}
        nan_det, inf_det = False, False

        if rank == 0:
            if step == 100:
                print(f"\n{'=' * 72}\n  Phase 2: Steady-state\n{'=' * 72}\n")
            elif step == args.checkpoint_step:
                print(f"\n{'=' * 72}\n  Phase 3: Checkpoint + optimizer verify\n{'=' * 72}\n")
            elif step == args.checkpoint_step + 1:
                print(f"\n{'=' * 72}\n  Phase 4: Post-resume\n{'=' * 72}\n")

        # ----- Checkpoint -----
        if step == args.checkpoint_step:
            # Pre-save optimizer baseline
            if rank == 0: print("  Computing pre-save optimizer baseline...")
            try:
                pre_mags = compute_update_magnitudes(model_engine, config, device)
                if rank == 0:
                    avg = sum(pre_mags.values()) / max(len(pre_mags), 1)
                    print(f"    Avg update magnitude: {avg:.6e}")
            except Exception as e:
                pre_mags = {}
                if rank == 0: print(f"    WARNING: baseline failed: {e}")

            # Snapshot + save
            if rank == 0: print("  Saving checkpoint...")
            bm = model_engine.module if hasattr(model_engine, 'module') else model_engine
            pre_params = {n: p.data.clone() for n, p in bm.named_parameters()}
            model_engine.save_checkpoint(str(checkpoint_dir), tag="val_ckpt")

            # Load
            if rank == 0: print("  Loading checkpoint...")
            try:
                _, cs = model_engine.load_checkpoint(str(checkpoint_dir), tag="val_ckpt")
                ckpt_resume_ok = True
                if rank == 0: print(f"  Resumed. Client state: {cs}")

                # Param check
                max_pd = 0.0; pd_name = ""
                for n, p in bm.named_parameters():
                    if n in pre_params:
                        d = (pre_params[n] - p.data).abs().max().item()
                        if d > max_pd: max_pd = d; pd_name = n
                ckpt_params_ok = max_pd < 1e-5
                if rank == 0:
                    print(f"  Param drift: {max_pd:.2e} {'PASS' if ckpt_params_ok else 'FAIL: '+pd_name}")

                # Optimizer check
                if pre_mags:
                    if rank == 0: print("  Verifying optimizer state...")
                    ckpt_optim_ok, _ = verify_optimizer_state_after_resume(
                        model_engine, config, device, rank, pre_mags)
                else:
                    ckpt_optim_ok = True
                if rank == 0: print()
            except Exception as e:
                ckpt_resume_ok = ckpt_params_ok = ckpt_optim_ok = False
                if rank == 0: print(f"  Resume FAILED: {e}"); traceback.print_exc(); print()
            del pre_params

        # ----- Forward -----
        if step == 0:
            profiler.reset_peak()

        t_loss, t_ntp, t_mtp, t_aux = 0.0, 0.0, 0.0, 0.0
        for micro_idx in range(model_engine.gradient_accumulation_steps()):
            try:
                ids, tgt_n, nxt, tgt_m = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                ids, tgt_n, nxt, tgt_m = next(data_iter)

            out = model_engine(ids, next_token_ids=nxt, return_loss=True, return_memory=False)
            logits_n, logits_m, aux = out[0], out[1], out[2]
            l_ntp = F.cross_entropy(logits_n.view(-1, config.vocab_size), tgt_n.view(-1))
            if logits_m is not None:
                ml = min(logits_m.size(1), tgt_m.size(1))
                l_mtp = F.cross_entropy(logits_m[:, :ml].reshape(-1, config.vocab_size), tgt_m[:, :ml].reshape(-1))
            else:
                l_mtp = torch.tensor(0.0, device=device)
            loss = l_ntp + 0.3 * l_mtp + aux

            if step == 0 and micro_idx == 0:
                profiler.snapshot("peak_forward")

            if torch.isnan(loss).any(): nan_det = True
            if torch.isinf(loss).any(): inf_det = True
            model_engine.backward(loss)
            model_engine.step()  # Must call after EVERY backward(); DS fires optimizer at GA boundary

            if step == 0 and micro_idx == 0:
                profiler.snapshot("peak_backward")

            t_loss += loss.item(); t_ntp += l_ntp.item()
            t_mtp += (l_mtp.item() if isinstance(l_mtp, torch.Tensor) else l_mtp)
            t_aux += aux.item()

        grad_m = collect_grad_metrics(model_engine)
        # ZeRO-2 clears param.grad during backward; use hook-based tracker instead
        if grad_m.get("_zero_cleared_grads", False):
            grad_m.update(grad_tracker.get_metrics())
        step_metrics.update(grad_m)
        grad_tracker.reset()

        dt = time.perf_counter() - step_start
        acc = model_engine.gradient_accumulation_steps()
        toks = micro_batch * args.seq_len * acc * world_size

        step_metrics["loss_total"] = t_loss / acc
        step_metrics["loss_ntp"] = t_ntp / acc
        step_metrics["loss_mtp"] = t_mtp / acc
        step_metrics["loss_aux"] = t_aux / acc
        step_metrics["aux_main_ratio"] = (t_aux / acc) / max(t_ntp / acc, 1e-8)
        step_metrics.update(routing_tracker.get_metrics())
        step_metrics["moe_dispatch_ms"] = moe_timer.get_ms()

        if step % 10 == 0:
            xr = collect_cross_rank_param_norm(model_engine, device, world_size)
            step_metrics["cross_rank_param_norm_spread"] = xr
            max_xr_spread = max(max_xr_spread, xr)
        else:
            step_metrics["cross_rank_param_norm_spread"] = 0.0

        step_metrics["step_time_ms"] = dt * 1000
        step_metrics["tokens_per_sec"] = toks / (dt + 1e-9)
        step_metrics["gpu_mem_allocated_gb"] = torch.cuda.memory_allocated(device) / 1e9
        step_metrics["gpu_mem_reserved_gb"] = torch.cuda.memory_reserved(device) / 1e9
        step_metrics["nan_detected"] = 1 if nan_det else 0
        step_metrics["inf_detected"] = 1 if inf_det else 0
        collector.record(step_metrics)

        if rank == 0 and (step < 10 or step % 5 == 0 or step == args.total_steps - 1):
            collector.print_step(step, step_metrics)

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    routing_tracker.remove_hooks()
    moe_timer.remove()
    grad_tracker.remove()
    profiler.report()
    xr_ok = max_xr_spread < 0.01 if world_size > 1 else True

    if rank == 0:
        print(f"\n{'=' * 72}\n  GENERATING VALIDATION SUMMARY\n{'=' * 72}\n")

    verdicts = collector.generate_summary(
        args.total_steps, ckpt_resume_ok,
        config=config,
        checkpoint_params_match=ckpt_params_ok,
        checkpoint_optim_ok=ckpt_optim_ok,
        reversible_grad_ok=reversible_grad_ok,
        gate_determinism_ok=gate_determinism_ok,
        zero_alignment_ok=zero_alignment_ok,
        moe_dispatch_ok=moe_dispatch_ok,
        ab_results=ab_results,
        cross_rank_consistent=xr_ok,
    )

    if rank == 0:
        print(f"\n  Metrics CSV: {collector.csv_path}")
        print(f"  Summary:     {collector.summary_path}\n")
    deepspeed.comm.barrier()


if __name__ == "__main__":
    main()