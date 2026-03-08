#!/usr/bin/env python3
"""
Benchmark OPUS sweep: baseline + 3 OPUS configs.
Modifies the YAML config in-place, runs deepspeed, collects metrics.

Usage: python3 benchmark_opus_sweep.py
"""
import json
import os
import subprocess
import sys
import time
import yaml
import copy
import shutil

TEST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(TEST_ROOT, "configs", "test_1b_nonrev_opus_4096_10steps.yaml")
RESULTS_BASE = os.path.join(TEST_ROOT, "results", "opus_sweep")
CODE_DIR = os.path.join(TEST_ROOT, "code")

# Configs to test: (name, opus_enabled, candidate_multiplier, n_proxy)
# candidate_multiplier * 8 GPUs = global raw candidates
# n_proxy * 8 GPUs = global proxy count
CONFIGS = [
    ("baseline_no_opus",  False, 8, 4),
    ("opus_64raw_32proxy", True,  8, 4),   # 64 candidates, 32 proxy
    ("opus_64raw_16proxy", True,  8, 2),   # 64 candidates, 16 proxy
    ("opus_56raw_8proxy",  True,  7, 1),   # 56 candidates, 8 proxy
]

MAX_STEPS = 10


def write_config(name, opus_enabled, candidate_multiplier, n_proxy):
    """Read base config, modify OPUS settings, write back."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    cfg["training"]["max_train_steps"] = MAX_STEPS
    cfg["training"]["profile_steps"] = []

    out_dir = os.path.join(RESULTS_BASE, name)
    os.makedirs(out_dir, exist_ok=True)
    cfg["training"]["metrics_jsonl_path"] = os.path.join(out_dir, "metrics.jsonl")
    cfg["checkpoint"]["output_dir"] = os.path.join(out_dir, "checkpoints")

    cfg["opus"]["enabled"] = opus_enabled
    cfg["opus"]["candidate_multiplier"] = candidate_multiplier
    cfg["opus"]["n_proxy"] = n_proxy
    cfg["opus"]["train_batch"] = 1

    # For baseline, set multiplier=1 so dataloader BS matches training BS
    if not opus_enabled:
        cfg["opus"]["candidate_multiplier"] = 1

    tmp_cfg = os.path.join(out_dir, "config.yaml")
    with open(tmp_cfg, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return tmp_cfg, out_dir


def run_benchmark(cfg_path, out_dir, name):
    """Run deepspeed with the given config, capture output."""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{TEST_ROOT}:{env.get('PYTHONPATH', '')}"
    env["TORCHDYNAMO_DISABLE"] = "1"
    env["T19_STEP_CUDA_SYNC"] = "1"
    env["T19_STEP_GC_COLLECT"] = "1"
    env["T19_STEP_EMPTY_CACHE"] = "1"
    env["T19_STEP_IPC_COLLECT"] = "0"
    env["T19_ZERO3_RELEASE_EVERY"] = "1"
    env["T19_ZERO3_FORCE_CLEAR_CONTAINERS"] = "0"
    env["T19_CLEAR_ROUTER_CACHE_EVERY"] = "1"
    env["T19_TRACK_CUDA_MEMORY"] = "1"
    env["T19_REV_CKPT_USE_REENTRANT"] = "0"

    log_path = os.path.join(out_dir, "train.log")
    cmd = [
        "deepspeed", "--num_gpus=8",
        "main.py", "--config", cfg_path,
    ]

    print(f"\n{'='*70}")
    print(f"  RUNNING: {name}")
    print(f"  Config:  {cfg_path}")
    print(f"  Log:     {log_path}")
    print(f"{'='*70}\n")

    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            cmd, cwd=CODE_DIR, env=env,
            stdout=log_f, stderr=subprocess.STDOUT,
            timeout=600,
        )

    if proc.returncode != 0:
        print(f"  FAILED (exit {proc.returncode}) — see {log_path}")
        return None

    return log_path


def parse_metrics(out_dir):
    """Parse metrics.jsonl and extract steady-state (last 5 steps) averages."""
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    if not os.path.exists(metrics_path):
        return None

    rows = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    train_rows = [r for r in rows if r.get("phase", "").startswith("train")]
    if len(train_rows) < 3:
        return None

    # Use last 5 steps for steady-state
    ss = train_rows[-5:] if len(train_rows) >= 5 else train_rows[-3:]

    def avg(key):
        vals = [r[key] for r in ss if key in r and r[key] is not None]
        return sum(vals) / len(vals) if vals else 0

    result = {
        "steps": len(train_rows),
        "loss": avg("loss_ntp"),
        "tok_s": avg("tokens_per_sec"),
        "dt_ms": avg("dt_ms"),
        "vram_gb": avg("gpu_alloc_gb"),
    }

    # OPUS phase times
    phase_keys = [
        "phase_scoring_fwd_ms", "phase_scoring_bwd_ms", "phase_zero_grad_ms",
        "phase_boltzmann_ms", "phase_scoring_total_ms",
        "phase_train_fwd_ms", "phase_train_ce_ms", "phase_train_bwd_ms",
        "phase_train_step_ms", "phase_train_total_ms",
        "phase_precond_refresh_ms", "phase_proxy_sample_ms",
    ]
    for k in phase_keys:
        v = avg(k)
        if v > 0:
            result[k] = v

    return result


def main():
    os.makedirs(RESULTS_BASE, exist_ok=True)

    all_results = {}

    for name, opus_enabled, cand_mult, n_proxy in CONFIGS:
        cfg_path, out_dir = write_config(name, opus_enabled, cand_mult, n_proxy)
        log_path = run_benchmark(cfg_path, out_dir, name)

        if log_path:
            metrics = parse_metrics(out_dir)
            all_results[name] = metrics
        else:
            all_results[name] = None

    # Print summary
    print(f"\n\n{'='*90}")
    print(f"  OPUS BENCHMARK SUMMARY  (8x A100-40GB, 1B Non-Rev, SL=4096, 10 steps)")
    print(f"{'='*90}")
    print(f"{'Config':<25} {'tok/s':>8} {'dt_ms':>8} {'loss':>8} {'VRAM':>6}"
          f" {'scr_fwd':>8} {'scr_bwd':>8} {'zero_g':>7} {'boltz':>6}"
          f" {'tr_fwd':>7} {'tr_ce':>6} {'tr_bwd':>7} {'tr_stp':>7}"
          f" {'scr_tot':>8} {'tr_tot':>7}")
    print("-" * 155)

    for name, _, _, _ in CONFIGS:
        m = all_results.get(name)
        if m is None:
            print(f"{name:<25} {'FAILED':>8}")
            continue
        print(
            f"{name:<25} {m['tok_s']:>8.0f} {m['dt_ms']:>8.0f} {m['loss']:>8.4f} {m['vram_gb']:>5.1f}G"
            f" {m.get('phase_scoring_fwd_ms',0):>8.0f} {m.get('phase_scoring_bwd_ms',0):>8.0f}"
            f" {m.get('phase_zero_grad_ms',0):>7.0f} {m.get('phase_boltzmann_ms',0):>6.0f}"
            f" {m.get('phase_train_fwd_ms',0):>7.0f} {m.get('phase_train_ce_ms',0):>6.0f}"
            f" {m.get('phase_train_bwd_ms',0):>7.0f} {m.get('phase_train_step_ms',0):>7.0f}"
            f" {m.get('phase_scoring_total_ms',0):>8.0f} {m.get('phase_train_total_ms',0):>7.0f}"
        )

    print(f"{'='*155}")

    # Save summary JSON
    summary_path = os.path.join(RESULTS_BASE, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
