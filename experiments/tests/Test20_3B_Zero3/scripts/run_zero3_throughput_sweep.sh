#!/usr/bin/env bash
set -euo pipefail

TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$TEST_ROOT/run.sh"
OUT_DIR="$TEST_ROOT/results/sweeps"
STAMP="$(date '+%Y%m%d_%H%M%S')"
SWEEP_DIR="$OUT_DIR/$STAMP"
mkdir -p "$SWEEP_DIR"

BASE_CFG="$TEST_ROOT/configs/test17_3b_moe_offload_1024_30steps.yaml"
PERF_CFG="$TEST_ROOT/configs/test17_3b_moe_perf_1024_30steps.yaml"
PERF_PARAM_CFG="$SWEEP_DIR/test17_3b_moe_perf_paramoffload_1024_30steps.yaml"

cp "$PERF_CFG" "$PERF_PARAM_CFG"
sed -i.bak 's|zero-3-moe-bf16-perf-noparamoffload.json|zero-3-moe-bf16-perf-paramoffload.json|g' "$PERF_PARAM_CFG"
rm -f "$PERF_PARAM_CFG.bak"

run_case() {
  local name="$1"
  local cfg="$2"
  shift 2
  local log="$SWEEP_DIR/${name}.log"
  echo "=== Running ${name} ==="
  (
    cd "$TEST_ROOT"
    FORCE_REWRITE_INIT=0 CFG="$cfg" "$@" "$RUNNER"
  ) >"$log" 2>&1 || true
  echo "=== Finished ${name}: $log ==="
}

run_case "A_baseline_hardened" "$BASE_CFG" env

run_case "B_baseline_no_hardening" "$BASE_CFG" env \
  T19_STEP_CUDA_SYNC=0 \
  T19_STEP_GC_COLLECT=0 \
  T19_STEP_EMPTY_CACHE=0 \
  T19_STEP_IPC_COLLECT=0 \
  T19_ZERO3_RELEASE_EVERY=0 \
  T19_CLEAR_ROUTER_CACHE_EVERY=0 \
  T19_TRACK_CUDA_MEMORY=0

run_case "C_perf_no_param_offload" "$PERF_CFG" env \
  T19_STEP_CUDA_SYNC=0 \
  T19_STEP_GC_COLLECT=0 \
  T19_STEP_EMPTY_CACHE=0 \
  T19_STEP_IPC_COLLECT=0 \
  T19_ZERO3_RELEASE_EVERY=0 \
  T19_CLEAR_ROUTER_CACHE_EVERY=0 \
  T19_TRACK_CUDA_MEMORY=0

run_case "D_perf_with_param_offload" "$PERF_PARAM_CFG" env \
  T19_STEP_CUDA_SYNC=0 \
  T19_STEP_GC_COLLECT=0 \
  T19_STEP_EMPTY_CACHE=0 \
  T19_STEP_IPC_COLLECT=0 \
  T19_ZERO3_RELEASE_EVERY=0 \
  T19_CLEAR_ROUTER_CACHE_EVERY=0 \
  T19_TRACK_CUDA_MEMORY=0

python3 - "$SWEEP_DIR" <<'PY'
import re
import statistics
import sys
from pathlib import Path

sweep = Path(sys.argv[1])
print(f"sweep_dir={sweep}")
print("case,steps,avg_tok_s,min_tok_s,max_tok_s,avg_dt_ms,oom,completed")

for log in sorted(sweep.glob("*.log")):
    lines = log.read_text(errors="ignore").splitlines()
    tok, dt = [], []
    for ln in lines:
        mt = re.search(r"tok/sec:\s*([0-9.]+)", ln)
        md = re.search(r"dt:\s*([0-9.]+)ms", ln)
        if mt:
            tok.append(float(mt.group(1)))
        if md:
            dt.append(float(md.group(1)))
    oom = any(("OutOfMemoryError" in ln or "CUDA out of memory" in ln) for ln in lines)
    completed = any("Test 14 completed" in ln for ln in lines)
    if tok and dt:
        print(
            f"{log.stem},{len(tok)},{statistics.mean(tok):.2f},{min(tok):.2f},{max(tok):.2f},"
            f"{statistics.mean(dt):.2f},{int(oom)},{int(completed)}"
        )
    else:
        print(f"{log.stem},0,NA,NA,NA,NA,{int(oom)},{int(completed)}")
PY
