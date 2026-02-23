"""
Live metrics watcher — run in a second terminal while training.
Shows loss, MoE router health, per-expert load, and per-GPU utilization.

Usage:
    python3 watch_metrics.py                          # default path
    python3 watch_metrics.py checkpoints_moe_test/metrics.jsonl
"""

import json
import sys
import time
from pathlib import Path

METRICS_FILE = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_moe_test/metrics.jsonl"

RED   = "\033[31m"
YLW   = "\033[33m"
GRN   = "\033[32m"
DIM   = "\033[2m"
RESET = "\033[0m"


def fmt(val, spec=".4f"):
    return f"{val:{spec}}" if val is not None else f"{DIM}null{RESET}"


def color_null_rate(v):
    """Highlight null_rate if outside healthy band (target ~0.167 with 4 nulls/24 slots)."""
    if v is None:
        return f"{DIM}null{RESET}"
    s = f"{v:.3f}"
    if v < 0.05 or v > 0.40:
        return f"{RED}{s}{RESET}"
    if v < 0.10 or v > 0.30:
        return f"{YLW}{s}{RESET}"
    return f"{GRN}{s}{RESET}"


def color_gpu(v):
    if v is None:
        return f"{DIM}null{RESET}"
    s = f"{v:.0f}%"
    if v < 50:
        return f"{YLW}{s}{RESET}"
    return s


def print_loss_header():
    print(
        f"\n{'step':>5} | {'loss':>8} | {'ntp':>8} | {'mtp':>8} | {'aux':>8} | "
        f"{'null_rtr':>8} | {'moe_rtr':>8} | {'tok/s':>7} | {'lr':>10}"
    )
    print("-" * 100)


def print_router_header():
    print(
        f"\n{'step':>5} | {'null_rate':>10} | {'avg_real':>8} | {'0real%':>7} | "
        f"{'L_bal':>7} | {'L_null':>7} | {'L_z':>7} | {'#gates':>6}"
    )
    print("-" * 78)


def print_expert_load(step, counts):
    if not counts:
        return
    total = sum(counts) or 1.0
    fracs = [c / total for c in counts]
    max_f = max(fracs)
    bars = []
    for i, f in enumerate(fracs):
        bar = "█" * int(f * 20)
        flag = f"{RED}*{RESET}" if f > 2 * (1 / len(counts)) else ""
        bars.append(f"E{i}:{f:.2f}{flag}{DIM}{bar}{RESET}")
    print(f"  {'step':>5} expert load | " + "  ".join(bars))


def print_gpu_header(n_gpus):
    gpu_cols = "  ".join(f"GPU{i}(util/mem)" for i in range(n_gpus))
    print(f"\n{'step':>5} | {gpu_cols}")
    print("-" * (8 + 20 * n_gpus))


def watch(path: str):
    p = Path(path)
    print(f"Watching: {p.resolve()}")
    print("Waiting for training to start...\n")

    seen_lines = 0
    step_count = 0       # reprint headers every N steps
    last_n_gpus = None

    while True:
        if not p.exists():
            time.sleep(1)
            continue

        lines = p.read_text().strip().splitlines()

        if len(lines) <= seen_lines:
            time.sleep(0.5)
            continue

        for line in lines[seen_lines:]:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            event = d.get("event")

            if event == "train_step":
                step     = d.get("global_step", "?")
                loss     = d.get("loss")
                ntp      = d.get("loss_ntp")
                mtp      = d.get("loss_mtp")
                aux      = d.get("loss_aux")
                null_rtr = d.get("loss_null_router")
                moe_rtr  = d.get("loss_moe_router")
                tok_s    = d.get("tokens_per_sec")
                lr       = d.get("lr")

                # MoE router stats
                null_rate  = d.get("moe_null_rate")
                avg_real   = d.get("moe_avg_real_experts")
                zero_frac  = d.get("moe_zero_real_frac")
                l_bal      = d.get("moe_L_bal")
                l_null     = d.get("moe_L_null")
                l_z        = d.get("moe_L_z")
                n_gates    = d.get("moe_num_gates")           # not logged yet, defensive
                exp_counts = d.get("moe_expert_counts")

                # All-GPU stats
                gpu_util_all = d.get("gpu_util_all_pct") or {}
                gpu_mem_all  = d.get("gpu_mem_all_gb") or {}
                n_gpus = len(gpu_util_all)

                # ── Loss line (reprint header every 20 steps) ──────────────
                if step_count % 20 == 0:
                    print_loss_header()
                print(
                    f"{step:>5} | {fmt(loss):>8} | {fmt(ntp):>8} | {fmt(mtp):>8} | "
                    f"{fmt(aux):>8} | {fmt(null_rtr):>8} | {fmt(moe_rtr):>8} | "
                    f"{fmt(tok_s, '.1f') if tok_s else fmt(None):>7} | "
                    f"{fmt(lr, '.2e') if lr else fmt(None):>10}"
                )

                # ── MoE router health ──────────────────────────────────────
                if null_rate is not None:
                    if step_count % 20 == 0:
                        print_router_header()
                    print(
                        f"{step:>5} | {color_null_rate(null_rate):>10} | "
                        f"{fmt(avg_real, '.2f'):>8} | "
                        f"{fmt(zero_frac, '.3f'):>7} | "
                        f"{fmt(l_bal, '.4f'):>7} | "
                        f"{fmt(l_null, '.4f'):>7} | "
                        f"{fmt(l_z, '.2f'):>7} | "
                        f"{str(n_gates) if n_gates else DIM+'?'+RESET:>6}"
                    )

                # ── Per-expert load bar ────────────────────────────────────
                if exp_counts:
                    print_expert_load(step, exp_counts)

                # ── Per-GPU utilization ────────────────────────────────────
                if n_gpus:
                    if n_gpus != last_n_gpus or step_count % 20 == 0:
                        print_gpu_header(n_gpus)
                        last_n_gpus = n_gpus
                    gpu_cols = "  ".join(
                        f"{color_gpu(gpu_util_all.get(str(i)))}/{gpu_mem_all.get(str(i), '?'):.1f}G"
                        if isinstance(gpu_mem_all.get(str(i)), float)
                        else f"{color_gpu(gpu_util_all.get(str(i)))}/?.?G"
                        for i in range(n_gpus)
                    )
                    print(f"{step:>5} | {gpu_cols}")

                print()  # blank line between steps for readability
                step_count += 1

            elif event in ("evaluation", "eval"):
                avg_loss = d.get("avg_loss") or d.get("val_loss")
                avg_ppl  = d.get("avg_perplexity") or d.get("perplexity")
                print(
                    f"\n  {GRN}[EVAL]{RESET} step={d.get('global_step')} "
                    f"val_loss={fmt(avg_loss)} val_ppl={fmt(avg_ppl, '.2f') if avg_ppl else fmt(None)}\n"
                )
                step_count = 0  # force header reprint after eval

            elif event == "checkpoint_saved":
                print(
                    f"\n  {DIM}[CKPT]{RESET} step={d.get('global_step')} "
                    f"tag={d.get('tag')} ({fmt(d.get('duration_s'), '.1f')}s)\n"
                )

        seen_lines = len(lines)
        time.sleep(0.5)


if __name__ == "__main__":
    try:
        watch(METRICS_FILE)
    except KeyboardInterrupt:
        print("\nStopped.")
