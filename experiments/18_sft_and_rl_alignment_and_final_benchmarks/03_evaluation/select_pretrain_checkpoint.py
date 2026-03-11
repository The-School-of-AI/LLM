#!/usr/bin/env python3
r"""
Pre-training Checkpoint Selector
Team 18: SFT, RL-Style Alignment & Final Post-Training Benchmarks

Ranks candidate pre-training checkpoints and selects the best one to SFT from.

Scoring (lower rank = better):
  1. Validation perplexity  (primary — lower is better)
  2. Benchmark average      (secondary — higher is better)
  3. Training stability     (filter — checkpoints with recent loss spikes are flagged)

Usage:
    # Evaluate a list of local HF-format checkpoints (bash/zsh)
    python select_pretrain_checkpoint.py \\
        --checkpoints /ckpts/step_80000 /ckpts/step_90000 /ckpts/step_100000 \\
        --output_json checkpoint_ranking.json

    # PowerShell equivalent
    python .\select_pretrain_checkpoint.py `
        --checkpoints C:\ckpts\step_80000 C:\ckpts\step_90000 C:\ckpts\step_100000 `
        --output_json .\checkpoint_ranking.json

    # Supply custom validation data (JSONL with "text" field, one doc per line)
    python .\select_pretrain_checkpoint.py `
        --checkpoints C:\ckpts\step_100000 `
        --val_data C:\data\val.jsonl `
        --output_json .\checkpoint_ranking.json

    # Skip benchmarks (perplexity-only, much faster)
    python .\select_pretrain_checkpoint.py `
        --checkpoints C:\ckpts\step_100000 `
        --skip_benchmarks `
        --output_json .\checkpoint_ranking.json

    # Supply a loss log CSV to check training stability
    #   CSV must have columns: step, loss
    python .\select_pretrain_checkpoint.py `
        --checkpoints C:\ckpts\step_100000 `
        --loss_log C:\logs\train_loss.csv `
        --output_json .\checkpoint_ranking.json

Notes:
    - Checkpoints must be in HuggingFace format (config.json + weights).
      If you have ZeRO/DeepSpeed checkpoints, convert first:
        python zero_to_fp32.py <checkpoint_dir> <output_dir>
    - Uses the team tokenizer at FINAL_TOKENIZER/ by default.
      Override with --tokenizer_path.
    - Benchmarks use lm-evaluation-harness (pip install lm-eval).
      Skipped automatically if lm_eval is not installed.
"""

import argparse
import json
import logging
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Directory containing this script
SCRIPT_DIR = Path(__file__).parent

# Team tokenizer (default — override with --tokenizer_path)
DEFAULT_TOKENIZER = str(SCRIPT_DIR.parent / "FINAL_TOKENIZER")

# Benchmarks for pre-training checkpoint selection.
#
# IMPORTANT: MMLU-Pro and GSM8K are NOT used here — they require instruction
# following and score near-random on raw PT models (~10-15% and ~1-2%
# respectively), making them useless for ranking. Use those benchmarks
# post-SFT via evaluate_smoke_test.py instead.
#
# These benchmarks are completion-style and work well on raw PT models:
#   - HellaSwag: sentence completion, 10K examples, ~30 min at 70B
#   - ARC-Challenge: reasoning MC, 1.2K examples, ~5 min at 70B
#   - WinoGrande: commonsense MC, 1.3K examples, ~5 min at 70B
#   - LAMBADA: next-word prediction, 5K examples, ~15 min at 70B
#
# Total benchmark time: ~55 min per checkpoint at 70B on 2xA100.
BENCHMARKS = {
    "hellaswag": {
        "task": "hellaswag",
        "num_fewshot": 10,
        "description": "HellaSwag (commonsense completion)",
        "weight": 1.0,
    },
    "arc_challenge": {
        "task": "arc_challenge",
        "num_fewshot": 25,
        "description": "ARC-Challenge (reasoning)",
        "weight": 1.0,
    },
    "winogrande": {
        "task": "winogrande",
        "num_fewshot": 5,
        "description": "WinoGrande (commonsense)",
        "weight": 1.0,
    },
    "lambada_openai": {
        "task": "lambada_openai",
        "num_fewshot": 0,
        "description": "LAMBADA (next-word prediction)",
        "weight": 1.0,
    },
}

# Validation block size — must match pre-training block size
VAL_BLOCK_SIZE = 4096

# Number of validation tokens to use for perplexity (8M = fast but reliable)
VAL_TOKENS = 8_000_000

# Stability window: how many steps before a checkpoint to scan for spikes
STABILITY_WINDOW = 500

# A loss spike is defined as: loss > rolling_mean * (1 + SPIKE_THRESHOLD)
SPIKE_THRESHOLD = 0.15


# ---------------------------------------------------------------------------
# Validation data helpers
# ---------------------------------------------------------------------------


def _load_val_texts_from_jsonl(path: str, max_chars: int = 50_000_000) -> List[str]:
    """Load text documents from a JSONL file (field: 'text')."""
    texts = []
    total = 0
    with open(path) as f:
        for line in f:
            if total >= max_chars:
                break
            try:
                doc = json.loads(line)
                text = doc.get("text") or doc.get("content") or ""
                if text:
                    texts.append(text)
                    total += len(text)
            except json.JSONDecodeError:
                continue
    logger.info(f"Loaded {len(texts)} docs from {path} ({total:,} chars)")
    return texts


def _stability_sort_bucket(stable: Optional[bool]) -> int:
    """Stable first, unchecked next, unstable last."""
    if stable is False:
        return 2
    if stable is None:
        return 1
    return 0


def _stability_label(stable: Optional[bool]) -> str:
    """Human-readable stability status."""
    if stable is True:
        return "YES"
    if stable is False:
        return "SPIKE"
    return "UNKNOWN"


def _load_val_texts_wikitext() -> List[str]:
    """
    Fall back to WikiText-103 test split (HuggingFace datasets).
    This is a standard, reproducible validation set.
    """
    try:
        from datasets import load_dataset

        logger.info("Loading WikiText-103 test split as validation data...")
        ds = load_dataset("wikitext", "wikitext-103-v1", split="test")
        texts = [row["text"] for row in ds if row["text"].strip()]
        logger.info(f"Loaded {len(texts)} WikiText-103 docs")
        return texts
    except Exception as e:
        logger.error(f"Failed to load WikiText-103: {e}")
        logger.error("Provide --val_data to specify a local validation file.")
        return []


def build_val_token_blocks(
    texts: List[str],
    tokenizer,
    block_size: int = VAL_BLOCK_SIZE,
    max_tokens: int = VAL_TOKENS,
) -> torch.Tensor:
    """
    Tokenize texts and pack into fixed-size blocks.
    Returns a tensor of shape (num_blocks, block_size).
    """
    if tokenizer.eos_token_id is None:
        raise ValueError(
            "Tokenizer must define eos_token_id so validation documents can be separated."
        )

    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        all_ids.append(tokenizer.eos_token_id)
        if len(all_ids) >= max_tokens:
            break

    all_ids = all_ids[:max_tokens]
    num_blocks = len(all_ids) // block_size
    if num_blocks == 0:
        raise ValueError(
            f"Not enough validation tokens ({len(all_ids)}) for block size {block_size}. "
            "Use a larger --val_data file."
        )

    ids_tensor = torch.tensor(all_ids[: num_blocks * block_size], dtype=torch.long)
    blocks = ids_tensor.view(num_blocks, block_size)
    logger.info(
        f"Built {num_blocks} validation blocks "
        f"({num_blocks * block_size:,} tokens, block_size={block_size})"
    )
    return blocks


# ---------------------------------------------------------------------------
# Perplexity measurement
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_perplexity(
    model_path: str,
    tokenizer_path: str,
    val_blocks: torch.Tensor,
    batch_size: int = 4,
    dtype: torch.dtype = torch.bfloat16,
) -> float:
    """
    Compute perplexity of a HF checkpoint on pre-tokenized validation blocks.

    Args:
        model_path: Path to HF-format checkpoint directory.
        tokenizer_path: Path to tokenizer directory.
        val_blocks: Tensor of shape (num_blocks, block_size).
        batch_size: Blocks per forward pass.
        dtype: Model dtype (bfloat16 recommended for 70B).

    Returns:
        Perplexity (float). Lower is better.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model for perplexity: {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("No GPU found — perplexity measurement will be slow.")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_blocks = val_blocks.shape[0]

    for i in range(0, num_blocks, batch_size):
        batch = val_blocks[i : i + batch_size].to(device)  # (B, T)
        # Labels = inputs shifted right (standard CLM)
        outputs = model(input_ids=batch, labels=batch)
        loss = outputs.loss  # mean cross-entropy over non-padding tokens
        batch_tokens = batch.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

        if (i // batch_size) % 20 == 0:
            ppl_so_far = math.exp(total_loss / total_tokens)
            logger.info(
                f"  [{i}/{num_blocks} blocks] running ppl = {ppl_so_far:.2f}"
            )

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    # Free GPU memory before next checkpoint
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Perplexity for {model_path}: {ppl:.4f}")
    return ppl


# ---------------------------------------------------------------------------
# Benchmark evaluation (lm-eval)
# ---------------------------------------------------------------------------


def _lm_eval_available() -> bool:
    try:
        import importlib

        return importlib.util.find_spec("lm_eval") is not None
    except Exception:
        return False


def run_benchmark(
    model_path: str,
    task: str,
    num_fewshot: int,
    output_dir: str,
    batch_size: str = "auto",
) -> Optional[float]:
    """
    Run a single lm-eval benchmark on a HF checkpoint.

    Returns:
        Score as a float in [0, 100], or None on failure.
    """
    output_path = os.path.join(output_dir, f"{task}.json")
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path}",
        "--tasks", task,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", batch_size,
        "--output_path", output_path,
    ]

    logger.info(f"Running benchmark: {task} on {model_path}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200
        )
        if result.returncode != 0:
            logger.error(f"lm_eval failed for {task}:\n{result.stderr[-2000:]}")
            return None
    except subprocess.TimeoutExpired:
        logger.error(f"Benchmark {task} timed out (2h limit)")
        return None

    if not Path(output_path).exists():
        logger.error(f"lm_eval output not found at {output_path}")
        return None

    try:
        with open(output_path) as f:
            data = json.load(f)
        results = data.get("results", {})
        for _task_name, task_data in results.items():
            for key in ["acc,none", "acc_norm,none", "exact_match,none", "acc"]:
                if key in task_data:
                    score = task_data[key]
                    if score is not None and score <= 1.0:
                        score *= 100.0
                    return float(score)
    except Exception as e:
        logger.error(f"Failed to parse lm_eval output: {e}")

    return None


def run_benchmarks(
    model_path: str,
    output_dir: str,
) -> Dict[str, Optional[float]]:
    """Run all configured benchmarks. Returns {name: score_or_None}."""
    scores = {}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for name, cfg in BENCHMARKS.items():
        scores[name] = run_benchmark(
            model_path=model_path,
            task=cfg["task"],
            num_fewshot=cfg["num_fewshot"],
            output_dir=output_dir,
        )
    return scores


def weighted_benchmark_avg(scores: Dict[str, Optional[float]]) -> Optional[float]:
    """Weighted average of benchmark scores. Returns None if all failed."""
    total_weight = 0.0
    weighted_sum = 0.0
    for name, score in scores.items():
        if score is not None:
            w = BENCHMARKS[name]["weight"]
            weighted_sum += score * w
            total_weight += w
    if total_weight == 0:
        return None
    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Training stability check
# ---------------------------------------------------------------------------


def check_stability(
    loss_log_path: str,
    checkpoint_step: int,
    window: int = STABILITY_WINDOW,
    spike_threshold: float = SPIKE_THRESHOLD,
) -> Tuple[bool, str]:
    """
    Check for loss spikes in the `window` steps before `checkpoint_step`.

    Args:
        loss_log_path: CSV file with columns: step, loss
        checkpoint_step: Training step number of the checkpoint.
        window: Number of steps to inspect before checkpoint_step.
        spike_threshold: Relative increase over rolling mean to flag as spike.

    Returns:
        (is_stable, description) — is_stable=False means spikes detected.
    """
    import csv

    steps, losses = [], []
    with open(loss_log_path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Loss log is empty or missing a header row")
        required_columns = {"step", "loss"}
        missing = required_columns.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "Loss log must contain columns: step, loss "
                f"(missing: {', '.join(sorted(missing))})"
            )
        for row in reader:
            if not row.get("step") or not row.get("loss"):
                continue
            try:
                s = int(row["step"])
                loss = float(row["loss"])
            except ValueError:
                continue
            if checkpoint_step - window <= s <= checkpoint_step:
                steps.append(s)
                losses.append(loss)

    if len(losses) < 10:
        return True, f"Only {len(losses)} loss points in window — stability unchecked"

    # Rolling mean over 50-step windows
    spike_steps = []
    win = 50
    for i in range(win, len(losses)):
        rolling_mean = sum(losses[i - win : i]) / win
        if losses[i] > rolling_mean * (1 + spike_threshold):
            spike_steps.append(steps[i])

    if spike_steps:
        return (
            False,
            f"Loss spikes detected at steps: {spike_steps[:5]}"
            + (" (and more)" if len(spike_steps) > 5 else ""),
        )
    return True, "Stable — no spikes in window"


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def rank_checkpoints(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Rank checkpoints by composite score.

    Composite score (lower = better):
        rank_ppl + rank_benchmark_avg

    Unstable checkpoints are sorted to the bottom regardless of score.
    """
    # Sort by perplexity for rank
    valid = [r for r in results if r.get("perplexity") is not None]
    valid.sort(key=lambda r: r["perplexity"])
    for i, r in enumerate(valid):
        r["rank_ppl"] = i + 1

    # Sort by benchmark avg for rank (higher = better → negate)
    bench_valid = [r for r in valid if r.get("benchmark_avg") is not None]
    bench_valid.sort(key=lambda r: -(r["benchmark_avg"] or 0))
    for i, r in enumerate(bench_valid):
        r["rank_benchmark"] = i + 1

    # Composite rank
    for r in valid:
        ppl_rank = r.get("rank_ppl", len(valid))
        bench_rank = r.get("rank_benchmark", len(valid))
        r["composite_rank"] = ppl_rank + bench_rank

    # Stable checkpoints first, then sort by composite rank
    valid.sort(
        key=lambda r: (
            _stability_sort_bucket(r.get("stable", True)),
            r.get("composite_rank", 9999),
        )
    )
    for i, r in enumerate(valid):
        r["final_rank"] = i + 1

    # Append any checkpoints that failed perplexity
    failed = [r for r in results if r.get("perplexity") is None]
    return valid + failed


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_ranking_table(ranked: List[Dict[str, Any]]):
    col_w = 36
    print("\n" + "=" * 120)
    print("PRE-TRAINING CHECKPOINT RANKING")
    print("=" * 120)
    header = (
        f"{'Rank':<6} {'Checkpoint':<{col_w}} {'Perplexity':>12} "
        f"{'BenchAvg':>10} {'HellaSwag':>10} {'ARC-C':>7} {'Wino':>7} {'LAMBADA':>9} {'Stable':>8}"
    )
    print(header)
    print("-" * 120)

    for r in ranked:
        rank = r.get("final_rank", "-")
        ckpt = Path(r["checkpoint"]).name[:col_w]
        ppl = f"{r['perplexity']:.4f}" if r.get("perplexity") else "FAILED"
        bench = (
            f"{r['benchmark_avg']:.1f}%" if r.get("benchmark_avg") is not None else "-"
        )
        benches = r.get("benchmarks", {})

        def fmt(key):
            v = benches.get(key)
            return f"{v:.1f}%" if v is not None else "-"

        stable = _stability_label(r.get("stable", True))

        print(
            f"{rank!s:<6} {ckpt:<{col_w}} {ppl:>12} "
            f"{bench:>10} {fmt('hellaswag'):>10} {fmt('arc_challenge'):>7} "
            f"{fmt('winogrande'):>7} {fmt('lambada_openai'):>9} {stable:>8}"
        )

    print("=" * 120)
    if ranked and ranked[0].get("perplexity"):
        winner = ranked[0]
        print(f"\nRECOMMENDED: {winner['checkpoint']}")
        print(
            f"  Perplexity: {winner['perplexity']:.4f} | "
            f"Benchmark avg: {winner.get('benchmark_avg') or 'N/A'}"
        )
        if not winner.get("stable", True):
            print("  WARNING: recommended checkpoint has loss spikes — verify manually")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Rank pre-training checkpoints for SFT selection"
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="One or more HF-format checkpoint directories to evaluate",
    )
    parser.add_argument(
        "--tokenizer_path",
        default=DEFAULT_TOKENIZER,
        help=f"Tokenizer directory (default: {DEFAULT_TOKENIZER})",
    )
    parser.add_argument(
        "--val_data",
        default=None,
        help="Validation JSONL file (field: 'text'). Defaults to WikiText-103.",
    )
    parser.add_argument(
        "--val_tokens",
        type=int,
        default=VAL_TOKENS,
        help=f"Max tokens for perplexity eval (default: {VAL_TOKENS:,})",
    )
    parser.add_argument(
        "--skip_benchmarks",
        action="store_true",
        help="Skip lm-eval benchmarks (perplexity only — much faster)",
    )
    parser.add_argument(
        "--loss_log",
        default=None,
        help="CSV file (columns: step, loss) for stability check",
    )
    parser.add_argument(
        "--output_json",
        required=True,
        help="Path to save full ranking results as JSON",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for perplexity evaluation (default: 4)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    for ckpt in args.checkpoints:
        if not Path(ckpt).exists():
            logger.error(f"Checkpoint not found: {ckpt}")
            sys.exit(1)
        if not (Path(ckpt) / "config.json").exists():
            logger.error(
                f"{ckpt} does not look like a HF checkpoint (no config.json). "
                "If this is a ZeRO checkpoint, convert with zero_to_fp32.py first."
            )
            sys.exit(1)

    if not Path(args.tokenizer_path).exists():
        logger.error(f"Tokenizer not found: {args.tokenizer_path}")
        sys.exit(1)

    if args.val_data and not Path(args.val_data).exists():
        logger.error(f"Validation data not found: {args.val_data}")
        sys.exit(1)

    if args.loss_log and not Path(args.loss_log).exists():
        logger.error(f"Loss log not found: {args.loss_log}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load tokenizer and build validation blocks (once, shared across ckpts)
    # ------------------------------------------------------------------
    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )

    if args.val_data:
        val_texts = _load_val_texts_from_jsonl(args.val_data)
    else:
        val_texts = _load_val_texts_wikitext()

    if not val_texts:
        logger.error("No validation texts loaded. Exiting.")
        sys.exit(1)

    val_blocks = build_val_token_blocks(
        val_texts, tokenizer, max_tokens=args.val_tokens
    )

    # ------------------------------------------------------------------
    # Evaluate each checkpoint
    # ------------------------------------------------------------------
    run_benchmarks_flag = not args.skip_benchmarks and _lm_eval_available()
    if not args.skip_benchmarks and not _lm_eval_available():
        logger.warning(
            "lm_eval not installed — skipping benchmarks. "
            "Install with: pip install lm-eval"
        )

    all_results = []
    for ckpt in args.checkpoints:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {ckpt}")
        logger.info(f"{'='*60}")

        result: Dict[str, Any] = {"checkpoint": ckpt}

        # 1. Perplexity
        try:
            ppl = compute_perplexity(
                model_path=ckpt,
                tokenizer_path=args.tokenizer_path,
                val_blocks=val_blocks,
                batch_size=args.batch_size,
            )
            result["perplexity"] = ppl
        except Exception as e:
            logger.error(f"Perplexity failed for {ckpt}: {e}")
            result["perplexity"] = None
            result["perplexity_error"] = str(e)

        # 2. Benchmarks
        result["benchmarks"] = {}
        if run_benchmarks_flag:
            bench_dir = str(
                Path(args.output_json).parent
                / f"bench_{Path(ckpt).name}"
            )
            scores = run_benchmarks(ckpt, bench_dir)
            result["benchmarks"] = scores
            result["benchmark_avg"] = weighted_benchmark_avg(scores)
        else:
            result["benchmark_avg"] = None

        # 3. Stability check
        if args.loss_log:
            # Try to infer step from checkpoint directory name (e.g. "step_100000")
            ckpt_name = Path(ckpt).name
            step = None
            for part in ckpt_name.replace("-", "_").split("_"):
                if part.isdigit():
                    step = int(part)
                    break
            if step is not None:
                try:
                    stable, note = check_stability(args.loss_log, step)
                    result["stable"] = stable
                    result["stability_note"] = note
                    logger.info(f"Stability ({ckpt}): {note}")
                except Exception as e:
                    note = f"Stability check failed: {e}"
                    logger.warning(note)
                    result["stable"] = None
                    result["stability_note"] = note
            else:
                logger.warning(
                    f"Could not infer step from checkpoint name '{ckpt_name}'. "
                    "Skipping stability check. Rename dir to include the step number."
                )
                result["stable"] = None
                result["stability_note"] = "Step not inferred from directory name"
        else:
            result["stable"] = True
            result["stability_note"] = "No loss log provided"

        all_results.append(result)

    # ------------------------------------------------------------------
    # Rank and output
    # ------------------------------------------------------------------
    ranked = rank_checkpoints(all_results)
    print_ranking_table(ranked)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"ranked_checkpoints": ranked}, f, indent=2)
    logger.info(f"Full results saved to {output_path}")


if __name__ == "__main__":
    main()
