"""
Batch size finder for Test 14 recurrence_model_1b (1.5B dense, DeltaNet + GSA).

Test 14 training uses DeepSpeed ZeRO-2 (see main.py: deepspeed.initialize, zero_optimization_stage).
This script runs single-GPU, plain PyTorch (no DeepSpeed/ZeRO). So:
  - Capacity M = largest batch that fits on one GPU *without* ZeRO (conservative).
  - With ZeRO-2, optimizer/grad state is sharded, so real training may allow larger
    micro_batch per GPU; use M as a safe lower bound or re-run capacity under ZeRO for cluster-accurate M.
  - Output micro_batch + grad_accum_steps plug into DeepSpeed config as
    train_micro_batch_size_per_gpu and gradient_accumulation_steps.

Two-phase plan:
  1. Capacity find: bracket + binary search for largest batch that fits (no OOM). → M
  2. Effective batch sweep: 256 to 4k; for E > M use grad_accum so we measure loss correctly.
     Same step-0 init for all. Output: recommended effective batch + micro_batch + grad_accum.

Kronecker + tokenizer (same as main.py). Aggressive cleanup after each run.
  TOKENIZER_PATH: dir containing tokenizer.json (default: code/src/tokenizer/)

Data: Phase 2 uses random tokens unless REAL_DATA_DATASET and REAL_DATA_CONFIG are set
  (e.g. wikitext, wikitext-103-raw-v1) for meaningful loss curves.
  Optional env: WARMUP_STEPS, REPEAT_EACH_E, REAL_DATA_MAX_TOKENS, DETERMINISTIC=1, RUN_CAPACITY_AGAIN=1.
  Treat this script as pre-filtering candidates; final check = one short DeepSpeed run per top 2 configs.
"""

import gc
import os
import sys
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any

# Add code root so we can import from src
_CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

import logging
import math
import time

import numpy as np
import torch
import torch.nn as nn

try:
    import psutil
except ImportError:
    psutil = None

from src.data import get_tokenizer
from src.models.recurrence_model_1b import (
    ModelConfig,
    Model1B,
    KroneckerConfig,
    KroneckerEmbeddings,
)

# Fixed config
SEQ_LEN = 4096
N_STEPS = 200
# Capacity: bracket then binary search; require this many steps to pass (avoids lazy init / workspace variance)
CAPACITY_STEPS_REQUIRED = 3
CAPACITY_BRACKET = [256, 512, 1024, 2048, 4096, 8192, 16384]
BINARY_SEARCH_TOLERANCE = 64
# Recommendation: pick best tok/s among configs within this fraction of best loss (e.g. 0.005 = 0.5%)
LOSS_TOLERANCE_FOR_BEST_TOK_S = 0.005
# Effective batch sizes to sweep (256 to 4k)
EFFECTIVE_BATCH_SWEEP = [256, 512, 1024, 2048, 4096]
TOKENIZER_PATH_ENV = os.environ.get("TOKENIZER_PATH", "")
# Warmup steps before timing (reduces kernel autotune / first-backward noise in tok/s)
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "5"))
# Repeat each E this many times and average loss/tok_s (1 = no repeat; 2 = more stable recommendation)
REPEAT_EACH_E = int(os.environ.get("REPEAT_EACH_E", "1"))
# Real data: set REAL_DATA_DATASET and REAL_DATA_CONFIG (e.g. wikitext, wikitext-103-raw-v1) for meaningful loss curves; else random tokens
REAL_DATA_DATASET = os.environ.get("REAL_DATA_DATASET", "").strip()
REAL_DATA_CONFIG = os.environ.get("REAL_DATA_CONFIG", "").strip()
REAL_DATA_MAX_TOKENS = int(os.environ.get("REAL_DATA_MAX_TOKENS", "50000000"))
# DETERMINISTIC=1: set seeds and (optional) reset real-data offset per (E, repeat) for comparable runs
DETERMINISTIC = os.environ.get("DETERMINISTIC", "").strip() in ("1", "true", "yes")
# RUN_CAPACITY_AGAIN=1: after sweep, run 1 step at M again to catch allocator fragmentation
RUN_CAPACITY_AGAIN = os.environ.get("RUN_CAPACITY_AGAIN", "").strip() in ("1", "true", "yes")

logger = logging.getLogger(__name__)
_log_file_path: Optional[str] = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_tokenizer_vocab: Optional[List[str]] = None
_pf_codec: Optional[KroneckerEmbeddings] = None
_vocab_size: int = 0
# Real data buffer (when REAL_DATA_DATASET/CONFIG set): 1D token ids, and offset for next slice
_real_data_buffer: Optional[torch.Tensor] = None
_real_data_offset: int = 0


def _get_tokenizer_path() -> str:
    if TOKENIZER_PATH_ENV.strip():
        path = os.path.abspath(TOKENIZER_PATH_ENV.strip())
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"TOKENIZER_PATH is not a directory: {path}\n"
                "Set TOKENIZER_PATH to the directory that contains tokenizer.json."
            )
        return path
    default = os.path.join(_CODE_ROOT, "src", "tokenizer")
    if not os.path.isdir(default):
        raise FileNotFoundError(
            f"Tokenizer directory not found: {default}\n"
            "Set TOKENIZER_PATH to the directory containing tokenizer.json."
        )
    return default


def _get_kronecker_setup():
    global _tokenizer_vocab, _pf_codec, _vocab_size
    if _tokenizer_vocab is not None and _pf_codec is not None:
        return _tokenizer_vocab, _pf_codec, _vocab_size
    tokenizer_path = _get_tokenizer_path()
    logger.info("Loading tokenizer from: %s", tokenizer_path)
    tokenizer = get_tokenizer(tokenizer_path)
    vocab_size = len(tokenizer)
    vocab_words = []
    for i in range(vocab_size):
        try:
            token = tokenizer.decode([i])
            vocab_words.append(token if token else f"<unk_{i}>")
        except Exception:
            vocab_words.append(f"<unk_{i}>")
    pf_config = KroneckerConfig(
        CHAR_DIM=256, POS_DIM=32, D=8192,
        length_normalize=True, truncate_long_words=True,
    )
    _pf_codec = KroneckerEmbeddings(pf_config)
    _tokenizer_vocab = vocab_words
    _vocab_size = vocab_size
    logger.info("Kronecker vocab size: %s (from tokenizer)", _vocab_size)
    return _tokenizer_vocab, _pf_codec, _vocab_size


def model_factory() -> nn.Module:
    vocab_words, pf_codec, vocab_size = _get_kronecker_setup()
    config = ModelConfig()
    config.vocab_size = vocab_size
    return Model1B(
        config, embedding_type="kronecker",
        bpe_vocab=vocab_words, pf_codec=pf_codec,
    ).to(DEVICE)


def loss_fn(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    x_input = batch[:, :-2].contiguous()
    y_ntp = batch[:, 1:-1].contiguous()
    y_mtp = batch[:, 2:].contiguous()
    logits_ntp, logits_mtp, aux_loss = model(
        x_input, next_token_ids=y_ntp,
        attention_mask=None, return_loss=True, return_memory=False, prev_memory_stream=None,
    )
    loss_ntp = torch.nn.functional.cross_entropy(
        logits_ntp.float().view(-1, logits_ntp.size(-1)), y_ntp.view(-1),
    )
    loss_mtp = torch.nn.functional.cross_entropy(
        logits_mtp.float().view(-1, logits_mtp.size(-1)), y_mtp.view(-1),
    )
    total = loss_ntp + 0.3 * loss_mtp
    if aux_loss is not None:
        total = total + aux_loss
    return total


def _tokenize_text_for_real_data(tokenizer: Any, text: str, max_length: int) -> List[int]:
    """
    Return a flat list of token ids for one text, compatible with HF PreTrainedTokenizer,
    tokenizers.Tokenizer, or any .encode(text) returning list-like ids.
    """
    text = (text or "").strip()
    if not text:
        return []
    ids: List[int] = []
    try:
        # HuggingFace PreTrainedTokenizer: __call__ returns BatchEncoding with input_ids
        enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors=None)
        if hasattr(enc, "get"):
            ids = enc.get("input_ids", [])
        else:
            ids = getattr(enc, "input_ids", [])
        if ids and isinstance(ids[0], (list, tuple)):
            ids = list(ids[0])
        else:
            ids = list(ids)
    except (TypeError, KeyError, AttributeError):
        try:
            # tokenizers.Tokenizer: .encode(text) -> Encoding with .ids
            enc = tokenizer.encode(text)
            ids = getattr(enc, "ids", None) or list(enc)
            ids = list(ids)[:max_length]
        except (TypeError, AttributeError):
            # Fallback: .encode(text) returning list-like of ids
            raw = tokenizer.encode(text)
            ids = getattr(raw, "ids", raw)
            ids = list(ids)[:max_length]
    return ids if isinstance(ids, list) else list(ids)


def _load_real_data_buffer() -> None:
    """Load a fixed buffer of real tokenized text for meaningful loss curves. Uses REAL_DATA_DATASET/CONFIG."""
    global _real_data_buffer, _vocab_size
    if _real_data_buffer is not None:
        return
    if not REAL_DATA_DATASET or not REAL_DATA_CONFIG:
        return
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets not installed; falling back to random data. pip install datasets")
        return
    tokenizer_path = _get_tokenizer_path()
    logger.info("Loading real data: %s / %s (max %s tokens)", REAL_DATA_DATASET, REAL_DATA_CONFIG, REAL_DATA_MAX_TOKENS)
    tok = get_tokenizer(tokenizer_path)
    ds = load_dataset(REAL_DATA_DATASET, REAL_DATA_CONFIG, split="train", trust_remote_code=True)
    all_ids = []
    for ex in ds:
        ids = _tokenize_text_for_real_data(tok, ex.get("text", ""), SEQ_LEN)
        if not ids:
            continue
        all_ids.extend(ids)
        if len(all_ids) >= REAL_DATA_MAX_TOKENS:
            break
    all_ids = all_ids[:REAL_DATA_MAX_TOKENS]
    if len(all_ids) < SEQ_LEN * 2:
        logger.warning("Real data has only %s tokens; falling back to random.", len(all_ids))
        return
    _real_data_buffer = torch.tensor(all_ids, dtype=torch.long)
    logger.info("  Real data buffer: %s tokens", len(_real_data_buffer))


def data_iterator(batch_size: int) -> torch.Tensor:
    global _real_data_offset
    if _vocab_size == 0:
        _get_kronecker_setup()
    if _real_data_buffer is not None:
        need = batch_size * SEQ_LEN
        L = len(_real_data_buffer)
        if need > L:
            raise RuntimeError(f"Real data buffer has {L} tokens; need {need} for batch_size={batch_size} seq_len={SEQ_LEN}")
        start = _real_data_offset % (L - need + 1)
        chunk = _real_data_buffer[start : start + need].clone().to(DEVICE)
        _real_data_offset = (start + need) % L
        return chunk.view(batch_size, SEQ_LEN)
    return torch.randint(0, _vocab_size, (batch_size, SEQ_LEN), device=DEVICE)


def _system_metrics() -> Dict[str, Any]:
    out = {}
    if torch.cuda.is_available():
        out["gpu_alloc_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        out["gpu_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
    if psutil is not None:
        vm = psutil.virtual_memory()
        out["ram_percent"] = round(vm.percent, 2)
        out["ram_used_gb"] = round(vm.used / (1024 ** 3), 2)
        out["ram_available_gb"] = round(vm.available / (1024 ** 3), 2)
        out["cpu_percent"] = round(psutil.cpu_percent(interval=None) or 0, 2)
    return out


def _log_system_metrics(prefix: str = ""):
    m = _system_metrics()
    logger.info("%s System: %s", prefix, "  ".join(f"{k}={v}" for k, v in m.items()))


def _full_cleanup():
    """Free GPU memory and force GC. Call only after dropping model/optimizer refs in caller scope (e.g. model, opt = None, None)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


# ─── Phase 1: Capacity find (bracket + binary search) ────────────────────────

def _run_k_steps(batch_size: int, model: nn.Module, opt: torch.optim.Optimizer, k: int = CAPACITY_STEPS_REQUIRED) -> bool:
    """Run k optimizer steps with given batch size. Returns True if all pass, False on OOM (avoids lazy init / workspace variance)."""
    try:
        for _ in range(k):
            batch = data_iterator(batch_size)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "OutOfMemoryError" in str(e):
            return False
        raise


def find_capacity() -> int:
    """
    Find largest batch size that runs without OOM.
    Bracket with CAPACITY_BRACKET, then binary search until width <= BINARY_SEARCH_TOLERANCE.
    Full cleanup after each try.
    """
    logger.info("\n--- Phase 1: Capacity find (largest batch that fits) ---")
    logger.info("  Requiring %s steps per try to avoid lazy init / workspace variance.", CAPACITY_STEPS_REQUIRED)
    # Bracket: find last OK and first OOM
    low_ok = 0
    high_oom = None
    for b in sorted(CAPACITY_BRACKET):
        model = None
        opt = None
        try:
            model = model_factory()
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
            ok = _run_k_steps(b, model, opt)
            model, opt = None, None
            _full_cleanup()
            if ok:
                low_ok = b
                logger.info("  Capacity try  B=%s  OK  (largest_ok_so_far=%s)", b, low_ok)
            else:
                high_oom = b
                logger.info("  Capacity try  B=%s  OOM  (bracket: [%s, %s])", b, low_ok, high_oom)
                break
        except torch.cuda.OutOfMemoryError:
            high_oom = b
            logger.info("  Capacity try  B=%s  OOM  (bracket: [%s, %s])", b, low_ok, high_oom)
            model, opt = None, None
            _full_cleanup()
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                high_oom = b
                logger.info("  Capacity try  B=%s  OOM  (bracket: [%s, %s])", b, low_ok, high_oom)
                model, opt = None, None
                _full_cleanup()
                break
            model, opt = None, None
            _full_cleanup()
            raise

    if high_oom is None:
        logger.info("  No OOM in bracket; capacity >= %s. Using %s as M.", CAPACITY_BRACKET[-1], CAPACITY_BRACKET[-1])
        return CAPACITY_BRACKET[-1]

    # Binary search in [low_ok, high_oom]
    while high_oom - low_ok > BINARY_SEARCH_TOLERANCE:
        mid = (low_ok + high_oom) // 2
        model, opt = None, None
        try:
            model = model_factory()
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
            ok = _run_k_steps(mid, model, opt)
            model, opt = None, None
            _full_cleanup()
            if ok:
                low_ok = mid
                logger.info("  Binary search  mid=%s  OK   -> new bracket [%s, %s]", mid, low_ok, high_oom)
            else:
                high_oom = mid
                logger.info("  Binary search  mid=%s  OOM  -> new bracket [%s, %s]", mid, low_ok, high_oom)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() and "OutOfMemoryError" not in str(e):
                model, opt = None, None
                _full_cleanup()
                raise
            high_oom = mid
            logger.info("  Binary search  mid=%s  OOM  -> new bracket [%s, %s]", mid, low_ok, high_oom)
            model, opt = None, None
            _full_cleanup()

    M = low_ok
    logger.info("  Capacity M = %s (largest batch that fits in one step)", M)
    _log_system_metrics(prefix="  After capacity ")
    return M


# ─── Phase 2: Effective batch sweep (with grad_accum when E > M) ──────────────

def run_effective_batch_sweep(
    M: int,
    initial_state: Dict[str, torch.Tensor],
    effective_batch_sizes: List[int],
    n_steps: int = N_STEPS,
) -> List[dict]:
    """
    For each effective batch E: same init, n_steps steps. Exact E per optimizer step:
    when E > M we use variable micro-batches (last may be smaller) so effective = E.
    Token-weighted step loss; tokens_processed for tok/s. Sync + peak memory logged.
    Uses real data buffer if REAL_DATA_DATASET/CONFIG set; otherwise random tokens.
    """
    if _vocab_size == 0:
        _get_kronecker_setup()
    if REAL_DATA_DATASET and REAL_DATA_CONFIG:
        _load_real_data_buffer()
    data_mode = f"real ({REAL_DATA_DATASET}/{REAL_DATA_CONFIG})" if _real_data_buffer is not None else "random tokens"
    logger.info("\n--- Phase 2: Effective batch sweep (exact E, grad_accum when E > M) ---")
    logger.info("  Data: %s", data_mode)
    logger.info("  M=%s  effective_batch_sizes=%s  n_steps=%s  warmup=%s  repeat_each_E=%s  deterministic=%s",
                M, effective_batch_sizes, n_steps, WARMUP_STEPS, REPEAT_EACH_E, DETERMINISTIC)
    results = []
    for E in sorted(effective_batch_sizes):
        model = None
        opt = None
        try:
            run_recs = []
            for repeat_idx in range(REPEAT_EACH_E):
                if DETERMINISTIC:
                    torch.manual_seed(42)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(42)
                    np.random.seed(42)
                    global _real_data_offset
                    _real_data_offset = 0
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                model = model_factory()
                model.load_state_dict(initial_state, strict=True)
                model.train()
                opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

                if E <= M:
                    grad_accum_steps = 1
                    micro_batch_max = E
                else:
                    grad_accum_steps = math.ceil(E / M)
                    micro_batch_max = M

                def run_one_step():
                    opt.zero_grad(set_to_none=True)
                    step_loss_weighted = 0.0
                    processed_this_step = 0
                    for accum in range(grad_accum_steps):
                        remaining = E - processed_this_step
                        cur_bs = min(micro_batch_max, remaining)
                        batch = data_iterator(cur_bs)
                        loss = loss_fn(model, batch)
                        weight = cur_bs / E
                        step_loss_weighted += loss.item() * weight
                        if grad_accum_steps > 1:
                            (loss * weight).backward()
                        else:
                            loss.backward()
                        processed_this_step += cur_bs
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    return step_loss_weighted, processed_this_step * SEQ_LEN

                # Warmup (not timed, not counted in loss/tokens)
                for _ in range(WARMUP_STEPS):
                    run_one_step()

                losses = []
                tokens_processed = 0
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for step in range(n_steps):
                    step_loss, tok = run_one_step()
                    losses.append(step_loss)
                    tokens_processed += tok

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                run_rec = {
                    "final_loss": float(np.mean(losses[-20:])),
                    "tokens_per_sec": tokens_processed / elapsed if elapsed > 0 else 0.0,
                    "tokens_processed": tokens_processed,
                    "elapsed_sec": elapsed,
                    "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else None,
                    "gpu_peak_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024 * 1024), 2) if torch.cuda.is_available() else None,
                }
                run_recs.append(run_rec)
                model, opt = None, None
                _full_cleanup()

            # Aggregate over repeats: tok/s = total_tokens / total_elapsed (honest); final_loss = mean
            total_tokens = sum(r["tokens_processed"] for r in run_recs)
            total_elapsed = sum(r["elapsed_sec"] for r in run_recs)
            micro_batch_max = M if E > M else E
            grad_accum_steps = math.ceil(E / M) if E > M else 1
            rec = {
                "effective_batch": E,
                "tokens_per_step": E * SEQ_LEN,
                "micro_batch": micro_batch_max,
                "grad_accum_steps": grad_accum_steps,
                "final_loss": float(np.mean([r["final_loss"] for r in run_recs])),
                "tokens_per_sec": total_tokens / total_elapsed if total_elapsed > 0 else 0.0,
                "tokens_processed": total_tokens,
                "elapsed_sec": total_elapsed,
                "oom": False,
            }
            if run_recs[0].get("gpu_peak_mb") is not None:
                rec["gpu_peak_mb"] = max(r["gpu_peak_mb"] for r in run_recs)
            if run_recs[0].get("gpu_peak_reserved_mb") is not None:
                rec["gpu_peak_reserved_mb"] = max(r["gpu_peak_reserved_mb"] for r in run_recs)
            results.append(rec)
            sys_m = _system_metrics()
            peak_str = ""
            if "gpu_peak_mb" in rec:
                peak_str = f"  gpu_peak_mb={rec['gpu_peak_mb']}"
            if "gpu_peak_reserved_mb" in rec:
                peak_str += f"  gpu_peak_reserved_mb={rec['gpu_peak_reserved_mb']}"
            logger.info(
                "  E=%s  micro_batch=%s  grad_accum=%s  final_loss=%.4f  tok/s=%.0f  elapsed=%.1fs%s  |  %s",
                E, micro_batch_max, grad_accum_steps, rec["final_loss"], rec["tokens_per_sec"], rec["elapsed_sec"], peak_str,
                "  ".join(f"{k}={v}" for k, v in sys_m.items()),
            )
        except torch.cuda.OutOfMemoryError as e:
            results.append({"effective_batch": E, "oom": True})
            logger.info("  E=%s  OOM — skipped  %s", E, str(e).strip()[:150])
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "OutOfMemoryError" in str(e):
                results.append({"effective_batch": E, "oom": True})
                logger.info("  E=%s  OOM — skipped  %s", E, str(e).strip()[:150])
            else:
                raise
        finally:
            model, opt = None, None
            _full_cleanup()

    return results


def recommend(results: List[dict]) -> dict:
    """Pick config with best tok/s among those within LOSS_TOLERANCE_FOR_BEST_TOK_S of best loss; fallback to knee."""
    valid = [r for r in results if not r.get("oom") and "final_loss" in r and "tokens_per_sec" in r]
    if not valid:
        return {}
    if len(valid) == 1:
        return valid[0]
    best_loss = min(r["final_loss"] for r in valid)
    threshold = best_loss * (1 + LOSS_TOLERANCE_FOR_BEST_TOK_S)
    within = [r for r in valid if r["final_loss"] <= threshold]
    if within:
        return max(within, key=lambda r: r["tokens_per_sec"])
    eff_arr = np.array([r["effective_batch"] for r in valid], dtype=float)
    loss_arr = np.array([r["final_loss"] for r in valid], dtype=float)
    eff_n = (eff_arr - eff_arr.min()) / (eff_arr.max() - eff_arr.min() + 1e-8)
    loss_n = (loss_arr - loss_arr.min()) / (loss_arr.max() - loss_arr.min() + 1e-8)
    if len(eff_n) >= 3:
        d2 = np.gradient(np.gradient(loss_n, eff_n), eff_n)
        knee_idx = int(np.argmax(np.abs(d2)))
    else:
        knee_idx = int(np.argmin(loss_arr))
    return valid[knee_idx]


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    global _log_file_path
    t_start = time.perf_counter()

    log_name = f"batch_size_finder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    _log_file_path = os.path.join(_CODE_ROOT, log_name)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(_log_file_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)

    if DETERMINISTIC:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            logger.warning("  torch.use_deterministic_algorithms(True) not available: %s", e)
    logger.info("Batch size finder — Test 14 recurrence_model_1b (1.5B dense, Kronecker + tokenizer)")
    logger.info("  NOTE: Test 14 training uses DeepSpeed ZeRO-2. This script runs single-GPU without ZeRO; M is a conservative lower bound.")
    logger.info("  SEQ_LEN=%s  N_STEPS=%s  DEVICE=%s  deterministic=%s  Log: %s", SEQ_LEN, N_STEPS, DEVICE, DETERMINISTIC, _log_file_path)
    _log_system_metrics(prefix="  Start ")

    # Phase 1: capacity
    M = find_capacity()

    # Save initial state (same step-0 for all Phase 2 runs)
    logger.info("\n--- Saving initial state for Phase 2 (same step-0 for all E) ---")
    model = model_factory()
    initial_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model = None
    _full_cleanup()
    logger.info("  Initial state saved. Cleaning up.")

    # Phase 2: effective batch sweep
    effective_batch_sizes = EFFECTIVE_BATCH_SWEEP
    env_sweep = os.environ.get("EFFECTIVE_BATCH_SWEEP", "").strip()
    if env_sweep:
        effective_batch_sizes = sorted(set(int(x.strip()) for x in env_sweep.split(",") if x.strip()))
    results = run_effective_batch_sweep(M, initial_state, effective_batch_sizes, n_steps=N_STEPS)
    rec = recommend(results)

    elapsed = time.perf_counter() - t_start
    n_oom = sum(1 for r in results if r.get("oom"))
    logger.info("\n--- Run summary ---")
    logger.info("  Wall time: %.1f s  |  Capacity M=%s  |  OOM skipped: %s/%s", elapsed, M, n_oom, len(results))
    _log_system_metrics(prefix="  End ")

    # Final: right batch size + grad accum details (from record; exact E via variable last micro-batch)
    logger.info("\n" + "=" * 70)
    if rec:
        E = rec["effective_batch"]
        micro_batch = rec.get("micro_batch", M)
        grad_accum_steps = rec.get("grad_accum_steps", math.ceil(E / M))
        tokens_per_step = E * SEQ_LEN
        logger.info("  RECOMMENDED FOR YOUR RUN (best tok/s within 0.5%% of best loss)")
        logger.info("  -------------------------")
        logger.info("  Effective batch size (samples per optimizer step):  %s  (exact)", E)
        logger.info("  Tokens per step (seq_len=%s):                      %s", SEQ_LEN, f"{tokens_per_step:,}")
        logger.info("  Micro batch (max per step, capacity M=%s):           %s", M, micro_batch)
        logger.info("  Grad accumulation steps:                           %s", grad_accum_steps)
        logger.info("  Final loss at this E:                              %.4f", rec["final_loss"])
        logger.info("  Tokens/sec (actual counted):                      %.0f", rec.get("tokens_per_sec", 0))
        logger.info("  DeepSpeed config: train_micro_batch_size_per_gpu=%s, gradient_accumulation_steps=%s", micro_batch, grad_accum_steps)
        logger.info("  (With DDP/ZeRO, use no_sync() on non-final accum steps when grad_accum > 1.)")
        logger.info("=" * 70)
    else:
        logger.info("  No valid runs (all OOM or no data); no recommendation.")
        logger.info("=" * 70)

    # Optional: re-run capacity (1 step at M) to catch allocator fragmentation
    if RUN_CAPACITY_AGAIN and M > 0 and torch.cuda.is_available():
        logger.info("\n--- Run capacity again (sanity check for fragmentation) ---")
        model = None
        opt = None
        try:
            model = model_factory()
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
            ok = _run_k_steps(M, model, opt, k=1)
            model, opt = None, None
            _full_cleanup()
            if ok:
                logger.info("  M=%s still runs 1 step OK (no fragmentation surprise).", M)
            else:
                logger.warning("  M=%s failed on re-run; possible allocator fragmentation. Consider using a fresh process or min(M1, M2).", M)
        except Exception as e:
            logger.warning("  Re-run capacity failed: %s (possible fragmentation).", e)
        finally:
            model, opt = None, None
            _full_cleanup()

    logger.info("\nFull log saved to: %s", _log_file_path)


if __name__ == "__main__":
    main()
