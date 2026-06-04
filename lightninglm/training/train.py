"""
Training utilities for DeepSpeed.

This module contains training, evaluation, and inference functions
for training language models with DeepSpeed optimization.
"""

import gc
import inspect
import json
import math
import os
import time
from contextlib import contextmanager
from datetime import datetime

import psutil
import torch
import torch.distributed as dist
from tqdm import tqdm

# FIX-PERF-04 (v3): FusedLinearCrossEntropyLoss — fuses lm_head matmul + CE.
# Never materialises [B*T, vocab] logits (saves ~17 GB per step).
# ZERO FALLBACK — if this import fails, training crashes immediately.
from .kernels.triton_cross_entropy import FusedLinearCrossEntropyLoss as _FusedLinearCE
from .profiler import StepProfiler
from .utils import is_main_process, print_rank_0

# _fused_ce is initialized inside train_epoch using max_chunk_gb from config


@contextmanager
def _null_ctx():
    yield


try:
    from deepspeed.profiling.flops_profiler import FlopsProfiler
except Exception:  # pragma: no cover - fallback for lightweight environments

    class FlopsProfiler:  # type: ignore
        def __init__(self, *_args, **_kwargs):
            pass

        def start_profile(self):
            pass

        def stop_profile(self):
            pass

        def get_total_flops(self):
            return 0

        def get_total_macs(self):
            return 0

        def get_total_params(self):
            return 0

        def print_model_profile(self, *args, **kwargs):
            pass

        def end_profile(self):
            pass


try:
    import pynvml

    _NVML_AVAILABLE = True
    pynvml.nvmlInit()
except Exception:
    _NVML_AVAILABLE = False


def _append_jsonl(path: str, payload: dict) -> None:
    """Append one metrics row to JSONL file from rank-0 only."""
    if not path or not is_main_process():
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _uses_custom_recurrence_forward(module) -> bool:
    """
    Detect recurrence models that use custom forward signature:
      forward(input_ids, next_token_ids=..., return_loss=..., ...)

    This covers reversible and non-reversible Model1B variants.
    """
    try:
        params = inspect.signature(module.forward).parameters
    except Exception:
        return False
    return "next_token_ids" in params and "return_loss" in params


def _format_log_timestamp() -> str:
    """Return wall-clock timestamp in logger style with millisecond precision."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]


def _get_learning_rate(model_engine):
    """
    Best-effort extraction of current learning rate from DeepSpeed engine.
    Returns float or None.
    """
    try:
        lr_val = model_engine.get_lr()
        if isinstance(lr_val, (list, tuple)):
            return float(lr_val[0]) if lr_val else None
        if lr_val is not None:
            return float(lr_val)
    except Exception:
        pass

    optimizer = getattr(model_engine, "optimizer", None)
    if optimizer is not None and hasattr(optimizer, "param_groups"):
        groups = optimizer.param_groups
        if groups and "lr" in groups[0]:
            return float(groups[0]["lr"])
    return None


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _try_call_noarg(obj, method_name: str):
    if obj is None:
        return False, None
    fn = getattr(obj, method_name, None)
    if not callable(fn):
        return False, None
    try:
        fn()
        return True, None
    except TypeError as exc:
        return False, f"{method_name} requires args ({exc})"
    except Exception as exc:
        return False, f"{method_name} failed ({type(exc).__name__}: {exc})"


def _release_zero3_runtime_buffers(model_engine):
    """
    Best-effort release of ZeRO-3 runtime caches.
    Guarded so unknown DeepSpeed builds still run cleanly.
    """
    called = []
    errors = []
    optimizer = getattr(model_engine, "optimizer", None)
    parameter_offload = getattr(optimizer, "parameter_offload", None)
    coordinator = getattr(parameter_offload, "partitioned_param_coordinator", None)

    candidates = [
        ("engine", model_engine, ("empty_partition_cache",)),
        ("optimizer", optimizer, ("empty_partition_cache",)),
        (
            "parameter_offload",
            parameter_offload,
            (
                "empty_partition_cache",
                "release_and_reset_all",
                "_release_and_reset_all",
            ),
        ),
        (
            "partitioned_param_coordinator",
            coordinator,
            (
                "empty_partition_cache",
                "release_and_reset_all",
                "_release_and_reset_all",
                "reset_step",
            ),
        ),
    ]

    for owner_name, obj, method_names in candidates:
        for method_name in method_names:
            ok, err = _try_call_noarg(obj, method_name)
            if ok:
                called.append(f"{owner_name}.{method_name}")
            elif err is not None:
                errors.append(f"{owner_name}.{err}")
    return called, errors


def _clear_router_topk_caches(module) -> int:
    """
    Drop reversible MoE router top-k caches after optimizer step.
    """
    if module is None:
        return 0
    cleared = 0
    for submodule in module.modules():
        if (
            hasattr(submodule, "_cached_topk_idx")
            and getattr(submodule, "_cached_topk_idx") is not None
        ):
            submodule._cached_topk_idx = None
            cleared += 1
    return cleared


def _force_clear_zero3_containers(model_engine):
    """
    Aggressive fallback: clear likely cache containers inside ZeRO internals.
    Intended for leak triage only (opt-in via env).
    """
    touched = []
    optimizer = getattr(model_engine, "optimizer", None)
    parameter_offload = getattr(optimizer, "parameter_offload", None)
    coordinator = getattr(parameter_offload, "partitioned_param_coordinator", None)
    targets = [
        ("optimizer", optimizer),
        ("parameter_offload", parameter_offload),
        ("partitioned_param_coordinator", coordinator),
    ]
    key_tokens = ("cache", "prefetch", "inflight", "queue", "trace")

    for owner_name, obj in targets:
        if obj is None or not hasattr(obj, "__dict__"):
            continue
        for key, value in obj.__dict__.items():
            k = str(key).lower()
            if not any(token in k for token in key_tokens):
                continue
            if isinstance(value, dict):
                if value:
                    value.clear()
                    touched.append(f"{owner_name}.{key}(dict)")
            elif isinstance(value, list):
                if value:
                    value.clear()
                    touched.append(f"{owner_name}.{key}(list)")
            elif isinstance(value, set):
                if value:
                    value.clear()
                    touched.append(f"{owner_name}.{key}(set)")
    return touched


def _opus_score_and_select(
    model_engine,
    opus_loader,
    opus_components,
    device,
    global_step,
):
    """
    Run one OPUS scoring pass: load candidates, score at 512 tokens, select ρ fraction.

    Returns:
        selected_input_ids: [n_selected, full_seq_len] — full 4096-token sequences
        selected_attention_mask: [n_selected, full_seq_len]
        selected_labels: [n_selected, full_seq_len]
        opus_metrics: dict with scoring diagnostics
    """
    from .opus import OpusGhostCollector

    preconditioner = opus_components["preconditioner"]
    sketcher = opus_components["sketcher"]
    selector = opus_components["selector"]
    proxy = opus_components["proxy"]
    n_proxy = opus_components["n_proxy"]
    score_seq_len = opus_components["score_seq_len"]
    score_layer_stride = opus_components["score_layer_stride"]

    # 0. Free VRAM before scoring — ZeRO-3 leaves memory nearly full
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1. Load a full batch of candidates from the OPUS loader (D1-D4)
    try:
        cand_batch = next(opus_loader._iter)
    except (StopIteration, AttributeError):
        opus_loader._iter = iter(opus_loader)
        cand_batch = next(opus_loader._iter)

    cand_ids = cand_batch["input_ids"]  # [N, 4096]
    cand_mask = cand_batch["attention_mask"]  # [N, 4096]
    cand_labels = cand_batch["labels"]  # [N, 4096]
    n_candidates = cand_ids.shape[0]

    # 2. Get proxy samples (at score_seq_len)
    proxy_ids = proxy.sample(
        device=device, k=n_proxy, seq_len=score_seq_len
    )  # [n_proxy, 512]

    # 3. Truncate candidates to score_seq_len for scoring
    cand_ids_short = cand_ids[:, :score_seq_len].to(
        device, non_blocking=True
    )  # [N, 512]
    proxy_ids = proxy_ids.to(device, non_blocking=True)  # [K, 512]

    # 4. Concatenate proxy + candidates for single forward/backward pass
    # Ghost hooks will split them by n_proxy boundary
    combined_ids = torch.cat([proxy_ids, cand_ids_short], dim=0)  # [K+N, 512]

    # 5. Create ghost collector and register hooks
    ghost = OpusGhostCollector(
        model=model_engine.module,
        n_proxy=n_proxy,
        n_candidates=n_candidates,
        preconditioner=preconditioner,
        sketcher=sketcher,
        device=device,
        score_layer_stride=score_layer_stride,
    )

    ghost.register()

    try:
        # 6. Forward pass at 512 tokens (scoring only, MTP disabled to save VRAM)
        torch.cuda.empty_cache()  # Reclaim fragmented memory before scoring
        model_engine.module.eval()

        with torch.enable_grad():
            uses_custom = _uses_custom_recurrence_forward(model_engine.module)
            if uses_custom:
                # MTP OFF for scoring: pass next_token_ids=None to skip MTP block.
                # Saves ~3-4GB VRAM (one transformer block of activations+grads).
                # Ghost hooks only fire on main stack layers, not MTP block.
                x_score = combined_ids[:, :-1].contiguous()
                y_ntp_score = combined_ids[:, 1:].contiguous()

                h_ntp, _, _ = model_engine.module(
                    x_score,
                    next_token_ids=None,  # MTP OFF — save VRAM during scoring
                    attention_mask=None,
                    return_loss=True,
                    return_memory=False,
                    prev_memory_stream=None,
                    return_hidden=True,
                )

                # Use fused linear CE to avoid materializing [N*T, vocab] logits
                # (~14 GB for BS=52 × 510 × 131072). Ghost hooks only fire on
                # internal layers, so this is safe.
                # ZeRO-3: lm_head weight is partitioned; gather + clone so that
                # backward doesn't hit the partitioned param (shape mismatch).
                # Clone is ~1GB (131072×4096×2 bf16) — acceptable.
                import deepspeed

                lm_param = model_engine.module.lm_head.weight
                B_s, T_s, H_s = h_ntp.shape
                with deepspeed.zero.GatheredParameters([lm_param], modifier_rank=None):
                    lm_weight = lm_param.data.clone()  # [V, H] full copy, no grad
                score_loss = _FusedLinearCE(
                    ignore_index=-100,
                    reduction="mean",
                    max_chunk_gb=1.0,
                )(h_ntp.view(-1, H_s), lm_weight, y_ntp_score.reshape(-1))
            else:
                x_score = combined_ids[:, :-1].contiguous()
                y_score = combined_ids[:, 1:].contiguous()
                outputs = model_engine.module(x_score, labels=y_score)
                score_loss = outputs.loss

            # 7. Backward to trigger ghost hooks
            score_loss.backward()

        # 8. Collect results from ghost hooks (local scores, no all_reduce)
        alignment_scores, candidate_sketches = ghost.results()

        _as_float = alignment_scores.float()
        _as_stats = {
            "align_mean": float(_as_float.mean().item()),
            "align_std": (
                float(_as_float.std().item()) if _as_float.numel() > 1 else 0.0
            ),
            "align_min": float(_as_float.min().item()),
            "align_max": float(_as_float.max().item()),
            "align_median": float(_as_float.median().item()),
            "n_layers_scored": len(candidate_sketches),
        }
        if is_main_process():
            print(
                f"[OPUS] Alignment stats: mean={_as_stats['align_mean']:.4f} "
                f"std={_as_stats['align_std']:.4f} "
                f"min={_as_stats['align_min']:.4f} max={_as_stats['align_max']:.4f} "
                f"median={_as_stats['align_median']:.4f} "
                f"layers={_as_stats['n_layers_scored']}",
                flush=True,
            )

        # 9. Run OPUS selector (local top-k with Gumbel noise)
        lr = _get_learning_rate(model_engine)
        selection_result = selector.select(
            alignment_scores=alignment_scores,
            candidate_sketches=candidate_sketches,
            learning_rate=lr,
        )

    finally:
        ghost.unregister()
        ghost.clear()
        # Zero scoring gradients (set_to_none=False to keep tensor objects
        # intact — set_to_none=True causes ZeRO-3 crash on param.grad.numel()).
        model_engine.module.zero_grad(set_to_none=False)
        # Reset ZeRO-3's internal gradient state so the training step's
        # backward doesn't find stale state from the scoring backward.
        if hasattr(model_engine, "optimizer") and hasattr(
            model_engine.optimizer, "reset_step"
        ):
            model_engine.optimizer.reset_step()
        model_engine.module.train()

    # 10. Index into FULL 4096-token sequences using selected local indices
    sel_idx = selection_result.selected_local_indices.cpu()
    selected_ids = cand_ids[sel_idx]  # [n_selected, 4096]
    selected_mask = cand_mask[sel_idx]  # [n_selected, 4096]
    selected_labels = cand_labels[sel_idx]  # [n_selected, 4096]

    opus_metrics = {
        "n_candidates": n_candidates,
        "n_selected": len(sel_idx),
        "used_fallback": selection_result.used_fallback,
        "score_loss": float(score_loss.detach().item()),
        **_as_stats,
        **selection_result.metrics,
    }

    # Clean up scoring tensors
    del combined_ids, cand_ids_short, proxy_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return selected_ids, selected_mask, selected_labels, opus_metrics


def train_epoch(
    model_engine,
    train_loader,
    epoch,
    max_steps=None,
    max_train_seconds=None,
    log_interval=10,
    enable_system_metrics=False,
    checkpoint_interval=None,
    output_dir=None,
    checkpoint_manager=None,
    start_step=0,
    global_step=0,
    metrics_jsonl_path=None,
    max_chunk_gb=8.0,
    profiler: "StepProfiler | None" = None,
    profile_steps: "set | None" = None,
    profile_output_dir: "str | None" = None,
    training_ops=None,
    spot_orchestrator=None,
    lr_scheduler=None,
    opus_components=None,
    opus_loader=None,
    aon_loader=None,
):
    """
    Train the model for one epoch.

    Args:
        model_engine: DeepSpeed model engine
        train_loader: DataLoader for training data
        epoch: Current epoch number
        max_steps: Maximum number of steps per epoch (None for full epoch)
        log_interval: Log every N steps
        checkpoint_interval: Save checkpoint every N steps (None to disable)
        output_dir: Directory to save checkpoints (required if checkpoint_interval is set)
        checkpoint_manager: S3CheckpointManager instance (optional, for S3 support)
        start_step: Step to start from (for resuming)
        global_step: Global step counter across all epochs

    Returns:
        Tuple of (average_loss, final_global_step)
    """
    model_engine.train()
    total_loss = 0
    steps = 0
    _epoch_wall_start = time.time()
    _total_tokens_processed = 0

    # ── OPUS amortized buffer ──────────────────────────────────────────────
    _opus_enabled = opus_components is not None
    _opus_buffer_ids = None  # [remaining, seq_len] — OPUS-selected sequences
    _opus_buffer_mask = None
    _opus_buffer_labels = None
    _opus_buffer_cursor = 0  # Next index to drain from buffer
    _opus_aon_iter = None
    _opus_loader_iter = None
    _opus_scoring_count = 0  # How many OPUS scoring passes have been run

    if _opus_enabled:
        _opus_cand_per_step = opus_components["candidates_per_step"]  # 24
        _opus_aon_per_step = opus_components["aon_per_step"]  # 8
        _opus_aon_iter = iter(aon_loader)
        _opus_loader_iter = iter(opus_loader)
        # Attach iterator to opus_loader for _opus_score_and_select
        opus_loader._iter = _opus_loader_iter
        print_rank_0(
            f"[OPUS] Amortized mode: {_opus_cand_per_step} OPUS + "
            f"{_opus_aon_per_step} AON = {_opus_cand_per_step + _opus_aon_per_step}/step"
        )

    # FIX-PERF-07: Use dynamic chunk size from config (default 4GB)
    # Autoresearch: softcap bounds logits to (-c, c) via tanh (Gemma2/nanochat style)
    _softcap = float(os.getenv("EXP_SOFTCAP", "0"))
    fused_ce_fn = _FusedLinearCE(
        ignore_index=-100, reduction="mean", max_chunk_gb=max_chunk_gb, softcap=_softcap
    )
    if _softcap > 0:
        print_rank_0(f"[EXP] Logit softcap = {_softcap}")
    # Autoresearch: MTP loss weight (default 0.3)
    _mtp_w = float(os.getenv("EXP_MTP_WEIGHT", "0"))
    _mtp_w = _mtp_w if _mtp_w > 0 else 0.3
    if abs(_mtp_w - 0.3) > 1e-6:
        print_rank_0(f"[EXP] MTP weight = {_mtp_w}")

    # Runtime hardening for step-wise VRAM growth (especially ZeRO-3 + reversible + MoE).
    cleanup_sync = _env_flag("T19_STEP_CUDA_SYNC", "1")
    cleanup_gc_collect = _env_flag("T19_STEP_GC_COLLECT", "1")
    cleanup_empty_cache = _env_flag("T19_STEP_EMPTY_CACHE", "1")
    cleanup_ipc_collect = _env_flag("T19_STEP_IPC_COLLECT", "0")
    zero3_release_every = max(0, _env_int("T19_ZERO3_RELEASE_EVERY", 1))
    clear_router_cache_every = max(0, _env_int("T19_CLEAR_ROUTER_CACHE_EVERY", 1))
    zero3_force_clear_containers = _env_flag("T19_ZERO3_FORCE_CLEAR_CONTAINERS", "0")
    track_cuda_memory = _env_flag("T19_TRACK_CUDA_MEMORY", "1")
    _zero3_release_logged = False
    _zero3_release_error_logged = False
    _zero3_force_clear_logged = False
    prev_alloc_gb = None
    prev_reserved_gb = None

    # --- Log library versions at training start ---
    import fla as _fla
    import triton as _triton

    print_rank_0(
        f"[versions] torch={torch.__version__} triton={_triton.__version__} "
        f"deepspeed={__import__('deepspeed').__version__} fla={_fla.__version__}"
    )

    print_rank_0(
        "[T19 hardening] "
        f"cuda_sync={int(cleanup_sync)} "
        f"gc_collect={int(cleanup_gc_collect)} "
        f"empty_cache={int(cleanup_empty_cache)} "
        f"ipc_collect={int(cleanup_ipc_collect)} "
        f"zero3_release_every={zero3_release_every} "
        f"clear_router_cache_every={clear_router_cache_every} "
        f"zero3_force_clear_containers={int(zero3_force_clear_containers)} "
        f"track_cuda_memory={int(track_cuda_memory)}"
    )

    # ── Step profiler setup ──────────────────────────────────────────────────
    # Auto-create a profiler if profile_steps were provided but no instance passed.
    _owns_profiler = False
    if profiler is None and profile_steps:
        local_rank = getattr(model_engine, "local_rank", 0)
        _pout = profile_output_dir or (
            os.path.dirname(metrics_jsonl_path) if metrics_jsonl_path else "results/run"
        )
        profiler = StepProfiler(
            rank=local_rank,
            profile_steps=set(profile_steps),
            output_dir=_pout,
        )
        profiler.activate()
        profiler.register_model(model_engine.module)
        _owns_profiler = True
        print_rank_0(f"[profiler] Enabled for steps: {sorted(profile_steps)}")

    # Only show progress bar on main process
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=True)

    profile_step = 10
    print_profile = True
    prof = FlopsProfiler(model_engine)

    # When resuming, create an iterator that skips already-completed steps
    # without pulling batches from the dataloader (which would advance
    # the curriculum shard state incorrectly).
    _loader_iter = iter(progress_bar)
    if start_step > 0:
        print_rank_0(
            f"  Resuming from step {start_step} (skipping dataloader fast-forward)"
        )

    for i in range(start_step, max_steps if max_steps is not None else 10**9):
        _opus_scored_this_step = False
        _opus_step_metrics = None

        if _opus_enabled:
            # ── OPUS: refill buffer if empty ───────────────────────────────
            _buf_remaining = (
                (_opus_buffer_ids.shape[0] - _opus_buffer_cursor)
                if _opus_buffer_ids is not None
                else 0
            )
            if _buf_remaining < _opus_cand_per_step:
                _score_t0 = time.time()
                print_rank_0(
                    f"[OPUS] Scoring pass #{_opus_scoring_count + 1} at step {i} "
                    f"(buffer had {_buf_remaining} remaining)"
                )
                try:
                    (
                        _opus_buffer_ids,
                        _opus_buffer_mask,
                        _opus_buffer_labels,
                        _opus_step_metrics,
                    ) = _opus_score_and_select(
                        model_engine=model_engine,
                        opus_loader=opus_loader,
                        opus_components=opus_components,
                        device=model_engine.device,
                        global_step=global_step,
                    )
                    _opus_buffer_cursor = 0
                    _opus_scoring_count += 1
                    _opus_scored_this_step = True
                    _score_dt = time.time() - _score_t0
                    print_rank_0(
                        f"[OPUS] Scored {_opus_step_metrics['n_candidates']} candidates → "
                        f"selected {_opus_step_metrics['n_selected']} in {_score_dt:.2f}s "
                        f"(fallback={_opus_step_metrics['used_fallback']})"
                    )
                    print_rank_0(
                        f"[OPUS] score_loss={_opus_step_metrics.get('score_loss', 'n/a'):.4f} "
                        f"align={_opus_step_metrics.get('alignment', 0):.4f} "
                        f"redun={_opus_step_metrics.get('redundancy', 0):.4f} "
                        f"selector_t={_opus_step_metrics.get('selector_time_s', 0):.2f}s "
                        f"align_mean={_opus_step_metrics.get('align_mean', 0):.4f} "
                        f"align_std={_opus_step_metrics.get('align_std', 0):.4f}"
                    )
                except Exception as e:
                    import traceback

                    print_rank_0(
                        f"[OPUS] Scoring failed: {e} — falling back to combined loader"
                    )
                    print_rank_0(f"[OPUS] Traceback:\n{traceback.format_exc()}")
                    _opus_enabled = False  # Disable for rest of epoch

            if _opus_enabled:
                # ── Drain OPUS buffer ──────────────────────────────────────
                end_idx = min(
                    _opus_buffer_cursor + _opus_cand_per_step, _opus_buffer_ids.shape[0]
                )
                opus_ids = _opus_buffer_ids[_opus_buffer_cursor:end_idx]
                opus_mask = _opus_buffer_mask[_opus_buffer_cursor:end_idx]
                opus_labels = _opus_buffer_labels[_opus_buffer_cursor:end_idx]
                _opus_buffer_cursor = end_idx

                # ── Fetch AON batch ────────────────────────────────────────
                try:
                    aon_batch = next(_opus_aon_iter)
                except StopIteration:
                    _opus_aon_iter = iter(aon_loader)
                    aon_batch = next(_opus_aon_iter)

                aon_ids = aon_batch["input_ids"]
                aon_mask = aon_batch["attention_mask"]
                aon_labels = aon_batch["labels"]

                # ── Combine OPUS + AON ─────────────────────────────────────
                batch = {
                    "input_ids": torch.cat([opus_ids, aon_ids], dim=0),
                    "attention_mask": torch.cat([opus_mask, aon_mask], dim=0),
                    "labels": torch.cat([opus_labels, aon_labels], dim=0),
                }
                # Log batch composition every 10 steps
                if i % 10 == 0:
                    _buf_total = (
                        _opus_buffer_ids.shape[0] if _opus_buffer_ids is not None else 0
                    )
                    print_rank_0(
                        f"[OPUS] Step {i}: batch={opus_ids.shape[0]} OPUS + "
                        f"{aon_ids.shape[0]} AON = {batch['input_ids'].shape[0]} total | "
                        f"buffer: {_opus_buffer_cursor}/{_buf_total} consumed "
                        f"(scoring #{_opus_scoring_count})"
                    )
            else:
                # OPUS disabled mid-epoch due to error, fall through to normal loader
                try:
                    batch = next(_loader_iter)
                except StopIteration:
                    break
        else:
            # ── Normal (non-OPUS) batch fetch ──────────────────────────────
            try:
                batch = next(_loader_iter)
            except StopIteration:
                break

        outputs = None
        h_ntp = None
        h_mtp = None
        aux_loss = None
        loss = None
        loss_ntp = None
        loss_mtp = None
        aux_term = None
        x_input = None
        y_ntp = None
        y_mtp = None

        if i == profile_step:
            print("Profile started")
            prof.start_profile()

        # ── Profiler: start step ─────────────────────────────────────────────
        _batch_tokens_for_profiler = (
            batch["attention_mask"].sum().item() if "attention_mask" in batch else 0
        )
        if profiler is not None:
            profiler.start_step(global_step + 1, tokens=int(_batch_tokens_for_profiler))

        # Measure step wall-clock time
        step_start_time = time.time()

        # ── Profiler: dataloader + device transfer ───────────────────────────
        _profiler_ctx = (
            profiler.phase("dataloader") if profiler is not None else _null_ctx()
        )
        with _profiler_ctx:
            input_ids = batch["input_ids"].to(model_engine.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(
                model_engine.device, non_blocking=True
            )
            labels = batch["labels"].to(model_engine.device, non_blocking=True)

        # Memory profiling on first step
        if i == 0:
            torch.cuda.reset_peak_memory_stats(model_engine.device)
            mem_before = torch.cuda.memory_allocated(model_engine.device) / 1e9
            print_rank_0(f"\n[MEMORY] Before forward pass: {mem_before:.2f}GB")

        # Forward pass
        # Recurrence models use a custom forward signature (not labels=...)
        uses_custom_forward = _uses_custom_recurrence_forward(model_engine.module)
        gsa_leak_frac = None
        gsa_leak_attempt_frac = None
        loss_ntp_value = None
        loss_mtp_value = None
        loss_aux_value = None

        if uses_custom_forward:
            # Reversible model: returns (h_ntp, h_mtp, aux_loss) hidden states
            # (NOT logits — lm_head is skipped; FusedLinearCE fuses matmul+CE below)
            x_input = input_ids[:, :-2].contiguous()
            y_ntp = input_ids[:, 1:-1].contiguous()
            y_mtp = input_ids[:, 2:].contiguous()

            # FIX: Call model_engine(...) not model_engine.module(...)
            # This ensures DeepSpeed's BF16, gradient hooks, and ZeRO all fire correctly.
            with profiler.phase("forward") if profiler is not None else _null_ctx():
                h_ntp, h_mtp, aux_loss = model_engine(
                    x_input,
                    next_token_ids=y_ntp,
                    attention_mask=(
                        attention_mask[:, :-2].contiguous()
                        if attention_mask is not None
                        else None
                    ),
                    return_loss=True,
                    return_memory=False,
                    prev_memory_stream=None,
                    return_hidden=True,  # Skip lm_head — we compute CE below
                )
            with (
                profiler.phase("gsa_leak_allreduce")
                if profiler is not None
                else _null_ctx()
            ):
                leak_frac_t = getattr(
                    model_engine.module, "last_gsa_leak_fraction", None
                )
                leak_attempt_t = getattr(
                    model_engine.module, "last_gsa_leak_attempt_fraction", None
                )
                if leak_frac_t is not None:
                    leak_frac_t = leak_frac_t.detach().float()
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(leak_frac_t, op=dist.ReduceOp.SUM)
                        leak_frac_t = leak_frac_t / dist.get_world_size()
                    gsa_leak_frac = float(leak_frac_t.item())
                if leak_attempt_t is not None:
                    leak_attempt_t = leak_attempt_t.detach().float()
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(leak_attempt_t, op=dist.ReduceOp.SUM)
                        leak_attempt_t = leak_attempt_t / dist.get_world_size()
                    gsa_leak_attempt_frac = float(leak_attempt_t.item())
            # Regression guard: if this ever fires, sparse selection let future
            # tokens through and the training loss is no longer trustworthy.
            if gsa_leak_frac is not None and gsa_leak_frac > 1e-12:
                raise RuntimeError(
                    f"GSA causal leak regression detected: gsa_leak_fraction={gsa_leak_frac:.6e}"
                )

            # Memory profiling after forward
            if i == 0:
                mem_after_fwd = torch.cuda.memory_allocated(model_engine.device) / 1e9
                mem_peak = torch.cuda.max_memory_allocated(model_engine.device) / 1e9
                print_rank_0(
                    f"[MEMORY] After forward pass: {mem_after_fwd:.2f}GB (peak: {mem_peak:.2f}GB)"
                )
                print_rank_0(
                    f"[MEMORY] Forward allocated: {(mem_after_fwd - mem_before):.2f}GB"
                )

            # 3. FusedLinearCE: fuses lm_head matmul + CE in one chunked kernel.
            # Never materialises [B*T, vocab] logits. Zero fallback.
            #
            # ZeRO-3: gather lm_head around CE compute, but avoid full weight cloning.
            # Cloning [V,H] each step adds a large transient allocation and can amplify
            # fragmentation before optimizer step.
            _lm_param = model_engine.module.lm_head.weight
            _is_zero3 = hasattr(_lm_param, "ds_id")
            if _is_zero3:
                try:
                    from deepspeed.zero import GatheredParameters
                except ImportError:
                    from deepspeed.runtime.zero.partition_parameters import (
                        GatheredParameters,
                    )
                _gather_ctx = GatheredParameters([_lm_param], modifier_rank=None)
            else:
                _gather_ctx = _null_ctx()

            with _gather_ctx:
                lm_weight = _lm_param.data

                B_seq, T_seq, H_dim = h_ntp.shape
                vocab_size = lm_weight.shape[0]

                with (
                    profiler.phase("fused_ce") if profiler is not None else _null_ctx()
                ):
                    loss_ntp = fused_ce_fn(
                        h_ntp.view(-1, H_dim),  # [B*T, H]
                        lm_weight,  # [V, H]
                        y_ntp.view(-1),  # [B*T]
                    )
                if i == 0:
                    mem_after_loss_ntp = (
                        torch.cuda.memory_allocated(model_engine.device) / 1e9
                    )
                    print_rank_0(f"[MEMORY] After loss_ntp: {mem_after_loss_ntp:.2f}GB")

                loss_mtp = None
                if h_mtp is not None:
                    B_m, T_m, H_m = h_mtp.shape
                    with (
                        profiler.phase("fused_ce_mtp")
                        if profiler is not None
                        else _null_ctx()
                    ):
                        loss_mtp = fused_ce_fn(
                            h_mtp.view(-1, H_m),  # [B*T, H]
                            lm_weight,  # [V, H]
                            y_mtp.view(-1),  # [B*T]
                        )

            # 4. NaN Watchdog — HARD CRASH (FIX: was silently continuing, corrupting weights)
            if (
                torch.isnan(loss_ntp)
                or (loss_mtp is not None and torch.isnan(loss_mtp))
                or (aux_loss is not None and torch.isnan(aux_loss))
            ):
                raise RuntimeError(
                    f"NaN detected at epoch {epoch}, step {i}: "
                    f"loss_ntp={loss_ntp.item():.4f}, "
                    f"loss_mtp={loss_mtp.item():.4f if loss_mtp is not None else 'None'}"
                )

            # 5. Combine Loss (NTP + 0.3*MTP + aux)
            loss = loss_ntp.squeeze()
            if loss_mtp is not None:
                loss = loss + _mtp_w * loss_mtp.squeeze()
            if aux_loss is not None and aux_loss.numel() > 0:
                # Defensive scalarization: some model variants may return
                # aux tensors with more than one element. Ensure 0-dim to avoid
                # broadcast shape mismatch ([] vs [1]) when adding to loss.
                aux_term = (
                    aux_loss if aux_loss.numel() == 1 else aux_loss.mean()
                ).squeeze()
                loss = loss + aux_term
                loss_aux_value = float(aux_term.detach().float().item())
            else:
                loss_aux_value = 0.0
            loss_ntp_value = float(loss_ntp.detach().float().item())
            loss_mtp_value = (
                float(loss_mtp.detach().float().item()) if loss_mtp is not None else 0.0
            )

            # Memory profiling before backward
            if i == 0:
                mem_before_bwd = torch.cuda.memory_allocated(model_engine.device) / 1e9
                print_rank_0(f"[MEMORY] Before backward: {mem_before_bwd:.2f}GB")
                print_rank_0("[MEMORY] === MEMORY SUMMARY ===")
                print_rank_0(
                    torch.cuda.memory_summary(
                        device=model_engine.device, abbreviated=True
                    )
                )

            # MTP loss (if enabled and logits_mtp is not None)
            # For now, we focus on NTP loss

        else:
            # Standard transformer model
            with profiler.phase("forward") if profiler is not None else _null_ctx():
                outputs = model_engine(
                    input_ids, attention_mask=attention_mask, labels=labels
                )
            loss = outputs.loss
            loss_ntp_value = float(loss.detach().float().item())
            loss_mtp_value = None
            loss_aux_value = None

        # Backward pass
        with profiler.phase("backward") if profiler is not None else _null_ctx():
            model_engine.backward(loss)

        # Update weights (includes allreduce in ZeRO-1)
        with profiler.phase("optim_step") if profiler is not None else _null_ctx():
            model_engine.step()

        # Apply staged LR schedule (overrides DeepSpeed's built-in scheduler)
        _scheduled_lr = None
        if lr_scheduler is not None:
            _scheduled_lr = lr_scheduler.step(model_engine.optimizer, global_step)

        # Collect MoE router stats BEFORE bias update clears them.
        _moe_null_rate = None
        _moe_expert_counts = None
        _unwrapped_for_stats = model_engine.module
        for _gate_mod in _unwrapped_for_stats.modules():
            if (
                hasattr(_gate_mod, "_last_null_rate")
                and _gate_mod._last_null_rate is not None
            ):
                _moe_null_rate = float(_gate_mod._last_null_rate.item())
            if (
                hasattr(_gate_mod, "_last_expert_counts")
                and _gate_mod._last_expert_counts is not None
            ):
                _moe_expert_counts = _gate_mod._last_expert_counts.tolist()
                break

        if i == 0:
            mem_after_step = torch.cuda.memory_allocated(model_engine.device) / 1e9
            print_rank_0(f"[MEMORY] After backward & step: {mem_after_step:.2f}GB")

        # Compute tokens per second for this step
        step_time = time.time() - step_start_time

        # ── Profiler: record total step time and finalize ────────────────────
        if profiler is not None:
            with torch.no_grad():
                _ptoks = attention_mask.sum().float()
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(_ptoks, op=dist.ReduceOp.SUM)
                profiler._current and profiler._current.add(
                    "step_total", step_time * 1000.0
                )
            profiler.end_step(tokens=int(_ptoks.item()))
        step_dt_ms = step_time * 1000.0
        with (
            profiler.phase("token_count_allreduce")
            if profiler is not None
            else _null_ctx()
        ):
            with torch.no_grad():
                # Count tokens in this batch using attention mask (1s for real tokens)
                tokens = attention_mask.sum().float()
                # Aggregate across all ranks if distributed is initialized
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
                tokens = tokens.item()
        tokens_per_sec = tokens / step_time if step_time > 0 else 0.0
        _total_tokens_processed += int(tokens)
        learning_rate = (
            _scheduled_lr
            if _scheduled_lr is not None
            else _get_learning_rate(model_engine)
        )
        loss_scalar = float(loss.item())
        gpu_alloc_gb = None
        gpu_reserved_gb = None
        gpu_peak_reserved_gb = None
        gpu_peak_alloc_gb = None
        gpu_alloc_delta_gb = None
        gpu_reserved_delta_gb = None
        if track_cuda_memory and torch.cuda.is_available():
            gpu_alloc_gb = torch.cuda.memory_allocated(model_engine.device) / 1e9
            gpu_reserved_gb = torch.cuda.memory_reserved(model_engine.device) / 1e9
            gpu_peak_reserved_gb = (
                torch.cuda.max_memory_reserved(model_engine.device) / 1e9
            )
            gpu_peak_alloc_gb = (
                torch.cuda.max_memory_allocated(model_engine.device) / 1e9
            )
            if prev_alloc_gb is not None:
                gpu_alloc_delta_gb = gpu_alloc_gb - prev_alloc_gb
            if prev_reserved_gb is not None:
                gpu_reserved_delta_gb = gpu_reserved_gb - prev_reserved_gb
            prev_alloc_gb = gpu_alloc_gb
            prev_reserved_gb = gpu_reserved_gb

        # Optional system metrics (CPU/GPU util & memory)
        gpu_util = gpu_mem_used = gpu_mem_total = None
        cpu_util = cpu_mem_used = cpu_mem_total = None
        if enable_system_metrics:
            with (
                profiler.phase("system_metrics")
                if profiler is not None
                else _null_ctx()
            ):
                # CPU metrics
                vm = psutil.virtual_memory()
                cpu_util = psutil.cpu_percent(interval=None)
                cpu_mem_used = vm.used / (1024**3)
                cpu_mem_total = vm.total / (1024**3)

                # GPU metrics (only on main process to avoid spam)
                if _NVML_AVAILABLE and is_main_process() and torch.cuda.is_available():
                    try:
                        # Collect metrics for all visible GPUs
                        n_devices = pynvml.nvmlDeviceGetCount()
                        gpu_rows = []
                        for idx in range(n_devices):
                            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            used_gb = mem_info.used / (1024**3)
                            total_gb = mem_info.total / (1024**3)
                            util = util_info.gpu
                            gpu_rows.append((idx, util, used_gb, total_gb))

                        # For scalar summary fields, use the device this rank is bound to
                        device_index = (
                            model_engine.local_rank
                            if hasattr(model_engine, "local_rank")
                            else torch.cuda.current_device()
                        )
                        _, gpu_util, gpu_mem_used, gpu_mem_total = next(
                            (r for r in gpu_rows if r[0] == int(device_index)),
                            (device_index, None, None, None),
                        )

                        # Build a neat table string for all GPUs
                        header = "GPU  Util(%)  Mem(GB Used/Total)"
                        lines = [header, "-" * len(header)]
                        for idx, util, used_gb, total_gb in gpu_rows:
                            lines.append(
                                f"{idx:<3}  {util:>6.0f}%  {used_gb:>5.1f}G/{total_gb:>5.1f}G"
                            )
                        gpu_table = "\n".join(lines)
                    except Exception:
                        gpu_table = None
                        # Fail silently if NVML query fails
                        pass

        if i == profile_step:
            print("Profile stoped\n")
            prof.stop_profile()
            flops = prof.get_total_flops()
            macs = prof.get_total_macs()
            params = prof.get_total_params()
            if print_profile:
                prof.print_model_profile(profile_step=profile_step)
            prof.end_profile()
        # MoE router stats (collected before bias update above).
        z_loss_value = (
            loss_aux_value
            if (loss_aux_value is not None and loss_aux_value > 0)
            else None
        )
        null_rate_value = _moe_null_rate
        expert_counts_list = _moe_expert_counts

        # Track metrics — use NTP loss (not combined) for avg_loss reporting
        total_loss += loss_ntp_value if loss_ntp_value is not None else loss_scalar
        steps += 1
        global_step += 1

        # Update progress bar
        postfix = {
            "loss": (
                f"{loss_ntp_value:.4f}"
                if loss_ntp_value is not None
                else f"{loss_scalar:.4f}"
            ),
            "global_step": global_step,
            "toks/s": f"{tokens_per_sec:.1f}",
        }
        if loss_mtp_value is not None:
            postfix["loss2"] = f"{loss_mtp_value:.4f}"
        if z_loss_value is not None:
            postfix["z_loss"] = f"{z_loss_value:.6f}"
        if null_rate_value is not None:
            postfix["null_rate"] = f"{null_rate_value:.3f}"
        if gsa_leak_frac is not None:
            postfix["gsa_leak"] = f"{gsa_leak_frac:.6f}"
        if gsa_leak_attempt_frac is not None:
            postfix["gsa_leak_try"] = f"{gsa_leak_attempt_frac:.6f}"
        if enable_system_metrics and gpu_mem_used is not None:
            postfix["gpu_util"] = f"{gpu_util:.0f}%"
            postfix["gpu_mem"] = f"{gpu_mem_used:.1f}G"
        if enable_system_metrics and cpu_util is not None:
            postfix["cpu_util"] = f"{cpu_util:.0f}%"
            postfix["cpu_mem"] = f"{cpu_mem_used:.1f}G"
        if gpu_alloc_gb is not None:
            postfix["vram_alloc"] = f"{gpu_alloc_gb:.1f}G"
        if gpu_peak_reserved_gb is not None:
            postfix["peak"] = f"{gpu_peak_reserved_gb:.1f}G"
        if gpu_alloc_delta_gb is not None:
            postfix["d_alloc"] = f"{gpu_alloc_delta_gb:+.2f}G"
        progress_bar.set_postfix(postfix)

        # Log periodically
        if i % log_interval == 0:
            with profiler.phase("log_write") if profiler is not None else _null_ctx():
                loss_str = (
                    f"{loss_ntp_value:.4f}" if loss_ntp_value is not None else "nan"
                )
                loss2_str = (
                    f"{loss_mtp_value:.4f}" if loss_mtp_value is not None else "nan"
                )
                vram_alloc_str = (
                    f"{gpu_alloc_gb:.1f}G" if gpu_alloc_gb is not None else "n/a"
                )
                peak_str = (
                    f"{gpu_peak_reserved_gb:.1f}"
                    if gpu_peak_reserved_gb is not None
                    else "n/a"
                )
                msg = (
                    f"Step: {global_step} | dt: {step_time:5.2f}s/it | "
                    f"loss={loss_str} | toks/s={tokens_per_sec:.1f} | "
                    f"loss2={loss2_str} | "
                    f"vram_alloc={vram_alloc_str} | peak={peak_str}"
                )
                if z_loss_value is not None:
                    msg += f" | aux_loss={z_loss_value:.6f}"
                if null_rate_value is not None:
                    msg += f" | null_rate={null_rate_value:.3f}"
                if expert_counts_list is not None:
                    ec_min = min(expert_counts_list)
                    ec_max = max(expert_counts_list)
                    ec_std = (
                        sum(
                            (c - sum(expert_counts_list) / len(expert_counts_list)) ** 2
                            for c in expert_counts_list
                        )
                        / len(expert_counts_list)
                    ) ** 0.5
                    msg += f" | exp_bal=({ec_min:.0f}/{ec_max:.0f}/std={ec_std:.1f})"
                if enable_system_metrics:
                    if gpu_util is not None:
                        msg += (
                            f", GPU Util: {gpu_util:.0f}%, "
                            f"GPU Mem: {gpu_mem_used:.1f}G/{gpu_mem_total:.1f}G"
                        )
                    if cpu_util is not None:
                        msg += (
                            f", CPU Util: {cpu_util:.0f}%, "
                            f"CPU Mem: {cpu_mem_used:.1f}G/{cpu_mem_total:.1f}G"
                        )
                print_rank_0(msg)
                _append_jsonl(
                    metrics_jsonl_path,
                    {
                        "phase": "train",
                        "epoch": epoch,
                        "step": i,
                        "global_step": global_step,
                        "loss": loss_scalar,
                        "loss_ntp": (
                            None if loss_ntp_value is None else float(loss_ntp_value)
                        ),
                        "loss2": (
                            None if loss_mtp_value is None else float(loss_mtp_value)
                        ),
                        "z_loss": z_loss_value,
                        "null_rate": null_rate_value,
                        "expert_counts": expert_counts_list,
                        "lr": None if learning_rate is None else float(learning_rate),
                        "dt_ms": float(step_dt_ms),
                        "tokens_per_sec": float(tokens_per_sec),
                        "tokens": int(tokens),
                        "gpu_util": None if gpu_util is None else float(gpu_util),
                        "gpu_mem_used_gb": (
                            None if gpu_mem_used is None else float(gpu_mem_used)
                        ),
                        "cpu_util": None if cpu_util is None else float(cpu_util),
                        "cpu_mem_used_gb": (
                            None if cpu_mem_used is None else float(cpu_mem_used)
                        ),
                        "gsa_leak_fraction": (
                            None if gsa_leak_frac is None else float(gsa_leak_frac)
                        ),
                        "gsa_leak_attempt_fraction": (
                            None
                            if gsa_leak_attempt_frac is None
                            else float(gsa_leak_attempt_frac)
                        ),
                        "gpu_alloc_gb": (
                            None if gpu_alloc_gb is None else float(gpu_alloc_gb)
                        ),
                        "gpu_reserved_gb": (
                            None if gpu_reserved_gb is None else float(gpu_reserved_gb)
                        ),
                        "gpu_peak_reserved_gb": (
                            None
                            if gpu_peak_reserved_gb is None
                            else float(gpu_peak_reserved_gb)
                        ),
                        "gpu_peak_alloc_gb": (
                            None
                            if gpu_peak_alloc_gb is None
                            else float(gpu_peak_alloc_gb)
                        ),
                        "gpu_alloc_delta_gb": (
                            None
                            if gpu_alloc_delta_gb is None
                            else float(gpu_alloc_delta_gb)
                        ),
                        "gpu_reserved_delta_gb": (
                            None
                            if gpu_reserved_delta_gb is None
                            else float(gpu_reserved_delta_gb)
                        ),
                        "opus_scored": _opus_scored_this_step,
                        "opus_metrics": _opus_step_metrics,
                    },
                )

                # Forward to TrainingOps observability stack
                if training_ops is not None:
                    try:
                        training_ops.log_step(
                            step=global_step,
                            metrics={
                                "loss": loss_scalar,
                                "lr": float(learning_rate) if learning_rate else 0,
                                "tokens_per_second": float(tokens_per_sec),
                                "dt_ms": float(step_dt_ms),
                                "gpu_alloc_gb": gpu_alloc_gb,
                                "gpu_peak_reserved_gb": gpu_peak_reserved_gb,
                            },
                            context={"phase": "train", "epoch": epoch, "step": i},
                        )
                    except Exception:
                        pass

                # Print full GPU table (all devices) when enabled and available
                if enable_system_metrics and is_main_process():
                    try:
                        # gpu_table is defined above when NVML succeeds; guard with getattr-style check
                        if _NVML_AVAILABLE and "gpu_table" in locals() and gpu_table:
                            print_rank_0("\nGPU Utilization / Memory (all devices):")
                            print_rank_0(gpu_table)
                    except Exception:
                        # Don't let logging issues break training
                        pass

        # Save checkpoint periodically
        if checkpoint_interval is not None and (i + 1) % checkpoint_interval == 0:
            with (
                profiler.phase("checkpoint_save")
                if profiler is not None
                else _null_ctx()
            ):
                checkpoint_tag = f"epoch{epoch}_step{i + 1}"
                print_rank_0(
                    f"\nSaving checkpoint at epoch {epoch}, step {i + 1}, global_step {global_step}..."
                )

                # Client state to save with checkpoint
                client_state = {
                    "epoch": epoch,
                    "step": i + 1,
                    "global_step": global_step,
                    "loss": loss_scalar,
                }

                if checkpoint_manager:
                    # Use S3CheckpointManager (will upload to S3 in background)
                    checkpoint_manager.save_checkpoint(
                        model_engine,
                        step=global_step,
                        tag=checkpoint_tag,
                        client_state=client_state,
                    )
                elif output_dir:
                    # Use basic checkpoint saving
                    save_checkpoint(model_engine, output_dir, tag=checkpoint_tag)

                # Log checkpoint to observability stack
                if training_ops is not None:
                    try:
                        _ckpt_path = output_dir or ""
                        training_ops.log_checkpoint(
                            step=global_step,
                            path=_ckpt_path,
                            loss=loss_scalar,
                            tag=checkpoint_tag,
                        )
                    except Exception:
                        pass

        # ── Spot-aware checkpoint: on-demand / periodic-timer / spot-termination ──
        _spot_ckpt_triggered = False
        if spot_orchestrator is not None and spot_orchestrator.should_checkpoint(
            global_step
        ):
            _spot_ckpt_triggered = True
            _spot_reason = spot_orchestrator.get_checkpoint_reason()
            print_rank_0(
                f"\nSpot checkpoint triggered: reason={_spot_reason}, global_step={global_step}"
            )

            # Capture detailed shard state for exact resume
            _shard_state = {"step_in_epoch": i + 1}
            if hasattr(train_loader, "dataset") and hasattr(
                train_loader.dataset, "get_shard_state"
            ):
                _shard_state.update(train_loader.dataset.get_shard_state())
            elif hasattr(train_loader, "dataset") and hasattr(
                train_loader.dataset, "_shard_pairs"
            ):
                _shard_state["num_shards"] = len(train_loader.dataset._shard_pairs)

            spot_orchestrator.save_full_checkpoint(
                model_engine=model_engine,
                global_step=global_step,
                epoch=epoch,
                step_in_epoch=i + 1,
                checkpoint_manager=checkpoint_manager,
                shard_state=_shard_state,
                extra_client_state={
                    "loss": loss_scalar,
                    "loss_ntp": loss_ntp_value,
                    "tokens_per_sec": tokens_per_sec,
                    "learning_rate": learning_rate,
                    "lr_scheduler_state": (
                        lr_scheduler.state_dict() if lr_scheduler is not None else None
                    ),
                },
                training_ops=training_ops,
            )

            # Spot termination: AWS kills us in ~2 min, exit after save
            if spot_orchestrator.is_spot_terminating:
                print_rank_0(
                    "Spot termination: checkpoint saved, exiting training loop."
                )
                break

            # On-demand (Ctrl+C / SIGUSR1): checkpoint saved, check if stop requested
            if _spot_reason.startswith("on_demand"):
                if spot_orchestrator.stop_requested:
                    print_rank_0(
                        "On-demand checkpoint saved — stopping training as requested."
                    )
                    break
                else:
                    print_rank_0("On-demand checkpoint saved — training continues.")

        # Best-effort release path for ZeRO-3 coordinator/offload caches.
        if zero3_release_every > 0 and (global_step % zero3_release_every == 0):
            called_methods, release_errors = _release_zero3_runtime_buffers(
                model_engine
            )
            if called_methods and (not _zero3_release_logged):
                print_rank_0(
                    "[T19 hardening] ZeRO release hooks active: "
                    + ", ".join(sorted(set(called_methods)))
                )
                _zero3_release_logged = True
            if release_errors and (not _zero3_release_error_logged):
                print_rank_0(
                    "[T19 hardening] ZeRO release probes unavailable/no-op on this build: "
                    + "; ".join(release_errors[:3])
                )
                _zero3_release_error_logged = True

        if (
            zero3_force_clear_containers
            and zero3_release_every > 0
            and (global_step % zero3_release_every == 0)
        ):
            cleared = _force_clear_zero3_containers(model_engine)
            if cleared and (not _zero3_force_clear_logged):
                print_rank_0(
                    "[T19 hardening] Aggressive ZeRO container clear active: "
                    + ", ".join(cleared[:8])
                )
                _zero3_force_clear_logged = True

        if clear_router_cache_every > 0 and (
            global_step % clear_router_cache_every == 0
        ):
            _ = _clear_router_topk_caches(getattr(model_engine, "module", None))

        # Drop references to large step tensors before GC/cuda allocator maintenance.
        batch = None
        input_ids = None
        attention_mask = None
        labels = None
        x_input = None
        y_ntp = None
        y_mtp = None
        h_ntp = None
        h_mtp = None
        aux_loss = None
        aux_term = None
        outputs = None
        loss = None
        loss_ntp = None
        loss_mtp = None
        leak_frac_t = None
        leak_attempt_t = None
        lm_weight = None
        _lm_param = None

        # Free circular-ref and allocator-tracked GPU memory at end of each step.
        if torch.cuda.is_available() and cleanup_sync:
            torch.cuda.synchronize()
        if cleanup_gc_collect:
            gc.collect()
        if torch.cuda.is_available() and cleanup_empty_cache:
            torch.cuda.empty_cache()
        if torch.cuda.is_available() and cleanup_ipc_collect:
            torch.cuda.ipc_collect()

        # Early stopping: wall-clock budget
        if (
            max_train_seconds is not None
            and (time.time() - _epoch_wall_start) >= max_train_seconds
        ):
            print_rank_0(
                f"[timer] Wall-clock budget of {max_train_seconds}s reached after {steps} steps."
            )
            break

    _epoch_wall_elapsed = time.time() - _epoch_wall_start
    avg_loss = total_loss / steps if steps > 0 else 0
    avg_tokens_per_sec = (
        _total_tokens_processed / _epoch_wall_elapsed if _epoch_wall_elapsed > 0 else 0
    )

    print_rank_0(f"Epoch {epoch} - Training Average Loss: {avg_loss:.4f}")
    print_rank_0(
        f"Epoch {epoch} - Wall clock: {_epoch_wall_elapsed:.1f}s, Steps: {steps}, Avg tok/s: {avg_tokens_per_sec:.0f}"
    )

    # ── Profiler: write reports and clean up ─────────────────────────────────
    if profiler is not None and profiler._history:
        _pout = profile_output_dir or (
            os.path.dirname(metrics_jsonl_path) if metrics_jsonl_path else "results/run"
        )
        profiler.write_report(os.path.join(_pout, "profile_report.txt"))
        profiler.write_jsonl(os.path.join(_pout, "profile.jsonl"))
        print_rank_0(f"[profiler] Report written to {_pout}/profile_report.txt")
    if _owns_profiler and profiler is not None:
        profiler.deactivate()

    return (
        avg_loss,
        global_step,
        {
            "training_seconds": _epoch_wall_elapsed,
            "num_steps": steps,
            "tokens_per_sec": avg_tokens_per_sec,
            "total_tokens": _total_tokens_processed,
            "learning_rate": _get_learning_rate(model_engine),
            "peak_vram_mb": (
                torch.cuda.max_memory_reserved(model_engine.device) / 1e6
                if torch.cuda.is_available()
                else 0
            ),
        },
    )


def evaluate(
    model_engine,
    eval_loader,
    phase="Validation",
    max_steps=None,
    metrics_jsonl_path=None,
):
    """Evaluate model on given loader. Returns (avg_loss, perplexity)."""
    model_engine.eval()
    total_loss = 0
    steps = 0
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if max_steps is not None and i >= max_steps:
                break
            input_ids = batch["input_ids"].to(model_engine.device)
            attention_mask = batch["attention_mask"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)

            uses_custom = _uses_custom_recurrence_forward(model_engine.module)
            if uses_custom:
                x = input_ids[:, :-2].contiguous()
                y = input_ids[:, 1:-1].contiguous()
                h_ntp, _, _ = model_engine(
                    x,
                    next_token_ids=y,
                    attention_mask=(
                        attention_mask[:, :-2].contiguous()
                        if attention_mask is not None
                        else None
                    ),
                    return_loss=True,
                    return_memory=False,
                    prev_memory_stream=None,
                    return_hidden=True,
                )
                from .kernels.triton_cross_entropy import FusedLinearCrossEntropyLoss

                ce_fn = FusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean")
                lm_w = model_engine.module.lm_head.weight
                B, T, H = h_ntp.shape
                loss = ce_fn(h_ntp.view(-1, H), lm_w, y.view(-1))
            else:
                outputs = model_engine(
                    input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
            total_loss += loss.item()
            steps += 1
    avg_loss = total_loss / steps if steps > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    print_rank_0(f"{phase}: avg_loss={avg_loss:.4f}, perplexity={perplexity:.2f}")
    model_engine.train()
    return avg_loss, perplexity


def generate_text(model_engine, tokenizer, prompt, max_new_tokens=100):
    """Simple greedy text generation."""
    model_engine.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model_engine.device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model_engine(input_ids)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    model_engine.train()
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def save_checkpoint(model_engine, output_dir, tag="latest"):
    """Save DeepSpeed checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    model_engine.save_checkpoint(output_dir, tag=tag)
    print_rank_0(f"Checkpoint saved: {output_dir}/{tag}")


def load_checkpoint(model_engine, output_dir, tag="latest"):
    """Load DeepSpeed checkpoint. Returns client_state dict or None."""
    try:
        _, client_state = model_engine.load_checkpoint(output_dir, tag=tag)
        print_rank_0(f"Checkpoint loaded: {output_dir}/{tag}")
        return client_state
    except Exception as e:
        print_rank_0(f"Failed to load checkpoint: {e}")
        return None
