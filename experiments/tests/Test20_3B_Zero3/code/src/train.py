"""
Training utilities for DeepSpeed.

This module contains training, evaluation, and inference functions
for training language models with DeepSpeed optimization.
"""

import gc
import inspect
import json
import os
import time
from datetime import datetime

import psutil
import torch
import torch.distributed as dist
from tqdm import tqdm

# FIX-PERF-04 (v3): FusedLinearCrossEntropyLoss — fuses lm_head matmul + CE.
# Never materialises [B*T, vocab] logits (saves ~17 GB per step).
# ZERO FALLBACK — if this import fails, training crashes immediately.
from .kernels.triton_cross_entropy import FusedLinearCrossEntropyLoss as _FusedLinearCE
# _fused_ce is initialized inside train_epoch using max_chunk_gb from config

from contextlib import contextmanager

from .utils import is_main_process, print_rank_0
from .profiler import StepProfiler


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
            ("empty_partition_cache", "release_and_reset_all", "_release_and_reset_all"),
        ),
        (
            "partitioned_param_coordinator",
            coordinator,
            ("empty_partition_cache", "release_and_reset_all", "_release_and_reset_all", "reset_step"),
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
        if hasattr(submodule, "_cached_topk_idx") and getattr(submodule, "_cached_topk_idx") is not None:
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


def train_epoch(
    model_engine,
    train_loader,
    epoch,
    max_steps=None,
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

    # FIX-PERF-07: Use dynamic chunk size from config (default 4GB)
    fused_ce_fn = _FusedLinearCE(ignore_index=-100, reduction='mean', max_chunk_gb=max_chunk_gb)

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
    import triton as _triton
    import fla as _fla
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
        _pout = profile_output_dir or (os.path.dirname(metrics_jsonl_path) if metrics_jsonl_path else "results/run")
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
    progress_bar = tqdm(
        train_loader, desc=f"Epoch {epoch}", disable=not is_main_process()
    )

    profile_step = 10
    print_profile= True
    prof = FlopsProfiler(model_engine)
    for i, batch in enumerate(progress_bar):
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

        # Skip steps if resuming
        if i < start_step:
            continue
        if i == profile_step:
            print ("Profile started")
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
        _profiler_ctx = profiler.phase("dataloader") if profiler is not None else _null_ctx()
        with _profiler_ctx:
            input_ids = batch["input_ids"].to(model_engine.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(
                model_engine.device, non_blocking=True
            )
            labels = batch["labels"].to(model_engine.device, non_blocking=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
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
                    attention_mask=attention_mask[:, :-2].contiguous() if attention_mask is not None else None,
                    return_loss=True,
                    return_memory=False,
                    prev_memory_stream=None,
                    return_hidden=True,   # Skip lm_head — we compute CE below
                )
            with profiler.phase("gsa_leak_allreduce") if profiler is not None else _null_ctx():
                leak_frac_t = getattr(model_engine.module, "last_gsa_leak_fraction", None)
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
                print_rank_0(f"[MEMORY] After forward pass: {mem_after_fwd:.2f}GB (peak: {mem_peak:.2f}GB)")
                print_rank_0(f"[MEMORY] Forward allocated: {(mem_after_fwd - mem_before):.2f}GB")

            # 3. FusedLinearCE: fuses lm_head matmul + CE in one chunked kernel.
            # Never materialises [B*T, vocab] logits. Zero fallback.
            #
            # ZeRO-3: gather lm_head around CE compute, but avoid full weight cloning.
            # Cloning [V,H] each step adds a large transient allocation and can amplify
            # fragmentation before optimizer step.
            _lm_param = model_engine.module.lm_head.weight
            _is_zero3 = hasattr(_lm_param, 'ds_id')
            if _is_zero3:
                try:
                    from deepspeed.zero import GatheredParameters
                except ImportError:
                    from deepspeed.runtime.zero.partition_parameters import GatheredParameters
                _gather_ctx = GatheredParameters([_lm_param], modifier_rank=None)
            else:
                _gather_ctx = _null_ctx()

            with _gather_ctx:
                lm_weight = _lm_param.data

                B_seq, T_seq, H_dim = h_ntp.shape
                vocab_size = lm_weight.shape[0]

                with profiler.phase("fused_ce") if profiler is not None else _null_ctx():
                    loss_ntp = fused_ce_fn(
                        h_ntp.view(-1, H_dim),          # [B*T, H]
                        lm_weight,                      # [V, H]
                        y_ntp.view(-1),                 # [B*T]
                    )
                if i == 0:
                    mem_after_loss_ntp = torch.cuda.memory_allocated(model_engine.device) / 1e9
                    print_rank_0(f"[MEMORY] After loss_ntp: {mem_after_loss_ntp:.2f}GB")

                loss_mtp = None
                if h_mtp is not None:
                    B_m, T_m, H_m = h_mtp.shape
                    with profiler.phase("fused_ce_mtp") if profiler is not None else _null_ctx():
                        loss_mtp = fused_ce_fn(
                            h_mtp.view(-1, H_m),         # [B*T, H]
                            lm_weight,                   # [V, H]
                            y_mtp.view(-1),              # [B*T]
                        )
            
            # 4. NaN Watchdog — HARD CRASH (FIX: was silently continuing, corrupting weights)
            if torch.isnan(loss_ntp) or (loss_mtp is not None and torch.isnan(loss_mtp)) or \
                    (aux_loss is not None and torch.isnan(aux_loss)):
                raise RuntimeError(
                    f"NaN detected at epoch {epoch}, step {i}: "
                    f"loss_ntp={loss_ntp.item():.4f}, "
                    f"loss_mtp={loss_mtp.item():.4f if loss_mtp is not None else 'None'}"
                )

            # 5. Combine Loss (NTP + 0.3*MTP + aux)
            loss = loss_ntp
            if loss_mtp is not None:
                loss = loss + 0.3 * loss_mtp
            if aux_loss is not None and aux_loss.numel() > 0:
                # Defensive scalarization: some model variants may return
                # aux tensors with more than one element.
                aux_term = aux_loss if aux_loss.numel() == 1 else aux_loss.mean()
                loss += aux_term
                loss_aux_value = float(aux_term.detach().float().item())
            else:
                loss_aux_value = 0.0
            loss_ntp_value = float(loss_ntp.detach().float().item())
            loss_mtp_value = float(loss_mtp.detach().float().item()) if loss_mtp is not None else 0.0
            
            # Memory profiling before backward
            if i == 0:
                mem_before_bwd = torch.cuda.memory_allocated(model_engine.device) / 1e9
                print_rank_0(f"[MEMORY] Before backward: {mem_before_bwd:.2f}GB")
                print_rank_0(f"[MEMORY] === MEMORY SUMMARY ===")
                print_rank_0(torch.cuda.memory_summary(device=model_engine.device, abbreviated=True))
            
            # MTP loss (if enabled and logits_mtp is not None)
            # For now, we focus on NTP loss
            
        else:
            # Standard transformer model
            with profiler.phase("forward") if profiler is not None else _null_ctx():
                outputs = model_engine(input_ids, attention_mask=attention_mask, labels=labels)
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
                profiler._current and profiler._current.add("step_total", step_time * 1000.0)
            profiler.end_step(tokens=int(_ptoks.item()))
        step_dt_ms = step_time * 1000.0
        with profiler.phase("token_count_allreduce") if profiler is not None else _null_ctx():
            with torch.no_grad():
                # Count tokens in this batch using attention mask (1s for real tokens)
                tokens = attention_mask.sum().float()
                # Aggregate across all ranks if distributed is initialized
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
                tokens = tokens.item()
        tokens_per_sec = tokens / step_time if step_time > 0 else 0.0
        learning_rate = _get_learning_rate(model_engine)
        loss_scalar = float(loss.item())
        gpu_alloc_gb = None
        gpu_reserved_gb = None
        gpu_alloc_delta_gb = None
        gpu_reserved_delta_gb = None
        if track_cuda_memory and torch.cuda.is_available():
            gpu_alloc_gb = torch.cuda.memory_allocated(model_engine.device) / 1e9
            gpu_reserved_gb = torch.cuda.memory_reserved(model_engine.device) / 1e9
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
            with profiler.phase("system_metrics") if profiler is not None else _null_ctx():
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
        # Track metrics
        total_loss += loss_scalar
        steps += 1
        global_step += 1

        # Update progress bar
        postfix = {
            "loss": f"{loss_ntp_value:.4f}" if loss_ntp_value is not None else f"{loss_scalar:.4f}",
            "global_step": global_step,
            "toks/s": f"{tokens_per_sec:.1f}",
        }
        if loss_mtp_value is not None:
            postfix["loss2"] = f"{loss_mtp_value:.4f}"
        if loss_aux_value is not None:
            postfix["r_loss"] = f"{loss_aux_value:.4f}"
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
        if gpu_alloc_delta_gb is not None:
            postfix["d_alloc"] = f"{gpu_alloc_delta_gb:+.2f}G"
        progress_bar.set_postfix(postfix)

        # Log periodically
        if i % log_interval == 0:
            with profiler.phase("log_write") if profiler is not None else _null_ctx():
                loss_str = f"{loss_ntp_value:.4f}" if loss_ntp_value is not None else "nan"
                loss2_str = f"{loss_mtp_value:.4f}" if loss_mtp_value is not None else "nan"
                r_loss_str = f"{loss_aux_value:.4f}" if loss_aux_value is not None else "nan"
                lr_str = f"{learning_rate:.2e}" if learning_rate is not None else "nan"
                msg = (
                    f"{_format_log_timestamp()} | step {global_step} | "
                    f"loss: {loss_str} | loss2: {loss2_str} | r_loss: {r_loss_str} | "
                    f"lr: {lr_str} | dt: {step_dt_ms:.2f}ms | tok/sec: {tokens_per_sec:9.2f}"
                )
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
                        "loss_ntp": None if loss_ntp_value is None else float(loss_ntp_value),
                        "loss2": None if loss_mtp_value is None else float(loss_mtp_value),
                        "r_loss": None if loss_aux_value is None else float(loss_aux_value),
                        "lr": None if learning_rate is None else float(learning_rate),
                        "dt_ms": float(step_dt_ms),
                        "tokens_per_sec": float(tokens_per_sec),
                        "tokens": int(tokens),
                        "gpu_util": None if gpu_util is None else float(gpu_util),
                        "gpu_mem_used_gb": None if gpu_mem_used is None else float(gpu_mem_used),
                        "cpu_util": None if cpu_util is None else float(cpu_util),
                        "cpu_mem_used_gb": None if cpu_mem_used is None else float(cpu_mem_used),
                        "gsa_leak_fraction": None if gsa_leak_frac is None else float(gsa_leak_frac),
                        "gsa_leak_attempt_fraction": (
                            None
                            if gsa_leak_attempt_frac is None
                            else float(gsa_leak_attempt_frac)
                        ),
                        "gpu_alloc_gb": None if gpu_alloc_gb is None else float(gpu_alloc_gb),
                        "gpu_reserved_gb": (
                            None if gpu_reserved_gb is None else float(gpu_reserved_gb)
                        ),
                        "gpu_alloc_delta_gb": (
                            None if gpu_alloc_delta_gb is None else float(gpu_alloc_delta_gb)
                        ),
                        "gpu_reserved_delta_gb": (
                            None
                            if gpu_reserved_delta_gb is None
                            else float(gpu_reserved_delta_gb)
                        ),
                    },
                )

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
            with profiler.phase("checkpoint_save") if profiler is not None else _null_ctx():
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

        # Best-effort release path for ZeRO-3 coordinator/offload caches.
        if zero3_release_every > 0 and (global_step % zero3_release_every == 0):
            called_methods, release_errors = _release_zero3_runtime_buffers(model_engine)
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

        if zero3_force_clear_containers and zero3_release_every > 0 and (global_step % zero3_release_every == 0):
            cleared = _force_clear_zero3_containers(model_engine)
            if cleared and (not _zero3_force_clear_logged):
                print_rank_0(
                    "[T19 hardening] Aggressive ZeRO container clear active: "
                    + ", ".join(cleared[:8])
                )
                _zero3_force_clear_logged = True

        if clear_router_cache_every > 0 and (global_step % clear_router_cache_every == 0):
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

        # Early stopping for demo/debugging
        if max_steps is not None and i >= max_steps:
            break

    avg_loss = total_loss / steps if steps > 0 else 0
    print_rank_0(f"Epoch {epoch} - Training Average Loss: {avg_loss:.4f}")

    # ── Profiler: write reports and clean up ─────────────────────────────────
    if profiler is not None and profiler._history:
        _pout = profile_output_dir or (os.path.dirname(metrics_jsonl_path) if metrics_jsonl_path else "results/run")
        profiler.write_report(os.path.join(_pout, "profile_report.txt"))
        profiler.write_jsonl(os.path.join(_pout, "profile.jsonl"))
        print_rank_0(f"[profiler] Report written to {_pout}/profile_report.txt")
    if _owns_profiler and profiler is not None:
        profiler.deactivate()

    return avg_loss, global_step


# ═══════════════════════════════════════════════════════════════════════════════
# OPUS: Optimizer-induced Projected Utility Selection
# ═══════════════════════════════════════════════════════════════════════════════


def _opus_forward_and_loss(model_engine, input_ids, fused_ce_fn):
    """
    Scoring-pass forward: NTP loss only (no MTP), return_hidden=True.
    Same loss formulation as training pass for OPUS consistency.
    """
    x_input = input_ids[:, :-2].contiguous()
    y_ntp = input_ids[:, 1:-1].contiguous()

    h_ntp, _h_mtp, aux_loss = model_engine(
        x_input,
        next_token_ids=None,       # NO MTP in scoring pass
        attention_mask=None,
        return_loss=True,
        return_memory=False,
        prev_memory_stream=None,
        return_hidden=True,
    )

    _lm_param = model_engine.module.lm_head.weight
    _is_zero3 = hasattr(_lm_param, 'ds_id')
    if _is_zero3:
        try:
            from deepspeed.zero import GatheredParameters
        except ImportError:
            from deepspeed.runtime.zero.partition_parameters import GatheredParameters
        _gather_ctx = GatheredParameters([_lm_param], modifier_rank=None)
    else:
        _gather_ctx = _null_ctx()

    with _gather_ctx:
        lm_weight = _lm_param.data
        B, T, H = h_ntp.shape
        loss = fused_ce_fn(
            h_ntp.view(-1, H),
            lm_weight,
            y_ntp.view(-1),
        )

    if aux_loss is not None and aux_loss.numel() > 0:
        aux_term = aux_loss if aux_loss.numel() == 1 else aux_loss.mean()
        loss = loss + aux_term

    return loss


def train_epoch_opus(
    model_engine,
    candidate_loader,
    proxy_provider,
    preconditioner_view,
    sketcher,
    selector,
    opus_n_proxy,
    opus_proxy_seq_len,
    opus_candidate_multiplier,
    opus_train_batch,
    opus_score_interval=1,
    epoch=0,
    max_steps=None,
    log_interval=10,
    global_step=0,
    metrics_jsonl_path=None,
    max_chunk_gb=8.0,
    profiler=None,
    profile_steps=None,
    profile_output_dir=None,
):
    """
    OPUS training epoch: scoring pass (512 tokens) + training pass (full length).

    Data flow per step:
      1. Load candidate_pool_size sequences at full length (4096)
      2. Sample n_proxy proxy sequences at proxy_seq_len (512)
      3. Scoring: truncate candidates to 512, combine with proxy → forward+backward
         with ghost hooks → alignment scores
      4. Boltzmann selection → pick best candidates
      5. Take first opus_train_batch for training (memory limit)
      6. Training: selected at full length → forward+backward → step
    """
    from .opus import (
        AdamWPreconditionerView,
        CountSketchProjector,
        OpusGhostCollector,
        OpusSelector,
        SelectionResult,
    )
    from .opus.distributed import get_rank, get_world_size

    model_engine.train()
    total_loss = 0
    steps = 0
    device = model_engine.device

    fused_ce_fn = _FusedLinearCE(ignore_index=-100, reduction='mean', max_chunk_gb=max_chunk_gb)

    # Runtime hardening flags (same as train_epoch)
    cleanup_sync = _env_flag("T19_STEP_CUDA_SYNC", "1")
    cleanup_gc_collect = _env_flag("T19_STEP_GC_COLLECT", "1")
    cleanup_empty_cache = _env_flag("T19_STEP_EMPTY_CACHE", "1")
    cleanup_ipc_collect = _env_flag("T19_STEP_IPC_COLLECT", "0")
    zero3_release_every = int(os.environ.get("T19_ZERO3_RELEASE_EVERY", "0"))
    zero3_force_clear_containers = _env_flag("T19_ZERO3_FORCE_CLEAR_CONTAINERS", "0")
    clear_router_cache_every = int(os.environ.get("T19_CLEAR_ROUTER_CACHE_EVERY", "0"))
    track_cuda_memory = _env_flag("T19_TRACK_CUDA_MEMORY", "0")
    prev_alloc_gb = None
    prev_reserved_gb = None
    _zero3_release_logged = False
    _zero3_release_error_logged = False
    _zero3_force_clear_logged = False

    # Step profiler
    _owns_profiler = False
    if profiler is None and profile_steps:
        local_rank = getattr(model_engine, "local_rank", 0)
        _pout = profile_output_dir or (os.path.dirname(metrics_jsonl_path) if metrics_jsonl_path else "results/run")
        profiler = StepProfiler(
            rank=local_rank,
            profile_steps=set(profile_steps),
            output_dir=_pout,
        )
        profiler.activate()
        profiler.register_model(model_engine.module)
        _owns_profiler = True

    progress_bar = tqdm(
        candidate_loader, desc=f"Epoch {epoch} [OPUS]", disable=not is_main_process()
    )

    world_size = get_world_size()
    rank = get_rank()

    score_interval = max(1, int(opus_score_interval))
    print_rank_0(
        f"[OPUS] candidate_multiplier={opus_candidate_multiplier} "
        f"n_proxy={opus_n_proxy}/GPU ({opus_n_proxy * world_size} global) "
        f"proxy_seq_len={opus_proxy_seq_len} "
        f"opus_train_batch={opus_train_batch}/GPU ({opus_train_batch * world_size} global) "
        f"score_interval={score_interval}"
    )

    for i, batch in enumerate(progress_bar):
        if profiler is not None:
            _batch_tokens = batch.get("attention_mask", batch["input_ids"]).sum().item() if isinstance(batch, dict) else 0
            profiler.start_step(global_step + 1, tokens=int(_batch_tokens))

        step_start_time = time.time()
        _phase_times = {}

        def _sync_time():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return time.time()

        # ── Load candidate pool (full length, e.g. 4096) ─────────────────────
        candidate_ids = batch["input_ids"].to(device, non_blocking=True)
        n_candidates = candidate_ids.size(0)

        # ══════════════════════════════════════════════════════════════════════
        # PASS 1: SCORING or RANDOM SELECTION
        # ══════════════════════════════════════════════════════════════════════
        do_scoring = (score_interval == 1) or ((global_step + 1) % score_interval == 1) or (global_step == 0)
        result = None

        if do_scoring:
            # Full OPUS scoring pass
            _t0 = _sync_time()
            preconditioner_view.refresh()
            _phase_times["precond_refresh"] = (_sync_time() - _t0) * 1000

            _t0 = _sync_time()
            proxy_ids = proxy_provider.sample(
                device=device,
                k=opus_n_proxy,
                seq_len=opus_proxy_seq_len,
            )
            n_proxy = proxy_ids.size(0)
            _phase_times["proxy_sample"] = (_sync_time() - _t0) * 1000

            _t0 = _sync_time()
            scoring_candidate_ids = candidate_ids[:, :opus_proxy_seq_len]
            combined_ids = torch.cat([proxy_ids, scoring_candidate_ids], dim=0)
            _phase_times["build_scoring"] = (_sync_time() - _t0) * 1000

            ghost = OpusGhostCollector(
                model=model_engine.module,
                n_proxy=n_proxy,
                n_candidates=n_candidates,
                preconditioner=preconditioner_view,
                sketcher=sketcher,
                device=device,
            )

            with ghost:
                _t0 = _sync_time()
                scoring_loss = _opus_forward_and_loss(model_engine, combined_ids, fused_ce_fn)
                _phase_times["scoring_fwd"] = (_sync_time() - _t0) * 1000

                _t0 = _sync_time()
                model_engine.backward(scoring_loss)
                _phase_times["scoring_bwd"] = (_sync_time() - _t0) * 1000

                _t0 = _sync_time()
                model_engine.zero_grad()
                _phase_times["zero_grad"] = (_sync_time() - _t0) * 1000

                alignment_scores, candidate_sketches = ghost.results()

            current_lr = _get_learning_rate(model_engine) or 3e-4
            _t0 = _sync_time()
            result = selector.select(
                alignment_scores=alignment_scores,
                candidate_sketches=candidate_sketches,
                learning_rate=current_lr,
            )
            _phase_times["boltzmann"] = (_sync_time() - _t0) * 1000

            local_indices = result.selected_local_indices
            if opus_train_batch is not None and local_indices.numel() > opus_train_batch:
                local_indices = local_indices[:opus_train_batch]

            del scoring_loss, alignment_scores, candidate_sketches, proxy_ids
            del scoring_candidate_ids, combined_ids
        else:
            # Random selection — skip scoring pass entirely
            _tb = opus_train_batch if opus_train_batch is not None else n_candidates
            local_indices = torch.randperm(n_candidates, device=device)[:_tb]

        _phase_times["scoring_total"] = sum(
            _phase_times.get(k, 0) for k in
            ["precond_refresh", "proxy_sample", "build_scoring", "scoring_fwd", "scoring_bwd", "zero_grad", "boltzmann"]
        )

        # ══════════════════════════════════════════════════════════════════════
        # PASS 2: TRAINING (forward+backward at full length, e.g. 4096)
        # ══════════════════════════════════════════════════════════════════════

        selected_ids = candidate_ids[local_indices]

        x_input = selected_ids[:, :-2].contiguous()
        y_ntp = selected_ids[:, 1:-1].contiguous()
        y_mtp = selected_ids[:, 2:].contiguous()

        _t0 = _sync_time()
        h_ntp, h_mtp, aux_loss = model_engine(
            x_input,
            next_token_ids=y_ntp,
            attention_mask=None,
            return_loss=True,
            return_memory=False,
            prev_memory_stream=None,
            return_hidden=True,
        )
        _phase_times["train_fwd"] = (_sync_time() - _t0) * 1000

        # FusedLinearCE with ZeRO-3 GatheredParameters
        _lm_param = model_engine.module.lm_head.weight
        _is_zero3 = hasattr(_lm_param, 'ds_id')
        if _is_zero3:
            try:
                from deepspeed.zero import GatheredParameters
            except ImportError:
                from deepspeed.runtime.zero.partition_parameters import GatheredParameters
            _gather_ctx = GatheredParameters([_lm_param], modifier_rank=None)
        else:
            _gather_ctx = _null_ctx()

        _t0 = _sync_time()
        with _gather_ctx:
            lm_weight = _lm_param.data
            B_seq, T_seq, H_dim = h_ntp.shape

            loss_ntp = fused_ce_fn(
                h_ntp.view(-1, H_dim),
                lm_weight,
                y_ntp.view(-1),
            )

            loss_mtp = None
            if h_mtp is not None:
                B_m, T_m, H_m = h_mtp.shape
                loss_mtp = fused_ce_fn(
                    h_mtp.view(-1, H_m),
                    lm_weight,
                    y_mtp.view(-1),
                )
        _phase_times["train_ce"] = (_sync_time() - _t0) * 1000

        # Combine losses
        loss = loss_ntp
        if loss_mtp is not None:
            loss = loss + 0.3 * loss_mtp
        loss_aux_value = 0.0
        if aux_loss is not None and aux_loss.numel() > 0:
            aux_term = aux_loss if aux_loss.numel() == 1 else aux_loss.mean()
            loss += aux_term
            loss_aux_value = float(aux_term.detach().float().item())

        if torch.isnan(loss):
            raise RuntimeError(f"NaN loss at step {i}, global_step {global_step}")

        loss_ntp_value = float(loss_ntp.detach().float().item())
        loss_mtp_value = float(loss_mtp.detach().float().item()) if loss_mtp is not None else 0.0

        # Backward + optimizer step
        _t0 = _sync_time()
        model_engine.backward(loss)
        _phase_times["train_bwd"] = (_sync_time() - _t0) * 1000

        _t0 = _sync_time()
        model_engine.step()
        _phase_times["train_step"] = (_sync_time() - _t0) * 1000

        _phase_times["train_total"] = sum(
            _phase_times.get(k, 0) for k in
            ["train_fwd", "train_ce", "train_bwd", "train_step"]
        )

        # ── Metrics & Logging ─────────────────────────────────────────────────
        step_time = time.time() - step_start_time
        loss_scalar = float(loss.item())
        total_loss += loss_scalar
        steps += 1
        global_step += 1

        # Token count for throughput (training tokens only)
        with torch.no_grad():
            tokens = torch.tensor(selected_ids.numel(), dtype=torch.float, device=device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
            tokens = tokens.item()
        tokens_per_sec = tokens / step_time if step_time > 0 else 0.0
        learning_rate = _get_learning_rate(model_engine)

        # Profiler end step
        if profiler is not None:
            profiler._current and profiler._current.add("step_total", step_time * 1000.0)
            profiler.end_step(tokens=int(tokens))

        # GPU memory tracking
        gpu_alloc_gb = gpu_alloc_delta_gb = None
        if track_cuda_memory and torch.cuda.is_available():
            gpu_alloc_gb = torch.cuda.memory_allocated(device) / 1e9
            if prev_alloc_gb is not None:
                gpu_alloc_delta_gb = gpu_alloc_gb - prev_alloc_gb
            prev_alloc_gb = gpu_alloc_gb

        # OPUS metrics
        if do_scoring and result is not None:
            m = result.metrics
            opus_alignment = m.get("alignment", 0.0)
            opus_redundancy = m.get("redundancy", 0.0)
            opus_entropy = m.get("entropy", 0.0)
            opus_selector_ms = m.get("selector_time_s", 0.0) * 1000
            opus_fallback = result.used_fallback
        else:
            opus_alignment = 0.0
            opus_redundancy = 0.0
            opus_entropy = 0.0
            opus_selector_ms = 0.0
            opus_fallback = False

        # Progress bar
        postfix = {
            "loss": f"{loss_ntp_value:.4f}",
            "toks/s": f"{tokens_per_sec:.0f}",
            "step": global_step,
        }
        if loss_mtp_value:
            postfix["mtp"] = f"{loss_mtp_value:.4f}"
        if gpu_alloc_gb is not None:
            postfix["vram"] = f"{gpu_alloc_gb:.1f}G"
        if gpu_alloc_delta_gb is not None:
            postfix["d"] = f"{gpu_alloc_delta_gb:+.2f}G"
        postfix["align"] = f"{opus_alignment:.2f}"
        postfix["sel_ms"] = f"{opus_selector_ms:.0f}"
        if opus_fallback:
            postfix["FB"] = "1"
        progress_bar.set_postfix(postfix)

        # JSONL log
        if i % log_interval == 0:
            lr_str = f"{learning_rate:.2e}" if learning_rate else "nan"
            print_rank_0(
                f"{_format_log_timestamp()} | step {global_step} | "
                f"loss: {loss_ntp_value:.4f} | loss2: {loss_mtp_value:.4f} | "
                f"r_loss: {loss_aux_value:.4f} | lr: {lr_str} | "
                f"dt: {step_time*1000:.0f}ms | tok/sec: {tokens_per_sec:.0f} | "
                f"opus_align: {opus_alignment:.3f} | "
                f"opus_sel: {opus_selector_ms:.0f}ms"
                + (" | [FALLBACK]" if opus_fallback else "")
            )
            # Phase timing breakdown
            print_rank_0(
                f"  PHASES: scoring_fwd={_phase_times.get('scoring_fwd',0):.0f}ms "
                f"scoring_bwd={_phase_times.get('scoring_bwd',0):.0f}ms "
                f"zero_grad={_phase_times.get('zero_grad',0):.0f}ms "
                f"boltzmann={_phase_times.get('boltzmann',0):.0f}ms "
                f"| scoring_total={_phase_times.get('scoring_total',0):.0f}ms "
                f"| train_fwd={_phase_times.get('train_fwd',0):.0f}ms "
                f"train_ce={_phase_times.get('train_ce',0):.0f}ms "
                f"train_bwd={_phase_times.get('train_bwd',0):.0f}ms "
                f"train_step={_phase_times.get('train_step',0):.0f}ms "
                f"| train_total={_phase_times.get('train_total',0):.0f}ms"
            )
            _append_jsonl(
                metrics_jsonl_path,
                {
                    "phase": "train_opus",
                    "epoch": epoch,
                    "step": i,
                    "global_step": global_step,
                    "loss": loss_scalar,
                    "loss_ntp": loss_ntp_value,
                    "loss2": loss_mtp_value,
                    "r_loss": loss_aux_value,
                    "lr": float(learning_rate) if learning_rate else None,
                    "dt_ms": step_time * 1000.0,
                    "tokens_per_sec": tokens_per_sec,
                    "tokens": int(tokens),
                    "gpu_alloc_gb": gpu_alloc_gb,
                    "gpu_alloc_delta_gb": gpu_alloc_delta_gb,
                    "opus_alignment": opus_alignment,
                    "opus_redundancy": opus_redundancy,
                    "opus_entropy": opus_entropy,
                    "opus_selector_ms": opus_selector_ms,
                    "opus_fallback": opus_fallback,
                    "opus_selected_local": int(local_indices.numel()),
                    "opus_candidates_local": n_candidates,
                    **{f"phase_{k}_ms": v for k, v in _phase_times.items()},
                },
            )

        # ── Post-step cleanup (same as train_epoch) ──────────────────────────
        if zero3_release_every > 0 and (global_step % zero3_release_every == 0):
            called_methods, release_errors = _release_zero3_runtime_buffers(model_engine)
            if called_methods and not _zero3_release_logged:
                print_rank_0("[T19] ZeRO release: " + ", ".join(sorted(set(called_methods))))
                _zero3_release_logged = True
        if zero3_force_clear_containers and zero3_release_every > 0 and (global_step % zero3_release_every == 0):
            _force_clear_zero3_containers(model_engine)
        if clear_router_cache_every > 0 and (global_step % clear_router_cache_every == 0):
            _clear_router_topk_caches(getattr(model_engine, "module", None))

        # Drop references
        batch = None; candidate_ids = None; selected_ids = None
        x_input = None; y_ntp = None; y_mtp = None
        h_ntp = None; h_mtp = None; aux_loss = None
        loss = None; loss_ntp = None; loss_mtp = None
        lm_weight = None; _lm_param = None; result = None; local_indices = None

        if torch.cuda.is_available() and cleanup_sync:
            torch.cuda.synchronize()
        if cleanup_gc_collect:
            gc.collect()
        if torch.cuda.is_available() and cleanup_empty_cache:
            torch.cuda.empty_cache()
        if torch.cuda.is_available() and cleanup_ipc_collect:
            torch.cuda.ipc_collect()

        if max_steps is not None and i >= max_steps:
            break

    avg_loss = total_loss / steps if steps > 0 else 0
    print_rank_0(f"Epoch {epoch} [OPUS] - Average Loss: {avg_loss:.4f}")

    if profiler is not None and profiler._history:
        _pout = profile_output_dir or (os.path.dirname(metrics_jsonl_path) if metrics_jsonl_path else "results/run")
        profiler.write_report(os.path.join(_pout, "profile_report.txt"))
        profiler.write_jsonl(os.path.join(_pout, "profile.jsonl"))
    if _owns_profiler and profiler is not None:
        profiler.deactivate()

    return avg_loss, global_step


def evaluate(
    model_engine,
    data_loader,
    phase="Evaluation",
    max_steps=None,
    metrics_jsonl_path=None,
):
    """
    Evaluate the model on a dataset.

    Args:
        model_engine: DeepSpeed model engine
        data_loader: DataLoader for evaluation data
        phase: Name of the evaluation phase (for logging)
        max_steps: Maximum number of steps (None for full evaluation)

    Returns:
        Tuple of (average_loss, average_perplexity)
    """
    model_engine.eval()
    total_loss = 0
    total_perplexity = 0
    steps = 0

    # Only show progress bar on main process
    progress_bar = tqdm(data_loader, desc=phase, disable=not is_main_process())

    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(model_engine.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(
                model_engine.device, non_blocking=True
            )
            labels = batch["labels"].to(model_engine.device, non_blocking=True)

            # Forward pass
            # Recurrence models use a custom forward signature (not labels=...)
            uses_custom_forward = _uses_custom_recurrence_forward(model_engine.module)

            if uses_custom_forward:
                # Reversible model: returns (logits_ntp, logits_mtp, aux_loss)
                x_input = input_ids[:, :-2].contiguous()
                y_ntp = input_ids[:, 1:-1].contiguous()
                y_mtp = input_ids[:, 2:].contiguous()
                
                # CRITICAL FIX: Bypass DeepSpeed's autocast wrapper
                with torch.amp.autocast('cuda',enabled=False):
                    logits_ntp, logits_mtp, aux_loss = model_engine.module(
                        x_input,
                        next_token_ids=y_ntp,
                        attention_mask=attention_mask[:, :-2].contiguous() if attention_mask is not None else None,
                        return_loss=True,
                        return_memory=False,
                        prev_memory_stream=None
                    )
                
                vocab_size = logits_ntp.size(-1)
                
                # Compute both NTP and MTP losses
                with torch.amp.autocast('cuda',enabled=False):
                    loss_ntp = torch.nn.functional.cross_entropy(
                        logits_ntp.float().view(-1, vocab_size), 
                        y_ntp.view(-1)
                    )
                del logits_ntp
                
                with torch.amp.autocast('cuda',enabled=False):
                    loss_mtp = torch.nn.functional.cross_entropy(
                        logits_mtp.float().view(-1, vocab_size), 
                        y_mtp.view(-1)
                    )
                del logits_mtp
                
                loss = loss_ntp + 0.3 * loss_mtp
                if aux_loss is not None and aux_loss.numel() > 0:
                    loss += aux_loss
            else:
                # Standard transformer model
                outputs = model_engine(
                    input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss

            # Track metrics
            total_loss += loss.item()
            total_perplexity += torch.exp(loss).item()
            steps += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Early stopping for demo/debugging
            if max_steps is not None and i >= max_steps:
                break

    avg_loss = total_loss / steps
    avg_perplexity = total_perplexity / steps

    print_rank_0(
        f"{phase} - Avg Loss: {avg_loss:.4f}, Avg Perplexity: {avg_perplexity:.4f}"
    )
    _append_jsonl(
        metrics_jsonl_path,
        {
            "phase": phase.lower(),
            "avg_loss": float(avg_loss),
            "avg_perplexity": float(avg_perplexity),
            "steps": int(steps),
        },
    )

    return avg_loss, avg_perplexity


def generate_text(
    model_engine,
    tokenizer,
    prompt="The history of artificial intelligence begins with",
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.92,
):
    """
    Generate text using the trained model.

    Args:
        model_engine: DeepSpeed model engine
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt for generation
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (lower = more conservative)
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter

    Returns:
        Dictionary with 'prompt', 'full_text', and 'generated_text'
    """
    model_engine.eval()

    print_rank_0(f'\nGenerating text from prompt: "{prompt}"')

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model_engine.device, non_blocking=True)
    attention_mask = inputs["attention_mask"].to(model_engine.device, non_blocking=True)

    print_rank_0(f"Input tokens: {input_ids.shape[1]}")

    # Generate (only on rank 0 to avoid redundant generation)
    if is_main_process():
        with torch.no_grad():
            # Recurrence models use a custom forward signature (not HF-style generate)
            uses_custom_forward = _uses_custom_recurrence_forward(model_engine.module)

            if uses_custom_forward:
                # Reversible models need custom generation logic
                # For now, we'll use a simple greedy generation approach
                print_rank_0("  Note: Using simplified generation for reversible model")
                
                generated_ids = input_ids.clone()
                for _ in range(max_new_tokens):
                    # Forward pass
                    logits_ntp, _, _ = model_engine.module(
                        generated_ids,
                        next_token_ids=None,
                        attention_mask=None,
                        return_loss=True
                    )
                    
                    # Get next token (greedy or sampling)
                    next_token_logits = logits_ntp[:, -1, :] / temperature
                    
                    if top_k > 0:
                        # Top-k sampling
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                        next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                    
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Append to generated sequence
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    
                    # Stop if EOS token is generated
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                
                output_ids = generated_ids
            else:
                # Standard transformer model
                output_ids = model_engine.module.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    no_repeat_ngram_size=2,
                    pad_token_id=tokenizer.eos_token_id,
                )

        # Decode
        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract generated portion
        generated_text = (
            full_output[len(input_text) :].strip()
            if len(full_output) > len(input_text)
            else ""
        )

        print_rank_0(
            f"\nGenerated {output_ids.shape[1] - input_ids.shape[1]} new tokens"
        )
        print_rank_0(f"\nFull Output:\n{full_output}")
        print_rank_0(f"\nGenerated Continuation:\n{generated_text}")
    else:
        # Return empty results for non-main processes
        input_text = ""
        full_output = ""
        generated_text = ""

    return {
        "prompt": input_text,
        "full_text": full_output,
        "generated_text": generated_text,
    }


def save_checkpoint(model_engine, output_dir, tag="final"):
    """
    Save model checkpoint.

    Args:
        model_engine: DeepSpeed model engine
        output_dir: Directory to save checkpoint
        tag: Tag for the checkpoint
    """
    print_rank_0(f"Saving checkpoint to {output_dir} with tag '{tag}'")
    model_engine.save_checkpoint(output_dir, tag=tag)
    print_rank_0("Checkpoint saved successfully")


def load_checkpoint(model_engine, checkpoint_dir, tag="final"):
    """
    Load model checkpoint.

    Args:
        model_engine: DeepSpeed model engine
        checkpoint_dir: Directory containing checkpoint
        tag: Tag of the checkpoint to load

    Returns:
        The loaded checkpoint metadata
    """
    print_rank_0(f"Loading checkpoint from {checkpoint_dir} with tag '{tag}'")
    _, client_sd = model_engine.load_checkpoint(checkpoint_dir, tag=tag)
    print_rank_0("Checkpoint loaded successfully")
    return client_sd
