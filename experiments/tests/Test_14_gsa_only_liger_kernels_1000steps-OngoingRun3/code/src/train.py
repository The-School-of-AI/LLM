"""
Training utilities for DeepSpeed.

This module contains training, evaluation, and inference functions
for training language models with DeepSpeed optimization.
"""

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
# _fused_ce is initialized inside train_epoch

from .utils import is_main_process, print_rank_0
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

    # FIX-PERF-07: Using streaming kernel (no high-level chunking needed)
    fused_ce_fn = _FusedLinearCE(ignore_index=-100, reduction='mean')

    # Only show progress bar on main process
    progress_bar = tqdm(
        train_loader, desc=f"Epoch {epoch}", disable=not is_main_process()
    )

    profile_step = 10
    print_profile= True
    prof = FlopsProfiler(model_engine)
    for i, batch in enumerate(progress_bar):
        # Skip steps if resuming
        if i < start_step:
            continue
        if i == profile_step:
            print ("Profile started")
            prof.start_profile()
        # Measure step wall-clock time
        step_start_time = time.time()
        # Move batch to device
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
            h_ntp, h_mtp, aux_loss = model_engine(
                x_input,
                next_token_ids=y_ntp,
                attention_mask=attention_mask[:, :-2].contiguous() if attention_mask is not None else None,
                return_loss=True,
                return_memory=False,
                prev_memory_stream=None,
                return_hidden=True,   # Skip lm_head — we compute CE below
            )
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
            lm_weight = model_engine.module.lm_head.weight  # [V, H]
            B_seq, T_seq, H_dim = h_ntp.shape
            vocab_size = lm_weight.shape[0]

            loss_ntp = fused_ce_fn(
                h_ntp.view(-1, H_dim),          # [B*T, H]
                lm_weight,                       # [V, H]
                y_ntp.view(-1),                  # [B*T]
            )
            if i == 0:
                mem_after_loss_ntp = torch.cuda.memory_allocated(model_engine.device) / 1e9
                print_rank_0(f"[MEMORY] After loss_ntp: {mem_after_loss_ntp:.2f}GB")

            loss_mtp = None
            if h_mtp is not None:
                B_m, T_m, H_m = h_mtp.shape
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
            outputs = model_engine(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss_ntp_value = float(loss.detach().float().item())
            loss_mtp_value = None
            loss_aux_value = None

        # Backward pass
        model_engine.backward(loss)

        # Update weights
        model_engine.step()
        
        if i == 0:
            mem_after_step = torch.cuda.memory_allocated(model_engine.device) / 1e9
            print_rank_0(f"[MEMORY] After backward & step: {mem_after_step:.2f}GB")

        # Compute tokens per second for this step
        step_time = time.time() - step_start_time
        step_dt_ms = step_time * 1000.0
        with torch.no_grad():
            # Count tokens in this batch using attention mask (1s for real tokens)
            tokens = attention_mask.sum().float()
            # Aggregate across all ranks if distributed is initialized
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
            tokens = tokens.item()
        tokens_per_sec = tokens / step_time if step_time > 0 else 0.0
        learning_rate = _get_learning_rate(model_engine)

        # Optional system metrics (CPU/GPU util & memory)
        gpu_util = gpu_mem_used = gpu_mem_total = None
        cpu_util = cpu_mem_used = cpu_mem_total = None
        if enable_system_metrics:
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
        total_loss += loss.item()
        steps += 1
        global_step += 1

        # Update progress bar
        postfix = {
            "loss": f"{loss_ntp_value:.4f}" if loss_ntp_value is not None else f"{loss.item():.4f}",
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
        progress_bar.set_postfix(postfix)

        # Log periodically
        if i % log_interval == 0:
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
                    "loss": float(loss.item()),
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
            checkpoint_tag = f"epoch{epoch}_step{i + 1}"
            print_rank_0(
                f"\nSaving checkpoint at epoch {epoch}, step {i + 1}, global_step {global_step}..."
            )

            # Client state to save with checkpoint
            client_state = {
                "epoch": epoch,
                "step": i + 1,
                "global_step": global_step,
                "loss": loss.item(),
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

        # Early stopping for demo/debugging
        if max_steps is not None and i >= max_steps:
            break

    avg_loss = total_loss / steps if steps > 0 else 0
    print_rank_0(f"Epoch {epoch} - Training Average Loss: {avg_loss:.4f}")

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
