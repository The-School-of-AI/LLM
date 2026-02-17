"""
Training utilities for DeepSpeed.

This module contains training, evaluation, and inference functions
for training language models with DeepSpeed optimization.
"""

import json as _json
import os
import torch
from tqdm import tqdm
import time
import torch.distributed as dist
import psutil

from .utils import is_main_process, print_rank_0
from deepspeed.profiling.flops_profiler import FlopsProfiler


def _jsonl_logger(output_dir: str):
    """Return an append function that writes one JSON line per call (rank 0 only)."""
    path = os.path.join(output_dir, "metrics.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    def _log(record: dict):
        if not is_main_process():
            return
        with open(path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(record, default=str) + "\n")

    return _log

try:
    import pynvml

    _NVML_AVAILABLE = True
    pynvml.nvmlInit()
except Exception:
    _NVML_AVAILABLE = False

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
):
    """
    Train the model for one epoch.

    DeepSpeed handles gradient accumulation internally:
    - model_engine.backward() accumulates gradients
    - model_engine.step() only updates weights every gradient_accumulation_steps
    - model_engine.is_gradient_accumulation_boundary() returns True on the step
      where weights are actually updated

    We iterate over the DataLoader one micro-batch at a time. Each micro-batch
    gets a DIFFERENT batch of data (not the same one repeated). global_step
    only increments on optimizer-step boundaries so that max_steps, checkpoint
    intervals, and logging all refer to optimizer steps (matching the reference
    training script's semantics).

    Args:
        model_engine: DeepSpeed model engine
        train_loader: DataLoader for training data
        epoch: Current epoch number
        max_steps: Maximum number of optimizer steps (None for full epoch)
        log_interval: Log every N optimizer steps
        checkpoint_interval: Save checkpoint every N optimizer steps (None to disable)
        output_dir: Directory to save checkpoints
        checkpoint_manager: S3CheckpointManager instance (optional)
        start_step: Optimizer step to start from (for resuming)
        global_step: Global optimizer step counter across all epochs

    Returns:
        Tuple of (average_loss, final_global_step)
    """
    model_engine.train()
    total_loss = 0
    optimizer_steps = 0
    grad_accum_steps = model_engine.gradient_accumulation_steps()

    # Structured JSONL logger (writes to output_dir/metrics.jsonl on rank 0)
    jsonl_log = _jsonl_logger(output_dir or "./logs")

    # Accumulators for averaging loss/tokens across micro-batches within one
    # optimizer step (matches reference script pattern)
    accum_loss = 0.0
    accum_tokens = 0
    step_start_time = time.time()

    # Only show progress bar on main process
    progress_bar = tqdm(
        train_loader, desc=f"Epoch {epoch}", disable=not is_main_process()
    )

    profile_step = 10
    print_profile = True
    prof = FlopsProfiler(model_engine)

    # Track micro-batch index within current optimizer step for resume skipping
    micro_batch_idx = 0
    # How many micro-batches to skip for resume
    skip_micro_batches = start_step * grad_accum_steps

    for i, batch in enumerate(progress_bar):
        # Skip micro-batches if resuming
        if i < skip_micro_batches:
            continue

        # Start profiling at the right optimizer step
        current_optimizer_step = global_step + 1  # what it will be after this step
        if current_optimizer_step == profile_step and micro_batch_idx == 0:
            print("Profile started")
            prof.start_profile()

        # Move batch to device
        input_ids = batch["input_ids"].to(model_engine.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(model_engine.device, non_blocking=True)
        labels = batch["labels"].to(model_engine.device, non_blocking=True)

        # Memory profiling on very first micro-batch
        if i == 0 or (i == skip_micro_batches and skip_micro_batches > 0):
            torch.cuda.reset_peak_memory_stats(model_engine.device)
            mem_before = torch.cuda.memory_allocated(model_engine.device) / 1e9
            print_rank_0(f"\n[MEMORY] Before forward pass: {mem_before:.2f}GB")

        # Forward pass
        is_reversible = hasattr(model_engine.module, 'stack') and hasattr(model_engine.module.stack, 'bootstrap_layer')

        if is_reversible:
            x_input = input_ids[:, :-2].contiguous()
            y_ntp = input_ids[:, 1:-1].contiguous()
            y_mtp = input_ids[:, 2:].contiguous()

            # Pass targets directly — model uses chunked cross-entropy internally
            # to avoid materializing [B, T, 131072] logit tensors (saves ~4+ GB).
            loss_ntp, loss_mtp, aux_loss = model_engine(
                x_input,
                next_token_ids=y_ntp,
                attention_mask=attention_mask[:, :-2].contiguous() if attention_mask is not None else None,
                return_loss=True,
                return_memory=False,
                prev_memory_stream=None,
                ntp_targets=y_ntp,
                mtp_targets=y_mtp,
            )

            # Memory profiling after first forward
            if i == 0 or (i == skip_micro_batches and skip_micro_batches > 0):
                mem_after_fwd = torch.cuda.memory_allocated(model_engine.device) / 1e9
                mem_peak = torch.cuda.max_memory_allocated(model_engine.device) / 1e9
                print_rank_0(f"[MEMORY] After forward pass: {mem_after_fwd:.2f}GB (peak: {mem_peak:.2f}GB)")
                print_rank_0(f"[MEMORY] Forward allocated: {(mem_after_fwd - mem_before):.2f}GB")

            # loss_ntp and loss_mtp are already scalar losses (chunked CE computed in model)
            # FAIL-FAST: Raise immediately on NaN to prevent parameter corruption
            if torch.isnan(loss_ntp) or torch.isnan(loss_mtp) or (aux_loss is not None and torch.isnan(aux_loss)):
                raise FloatingPointError(
                    f"NaN loss detected at epoch {epoch}, micro-batch {i}!\n"
                    f"  loss_ntp={loss_ntp.item()}, loss_mtp={loss_mtp.item()}, aux_loss={aux_loss.item() if aux_loss is not None else None}"
                )

            loss = loss_ntp + 0.3 * loss_mtp
            if aux_loss is not None and aux_loss.numel() > 0:
                loss += aux_loss

            # Free intermediate tensors
            del x_input, y_ntp, y_mtp

        else:
            outputs = model_engine(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        # DeepSpeed backward — internally divides by grad_accum_steps and
        # accumulates gradients
        model_engine.backward(loss)

        # DeepSpeed step — only updates weights at accumulation boundary
        model_engine.step()

        # Accumulate metrics for this micro-batch
        accum_loss += loss.item()
        with torch.no_grad():
            tokens = attention_mask.sum().item()
            accum_tokens += int(tokens)

        # Free batch tensors
        del input_ids, attention_mask, labels

        micro_batch_idx += 1

        # Check if we just completed an optimizer step
        if model_engine.is_gradient_accumulation_boundary():
            # FAIL-FAST: Check all parameters are finite after optimizer step
            with torch.no_grad():
                for name, param in model_engine.module.named_parameters():
                    if param is not None and not torch.isfinite(param).all():
                        raise FloatingPointError(
                            f"Non-finite parameter detected after optimizer step at epoch {epoch}, "
                            f"micro-batch {i}, global_step {global_step + 1}: {name}"
                        )
            
            step_time = time.time() - step_start_time
            global_step += 1
            optimizer_steps += 1

            # Average loss across micro-batches in this optimizer step
            avg_step_loss = accum_loss / grad_accum_steps

            # tok/s = total tokens across ALL micro-batches in this step / wall time
            # Also aggregate across ranks for multi-GPU
            step_tokens_tensor = torch.tensor(float(accum_tokens), device=model_engine.device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(step_tokens_tensor, op=dist.ReduceOp.SUM)
            total_step_tokens = step_tokens_tensor.item()
            tokens_per_sec = total_step_tokens / step_time if step_time > 0 else 0.0

            total_loss += avg_step_loss

            # System metrics
            gpu_util = gpu_mem_used = gpu_mem_total = None
            cpu_util = cpu_mem_used = cpu_mem_total = None
            gpu_table = None
            if enable_system_metrics:
                vm = psutil.virtual_memory()
                cpu_util = psutil.cpu_percent(interval=None)
                cpu_mem_used = vm.used / (1024**3)
                cpu_mem_total = vm.total / (1024**3)

                if _NVML_AVAILABLE and is_main_process() and torch.cuda.is_available():
                    try:
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

                        device_index = (
                            model_engine.local_rank
                            if hasattr(model_engine, "local_rank")
                            else torch.cuda.current_device()
                        )
                        _, gpu_util, gpu_mem_used, gpu_mem_total = next(
                            (r for r in gpu_rows if r[0] == int(device_index)),
                            (device_index, None, None, None),
                        )

                        header = "GPU  Util(%)  Mem(GB Used/Total)"
                        lines = [header, "-" * len(header)]
                        for idx, util, used_gb, total_gb in gpu_rows:
                            lines.append(
                                f"{idx:<3}  {util:>6.0f}%  {used_gb:>5.1f}G/{total_gb:>5.1f}G"
                            )
                        gpu_table = "\n".join(lines)
                    except Exception:
                        pass

            # Profiler stop
            if global_step == profile_step:
                print("Profile stopped\n")
                prof.stop_profile()
                if print_profile:
                    prof.print_model_profile(profile_step=profile_step)
                prof.end_profile()

            # Update progress bar
            postfix = {
                "loss": f"{avg_step_loss:.4f}",
                "global_step": global_step,
                "toks/s": f"{tokens_per_sec:.1f}",
            }
            if enable_system_metrics and gpu_mem_used is not None:
                postfix["gpu_util"] = f"{gpu_util:.0f}%"
                postfix["gpu_mem"] = f"{gpu_mem_used:.1f}G"
            if enable_system_metrics and cpu_util is not None:
                postfix["cpu_util"] = f"{cpu_util:.0f}%"
                postfix["cpu_mem"] = f"{cpu_mem_used:.1f}G"
            progress_bar.set_postfix(postfix)

            # Log at optimizer-step granularity
            if optimizer_steps % log_interval == 0:
                msg = (
                    f"Epoch {epoch}, Global Step {global_step}, "
                    f"Loss: {avg_step_loss:.4f}, Tokens/s: {tokens_per_sec:.1f}, "
                    f"Tokens: {int(total_step_tokens)}"
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

                if enable_system_metrics and is_main_process() and gpu_table:
                    print_rank_0("\nGPU Utilization / Memory (all devices):")
                    print_rank_0(gpu_table)

            # Structured JSONL metrics (every optimizer step, rank 0 only)
            jsonl_log({
                "epoch": epoch,
                "global_step": global_step,
                "loss": avg_step_loss,
                "tokens_per_sec": tokens_per_sec,
                "tokens": int(total_step_tokens),
                "step_time_s": step_time,
                "gpu_util_pct": gpu_util,
                "gpu_mem_gb": gpu_mem_used,
                "cpu_util_pct": cpu_util,
                "lr": model_engine.get_lr()[0] if hasattr(model_engine, "get_lr") else None,
                "timestamp": time.time(),
            })

            # Save checkpoint at optimizer-step granularity
            if checkpoint_interval is not None and global_step % checkpoint_interval == 0:
                checkpoint_tag = f"epoch{epoch}_step{global_step}"
                print_rank_0(
                    f"\nSaving checkpoint at epoch {epoch}, global_step {global_step}..."
                )
                client_state = {
                    "epoch": epoch,
                    "step": optimizer_steps,
                    "global_step": global_step,
                    "loss": avg_step_loss,
                }

                if checkpoint_manager:
                    checkpoint_manager.save_checkpoint(
                        model_engine,
                        step=global_step,
                        tag=checkpoint_tag,
                        client_state=client_state,
                    )
                elif output_dir:
                    save_checkpoint(model_engine, output_dir, tag=checkpoint_tag)

            # Reset accumulators for next optimizer step
            accum_loss = 0.0
            accum_tokens = 0
            step_start_time = time.time()
            micro_batch_idx = 0

            # Early stopping — max_steps refers to optimizer steps
            if max_steps is not None and optimizer_steps >= max_steps:
                break

    avg_loss = total_loss / optimizer_steps if optimizer_steps > 0 else 0
    print_rank_0(f"Epoch {epoch} - Training Average Loss: {avg_loss:.4f}")

    return avg_loss, global_step


def evaluate(model_engine, data_loader, phase="Evaluation", max_steps=None):
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
            attention_mask = batch["attention_mask"].to(model_engine.device, non_blocking=True)
            labels = batch["labels"].to(model_engine.device, non_blocking=True)

            # Forward pass
            # Check if this is a reversible model
            is_reversible = hasattr(model_engine.module, 'stack') and hasattr(model_engine.module.stack, 'bootstrap_layer')

            if is_reversible:
                # Reversible model: returns (logits_ntp, logits_mtp, aux_loss)
                x_input = input_ids[:, :-2].contiguous()
                y_ntp = input_ids[:, 1:-1].contiguous()
                y_mtp = input_ids[:, 2:].contiguous()

                # No autocast — model is pre-cast to bf16 (see train_epoch comment)
                logits_ntp, logits_mtp, aux_loss = model_engine.module(
                    x_input,
                    next_token_ids=y_ntp,
                    attention_mask=attention_mask[:, :-2].contiguous() if attention_mask is not None else None,
                    return_loss=True,
                    return_memory=False,
                    prev_memory_stream=None
                )

                vocab_size = logits_ntp.size(-1)
                loss_ntp = torch.nn.functional.cross_entropy(
                    logits_ntp.float().view(-1, vocab_size),
                    y_ntp.view(-1)
                )
                del logits_ntp

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
            # Check if this is a reversible model
            is_reversible = hasattr(model_engine.module, 'stack') and hasattr(model_engine.module.stack, 'bootstrap_layer')

            if is_reversible:
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
