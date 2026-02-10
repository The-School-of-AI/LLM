"""
Training utilities for DeepSpeed.

This module contains training, evaluation, and inference functions
for training language models with DeepSpeed optimization.
"""

import torch
from tqdm import tqdm
import time
import torch.distributed as dist
import psutil

from .utils import is_main_process, print_rank_0
from deepspeed.profiling.flops_profiler import FlopsProfiler

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
        input_ids = batch["input_ids"].to(model_engine.device)
        attention_mask = batch["attention_mask"].to(model_engine.device)
        labels = batch["labels"].to(model_engine.device)

        # Forward pass
        # Check if this is a reversible model (has custom forward signature)
        is_reversible = hasattr(model_engine.module, 'stack') and hasattr(model_engine.module.stack, 'bootstrap_layer')
        
        if is_reversible:
            # Reversible model: returns (logits_ntp, logits_mtp, aux_loss)
            logits_ntp, logits_mtp, aux_loss = model_engine(
                input_ids, 
                next_token_ids=None,  # Will be computed from shifted labels if needed
                attention_mask=attention_mask,
                return_loss=True
            )
            
            # Compute cross-entropy loss for next token prediction
            # Shift logits and labels for causal LM
            shift_logits = logits_ntp[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            ntp_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Add auxiliary loss (from MoE routing, etc.)
            if aux_loss is not None and aux_loss.numel() > 0:
                loss = ntp_loss + aux_loss
            else:
                loss = ntp_loss
            
            # MTP loss (if enabled and logits_mtp is not None)
            # For now, we focus on NTP loss
            
        else:
            # Standard transformer model
            outputs = model_engine(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        # Backward pass
        model_engine.backward(loss)

        # Update weights
        model_engine.step()

        # Compute tokens per second for this step
        step_time = time.time() - step_start_time
        with torch.no_grad():
            # Count tokens in this batch using attention mask (1s for real tokens)
            tokens = attention_mask.sum().float()
            # Aggregate across all ranks if distributed is initialized
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
            tokens = tokens.item()
        tokens_per_sec = tokens / step_time if step_time > 0 else 0.0

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
            "loss": f"{loss.item():.4f}",
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

        # Log periodically
        if i % log_interval == 0:
            msg = (
                f"Epoch {epoch}, Step {i}, Global Step {global_step}, "
                f"Loss: {loss.item():.4f}, Tokens/s: {tokens_per_sec:.1f}, Tokens: {int(tokens)}"
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
            input_ids = batch["input_ids"].to(model_engine.device)
            attention_mask = batch["attention_mask"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)

            # Forward pass
            # Check if this is a reversible model
            is_reversible = hasattr(model_engine.module, 'stack') and hasattr(model_engine.module.stack, 'bootstrap_layer')
            
            if is_reversible:
                # Reversible model: returns (logits_ntp, logits_mtp, aux_loss)
                logits_ntp, logits_mtp, aux_loss = model_engine(
                    input_ids,
                    next_token_ids=None,
                    attention_mask=attention_mask,
                    return_loss=True
                )
                
                # Compute cross-entropy loss for next token prediction
                shift_logits = logits_ntp[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss()
                ntp_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                # Add auxiliary loss
                if aux_loss is not None and aux_loss.numel() > 0:
                    loss = ntp_loss + aux_loss
                else:
                    loss = ntp_loss
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
    input_ids = inputs["input_ids"].to(model_engine.device)
    attention_mask = inputs["attention_mask"].to(model_engine.device)

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
