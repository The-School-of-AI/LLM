"""
Training Setup and Checkpointing for SmolLM

Includes:
- Training setup (optimizer, device, etc.)
- Checkpoint save/load with proper state management
- Learning rate schedules
"""

import glob
import math
import os
import time
from typing import Tuple

import torch

# ============================================================================
# TRAINING SETUP
# ============================================================================


def setup_training(model, device_type="mps"):
    """
    Setup training components - optimized for Mac M1 (MPS).

    Args:
        model: Model to train
        device_type: Device type ("mps", "cuda", or "cpu")

    Returns:
        Tuple of (device, optimizer, lr_scheduler, scaler)
    """
    # Prioritize MPS for Mac M1
    if (
        device_type == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    elif device_type == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    # AdamW optimizer with Karpathy-style settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.0,  # 0.0 for MoE models (Paper uses 0.1 for dense models)
        eps=1e-10,  # Karpathy-style (vs PyTorch default 1e-8)
    )

    # No scheduler - we'll use manual LR schedule
    lr_scheduler = None

    # No scaler - deepscreen doesn't use GradScaler, just autocast directly
    scaler = None

    return device, optimizer, lr_scheduler, scaler


# ============================================================================
# LEARNING RATE SCHEDULE
# ============================================================================


def get_learning_rate(step: int, config) -> float:
    """
    Calculate learning rate for given step using cosine schedule with warmup.

    Args:
        step: Current training step
        config: Training configuration

    Returns:
        Learning rate for this step
    """
    max_lr = config.training.max_lr
    min_lr = config.training.min_lr
    warmup_steps = config.training.warmup_steps

    # Special plateau for stability (500-530 steps)
    if 500 <= step <= 530:
        return 3.0e-5

    # Warmup phase
    elif step < warmup_steps:
        return max_lr * step / warmup_steps

    # Main cosine decay (warmup -> 10000 steps)
    elif step <= 10000:
        progress = (step - warmup_steps) / (10000 - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    # Fine-tuning cosine tail (10000 -> 12000 steps)
    elif step <= 12000:
        tail_start_lr = 3.0e-5
        tail_end_lr = 1.0e-6
        progress = (step - 10000) / (12000 - 10000)
        progress = min(max(progress, 0.0), 1.0)
        return tail_end_lr + 0.5 * (tail_start_lr - tail_end_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    # Final plateau
    else:
        return 1.0e-6


def update_learning_rate(optimizer, lr: float):
    """Update optimizer learning rate"""
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ============================================================================
# CHECKPOINTING
# ============================================================================


def save_checkpoint(
    model,
    optimizer,
    lr_scheduler,
    step,
    loss,
    embedding_type,
    save_dir="checkpoints",
    ops=None,
    s3_key=None,
    tag="temporary",
):
    """
    Save model checkpoint with embedding_type prefix.

    Args:
        model: Model to save
        optimizer: Optimizer state
        lr_scheduler: LR scheduler state (can be None)
        step: Current training step
        loss: Current loss value
        embedding_type: Type of embedding ("fourier" or "baseline")
        save_dir: Directory to save checkpoints
        ops: TrainingOps instance (optional). If provided, registers the
             checkpoint in the observability stack automatically.
        s3_key: S3 URI if checkpoint will be uploaded (optional).
        tag: Governance tag: "temporary", "growth", "lora", "release_candidate".
    """
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            lr_scheduler.state_dict() if lr_scheduler is not None else None
        ),
        "loss": loss,
        "embedding_type": embedding_type,
    }

    # Save with embedding_type prefix and step number
    path = os.path.join(save_dir, f"{embedding_type}_checkpoint_step_{step:07d}.pt")
    t0 = time.time()
    torch.save(checkpoint, path)
    duration_s = time.time() - t0
    size_bytes = os.path.getsize(path)

    # Also save as latest for easy resuming
    latest_path = os.path.join(save_dir, f"{embedding_type}_latest.pt")
    torch.save(checkpoint, latest_path)

    print(f"💾 Checkpoint saved: {path}")
    print(f"💾 Latest checkpoint: {latest_path}")

    # Register with P12 observability (if TrainingOps is available)
    if ops is not None:
        ops.log_checkpoint(
            step=step,
            path=path,
            s3_key=s3_key,
            loss=float(loss) if loss is not None else 0.0,
            tag=tag,
            duration_s=duration_s,
            size_bytes=size_bytes,
        )


def load_checkpoint(
    model,
    optimizer,
    lr_scheduler,
    embedding_type,
    save_dir="checkpoints",
    checkpoint_path=None,
) -> Tuple[int, float]:
    """
    Load checkpoint from a specific path or find the latest checkpoint.

    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        lr_scheduler: LR scheduler to load state into (can be None)
        embedding_type: Type of embedding (for auto-discovery)
        save_dir: Directory to search for checkpoints
        checkpoint_path: Optional specific checkpoint path to load

    Returns:
        Tuple of (step, loss)
    """
    # If a specific checkpoint path is provided, use it directly
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint file not found: {checkpoint_path}")
            return 0, 0.0
        latest_path = checkpoint_path
    else:
        # Auto-discover latest checkpoint
        latest_path = os.path.join(save_dir, f"{embedding_type}_latest.pt")

        if not os.path.exists(latest_path):
            # Try to find the highest step checkpoint
            pattern = os.path.join(save_dir, f"{embedding_type}_checkpoint_step_*.pt")
            checkpoints = glob.glob(pattern)
            if not checkpoints:
                return 0, 0.0

            # Sort by step number (extract from filename)
            def get_step(path):
                basename = os.path.basename(path)
                try:
                    step_str = basename.split("_step_")[1].split(".pt")[0]
                    return int(step_str)
                except Exception:
                    return 0

            checkpoints.sort(key=get_step, reverse=True)
            latest_path = checkpoints[0]

    try:
        checkpoint = torch.load(latest_path, map_location="cpu")

        # Load model state (with strict=False to handle architecture changes)
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        if missing_keys:
            print(f"[RESUME] Warning: Missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"[RESUME] Warning: Unexpected keys: {unexpected_keys[:5]}...")

        # Load optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state (if exists and not None)
        if lr_scheduler is not None and "scheduler_state_dict" in checkpoint:
            if checkpoint["scheduler_state_dict"] is not None:
                lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        step = checkpoint.get("step", 0)
        loss = checkpoint.get("loss", 0.0)

        print(f"✅ Loaded checkpoint from {latest_path} at step {step}")
        return step, loss

    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        import traceback

        traceback.print_exc()
        return 0, 0.0


# ============================================================================
# TOKENIZER SETUP
# ============================================================================


def setup_tokenizer(tokenizer, target_vocab_size=50272, cache_dir="checkpoints_70b"):
    """
    Setup tokenizer with Qwen special tokens and pad to target vocab size.
    CACHED: Only does padding once, then loads from cache.

    Args:
        tokenizer: Base tokenizer (e.g., GPT2Tokenizer)
        target_vocab_size: Target vocabulary size
        cache_dir: Directory to cache the tokenizer

    Returns:
        Modified tokenizer
    """
    # Check for cached tokenizer
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenizer_padded_{target_vocab_size}.pt"

    if os.path.exists(cache_file):
        print(f"   ✓ Loading cached tokenizer from {cache_file}")
        tokenizer = torch.load(cache_file)
        print(f"   ✓ Tokenizer loaded: {len(tokenizer):,} tokens")
        return tokenizer

    print("   ⚠️  No tokenizer cache, creating padded tokenizer (takes ~30s)...")

    # Add Qwen-style special tokens
    special_tokens = {
        "additional_special_tokens": [
            "<|im_start|>",
            "<|im_end|>",
            "<|assistant|>",
            "<|user|>",
        ]
    }
    tokenizer.add_special_tokens(special_tokens)

    # Pad vocab to target size
    base_vocab_size = len(tokenizer)
    if base_vocab_size < target_vocab_size:
        num_padding = target_vocab_size - base_vocab_size
        print(f"   🔄 Adding {num_padding:,} padding tokens in ONE batch...")

        # Add ALL tokens at once (much faster than batches!)
        padding_tokens = [f"<|pad_{i}|>" for i in range(num_padding)]
        tokenizer.add_special_tokens({"additional_special_tokens": padding_tokens})

        print(f"   ✓ Added {num_padding:,} padding tokens")

    # CRITICAL: Use a SEPARATE pad token, NOT eos_token!
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|pad_0|>"

    print(f"   📚 Tokenizer vocab size: {len(tokenizer):,}")

    # Cache for next time
    print(f"   💾 Caching tokenizer to {cache_file}...")
    torch.save(tokenizer, cache_file)
    print("   ✓ Cached! Next run loads instantly.")

    return tokenizer
