"""
Local test script for loss spike detection and recovery.

Trains a small (~25M param) GPT-style model on wikitext-2 using plain PyTorch
(no DeepSpeed, no Triton, no CUDA required). Exercises the full spike recovery
pipeline: loss detection, grad norm monitoring, embedding norm tracking,
automatic escalation, cooldown, and checkpoint rollback.

Usage:
    cd llm
    uv run python tests/local_spike_recovery_test.py

Works on M1 Mac (MPS), Linux (CPU/CUDA), etc.
"""

import os
import math
import tempfile
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from llm.loss_spike_recovery import (
    LossSpikeDetector,
    RecoveryAction,
    auto_select_action,
    compute_grad_norm,
    compute_embedding_norms,
)


# ---------------------------------------------------------------------------
# Minimal GPT model (~25M params with default config)
# ---------------------------------------------------------------------------


@dataclass
class SmallGPTConfig:
    vocab_size: int = 32000
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    intermediate_size: int = 1536
    max_seq_len: int = 256
    dropout: float = 0.0


class SmallGPT(nn.Module):
    """Minimal GPT for local spike recovery testing."""

    def __init__(self, config: SmallGPTConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, config: SmallGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: SmallGPTConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, config: SmallGPTConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Data loading (wikitext-2 from HuggingFace)
# ---------------------------------------------------------------------------


def load_wikitext2(tokenizer, max_length=256, batch_size=4):
    """Load wikitext-2-raw-v1 from HuggingFace and create a DataLoader."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

    def tokenize_fn(examples):
        tok = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None,
        )
        tok["labels"] = [ids.copy() for ids in tok["input_ids"]]
        return tok

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# ---------------------------------------------------------------------------
# Training loop with spike recovery
# ---------------------------------------------------------------------------


def train_with_spike_recovery(
    num_steps: int = 100,
    inject_spike_at: int | None = 30,
    inject_grad_spike_at: int | None = 60,
    batch_size: int = 4,
    max_length: int = 128,
):
    """
    Train a small GPT on wikitext-2 with full spike recovery instrumentation.

    Args:
        num_steps: Total training steps to run.
        inject_spike_at: Step at which to inject a fake loss spike (None to disable).
        inject_grad_spike_at: Step at which to inject a gradient spike (None to disable).
        batch_size: Micro-batch size.
        max_length: Sequence length.
    """
    print("=" * 70)
    print("  Local Spike Recovery Test")
    print("=" * 70)

    # --- Device selection ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n  Device: {device}")

    # --- Tokenizer ---
    from transformers import AutoTokenizer
    print("  Loading tokenizer (GPT-2)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # --- Model ---
    config = SmallGPTConfig(vocab_size=vocab_size, max_seq_len=max_length)
    model = SmallGPT(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,} (~{total_params / 1e6:.1f}M)")

    # --- Data ---
    print("  Loading wikitext-2...")
    train_loader = load_wikitext2(tokenizer, max_length=max_length, batch_size=batch_size)
    print(f"  Training batches: {len(train_loader)}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    # --- Spike detector ---
    detector = LossSpikeDetector(
        window_size=10,
        z_threshold=2.5,
        min_spike_ratio=2.0,
        min_abs_delta=0.3,
        cooldown_steps=5,
    )
    grad_norm_threshold = 50.0
    patience_skip = 2
    patience_lr = 5
    lr_reduction_factor = 0.5

    # --- Checkpoint state (simulated) ---
    checkpoint_dir = tempfile.mkdtemp(prefix="spike_test_ckpt_")
    last_checkpoint_tag = None
    last_checkpoint_state = None

    def save_checkpoint(step):
        nonlocal last_checkpoint_tag, last_checkpoint_state
        tag = f"step_{step}"
        last_checkpoint_tag = tag
        last_checkpoint_state = {
            "step": step,
            "model_state": {k: v.clone() for k, v in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
        }
        print(f"    [CKPT] Saved checkpoint '{tag}'")

    def load_checkpoint():
        nonlocal last_checkpoint_state
        if last_checkpoint_state is None:
            raise RuntimeError("No checkpoint available")
        model.load_state_dict(last_checkpoint_state["model_state"])
        optimizer.load_state_dict(last_checkpoint_state["optimizer_state"])
        detector.reset()
        return last_checkpoint_state["step"]

    # --- Training ---
    print(f"\n  Starting training for {num_steps} steps...")
    if inject_spike_at:
        print(f"  Will inject loss spike at step {inject_spike_at}")
    if inject_grad_spike_at:
        print(f"  Will inject grad spike at step {inject_grad_spike_at}")
    print()

    model.train()
    data_iter = iter(train_loader)
    global_step = 0
    spike_events = []

    while global_step < num_steps:
        # Get next batch (cycle through data)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward
        logits = model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=tokenizer.pad_token_id,
        )

        # Inject fake loss spike
        actual_loss = loss.item()
        loss_for_detection = actual_loss
        if inject_spike_at and global_step == inject_spike_at:
            loss_for_detection = actual_loss * 5.0
            print(f"  [INJECT] Fake loss spike at step {global_step}: "
                  f"{actual_loss:.4f} -> {loss_for_detection:.4f}")

        # --- Loss spike detection ---
        if detector.update(loss_for_detection):
            stats = detector.get_stats()
            action = auto_select_action(
                spike_count=detector.spike_count + 1,
                patience_skip=patience_skip,
                patience_lr=patience_lr,
                last_checkpoint_tag=last_checkpoint_tag,
            )
            print(f"  [LOSS SPIKE] step={global_step} "
                  f"loss={stats['current_loss']:.4f} "
                  f"(mean={stats['window_mean']:.4f}, "
                  f"ratio={stats['spike_ratio']:.2f}x) "
                  f"spike #{detector.spike_count + 1} -> {action.name}")
            spike_events.append(("LOSS_SPIKE", global_step, action.name))
            detector.record_spike_action()

            if action == RecoveryAction.SKIP_BATCH:
                global_step += 1
                continue
            elif action == RecoveryAction.REDUCE_LR:
                for pg in optimizer.param_groups:
                    pg["lr"] *= lr_reduction_factor
                print(f"    LR reduced -> {optimizer.param_groups[0]['lr']:.2e}")
                global_step += 1
                continue
            elif action == RecoveryAction.ROLLBACK_CHECKPOINT:
                if last_checkpoint_state:
                    restored_step = load_checkpoint()
                    print(f"    Rolled back to step {restored_step}")
                global_step += 1
                continue

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Inject gradient spike
        if inject_grad_spike_at and global_step == inject_grad_spike_at:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data *= 100.0
            print(f"  [INJECT] Gradient spike at step {global_step}")

        # --- Gradient norm check ---
        grad_norm = compute_grad_norm(model)
        if grad_norm > grad_norm_threshold:
            action = auto_select_action(
                spike_count=detector.spike_count + 1,
                patience_skip=patience_skip,
                patience_lr=patience_lr,
                last_checkpoint_tag=last_checkpoint_tag,
            )
            print(f"  [GRAD SPIKE] step={global_step} "
                  f"grad_norm={grad_norm:.2f} (threshold={grad_norm_threshold}) "
                  f"-> {action.name}")
            spike_events.append(("GRAD_SPIKE", global_step, action.name))
            detector.record_spike_action()

            if action in (RecoveryAction.SKIP_BATCH, RecoveryAction.REDUCE_LR):
                optimizer.zero_grad()
                if action == RecoveryAction.REDUCE_LR:
                    for pg in optimizer.param_groups:
                        pg["lr"] *= lr_reduction_factor
                    print(f"    LR reduced -> {optimizer.param_groups[0]['lr']:.2e}")
                global_step += 1
                continue

        # Optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Embedding norms (every 10 steps)
        if global_step % 10 == 0:
            emb_norms = compute_embedding_norms(model)
            emb_str = ", ".join(f"{k}={v:.2f}" for k, v in emb_norms.items())
            if emb_str:
                emb_str = f" | {emb_str}"
        else:
            emb_str = ""

        # Logging
        if global_step % 5 == 0:
            print(f"  step {global_step:4d}  loss={actual_loss:.4f}  "
                  f"grad_norm={grad_norm:.2f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}{emb_str}")

        # Checkpoint every 20 steps
        if global_step > 0 and global_step % 20 == 0:
            save_checkpoint(global_step)

        global_step += 1

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  Training Complete")
    print("=" * 70)
    print(f"  Total steps: {global_step}")
    print(f"  Final LR: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"  Spike events: {len(spike_events)}")
    for event_type, step, action in spike_events:
        print(f"    step {step}: {event_type} -> {action}")

    final_norms = compute_embedding_norms(model)
    print(f"  Final embedding norms:")
    for k, v in final_norms.items():
        print(f"    {k}: {v:.4f}")
    print()

    return spike_events


if __name__ == "__main__":
    events = train_with_spike_recovery(
        num_steps=80,
        inject_spike_at=30,
        inject_grad_spike_at=55,
        batch_size=4,
        max_length=128,
    )
