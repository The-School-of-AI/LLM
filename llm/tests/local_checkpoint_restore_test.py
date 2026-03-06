"""
Integration test for optimizer & scheduler checkpoint restore.

Trains a small (~28M param) GPT-style model on wikitext-2 using plain PyTorch
(no DeepSpeed, no Triton, no CUDA required). Saves a checkpoint, loads it into
a fresh model, and verifies the restore using the actual production
verify_optimizer_scheduler_restored function from llm.utils.

Usage:
    cd llm
    uv run python tests/local_checkpoint_restore_test.py

Works on M1 Mac (MPS), Linux (CPU/CUDA), etc.
"""

import tempfile
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from llm.utils import verify_optimizer_scheduler_restored


@dataclass
class SmallGPTConfig:
    vocab_size: int = 32000
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    intermediate_size: int = 512
    max_seq_len: int = 128
    dropout: float = 0.0


class TransformerBlock(nn.Module):
    def __init__(self, config: SmallGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = nn.MultiheadAttention(
            config.hidden_size, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )

    def forward(self, x, mask=None):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x


class SmallGPT(nn.Module):
    def __init__(self, config: SmallGPTConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=input_ids.device)
        for layer in self.layers:
            x = layer(x, mask=causal_mask)
        x = self.norm(x)
        return self.lm_head(x)


def load_wikitext2(tokenizer, max_length=128, batch_size=4):
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


def train_loop(model, optimizer, scheduler, train_loader, device, num_steps):
    model.train()
    data_iter = iter(train_loader)
    losses = []

    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if step % 5 == 0:
            print(f"  step {step:3d}  loss={loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

    return losses


def run_checkpoint_restore_test(
    num_train_steps: int = 20,
    num_continue_steps: int = 10,
    batch_size: int = 4,
    max_length: int = 128,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    config = SmallGPTConfig(vocab_size=vocab_size, max_seq_len=max_length)
    model = SmallGPT(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count:,}")

    train_loader = load_wikitext2(tokenizer, max_length=max_length, batch_size=batch_size)

    total_steps = num_train_steps + num_continue_steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # ========== Phase 1: Train and save ==========
    print(f"\n{'=' * 60}")
    print(f"Phase 1: Training for {num_train_steps} steps")
    print(f"{'=' * 60}")

    train_loop(model, optimizer, scheduler, train_loader, device, num_train_steps)

    lr_before_save = scheduler.get_last_lr()[0]
    sched_epoch_before = scheduler.state_dict()["last_epoch"]

    checkpoint_dir = tempfile.mkdtemp(prefix="ckpt_restore_test_")
    ckpt_path = f"{checkpoint_dir}/checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": num_train_steps,
        },
        ckpt_path,
    )
    print(f"\nCheckpoint saved to {ckpt_path}")
    print(f"  LR at save: {lr_before_save:.2e}")
    print(f"  Scheduler last_epoch: {sched_epoch_before}")

    # ========== Phase 2: Fresh model + load + verify ==========
    print(f"\n{'=' * 60}")
    print(f"Phase 2: Loading checkpoint into fresh model")
    print(f"{'=' * 60}")

    model2 = SmallGPT(config).to(device)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=total_steps)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model2.load_state_dict(ckpt["model"])
    optimizer2.load_state_dict(ckpt["optimizer"])
    scheduler2.load_state_dict(ckpt["scheduler"])

    result = verify_optimizer_scheduler_restored(
        optimizer2, scheduler2, expected_global_step=ckpt["global_step"]
    )
    print(f"\nverify_optimizer_scheduler_restored result:")
    print(f"  restored_count: {result['restored_count']}/{result['total_count']}")
    print(f"  current_lr:     {result['current_lr']:.2e}")
    print(f"  last_epoch:     {result['last_epoch']}")

    lr_after_load = scheduler2.get_last_lr()[0]
    assert lr_after_load == lr_before_save, (
        f"LR mismatch after restore: {lr_after_load} != {lr_before_save}"
    )
    assert result["restored_count"] == result["total_count"], (
        f"Not all params restored: {result['restored_count']}/{result['total_count']}"
    )
    assert result["last_epoch"] == num_train_steps, (
        f"Scheduler epoch mismatch: {result['last_epoch']} != {num_train_steps}"
    )
    print("All assertions passed.")

    # ========== Phase 3: Continue training ==========
    print(f"\n{'=' * 60}")
    print(f"Phase 3: Continuing training for {num_continue_steps} steps")
    print(f"{'=' * 60}")

    train_loop(model2, optimizer2, scheduler2, train_loader, device, num_continue_steps)

    lr_final = scheduler2.get_last_lr()[0]
    epoch_final = scheduler2.state_dict()["last_epoch"]
    assert epoch_final == num_train_steps + num_continue_steps, (
        f"Scheduler epoch after continue: {epoch_final} != {num_train_steps + num_continue_steps}"
    )

    print(f"\n{'=' * 60}")
    print(f"PASSED: Optimizer & scheduler restore verified end-to-end")
    print(f"  Final LR: {lr_final:.2e}")
    print(f"  Final scheduler epoch: {epoch_final}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_checkpoint_restore_test()
