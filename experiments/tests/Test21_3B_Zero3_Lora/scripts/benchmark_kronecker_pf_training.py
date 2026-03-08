#!/usr/bin/env python3
"""
Benchmark Kronecker PF_table at ACTUAL per-GPU training micro-batch sizes.

Shows the real overhead of gpu_dynamic vs gpu_table as a percentage of
estimated step time for each model configuration.

Usage (on AWS):
  cd /mnt/local-nvme/LLM/experiments/tests/Test19/code
  python3 ../scripts/benchmark_kronecker_pf_training.py

Single-GPU benchmark, no DeepSpeed required.
"""

import os
import sys
import time

import numpy as np
import torch

CODE_DIR = os.path.join(os.path.dirname(__file__), "..", "code")
sys.path.insert(0, CODE_DIR)

from src.models.recurrence_model_3b_moe import KroneckerConfig, KroneckerEmbeddings


# ---------------------------------------------------------------------------
# gpu_table: precomputed full PF table on GPU (current approach)
# ---------------------------------------------------------------------------
class EmbedGPUTable(torch.nn.Module):
    def __init__(self, pf_table_bf16: torch.Tensor):
        super().__init__()
        self.register_buffer("PF_table", pf_table_bf16, persistent=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.PF_table[token_ids]


# ---------------------------------------------------------------------------
# gpu_dynamic: compute PF on-the-fly from compact byte buffers
# ---------------------------------------------------------------------------
class EmbedGPUDynamic(torch.nn.Module):
    def __init__(self, vocab_words, pf_codec: KroneckerEmbeddings, chunk_tokens: int = 2048):
        super().__init__()
        self.pos_dim = pf_codec.POS_DIM
        self.char_dim = pf_codec.CHAR_DIM
        self.D = pf_codec.D
        self.length_normalize = pf_codec.cfg.length_normalize
        self.chunk_tokens = chunk_tokens

        # Build compact byte table: [V, POS_DIM] uint8 + [V] int16
        token_bytes = np.zeros((len(vocab_words), self.pos_dim), dtype=np.uint8)
        token_lens = np.zeros((len(vocab_words),), dtype=np.int16)
        for i, word in enumerate(vocab_words):
            if not word:
                continue
            byte_seq = word.encode("utf-8")
            if len(byte_seq) > self.pos_dim:
                if pf_codec.cfg.truncate_long_words:
                    byte_seq = pf_codec._utf8_safe_truncate(byte_seq, self.pos_dim)
                else:
                    byte_seq = byte_seq[:self.pos_dim]
            L = len(byte_seq)
            if L == 0:
                continue
            token_bytes[i, :L] = np.frombuffer(byte_seq, dtype=np.uint8, count=L)
            token_lens[i] = L

        self.register_buffer("_token_bytes", torch.from_numpy(token_bytes), persistent=False)
        self.register_buffer("_token_lens", torch.from_numpy(token_lens), persistent=False)
        self.register_buffer("_pos_ids", torch.arange(self.pos_dim, dtype=torch.long), persistent=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        flat_ids = token_ids.reshape(-1)
        total = flat_ids.numel()
        device = token_ids.device
        out = torch.empty((total, self.D), device=device, dtype=torch.bfloat16)

        for start in range(0, total, self.chunk_tokens):
            end = min(start + self.chunk_tokens, total)
            ids_chunk = flat_ids[start:end]
            chunk_size = ids_chunk.shape[0]

            bytes_chunk = self._token_bytes.index_select(0, ids_chunk).to(torch.long)
            lens_chunk = self._token_lens.index_select(0, ids_chunk).to(torch.long)

            pf_chunk = torch.zeros((chunk_size, self.D), device=device, dtype=torch.float32)
            pos = self._pos_ids.unsqueeze(0).expand(chunk_size, -1)
            lin_idx = bytes_chunk * self.pos_dim + pos

            valid = pos < lens_chunk.unsqueeze(1)

            if self.length_normalize:
                scales = torch.rsqrt(lens_chunk.clamp_min(1).to(torch.float32))
                src = valid.to(torch.float32) * scales.unsqueeze(1)
            else:
                src = valid.to(torch.float32)

            pf_chunk.scatter_add_(dim=1, index=lin_idx, src=src)

            pf_centered = pf_chunk - pf_chunk.mean(dim=-1, keepdim=True)
            pf_std = pf_centered.std(dim=-1, keepdim=True) + 1e-6
            out[start:end] = (pf_centered / pf_std).to(torch.bfloat16)

        return out.view(*token_ids.shape, self.D)


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------
def benchmark(fn, warmup=10, iters=50):
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    median = times[len(times) // 2]
    return median


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # --- Build vocabulary ---
    print("Loading tokenizer...")
    tokenizer_dir = os.path.join(CODE_DIR, "src", "tokenizer")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    vocab_size = 131072

    bpe_vocab = []
    for i in range(vocab_size):
        try:
            token = tokenizer.decode([i])
            bpe_vocab.append(token if token else "<unk_{}>".format(i))
        except Exception:
            bpe_vocab.append("<unk_{}>".format(i))

    # --- Build PF codec ---
    pf_config = KroneckerConfig(CHAR_DIM=256, POS_DIM=32, D=8192,
                                 length_normalize=True, truncate_long_words=True)
    pf_codec = KroneckerEmbeddings(pf_config)

    # --- Build gpu_table ---
    print("Building gpu_table...")
    PF_np = pf_codec.encode_batch(bpe_vocab)
    if PF_np.dtype != np.float32:
        PF_np = PF_np.astype(np.float32, copy=False)
    pf_tensor = torch.from_numpy(PF_np)
    pf_centered = pf_tensor - pf_tensor.mean(dim=-1, keepdim=True)
    pf_std = pf_centered.std(dim=-1, keepdim=True) + 1e-6
    pf_norm = (pf_centered / pf_std).to(torch.bfloat16)
    embed_table = EmbedGPUTable(pf_norm).to(device)
    del PF_np, pf_tensor, pf_centered, pf_std, pf_norm

    # --- Build gpu_dynamic ---
    print("Building gpu_dynamic buffers...")
    embed_dyn = EmbedGPUDynamic(bpe_vocab, pf_codec, chunk_tokens=2048).to(device)

    # --- Memory ---
    table_mem_gb = embed_table.PF_table.numel() * 2 / 1e9
    dyn_mem_mb = (embed_dyn._token_bytes.numel() * 1 + embed_dyn._token_lens.numel() * 2) / 1e6
    print("Memory: gpu_table={:.2f} GB, gpu_dynamic={:.1f} MB".format(table_mem_gb, dyn_mem_mb))

    # --- Actual training configs ---
    # (batch, seqlen, description, estimated_step_ms)
    configs = [
        (1, 4096,   "8B current (micro=1, SL=4k)",    9000),
        (1, 16384,  "70B LoRA (micro=1, SL=16k)",     25000),
        (1, 65536,  "Future (micro=1, SL=64k)",       80000),
        (2, 4096,   "3B current (micro=2, SL=4k)",    2500),
        (4, 4096,   "1B current (micro=4, SL=4k)",    1600),
    ]

    print("\n" + "=" * 80)
    print("  ACTUAL TRAINING MICRO-BATCH SIZES — Kronecker PF Overhead")
    print("=" * 80)
    header = "{:<38s} {:>8s} {:>8s} {:>8s} {:>8s} {:>10s}".format(
        "Config", "Tokens", "Table", "Dynamic", "Extra", "% of step")
    print(header)
    print("-" * 80)

    for B, T, desc, step_ms in configs:
        total_tokens = B * T

        # Check if output would OOM (rough estimate: tokens * 8192 * 2 bytes)
        out_bytes = total_tokens * 8192 * 2
        if out_bytes > 30e9:  # > 30 GB output tensor, skip
            print("{:<38s} {:>8s}  SKIPPED (output tensor too large)".format(desc, "{:,}".format(total_tokens)))
            continue

        token_ids = torch.randint(0, vocab_size, (B, T), device=device)

        med_table = benchmark(lambda: embed_table(token_ids), warmup=10, iters=50)
        med_dyn = benchmark(lambda: embed_dyn(token_ids), warmup=10, iters=50)
        extra_ms = med_dyn - med_table
        pct = extra_ms / step_ms * 100

        print("{:<38s} {:>8s} {:>7.2f}ms {:>7.2f}ms {:>7.2f}ms {:>9.3f}%".format(
            desc,
            "{:,}".format(total_tokens),
            med_table,
            med_dyn,
            extra_ms,
            pct,
        ))

        del token_ids
        torch.cuda.empty_cache()

    print("=" * 80)
    print("\nConclusion: Extra time from gpu_dynamic as % of full training step.")
    print("Memory saved: {:.2f} GB per GPU ({:.1f} MB buffers replace {:.2f} GB table).".format(
        table_mem_gb - dyn_mem_mb / 1000, dyn_mem_mb, table_mem_gb))
    print("Total across 8 GPUs: {:.1f} GB saved.".format((table_mem_gb - dyn_mem_mb / 1000) * 8))
    print("Done.")


if __name__ == "__main__":
    main()
