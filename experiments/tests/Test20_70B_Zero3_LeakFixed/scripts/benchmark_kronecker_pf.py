#!/usr/bin/env python3
"""
Benchmark Kronecker PF_table embedding modes:
  1. gpu_table   — full precomputed table on GPU (current production)
  2. gpu_dynamic — compute PF on GPU each step from compact byte buffers

Usage (on AWS):
  cd /mnt/local-nvme/LLM/experiments/tests/Test19/code
  python3 ../scripts/benchmark_kronecker_pf.py

Single-GPU benchmark, no DeepSpeed required.
"""

import math
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

            # Look up byte sequences and lengths for this chunk
            bytes_chunk = self._token_bytes.index_select(0, ids_chunk).to(torch.long)  # [C, POS_DIM]
            lens_chunk = self._token_lens.index_select(0, ids_chunk).to(torch.long)    # [C]

            # Build sparse PF vectors via scatter_add
            # Each byte at position p contributes to index: byte_value * POS_DIM + p
            pf_chunk = torch.zeros((chunk_size, self.D), device=device, dtype=torch.float32)
            pos = self._pos_ids.unsqueeze(0).expand(chunk_size, -1)  # [C, POS_DIM]
            lin_idx = bytes_chunk * self.pos_dim + pos                # [C, POS_DIM]

            # Mask: only positions < token length are valid
            valid = pos < lens_chunk.unsqueeze(1)  # [C, POS_DIM]

            # Length normalization: 1/sqrt(L)
            if self.length_normalize:
                scales = torch.rsqrt(lens_chunk.clamp_min(1).to(torch.float32))  # [C]
                src = valid.to(torch.float32) * scales.unsqueeze(1)  # [C, POS_DIM]
            else:
                src = valid.to(torch.float32)

            pf_chunk.scatter_add_(dim=1, index=lin_idx, src=src)

            # Normalize: zero mean, unit std
            pf_centered = pf_chunk - pf_chunk.mean(dim=-1, keepdim=True)
            pf_std = pf_centered.std(dim=-1, keepdim=True) + 1e-6
            out[start:end] = (pf_centered / pf_std).to(torch.bfloat16)

        return out.view(*token_ids.shape, self.D)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def benchmark(fn, warmup=10, iters=50, label=""):
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
    mean = sum(times) / len(times)
    return median, mean


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # --- Build vocabulary (same as main.py) ---
    print("Loading tokenizer...")
    tokenizer_dir = os.path.join(CODE_DIR, "src", "tokenizer")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    vocab_size = 131072

    bpe_vocab = []
    for i in range(vocab_size):
        try:
            token = tokenizer.decode([i])
            bpe_vocab.append(token if token else f"<unk_{i}>")
        except Exception:
            bpe_vocab.append(f"<unk_{i}>")

    # --- Build PF codec ---
    pf_config = KroneckerConfig(CHAR_DIM=256, POS_DIM=32, D=8192,
                                 length_normalize=True, truncate_long_words=True)
    pf_codec = KroneckerEmbeddings(pf_config)

    # --- Build gpu_table embedding ---
    print("Building gpu_table (full PF_table)...")
    t0 = time.perf_counter()
    PF_np = pf_codec.encode_batch(bpe_vocab)
    if PF_np.dtype != np.float32:
        PF_np = PF_np.astype(np.float32, copy=False)
    pf_tensor = torch.from_numpy(PF_np)
    pf_centered = pf_tensor - pf_tensor.mean(dim=-1, keepdim=True)
    pf_std = pf_centered.std(dim=-1, keepdim=True) + 1e-6
    pf_norm = (pf_centered / pf_std).to(torch.bfloat16)
    embed_table = EmbedGPUTable(pf_norm).to(device)
    init_table_s = time.perf_counter() - t0
    del PF_np, pf_tensor, pf_centered, pf_std, pf_norm

    # --- Build gpu_dynamic embeddings (chunk=2048 and chunk=1024) ---
    print("Building gpu_dynamic...")
    t0 = time.perf_counter()
    embed_dyn_2048 = EmbedGPUDynamic(bpe_vocab, pf_codec, chunk_tokens=2048).to(device)
    init_dyn_s = time.perf_counter() - t0

    embed_dyn_1024 = EmbedGPUDynamic(bpe_vocab, pf_codec, chunk_tokens=1024).to(device)

    # --- Memory ---
    table_mem_gb = embed_table.PF_table.numel() * 2 / 1e9  # bf16
    dyn_bytes_mem = embed_dyn_2048._token_bytes.numel() * 1 / 1e9
    dyn_lens_mem = embed_dyn_2048._token_lens.numel() * 2 / 1e9
    dyn_mem_gb = dyn_bytes_mem + dyn_lens_mem

    print(f"\nMemory:")
    print(f"  gpu_table:   {table_mem_gb:.2f} GB  ({embed_table.PF_table.shape})")
    print(f"  gpu_dynamic: {dyn_mem_gb*1000:.1f} MB  (bytes={embed_dyn_2048._token_bytes.shape}, lens={embed_dyn_2048._token_lens.shape})")
    print(f"  Savings:     {table_mem_gb - dyn_mem_gb:.2f} GB ({(1 - dyn_mem_gb/table_mem_gb)*100:.0f}%)")

    # --- Benchmark configs ---
    configs = [
        (1, 4096, "B=1, T=4096 (8B micro-batch)"),
        (2, 4096, "B=2, T=4096 (3B micro-batch)"),
        (3, 4096, "B=3, T=4096 (from screenshot)"),
        (8, 4096, "B=8, T=4096 (global batch)"),
    ]

    for B, T, desc in configs:
        token_ids = torch.randint(0, vocab_size, (B, T), device=device)
        total_tokens = B * T

        print(f"\n{'='*70}")
        print(f"  {desc}  ({total_tokens:,} tokens)")
        print(f"{'='*70}")

        # Correctness check
        out_table = embed_table(token_ids)
        out_dyn = embed_dyn_2048(token_ids)
        max_diff = (out_table.float() - out_dyn.float()).abs().max().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            out_table.float().reshape(-1, 8192),
            out_dyn.float().reshape(-1, 8192),
            dim=-1
        ).mean().item()
        print(f"  Correctness: max_diff={max_diff:.4f}, cos_sim={cos_sim:.6f}")

        # gpu_table
        med, avg = benchmark(lambda: embed_table(token_ids), warmup=10, iters=50)
        tps_table = total_tokens / (med / 1000)
        print(f"  gpu_table:          {med:6.2f} ms median | {tps_table/1e6:.2f}M tok/s")

        # gpu_dynamic chunk=2048
        med, avg = benchmark(lambda: embed_dyn_2048(token_ids), warmup=10, iters=50)
        tps_dyn = total_tokens / (med / 1000)
        print(f"  gpu_dynamic(2048):  {med:6.2f} ms median | {tps_dyn/1e6:.2f}M tok/s")

        # gpu_dynamic chunk=1024
        med, avg = benchmark(lambda: embed_dyn_1024(token_ids), warmup=10, iters=50)
        tps_dyn2 = total_tokens / (med / 1000)
        print(f"  gpu_dynamic(1024):  {med:6.2f} ms median | {tps_dyn2/1e6:.2f}M tok/s")

    # --- Peak GPU memory ---
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Measure gpu_table peak
    torch.cuda.reset_peak_memory_stats()
    token_ids = torch.randint(0, vocab_size, (3, 4096), device=device)
    _ = embed_table(token_ids)
    peak_table = torch.cuda.max_memory_allocated() / 1e9

    # Clear and measure gpu_dynamic peak
    del embed_table
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = embed_dyn_2048(token_ids)
    peak_dyn = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n{'='*70}")
    print(f"PEAK GPU MEMORY (B=3, T=4096)")
    print(f"{'='*70}")
    print(f"  gpu_table:   {peak_table:.2f} GB")
    print(f"  gpu_dynamic: {peak_dyn:.2f} GB")
    print(f"  Savings:     {peak_table - peak_dyn:.2f} GB")

    print(f"\n{'='*70}")
    print(f"INIT TIME")
    print(f"{'='*70}")
    print(f"  gpu_table:   {init_table_s:.2f} s (encode_batch for {vocab_size:,} tokens)")
    print(f"  gpu_dynamic: {init_dyn_s:.2f} s (build byte tables)")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
