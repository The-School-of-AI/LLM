"""
Prototype: 4-bit NF4 LoRA training step for MoE model.

Demonstrates memory footprint of NF4 base weights + bf16 LoRA on a proxy
MoE architecture matching our 70B model's expert layout.

Usage:
    python scripts/prototype_4bit_lora.py [--num-experts 260] [--hidden 4096]

Requirements:
    pip install bitsandbytes>=0.43.0 torch>=2.1

Notes:
    - Uses a PROXY model (not the full 70B) to validate the quantization
      approach and measure memory savings.
    - The proxy matches our MoE expert weight layout: [E, K, N] parameters
      with gate/up/down projections and shared expert FFN.
    - Scale memory estimates by (full_params / proxy_params) for 70B.
"""

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# NF4 Quantization Utilities
# ---------------------------------------------------------------------------

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    print("[WARN] bitsandbytes not installed. Using simulated quantization.")


@dataclass
class NF4Config:
    bits: int = 4
    quant_type: str = "nf4"
    double_quant: bool = True
    block_size: int = 64
    compute_dtype: torch.dtype = torch.bfloat16


def quantize_parameter_nf4(param: torch.Tensor, config: NF4Config):
    """Quantize a parameter tensor to NF4 using bitsandbytes."""
    if not HAS_BNB:
        # Simulated: just store in fp16 (half the bf16 size as rough proxy)
        return param.half(), None

    flat = param.contiguous().view(-1).to(config.compute_dtype)
    quantized, quant_state = bnb.functional.quantize_4bit(
        flat,
        quant_type=config.quant_type,
        compress_statistics=config.double_quant,
        blocksize=config.block_size,
    )
    return quantized, quant_state


def dequantize_parameter_nf4(quantized, quant_state, original_shape, config: NF4Config):
    """Dequantize NF4 parameter back to compute dtype."""
    if not HAS_BNB:
        return quantized.to(config.compute_dtype).view(original_shape)

    deq = bnb.functional.dequantize_4bit(
        quantized, quant_state, quant_type=config.quant_type,
    )
    return deq.view(original_shape).to(config.compute_dtype)


# ---------------------------------------------------------------------------
# Proxy MoE Layer (matches 70B expert layout)
# ---------------------------------------------------------------------------

class ProxyMoEExperts(nn.Module):
    """
    Proxy MoE expert layer with optional NF4 quantization.
    Matches our 70B model's expert weight layout: [E, K, N] parameters.
    """

    def __init__(self, num_experts, d_model, d_hidden, quantize=False, config=None):
        super().__init__()
        self.E = num_experts
        self.K = d_model
        self.N = d_hidden
        self.quantize = quantize
        self.nf4_config = config or NF4Config()

        # Initialize bf16 weights
        W_gate = torch.randn(num_experts, d_model, d_hidden, dtype=torch.bfloat16) * 0.02
        W_up = torch.randn(num_experts, d_model, d_hidden, dtype=torch.bfloat16) * 0.02
        W_down = torch.randn(num_experts, d_hidden, d_model, dtype=torch.bfloat16) * 0.02

        if quantize:
            # Store as NF4
            self._W_gate_q, self._W_gate_qs = quantize_parameter_nf4(W_gate, self.nf4_config)
            self._W_up_q, self._W_up_qs = quantize_parameter_nf4(W_up, self.nf4_config)
            self._W_down_q, self._W_down_qs = quantize_parameter_nf4(W_down, self.nf4_config)
            self._W_gate_shape = W_gate.shape
            self._W_up_shape = W_up.shape
            self._W_down_shape = W_down.shape
            # Register as buffers (not parameters — frozen)
            self.register_buffer("W_gate_q", self._W_gate_q)
            self.register_buffer("W_up_q", self._W_up_q)
            self.register_buffer("W_down_q", self._W_down_q)
        else:
            self.W_gate = nn.Parameter(W_gate, requires_grad=False)
            self.W_up = nn.Parameter(W_up, requires_grad=False)
            self.W_down = nn.Parameter(W_down, requires_grad=False)

    def get_expert_weights(self):
        """Return dequantized expert weights."""
        if self.quantize:
            W_g = dequantize_parameter_nf4(
                self._W_gate_q, self._W_gate_qs, self._W_gate_shape, self.nf4_config)
            W_u = dequantize_parameter_nf4(
                self._W_up_q, self._W_up_qs, self._W_up_shape, self.nf4_config)
            W_d = dequantize_parameter_nf4(
                self._W_down_q, self._W_down_qs, self._W_down_shape, self.nf4_config)
            return W_g, W_u, W_d
        return self.W_gate, self.W_up, self.W_down


class ProxyLoRAMoELayer(nn.Module):
    """
    Proxy MoE layer with LoRA adaptation on expert weights.
    Demonstrates the full forward+backward memory footprint.
    """

    def __init__(self, num_experts, d_model, d_hidden, d_shared,
                 top_k=8, lora_rank=16, quantize=False):
        super().__init__()
        self.E = num_experts
        self.K = d_model
        self.N = d_hidden
        self.top_k = top_k
        self.lora_rank = lora_rank
        self.scaling = 32.0 / lora_rank  # alpha=32

        # Expert weights (quantized or bf16)
        self.experts = ProxyMoEExperts(num_experts, d_model, d_hidden, quantize=quantize)

        # Shared expert (always bf16 — small)
        self.shared_gate = nn.Linear(d_model, d_shared, bias=False)
        self.shared_up = nn.Linear(d_model, d_shared, bias=False)
        self.shared_down = nn.Linear(d_shared, d_model, bias=False)

        # Router
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # LoRA on expert weights (bf16, trainable)
        dtype = torch.bfloat16
        self.lora_A_gate = nn.Parameter(torch.empty(num_experts, lora_rank, d_model, dtype=dtype))
        self.lora_B_gate = nn.Parameter(torch.zeros(num_experts, d_hidden, lora_rank, dtype=dtype))
        self.lora_A_up = nn.Parameter(torch.empty(num_experts, lora_rank, d_model, dtype=dtype))
        self.lora_B_up = nn.Parameter(torch.zeros(num_experts, d_hidden, lora_rank, dtype=dtype))
        self.lora_A_down = nn.Parameter(torch.empty(num_experts, lora_rank, d_hidden, dtype=dtype))
        self.lora_B_down = nn.Parameter(torch.zeros(num_experts, d_model, lora_rank, dtype=dtype))

        # Init LoRA A with Kaiming
        for p in [self.lora_A_gate, self.lora_A_up, self.lora_A_down]:
            nn.init.kaiming_uniform_(p.view(-1, p.shape[-1]), a=math.sqrt(5))

        # Freeze non-LoRA
        for name, param in self.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

    def forward(self, x):
        B, T, D = x.shape
        N = B * T

        # Shared expert
        shared_h = F.silu(self.shared_gate(x)) * self.shared_up(x)
        shared_out = self.shared_down(shared_h)

        # Router (simplified — no null routing for prototype)
        logits = self.gate(x)
        topk_weight, topk_idx = torch.topk(F.softmax(logits, dim=-1), self.top_k, dim=-1)
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

        # Flatten and sort by expert
        flat_x = x.reshape(N, D)
        flat_idx = topk_idx.reshape(-1)
        flat_weight = topk_weight.reshape(-1)
        token_idx = torch.arange(N, device=x.device).repeat_interleave(self.top_k)

        sort_idx = flat_idx.argsort()
        sorted_x = flat_x[token_idx[sort_idx]]
        sorted_experts = flat_idx[sort_idx]
        sorted_weights = flat_weight[sort_idx]
        expert_counts = torch.bincount(sorted_experts, minlength=self.E)

        # Expert compute with LoRA (per-expert loop for clarity)
        W_gate, W_up, W_down = self.experts.get_expert_weights()
        M = sorted_x.shape[0]
        out = torch.zeros(M, D, device=x.device, dtype=x.dtype)
        offset = 0
        for e in range(self.E):
            cnt = expert_counts[e].item()
            if cnt == 0:
                continue
            xe = sorted_x[offset:offset + cnt]

            # Base + LoRA for gate
            g = xe @ W_gate[e].T + (xe @ self.lora_A_gate[e].T @ self.lora_B_gate[e].T) * self.scaling
            u = xe @ W_up[e].T + (xe @ self.lora_A_up[e].T @ self.lora_B_up[e].T) * self.scaling
            h = F.silu(g) * u
            d = h @ W_down[e].T + (h @ self.lora_A_down[e].T @ self.lora_B_down[e].T) * self.scaling
            out[offset:offset + cnt] = d
            offset += cnt

        # Scatter back
        weighted_out = out * sorted_weights.unsqueeze(-1)
        routed_out = torch.zeros(N, D, device=x.device, dtype=x.dtype)
        routed_out.index_add_(0, token_idx[sort_idx], weighted_out)

        return shared_out + routed_out.view(B, T, D)


# ---------------------------------------------------------------------------
# Memory Measurement
# ---------------------------------------------------------------------------

def print_memory(label=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  [{label:30s}] alloc={alloc:6.2f} GB  reserved={reserved:6.2f} GB  peak={peak:6.2f} GB")
    else:
        print(f"  [{label}] CUDA not available")


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())
    return total, trainable, buffers


def run_experiment(num_experts, d_model, d_hidden, d_shared, top_k,
                   lora_rank, seq_len, batch_size, quantize):
    tag = "NF4" if quantize else "bf16"
    print(f"\n{'='*70}")
    print(f"  MoE LoRA Experiment: {tag}")
    print(f"  experts={num_experts}, d_model={d_model}, d_hidden={d_hidden}")
    print(f"  top_k={top_k}, lora_rank={lora_rank}, BS={batch_size}, SL={seq_len}")
    print(f"{'='*70}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    print_memory("clean start")

    model = ProxyLoRAMoELayer(
        num_experts=num_experts,
        d_model=d_model,
        d_hidden=d_hidden,
        d_shared=d_shared,
        top_k=top_k,
        lora_rank=lora_rank,
        quantize=quantize,
    ).cuda().bfloat16()

    total, trainable, buffers = count_params(model)
    print(f"\n  Parameters: total={total:,} trainable={trainable:,} buffers={buffers:,}")
    print(f"  Trainable: {trainable * 2 / 1e6:.1f} MB (bf16)")
    print_memory("after model load")

    # Forward pass
    x = torch.randn(batch_size, seq_len, d_model, device="cuda", dtype=torch.bfloat16)
    print_memory("after input alloc")

    out = model(x)
    loss = out.sum()
    print_memory("after forward")

    # Backward pass
    loss.backward()
    print_memory("after backward")

    # Check gradients exist for LoRA params
    lora_grads = {n: p.grad is not None for n, p in model.named_parameters() if "lora_" in n}
    has_all_grads = all(lora_grads.values())
    print(f"\n  LoRA gradients computed: {has_all_grads} ({sum(lora_grads.values())}/{len(lora_grads)})")

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n  PEAK VRAM: {peak:.2f} GB")

    del model, x, out, loss
    torch.cuda.empty_cache()
    return peak


def main():
    parser = argparse.ArgumentParser(description="4-bit LoRA MoE memory prototype")
    parser.add_argument("--num-experts", type=int, default=20,
                        help="Number of experts (default: 20 for proxy; 260 for full)")
    parser.add_argument("--hidden", type=int, default=4096, help="Model hidden size")
    parser.add_argument("--expert-hidden", type=int, default=1024, help="Expert FFN width")
    parser.add_argument("--shared-hidden", type=int, default=2048, help="Shared expert width")
    parser.add_argument("--top-k", type=int, default=8, help="Top-k routing")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required for memory measurement.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"\nGPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"bitsandbytes: {'available' if HAS_BNB else 'NOT available (simulated)'}")

    # Run bf16 baseline
    peak_bf16 = run_experiment(
        args.num_experts, args.hidden, args.expert_hidden, args.shared_hidden,
        args.top_k, args.lora_rank, args.seq_len, args.batch_size,
        quantize=False,
    )

    # Run NF4 quantized
    peak_nf4 = run_experiment(
        args.num_experts, args.hidden, args.expert_hidden, args.shared_hidden,
        args.top_k, args.lora_rank, args.seq_len, args.batch_size,
        quantize=True,
    )

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  bf16 peak VRAM:  {peak_bf16:.2f} GB")
    print(f"  NF4  peak VRAM:  {peak_nf4:.2f} GB")
    print(f"  Savings:         {peak_bf16 - peak_nf4:.2f} GB ({(1 - peak_nf4/peak_bf16)*100:.1f}%)")

    # Extrapolate to full 70B model
    scale = 260 / args.num_experts  # Expert count scaling
    print(f"\n  Extrapolation to 260 experts (scale={scale:.1f}x):")
    print(f"    bf16 est: {peak_bf16 * scale:.1f} GB (expert weights only)")
    print(f"    NF4  est: {peak_nf4 * scale:.1f} GB (expert weights only)")
    print(f"    Note: Full 70B includes attention, mHC, embeddings — add ~20 GB")


if __name__ == "__main__":
    main()
