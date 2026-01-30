#!/usr/bin/env python3
"""
MoE FLOP Estimator
==================
Comprehensive FLOPs calculation for MoE architectures with:
- Per-component breakdown (attention, FFN, router, experts)
- Forward and backward pass estimation
- FLOPs/token and FLOPs/second calculations
- GPU utilization estimation (MFU)

Usage:
    from estimators.flops_estimator import FLOPEstimator
    
    estimator = FLOPEstimator(config)
    report = estimator.full_report()
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import math


@dataclass
class ModelConfig:
    """Model configuration for FLOP estimation."""
    
    # Architecture
    hidden_size: int = 4096
    num_layers: int = 40
    num_attention_heads: int = 32
    num_kv_heads: int = 8  # GQA
    intermediate_size: int = 2048
    vocab_size: int = 32000
    max_seq_length: int = 4096
    
    # MoE Configuration
    moe_enabled: bool = True
    num_routed_experts: int = 64
    num_shared_experts: int = 4
    num_null_experts: int = 2
    top_k: int = 4
    moe_layer_frequency: int = 1  # 1 = every layer
    
    # Router
    router_num_heads: int = 4
    router_head_dim: int = 128
    
    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    # Hardware (for utilization estimates)
    gpu_flops_bf16: float = 312e12  # A100 bf16 TFLOPS
    num_gpus: int = 32


@dataclass
class FLOPBreakdown:
    """Detailed FLOP breakdown by component."""
    
    # Per-token FLOPs
    embedding_forward: int = 0
    embedding_backward: int = 0
    
    attention_qkv_proj: int = 0
    attention_scores: int = 0
    attention_softmax: int = 0
    attention_weighted_sum: int = 0
    attention_output_proj: int = 0
    attention_total: int = 0
    
    router_forward: int = 0
    router_topk: int = 0
    router_total: int = 0
    
    expert_shared: int = 0
    expert_routed: int = 0
    expert_null: int = 0  # Near-zero
    expert_gating: int = 0  # G1 + G2
    expert_combine: int = 0
    expert_total: int = 0
    
    layer_norm: int = 0
    
    # Totals
    forward_per_token: int = 0
    backward_per_token: int = 0
    total_per_token: int = 0
    
    # Per-batch
    forward_per_batch: int = 0
    backward_per_batch: int = 0
    total_per_batch: int = 0
    
    # Training step
    forward_per_step: int = 0
    backward_per_step: int = 0
    total_per_step: int = 0


class FLOPEstimator:
    """
    Comprehensive FLOP estimator for MoE models.
    
    Calculation methodology:
    - Matrix multiply: 2 * M * N * K FLOPs
    - Attention scores: 2 * batch * heads * seq * seq
    - Softmax: ~5 * elements (exp, sum, div)
    - LayerNorm: ~5 * elements
    - Backward pass: ~2x forward pass
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._breakdown = None
    
    def _matmul_flops(self, m: int, n: int, k: int) -> int:
        """FLOPs for matrix multiplication [M, K] @ [K, N] = [M, N]."""
        return 2 * m * n * k
    
    def _attention_flops_per_layer(self) -> Dict[str, int]:
        """Calculate attention FLOPs for one layer."""
        cfg = self.config
        B, S, H = cfg.batch_size, cfg.max_seq_length, cfg.hidden_size
        num_heads = cfg.num_attention_heads
        num_kv_heads = cfg.num_kv_heads
        head_dim = H // num_heads
        
        # Q projection: [B*S, H] @ [H, H] = [B*S, H]
        q_proj = self._matmul_flops(B * S, H, H)
        
        # K, V projections: [B*S, H] @ [H, kv_dim]
        kv_dim = num_kv_heads * head_dim
        k_proj = self._matmul_flops(B * S, kv_dim, H)
        v_proj = self._matmul_flops(B * S, kv_dim, H)
        
        # Attention scores: [B, heads, S, head_dim] @ [B, heads, head_dim, S]
        # With GQA, we repeat KV heads
        attn_scores = self._matmul_flops(B * num_heads * S, S, head_dim)
        
        # Softmax: ~5 ops per element
        softmax = 5 * B * num_heads * S * S
        
        # Weighted sum: [B, heads, S, S] @ [B, heads, S, head_dim]
        weighted_sum = self._matmul_flops(B * num_heads * S, head_dim, S)
        
        # Output projection: [B*S, H] @ [H, H]
        out_proj = self._matmul_flops(B * S, H, H)
        
        return {
            'qkv_proj': q_proj + k_proj + v_proj,
            'scores': attn_scores,
            'softmax': softmax,
            'weighted_sum': weighted_sum,
            'output_proj': out_proj,
            'total': q_proj + k_proj + v_proj + attn_scores + softmax + weighted_sum + out_proj
        }
    
    def _router_flops_per_layer(self) -> Dict[str, int]:
        """Calculate router FLOPs for one MoE layer."""
        cfg = self.config
        B, S, H = cfg.batch_size, cfg.max_seq_length, cfg.hidden_size
        num_experts = cfg.num_routed_experts + cfg.num_null_experts
        
        # GSA Router: multi-head scoring
        # Query projection: [B*S, H] @ [H, num_heads * head_dim]
        query_dim = cfg.router_num_heads * cfg.router_head_dim
        query_proj = self._matmul_flops(B * S, query_dim, H)
        
        # Head weight projection: [B*S, H] @ [H, num_heads]
        head_weight = self._matmul_flops(B * S, cfg.router_num_heads, H)
        
        # Expert affinity: [B*S, num_heads, head_dim] @ [num_experts, num_heads, head_dim]
        # Simplified: B*S * num_heads * head_dim * num_experts * 2
        affinity = 2 * B * S * cfg.router_num_heads * cfg.router_head_dim * num_experts
        
        # Sigmoid: ~4 ops per element
        sigmoid = 4 * B * S * num_experts
        
        # Top-k selection: ~k * log(num_experts) comparisons
        topk = int(cfg.top_k * math.log2(num_experts) * B * S)
        
        return {
            'forward': query_proj + head_weight + affinity + sigmoid,
            'topk': topk,
            'total': query_proj + head_weight + affinity + sigmoid + topk
        }
    
    def _expert_flops_per_layer(self) -> Dict[str, int]:
        """Calculate expert FLOPs for one MoE layer."""
        cfg = self.config
        B, S, H = cfg.batch_size, cfg.max_seq_length, cfg.hidden_size
        I = cfg.intermediate_size
        
        # SwiGLU FFN: 3 projections
        # W1 (gate): [B*S, H] @ [H, I]
        # W3 (up): [B*S, H] @ [H, I]
        # W2 (down): [B*S, I] @ [I, H]
        # Plus SiLU activation: ~4 ops per element
        
        single_expert = (
            self._matmul_flops(B * S, I, H) +  # W1
            self._matmul_flops(B * S, I, H) +  # W3
            4 * B * S * I +                     # SiLU
            B * S * I +                         # element-wise multiply
            self._matmul_flops(B * S, H, I)    # W2
        )
        
        # Shared experts: always active
        shared_flops = single_expert * cfg.num_shared_experts
        
        # Routed experts: only top-k active
        routed_flops = single_expert * cfg.top_k
        
        # Null experts: near-zero compute (just scaling)
        # Assume ~10% of tokens route to null, saving compute
        null_rate = 0.10
        null_flops = int(B * S * H * null_rate)  # Just x * scale
        
        # Dual gating (G1 + G2): 2 small projections per expert
        gating_dim = int(H * 0.25)  # 25% of hidden
        gating_flops = (
            2 * self._matmul_flops(B * S, gating_dim, H) +  # G1, G2 projections
            2 * 4 * B * S * gating_dim +                    # Sigmoids
            2 * B * S * H                                   # Element-wise multiply
        ) * (cfg.num_shared_experts + cfg.top_k)
        
        # Combine weighted outputs: top_k + shared multiplications and adds
        combine_flops = (cfg.top_k + cfg.num_shared_experts) * B * S * H * 2
        
        return {
            'shared': shared_flops,
            'routed': routed_flops,
            'null': null_flops,
            'gating': gating_flops,
            'combine': combine_flops,
            'total': shared_flops + routed_flops + null_flops + gating_flops + combine_flops
        }
    
    def _layernorm_flops(self) -> int:
        """FLOPs for one LayerNorm."""
        cfg = self.config
        B, S, H = cfg.batch_size, cfg.max_seq_length, cfg.hidden_size
        # ~5 ops: mean, variance, normalize, scale, shift
        return 5 * B * S * H
    
    def _embedding_flops(self) -> Dict[str, int]:
        """FLOPs for embedding layer."""
        cfg = self.config
        B, S, H, V = cfg.batch_size, cfg.max_seq_length, cfg.hidden_size, cfg.vocab_size
        
        # Forward: lookup (negligible) + output projection
        # LM head: [B*S, H] @ [H, V]
        lm_head = self._matmul_flops(B * S, V, H)
        
        return {
            'forward': lm_head,
            'backward': lm_head * 2  # Gradients for both weight and input
        }
    
    def calculate_breakdown(self) -> FLOPBreakdown:
        """Calculate detailed FLOP breakdown."""
        cfg = self.config
        breakdown = FLOPBreakdown()
        
        # Embeddings
        embed = self._embedding_flops()
        breakdown.embedding_forward = embed['forward']
        breakdown.embedding_backward = embed['backward']
        
        # Per-layer components
        attn = self._attention_flops_per_layer()
        breakdown.attention_qkv_proj = attn['qkv_proj']
        breakdown.attention_scores = attn['scores']
        breakdown.attention_softmax = attn['softmax']
        breakdown.attention_weighted_sum = attn['weighted_sum']
        breakdown.attention_output_proj = attn['output_proj']
        breakdown.attention_total = attn['total']
        
        # MoE layers
        num_moe_layers = cfg.num_layers // cfg.moe_layer_frequency
        
        if cfg.moe_enabled:
            router = self._router_flops_per_layer()
            breakdown.router_forward = router['forward']
            breakdown.router_topk = router['topk']
            breakdown.router_total = router['total']
            
            expert = self._expert_flops_per_layer()
            breakdown.expert_shared = expert['shared']
            breakdown.expert_routed = expert['routed']
            breakdown.expert_null = expert['null']
            breakdown.expert_gating = expert['gating']
            breakdown.expert_combine = expert['combine']
            breakdown.expert_total = expert['total']
        else:
            # Dense FFN
            B, S, H, I = cfg.batch_size, cfg.max_seq_length, cfg.hidden_size, cfg.intermediate_size
            dense_ffn = (
                self._matmul_flops(B * S, I, H) * 2 +  # W1, W3
                self._matmul_flops(B * S, H, I)       # W2
            )
            breakdown.expert_total = dense_ffn
        
        # LayerNorm (2 per layer: pre-attn, pre-ffn)
        breakdown.layer_norm = 2 * self._layernorm_flops()
        
        # Total forward per layer
        layer_forward = (
            breakdown.attention_total +
            breakdown.router_total +
            breakdown.expert_total +
            breakdown.layer_norm
        )
        
        # Total forward
        breakdown.forward_per_batch = (
            breakdown.embedding_forward +
            layer_forward * cfg.num_layers
        )
        
        # Backward is ~2x forward
        breakdown.backward_per_batch = breakdown.forward_per_batch * 2
        
        # Total per batch
        breakdown.total_per_batch = breakdown.forward_per_batch + breakdown.backward_per_batch
        
        # Per token
        tokens_per_batch = cfg.batch_size * cfg.max_seq_length
        breakdown.forward_per_token = breakdown.forward_per_batch // tokens_per_batch
        breakdown.backward_per_token = breakdown.backward_per_batch // tokens_per_batch
        breakdown.total_per_token = breakdown.total_per_batch // tokens_per_batch
        
        # Per training step (with gradient accumulation)
        breakdown.forward_per_step = breakdown.forward_per_batch * cfg.gradient_accumulation_steps
        breakdown.backward_per_step = breakdown.backward_per_batch * cfg.gradient_accumulation_steps
        breakdown.total_per_step = breakdown.total_per_batch * cfg.gradient_accumulation_steps
        
        self._breakdown = breakdown
        return breakdown
    
    def estimate_throughput(self) -> Dict[str, float]:
        """Estimate training throughput and GPU utilization."""
        if self._breakdown is None:
            self.calculate_breakdown()
        
        cfg = self.config
        bd = self._breakdown
        
        # Total GPU compute capacity
        total_gpu_flops = cfg.gpu_flops_bf16 * cfg.num_gpus
        
        # Theoretical max throughput (tokens/second)
        flops_per_token = bd.total_per_token
        theoretical_tokens_per_sec = total_gpu_flops / flops_per_token
        
        # Realistic throughput (accounting for memory bandwidth, communication)
        # Typical MFU for MoE: 30-45%
        mfu_estimate = 0.35
        realistic_tokens_per_sec = theoretical_tokens_per_sec * mfu_estimate
        
        # Steps per second
        tokens_per_step = cfg.batch_size * cfg.max_seq_length * cfg.gradient_accumulation_steps
        steps_per_sec = realistic_tokens_per_sec / tokens_per_step
        
        # Time estimates
        tokens_per_day = realistic_tokens_per_sec * 86400
        
        return {
            'flops_per_token': flops_per_token,
            'total_gpu_tflops': total_gpu_flops / 1e12,
            'theoretical_tokens_per_sec': theoretical_tokens_per_sec,
            'estimated_mfu': mfu_estimate,
            'realistic_tokens_per_sec': realistic_tokens_per_sec,
            'steps_per_sec': steps_per_sec,
            'tokens_per_day': tokens_per_day,
            'tokens_per_day_billions': tokens_per_day / 1e9,
        }
    
    def full_report(self) -> Dict:
        """Generate comprehensive FLOP report."""
        breakdown = self.calculate_breakdown()
        throughput = self.estimate_throughput()
        
        return {
            'config': {
                'model': f"{self.config.num_layers}L-{self.config.hidden_size}H",
                'moe': f"{self.config.num_routed_experts}E-top{self.config.top_k}" if self.config.moe_enabled else "Dense",
                'batch_size': self.config.batch_size,
                'seq_length': self.config.max_seq_length,
                'num_gpus': self.config.num_gpus,
            },
            'flops_breakdown': {
                'attention': {
                    'qkv_projection': f"{breakdown.attention_qkv_proj / 1e9:.2f}B",
                    'scores': f"{breakdown.attention_scores / 1e9:.2f}B",
                    'softmax': f"{breakdown.attention_softmax / 1e9:.2f}B",
                    'weighted_sum': f"{breakdown.attention_weighted_sum / 1e9:.2f}B",
                    'output_projection': f"{breakdown.attention_output_proj / 1e9:.2f}B",
                    'total_per_layer': f"{breakdown.attention_total / 1e9:.2f}B",
                },
                'router': {
                    'forward': f"{breakdown.router_forward / 1e9:.2f}B",
                    'topk_selection': f"{breakdown.router_topk / 1e6:.2f}M",
                    'total_per_layer': f"{breakdown.router_total / 1e9:.2f}B",
                },
                'experts': {
                    'shared': f"{breakdown.expert_shared / 1e9:.2f}B",
                    'routed_topk': f"{breakdown.expert_routed / 1e9:.2f}B",
                    'null': f"{breakdown.expert_null / 1e6:.2f}M",
                    'gating': f"{breakdown.expert_gating / 1e9:.2f}B",
                    'combine': f"{breakdown.expert_combine / 1e9:.2f}B",
                    'total_per_layer': f"{breakdown.expert_total / 1e9:.2f}B",
                },
                'other': {
                    'layer_norm_per_layer': f"{breakdown.layer_norm / 1e9:.2f}B",
                    'embedding': f"{breakdown.embedding_forward / 1e9:.2f}B",
                },
            },
            'totals': {
                'forward_per_token': f"{breakdown.forward_per_token / 1e9:.2f}B FLOPs",
                'backward_per_token': f"{breakdown.backward_per_token / 1e9:.2f}B FLOPs",
                'total_per_token': f"{breakdown.total_per_token / 1e9:.2f}B FLOPs",
                'forward_per_batch': f"{breakdown.forward_per_batch / 1e12:.2f}T FLOPs",
                'total_per_step': f"{breakdown.total_per_step / 1e12:.2f}T FLOPs",
            },
            'throughput': {
                'gpu_capacity': f"{throughput['total_gpu_tflops']:.1f} TFLOPS",
                'theoretical_tokens_per_sec': f"{throughput['theoretical_tokens_per_sec']:.0f}",
                'estimated_mfu': f"{throughput['estimated_mfu']*100:.0f}%",
                'realistic_tokens_per_sec': f"{throughput['realistic_tokens_per_sec']:.0f}",
                'steps_per_sec': f"{throughput['steps_per_sec']:.3f}",
                'tokens_per_day': f"{throughput['tokens_per_day_billions']:.2f}B",
            },
        }
    
    def print_report(self):
        """Print formatted FLOP report."""
        report = self.full_report()
        
        print("=" * 60)
        print("MoE FLOP ESTIMATION REPORT")
        print("=" * 60)
        
        print("\n📊 Configuration:")
        for k, v in report['config'].items():
            print(f"  {k}: {v}")
        
        print("\n🔢 FLOPs Breakdown (per layer, per batch):")
        
        print("\n  Attention:")
        for k, v in report['flops_breakdown']['attention'].items():
            print(f"    {k}: {v}")
        
        print("\n  Router:")
        for k, v in report['flops_breakdown']['router'].items():
            print(f"    {k}: {v}")
        
        print("\n  Experts:")
        for k, v in report['flops_breakdown']['experts'].items():
            print(f"    {k}: {v}")
        
        print("\n📈 Totals:")
        for k, v in report['totals'].items():
            print(f"  {k}: {v}")
        
        print("\n⚡ Throughput Estimates:")
        for k, v in report['throughput'].items():
            print(f"  {k}: {v}")
        
        print("=" * 60)


# Preset configurations
CONFIGS = {
    '3b_moe': ModelConfig(
        hidden_size=2048,
        num_layers=24,
        num_attention_heads=16,
        num_kv_heads=4,
        intermediate_size=2048,
        num_routed_experts=8,
        num_shared_experts=2,
        num_null_experts=1,
        top_k=2,
        num_gpus=1,
    ),
    '8b_moe': ModelConfig(
        hidden_size=4096,
        num_layers=40,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=2048,
        num_routed_experts=8,
        num_shared_experts=2,
        num_null_experts=1,
        top_k=2,
        num_gpus=4,
    ),
    '70b_moe': ModelConfig(
        hidden_size=4096,
        num_layers=40,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=2048,
        num_routed_experts=64,
        num_shared_experts=4,
        num_null_experts=2,
        top_k=4,
        num_gpus=32,
    ),
}


if __name__ == "__main__":
    import sys
    
    model = sys.argv[1] if len(sys.argv) > 1 else '70b_moe'
    
    if model in CONFIGS:
        config = CONFIGS[model]
    else:
        print(f"Unknown model: {model}")
        print(f"Available: {list(CONFIGS.keys())}")
        sys.exit(1)
    
    estimator = FLOPEstimator(config)
    estimator.print_report()
