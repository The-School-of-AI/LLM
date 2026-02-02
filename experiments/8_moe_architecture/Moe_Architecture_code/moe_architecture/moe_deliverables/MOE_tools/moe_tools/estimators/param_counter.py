#!/usr/bin/env python3
"""
MoE Parameter Counter
=====================
Detailed parameter counting with:
- Per-component breakdown
- Active vs total parameters
- Parameter efficiency metrics

Usage:
    from estimators.param_counter import ParamCounter
    
    counter = ParamCounter(config)
    report = counter.full_report()
"""

from dataclasses import dataclass
from typing import Dict, Optional
import json


@dataclass
class ModelConfig:
    """Model configuration for parameter counting."""
    
    # Architecture
    hidden_size: int = 2048
    num_layers: int = 40
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    intermediate_size: int = 512
    vocab_size: int = 32000
    
    # MoE Configuration
    moe_enabled: bool = True
    num_routed_experts: int = 512
    num_shared_experts: int = 4
    num_null_experts: int = 256
    top_k: int = 8
    moe_layer_frequency: int = 1
    
    # Router
    router_num_heads: int = 4
    router_head_dim: int = 128
    use_dual_gating: bool = False


@dataclass
class ParamBreakdown:
    """Parameter breakdown by component."""
    
    # Embeddings
    input_embedding: int = 0
    output_embedding: int = 0  # Often tied with input
    embedding_total: int = 0
    
    # Per-layer components
    attention_q: int = 0
    attention_k: int = 0
    attention_v: int = 0
    attention_o: int = 0
    attention_total_per_layer: int = 0
    
    router_per_layer: int = 0
    
    expert_single: int = 0
    shared_experts_per_layer: int = 0
    routed_experts_per_layer: int = 0
    gating_per_layer: int = 0
    
    layernorm_per_layer: int = 0
    
    # Totals
    attention_total: int = 0
    router_total: int = 0
    shared_expert_total: int = 0
    routed_expert_total: int = 0
    gating_total: int = 0
    layernorm_total: int = 0
    
    # Grand totals
    total_params: int = 0
    active_params: int = 0
    
    # Efficiency
    active_ratio: float = 0.0


class ParamCounter:
    """
    Parameter counter for MoE models.
    
    Provides detailed breakdown of:
    - Total parameters (all experts)
    - Active parameters (shared + top-k routed)
    - Per-component analysis
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._breakdown = None
    
    def _count_embedding_params(self) -> Dict[str, int]:
        """Count embedding parameters."""
        cfg = self.config
        
        input_emb = cfg.vocab_size * cfg.hidden_size
        output_emb = input_emb  # Tied weights
        
        return {
            'input': input_emb,
            'output': output_emb,
            'total': input_emb,  # Tied, so count once
        }
    
    def _count_attention_params(self) -> Dict[str, int]:
        """Count attention parameters per layer."""
        cfg = self.config
        H = cfg.hidden_size
        num_kv_heads = cfg.num_kv_heads
        head_dim = H // cfg.num_attention_heads
        kv_dim = num_kv_heads * head_dim
        
        q_params = H * H  # [H, H]
        k_params = H * kv_dim  # [H, kv_dim]
        v_params = H * kv_dim  # [H, kv_dim]
        o_params = H * H  # [H, H]
        
        return {
            'q': q_params,
            'k': k_params,
            'v': v_params,
            'o': o_params,
            'total': q_params + k_params + v_params + o_params,
        }
    
    def _count_router_params(self) -> int:
        """Count router parameters per MoE layer."""
        if not self.config.moe_enabled:
            return 0
        
        cfg = self.config
        H = cfg.hidden_size
        num_experts = cfg.num_routed_experts + cfg.num_null_experts
        
        # GSA router components
        query_proj = H * cfg.router_num_heads * cfg.router_head_dim
        head_weight = H * cfg.router_num_heads
        expert_keys = num_experts * cfg.router_num_heads * cfg.router_head_dim
        expert_bias = num_experts
        
        return query_proj + head_weight + expert_keys + expert_bias
    
    def _count_expert_params(self) -> int:
        """Count single expert (SwiGLU FFN) parameters."""
        cfg = self.config
        H = cfg.hidden_size
        I = cfg.intermediate_size
        
        # W1 (gate), W2 (down), W3 (up)
        # Plus biases if used (usually not in modern LLMs)
        return 3 * H * I
    
    def _count_gating_params(self) -> int:
        """Count dual gating parameters per layer."""
        if not self.config.use_dual_gating or not self.config.moe_enabled:
            return 0
        
        cfg = self.config
        H = cfg.hidden_size
        gating_dim = int(H * 0.25)
        
        # G1 + G2 per active expert type
        # Applied to both shared and top-k routed
        num_gated = cfg.num_shared_experts + cfg.top_k
        return 2 * H * gating_dim * num_gated
    
    def _count_layernorm_params(self) -> int:
        """Count LayerNorm parameters per layer."""
        # 2 LayerNorms per layer (pre-attention, pre-FFN)
        # Each has scale and shift (gamma, beta)
        return 2 * 2 * self.config.hidden_size
    
    def calculate_breakdown(self) -> ParamBreakdown:
        """Calculate detailed parameter breakdown."""
        cfg = self.config
        breakdown = ParamBreakdown()
        
        # Embeddings
        emb = self._count_embedding_params()
        breakdown.input_embedding = emb['input']
        breakdown.output_embedding = emb['output']
        breakdown.embedding_total = emb['total']
        
        # Attention per layer
        attn = self._count_attention_params()
        breakdown.attention_q = attn['q']
        breakdown.attention_k = attn['k']
        breakdown.attention_v = attn['v']
        breakdown.attention_o = attn['o']
        breakdown.attention_total_per_layer = attn['total']
        
        # Router per layer
        breakdown.router_per_layer = self._count_router_params()
        
        # Experts
        breakdown.expert_single = self._count_expert_params()
        num_moe_layers = cfg.num_layers // cfg.moe_layer_frequency
        
        breakdown.shared_experts_per_layer = breakdown.expert_single * cfg.num_shared_experts
        breakdown.routed_experts_per_layer = breakdown.expert_single * cfg.num_routed_experts
        
        # Gating
        breakdown.gating_per_layer = self._count_gating_params()
        
        # LayerNorm
        breakdown.layernorm_per_layer = self._count_layernorm_params()
        
        # Totals across all layers
        breakdown.attention_total = breakdown.attention_total_per_layer * cfg.num_layers
        breakdown.router_total = breakdown.router_per_layer * num_moe_layers
        breakdown.shared_expert_total = breakdown.shared_experts_per_layer * num_moe_layers
        breakdown.routed_expert_total = breakdown.routed_experts_per_layer * num_moe_layers
        breakdown.gating_total = breakdown.gating_per_layer * num_moe_layers
        breakdown.layernorm_total = breakdown.layernorm_per_layer * cfg.num_layers
        
        # Grand total
        breakdown.total_params = (
            breakdown.embedding_total +
            breakdown.attention_total +
            breakdown.router_total +
            breakdown.shared_expert_total +
            breakdown.routed_expert_total +
            breakdown.gating_total +
            breakdown.layernorm_total
        )
        
        # Active parameters (per token)
        # Embeddings + Attention + Router + Shared experts + top-k routed + Gating + LayerNorm
        active_routed = breakdown.expert_single * cfg.top_k * num_moe_layers
        breakdown.active_params = (
            breakdown.embedding_total +
            breakdown.attention_total +
            breakdown.router_total +
            breakdown.shared_expert_total +
            active_routed +
            breakdown.gating_total +
            breakdown.layernorm_total
        )
        
        # Efficiency ratio
        breakdown.active_ratio = breakdown.active_params / breakdown.total_params
        
        self._breakdown = breakdown
        return breakdown
    
    def full_report(self) -> Dict:
        """Generate comprehensive parameter report."""
        breakdown = self.calculate_breakdown()
        cfg = self.config
        
        def fmt_params(p: int) -> str:
            if p >= 1e9:
                return f"{p/1e9:.2f}B"
            elif p >= 1e6:
                return f"{p/1e6:.2f}M"
            else:
                return f"{p/1e3:.2f}K"
        
        def fmt_pct(p: int, total: int) -> str:
            return f"{100*p/total:.1f}%"
        
        return {
            'config': {
                'model': f"{cfg.num_layers}L-{cfg.hidden_size}H",
                'moe': f"{cfg.num_routed_experts}E-top{cfg.top_k}" if cfg.moe_enabled else "Dense",
                'intermediate': cfg.intermediate_size,
                'vocab_size': cfg.vocab_size,
            },
            'per_component': {
                'embeddings': {
                    'count': fmt_params(breakdown.embedding_total),
                    'pct_of_total': fmt_pct(breakdown.embedding_total, breakdown.total_params),
                },
                'attention': {
                    'q_per_layer': fmt_params(breakdown.attention_q),
                    'k_per_layer': fmt_params(breakdown.attention_k),
                    'v_per_layer': fmt_params(breakdown.attention_v),
                    'o_per_layer': fmt_params(breakdown.attention_o),
                    'total_per_layer': fmt_params(breakdown.attention_total_per_layer),
                    'total_all_layers': fmt_params(breakdown.attention_total),
                    'pct_of_total': fmt_pct(breakdown.attention_total, breakdown.total_params),
                },
                'router': {
                    'per_layer': fmt_params(breakdown.router_per_layer),
                    'total': fmt_params(breakdown.router_total),
                    'pct_of_total': fmt_pct(breakdown.router_total, breakdown.total_params),
                },
                'experts': {
                    'single_expert': fmt_params(breakdown.expert_single),
                    'shared_per_layer': fmt_params(breakdown.shared_experts_per_layer),
                    'routed_per_layer': fmt_params(breakdown.routed_experts_per_layer),
                    'shared_total': fmt_params(breakdown.shared_expert_total),
                    'routed_total': fmt_params(breakdown.routed_expert_total),
                    'pct_shared': fmt_pct(breakdown.shared_expert_total, breakdown.total_params),
                    'pct_routed': fmt_pct(breakdown.routed_expert_total, breakdown.total_params),
                },
                'gating': {
                    'per_layer': fmt_params(breakdown.gating_per_layer),
                    'total': fmt_params(breakdown.gating_total),
                    'pct_of_total': fmt_pct(breakdown.gating_total, breakdown.total_params),
                },
                'layernorm': {
                    'per_layer': fmt_params(breakdown.layernorm_per_layer),
                    'total': fmt_params(breakdown.layernorm_total),
                    'pct_of_total': fmt_pct(breakdown.layernorm_total, breakdown.total_params),
                },
            },
            'summary': {
                'total_params': fmt_params(breakdown.total_params),
                'active_params': fmt_params(breakdown.active_params),
                'active_ratio': f"{breakdown.active_ratio*100:.1f}%",
                'sparsity': f"{(1-breakdown.active_ratio)*100:.1f}%",
            },
            'raw': {
                'total_params': breakdown.total_params,
                'active_params': breakdown.active_params,
            }
        }
    
    def print_report(self):
        """Print formatted parameter report."""
        report = self.full_report()
        
        print("=" * 60)
        print("MoE PARAMETER COUNT REPORT")
        print("=" * 60)
        
        print("\n📊 Configuration:")
        for k, v in report['config'].items():
            print(f"  {k}: {v}")
        
        print("\n📦 Per-Component Breakdown:")
        
        print("\n  Embeddings:")
        for k, v in report['per_component']['embeddings'].items():
            print(f"    {k}: {v}")
        
        print("\n  Attention:")
        for k, v in report['per_component']['attention'].items():
            print(f"    {k}: {v}")
        
        print("\n  Router:")
        for k, v in report['per_component']['router'].items():
            print(f"    {k}: {v}")
        
        print("\n  Experts:")
        for k, v in report['per_component']['experts'].items():
            print(f"    {k}: {v}")
        
        print("\n  Gating (G1+G2):")
        for k, v in report['per_component']['gating'].items():
            print(f"    {k}: {v}")
        
        print("\n📈 Summary:")
        for k, v in report['summary'].items():
            print(f"  {k}: {v}")
        
        print("=" * 60)


# Preset configurations
CONFIGS = {
    '3b_moe': ModelConfig(
        hidden_size=2048,
        num_layers=20,
        num_attention_heads=16,
        num_kv_heads=4,
        intermediate_size=512,
        num_routed_experts=40,
        num_shared_experts=2,
        num_null_experts=20,
        top_k=4,
    ),
    '8b_moe': ModelConfig(
        hidden_size=2048,
        num_layers=40,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=512,
        num_routed_experts=40,
        num_shared_experts=2,
        num_null_experts=20,
        top_k=4,
    ),
    '70b_moe': ModelConfig(
        hidden_size=2048,
        num_layers=40,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=512,
        num_routed_experts=512,
        num_shared_experts=4,
        num_null_experts=256,
        top_k=8,
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
    
    counter = ParamCounter(config)
    counter.print_report()
