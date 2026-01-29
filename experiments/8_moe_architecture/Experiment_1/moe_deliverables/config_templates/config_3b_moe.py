# ============================================================================
# MoE 3B Configuration (Stage 2: Learn Routing)
# ============================================================================
# Canonical configuration for 3B MoE-8 model
# Total Parameters: ~3.0B
# Active Parameters: ~1.2B per token
# ============================================================================

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MoE3BConfig:
    """
    Canonical configuration for 3B MoE-8 model.
    
    Growth stage: Stage 2 (Learn Routing)
    Previous: 1B Dense
    Next: 8B MoE-8
    """
    
    # ==================== Model Identity ====================
    model_name: str = "3B-MoE-8"
    model_stage: int = 2
    description: str = "Learn routing patterns with 8 experts"
    
    # ==================== Architecture ====================
    hidden_size: int = 2048
    num_layers: int = 24
    num_attention_heads: int = 16
    num_kv_heads: int = 4  # GQA: 4 KV heads for 16 Q heads
    intermediate_size: int = 5504  # ~2.7× hidden for SwiGLU
    vocab_size: int = 32000
    max_position_embeddings: int = 4096
    
    # ==================== MoE Configuration ====================
    moe_enabled: bool = True
    moe_layer_frequency: int = 1  # MoE on every layer
    
    # Expert counts
    num_routed_experts: int = 8
    num_shared_experts: int = 2
    num_null_experts: int = 1
    
    # Routing
    top_k: int = 2  # √8 × 0.5 ≈ 1.4 → 2
    router_type: str = "gsa"  # GSA multi-head sigmoid
    router_num_heads: int = 4
    router_head_dim: int = 64
    
    # Adaptive top-k
    adaptive_top_k: bool = True
    adaptive_k_min: int = 1
    adaptive_k_max: int = 3
    
    # Expert gating
    use_dual_gating: bool = True  # G1 + G2 for collapse prevention
    gating_hidden_mult: float = 0.25  # Gate projection size
    
    # ==================== Null Expert ====================
    null_expert_scale: float = 0.001  # Near-zero output
    null_routing_target_junk: float = 0.70  # 70% junk → null
    null_routing_max_signal: float = 0.10  # <10% signal → null
    
    # ==================== Load Balancing ====================
    load_balance_type: str = "bias_adjustment"  # Loss-free
    load_balance_alpha: float = 0.001  # Bias adjustment rate
    target_utilization: float = 0.125  # 1/8 for 8 experts
    
    # ==================== Attention ====================
    attention_type: str = "gqa"  # Grouped Query Attention
    attention_dropout: float = 0.0
    use_rope: bool = True
    rope_theta: float = 10000.0
    
    # ==================== Normalization ====================
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-5
    
    # ==================== Training ====================
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    flash_attention: bool = True
    
    # ==================== Initialization ====================
    initializer_range: float = 0.02
    expert_noise_std: float = 1e-4  # For symmetry breaking
    
    # ==================== Compute Budget ====================
    max_flops_per_token: int = int(2.4e9)  # 2.4B FLOPs
    max_active_params: int = int(1.5e9)   # 1.5B params
    max_total_params: int = int(4e9)      # 4B params
    
    # ==================== Telemetry Thresholds ====================
    telemetry_config: dict = field(default_factory=lambda: {
        'dead_expert_threshold': 0.01,      # <1% = dead
        'overload_threshold': 3.0,          # >3× expected = overload
        'min_entropy': 0.70,                # Normalized routing entropy
        'max_gini': 0.50,                   # Load balance coefficient
        'alert_on_collapse': True,
        'log_interval': 100,
    })
    
    # ==================== Token Classification ====================
    junk_token_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    
    def __post_init__(self):
        """Validate configuration."""
        # Validate compute budget
        active_params = self.compute_active_params()
        flops_per_token = 2 * active_params
        
        assert flops_per_token <= self.max_flops_per_token, \
            f"FLOPs/token {flops_per_token} exceeds budget {self.max_flops_per_token}"
        assert active_params <= self.max_active_params, \
            f"Active params {active_params} exceeds budget {self.max_active_params}"
        
        # Validate MoE config
        assert self.top_k <= self.num_routed_experts, \
            f"top_k ({self.top_k}) cannot exceed num_routed_experts ({self.num_routed_experts})"
    
    def compute_active_params(self) -> int:
        """Compute active parameters per token."""
        # Embeddings (always active)
        embed_params = self.vocab_size * self.hidden_size
        
        # Attention (always active)
        attn_params_per_layer = (
            self.hidden_size * self.hidden_size +  # Q
            self.hidden_size * (self.hidden_size // (self.num_attention_heads // self.num_kv_heads)) * 2 +  # K, V
            self.hidden_size * self.hidden_size  # O
        )
        
        # FFN/Expert (only active experts)
        expert_params = 3 * self.hidden_size * self.intermediate_size  # SwiGLU
        active_experts = self.num_shared_experts + self.top_k
        ffn_params_per_layer = expert_params * active_experts
        
        # Total
        total_attn = attn_params_per_layer * self.num_layers
        total_ffn = ffn_params_per_layer * self.num_layers
        
        return embed_params + total_attn + total_ffn
    
    def compute_total_params(self) -> int:
        """Compute total parameters."""
        embed_params = self.vocab_size * self.hidden_size
        
        attn_params_per_layer = (
            self.hidden_size * self.hidden_size +
            self.hidden_size * (self.hidden_size // (self.num_attention_heads // self.num_kv_heads)) * 2 +
            self.hidden_size * self.hidden_size
        )
        
        expert_params = 3 * self.hidden_size * self.intermediate_size
        total_experts = self.num_routed_experts + self.num_shared_experts
        ffn_params_per_layer = expert_params * total_experts
        
        # Router params
        router_params_per_layer = (
            self.hidden_size * self.router_num_heads * self.router_head_dim +  # Query proj
            self.router_num_heads * self.router_head_dim * (self.num_routed_experts + self.num_null_experts)  # Expert keys
        )
        
        total_attn = attn_params_per_layer * self.num_layers
        total_ffn = ffn_params_per_layer * self.num_layers
        total_router = router_params_per_layer * self.num_layers
        
        return embed_params + total_attn + total_ffn + total_router


# ============================================================================
# Plan B Fallback Configuration
# ============================================================================
@dataclass
class MoE3BFallbackConfig(MoE3BConfig):
    """
    Fallback configuration if primary config shows instability.
    
    Changes from primary:
    - Disable adaptive top-k
    - Increase shared experts
    - Lower expert count
    - More conservative load balancing
    """
    
    model_name: str = "3B-MoE-8-Fallback"
    description: str = "Conservative fallback configuration"
    
    # More conservative MoE
    num_routed_experts: int = 8
    num_shared_experts: int = 3  # Increased from 2
    top_k: int = 2
    
    # Disable adaptive routing
    adaptive_top_k: bool = False
    
    # More aggressive load balancing
    load_balance_alpha: float = 0.002  # Faster adjustment
    
    # Disable dual gating (simpler)
    use_dual_gating: bool = False
    
    # Looser null targets
    null_routing_target_junk: float = 0.50
    null_routing_max_signal: float = 0.15


# ============================================================================
# Export
# ============================================================================
def get_3b_config(fallback: bool = False) -> MoE3BConfig:
    """Get 3B MoE configuration."""
    if fallback:
        return MoE3BFallbackConfig()
    return MoE3BConfig()


if __name__ == "__main__":
    config = get_3b_config()
    print(f"Model: {config.model_name}")
    print(f"Total Params: {config.compute_total_params() / 1e9:.2f}B")
    print(f"Active Params: {config.compute_active_params() / 1e9:.2f}B")
    print(f"FLOPs/Token: {2 * config.compute_active_params() / 1e9:.2f}B")
