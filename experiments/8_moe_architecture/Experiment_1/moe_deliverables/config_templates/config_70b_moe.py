# ============================================================================
# MoE 70B Configuration (Stage 4: Expert Expansion)
# ============================================================================
# Canonical configuration for 70B MoE-64 model
# Total Parameters: ~70B
# Active Parameters: ~12B per token
# ============================================================================

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MoE70BConfig:
    """
    Canonical configuration for 70B MoE-64 model.
    
    Growth stage: Stage 4 (Expert Expansion)
    Previous: 8B MoE-8
    Next: N/A (final stage)
    
    Key changes from 8B:
    - Expert expansion: 8 → 64 (8 parents × 8 children)
    - Shared experts: 2 → 4
    - Null experts: 1 → 2
    - Top-K: 2 → 4
    - Layers: 48 → 80
    """
    
    # ==================== Model Identity ====================
    model_name: str = "70B-MoE-64"
    model_stage: int = 4
    description: str = "Full-scale expert expansion with 64 experts"
    
    # ==================== Architecture ====================
    hidden_size: int = 4096
    num_layers: int = 80
    num_attention_heads: int = 32
    num_kv_heads: int = 8  # GQA: 8 KV heads for 32 Q heads
    intermediate_size: int = 11008  # ~2.7× hidden for SwiGLU
    vocab_size: int = 32000
    max_position_embeddings: int = 8192
    
    # ==================== MoE Configuration ====================
    moe_enabled: bool = True
    moe_layer_frequency: int = 1  # MoE on every layer
    
    # Expert counts
    num_routed_experts: int = 64  # 8 parents × 8 children
    num_shared_experts: int = 4
    num_null_experts: int = 2
    
    # Routing
    top_k: int = 4  # √64 × 0.5 = 4
    router_type: str = "gsa"
    router_num_heads: int = 4
    router_head_dim: int = 128  # Larger for more experts
    
    # Adaptive top-k
    adaptive_top_k: bool = True
    adaptive_k_min: int = 2
    adaptive_k_max: int = 6
    
    # Expert gating
    use_dual_gating: bool = True
    gating_hidden_mult: float = 0.25
    
    # ==================== Expert Hierarchy ====================
    # Tracks parent-child relationships for expansion
    expert_hierarchy: dict = field(default_factory=lambda: {
        'parents_per_stage': 8,
        'children_per_parent': 8,
        'parent_indices': list(range(8)),  # Original 8 parents from 8B
        'child_mapping': {
            # parent_idx: [child_indices]
            0: list(range(0, 8)),
            1: list(range(8, 16)),
            2: list(range(16, 24)),
            3: list(range(24, 32)),
            4: list(range(32, 40)),
            5: list(range(40, 48)),
            6: list(range(48, 56)),
            7: list(range(56, 64)),
        }
    })
    
    # ==================== Null Expert ====================
    null_expert_scale: float = 0.001
    null_routing_target_junk: float = 0.70
    null_routing_max_signal: float = 0.10
    
    # ==================== Load Balancing ====================
    load_balance_type: str = "bias_adjustment"
    load_balance_alpha: float = 0.0005  # Slower for more experts
    target_utilization: float = 0.015625  # 1/64 for 64 experts
    
    # Hierarchical balancing
    hierarchical_balance: bool = True  # Balance within parent groups
    parent_balance_weight: float = 0.5  # Weight for parent-level balance
    
    # ==================== Attention ====================
    attention_type: str = "gqa"
    attention_dropout: float = 0.0
    use_rope: bool = True
    rope_theta: float = 500000.0  # Higher for longer context
    
    # ==================== Normalization ====================
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-5
    
    # ==================== Training ====================
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    flash_attention: bool = True
    
    # ==================== Distributed Training ====================
    distributed_config: dict = field(default_factory=lambda: {
        'min_gpus': 32,
        'recommended_gpus': 64,
        'expert_parallel_size': 1,  # Can increase for more GPUs
        'pipeline_parallel_size': 4,
        'tensor_parallel_size': 1,
        'zero_stage': 3,
        'expert_offload': False,  # Enable if memory constrained
    })
    
    # ==================== Initialization ====================
    initializer_range: float = 0.02
    expert_noise_std: float = 1e-3  # Larger for expansion
    
    # ==================== Compute Budget ====================
    max_flops_per_token: int = int(24e9)  # 24B FLOPs
    max_active_params: int = int(15e9)    # 15B params
    max_total_params: int = int(80e9)     # 80B params
    
    # ==================== Telemetry Thresholds ====================
    telemetry_config: dict = field(default_factory=lambda: {
        'dead_expert_threshold': 0.005,     # <0.5% = dead (stricter for 64 experts)
        'overload_threshold': 4.0,          # >4× expected = overload
        'min_entropy': 0.65,                # Slightly lower for 64 experts
        'max_gini': 0.55,                   # Slightly relaxed
        'alert_on_collapse': True,
        'log_interval': 50,                 # More frequent for large model
        'parent_utilization_check': True,   # Check parent groups balance
    })
    
    # ==================== Token Classification ====================
    junk_token_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    
    # ==================== Expansion Configuration ====================
    expansion_config: dict = field(default_factory=lambda: {
        'source_stage': 3,
        'source_model': '8B-MoE-8',
        'expansion_type': 'hierarchical',
        'children_per_parent': 8,
        'router_key_noise': 0.1,
        'weight_noise': 1e-3,
        'warmup_steps': 1000,
        'router_lr_mult': 0.1,  # Lower LR for router during warmup
    })
    
    def __post_init__(self):
        """Validate configuration."""
        active_params = self.compute_active_params()
        flops_per_token = 2 * active_params
        
        assert flops_per_token <= self.max_flops_per_token, \
            f"FLOPs/token {flops_per_token} exceeds budget {self.max_flops_per_token}"
        assert active_params <= self.max_active_params, \
            f"Active params {active_params} exceeds budget {self.max_active_params}"
        
        # Validate expert hierarchy
        assert self.num_routed_experts == \
            self.expert_hierarchy['parents_per_stage'] * self.expert_hierarchy['children_per_parent']
    
    def compute_active_params(self) -> int:
        """Compute active parameters per token."""
        embed_params = self.vocab_size * self.hidden_size
        
        attn_params_per_layer = (
            self.hidden_size * self.hidden_size +
            self.hidden_size * (self.hidden_size // (self.num_attention_heads // self.num_kv_heads)) * 2 +
            self.hidden_size * self.hidden_size
        )
        
        expert_params = 3 * self.hidden_size * self.intermediate_size
        active_experts = self.num_shared_experts + self.top_k
        ffn_params_per_layer = expert_params * active_experts
        
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
        
        router_params_per_layer = (
            self.hidden_size * self.router_num_heads * self.router_head_dim +
            self.router_num_heads * self.router_head_dim * (self.num_routed_experts + self.num_null_experts)
        )
        
        total_attn = attn_params_per_layer * self.num_layers
        total_ffn = ffn_params_per_layer * self.num_layers
        total_router = router_params_per_layer * self.num_layers
        
        return embed_params + total_attn + total_ffn + total_router


# ============================================================================
# Plan B Fallback Configuration
# ============================================================================
@dataclass
class MoE70BFallbackConfig(MoE70BConfig):
    """
    Fallback configuration if primary shows instability.
    
    Changes from primary:
    - Fewer experts: 64 → 32 (4 parents × 8 children)
    - More shared experts: 4 → 6
    - Lower top-k: 4 → 3
    - Disable adaptive top-k
    - More conservative load balancing
    - Enable expert offloading
    """
    
    model_name: str = "70B-MoE-32-Fallback"
    description: str = "Conservative fallback with fewer experts"
    
    # Reduced expert count
    num_routed_experts: int = 32  # 4 parents × 8 children
    num_shared_experts: int = 6   # Increased
    num_null_experts: int = 2
    top_k: int = 3  # Reduced
    
    # Updated hierarchy
    expert_hierarchy: dict = field(default_factory=lambda: {
        'parents_per_stage': 4,  # Reduced
        'children_per_parent': 8,
        'parent_indices': [0, 2, 4, 6],  # Skip every other parent
        'child_mapping': {
            0: list(range(0, 8)),
            1: list(range(8, 16)),
            2: list(range(16, 24)),
            3: list(range(24, 32)),
        }
    })
    
    # Disable adaptive routing
    adaptive_top_k: bool = False
    
    # More aggressive load balancing
    load_balance_alpha: float = 0.001
    target_utilization: float = 0.03125  # 1/32
    
    # Disable dual gating
    use_dual_gating: bool = False
    
    # Looser null targets
    null_routing_target_junk: float = 0.50
    null_routing_max_signal: float = 0.15
    
    # Enable offloading for memory safety
    distributed_config: dict = field(default_factory=lambda: {
        'min_gpus': 16,
        'recommended_gpus': 32,
        'expert_parallel_size': 1,
        'pipeline_parallel_size': 4,
        'tensor_parallel_size': 1,
        'zero_stage': 3,
        'expert_offload': True,  # Enabled
    })
    
    # Adjusted compute budget
    max_active_params: int = int(12e9)  # Reduced active
    

# ============================================================================
# Alternative: Higher Sparsity Configuration
# ============================================================================
@dataclass
class MoE70BHighSparsityConfig(MoE70BConfig):
    """
    High sparsity variant for maximum efficiency.
    
    Changes:
    - Lower top-k: 4 → 2
    - Fewer shared: 4 → 2
    - Higher null absorption
    - Significant FLOPs reduction
    """
    
    model_name: str = "70B-MoE-64-HighSparsity"
    description: str = "High sparsity for maximum efficiency"
    
    # Aggressive sparsity
    top_k: int = 2
    num_shared_experts: int = 2
    
    # Higher null absorption
    null_routing_target_junk: float = 0.80
    null_routing_max_signal: float = 0.05
    
    # Adjusted budgets
    max_flops_per_token: int = int(16e9)  # Reduced
    max_active_params: int = int(8e9)     # Reduced


# ============================================================================
# Export
# ============================================================================
def get_70b_config(variant: str = "primary") -> MoE70BConfig:
    """
    Get 70B MoE configuration.
    
    Args:
        variant: "primary", "fallback", or "high_sparsity"
    """
    if variant == "fallback":
        return MoE70BFallbackConfig()
    elif variant == "high_sparsity":
        return MoE70BHighSparsityConfig()
    return MoE70BConfig()


if __name__ == "__main__":
    config = get_70b_config()
    print(f"Model: {config.model_name}")
    print(f"Total Params: {config.compute_total_params() / 1e9:.2f}B")
    print(f"Active Params: {config.compute_active_params() / 1e9:.2f}B")
    print(f"FLOPs/Token: {2 * config.compute_active_params() / 1e9:.2f}B")
    print(f"Experts: {config.num_routed_experts} routed + {config.num_shared_experts} shared + {config.num_null_experts} null")
