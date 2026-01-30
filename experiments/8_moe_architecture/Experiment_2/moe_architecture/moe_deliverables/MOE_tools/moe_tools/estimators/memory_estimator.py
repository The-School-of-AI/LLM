#!/usr/bin/env python3
"""
MoE Memory Estimator
====================
Comprehensive memory estimation for MoE architectures with:
- Per-component breakdown (weights, gradients, optimizer, activations)
- Distributed training support (ZeRO, PP, EP, TP)
- Activation checkpointing
- Flash Attention savings

Usage:
    from estimators.memory_estimator import MemoryEstimator
    
    estimator = MemoryEstimator(config)
    report = estimator.full_report()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math


class ZeROStage(Enum):
    """ZeRO optimization stages."""
    NONE = 0
    OPTIMIZER = 1    # Shard optimizer states
    GRADIENT = 2     # + Shard gradients
    PARAMETER = 3    # + Shard parameters


@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    
    num_gpus: int = 32
    
    # Parallelism dimensions
    tensor_parallel_size: int = 1      # TP: Split attention/FFN across GPUs
    pipeline_parallel_size: int = 4    # PP: Split layers across GPUs
    expert_parallel_size: int = 1      # EP: Split experts across GPUs
    
    # ZeRO stage
    zero_stage: ZeROStage = ZeROStage.PARAMETER
    
    # Optimizations
    activation_checkpointing: bool = True
    flash_attention: bool = True
    cpu_offload: bool = False
    expert_offload: bool = False
    
    @property
    def data_parallel_size(self) -> int:
        """Calculate data parallel size."""
        return self.num_gpus // (
            self.tensor_parallel_size * 
            self.pipeline_parallel_size * 
            self.expert_parallel_size
        )


@dataclass
class ModelConfig:
    """Model configuration for memory estimation."""
    
    # Architecture
    hidden_size: int = 4096
    num_layers: int = 40
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    intermediate_size: int = 2048
    vocab_size: int = 32000
    max_seq_length: int = 4096
    
    # MoE Configuration
    moe_enabled: bool = True
    num_routed_experts: int = 64
    num_shared_experts: int = 4
    num_null_experts: int = 2
    top_k: int = 4
    moe_layer_frequency: int = 1
    
    # Router
    router_num_heads: int = 4
    router_head_dim: int = 128
    use_dual_gating: bool = False
    
    # Training
    batch_size: int = 8
    dtype_bytes: int = 2  # bf16 = 2 bytes
    optimizer: str = "adamw"  # 8 bytes per param (m + v in fp32)


@dataclass
class MemoryBreakdown:
    """Detailed memory breakdown in bytes."""
    
    # Model weights
    embedding_weights: int = 0
    attention_weights: int = 0
    router_weights: int = 0
    shared_expert_weights: int = 0
    routed_expert_weights: int = 0
    gating_weights: int = 0
    layernorm_weights: int = 0
    total_weights: int = 0
    
    # Gradients
    total_gradients: int = 0
    
    # Optimizer states
    optimizer_states: int = 0
    
    # Activations
    attention_activations: int = 0
    attention_scores: int = 0  # O(S²) - the killer
    ffn_activations: int = 0
    router_activations: int = 0
    total_activations: int = 0
    
    # With checkpointing
    activations_with_checkpoint: int = 0
    
    # Buffers and overhead
    buffers: int = 0
    
    # Totals
    static_memory: int = 0  # Weights + grads + optimizer
    peak_memory: int = 0    # Static + activations
    
    # Per-GPU (after distribution)
    per_gpu_weights: int = 0
    per_gpu_gradients: int = 0
    per_gpu_optimizer: int = 0
    per_gpu_activations: int = 0
    per_gpu_total: int = 0


class MemoryEstimator:
    """
    Comprehensive memory estimator for MoE models.
    
    Memory components:
    1. Model weights: params × dtype_bytes
    2. Gradients: params × dtype_bytes
    3. Optimizer states: params × 8 (AdamW: m + v in fp32)
    4. Activations: varies by layer type
    5. Buffers: ~10% of static memory
    """
    
    def __init__(self, model_config: ModelConfig, dist_config: DistributedConfig):
        self.model = model_config
        self.dist = dist_config
        self._breakdown = None
    
    def _count_embedding_params(self) -> int:
        """Count embedding layer parameters."""
        # Input embedding + output projection (often tied)
        return self.model.vocab_size * self.model.hidden_size
    
    def _count_attention_params_per_layer(self) -> int:
        """Count attention parameters for one layer."""
        H = self.model.hidden_size
        num_kv_heads = self.model.num_kv_heads
        head_dim = H // self.model.num_attention_heads
        kv_dim = num_kv_heads * head_dim
        
        # Q, K, V, O projections
        q_params = H * H
        k_params = H * kv_dim
        v_params = H * kv_dim
        o_params = H * H
        
        return q_params + k_params + v_params + o_params
    
    def _count_router_params_per_layer(self) -> int:
        """Count router parameters for one MoE layer."""
        if not self.model.moe_enabled:
            return 0
        
        H = self.model.hidden_size
        num_experts = self.model.num_routed_experts + self.model.num_null_experts
        
        # GSA router
        query_proj = H * self.model.router_num_heads * self.model.router_head_dim
        head_weight_proj = H * self.model.router_num_heads
        expert_keys = num_experts * self.model.router_num_heads * self.model.router_head_dim
        expert_bias = num_experts
        
        return query_proj + head_weight_proj + expert_keys + expert_bias
    
    def _count_expert_params(self) -> int:
        """Count single expert parameters (SwiGLU FFN)."""
        H = self.model.hidden_size
        I = self.model.intermediate_size
        
        # W1 (gate), W2 (down), W3 (up)
        return 3 * H * I
    
    def _count_gating_params_per_layer(self) -> int:
        """Count dual gating parameters per layer."""
        if not self.model.use_dual_gating or not self.model.moe_enabled:
            return 0
        
        H = self.model.hidden_size
        gating_dim = int(H * 0.25)
        
        # G1 (output gate) + G2 (input gate) per active expert
        num_gated_experts = self.model.num_shared_experts + self.model.top_k
        return 2 * H * gating_dim * num_gated_experts
    
    def _count_layernorm_params_per_layer(self) -> int:
        """Count LayerNorm parameters (scale + bias)."""
        return 2 * self.model.hidden_size  # 2 LayerNorms per layer
    
    def _calculate_attention_activations(self) -> Tuple[int, int]:
        """
        Calculate attention activation memory.
        
        Returns:
            (activations_without_scores, attention_scores)
        
        Note: Attention scores are O(S²) - the memory killer!
        """
        B = self.model.batch_size
        S = self.model.max_seq_length
        H = self.model.hidden_size
        num_heads = self.model.num_attention_heads
        dtype = self.model.dtype_bytes
        
        # Q, K, V after projection
        qkv_activations = 3 * B * S * H * dtype
        
        # Attention scores: [B, heads, S, S] - O(S²)!
        if self.dist.flash_attention:
            # Flash Attention: O(S) instead of O(S²)
            attn_scores = B * num_heads * S * 32  # Just state, not full matrix
        else:
            attn_scores = B * num_heads * S * S * dtype
        
        # Output activations
        output_activations = B * S * H * dtype
        
        return qkv_activations + output_activations, attn_scores
    
    def _calculate_ffn_activations(self) -> int:
        """Calculate FFN/expert activation memory."""
        B = self.model.batch_size
        S = self.model.max_seq_length
        H = self.model.hidden_size
        I = self.model.intermediate_size
        dtype = self.model.dtype_bytes
        
        if self.model.moe_enabled:
            # Active experts only
            num_active = self.model.num_shared_experts + self.model.top_k
        else:
            num_active = 1
        
        # Intermediate activations per expert
        per_expert = B * S * I * dtype
        
        return per_expert * num_active
    
    def _calculate_router_activations(self) -> int:
        """Calculate router activation memory."""
        if not self.model.moe_enabled:
            return 0
        
        B = self.model.batch_size
        S = self.model.max_seq_length
        num_experts = self.model.num_routed_experts + self.model.num_null_experts
        dtype = self.model.dtype_bytes
        
        # Scores, indices, weights
        scores = B * S * num_experts * dtype
        indices = B * S * self.model.top_k * 4  # int32
        weights = B * S * self.model.top_k * dtype
        
        return scores + indices + weights
    
    def calculate_breakdown(self) -> MemoryBreakdown:
        """Calculate detailed memory breakdown."""
        breakdown = MemoryBreakdown()
        dtype = self.model.dtype_bytes
        
        # ============ Model Weights ============
        
        # Embeddings
        breakdown.embedding_weights = self._count_embedding_params() * dtype
        
        # Attention (all layers)
        breakdown.attention_weights = (
            self._count_attention_params_per_layer() * 
            self.model.num_layers * dtype
        )
        
        # Router (MoE layers only)
        num_moe_layers = self.model.num_layers // self.model.moe_layer_frequency
        breakdown.router_weights = (
            self._count_router_params_per_layer() * 
            num_moe_layers * dtype
        )
        
        # Experts
        expert_params = self._count_expert_params()
        breakdown.shared_expert_weights = (
            expert_params * self.model.num_shared_experts * 
            num_moe_layers * dtype
        )
        breakdown.routed_expert_weights = (
            expert_params * self.model.num_routed_experts * 
            num_moe_layers * dtype
        )
        
        # Gating
        breakdown.gating_weights = (
            self._count_gating_params_per_layer() * 
            num_moe_layers * dtype
        )
        
        # LayerNorm
        breakdown.layernorm_weights = (
            self._count_layernorm_params_per_layer() * 
            self.model.num_layers * dtype
        )
        
        # Total weights
        breakdown.total_weights = (
            breakdown.embedding_weights +
            breakdown.attention_weights +
            breakdown.router_weights +
            breakdown.shared_expert_weights +
            breakdown.routed_expert_weights +
            breakdown.gating_weights +
            breakdown.layernorm_weights
        )
        
        # ============ Gradients ============
        breakdown.total_gradients = breakdown.total_weights  # Same size as weights
        
        # ============ Optimizer States ============
        # AdamW: 8 bytes per param (m + v in fp32)
        total_params = breakdown.total_weights // dtype
        breakdown.optimizer_states = total_params * 8
        
        # ============ Activations ============
        attn_act, attn_scores = self._calculate_attention_activations()
        breakdown.attention_activations = attn_act * self.model.num_layers
        breakdown.attention_scores = attn_scores * self.model.num_layers
        breakdown.ffn_activations = self._calculate_ffn_activations() * num_moe_layers
        breakdown.router_activations = self._calculate_router_activations() * num_moe_layers
        
        breakdown.total_activations = (
            breakdown.attention_activations +
            breakdown.attention_scores +
            breakdown.ffn_activations +
            breakdown.router_activations
        )
        
        # With activation checkpointing: ~1/3 of activations
        if self.dist.activation_checkpointing:
            breakdown.activations_with_checkpoint = breakdown.total_activations // 3
        else:
            breakdown.activations_with_checkpoint = breakdown.total_activations
        
        # ============ Buffers ============
        breakdown.static_memory = (
            breakdown.total_weights +
            breakdown.total_gradients +
            breakdown.optimizer_states
        )
        breakdown.buffers = int(breakdown.static_memory * 0.10)
        
        # ============ Peak Memory ============
        breakdown.peak_memory = (
            breakdown.static_memory +
            breakdown.activations_with_checkpoint +
            breakdown.buffers
        )
        
        # ============ Per-GPU with Distribution ============
        self._apply_distribution(breakdown)
        
        self._breakdown = breakdown
        return breakdown
    
    def _apply_distribution(self, breakdown: MemoryBreakdown):
        """Apply distributed training sharding to memory estimates."""
        PP = self.dist.pipeline_parallel_size
        TP = self.dist.tensor_parallel_size
        EP = self.dist.expert_parallel_size
        DP = self.dist.data_parallel_size
        zero = self.dist.zero_stage
        
        # Weights sharding
        # PP: layers split
        # TP: attention/FFN split
        # EP: experts split
        weight_shard_factor = PP * TP
        expert_shard_factor = PP * EP
        
        # Attention weights: sharded by PP * TP
        attn_per_gpu = (
            breakdown.attention_weights +
            breakdown.layernorm_weights +
            breakdown.embedding_weights
        ) // weight_shard_factor
        
        # Expert weights: sharded by PP * EP
        expert_per_gpu = (
            breakdown.shared_expert_weights +
            breakdown.routed_expert_weights +
            breakdown.router_weights +
            breakdown.gating_weights
        ) // expert_shard_factor
        
        breakdown.per_gpu_weights = attn_per_gpu + expert_per_gpu
        
        # ZeRO sharding for gradients
        if zero.value >= ZeROStage.GRADIENT.value:
            breakdown.per_gpu_gradients = breakdown.per_gpu_weights // DP
        else:
            breakdown.per_gpu_gradients = breakdown.per_gpu_weights
        
        # ZeRO sharding for optimizer
        if zero.value >= ZeROStage.OPTIMIZER.value:
            # Optimizer states sharded by DP
            breakdown.per_gpu_optimizer = breakdown.optimizer_states // (PP * TP * EP * DP)
        else:
            breakdown.per_gpu_optimizer = breakdown.optimizer_states // (PP * TP * EP)
        
        # ZeRO-3: also shard weights across DP
        if zero == ZeROStage.PARAMETER:
            breakdown.per_gpu_weights = breakdown.per_gpu_weights // DP
            breakdown.per_gpu_gradients = breakdown.per_gpu_gradients // DP
        
        # Activations: not sharded by ZeRO, only by PP
        # Each pipeline stage only stores activations for its layers
        layers_per_gpu = self.model.num_layers // PP
        breakdown.per_gpu_activations = (
            breakdown.activations_with_checkpoint * layers_per_gpu // self.model.num_layers
        )
        
        # Total per GPU
        breakdown.per_gpu_total = (
            breakdown.per_gpu_weights +
            breakdown.per_gpu_gradients +
            breakdown.per_gpu_optimizer +
            breakdown.per_gpu_activations +
            breakdown.buffers // (PP * TP * EP)
        )
    
    def full_report(self) -> Dict:
        """Generate comprehensive memory report."""
        breakdown = self.calculate_breakdown()
        
        def to_gb(bytes_val: int) -> str:
            return f"{bytes_val / 1e9:.2f} GB"
        
        def to_tb(bytes_val: int) -> str:
            return f"{bytes_val / 1e12:.3f} TB"
        
        total_params = breakdown.total_weights // self.model.dtype_bytes
        
        return {
            'config': {
                'model': f"{self.model.num_layers}L-{self.model.hidden_size}H",
                'moe': f"{self.model.num_routed_experts}E-top{self.model.top_k}" if self.model.moe_enabled else "Dense",
                'total_params': f"{total_params / 1e9:.2f}B",
                'num_gpus': self.dist.num_gpus,
                'parallelism': f"TP={self.dist.tensor_parallel_size}, PP={self.dist.pipeline_parallel_size}, EP={self.dist.expert_parallel_size}, DP={self.dist.data_parallel_size}",
                'zero_stage': self.dist.zero_stage.name,
            },
            'weights_breakdown': {
                'embeddings': to_gb(breakdown.embedding_weights),
                'attention': to_gb(breakdown.attention_weights),
                'router': to_gb(breakdown.router_weights),
                'shared_experts': to_gb(breakdown.shared_expert_weights),
                'routed_experts': to_gb(breakdown.routed_expert_weights),
                'gating': to_gb(breakdown.gating_weights),
                'layernorm': to_gb(breakdown.layernorm_weights),
                'total': to_gb(breakdown.total_weights),
            },
            'training_memory': {
                'weights': to_gb(breakdown.total_weights),
                'gradients': to_gb(breakdown.total_gradients),
                'optimizer_states': to_gb(breakdown.optimizer_states),
                'static_total': to_gb(breakdown.static_memory),
            },
            'activations': {
                'attention': to_gb(breakdown.attention_activations),
                'attention_scores_O_S2': to_gb(breakdown.attention_scores),
                'ffn_experts': to_gb(breakdown.ffn_activations),
                'router': to_gb(breakdown.router_activations),
                'total_no_checkpoint': to_gb(breakdown.total_activations),
                'with_checkpoint': to_gb(breakdown.activations_with_checkpoint),
                'flash_attention': "Enabled" if self.dist.flash_attention else "Disabled",
            },
            'peak_memory': {
                'total_single_gpu': to_tb(breakdown.peak_memory),
                'buffers': to_gb(breakdown.buffers),
            },
            'distributed': {
                'weights_per_gpu': to_gb(breakdown.per_gpu_weights),
                'gradients_per_gpu': to_gb(breakdown.per_gpu_gradients),
                'optimizer_per_gpu': to_gb(breakdown.per_gpu_optimizer),
                'activations_per_gpu': to_gb(breakdown.per_gpu_activations),
                'total_per_gpu': to_gb(breakdown.per_gpu_total),
                'cluster_total': to_tb(breakdown.per_gpu_total * self.dist.num_gpus),
            },
            'recommendations': self._get_recommendations(breakdown),
        }
    
    def _get_recommendations(self, breakdown: MemoryBreakdown) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        per_gpu_gb = breakdown.per_gpu_total / 1e9
        
        if per_gpu_gb > 80:
            recommendations.append("⚠️ Per-GPU memory exceeds 80GB. Consider:")
            recommendations.append("  - Increase pipeline parallelism")
            recommendations.append("  - Enable ZeRO-3 if not already")
            recommendations.append("  - Reduce batch size")
        elif per_gpu_gb > 70:
            recommendations.append("⚡ Memory usage is high (>70GB). Monitor closely.")
        else:
            recommendations.append("✅ Memory usage looks healthy for A100-80GB")
        
        if not self.dist.activation_checkpointing:
            savings = (breakdown.total_activations - breakdown.total_activations // 3) / 1e9
            recommendations.append(f"💡 Enable activation checkpointing to save ~{savings:.1f}GB")
        
        if not self.dist.flash_attention:
            recommendations.append("💡 Enable Flash Attention to reduce O(S²) attention scores memory")
        
        return recommendations
    
    def print_report(self):
        """Print formatted memory report."""
        report = self.full_report()
        
        print("=" * 60)
        print("MoE MEMORY ESTIMATION REPORT")
        print("=" * 60)
        
        print("\n📊 Configuration:")
        for k, v in report['config'].items():
            print(f"  {k}: {v}")
        
        print("\n💾 Weights Breakdown:")
        for k, v in report['weights_breakdown'].items():
            print(f"  {k}: {v}")
        
        print("\n🔄 Training Memory (Static):")
        for k, v in report['training_memory'].items():
            print(f"  {k}: {v}")
        
        print("\n📈 Activations:")
        for k, v in report['activations'].items():
            print(f"  {k}: {v}")
        
        print("\n🖥️ Per-GPU Memory (Distributed):")
        for k, v in report['distributed'].items():
            print(f"  {k}: {v}")
        
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  {rec}")
        
        print("=" * 60)


# Preset configurations
def get_config(model_size: str) -> Tuple[ModelConfig, DistributedConfig]:
    """Get preset configurations."""
    configs = {
        '3b_moe': (
            ModelConfig(
                hidden_size=2048,
                num_layers=24,
                num_attention_heads=16,
                num_kv_heads=4,
                intermediate_size=2048,
                num_routed_experts=8,
                num_shared_experts=2,
                num_null_experts=1,
                top_k=2,
            ),
            DistributedConfig(
                num_gpus=1,
                pipeline_parallel_size=1,
                zero_stage=ZeROStage.NONE,
            )
        ),
        '8b_moe': (
            ModelConfig(
                hidden_size=4096,
                num_layers=40,
                num_attention_heads=32,
                num_kv_heads=8,
                intermediate_size=2048,
                num_routed_experts=8,
                num_shared_experts=2,
                num_null_experts=1,
                top_k=2,
            ),
            DistributedConfig(
                num_gpus=4,
                pipeline_parallel_size=1,
                zero_stage=ZeROStage.GRADIENT,
            )
        ),
        '70b_moe': (
            ModelConfig(
                hidden_size=4096,
                num_layers=40,
                num_attention_heads=32,
                num_kv_heads=8,
                intermediate_size=2048,
                num_routed_experts=64,
                num_shared_experts=4,
                num_null_experts=2,
                top_k=4,
            ),
            DistributedConfig(
                num_gpus=32,
                pipeline_parallel_size=4,
                zero_stage=ZeROStage.PARAMETER,
            )
        ),
    }
    
    return configs.get(model_size, configs['70b_moe'])


if __name__ == "__main__":
    import sys
    
    model = sys.argv[1] if len(sys.argv) > 1 else '70b_moe'
    model_config, dist_config = get_config(model)
    
    estimator = MemoryEstimator(model_config, dist_config)
    estimator.print_report()
