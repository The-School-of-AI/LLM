"""
MoE Model Configuration
=======================

Base configuration class for all model variants (1B Dense, 3B MoE, 8B MoE, 70B MoE).

Configuration Philosophy:
- All hyperparameters are explicit and documented
- Derived values are computed automatically
- Validation ensures consistency
- Team integration points are clearly marked

References:
- DeepSeek-V3: https://arxiv.org/abs/2401.06066
- GSA Paper: arXiv:2601.15305v1
- MoE Load Balancing: arXiv:2406.13233
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
import math
import json


class ModelType(Enum):
    """Model architecture type."""
    DENSE = "dense"
    MOE = "moe"


class RouterType(Enum):
    """Router architecture type."""
    NONE = "none"           # Dense model, no router
    GSA = "gsa"             # Gated Sparse Attention style
    TOPK_SOFTMAX = "topk"   # Traditional top-k softmax
    EXPERT_CHOICE = "ec"    # Expert choice (not recommended for autoregressive)
    NULL_EXPERT = "null_expert"  # Token-choice with null expert copies (data sparsity)


@dataclass
class TokenizerConfig:
    """
    Tokenizer configuration (Team 6 Integration).
    
    Defines token ID bands and special token registry for proper
    null-routing decisions.
    """
    vocab_size: int = 32000
    
    # Special token IDs (Team 6 specification)
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    
    # Token ID bands for routing decisions
    # These help router identify token types for null routing
    special_token_range: Tuple[int, int] = (0, 100)       # IDs 0-99: special tokens
    punctuation_range: Tuple[int, int] = (100, 200)       # IDs 100-199: punctuation
    number_range: Tuple[int, int] = (200, 300)            # IDs 200-299: numbers
    common_word_range: Tuple[int, int] = (300, 1000)      # IDs 300-999: common words
    # IDs 1000+: regular vocabulary
    
    # Junk token classification (for null routing targets)
    junk_token_ids: List[int] = field(default_factory=lambda: [0])  # Padding
    
    def is_junk_token(self, token_id: int) -> bool:
        """Check if token should be considered junk for null routing."""
        if token_id in self.junk_token_ids:
            return True
        if self.special_token_range[0] <= token_id < self.special_token_range[1]:
            return True
        return False
    
    def get_token_type(self, token_id: int) -> str:
        """Get token type for telemetry."""
        if token_id in self.junk_token_ids or token_id == self.pad_token_id:
            return "junk"
        if self.special_token_range[0] <= token_id < self.special_token_range[1]:
            return "special"
        if self.punctuation_range[0] <= token_id < self.punctuation_range[1]:
            return "punctuation"
        if self.number_range[0] <= token_id < self.number_range[1]:
            return "number"
        if self.common_word_range[0] <= token_id < self.common_word_range[1]:
            return "common"
        return "regular"


@dataclass
class RouterConfig:
    """
    Router configuration for MoE routing.
    
    Supports:
    - GSA-style router (arXiv:2601.15305v1)
    - Token-choice with null experts for data sparsity (arXiv:2601.15370v1)
    """
    router_type: RouterType = RouterType.NULL_EXPERT
    
    # GSA Router Architecture
    num_router_heads: int = 4           # H_I in GSA paper (indexer heads)
    router_dim: int = 64                # d_I in GSA paper (low-dim projection)
    
    # Top-K Configuration
    top_k: int = 2                      # Base number of active experts
    top_k_min: int = 1                  # Minimum for adaptive top-k
    top_k_max: int = 4                  # Maximum for adaptive top-k
    use_adaptive_top_k: bool = True     # Enable GSA adaptive sparsity
    
    # Load Balancing (Loss-Free per Team 7 spec)
    use_aux_loss: bool = False          # NO auxiliary loss (loss-free)
    aux_loss_weight: float = 0.0        # Disabled
    bias_update_speed: float = 0.001    # γ in DeepSeek-V3
    bias_clamp_min: float = -2.0        # Minimum expert bias
    bias_clamp_max: float = 2.0         # Maximum expert bias
    
    # Null Expert Configuration
    null_bias_init: float = 0.1         # Slight preference for null routing
    null_target_junk_rate: Tuple[float, float] = (0.6, 0.8)   # Target: 60-80% junk→null
    null_target_signal_rate: Tuple[float, float] = (0.0, 0.1) # Target: <10% signal→null

    # Null Expert Router (data sparsity)
    data_sparsity: float = 1.0          # ρ in paper (1.0 = no null copies)
    null_copies: int = 20                # M copies of null logit (overrides data_sparsity)
    
    # Router Stability
    router_z_loss_weight: float = 0.0   # Z-loss for router stability (optional)
    score_scale: float = 1.0            # Score scaling factor
    
    # Variance EMA for adaptive top-k
    variance_ema_decay: float = 0.99


@dataclass  
class ExpertConfig:
    """
    Expert (FFN) Configuration.
    
    Each expert is a SwiGLU FFN with optional dual gating (G1+G2)
    from the GSA paper for collapse prevention.
    """
    intermediate_size: int = 512       # FFN intermediate dimension
    
    # Dual Gating (from GSA paper Section 3.5)
    use_dual_gating: bool = False        # Enable G1 (output) + G2 (input) gates
    gate_bias_init: float = 0.0         # Initialize for σ(·) ≈ 0.5
    
    # Expert Initialization
    expert_init_std: float = 0.02       # Weight initialization std
    noise_std_for_expansion: float = 1e-4  # Noise for symmetry breaking


@dataclass
class AttentionConfig:
    """
    Attention configuration (GQA or GSA).
    
    Uses GQA for efficient KV cache during inference.
    GSA adds gated sparse attention components per arXiv:2601.15305v1.
    """
    attention_type: str = "gqa"        # "gqa" (dense) or "gsa" (gated sparse attention)
    num_attention_heads: int = 16       # Query heads
    num_kv_heads: int = 4               # Key-Value heads (GQA)
    head_dim: int = 128                 # Dimension per head
    
    # RoPE Configuration
    rope_theta: float = 10000.0         # RoPE base frequency
    rope_scaling: Optional[Dict] = None  # For context extension
    
    # Attention Parameters
    attention_dropout: float = 0.0
    attention_bias: bool = False        # No bias in attention projections

    # GSA (Gated Sparse Attention) Parameters
    gsa_indexer_dim: int = 64           # d_I (low-dim indexer projection)
    gsa_indexer_heads: int = 4          # H_I (indexer heads)
    gsa_k_base: int = 2048              # Base selection budget
    gsa_k_min: int = 256                # Minimum selection budget
    gsa_k_max: int = 4096               # Maximum selection budget
    gsa_variance_ema_decay: float = 0.99
    gsa_gate_bias_init: float = 0.0     # Initialize gate bias for σ(·) ≈ 0.5


@dataclass
class ComputeBudget:
    """
    Compute Budget Configuration (Team Coordination).
    
    Hard ceilings for compute to ensure architecture fits
    within allocated resources.
    """
    # FLOPs budget per forward pass (approximate)
    max_flops_per_token: int = int(2e12)  # 2 TFLOPs per token
    
    # Memory budget
    max_params_total: int = int(70e9)     # Total parameters
    max_params_active: int = int(15e9)    # Active parameters per forward
    
    # Training budget
    target_tokens: int = int(2e12)        # Training tokens
    
    # Inference constraints
    max_sequence_length: int = 4096       # Maximum context
    max_batch_size: int = 32              # Maximum batch size


@dataclass
class TelemetryConfig:
    """
    Telemetry Configuration (Team 7 Integration).
    
    Defines monitoring and health check parameters for
    null-routing and router stability.
    """
    # Logging frequency
    log_every_n_steps: int = 100
    
    # Health check thresholds
    dead_expert_threshold: float = 0.01   # <1% utilization = dead
    overload_expert_threshold: float = 3.0 # >3× average = overloaded
    min_router_entropy: float = 0.7       # Normalized entropy threshold
    max_gini_coefficient: float = 0.5     # Load balance threshold
    
    # Null routing alerts
    junk_null_rate_alert_low: float = 0.5   # Alert if junk→null < 50%
    junk_null_rate_alert_high: float = 0.9  # Alert if junk→null > 90%
    signal_null_rate_alert: float = 0.15    # Alert if signal→null > 15%
    
    # Auto-correction
    enable_auto_correction: bool = True
    correction_strength: float = 0.1


@dataclass
class MoEModelConfig:
    """
    Complete MoE Model Configuration.
    
    This is the main configuration class that combines all sub-configurations
    and provides validation and derived value computation.
    
    Growth Cadence:
    - Stage 1: 1B Dense (foundation)
    - Stage 2: 3B MoE-8 (learn routing)
    - Stage 3: 8B MoE-8 (scale dimensions)
    - Stage 4: 70B MoE-64 (expand experts)
    """
    
    # Model identification
    model_name: str = "moe_model"
    model_type: ModelType = ModelType.MOE
    stage: int = 2  # Growth stage (1-4)
    
    # Core dimensions
    hidden_size: int = 2048
    num_layers: int = 20
    
    # MoE Configuration
    num_routed_experts: int = 40
    num_shared_experts: int = 2
    num_null_experts: int = 1
    moe_layer_frequency: int = 1        # MoE every N layers (1 = all, 2 = every other)
    
    # Sub-configurations
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    compute_budget: ComputeBudget = field(default_factory=ComputeBudget)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    
    # Training Configuration
    max_position_embeddings: int = 4096
    hidden_dropout: float = 0.0
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    
    # Precision
    torch_dtype: str = "bfloat16"
    
    def __post_init__(self):
        """Validate and compute derived values."""
        self._validate()
        self._compute_derived_values()
    
    def _validate(self):
        """Validate configuration consistency."""
        # Check hidden size divisibility
        assert self.hidden_size % self.attention.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.attention.num_attention_heads})"
        
        # Check GQA configuration
        assert self.attention.num_attention_heads % self.attention.num_kv_heads == 0, \
            f"num_attention_heads ({self.attention.num_attention_heads}) must be divisible by num_kv_heads ({self.attention.num_kv_heads})"

        # Check attention type
        if self.attention.attention_type not in {"gqa", "gsa"}:
            raise ValueError(
                f"Unsupported attention_type: {self.attention.attention_type}. "
                "Use 'gqa' or 'gsa'."
            )
        
        # Check expert configuration for MoE
        if self.model_type == ModelType.MOE:
            assert self.num_routed_experts > 0, "MoE requires at least 1 routed expert"
            assert self.router.top_k <= self.num_routed_experts + self.num_null_experts, \
                f"top_k ({self.router.top_k}) cannot exceed total routable experts"
            # if self.router.router_type == RouterType.NULL_EXPERT:
                # assert self.num_null_experts == 1, (
                #     "NULL_EXPERT router expects a single null expert (num_null_experts=1)."
                # )
        
        # Check head dimension
        computed_head_dim = self.hidden_size // self.attention.num_attention_heads
        if self.attention.head_dim != computed_head_dim:
            print(f"Warning: head_dim ({self.attention.head_dim}) overridden to {computed_head_dim}")
            self.attention.head_dim = computed_head_dim
    
    def _compute_derived_values(self):
        """Compute derived configuration values."""
        # Total experts in routing pool
        self.total_routable_experts = self.num_routed_experts + self.num_null_experts
        self.total_experts = self.num_routed_experts + self.num_shared_experts + self.num_null_experts
        
        # Number of MoE layers (exact count based on placement rule)
        self.num_moe_layers = self.num_moe_layers_count
        self.num_dense_layers = self.num_layers - self.num_moe_layers
        
        # Expert size (SwiGLU has 3 weight matrices)
        self.expert_params = 3 * self.hidden_size * self.expert.intermediate_size
        if self.expert.use_dual_gating:
            # Two gating projections with bias terms
            self.expert_params += 2 * (self.hidden_size * self.hidden_size + self.hidden_size)
        
        # Approximate total parameters
        self._estimate_parameters()
        
        # Active expert ratio
        if self.model_type == ModelType.MOE:
            active_experts = self.num_shared_experts + self.router.top_k
            self.active_expert_ratio = active_experts / self.total_experts
        else:
            self.active_expert_ratio = 1.0
    
    def _estimate_parameters(self):
        """Estimate total and active parameters."""
        # Embedding parameters
        embed_params = self.tokenizer.vocab_size * self.hidden_size * 2  # input + output
        
        # Attention parameters per layer (with GQA)
        q_params = self.hidden_size * self.hidden_size
        kv_params = self.hidden_size * (self.hidden_size // (self.attention.num_attention_heads // self.attention.num_kv_heads)) * 2
        o_params = self.hidden_size * self.hidden_size
        attn_params_per_layer = q_params + kv_params + o_params

        if self.attention.attention_type == "gsa":
            # GSA extra parameters: value gate, output gate, and indexer
            kv_dim = self.attention.num_kv_heads * self.attention.head_dim
            q_dim = self.attention.num_attention_heads * self.attention.head_dim
            gate_v_params = self.hidden_size * kv_dim + kv_dim
            gate_o_params = self.hidden_size * q_dim + q_dim
            indexer_params = (
                2 * (self.hidden_size * self.attention.gsa_indexer_heads * self.attention.gsa_indexer_dim) +
                (self.hidden_size * self.attention.gsa_indexer_heads + self.attention.gsa_indexer_heads) +
                self.attention.gsa_indexer_heads
            )
            attn_params_per_layer += gate_v_params + gate_o_params + indexer_params
        
        # LayerNorm parameters per layer
        norm_params_per_layer = self.hidden_size * 4  # 2 norms × 2 params each
        
        # FFN/Expert parameters
        expert_ffn_params = 3 * self.hidden_size * self.expert.intermediate_size
        expert_gating_params = 0
        if self.expert.use_dual_gating:
            expert_gating_params = 2 * (self.hidden_size * self.hidden_size + self.hidden_size)
        moe_expert_params = expert_ffn_params + expert_gating_params
        dense_ffn_params = expert_ffn_params

        if self.model_type == ModelType.MOE:
            # MoE layers
            moe_params_per_layer = (
                self.num_routed_experts * moe_expert_params +
                self.num_shared_experts * moe_expert_params +
                self.hidden_size * self.total_routable_experts  # Router
            )
            # Dense layers (if any)
            dense_ffn_per_layer = dense_ffn_params
            
            total_layer_params = (
                self.num_moe_layers * (attn_params_per_layer + moe_params_per_layer + norm_params_per_layer) +
                self.num_dense_layers * (attn_params_per_layer + dense_ffn_per_layer + norm_params_per_layer)
            )
            
            # Active parameters (per forward pass)
            active_moe_per_layer = (
                self.num_shared_experts * moe_expert_params +
                self.router.top_k * moe_expert_params
            )
            self.estimated_active_params = (
                embed_params +
                self.num_moe_layers * (attn_params_per_layer + active_moe_per_layer + norm_params_per_layer) +
                self.num_dense_layers * (attn_params_per_layer + dense_ffn_per_layer + norm_params_per_layer)
            )
        else:
            # Dense model
            ffn_per_layer = dense_ffn_params
            total_layer_params = self.num_layers * (attn_params_per_layer + ffn_per_layer + norm_params_per_layer)
            self.estimated_active_params = embed_params + total_layer_params
        
        self.estimated_total_params = embed_params + total_layer_params
    
    @property
    def num_moe_layers_count(self) -> int:
        """Get actual count of MoE layers."""
        if self.model_type == ModelType.DENSE:
            return 0
        return sum(1 for i in range(self.num_layers) if i % self.moe_layer_frequency == 0)
    
    def is_moe_layer(self, layer_idx: int) -> bool:
        """Check if a layer should be MoE."""
        if self.model_type == ModelType.DENSE:
            return False
        return layer_idx % self.moe_layer_frequency == 0
    
    def get_expert_indices(self) -> Dict[str, List[int]]:
        """Get expert indices by type."""
        routed_start = 0
        routed_end = self.num_routed_experts
        null_start = routed_end
        null_end = null_start + self.num_null_experts
        
        return {
            'routed': list(range(routed_start, routed_end)),
            'null': list(range(null_start, null_end)),
            'shared': list(range(self.num_shared_experts)),  # Separate indexing
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type.value,
            'stage': self.stage,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_routed_experts': self.num_routed_experts,
            'num_shared_experts': self.num_shared_experts,
            'num_null_experts': self.num_null_experts,
            'moe_layer_frequency': self.moe_layer_frequency,
            'intermediate_size': self.expert.intermediate_size,
            'attention_type': self.attention.attention_type,
            'num_attention_heads': self.attention.num_attention_heads,
            'num_kv_heads': self.attention.num_kv_heads,
            'head_dim': self.attention.head_dim,
            'gsa_indexer_dim': self.attention.gsa_indexer_dim,
            'gsa_indexer_heads': self.attention.gsa_indexer_heads,
            'gsa_k_base': self.attention.gsa_k_base,
            'gsa_k_min': self.attention.gsa_k_min,
            'gsa_k_max': self.attention.gsa_k_max,
            'gsa_variance_ema_decay': self.attention.gsa_variance_ema_decay,
            'gsa_gate_bias_init': self.attention.gsa_gate_bias_init,
            'vocab_size': self.tokenizer.vocab_size,
            'max_position_embeddings': self.max_position_embeddings,
            'router_type': self.router.router_type.value,
            'top_k': self.router.top_k,
            'data_sparsity': self.router.data_sparsity,
            'null_copies': self.router.null_copies,
            'estimated_total_params': self.estimated_total_params,
            'estimated_active_params': self.estimated_active_params,
        }
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MoEModelConfig':
        """Create configuration from dictionary."""
        # Handle nested configs
        config_dict['model_type'] = ModelType(config_dict.get('model_type', 'moe'))
        
        # Create sub-configs
        tokenizer = TokenizerConfig(vocab_size=config_dict.get('vocab_size', 32000))
        router = RouterConfig(
            router_type=RouterType(config_dict.get('router_type', 'gsa')),
            top_k=config_dict.get('top_k', 2),
            data_sparsity=config_dict.get('data_sparsity', 1.0),
            null_copies=config_dict.get('null_copies', 0),
        )
        expert = ExpertConfig(intermediate_size=config_dict.get('intermediate_size', 5504))
        attention = AttentionConfig(
            attention_type=config_dict.get('attention_type', 'gqa'),
            num_attention_heads=config_dict.get('num_attention_heads', 16),
            num_kv_heads=config_dict.get('num_kv_heads', 4),
            head_dim=config_dict.get('head_dim', 128),
            gsa_indexer_dim=config_dict.get('gsa_indexer_dim', 64),
            gsa_indexer_heads=config_dict.get('gsa_indexer_heads', 4),
            gsa_k_base=config_dict.get('gsa_k_base', 2048),
            gsa_k_min=config_dict.get('gsa_k_min', 256),
            gsa_k_max=config_dict.get('gsa_k_max', 4096),
            gsa_variance_ema_decay=config_dict.get('gsa_variance_ema_decay', 0.99),
            gsa_gate_bias_init=config_dict.get('gsa_gate_bias_init', 0.0),
        )
        
        return cls(
            model_name=config_dict.get('model_name', 'moe_model'),
            model_type=config_dict['model_type'],
            stage=config_dict.get('stage', 2),
            hidden_size=config_dict.get('hidden_size', 2048),
            num_layers=config_dict.get('num_layers', 24),
            num_routed_experts=config_dict.get('num_routed_experts', 8),
            num_shared_experts=config_dict.get('num_shared_experts', 2),
            num_null_experts=config_dict.get('num_null_experts', 1),
            moe_layer_frequency=config_dict.get('moe_layer_frequency', 1),
            max_position_embeddings=config_dict.get('max_position_embeddings', 4096),
            tokenizer=tokenizer,
            router=router,
            expert=expert,
            attention=attention,
        )
    
    def summary(self) -> str:
        """Get human-readable configuration summary."""
        lines = [
            f"=" * 60,
            f"Model Configuration: {self.model_name}",
            f"=" * 60,
            f"",
            f"Architecture:",
            f"  Type: {self.model_type.value.upper()}",
            f"  Stage: {self.stage}",
            f"  Hidden Size: {self.hidden_size}",
            f"  Layers: {self.num_layers} ({self.num_moe_layers} MoE, {self.num_dense_layers} Dense)",
            f"  Attention Heads: {self.attention.num_attention_heads} (KV: {self.attention.num_kv_heads})",
            f"",
        ]
        
        if self.model_type == ModelType.MOE:
            lines.extend([
                f"MoE Configuration:",
                f"  Routed Experts: {self.num_routed_experts}",
                f"  Shared Experts: {self.num_shared_experts}",
                f"  Null Experts: {self.num_null_experts}",
                f"  Top-K: {self.router.top_k}",
                f"  Router Type: {self.router.router_type.value}",
                f"  Active Ratio: {self.active_expert_ratio:.1%}",
                f"",
            ])
        
        lines.extend([
            f"Parameters (Estimated):",
            f"  Total: {self.estimated_total_params / 1e9:.2f}B",
            f"  Active: {self.estimated_active_params / 1e9:.2f}B",
            f"",
            f"=" * 60,
        ])
        
        return "\n".join(lines)


# Convenience function to print config
def print_config(config: MoEModelConfig):
    """Print configuration summary."""
    print(config.summary())
