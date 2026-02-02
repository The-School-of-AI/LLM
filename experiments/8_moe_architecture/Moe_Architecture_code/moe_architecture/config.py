"""
MoE Architecture Configuration
==============================

This module defines configurations for all 4 stages of the growth cadence:
- Stage 1: 1B Dense (Foundation)
- Stage 2: 3B MoE-8 (Learn Routing)
- Stage 3: 8B MoE-8 (Scale Dimensions)
- Stage 4: 70B MoE-64 (Expert Expansion)

Configuration Philosophy:
-------------------------
All configurations follow mathematical principles derived from:
1. DeepSeek-V2/V3 architecture papers
2. GSA (Gated Sparse Attention) paper insights
3. Parameter budget mathematics (see moe_expert_count_mathematics.md)

Key Formulas:
- Top-K: K = 0.5 × √N (DeepSeek empirical finding)
- Shared Experts: N_shared = α × K where α ∈ [0.5, 1.0]
- Null Experts: N_null = ⌈junk_rate × K / target_utilization⌉
- Expert Count: N = 2^round(log2(param_ratio × 4))

Team Integration:
- Team 6: Tokenizer constraints, ID bands, special token registry
- Team 7: Null-routing telemetry, routing health gates, loss-free control

Author: Team 8 - MoE Architecture
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum
import math


# =============================================================================
# Enumerations
# =============================================================================

class ModelStage(Enum):
    """Growth cadence stages."""
    DENSE_1B = "1b_dense"
    MOE_3B = "3b_moe"
    MOE_8B = "8b_moe"
    MOE_70B = "70b_moe"


class RouterType(Enum):
    """Router architecture types."""
    GSA_GATED = "gsa_gated"        # GSA-inspired gated lightning router
    DEEPSEEK_V3 = "deepseek_v3"   # DeepSeek-V3 style sigmoid + bias
    STANDARD_TOPK = "standard"     # Standard softmax top-k


class ExpertType(Enum):
    """Expert types in the MoE block."""
    SHARED = "shared"      # Always active experts
    ROUTED = "routed"      # Selected by router
    NULL = "null"          # Zero-compute absorber


class LoadBalanceStrategy(Enum):
    """Load balancing strategies."""
    LOSS_FREE_BIAS = "loss_free_bias"  # Bias-only adjustment (recommended)
    AUX_LOSS = "aux_loss"              # Traditional auxiliary loss
    HYBRID = "hybrid"                   # Tiny aux loss + bias (safety net)


# =============================================================================
# Team Integration Configurations
# =============================================================================

@dataclass
class Team6TokenizerConfig:
    """
    Team 6 Integration: Tokenizer Constraints
    
    Defines token ID bands and special token registry for proper
    null routing decisions and telemetry classification.
    """
    vocab_size: int = 32000
    
    # Special token IDs
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    
    # Token ID bands for classification
    # These help the router identify token types for null routing
    special_token_range: Tuple[int, int] = (0, 10)      # [PAD], [BOS], [EOS], etc.
    punctuation_range: Tuple[int, int] = (10, 100)      # Punctuation marks
    digit_range: Tuple[int, int] = (100, 150)           # Digits 0-9 and variants
    common_word_range: Tuple[int, int] = (150, 1000)    # Most frequent words
    code_keyword_range: Tuple[int, int] = (1000, 1500)  # def, class, return, etc.
    
    # Token classification for telemetry
    junk_token_ids: List[int] = field(default_factory=lambda: [0])  # Tokens to route to null
    signal_token_ids: List[int] = field(default_factory=list)       # Important tokens
    
    # Maximum sequence length
    max_seq_length: int = 4096
    
    def is_junk_token(self, token_id: int) -> bool:
        """Check if token should be considered junk (null-routable)."""
        if token_id in self.junk_token_ids:
            return True
        if self.special_token_range[0] <= token_id < self.special_token_range[1]:
            return True
        return False
    
    def get_token_class(self, token_id: int) -> str:
        """Get token classification for telemetry."""
        if token_id in self.junk_token_ids or token_id == self.pad_token_id:
            return "JUNK"
        if self.special_token_range[0] <= token_id < self.special_token_range[1]:
            return "SPECIAL"
        if self.punctuation_range[0] <= token_id < self.punctuation_range[1]:
            return "PUNCTUATION"
        if self.code_keyword_range[0] <= token_id < self.code_keyword_range[1]:
            return "CODE"
        return "GENERAL"


@dataclass
class Team7TelemetryConfig:
    """
    Team 7 Integration: Null-Routing Telemetry & Health Gates
    
    Defines telemetry collection, health monitoring thresholds,
    and loss-free control plugin interface.
    """
    # Telemetry collection
    enabled: bool = True
    log_interval: int = 100          # Log every N steps
    detailed_logging: bool = False   # Per-token routing logs
    
    # Null routing targets (from spec)
    junk_null_target_min: float = 0.60    # Minimum 60% junk → null
    junk_null_target_max: float = 0.80    # Maximum 80% junk → null
    signal_null_target_max: float = 0.10  # Maximum 10% signal → null
    
    # Health gate thresholds
    dead_expert_threshold: float = 0.01   # Expert is "dead" if < 1% utilization
    overload_threshold: float = 3.0       # Expert overloaded if > 3× expected
    entropy_min_threshold: float = 0.70   # Normalized entropy should be > 0.7
    gini_max_threshold: float = 0.50      # Gini coefficient should be < 0.5
    
    # Load balance control
    bias_update_speed: float = 0.001      # γ from DeepSeek-V3
    bias_min: float = -2.0
    bias_max: float = 2.0
    
    # Alerts and actions
    alert_on_dead_expert: bool = True
    alert_on_collapse: bool = True
    auto_revive_dead_experts: bool = True
    
    # Plugin interface hooks
    pre_routing_hook: Optional[str] = None   # Module path for pre-routing plugin
    post_routing_hook: Optional[str] = None  # Module path for post-routing plugin


@dataclass  
class ComputeBudgetConfig:
    """
    Compute Budget Constraints
    
    Hard ceilings to ensure architecture fits compute budget
    and coordinates with training teams.
    """
    # FLOPs budget (approximate)
    max_flops_per_token: Optional[int] = None  # Set by training team
    
    # Memory constraints
    max_memory_gb: Optional[float] = None      # GPU memory ceiling
    activation_checkpointing: bool = True      # Trade compute for memory
    
    # Batch size constraints
    min_batch_size: int = 1
    max_batch_size: int = 256
    
    # Sequence length
    max_seq_length: int = 4096
    
    # Expert computation budget
    max_experts_per_token: int = 8    # Hard ceiling on active experts
    
    # Training constraints
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"     # bf16, fp16, fp32


# =============================================================================
# Router Configuration
# =============================================================================

@dataclass
class GSARouterConfig:
    """
    GSA-Inspired Gated Lightning Router Configuration
    
    Based on GSA paper (arXiv:2601.15305v1) and DeepSeek-V3.
    
    Key innovations:
    1. Multi-head routing (H_I heads, each with d_I dimension)
    2. Sigmoid activations (bounded scores)
    3. Query-dependent head weights
    4. Adaptive top-k based on score variance
    5. Loss-free load balancing via bias
    
    Mathematical basis:
    - Score: I_{t,e} = Σⱼ σ(h_t W_j^w) · σ(q_{t,j} · k_e + b_j)
    - Bounded in (0, num_heads) for stability
    """
    router_type: RouterType = RouterType.GSA_GATED
    
    # Multi-head configuration (from GSA paper Table 1)
    num_router_heads: int = 4         # H_I: Number of indexer heads
    router_dim: int = 64              # d_I: Low-dimensional projection
    
    # Top-k configuration
    top_k: int = 4                    # Base number of active experts
    top_k_min: int = 1                # Minimum for adaptive k
    top_k_max: int = 8                # Maximum for adaptive k
    use_adaptive_k: bool = True       # Enable variance-based adaptive k
    
    # Gating configuration
    use_sigmoid_gating: bool = True   # Use sigmoid (True) vs softmax (False)
    gate_temperature: float = 1.0     # Temperature for gating
    
    # Bias configuration (loss-free load balancing)
    initial_bias: float = 0.0         # Initial expert bias
    null_bias_init: float = 0.1       # Slightly higher for null experts
    bias_update_speed: float = 0.001  # γ from DeepSeek-V3
    bias_clamp_min: float = -2.0
    bias_clamp_max: float = 2.0
    
    # Adaptive sparsity (from GSA paper Section 3.4)
    variance_ema_decay: float = 0.99  # EMA decay for variance tracking
    
    # Initialization
    router_init_std: float = 0.02     # Std for weight initialization
    
    # Auxiliary loss (optional safety net)
    use_aux_loss: bool = False        # Disable by default (loss-free)
    aux_loss_weight: float = 0.0001   # α from DeepSeek-V3 (tiny if used)


# =============================================================================
# Expert Configuration
# =============================================================================

@dataclass
class ExpertConfig:
    """
    Expert FFN Configuration
    
    Supports three types:
    1. Routed experts: Selected by router, specialized processing
    2. Shared experts: Always active, common pattern handling
    3. Null experts: Zero-compute, junk absorption
    
    All experts use SwiGLU activation with optional dual gating (G1+G2)
    from GSA paper for collapse prevention.
    """
    # Expert counts
    num_routed_experts: int = 40       # Selected by router
    num_shared_experts: int = 2       # Always active
    num_null_experts: int = 1         # Zero-compute
    
    # Expert dimensions
    intermediate_size: int = 512     # FFN intermediate dimension
    
    # Dual gating (from GSA paper Section 3.5)
    use_dual_gating: bool = False      # Enable G1 (output) + G2 (input) gates
    gate_bias_init: float = 0.0       # Initialize for σ(·) ≈ 0.5
    
    # Expert initialization
    expert_init_std: float = 0.02
    
    # Null expert configuration
    null_scale_init: float = 0.001    # Tiny scale for gradient flow
    
    @property
    def total_experts(self) -> int:
        """Total experts in routing pool (routed + null, NOT shared)."""
        return self.num_routed_experts + self.num_null_experts
    
    @property
    def all_experts(self) -> int:
        """All experts including shared."""
        return self.num_routed_experts + self.num_shared_experts + self.num_null_experts


# =============================================================================
# Attention Configuration  
# =============================================================================

@dataclass
class AttentionConfig:
    """
    Multi-Head Attention Configuration with GQA
    
    Uses Grouped-Query Attention (GQA) for efficient KV cache.
    GQA ratio determines how many query heads share each KV head.
    """
    num_attention_heads: int = 16     # Query heads
    num_kv_heads: int = 4             # Key-Value heads (GQA)
    head_dim: int = 128               # Dimension per head
    
    # Attention settings
    attention_dropout: float = 0.0
    use_flash_attention: bool = True  # Use FlashAttention-2 if available
    
    # Positional encoding
    rope_theta: float = 10000.0       # RoPE base frequency
    rope_scaling: Optional[Dict] = None  # For extended context
    
    # Sliding window (optional)
    sliding_window: Optional[int] = None
    
    @property
    def gqa_ratio(self) -> int:
        """Number of query heads per KV head."""
        return self.num_attention_heads // self.num_kv_heads


# =============================================================================
# Main Model Configuration
# =============================================================================

@dataclass
class MoEModelConfig:
    """
    Complete MoE Model Configuration
    
    Combines all sub-configurations into a unified model config.
    Includes helper methods for validation and scaling.
    """
    # Model identification
    model_name: str = "moe_model"
    stage: ModelStage = ModelStage.MOE_3B
    
    # Core dimensions
    hidden_size: int = 2048
    num_layers: int = 20
    vocab_size: int = 32000
    
    # Sub-configurations
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    router: GSARouterConfig = field(default_factory=GSARouterConfig)
    
    # Team integrations
    tokenizer: Team6TokenizerConfig = field(default_factory=Team6TokenizerConfig)
    telemetry: Team7TelemetryConfig = field(default_factory=Team7TelemetryConfig)
    compute_budget: ComputeBudgetConfig = field(default_factory=ComputeBudgetConfig)
    
    # MoE layer placement
    moe_layer_freq: int = 1           # MoE every N layers (1 = all layers)
    first_moe_layer: int = 0          # First layer with MoE (0-indexed)
    
    # Model settings
    hidden_dropout: float = 0.0
    embedding_dropout: float = 0.0
    
    # Normalization
    rms_norm_eps: float = 1e-6
    
    # Initialization
    initializer_range: float = 0.02
    
    # Training settings
    tie_word_embeddings: bool = False
    use_cache: bool = True
    
    # Load balancing
    load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.LOSS_FREE_BIAS
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
        self._compute_derived_values()
    
    def _validate(self):
        """Validate configuration values."""
        assert self.hidden_size > 0, "hidden_size must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert self.hidden_size % self.attention.num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads"
        assert self.attention.num_attention_heads % self.attention.num_kv_heads == 0, \
            "num_attention_heads must be divisible by num_kv_heads"
        
        # Validate expert configuration
        if self.is_moe:
            assert self.expert.num_routed_experts > 0, "MoE requires routed experts"
            assert self.router.top_k <= self.expert.total_experts, \
                "top_k cannot exceed total routable experts"
    
    def _compute_derived_values(self):
        """Compute derived configuration values."""
        # Ensure head_dim matches hidden_size / num_heads
        expected_head_dim = self.hidden_size // self.attention.num_attention_heads
        if self.attention.head_dim != expected_head_dim:
            self.attention.head_dim = expected_head_dim
    
    @property
    def is_moe(self) -> bool:
        """Check if model uses MoE."""
        return self.expert.num_routed_experts > 0
    
    @property
    def num_moe_layers(self) -> int:
        """Number of MoE layers in the model."""
        if not self.is_moe:
            return 0
        return len([i for i in range(self.first_moe_layer, self.num_layers, self.moe_layer_freq)])
    
    @property
    def num_dense_layers(self) -> int:
        """Number of dense (non-MoE) layers."""
        return self.num_layers - self.num_moe_layers
    
    def get_layer_type(self, layer_idx: int) -> str:
        """Get layer type (moe or dense) for given index."""
        if not self.is_moe:
            return "dense"
        if layer_idx < self.first_moe_layer:
            return "dense"
        if (layer_idx - self.first_moe_layer) % self.moe_layer_freq == 0:
            return "moe"
        return "dense"
    
    def estimate_parameters(self) -> Dict[str, int]:
        """Estimate parameter count by component."""
        # Embedding parameters
        embed_params = self.vocab_size * self.hidden_size
        if not self.tie_word_embeddings:
            embed_params *= 2  # Input + output embeddings
        
        # Attention parameters per layer (with GQA)
        q_params = self.hidden_size * self.hidden_size
        kv_params = self.hidden_size * (self.hidden_size // self.attention.gqa_ratio) * 2
        o_params = self.hidden_size * self.hidden_size
        attn_per_layer = q_params + kv_params + o_params
        
        # FFN parameters per expert (SwiGLU: W1, W2, W3)
        expert_params = 3 * self.hidden_size * self.expert.intermediate_size
        
        # Gating parameters if enabled
        if self.expert.use_dual_gating:
            expert_params += 2 * self.hidden_size * self.hidden_size
        
        # Dense layer FFN
        dense_ffn_params = expert_params
        
        # MoE layer parameters
        shared_params = self.expert.num_shared_experts * expert_params
        routed_params = self.expert.num_routed_experts * expert_params
        router_params = self.hidden_size * self.expert.total_experts  # Approximate
        moe_params = shared_params + routed_params + router_params
        
        # Layer norm parameters (2 per layer: attention + FFN)
        norm_per_layer = 2 * self.hidden_size
        
        # Total by component
        total_attn = self.num_layers * attn_per_layer
        total_dense_ffn = self.num_dense_layers * dense_ffn_params
        total_moe = self.num_moe_layers * moe_params
        total_norm = self.num_layers * norm_per_layer
        
        return {
            "embeddings": embed_params,
            "attention": total_attn,
            "dense_ffn": total_dense_ffn,
            "moe_shared": self.num_moe_layers * shared_params,
            "moe_routed": self.num_moe_layers * routed_params,
            "moe_router": self.num_moe_layers * router_params,
            "layer_norm": total_norm,
            "total": embed_params + total_attn + total_dense_ffn + total_moe + total_norm
        }
    
    def estimate_active_parameters(self) -> int:
        """Estimate active parameters per forward pass."""
        params = self.estimate_parameters()
        
        # Active = embeddings + attention + norms + active FFN
        active = params["embeddings"] + params["attention"] + params["layer_norm"]
        
        # Dense layers: full FFN active
        expert_size = 3 * self.hidden_size * self.expert.intermediate_size
        if self.expert.use_dual_gating:
            expert_size += 2 * self.hidden_size * self.hidden_size
        active += self.num_dense_layers * expert_size
        
        # MoE layers: shared + top_k experts
        active_experts = self.expert.num_shared_experts + self.router.top_k
        active += self.num_moe_layers * active_experts * expert_size
        
        return active


# =============================================================================
# Pre-defined Configurations for Each Stage
# =============================================================================

def get_1b_dense_config() -> MoEModelConfig:
    """
    Stage 1: 1B Dense Foundation Model
    
    Purpose: Establish foundation weights for MoE expansion
    Architecture: Standard transformer with SwiGLU FFN
    """
    return MoEModelConfig(
        model_name="moe_1b_dense",
        stage=ModelStage.DENSE_1B,
        
        # Core dimensions
        hidden_size=2048,
        num_layers=20,
        vocab_size=32000,
        
        # Attention (GQA 4:1)
        attention=AttentionConfig(
            num_attention_heads=16,
            num_kv_heads=4,
            head_dim=128,
        ),
        
        # No MoE (dense model)
        expert=ExpertConfig(
            num_routed_experts=0,
            num_shared_experts=0,
            num_null_experts=0,
            intermediate_size=512,
            use_dual_gating=False,
        ),
        
        # Router not used
        router=GSARouterConfig(top_k=0),
        
        # MoE disabled
        moe_layer_freq=999,  # Effectively never
    )


def get_3b_moe_config() -> MoEModelConfig:
    """
    Stage 2: 3B MoE-8 (Learn Routing)
    
    Purpose: Learn expert routing with 8 experts
    Transition: 1B Dense → 3B MoE (Expert Explosion 1→8)
    
    Mathematical basis:
    - N_experts = 8 (minimum for meaningful specialization)
    - N_shared = 2 (25% of effective capacity)
    - N_null = 1 (⌈0.20 × 2 / 0.6⌉)
    - Top-K = 2 (√8 ≈ 2.8 → 2)
    """
    return MoEModelConfig(
        model_name="moe_3b_moe8",
        stage=ModelStage.MOE_3B,
        
        # Core dimensions (same as 1B)
        hidden_size=2048,
        num_layers=20,
        vocab_size=32000,
        
        # Attention (same as 1B)
        attention=AttentionConfig(
            num_attention_heads=16,
            num_kv_heads=4,
            head_dim=128,
        ),
        
        # Expert configuration
        expert=ExpertConfig(
            num_routed_experts=40,
            num_shared_experts=2,
            num_null_experts=1,
            intermediate_size=512,
            use_dual_gating=False,
        ),
        
        # GSA Router
        router=GSARouterConfig(
            router_type=RouterType.GSA_GATED,
            num_router_heads=4,
            router_dim=512,
            top_k=8,
            top_k_min=1,
            top_k_max=8,
            use_adaptive_k=True,
            use_sigmoid_gating=True,
            null_bias_init=0.1,
        ),
        
        # MoE on every layer
        moe_layer_freq=1,
        
        # Load balancing
        load_balance_strategy=LoadBalanceStrategy.LOSS_FREE_BIAS,
    )


def get_8b_moe_config() -> MoEModelConfig:
    """
    Stage 3: 8B MoE-8 (Scale Dimensions)
    
    Purpose: Scale model capacity while preserving routing
    Transition: 3B MoE → 8B MoE (Dimension scaling, same 8 experts)
    
    Key changes from 3B:
    - hidden_size: 2048
    - num_layers: 20 → 40 (2×)
    - intermediate_size: 512
    - Experts stay at 40 (routing preserved!)
    """
    return MoEModelConfig(
        model_name="moe_8b_moe8",
        stage=ModelStage.MOE_8B,
        
        # Core dimensions (2× scale)
        hidden_size=2048,
        num_layers=40,
        vocab_size=32000,
        
        # Attention (2× heads for wider hidden)
        attention=AttentionConfig(
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
        ),
        
        # Expert configuration (SAME expert count, larger dims)
        expert=ExpertConfig(
            num_routed_experts=40,     # Same as 3B!
            num_shared_experts=2,     # Same as 3B!
            num_null_experts=1,       # Same as 3B!
            intermediate_size=512,  # 2× larger
            use_dual_gating=False,
        ),
        
        # GSA Router (scaled for larger hidden)
        router=GSARouterConfig(
            router_type=RouterType.GSA_GATED,
            num_router_heads=4,
            router_dim=512,           # 2× for larger hidden
            top_k=8,                  # Same as 3B!
            top_k_min=1,
            top_k_max=4,
            use_adaptive_k=True,
            use_sigmoid_gating=True,
            null_bias_init=0.1,
        ),
        
        # MoE on every layer
        moe_layer_freq=1,
        
        # Load balancing
        load_balance_strategy=LoadBalanceStrategy.LOSS_FREE_BIAS,
    )


def get_70b_moe_config() -> MoEModelConfig:
    """
    Stage 4: 70B MoE-64 (Expert Expansion)
    
    Purpose: Fine-grained specialization with 64 experts
    Transition: 8B MoE-8 → 70B MoE-64 (Expert Expansion 8→64)
    
    Mathematical basis:
    - N_experts = 64 (8 parents × 8 children)
    - N_shared = 4 (6.25% of routed)
    - N_null = 2 (⌈0.20 × 4 / 0.6⌉)
    - Top-K = 4 (0.5 × √64 = 4)
    
    Key changes from 8B:
    - Experts: 512 
    - num_layers: 40
    - Top-K: 2 → 4
    """
    return MoEModelConfig(
        model_name="moe_70b_moe64",
        stage=ModelStage.MOE_70B,
        
        # Core dimensions (same hidden as 8B)
        hidden_size=2048,
        num_layers=40,
        vocab_size=32000,
        
        # Attention (same as 8B)
        attention=AttentionConfig(
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
        ),
        
        # Expert configuration (8× expert expansion!)
        expert=ExpertConfig(
            num_routed_experts=512,    # 8× expansion
            num_shared_experts=4,     # 2× for larger pool
            num_null_experts=1,       # 2 for more junk absorption
            intermediate_size=512,
            use_dual_gating=False,
        ),
        
        # GSA Router for 64 experts
        router=GSARouterConfig(
            router_type=RouterType.GSA_GATED,
            num_router_heads=4,
            router_dim=128,
            top_k=4,                  # 2× for more experts
            top_k_min=2,
            top_k_max=6,
            use_adaptive_k=True,
            use_sigmoid_gating=True,
            null_bias_init=0.05,      # Lower for larger pool
        ),
        
        # MoE on every layer
        moe_layer_freq=1,
        
        # Load balancing with safety net at scale
        load_balance_strategy=LoadBalanceStrategy.HYBRID,
        
        # Extended context for 70B
        compute_budget=ComputeBudgetConfig(
            max_seq_length=8192,
            activation_checkpointing=True,
            mixed_precision="bf16",
        ),
        
        # Telemetry more critical at scale
        telemetry=Team7TelemetryConfig(
            enabled=True,
            log_interval=50,
            detailed_logging=True,
        ),
    )


# =============================================================================
# Configuration Registry
# =============================================================================

CONFIG_REGISTRY = {
    "1b_dense": get_1b_dense_config,
    "3b_moe": get_3b_moe_config,
    "8b_moe": get_8b_moe_config,
    "70b_moe": get_70b_moe_config,
}


def get_config(name: str) -> MoEModelConfig:
    """Get configuration by name."""
    if name not in CONFIG_REGISTRY:
        available = list(CONFIG_REGISTRY.keys())
        raise ValueError(f"Unknown config '{name}'. Available: {available}")
    return CONFIG_REGISTRY[name]()


def print_config_summary(config: MoEModelConfig):
    """Print a summary of the configuration."""
    params = config.estimate_parameters()
    active = config.estimate_active_parameters()
    
    print(f"\n{'='*60}")
    print(f" {config.model_name.upper()} Configuration Summary")
    print(f"{'='*60}")
    print(f"\n📊 Model Architecture:")
    print(f"   Stage: {config.stage.value}")
    print(f"   Hidden Size: {config.hidden_size}")
    print(f"   Num Layers: {config.num_layers} ({config.num_moe_layers} MoE, {config.num_dense_layers} Dense)")
    print(f"   Vocab Size: {config.vocab_size}")
    
    print(f"\n🔍 Attention:")
    print(f"   Query Heads: {config.attention.num_attention_heads}")
    print(f"   KV Heads: {config.attention.num_kv_heads} (GQA ratio: {config.attention.gqa_ratio}:1)")
    print(f"   Head Dim: {config.attention.head_dim}")
    
    if config.is_moe:
        print(f"\n🎯 MoE Configuration:")
        print(f"   Routed Experts: {config.expert.num_routed_experts}")
        print(f"   Shared Experts: {config.expert.num_shared_experts}")
        print(f"   Null Experts: {config.expert.num_null_experts}")
        print(f"   Top-K: {config.router.top_k}")
        print(f"   Dual Gating: {config.expert.use_dual_gating}")
        print(f"   Router Type: {config.router.router_type.value}")
    
    print(f"\n📈 Parameter Estimates:")
    print(f"   Total Parameters: {params['total'] / 1e9:.2f}B")
    print(f"   Active Parameters: {active / 1e9:.2f}B")
    print(f"   Active Ratio: {active / params['total'] * 100:.1f}%")
    
    print(f"\n   Breakdown:")
    for key, value in params.items():
        if key != 'total':
            print(f"     {key}: {value / 1e6:.1f}M")
    
    print(f"\n{'='*60}\n")


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    # Print summaries for all configurations
    for name in CONFIG_REGISTRY:
        config = get_config(name)
        print_config_summary(config)
