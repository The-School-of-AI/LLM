"""
LLM Architecture Configuration
==============================

Centralized configuration for all model components.
Supports dynamic component selection via config flags.

Target: 1B Parameter Model
Inspired by: Qwen3 1.7B, SmolLM2, LLaMA 3, DeepSeek V3
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, List, Dict, Any
from enum import Enum
import json
import yaml
from pathlib import Path


class AttentionType(Enum):
    """Available attention mechanisms."""
    GROUPED_QUERY = "grouped_query"      # GQA (default, like Qwen3/LLaMA3)
    GATED_SPARSE = "gated_sparse"        # GSA from paper 2601.15305v1 (original implementation)
    DEEPSEEK_GSA = "deepseek_gsa"        # DeepSeek-style GSA (corrected implementation)
    DEEPSEEK_SPARSE = "deepseek_sparse"  # DeepSeek V3 MLA


class PositionEmbeddingType(Enum):
    """Position embedding types."""
    ROPE = "rope"           # Standard RoPE
    YARN = "yarn"           # YaRN for extended context
    ALIBI = "alibi"         # ALiBi (alternative)


class FFNType(Enum):
    """Feed-forward network types."""
    SWIGLU = "swiglu"       # SwiGLU (default)
    GELU = "gelu"           # Standard GELU
    MOE = "moe"             # Mixture of Experts


class ConnectionType(Enum):
    """Layer connection types."""
    RESIDUAL = "residual"   # Standard residual
    MHC = "mhc"             # Manifold Hyper-Connections (2512.24880)


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms."""
    
    # Attention type selection
    attention_type: AttentionType = AttentionType.GROUPED_QUERY
    
    # Common attention params
    num_attention_heads: int = 16
    num_key_value_heads: int = 4  # For GQA, set equal to num_attention_heads for MHA
    head_dim: int = 64
    attention_dropout: float = 0.0
    attention_bias: bool = False  # Modern LLMs don't use bias
    
    # GSA specific (Gated Sparse Attention - paper 2601.15305v1)
    gsa_indexer_dim: int = 64          # d_I: Low-dim indexer projection
    gsa_num_indexer_heads: int = 4     # H_I: Number of indexer heads
    gsa_k_base: int = 2048             # Base selection budget
    gsa_k_min: int = 256               # Minimum k (high confidence)
    gsa_k_max: int = 4096              # Maximum k (low confidence)

    # DeepSeek GSA specific (corrected implementation)
    gsa_use_adaptive_k: bool = True              # Enable adaptive k selection
    gsa_adaptive_k_method: str = "variance"      # "variance", "entropy", or "learned"
    gsa_adaptive_k_temperature: float = 1.0      # Temperature for adaptive scaling
    gsa_use_value_gate: bool = True              # Enable G2 (value gate)
    gsa_use_output_gate: bool = True             # Enable G1 (output gate)
    gsa_gate_activation: str = "sigmoid"         # Gate activation function
    gsa_gate_bias_init: float = 0.5              # Initial gate bias
    gsa_indexer_activation: str = "sigmoid"      # Indexer activation ("sigmoid" or "relu")
    gsa_use_triton_kernels: bool = True          # Use Triton kernels for long sequences (if available)

    # DeepSeek Sparse Attention specific
    ds_compressed_dim: int = 512      # Compressed KV dimension
    ds_rope_head_dim: int = 32        # RoPE dimension for decoupled attention
    ds_num_shared_experts: int = 1
    ds_q_lora_rank: int = 0           # 0 = no LoRA compression


@dataclass
class PositionConfig:
    """Configuration for position embeddings."""
    
    position_type: PositionEmbeddingType = PositionEmbeddingType.ROPE
    
    # RoPE params
    rope_theta: float = 10000.0
    rope_scaling_factor: float = 1.0
    
    # YaRN specific params (for extended context)
    yarn_scale: float = 1.0
    yarn_original_max_position: int = 4096
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_mscale: float = 1.0
    yarn_mscale_all_dim: float = 0.0


@dataclass
class FFNConfig:
    """Configuration for feed-forward networks."""
    
    ffn_type: FFNType = FFNType.SWIGLU
    intermediate_size: int = 4096  # Usually 4x hidden_size for SwiGLU: 8/3 * hidden
    ffn_dropout: float = 0.0
    ffn_bias: bool = False
    
    # MoE specific
    moe_num_experts: int = 8
    moe_num_experts_per_tok: int = 2
    moe_aux_loss_coef: float = 0.01


@dataclass
class ConnectionConfig:
    """
    Configuration for layer connections.
    
    For mHC (Manifold-Constrained Hyper-Connections from paper 2512.24880):
    - mhc_expansion_rate (n): Number of parallel streams (default 4)
    - mhc_alpha_init: Initial value for gating factors (default 0.01)
    - mhc_sinkhorn_iters: Iterations for doubly stochastic projection (default 20)
    
    Parameter overhead per mHC module: ~nC(2n + n²) + constants
    For n=4, C=2048: ~205K params/module, ~410K/layer, ~9.8M total (24 layers)
    This is <1% overhead for a 1B model!
    """
    
    connection_type: ConnectionType = ConnectionType.RESIDUAL
    
    # mHC parameters (from DeepSeek paper 2512.24880v2)
    mhc_expansion_rate: int = 4        # n: number of streams (paper uses 4)
    mhc_alpha_init: float = 0.01       # α: gating factor init (paper uses 0.01)
    mhc_sinkhorn_iters: int = 20       # Sinkhorn-Knopp iterations (paper uses 20)


@dataclass
class HeadConfig:
    """
    Configuration for output heads.

    Design (following DeepSeek V3, arXiv:2412.19437):
    - Untied embeddings: Input and output embeddings are always separate
    - This ensures consistent behavior as model scales (FFN grows, head stays same)
    - Better quality than tied weights, especially for larger models
    """

    # Multi-token prediction (DeepSeek style)
    use_multi_token_prediction: bool = False
    num_predict_tokens: int = 1  # >1 enables multi-token prediction
    mtp_loss_weight: float = 0.3  # Weight for auxiliary MTP loss


@dataclass
class ModelConfig:
    """
    Complete model configuration.
    
    Default: ~1B parameter dense model similar to Qwen3/SmolLM2
    """
    
    # Model identification
    model_name: str = "LLM-1B-Base"
    model_version: str = "1.0.0"
    
    # Core architecture
    vocab_size: int = 128000  # Divisible by 64 for efficiency
    hidden_size: int = 2048
    num_hidden_layers: int = 16
    max_position_embeddings: int = 4096
    
    # Normalization
    rms_norm_eps: float = 1e-6
    use_pre_norm: bool = True  # Pre-LayerNorm (modern standard)
    
    # Initialization
    initializer_range: float = 0.02
    
    # Dropout
    hidden_dropout: float = 0.0
    
    # Component configs
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    position: PositionConfig = field(default_factory=PositionConfig)
    ffn: FFNConfig = field(default_factory=FFNConfig)
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    
    # Precision
    dtype: str = "bfloat16"  # bfloat16, float16, float32
    
    def __post_init__(self):
        """Validate and adjust configuration."""
        # Ensure head_dim consistency
        if self.attention.head_dim * self.attention.num_attention_heads != self.hidden_size:
            self.attention.head_dim = self.hidden_size // self.attention.num_attention_heads
            
        # Ensure intermediate_size is set properly for SwiGLU
        if self.ffn.ffn_type == FFNType.SWIGLU and self.ffn.intermediate_size == 0:
            # SwiGLU optimal: hidden_size * 8/3, rounded to multiple of 256
            self.ffn.intermediate_size = int(self.hidden_size * 8 / 3)
            self.ffn.intermediate_size = ((self.ffn.intermediate_size + 255) // 256) * 256
            
    @property
    def num_parameters(self) -> int:
        """Estimate total parameters."""
        # Embedding
        embed_params = self.vocab_size * self.hidden_size
        
        # Per layer
        # Attention: Q, K, V, O projections
        attn_params = 4 * self.hidden_size * self.hidden_size
        # FFN: gate, up, down for SwiGLU
        ffn_params = 3 * self.hidden_size * self.ffn.intermediate_size
        # Norms
        norm_params = 2 * self.hidden_size
        
        layer_params = attn_params + ffn_params + norm_params
        total_layer_params = layer_params * self.num_hidden_layers
        
        # LM head (tied with embeddings if enabled)
        head_params = 0 if self.head.tie_word_embeddings else embed_params
        
        # Final norm
        final_norm = self.hidden_size
        
        return embed_params + total_layer_params + head_params + final_norm
    
    @property
    def num_parameters_billions(self) -> float:
        """Parameters in billions."""
        return self.num_parameters / 1e9
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        def enum_to_str(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, dict):
                return {k: enum_to_str(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [enum_to_str(item) for item in obj]
            elif hasattr(obj, '__dataclass_fields__'):
                return {k: enum_to_str(v) for k, v in obj.__dict__.items()}
            return obj
            
        return enum_to_str(self.__dict__)
    
    def save(self, path: str):
        """Save configuration to file."""
        path = Path(path)
        data = self.to_dict()
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
            
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary."""
        # Convert string enums back
        if 'attention' in data:
            if 'attention_type' in data['attention']:
                data['attention']['attention_type'] = AttentionType(data['attention']['attention_type'])
            data['attention'] = AttentionConfig(**data['attention'])
            
        if 'position' in data:
            if 'position_type' in data['position']:
                data['position']['position_type'] = PositionEmbeddingType(data['position']['position_type'])
            data['position'] = PositionConfig(**data['position'])
            
        if 'ffn' in data:
            if 'ffn_type' in data['ffn']:
                data['ffn']['ffn_type'] = FFNType(data['ffn']['ffn_type'])
            data['ffn'] = FFNConfig(**data['ffn'])
            
        if 'connection' in data:
            if 'connection_type' in data['connection']:
                data['connection']['connection_type'] = ConnectionType(data['connection']['connection_type'])
            data['connection'] = ConnectionConfig(**data['connection'])
            
        if 'head' in data:
            data['head'] = HeadConfig(**data['head'])
            
        return cls(**data)
    
    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        """Load configuration from file."""
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
            
        return cls.from_dict(data)


# =============================================================================
# Preset Configurations
# =============================================================================

def get_1b_base_config() -> ModelConfig:
    """1B base model - dense, standard architecture."""
    return ModelConfig(
        model_name="LLM-1B-Base",
        vocab_size=128000,
        hidden_size=2048,
        num_hidden_layers=16,
        max_position_embeddings=4096,
        attention=AttentionConfig(
            attention_type=AttentionType.GROUPED_QUERY,
            num_attention_heads=16,
            num_key_value_heads=4,
        ),
        position=PositionConfig(
            position_type=PositionEmbeddingType.YARN,
            yarn_original_max_position=4096,
            yarn_scale=8.0,
        ),
        ffn=FFNConfig(
            ffn_type=FFNType.SWIGLU,
            intermediate_size=4096,  # ~2.7x hidden for SwiGLU
        ),
    )


def get_1b_gsa_config() -> ModelConfig:
    """1B model with Gated Sparse Attention (paper 2601.15305v1)."""
    config = get_1b_base_config()
    config.model_name = "LLM-1B-GSA"
    config.attention.attention_type = AttentionType.GATED_SPARSE
    # Indexer parameters (Table 1 in paper)
    config.attention.gsa_indexer_dim = 64       # d_I
    config.attention.gsa_num_indexer_heads = 4  # H_I
    # Adaptive sparsity parameters
    config.attention.gsa_k_base = 2048          # Base selection budget
    config.attention.gsa_k_min = 256            # Min k (confident)
    config.attention.gsa_k_max = 4096           # Max k (uncertain)
    return config


def get_1b_deepseek_gsa_config() -> ModelConfig:
    """
    1B model with DeepSeek-style GSA (corrected implementation).

    Default k values are tuned for CUDA GPUs with 40GB+ VRAM.
    For MPS or limited memory, override via CLI:
        --gsa-k-base 128 --gsa-k-max 256

    Memory scaling guide (for seq_length=4096):
    - k_base=256, k_max=512:  ~8GB VRAM per batch
    - k_base=512, k_max=1024: ~16GB VRAM per batch
    - k_base=1024, k_max=2048: ~32GB VRAM per batch
    """
    config = get_1b_base_config()
    config.model_name = "LLM-1B-DeepSeek-GSA"
    config.attention.attention_type = AttentionType.DEEPSEEK_GSA
    # Indexer parameters
    config.attention.gsa_indexer_dim = 64
    config.attention.gsa_num_indexer_heads = 4
    config.attention.gsa_indexer_activation = "sigmoid"
    # Adaptive sparsity - defaults tuned for CUDA with good VRAM
    # k values scale memory linearly: O(batch * seq * k * heads * head_dim)
    config.attention.gsa_k_base = 512   # Good balance for 40GB+ GPUs
    config.attention.gsa_k_min = 64     # Minimum tokens to attend to
    config.attention.gsa_k_max = 1024   # Cap for very long sequences
    config.attention.gsa_use_adaptive_k = True
    config.attention.gsa_adaptive_k_method = "variance"
    config.attention.gsa_adaptive_k_temperature = 1.0
    # Gating
    config.attention.gsa_use_value_gate = True
    config.attention.gsa_use_output_gate = True
    config.attention.gsa_gate_activation = "sigmoid"
    config.attention.gsa_gate_bias_init = 0.5
    return config


def get_1b_deepseek_config() -> ModelConfig:
    """1B model with DeepSeek V3 Sparse Attention."""
    config = get_1b_base_config()
    config.model_name = "LLM-1B-DeepSeek"
    config.attention.attention_type = AttentionType.DEEPSEEK_SPARSE
    config.attention.ds_compressed_dim = 512
    return config


def get_1b_mhc_config() -> ModelConfig:
    """1B model with Manifold Hyper-Connections."""
    config = get_1b_base_config()
    config.model_name = "LLM-1B-mHC"
    config.connection.connection_type = ConnectionType.MHC
    config.connection.mhc_expansion_rate = 4.0
    return config


def get_1b_mtp_config() -> ModelConfig:
    """1B model with Multi-Token Prediction."""
    config = get_1b_base_config()
    config.model_name = "LLM-1B-MTP"
    config.head.use_multi_token_prediction = True
    config.head.num_predict_tokens = 4
    return config


def get_1b_yarn_config() -> ModelConfig:
    """1B model with YaRN for extended context (32K)."""
    config = get_1b_base_config()
    config.model_name = "LLM-1B-YaRN"
    config.max_position_embeddings = 32768
    config.position.position_type = PositionEmbeddingType.YARN
    config.position.yarn_original_max_position = 4096
    config.position.yarn_scale = 8.0
    return config


def get_1b_deepseek_gsa_128k_config() -> ModelConfig:
    """
    1B model with DeepSeek GSA optimized for 128K context length.

    Uses Triton kernels by default for memory efficiency.
    YaRN extends 4K base to 128K with scale factor 32.

    Memory requirements (approximate):
    - Triton kernels: ~40GB VRAM for batch_size=1
    - PyTorch fallback: ~60GB+ VRAM (use Triton for long sequences)
    """
    config = get_1b_deepseek_gsa_config()
    config.model_name = "LLM-1B-DeepSeek-GSA-128K"
    config.max_position_embeddings = 131072  # 128K
    # YaRN configuration for 128K context
    config.position.position_type = PositionEmbeddingType.YARN
    config.position.yarn_original_max_position = 4096
    config.position.yarn_scale = 32.0  # 4K -> 128K
    config.position.yarn_beta_fast = 32.0
    config.position.yarn_beta_slow = 1.0
    config.position.yarn_mscale = 1.0
    # GSA tuned for long sequences - larger k for better quality
    config.attention.gsa_k_base = 1024
    config.attention.gsa_k_min = 128
    config.attention.gsa_k_max = 2048
    # Triton kernels required for memory efficiency at this scale
    config.attention.gsa_use_triton_kernels = True
    return config


def get_1b_deepseek_gsa_256k_config() -> ModelConfig:
    """
    1B model with DeepSeek GSA optimized for 256K context length.

    Uses Triton kernels by default for memory efficiency.
    YaRN extends 4K base to 256K with scale factor 64.

    Memory requirements (approximate):
    - Triton kernels: ~60GB+ VRAM for batch_size=1
    - Recommended: Use gradient checkpointing and small batch sizes
    """
    config = get_1b_deepseek_gsa_config()
    config.model_name = "LLM-1B-DeepSeek-GSA-256K"
    config.max_position_embeddings = 262144  # 256K
    # YaRN configuration for 256K context
    config.position.position_type = PositionEmbeddingType.YARN
    config.position.yarn_original_max_position = 4096
    config.position.yarn_scale = 64.0  # 4K -> 256K
    config.position.yarn_beta_fast = 32.0
    config.position.yarn_beta_slow = 1.0
    config.position.yarn_mscale = 0.707  # sqrt(0.5) for very long contexts
    # GSA tuned for very long sequences
    config.attention.gsa_k_base = 1024
    config.attention.gsa_k_min = 128
    config.attention.gsa_k_max = 4096
    # Triton kernels required for memory efficiency at this scale
    config.attention.gsa_use_triton_kernels = True
    return config


def get_1b_full_config() -> ModelConfig:
    """1B model with ALL advanced features enabled."""
    config = ModelConfig(
        model_name="LLM-1B-Full",
        vocab_size=128000,
        hidden_size=2048,
        num_hidden_layers=16,
        max_position_embeddings=32768,
        attention=AttentionConfig(
            attention_type=AttentionType.GATED_SPARSE,
            num_attention_heads=16,
            num_key_value_heads=4,
            gsa_indexer_dim=64,
            gsa_num_indexer_heads=4,
            gsa_k_base=2048,
            gsa_k_min=256,
            gsa_k_max=4096,
        ),
        position=PositionConfig(
            position_type=PositionEmbeddingType.YARN,
            yarn_original_max_position=4096,
            yarn_scale=8.0,
        ),
        ffn=FFNConfig(
            ffn_type=FFNType.SWIGLU,
            intermediate_size=4096,
        ),
        connection=ConnectionConfig(
            connection_type=ConnectionType.MHC,
            mhc_expansion_rate=4.0,
        ),
        head=HeadConfig(
            use_multi_token_prediction=True,
            num_predict_tokens=2,
        ),
    )
    return config


# Configuration presets registry
PRESET_CONFIGS = {
    "1b-base": get_1b_base_config,
    "1b-gsa": get_1b_gsa_config,
    "1b-deepseek-gsa": get_1b_deepseek_gsa_config,        # DeepSeek-style GSA (recommended)
    "1b-deepseek-gsa-128k": get_1b_deepseek_gsa_128k_config,  # 128K context
    "1b-deepseek-gsa-256k": get_1b_deepseek_gsa_256k_config,  # 256K context
    "1b-deepseek": get_1b_deepseek_config,                # DeepSeek MLA
    "1b-mhc": get_1b_mhc_config,
    "1b-mtp": get_1b_mtp_config,
    "1b-yarn": get_1b_yarn_config,
    "1b-full": get_1b_full_config,
}


def get_preset_config(name: str) -> ModelConfig:
    """Get a preset configuration by name."""
    if name not in PRESET_CONFIGS:
        available = ", ".join(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset: {name}. Available: {available}")
    return PRESET_CONFIGS[name]()
