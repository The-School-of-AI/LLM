"""
LLM Architecture Configuration
==============================

Centralized configuration for all model components.
Supports dynamic component selection via config flags.

Target: 1B Parameter Model
Inspired by: Qwen3 1.7B, SmolLM2, LLaMA 3, DeepSeek V3
"""

import json
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class AttentionType(Enum):
    """Available attention mechanisms."""

    GROUPED_QUERY = "grouped_query"  # GQA (default, like Qwen3/LLaMA3)
    GATED_SPARSE = (
        "gated_sparse"  # GSA from paper 2601.15305v1 (original implementation)
    )
    DEEPSEEK_GSA = "deepseek_gsa"  # DeepSeek-style GSA (corrected implementation)
    DEEPSEEK_SPARSE = "deepseek_sparse"  # DeepSeek V3 MLA
    GATED_DELTANET = (
        "gated_deltanet"  # Gated DeltaNet O(N) linear attention (2412.06464)
    )
    REFERENCE_GSA = "reference_gsa"  # Reference GSA matching Test_Code


class PositionEmbeddingType(Enum):
    """Position embedding types."""

    ROPE = "rope"  # Standard RoPE
    YARN = "yarn"  # YaRN for extended context
    ALIBI = "alibi"  # ALiBi (alternative)


class FFNType(Enum):
    """Feed-forward network types."""

    SWIGLU = "swiglu"  # SwiGLU (default)
    GELU = "gelu"  # Standard GELU
    MOE = "moe"  # Mixture of Experts


class ConnectionType(Enum):
    """Layer connection types."""

    RESIDUAL = "residual"  # Standard residual
    MHC = "mhc"  # Manifold Hyper-Connections (2512.24880)
    MHC_V2 = "mhc_v2"  # MHC V2 (norm inside, matching Test_Code)


class EmbeddingType(Enum):
    """Embedding types."""

    STANDARD = "standard"  # Standard nn.Embedding
    KRONECKER = "kronecker"  # Kronecker product embeddings (D=8192)


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms."""

    # Attention type selection
    attention_type: AttentionType = AttentionType.GROUPED_QUERY

    # Common GQA params (used for GQA layers and GSA layers)
    num_attention_heads: int = 16  # Q heads
    num_key_value_heads: int = 2  # KV heads for GQA (16Q / 2KV)
    head_dim: int = 256  # Per-head dimension
    attention_dropout: float = 0.0
    attention_bias: bool = False  # Modern LLMs don't use bias

    # --- DeltaNet Configuration (arXiv:2412.06464) ---
    # Gated DeltaNet: O(N) linear attention with gated delta rule
    delta_v_heads: int = 32  # V/O heads (hidden_size / delta_head_dim)
    delta_qk_heads: int = 16  # QK heads (delta_v_heads / 2)
    delta_head_dim: int = 128  # Head dimension for DeltaNet
    delta_gate_dim: int = 384  # Beta gate dimension (9.4% of hidden_size)

    # --- GSA Configuration (arXiv:2601.15305v1) ---
    # GSA layers use full MHA: num_heads × head_dim = hidden_size
    gsa_num_heads: int = 16  # GSA attention heads
    gsa_head_dim: int = 256  # GSA head dimension
    gsa_indexer_dim: int = 64  # d_I: Low-dim indexer projection
    gsa_num_indexer_heads: int = 4  # H_I: Number of indexer heads
    gsa_k_base: int = 512  # Base selection budget
    gsa_k_min: int = 32  # Minimum k (high confidence)
    gsa_k_max: int = 1024  # Maximum k (low confidence)

    # --- Mixer Configuration ---
    # Hybrid DeltaNet + GSA per-layer mixing
    mixer_delta_ratio: float = 0.75  # 75% DeltaNet layers
    mixer_gsa_ratio: float = 0.25  # 25% GSA layers

    # DeepSeek GSA specific (corrected implementation)
    gsa_use_adaptive_k: bool = True  # Enable adaptive k selection
    gsa_adaptive_k_method: str = "variance"  # "variance", "entropy", or "learned"
    gsa_adaptive_k_temperature: float = 1.0  # Temperature for adaptive scaling
    gsa_use_value_gate: bool = True  # Enable G2 (value gate)
    gsa_use_output_gate: bool = True  # Enable G1 (output gate)
    gsa_gate_activation: str = "sigmoid"  # Gate activation function
    gsa_gate_bias_init: float = 0.5  # Initial gate bias
    gsa_indexer_activation: str = "sigmoid"  # Indexer activation ("sigmoid" or "relu")
    gsa_use_triton_kernels: bool = (
        True  # Use Triton kernels for long sequences (if available)
    )
    gsa_sparse_backend: str = "auto"  # "auto", "triton", "pytorch", "flash", "dense"
    gsa_triton_min_seq_len: int = (
        512  # Use Triton only above this seq length in auto mode
    )
    gsa_prefer_flash: bool = True  # Prefer Flash/Efficient SDPA backend on CUDA
    gsa_sdpa_chunk_size: int = 16  # Query chunk size for SDPA sparse gather path

    # DeepSeek Sparse Attention specific
    ds_compressed_dim: int = 512  # Compressed KV dimension
    ds_rope_head_dim: int = 32  # RoPE dimension for decoupled attention
    ds_num_shared_experts: int = 1
    ds_q_lora_rank: int = 0  # 0 = no LoRA compression


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
    intermediate_size: int = 2048  # Dense shared expert FFN (FFN=2048)
    ffn_dropout: float = 0.0
    ffn_bias: bool = False

    # MoE specific
    moe_num_experts: int = 0  # 0 = dense model (no routed experts)
    moe_num_experts_per_tok: int = 0
    moe_expert_intermediate_size: Optional[int] = (
        None  # Routed expert FFN width; defaults to intermediate_size
    )
    moe_aux_loss_coef: float = 0.01
    moe_data_sparsity: float = 0.5  # Null expert data sparsity (rho)


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

    Memory impact of mhc_expansion_rate (per activation tensor, B=1, T=4096, D=4096, bf16):
    - n=1 (residual):  B×T×D×2      =  32 MB  (no mHC, standard residual)
    - n=2 (2 streams): B×T×n×D×2    =  64 MB  (2x memory, good memory-speed tradeoff)
    - n=4 (4 streams): B×T×n×D×2    = 128 MB  (4x memory, paper default)

    Use n=2 for memory-constrained settings (e.g. long context on limited VRAM).
    Use n=4 for full expressiveness (paper default, recommended if VRAM allows).
    """

    connection_type: ConnectionType = ConnectionType.MHC

    # mHC parameters (from DeepSeek paper 2512.24880v2)
    mhc_expansion_rate: int = (
        4  # n: number of streams (paper uses 4, use 2 for memory savings)
    )
    mhc_alpha_init: float = 0.01  # α: gating factor init (paper uses 0.01)
    mhc_sinkhorn_iters: int = 20  # Sinkhorn-Knopp iterations (paper uses 20)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding layer."""

    embedding_type: EmbeddingType = EmbeddingType.STANDARD
    kronecker_pf_dim: int = 8192  # D = CHAR_DIM(256) * POS_DIM(32)


@dataclass
class IntegrationConfig:
    """Configuration for reversible integration."""

    use_reversible: bool = False
    step_size: float = 0.25
    a: float = 0.5
    noise_eps: float = 0.0
    bootstrap: str = "euler"


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
    use_multi_token_prediction: bool = True
    num_predict_tokens: int = 2  # MTP layers (1 backbone NTP + 1 MTP = 2 predictions)
    mtp_loss_weight: float = 0.3  # Weight for auxiliary MTP loss
    mtp_block_type: str = "full_transformer"  # "full_transformer" or "linear"

    # Weight tying
    tie_word_embeddings: bool = True  # Tie input/output embeddings


@dataclass
class ModelConfig:
    """
    Complete model configuration.

    Default: 1B Dense (FFN=2048) with hybrid DeltaNet + GSA mixer.
    Architecture: 8 backbone layers + 1 MTP layer = 9 total computational layers.
    Inspired by: DeepSeek V3, Gated DeltaNet (2412.06464), GSA (2601.15305v1)
    """

    # Model identification
    model_name: str = "LLM-1B-Dense"
    model_version: str = "1.0.0"

    # Core architecture
    vocab_size: int = 131072  # 2^17, divisible by 64
    hidden_size: int = 4096  # Effective hidden dimension
    num_hidden_layers: int = 8  # Backbone layers
    max_position_embeddings: int = 8192

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
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)

    # Precision
    dtype: str = "bfloat16"  # bfloat16, float16, float32

    def __post_init__(self):
        """Validate and adjust configuration."""
        # Ensure GQA head_dim consistency: Q_heads × head_dim == hidden_size
        if (
            self.attention.head_dim * self.attention.num_attention_heads
            != self.hidden_size
        ):
            self.attention.head_dim = (
                self.hidden_size // self.attention.num_attention_heads
            )

        # Ensure DeltaNet V heads consistency: V_heads × delta_head_dim == hidden_size
        if (
            self.attention.delta_v_heads * self.attention.delta_head_dim
            != self.hidden_size
        ):
            self.attention.delta_v_heads = (
                self.hidden_size // self.attention.delta_head_dim
            )

        # Ensure GSA head consistency: gsa_num_heads × gsa_head_dim == hidden_size
        if (
            self.attention.gsa_num_heads * self.attention.gsa_head_dim
            != self.hidden_size
        ):
            self.attention.gsa_num_heads = (
                self.hidden_size // self.attention.gsa_head_dim
            )

        # Ensure mixer ratios sum to 1.0
        total_ratio = self.attention.mixer_delta_ratio + self.attention.mixer_gsa_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            self.attention.mixer_gsa_ratio = 1.0 - self.attention.mixer_delta_ratio

        # Ensure intermediate_size is set properly for SwiGLU
        if self.ffn.ffn_type == FFNType.SWIGLU and self.ffn.intermediate_size == 0:
            # SwiGLU optimal: hidden_size * 8/3, rounded to multiple of 256
            self.ffn.intermediate_size = int(self.hidden_size * 8 / 3)
            self.ffn.intermediate_size = (
                (self.ffn.intermediate_size + 255) // 256
            ) * 256

    @property
    def num_deltanet_layers(self) -> int:
        """Number of DeltaNet layers (75% by default)."""
        return int(self.num_hidden_layers * self.attention.mixer_delta_ratio)

    @property
    def num_gsa_layers(self) -> int:
        """Number of GSA layers (25% by default)."""
        return self.num_hidden_layers - self.num_deltanet_layers

    @property
    def num_parameters(self) -> int:
        """Estimate total active parameters."""
        H = self.hidden_size
        attn = self.attention

        # Embedding (input only if tied, input + output if untied)
        embed_params = self.vocab_size * H
        head_params = 0 if self.head.tie_word_embeddings else embed_params

        # --- DeltaNet layer params ---
        # V projection: H × (delta_v_heads × delta_head_dim)
        delta_v = H * (attn.delta_v_heads * attn.delta_head_dim)
        # QK projections: 2 × H × (delta_qk_heads × delta_head_dim)
        delta_qk = 2 * H * (attn.delta_qk_heads * attn.delta_head_dim)
        # O projection: (delta_v_heads × delta_head_dim) × H
        delta_o = (attn.delta_v_heads * attn.delta_head_dim) * H
        # Gate: H × delta_gate_dim
        delta_gate = H * attn.delta_gate_dim
        delta_mixer = delta_v + delta_qk + delta_o + delta_gate

        # --- GQA/GSA layer params ---
        # Q: H × (num_attention_heads × head_dim)
        gqa_q = H * (attn.num_attention_heads * attn.head_dim)
        # KV: 2 × H × (num_key_value_heads × head_dim)
        gqa_kv = 2 * H * (attn.num_key_value_heads * attn.head_dim)
        # O: (num_attention_heads × head_dim) × H
        gqa_o = (attn.num_attention_heads * attn.head_dim) * H
        gsa_mixer = gqa_q + gqa_kv + gqa_o

        # Average mixer per layer
        n_delta = self.num_deltanet_layers
        n_gsa = self.num_gsa_layers
        total_mixer = (delta_mixer * n_delta) + (gsa_mixer * n_gsa)

        # FFN: gate, up, down for SwiGLU (shared expert)
        ffn_params = 3 * H * self.ffn.intermediate_size

        # Norms per layer (4: pre-attn, post-attn, pre-ffn, post-ffn)
        norm_params = 4 * H

        total_layer_params = (
            total_mixer + (ffn_params + norm_params) * self.num_hidden_layers
        )

        # Final norm
        final_norm = H

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
            elif hasattr(obj, "__dataclass_fields__"):
                return {k: enum_to_str(v) for k, v in obj.__dict__.items()}
            return obj

        return enum_to_str(self.__dict__)

    def save(self, path: str):
        """Save configuration to file."""
        path = Path(path)
        data = self.to_dict()

        if path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        elif path.suffix in [".yaml", ".yml"]:
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create from dictionary."""
        # Filter out non-model config keys (e.g., training config)
        valid_fields = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in valid_fields}

        # Convert string enums back
        if "attention" in data:
            if "attention_type" in data["attention"]:
                data["attention"]["attention_type"] = AttentionType(
                    data["attention"]["attention_type"]
                )
            data["attention"] = AttentionConfig(**data["attention"])

        if "position" in data:
            if "position_type" in data["position"]:
                data["position"]["position_type"] = PositionEmbeddingType(
                    data["position"]["position_type"]
                )
            data["position"] = PositionConfig(**data["position"])

        if "ffn" in data:
            if "ffn_type" in data["ffn"]:
                data["ffn"]["ffn_type"] = FFNType(data["ffn"]["ffn_type"])
            data["ffn"] = FFNConfig(**data["ffn"])

        if "connection" in data:
            if "connection_type" in data["connection"]:
                data["connection"]["connection_type"] = ConnectionType(
                    data["connection"]["connection_type"]
                )
            data["connection"] = ConnectionConfig(**data["connection"])

        if "head" in data:
            data["head"] = HeadConfig(**data["head"])

        if "embedding" in data:
            if "embedding_type" in data["embedding"]:
                data["embedding"]["embedding_type"] = EmbeddingType(
                    data["embedding"]["embedding_type"]
                )
            data["embedding"] = EmbeddingConfig(**data["embedding"])

        if "integration" in data:
            data["integration"] = IntegrationConfig(**data["integration"])

        return cls(**data)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        """Load configuration from file."""
        path = Path(path)

        if path.suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)
        elif path.suffix in [".yaml", ".yml"]:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

        return cls.from_dict(data)


# =============================================================================
# Preset Configurations
# =============================================================================


def get_1b_base_config() -> ModelConfig:
    """
    1B Dense (FFN=2048) - Hybrid DeltaNet + GSA architecture.

    Architecture from weight calculator:
    - 8 backbone layers + 1 MTP layer = 9 total computational layers
    - 75% DeltaNet (6 layers) / 25% GSA (2 layers)
    - GQA: 16Q / 2KV × 256 head_dim
    - DeltaNet: 32V / 16QK × 128 head_dim, gate_dim=384
    - GSA: 16 heads × 256 dim
    - FFN=2048 (dense shared expert), no routed experts
    - mHC connections, MTP (Full Transformer)
    - Critical Depth Limit: 19.66

    Param budget:
    - Embeddings: 0.537B | Mixer Δ+GSA: 0.607B | mHC: 0.007B
    - Shared Expert: 0.201B | MTP: 0.161B | Total ~1B active
    """
    return ModelConfig(
        model_name="LLM-1B-Dense",
        vocab_size=131072,
        hidden_size=4096,
        num_hidden_layers=8,
        max_position_embeddings=8192,
        attention=AttentionConfig(
            attention_type=AttentionType.GROUPED_QUERY,
            # GQA: 16Q / 2KV × 256
            num_attention_heads=16,
            num_key_value_heads=2,
            head_dim=256,
            # DeltaNet: 32V / 16QK × 128, gate=384
            delta_v_heads=32,
            delta_qk_heads=16,
            delta_head_dim=128,
            delta_gate_dim=384,
            # GSA: 16 × 256
            gsa_num_heads=16,
            gsa_head_dim=256,
            gsa_k_base=512,
            gsa_k_min=32,
            gsa_k_max=1024,
            # Mixer: 75% Delta / 25% GSA
            mixer_delta_ratio=0.75,
            mixer_gsa_ratio=0.25,
        ),
        position=PositionConfig(
            position_type=PositionEmbeddingType.YARN,
            yarn_original_max_position=8192,
            yarn_scale=8.0,
        ),
        ffn=FFNConfig(
            ffn_type=FFNType.SWIGLU,
            intermediate_size=2048,  # FFN=2048 (dense shared expert)
            moe_num_experts=0,  # Dense model, no routed experts
            moe_num_experts_per_tok=0,
        ),
        connection=ConnectionConfig(
            connection_type=ConnectionType.MHC,
            mhc_expansion_rate=4,
        ),
        head=HeadConfig(
            use_multi_token_prediction=True,
            num_predict_tokens=2,
            mtp_block_type="full_transformer",
            tie_word_embeddings=True,
        ),
    )


def get_1b_gsa_config() -> ModelConfig:
    """1B model with Gated Sparse Attention (paper 2601.15305v1)."""
    config = get_1b_base_config()
    config.model_name = "LLM-1B-GSA"
    config.attention.attention_type = AttentionType.GATED_SPARSE
    # Indexer parameters (Table 1 in paper)
    config.attention.gsa_indexer_dim = 64  # d_I
    config.attention.gsa_num_indexer_heads = 4  # H_I
    # Adaptive sparsity parameters
    config.attention.gsa_k_base = 2048  # Base selection budget
    config.attention.gsa_k_min = 256  # Min k (confident)
    config.attention.gsa_k_max = 4096  # Max k (uncertain)
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
    config.attention.gsa_k_base = 512  # Good balance for 40GB+ GPUs
    config.attention.gsa_k_min = 64  # Minimum tokens to attend to
    config.attention.gsa_k_max = 1024  # Cap for very long sequences
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
    """1B model with ALL advanced features enabled (same as base for 1B dense)."""
    config = get_1b_base_config()
    config.model_name = "LLM-1B-Full"
    config.max_position_embeddings = 32768
    config.position.yarn_original_max_position = 8192
    config.position.yarn_scale = 8.0
    return config


def get_1b_reference_config() -> ModelConfig:
    """
    1B Dense Reference Architecture matching Test_Code/model_1b.py.

    Architecture:
    - 8 backbone layers: 75% DeltaNet (6) + 25% GSA (2)
    - mHC V2 connections (norm inside, alpha=0.1)
    - Full Transformer MTP block
    - Reversible Midpoint Integration
    - YARN RoPE for 256k context
    - Dense FFN (no MoE, intermediate=2048)
    - Standard or Kronecker embeddings

    Param budget: ~1B active parameters
    """
    return ModelConfig(
        model_name="LLM-1B-Reference",
        vocab_size=131072,
        hidden_size=4096,
        num_hidden_layers=8,
        max_position_embeddings=262144,  # 256k context
        attention=AttentionConfig(
            attention_type=AttentionType.GATED_DELTANET,
            # GQA params (for fallback)
            num_attention_heads=16,
            num_key_value_heads=2,
            head_dim=256,
            # DeltaNet: 32V × 128 head_dim
            delta_v_heads=32,
            delta_qk_heads=16,
            delta_head_dim=128,
            delta_gate_dim=384,
            # GSA: 16 heads, d_idx=32 (hardcoded in ReferenceGSA)
            gsa_num_heads=16,
            gsa_head_dim=256,
            gsa_indexer_dim=32,  # d_idx=32 matching Test_Code
            gsa_num_indexer_heads=4,
            gsa_k_base=512,
            gsa_k_min=32,
            gsa_k_max=1024,
            gsa_use_triton_kernels=True,
            gsa_sparse_backend="auto",
            gsa_triton_min_seq_len=512,
            gsa_prefer_flash=True,
            gsa_sdpa_chunk_size=16,
            # Mixer: 75% DeltaNet / 25% GSA
            mixer_delta_ratio=0.75,
            mixer_gsa_ratio=0.25,
        ),
        position=PositionConfig(
            position_type=PositionEmbeddingType.YARN,
            rope_theta=10000.0,
            rope_scaling_factor=32.0,
            yarn_original_max_position=8192,
            yarn_scale=32.0,
        ),
        ffn=FFNConfig(
            ffn_type=FFNType.SWIGLU,
            intermediate_size=2048,
            moe_expert_intermediate_size=1024,  # Routed experts (future MoE mode)
            moe_num_experts=0,
            moe_num_experts_per_tok=0,
            moe_data_sparsity=0.0,
        ),
        connection=ConnectionConfig(
            connection_type=ConnectionType.MHC_V2,
            mhc_expansion_rate=4,
            mhc_alpha_init=0.1,  # Test_Code uses 0.1
            mhc_sinkhorn_iters=20,
        ),
        head=HeadConfig(
            use_multi_token_prediction=True,
            num_predict_tokens=2,
            mtp_block_type="full_transformer",
            tie_word_embeddings=False,  # Cannot tie with Kronecker
        ),
        embedding=EmbeddingConfig(
            embedding_type=EmbeddingType.STANDARD,
        ),
        integration=IntegrationConfig(
            use_reversible=True,
            step_size=0.25,
            a=0.5,
            noise_eps=0.0,
            bootstrap="euler",
        ),
    )


# Configuration presets registry
PRESET_CONFIGS = {
    "1b-base": get_1b_base_config,
    "1b-gsa": get_1b_gsa_config,
    "1b-deepseek-gsa": get_1b_deepseek_gsa_config,  # DeepSeek-style GSA (recommended)
    "1b-deepseek-gsa-128k": get_1b_deepseek_gsa_128k_config,  # 128K context
    "1b-deepseek-gsa-256k": get_1b_deepseek_gsa_256k_config,  # 256K context
    "1b-deepseek": get_1b_deepseek_config,  # DeepSeek MLA
    "1b-mhc": get_1b_mhc_config,
    "1b-mtp": get_1b_mtp_config,
    "1b-yarn": get_1b_yarn_config,
    "1b-full": get_1b_full_config,
    "1b-reference": get_1b_reference_config,  # Test_Code reference architecture
}


def get_preset_config(name: str) -> ModelConfig:
    """Get a preset configuration by name."""
    if name not in PRESET_CONFIGS:
        available = ", ".join(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset: {name}. Available: {available}")
    return PRESET_CONFIGS[name]()
