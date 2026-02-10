import pandas as pd
import numpy as np
import math
from dataclasses import dataclass

# ==========================================
# 1. EXACT Lightning ARCHITECTURE CALCULATOR
# ==========================================

@dataclass
class LightningConfig:
    # --- Global Architecture ---
    model_name: str = "LightningLM1.0"
    vocab_size: int = 131072
    hidden_size: int = 4096
    target_params: float = 70e9
    tie_word_embeddings: bool = True

    # --- Multi Token Prediction (MTP) ---
    enable_mtp: bool = True  # Enable multi-token prediction
    mtp_num_predictions: int = 2  # Number of future tokens to predict (n=1 means standard, n>1 enables MTP)
    mtp_projection_type: str = "linear"  # "linear" or "identity" - how to project intermediate layers

    # --- Scaling Constants ---
    depth_scale_constant: float = 2.43  # k in D_crit = k * ln(W)
    num_layer_norms: int = 4  # Layer norms per layer
    expert_weight_matrices: int = 3  # up_proj, down_proj, gate_proj

    # --- Manual Overrides (None = auto-calculate) ---
    num_layers_override: int = None  # Override auto-calculated layer count
    num_experts_override: int = None  # Override auto-calculated expert count

    # --- Theoretical Formula Constants (Section B) ---
    theory_attn_constant: float = 4.0  # Attention/Mixer weight coefficient
    theory_expert_constant: float = 8.0  # Expert weight coefficient per expert 

    # --- MoE Configuration ---
    # STRATEGY: "Sweet Spot" (Lane B)
    # Shared Expert Intermediate = 1024 (Provides strong baseline signal)
    # Routed Expert Intermediate = 512 (Granular specialization)
    # Active Routed Experts = 4 (Keeps compute balanced)
    num_routed_experts_active: int = 5
    num_shared_experts: int = 1
    expert_intermediate_size: int = 1024
    shared_expert_intermediate_size: int = 1280 

    # --- Attention / Mixer Configuration ---
    attention_type: str = "gqa"  # "gqa", "gsa", or "deepseek" - attention mechanism type for non-DeltaNet layers
    deltanet_layer_ratio: float = 0.75  # Ratio of DeltaNet layers (rest use attention_type)

    # GQA Head Configuration (auto-computed from hidden_size if None)
    attn_head_dim: int = 256  # Head dimension (typically 128 or 256)
    attn_q_heads: int = None  # Auto: hidden_size / attn_head_dim
    attn_kv_heads: int = None  # Auto: attn_q_heads / 8 (8:1 GQA ratio)

    # DeepSeek specific (only used when attention_type="deepseek")
    deepseek_shared_heads: int = None  # Auto: 1/4 of total heads
    deepseek_sparse_heads: int = None  # Auto: 3/4 of total heads

    # DeltaNet Head Configuration (auto-computed from hidden_size if None)
    delta_head_dim: int = 128  # Head dimension (typically 64 or 128)
    delta_v_heads: int = None  # Auto: hidden_size / delta_head_dim
    delta_qk_heads: int = None  # Auto: delta_v_heads / 2 (2:1 ratio)
    delta_gate_dim: int = 384  # Beta gate dimension (9.4% of hidden_size for 4096)

    # GSA (Gated Sparse Attention) Configuration
    gsa_num_heads: int = None  # Auto: hidden_size / gsa_head_dim
    gsa_head_dim: int = 256  # Head dimension for GSA
    gsa_indexer_heads: int = 4  # Lightning indexer heads
    gsa_indexer_dim: int = 32  # Indexer dimension

    # mHC (Multi-Head Composition) Configuration
    n_streams: int = 4  # Number of streams for mHC
    sinkhorn_iters: int = 20  # Sinkhorn iterations (doesn't affect params)

    # DeltaNet Auxiliary Parameters
    delta_conv_size: int = 4  # Short convolution kernel size

class LightningCalculator:
    def __init__(self, config: LightningConfig):
        self.cfg = config

        # Auto-compute head configurations if not specified
        self._auto_configure_heads()

        # Calculate depth first (needed for MTP validation)
        self.d_crit = config.depth_scale_constant * math.log(config.hidden_size)

        # Use override if provided, otherwise calculate from critical depth
        if config.num_layers_override is not None:
            self.num_layers = config.num_layers_override
            print(f"⚙️  Using manual layer override: {self.num_layers} layers (critical depth suggests {int(math.floor(self.d_crit))})")
        else:
            self.num_layers = int(math.floor(self.d_crit))

        self.num_delta_layers = int(self.num_layers * config.deltanet_layer_ratio)
        self.num_attn_layers = self.num_layers - self.num_delta_layers

        # Validate configuration
        self._validate_config()

    def _auto_configure_heads(self):
        """Auto-compute head configurations based on hidden_size if not explicitly set."""
        cfg = self.cfg

        # GQA: Compute Q heads from hidden_size and head_dim
        if cfg.attn_q_heads is None:
            cfg.attn_q_heads = int(cfg.hidden_size // cfg.attn_head_dim)

        # GQA: Compute KV heads (8:1 ratio with Q heads)
        if cfg.attn_kv_heads is None:
            cfg.attn_kv_heads = max(1, cfg.attn_q_heads // 8)

        # DeepSeek: Compute heads (1/4 shared, 3/4 sparse)
        if cfg.deepseek_shared_heads is None:
            total_heads = int(cfg.hidden_size // cfg.attn_head_dim)
            cfg.deepseek_shared_heads = max(1, total_heads // 4)

        if cfg.deepseek_sparse_heads is None:
            total_heads = int(cfg.hidden_size // cfg.attn_head_dim)
            cfg.deepseek_sparse_heads = total_heads - cfg.deepseek_shared_heads

        # DeltaNet: Compute V heads from hidden_size and head_dim
        if cfg.delta_v_heads is None:
            cfg.delta_v_heads = int(cfg.hidden_size // cfg.delta_head_dim)

        # DeltaNet: Compute QK heads (2:1 ratio with V heads)
        if cfg.delta_qk_heads is None:
            cfg.delta_qk_heads = max(1, cfg.delta_v_heads // 2)

        # GSA: Compute heads from hidden_size and head_dim
        if cfg.gsa_num_heads is None:
            cfg.gsa_num_heads = int(cfg.hidden_size // cfg.gsa_head_dim)

    def _validate_config(self):
        """Validate architectural constraints and configuration consistency."""
        cfg = self.cfg

        # === Attention Architecture Validation ===
        assert cfg.attention_type in ["gqa", "gsa", "deepseek"], (
            f"Attention type must be 'gqa', 'gsa', or 'deepseek', got '{cfg.attention_type}'."
        )

        if cfg.attention_type == "gqa":
            attn_qo_dim = cfg.attn_q_heads * cfg.attn_head_dim
            assert attn_qo_dim == cfg.hidden_size, (
                f"Attention Q/O dimension mismatch: {cfg.attn_q_heads} heads × {cfg.attn_head_dim} dim "
                f"= {attn_qo_dim}, but hidden_size = {cfg.hidden_size}. "
                f"For GQA: Q_heads × head_dim must equal hidden_size."
            )

            attn_kv_dim = cfg.attn_kv_heads * cfg.attn_head_dim
            assert attn_kv_dim <= cfg.hidden_size, (
                f"Attention KV dimension too large: {cfg.attn_kv_heads} heads × {cfg.attn_head_dim} dim "
                f"= {attn_kv_dim} > hidden_size {cfg.hidden_size}."
            )

            assert cfg.attn_q_heads % cfg.attn_kv_heads == 0, (
                f"GQA requires Q heads ({cfg.attn_q_heads}) to be divisible by KV heads ({cfg.attn_kv_heads})."
            )
        else:  # deepseek
            total_heads = cfg.deepseek_shared_heads + cfg.deepseek_sparse_heads
            attn_qo_dim = total_heads * cfg.attn_head_dim
            assert attn_qo_dim == cfg.hidden_size, (
                f"DeepSeek: (shared_heads + sparse_heads) × head_dim must equal hidden_size. "
                f"Got: ({cfg.deepseek_shared_heads} + {cfg.deepseek_sparse_heads}) × {cfg.attn_head_dim} "
                f"= {attn_qo_dim}, but hidden_size = {cfg.hidden_size}."
            )

        # === DeltaNet Architecture Validation ===
        delta_vo_dim = cfg.delta_v_heads * cfg.delta_head_dim
        assert delta_vo_dim == cfg.hidden_size, (
            f"DeltaNet V/O dimension mismatch: {cfg.delta_v_heads} heads × {cfg.delta_head_dim} dim "
            f"= {delta_vo_dim}, but hidden_size = {cfg.hidden_size}. "
            f"V_heads × head_dim must equal hidden_size."
        )

        delta_qk_dim = cfg.delta_qk_heads * cfg.delta_head_dim
        assert delta_qk_dim <= cfg.hidden_size, (
            f"DeltaNet QK dimension too large: {cfg.delta_qk_heads} heads × {cfg.delta_head_dim} dim "
            f"= {delta_qk_dim} > hidden_size {cfg.hidden_size}."
        )

        # Gate dimension should be reasonable (typically 128-512 for 8192 hidden)
        gate_ratio = cfg.delta_gate_dim / cfg.hidden_size
        assert 0.01 <= gate_ratio <= 0.5, (
            f"DeltaNet gate_dim ({cfg.delta_gate_dim}) seems unusual for hidden_size ({cfg.hidden_size}). "
            f"Ratio is {gate_ratio:.3f}, expected 0.01-0.5 (typically ~0.016-0.063)."
        )

        # === GSA Architecture Validation ===
        gsa_qo_dim = cfg.gsa_num_heads * cfg.gsa_head_dim
        assert gsa_qo_dim == cfg.hidden_size, (
            f"GSA Q/O dimension mismatch: {cfg.gsa_num_heads} heads × {cfg.gsa_head_dim} dim "
            f"= {gsa_qo_dim}, but hidden_size = {cfg.hidden_size}. "
            f"For GSA: num_heads × head_dim must equal hidden_size."
        )

        # === MoE Validation ===
        # Allow 0 experts for dense models (no MoE)
        assert cfg.num_routed_experts_active >= 0, (
            f"Active experts must be non-negative, got {cfg.num_routed_experts_active}."
        )

        assert cfg.num_shared_experts >= 0, (
            f"Shared experts count must be non-negative, got {cfg.num_shared_experts}."
        )

        # If using MoE (experts > 0), validate intermediate sizes
        if cfg.num_routed_experts_active > 0 or cfg.num_shared_experts > 0:
            assert cfg.expert_intermediate_size > 0, (
                f"Expert intermediate size must be positive, got {cfg.expert_intermediate_size}."
            )

            assert cfg.shared_expert_intermediate_size > 0, (
                f"Shared expert intermediate size must be positive, got {cfg.shared_expert_intermediate_size}."
            )

        # === Layer Ratio Validation ===
        assert 0.0 <= cfg.deltanet_layer_ratio <= 1.0, (
            f"DeltaNet layer ratio must be in [0, 1], got {cfg.deltanet_layer_ratio}."
        )

        # === Scaling Constants Validation ===
        assert cfg.depth_scale_constant > 0, (
            f"Depth scale constant must be positive, got {cfg.depth_scale_constant}."
        )

        assert cfg.num_layer_norms > 0, (
            f"Number of layer norms must be positive, got {cfg.num_layer_norms}."
        )

        assert cfg.expert_weight_matrices == 3, (
            f"Expert weight matrices must be 3 (up/down/gate), got {cfg.expert_weight_matrices}."
        )

        # === Target Parameters Validation ===
        assert cfg.target_params > 0, (
            f"Target parameters must be positive, got {cfg.target_params}."
        )

        min_params = cfg.vocab_size * cfg.hidden_size  # Just embeddings
        assert cfg.target_params >= min_params, (
            f"Target params ({cfg.target_params/1e9:.1f}B) too small. "
            f"Embeddings alone need {min_params/1e9:.1f}B."
        )

        # === Multi Token Prediction Validation ===
        if cfg.enable_mtp:
            assert cfg.mtp_num_predictions >= 2, (
                f"MTP requires at least 2 predictions (got {cfg.mtp_num_predictions}). "
                f"Use enable_mtp=False for standard single-token prediction."
            )

            # Note: MTP block is a full transformer layer (DeepSeek-V3 style)
            # It predicts t+2 given hidden state h_t and embedding of t+1

        print(f"✓ Configuration validated successfully")
        print(f"  - GQA: {cfg.attn_q_heads}Q/{cfg.attn_kv_heads}KV heads × {cfg.attn_head_dim} dim = {attn_qo_dim}")
        print(f"  - DeltaNet: {cfg.delta_v_heads}V/{cfg.delta_qk_heads}QK heads × {cfg.delta_head_dim} dim = {delta_vo_dim}")
        print(f"  - GSA: {cfg.gsa_num_heads} heads × {cfg.gsa_head_dim} dim = {gsa_qo_dim}")
        print(f"  - Gate dimension: {cfg.delta_gate_dim} ({gate_ratio*100:.1f}% of hidden)")
        if cfg.enable_mtp:
            total_layers = self.num_layers + 1  # Backbone + MTP block
            print(f"  - MTP: {cfg.mtp_num_predictions} predictions with full transformer block")
            print(f"  - Total computational layers: {self.num_layers} backbone + 1 MTP = {total_layers}")
        print()

    def calc_embeddings(self):
        input_embed = self.cfg.vocab_size * self.cfg.hidden_size
        output_head = 0 if self.cfg.tie_word_embeddings else self.cfg.vocab_size * self.cfg.hidden_size
        return input_embed, output_head

    def calc_mtp_block(self, num_experts_total=0):
        """Calculate Multi Token Prediction block parameters (DeepSeek-V3 style).

        MTP block is a FULL transformer block with:
        - Fusion projection: 2*hidden → hidden (concatenates h_t and emb_{t+1})
        - Full attention layer (same as backbone layers)
        - Full MLP/MoE layer (same as backbone layers)
        - mHC wrappers (2 per block: attention + MLP)
        - RMS norms
        - Router (if using MoE)
        - Shared + routed experts (if using MoE)

        This is essentially adding ONE extra transformer layer to the backbone.

        Args:
            num_experts_total: Total number of routed experts (for MoE calculation)

        Returns:
            int: Total parameters in the MTP block (0 if MTP disabled)
        """
        if not self.cfg.enable_mtp or self.cfg.mtp_num_predictions <= 1:
            return 0

        # 1. Fusion projection: 2*hidden → hidden
        fusion_proj = (2 * self.cfg.hidden_size) * self.cfg.hidden_size

        # 2. Attention layer (same calculation as backbone)
        if self.cfg.attention_type == "gsa":
            attn_mass = self.calc_gsa_layer()
        else:  # gqa or deepseek
            attn_mass = self.calc_standard_attn_layer()

        # 3. mHC wrappers (2 per block: one for attention, one for MLP)
        mhc_per_block = 2 * self.calc_mhc_wrapper()

        # 4. Norms (same as backbone layers)
        norms = self.cfg.num_layer_norms * self.cfg.hidden_size

        # 5. Router (if using MoE)
        router_mass = self.cfg.hidden_size * num_experts_total if num_experts_total > 0 else 0

        # 6. MoE experts (shared + routed)
        routed_exp_size = self.cfg.expert_weight_matrices * self.cfg.hidden_size * self.cfg.expert_intermediate_size
        shared_exp_size = self.cfg.expert_weight_matrices * self.cfg.hidden_size * self.cfg.shared_expert_intermediate_size

        # MTP block has same MoE structure as backbone
        total_exp_mass = (num_experts_total * routed_exp_size) + shared_exp_size

        # Total MTP block parameters
        mtp_block_params = fusion_proj + attn_mass + mhc_per_block + norms + router_mass + total_exp_mass

        return mtp_block_params

    def calc_mtp_params(self):
        """Wrapper for backward compatibility. Returns (0, 0) since MTP is now calculated with experts."""
        # MTP block is now calculated inside solve_for_experts and generate_report
        # This method is kept for backward compatibility but returns 0
        return 0, 0

    def calc_standard_attn_layer(self):
        w_q = self.cfg.hidden_size * (self.cfg.attn_q_heads * self.cfg.attn_head_dim)
        w_kv = 2 * self.cfg.hidden_size * (self.cfg.attn_kv_heads * self.cfg.attn_head_dim)
        w_o = (self.cfg.attn_q_heads * self.cfg.attn_head_dim) * self.cfg.hidden_size
        return w_q + w_kv + w_o

    def calc_deltanet_layer(self):
        """Calculate DeltaNet layer parameters.

        DeltaNet includes:
        - Core projections: Q, K, V, O
        - Output gate: G
        - Gate projections: b_proj (beta), gk_proj (for alpha)
        - Short convolutions: q_conv, k_conv, v_conv
        - Small params: A_log, D, dt_bias (per-head)
        """
        key_dim = self.cfg.delta_qk_heads * self.cfg.delta_head_dim
        value_dim = self.cfg.delta_v_heads * self.cfg.delta_head_dim

        # Core projections
        w_q = self.cfg.hidden_size * key_dim
        w_k = self.cfg.hidden_size * key_dim
        w_v = self.cfg.hidden_size * value_dim
        w_g = self.cfg.hidden_size * value_dim  # Output gate (was missing!)
        w_o = value_dim * self.cfg.hidden_size

        # Gate projections (beta and alpha computation)
        w_b_proj = self.cfg.hidden_size * self.cfg.delta_v_heads  # Beta writing strength
        w_gk_proj = self.cfg.hidden_size * self.cfg.delta_v_heads  # For alpha computation

        # Short convolutions (depthwise, so channels = input_dim)
        conv_q = key_dim * self.cfg.delta_conv_size  # q_conv1d
        conv_k = key_dim * self.cfg.delta_conv_size  # k_conv1d
        conv_v = value_dim * self.cfg.delta_conv_size  # v_conv1d

        # Small per-head parameters
        small_params = self.cfg.delta_v_heads * 3  # A_log, D, dt_bias (one per head)

        return w_q + w_k + w_v + w_g + w_o + w_b_proj + w_gk_proj + conv_q + conv_k + conv_v + small_params

    def calc_mhc_wrapper(self):
        """Calculate mHC (Multi-Head Composition) wrapper parameters.

        Each layer has 2 mHC wrappers (one for attention, one for MLP).
        Each mHC wrapper includes:
        - phi_pre: (n_streams * hidden_size) -> n_streams
        - phi_post: (n_streams * hidden_size) -> n_streams
        - phi_res: (n_streams * hidden_size) -> (n_streams * n_streams)
        - Biases: b_pre, b_post, b_res
        - Alpha parameters: alpha_pre, alpha_post, alpha_res
        - RMS norm: n_streams * hidden_size parameters
        """
        d_in = self.cfg.n_streams * self.cfg.hidden_size

        # Projection weights
        phi_pre = d_in * self.cfg.n_streams
        phi_post = d_in * self.cfg.n_streams
        phi_res = d_in * (self.cfg.n_streams * self.cfg.n_streams)

        # Biases
        b_pre = self.cfg.n_streams
        b_post = self.cfg.n_streams
        b_res = self.cfg.n_streams * self.cfg.n_streams

        # Alpha parameters (3 scalars)
        alphas = 3

        # RMS norm
        rms_norm = d_in

        return phi_pre + phi_post + phi_res + b_pre + b_post + b_res + alphas + rms_norm

    def calc_gsa_layer(self):
        """Calculate GSA (Gated Sparse Attention) layer parameters.

        GSA includes:
        - Standard attention projections (Q, K, V, O)
        - Lightning Indexer (W_Iq, W_Ik, W_Iw, gate_bias)
        - Dual Gating (W_gv, W_go)
        """
        # Standard attention projections (full multi-head, not GQA)
        w_q = self.cfg.hidden_size * self.cfg.hidden_size
        w_k = self.cfg.hidden_size * self.cfg.hidden_size
        w_v = self.cfg.hidden_size * self.cfg.hidden_size
        w_o = self.cfg.hidden_size * self.cfg.hidden_size

        # Lightning Indexer components
        w_iq = self.cfg.hidden_size * (self.cfg.gsa_indexer_heads * self.cfg.gsa_indexer_dim)  # W_Iq
        w_ik = self.cfg.hidden_size * self.cfg.gsa_indexer_dim  # W_Ik
        w_iw = self.cfg.hidden_size * self.cfg.gsa_indexer_heads  # W_Iw
        gate_bias = self.cfg.gsa_indexer_heads  # gate_bias parameter

        # Dual Gating
        w_gv = self.cfg.hidden_size * self.cfg.hidden_size  # W_gv (value gate)
        w_go = self.cfg.hidden_size * self.cfg.hidden_size  # W_go (output gate)

        return w_q + w_k + w_v + w_o + w_iq + w_ik + w_iw + gate_bias + w_gv + w_go

    def solve_for_experts(self):
        in_emb, out_emb = self.calc_embeddings()
        total_emb = in_emb + out_emb

        # MTP fusion projection (2*hidden → hidden, separate from layer calculations)
        mtp_fusion_proj = 0
        if self.cfg.enable_mtp and self.cfg.mtp_num_predictions > 1:
            mtp_fusion_proj = (2 * self.cfg.hidden_size) * self.cfg.hidden_size

        # Total number of transformer layers (backbone + MTP block if enabled)
        num_total_layers = self.num_layers
        if self.cfg.enable_mtp and self.cfg.mtp_num_predictions > 1:
            num_total_layers += 1  # Add 1 for MTP block

        # Choose attention calculation based on attention_type
        if self.cfg.attention_type == "gsa":
            attn_mass = self.calc_gsa_layer()
        else:  # gqa or deepseek
            attn_mass = self.calc_standard_attn_layer()

        delta_mass = self.calc_deltanet_layer()
        avg_mixer = ((attn_mass * self.num_attn_layers) + (delta_mass * self.num_delta_layers)) / self.num_layers

        # mHC wrappers (2 per layer: one for attention, one for MLP)
        mhc_per_layer = 2 * self.calc_mhc_wrapper()

        shared_exp_size = self.cfg.expert_weight_matrices * self.cfg.hidden_size * self.cfg.shared_expert_intermediate_size
        routed_exp_size = self.cfg.expert_weight_matrices * self.cfg.hidden_size * self.cfg.expert_intermediate_size
        norms = self.cfg.num_layer_norms * self.cfg.hidden_size

        # Remaining budget after embeddings and MTP fusion projection
        remaining_budget = self.cfg.target_params - total_emb - mtp_fusion_proj

        # Average budget per layer (backbone + MTP if enabled)
        layer_budget = remaining_budget / num_total_layers

        # Expert cost includes W params for Router
        numerator = layer_budget - avg_mixer - mhc_per_layer - shared_exp_size - norms
        denominator = routed_exp_size + self.cfg.hidden_size

        assert numerator > 0, (
            f"Insufficient parameter budget for experts! "
            f"Layer budget: {layer_budget/1e6:.1f}M, "
            f"Mixer: {avg_mixer/1e6:.1f}M, "
            f"Shared expert: {shared_exp_size/1e6:.1f}M, "
            f"Norms: {norms/1e6:.1f}M. "
            f"Remaining: {numerator/1e6:.1f}M (need > 0)"
        )

        e_total = numerator / denominator

        assert e_total >= self.cfg.num_routed_experts_active, (
            f"Solved expert count ({e_total:.1f}) is less than active experts required ({self.cfg.num_routed_experts_active}). "
            f"Configuration is infeasible - reduce model size or increase target params."
        )

        return int(e_total)

    def generate_report(self, num_experts_total):
        input_emb, output_head = self.calc_embeddings()
        total_emb_mass = input_emb + output_head

        # Calculate MTP block parameters (includes full transformer layer + fusion proj)
        mtp_block_params = self.calc_mtp_block(num_experts_total)

        # Choose attention calculation based on attention_type
        if self.cfg.attention_type == "gsa":
            attn_mass = self.calc_gsa_layer()
        else:  # gqa or deepseek
            attn_mass = self.calc_standard_attn_layer()

        delta_mass = self.calc_deltanet_layer()
        avg_mixer = ((attn_mass * self.num_attn_layers) + (delta_mass * self.num_delta_layers)) / self.num_layers

        # mHC wrappers (2 per layer: one for attention, one for MLP)
        mhc_per_layer = 2 * self.calc_mhc_wrapper()

        routed_exp_size = self.cfg.expert_weight_matrices * self.cfg.hidden_size * self.cfg.expert_intermediate_size
        shared_exp_size = self.cfg.expert_weight_matrices * self.cfg.hidden_size * self.cfg.shared_expert_intermediate_size

        router_mass = self.cfg.hidden_size * num_experts_total
        norm_mass = self.cfg.num_layer_norms * self.cfg.hidden_size

        active_exp_mass = (self.cfg.num_routed_experts_active * routed_exp_size) + shared_exp_size
        layer_active = avg_mixer + mhc_per_layer + router_mass + norm_mass + active_exp_mass

        total_exp_mass = (num_experts_total * routed_exp_size) + shared_exp_size
        layer_total = avg_mixer + mhc_per_layer + router_mass + norm_mass + total_exp_mass

        # Total params: embeddings + (backbone layers * per-layer params) + MTP block
        total_params = total_emb_mass + (self.num_layers * layer_total) + mtp_block_params

        # Active params: embeddings + (backbone layers * active per-layer params) + MTP block (MTP is always active)
        # For MTP block active params, we only count the active experts
        if mtp_block_params > 0:
            # Calculate MTP block active params (fusion + attention + mhc + norms + router + active experts)
            fusion_proj = (2 * self.cfg.hidden_size) * self.cfg.hidden_size
            if self.cfg.attention_type == "gsa":
                mtp_attn_mass = self.calc_gsa_layer()
            else:
                mtp_attn_mass = self.calc_standard_attn_layer()
            mtp_mhc = 2 * self.calc_mhc_wrapper()
            mtp_norms = self.cfg.num_layer_norms * self.cfg.hidden_size
            mtp_router = self.cfg.hidden_size * num_experts_total if num_experts_total > 0 else 0
            mtp_active_experts = active_exp_mass
            mtp_block_active = fusion_proj + mtp_attn_mass + mtp_mhc + mtp_norms + mtp_router + mtp_active_experts
        else:
            mtp_block_active = 0

        active_params = total_emb_mass + (self.num_layers * layer_active) + mtp_block_active

        data = [
            ["Vocab Embeddings (Input)", 1, f"{input_emb/1e9:.3f} B", f"{input_emb/1e9:.3f} B"],
            ["LM Head (Output)", 1, f"{output_head/1e9:.3f} B", f"{output_head/1e9:.3f} B"],
        ]

        # Create mixer label based on attention type
        attn_label = self.cfg.attention_type.upper()
        mixer_label = f"Mixer ({int(self.cfg.deltanet_layer_ratio*100)}% Delta / {int((1-self.cfg.deltanet_layer_ratio)*100)}% {attn_label})"

        # Backbone layers row
        data.extend([
            [mixer_label, self.num_layers, f"{avg_mixer/1e6:.1f} M", f"{(avg_mixer * self.num_layers)/1e9:.3f} B"],
            [f"mHC Wrappers (2×{self.cfg.n_streams} streams)", self.num_layers, f"{mhc_per_layer/1e6:.1f} M", f"{(mhc_per_layer * self.num_layers)/1e9:.3f} B"],
            ["Router & Norms", self.num_layers, f"{(router_mass + norm_mass)/1e6:.1f} M", f"{((router_mass + norm_mass)*self.num_layers)/1e9:.3f} B"],
            [f"Shared Expert (Dense {self.cfg.shared_expert_intermediate_size})", self.num_layers, f"{shared_exp_size/1e6:.1f} M", f"{(shared_exp_size * self.num_layers)/1e9:.3f} B"],
            [f"Routed Experts (Active: {self.cfg.num_routed_experts_active})", self.num_layers, f"{(self.cfg.num_routed_experts_active * routed_exp_size)/1e6:.1f} M", f"{(self.cfg.num_routed_experts_active * routed_exp_size * self.num_layers)/1e9:.3f} B"],
            [f"Routed Experts (Inactive: {num_experts_total - self.cfg.num_routed_experts_active})", self.num_layers, "0 M", f"{((num_experts_total - self.cfg.num_routed_experts_active) * routed_exp_size * self.num_layers)/1e9:.3f} B"],
        ])

        # Add MTP block row if enabled (shown as an additional full transformer layer)
        if self.cfg.enable_mtp and mtp_block_params > 0:
            data.append([
                f"MTP Block (Full Transformer for t+2 prediction)",
                1,
                f"{mtp_block_params/1e6:.1f} M",
                f"{mtp_block_params/1e9:.3f} B"
            ])

        data.extend([
            ["-----------------", "", "-------", "-------"],
            ["TOTAL ACTIVE PARAMETERS", "", "->", f"{active_params/1e9:.3f} B"],
            ["TOTAL MODEL PARAMETERS", "", "->", f"{total_params/1e9:.3f} B"],
        ])
        return pd.DataFrame(data, columns=["Component", "Layers", "Params Per Layer (Active)", "Total Contribution"]), num_experts_total

    def calculate_moe_dimensions(self, num_experts):
        # Section B Logic: Theoretical Optimal Dimensions (Standard Reference)
        w_range = np.linspace(1024, 16384, 1000)
        results = []
        for w in w_range:
            d_crit = self.cfg.depth_scale_constant * np.log(w)
            d_req = self.cfg.target_params / ((self.cfg.theory_attn_constant + self.cfg.theory_expert_constant * num_experts) * w**2)
            if d_req <= d_crit * 1.2 and d_req >= 1:
                results.append((w, d_req, d_crit))

        if not results: return {"Error": "No valid config found"}
        results.sort(key=lambda x: x[0])
        best_w, best_d, d_crit_val = results[0]
        n_active = best_d * (self.cfg.theory_attn_constant + self.cfg.theory_expert_constant * self.cfg.num_routed_experts_active) * (best_w**2)
        
        return {
            "Target Params": f"{self.cfg.target_params/1e9:.0f}B",
            "Total Experts": num_experts,
            "Optimal Width (W)": int(best_w),
            "Optimal Layers (D)": int(np.ceil(best_d)),
            "Critical Depth Limit": round(d_crit_val, 2),
            "Theoretical Active Params": f"{n_active/1e9:.3f}B"
        }

# --- Execution ---
config = LightningConfig()

# 70B Configuration (with GSA)
config70b = LightningConfig(
    vocab_size=131072,
    hidden_size=4096,
    target_params=70e9,
    attention_type="gsa",  # 25% GSA layers
    deltanet_layer_ratio=0.75,  # 75% DeltaNet layers
    num_routed_experts_active=5,
    expert_intermediate_size=1024,
    shared_expert_intermediate_size=1280,
    enable_mtp=True,
    mtp_num_predictions=2,
)

# 8B Configuration
config8b = LightningConfig(
    vocab_size=131072,
    hidden_size=4096,
    target_params=8e9,
    attention_type="gsa",  # 25% GSA layers
    deltanet_layer_ratio=0.75,  # 75% DeltaNet layers
    num_routed_experts_active=2,
    expert_intermediate_size=1024,
    shared_expert_intermediate_size=1280,
    enable_mtp=True,
    mtp_num_predictions=2,
    num_experts_override=20,
    num_layers_override=20,
)

# 3B Configuration
config3b = LightningConfig(
    vocab_size=131072,
    hidden_size=4096,
    target_params=3e9,
    attention_type="gsa",  # 25% GSA layers
    deltanet_layer_ratio=0.75,  # 75% DeltaNet layers
    num_routed_experts_active=2,
    expert_intermediate_size=1024,
    shared_expert_intermediate_size=1280,
    enable_mtp=True,
    mtp_num_predictions=2,
    num_experts_override=20,
    num_layers_override=8,
)

# 1B Configuration (Dense - No MoE)
config1b = LightningConfig(
    vocab_size=131072,
    hidden_size=4096,
    target_params=1e9,
    attention_type="gsa",  # 25% GSA layers
    deltanet_layer_ratio=0.75,  # 75% DeltaNet layers
    num_routed_experts_active=0,
    num_shared_experts=0,  # No MoE, pure dense model
    expert_intermediate_size=1024,
    shared_expert_intermediate_size=1280,  # Acts as dense FFN when MoE is disabled
    enable_mtp=True,
    mtp_num_predictions=2,
    num_experts_override=0,
    num_layers_override=8,
)

config = config1b  # Default: 1B dense model

calc = LightningCalculator(config)

# Use expert override if provided, otherwise solve for optimal expert count
if config.num_experts_override is not None:
    e_total_solved = config.num_experts_override
    print(f"⚙️  Using manual expert override: {e_total_solved} total experts")
else:
    e_total_solved = calc.solve_for_experts()

report_df, solved_experts = calc.generate_report(e_total_solved)

print("\n=========================================\n")
print(f"--- [A] LightningLM1.0 Exact Calculation (Tied: {config.tie_word_embeddings}) ---")
print(report_df.to_string(index=False))
print("\n\n")
print("--- [B] Theoretical Optimal Dimensions (Standard Dense Experts) ---")
theo_stats = calc.calculate_moe_dimensions(solved_experts)
print(pd.DataFrame.from_dict(theo_stats, orient='index').to_string(header=False))
print("\n=========================================\n")