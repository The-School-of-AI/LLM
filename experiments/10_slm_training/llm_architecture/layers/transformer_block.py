"""
Transformer Block
==================

Modular transformer layer that combines:
- Attention (GQA, GSA, or DeepSeek Sparse)
- FFN (SwiGLU)
- Normalization (RMSNorm)
- Connections (Residual or mHC)

Configuration-driven architecture selection.

When connection_type=MHC, the n-stream state [B, S, n, C] persists
across layers as the paper (arXiv:2512.24880v2) intends.  Expansion
happens once before the first layer and collapse once after the last.
"""

import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn

sys.path.append("..")

from components.attention.deepseek_gsa import DeepSeekGSA, DeepSeekGSAConfig
from components.attention.deepseek_sparse_attention import DeepSeekSparseAttention
from components.attention.gated_sparse_attention import GatedSparseAttention

# Import all attention variants
from components.attention.grouped_query_attention import GroupedQueryAttention

# Import FFN
from components.ffn.swiglu_ffn import SwiGLUFFN

# Import normalization
from components.normalization.rms_norm import RMSNorm

# Use Triton-optimized version when available
try:
    from components.kernels.triton_normalization import TritonRMSNorm

    USE_TRITON_NORM = True
except ImportError:
    USE_TRITON_NORM = False
    TritonRMSNorm = RMSNorm  # Fallback

# Import connections
from components.connections.mhc import MHCSublayerConnection, ResidualConnection

# Import config
from config.model_config import AttentionType, ConnectionType, ModelConfig

# =============================================================================
# Shared helpers for creating attention / FFN from config
# =============================================================================


def _create_attention(config: ModelConfig, layer_idx: int) -> nn.Module:
    """Create attention module based on config."""
    attn_config = config.attention
    pos_config = config.position

    common_args = {
        "hidden_size": config.hidden_size,
        "num_attention_heads": attn_config.num_attention_heads,
        "num_key_value_heads": attn_config.num_key_value_heads,
        "head_dim": attn_config.head_dim,
        "max_position_embeddings": config.max_position_embeddings,
        "rope_theta": pos_config.rope_theta,
        "attention_dropout": attn_config.attention_dropout,
        "attention_bias": attn_config.attention_bias,
        "layer_idx": layer_idx,
    }

    if attn_config.attention_type == AttentionType.GROUPED_QUERY:
        return GroupedQueryAttention(**common_args)

    elif attn_config.attention_type == AttentionType.GATED_SPARSE:
        return GatedSparseAttention(
            **common_args,
            indexer_dim=attn_config.gsa_indexer_dim,
            num_indexer_heads=attn_config.gsa_num_indexer_heads,
            k_base=attn_config.gsa_k_base,
            k_min=attn_config.gsa_k_min,
            k_max=attn_config.gsa_k_max,
        )

    elif attn_config.attention_type == AttentionType.DEEPSEEK_GSA:
        from config.model_config import PositionEmbeddingType

        use_yarn = pos_config.position_type == PositionEmbeddingType.YARN

        gsa_config = DeepSeekGSAConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=attn_config.num_attention_heads,
            num_key_value_heads=attn_config.num_key_value_heads,
            head_dim=attn_config.head_dim,
            indexer_dim=attn_config.gsa_indexer_dim,
            num_indexer_heads=attn_config.gsa_num_indexer_heads,
            indexer_activation=getattr(
                attn_config, "gsa_indexer_activation", "sigmoid"
            ),
            k_base=attn_config.gsa_k_base,
            k_min=attn_config.gsa_k_min,
            k_max=attn_config.gsa_k_max,
            use_adaptive_k=getattr(attn_config, "gsa_use_adaptive_k", True),
            adaptive_k_method=getattr(attn_config, "gsa_adaptive_k_method", "variance"),
            adaptive_k_temperature=getattr(
                attn_config, "gsa_adaptive_k_temperature", 1.0
            ),
            use_value_gate=getattr(attn_config, "gsa_use_value_gate", True),
            use_output_gate=getattr(attn_config, "gsa_use_output_gate", True),
            gate_activation=getattr(attn_config, "gsa_gate_activation", "sigmoid"),
            gate_bias_init=getattr(attn_config, "gsa_gate_bias_init", 0.5),
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=pos_config.rope_theta,
            use_yarn=use_yarn,
            yarn_scale=pos_config.yarn_scale,
            yarn_original_max_position=pos_config.yarn_original_max_position,
            yarn_beta_fast=pos_config.yarn_beta_fast,
            yarn_beta_slow=pos_config.yarn_beta_slow,
            yarn_mscale=pos_config.yarn_mscale,
            yarn_mscale_all_dim=pos_config.yarn_mscale_all_dim,
            use_dynamic_yarn=False,
            attention_dropout=attn_config.attention_dropout,
            attention_bias=attn_config.attention_bias,
            num_layers=config.num_hidden_layers,
            layer_idx=layer_idx,
            use_triton_kernels=getattr(attn_config, "gsa_use_triton_kernels", True),
        )
        return DeepSeekGSA(gsa_config)

    elif attn_config.attention_type == AttentionType.DEEPSEEK_SPARSE:
        return DeepSeekSparseAttention(
            **common_args,
            compressed_dim=attn_config.ds_compressed_dim,
            rope_head_dim=attn_config.ds_rope_head_dim,
            q_lora_rank=attn_config.ds_q_lora_rank,
        )

    else:
        raise ValueError(f"Unknown attention type: {attn_config.attention_type}")


def _create_ffn(config: ModelConfig) -> nn.Module:
    """Create FFN module based on config."""
    ffn_config = config.ffn
    return SwiGLUFFN(
        hidden_size=config.hidden_size,
        intermediate_size=ffn_config.intermediate_size,
        bias=ffn_config.ffn_bias,
        dropout=ffn_config.ffn_dropout,
    )


# =============================================================================
# Standard TransformerBlock (Residual connections)
# =============================================================================


class TransformerBlock(nn.Module):
    """
    Single transformer layer with standard residual connections.

    Architecture:
        Pre-norm: norm -> attention -> residual -> norm -> ffn -> residual

    Input / output shape: [batch, seq_len, hidden_size]
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.input_layernorm = TritonRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention = _create_attention(config, layer_idx)
        self.post_attention_layernorm = TritonRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.ffn = _create_ffn(config)

        self.attn_connection = ResidualConnection(dropout=config.hidden_dropout)
        self.ffn_connection = ResidualConnection(dropout=config.hidden_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        residual = hidden_states

        # Pre-norm + attention
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = self.attn_connection(residual, attn_output)

        # Pre-norm + FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.ffn_connection(residual, ffn_output)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


# =============================================================================
# MHC TransformerBlock (persistent n-stream, paper arXiv:2512.24880v2)
# =============================================================================


class MHCTransformerBlock(nn.Module):
    """
    Transformer layer with Manifold-Constrained Hyper-Connections.

    The n-stream state [B, S, n, C] flows through the entire model.
    Each sublayer (attention, FFN) has its own mHC module that:
      1. Aggregates n streams -> C  (H_pre @ x)
      2. Runs the sublayer on the aggregated C-dim input
      3. Updates the n-stream state  (H_res @ x + H_post^T @ F_output)

    This preserves the compositional H_res property across layers
    (paper Eq. 4) that is essential for mHC's benefit.

    Input / output shape: [batch, seq_len, n, hidden_size]
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        conn_config = config.connection

        # Norms operate on the C-dim aggregated input
        self.input_layernorm = TritonRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = TritonRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Sublayers
        self.attention = _create_attention(config, layer_idx)
        self.ffn = _create_ffn(config)

        # Per-sublayer mHC connections (each has its own H_pre, H_post, H_res)
        mhc_kwargs = dict(
            hidden_size=config.hidden_size,
            expansion_rate=conn_config.mhc_expansion_rate,
            alpha_init=conn_config.mhc_alpha_init,
            sinkhorn_iters=conn_config.mhc_sinkhorn_iters,
        )
        self.attn_mhc = MHCSublayerConnection(**mhc_kwargs)
        self.ffn_mhc = MHCSublayerConnection(**mhc_kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            hidden_states: [batch, seq_len, n, hidden_size]  (n-stream)
            ...

        Returns:
            hidden_states: [batch, seq_len, n, hidden_size]  (n-stream, updated)
            attn_weights, present_key_value (optional)
        """
        # --- Attention sublayer ---
        # Step 1: Aggregate n streams -> [B, S, C]
        attn_input, attn_cache = self.attn_mhc.get_layer_input(hidden_states)

        # Step 2: Pre-norm + attention (operates on C-dim)
        normed = self.input_layernorm(attn_input)
        attn_output, attn_weights, present_key_value = self.attention(
            hidden_states=normed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        # Step 3: mHC residual -> updated n-stream
        hidden_states = self.attn_mhc.apply_residual(
            hidden_states, attn_output, attn_cache
        )

        # --- FFN sublayer ---
        # Step 1: Aggregate n streams -> [B, S, C]
        ffn_input, ffn_cache = self.ffn_mhc.get_layer_input(hidden_states)

        # Step 2: Pre-norm + FFN (operates on C-dim)
        normed = self.post_attention_layernorm(ffn_input)
        ffn_output = self.ffn(normed)

        # Step 3: mHC residual -> updated n-stream
        hidden_states = self.ffn_mhc.apply_residual(
            hidden_states, ffn_output, ffn_cache
        )

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


# =============================================================================
# TransformerBlockList
# =============================================================================


class TransformerBlockList(nn.Module):
    """
    List of transformer blocks with layer-wise configuration support.

    When connection_type=MHC:
      - Expands [B, S, C] -> [B, S, n, C] once before the first layer
      - All layers operate on the persistent n-stream state
      - Collapses [B, S, n, C] -> [B, S, C] once after the last layer
      This matches the paper's Eq. (3)/(4) design.

    When connection_type=RESIDUAL:
      - Standard [B, S, C] throughout, no change from before.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.use_mhc = config.connection.connection_type == ConnectionType.MHC

        if self.use_mhc:
            self.expansion_rate = int(config.connection.mhc_expansion_rate)
            self.layers = nn.ModuleList(
                [
                    MHCTransformerBlock(config, layer_idx)
                    for layer_idx in range(config.num_hidden_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    TransformerBlock(config, layer_idx)
                    for layer_idx in range(config.num_hidden_layers)
                ]
            )

    def _expand(self, x: torch.Tensor) -> torch.Tensor:
        """Expand [B, S, C] -> [B, S, n, C] by replicating across streams."""
        return x.unsqueeze(2).expand(-1, -1, self.expansion_rate, -1).contiguous()

    def _collapse(self, x: torch.Tensor) -> torch.Tensor:
        """Collapse [B, S, n, C] -> [B, S, C] by averaging streams."""
        return x.mean(dim=2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through all layers.

        Input:  hidden_states [B, S, C]
        Output: hidden_states [B, S, C]  (collapsed if mHC)
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_cache = () if use_cache else None

        # --- Expand once for mHC ---
        if self.use_mhc:
            hidden_states = self._expand(hidden_states)

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                # Store collapsed C-dim view for compatibility
                h = self._collapse(hidden_states) if self.use_mhc else hidden_states
                all_hidden_states += (h,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_cache += (layer_outputs[-1],)

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        # --- Collapse once for mHC ---
        if self.use_mhc:
            hidden_states = self._collapse(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return (hidden_states, next_cache, all_hidden_states, all_attentions)
