"""
Lightning Decoder Layer + MTP Transformer Block
==================================================

Hybrid DeltaNet + GSA decoder layer with mHC connections,
and a full transformer block for Multi-Token Prediction.

Matching Test_Code/model_1b.py lines 1155-1336.

Components:
- LightningDecoderLayer: Hybrid DeltaNet/GSA with mHC wrappers and force() for reversible integration
- MTPTransformerBlock: Full transformer block for MTP (fusion + DeltaNet + MLP + mHC)
"""

import torch
import torch.nn as nn
from components.attention.gated_deltanet import GatedDeltaNet
from components.attention.reference_gsa import ReferenceGSA
from components.connections.mhc_v2 import MHCCoeffsV2, MHCSublayerV2, RMSNorm
from components.ffn.moe_ffn import LightningMLP, MoEFFN, MoEGate

# ============================================================================
# Decoder Layer (Hybrid DeltaNet + GSA)
# ============================================================================


class LightningDecoderLayer(nn.Module):
    """
    Decoder layer that can be either DeltaNet or GSA.
    Type is determined at initialization.

    Has force() method for ReversibleMidpointStack integration.
    """

    def __init__(self, config, layer_type: str):
        """
        Args:
            config: ModelConfig or equivalent with required attributes
            layer_type: "deltanet" or "gsa"
        """
        super().__init__()
        self.layer_type = layer_type
        self.n_streams = config.connection.mhc_expansion_rate

        # Extract config values
        hidden_size = config.hidden_size
        attn_config = config.attention
        pos_config = config.position
        ffn_config = config.ffn
        conn_config = config.connection

        # Select attention mechanism
        if layer_type == "deltanet":
            attn = GatedDeltaNet(
                hidden_size=hidden_size,
                num_heads=attn_config.delta_v_heads,
                head_dim=attn_config.delta_head_dim,
                max_seq_len=config.max_position_embeddings,
                rope_base=pos_config.rope_theta,
                rope_original_max=pos_config.yarn_original_max_position,
                rope_scaling_factor=pos_config.rope_scaling_factor,
                conv_size=4,
                use_output_norm=True,
            )
        elif layer_type == "gsa":
            attn = ReferenceGSA(
                hidden_size=hidden_size,
                num_heads=attn_config.gsa_num_heads,
                max_seq_len=config.max_position_embeddings,
                rope_base=pos_config.rope_theta,
                k_base=attn_config.gsa_k_base,
                k_min=attn_config.gsa_k_min,
                k_max=attn_config.gsa_k_max,
                indexer_heads=attn_config.gsa_num_indexer_heads,
                rope_original_max=pos_config.yarn_original_max_position,
                rope_scaling_factor=pos_config.rope_scaling_factor,
                use_triton_kernels=getattr(attn_config, "gsa_use_triton_kernels", True),
                sparse_backend=getattr(attn_config, "gsa_sparse_backend", "auto"),
                triton_min_seq_len=getattr(attn_config, "gsa_triton_min_seq_len", 512),
                prefer_flash=getattr(attn_config, "gsa_prefer_flash", True),
                sdpa_chunk_size=getattr(attn_config, "gsa_sdpa_chunk_size", 16),
            )
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        # MLP (MoE or dense shared expert)
        mlp = LightningMLP(
            hidden_size=hidden_size,
            intermediate_size=ffn_config.intermediate_size,
            num_experts=ffn_config.moe_num_experts,
            num_shared_experts=1,
            top_k=ffn_config.moe_num_experts_per_tok,
            data_sparsity=getattr(ffn_config, "moe_data_sparsity", 0.5),
            expert_intermediate_size=getattr(
                ffn_config, "moe_expert_intermediate_size", None
            ),
        )

        # mHC Wrappers (norm is INSIDE MHCSublayerV2)
        sinkhorn_iters = conn_config.mhc_sinkhorn_iters

        self.attn_block = MHCSublayerV2(
            d_model=hidden_size,
            n_streams=self.n_streams,
            sublayer=attn,
            norm=RMSNorm(hidden_size),
            iters=sinkhorn_iters,
        )

        self.mlp_block = MHCSublayerV2(
            d_model=hidden_size,
            n_streams=self.n_streams,
            sublayer=mlp,
            norm=RMSNorm(hidden_size),
            iters=sinkhorn_iters,
        )

    def force(self, x):
        """
        Compute residual delta for reversible integration.

        Required by ReversibleMidpointStack.

        Args:
            x: (B, T, n_streams, D)

        Returns:
            delta: (B, T, n_streams, D) - the residual change
            aux: Scalar auxiliary loss
        """
        h, aux1 = self.attn_block(x, attention_mask=None)
        out, aux2 = self.mlp_block(h, attention_mask=None)

        delta = out - x

        aux = None
        if aux1 is not None:
            aux = aux1
        if aux2 is not None:
            if aux is None:
                aux = aux2
            else:
                aux = aux + aux2

        if aux is None:
            aux = x.new_zeros((), dtype=torch.float32)

        return delta, aux

    def forward(self, x_stream, attention_mask=None):
        """
        Standard forward pass.

        Args:
            x_stream: (B, T, n_streams, D)
            attention_mask: Optional

        Returns:
            x_stream: (B, T, n_streams, D)
            total_aux: Scalar auxiliary loss
        """
        x_stream, aux1 = self.attn_block(x_stream, attention_mask=attention_mask)
        x_stream, aux2 = self.mlp_block(x_stream, attention_mask=None)

        total_aux = None
        if aux1 is not None or aux2 is not None:
            total_aux = (aux1 if aux1 is not None else 0) + (
                aux2 if aux2 is not None else 0
            )

        return x_stream, total_aux


# ============================================================================
# Multi-Token Prediction Block (Full Transformer)
# ============================================================================


class MTPTransformerBlock(nn.Module):
    """
    MTP block for predicting t+2 from [h_t; emb_{t+1}].

    This is a FULL transformer block (not just linear projections):
    - Fusion: concat([h_t, emb_{t+1}]) -> linear -> fused representation
    - DeltaNet attention with mHC wrapper
    - MLP with mHC wrapper
    - Sparse stream init, collapse by mean

    Returns hidden states (NOT logits - lm_head is applied externally).

    Reference: Test_Code/model_1b.py lines 1256-1336
    """

    def __init__(self, config):
        """
        Args:
            config: ModelConfig with required attributes
        """
        super().__init__()

        hidden_size = config.hidden_size
        attn_config = config.attention
        pos_config = config.position
        ffn_config = config.ffn
        conn_config = config.connection

        self.n_streams = conn_config.mhc_expansion_rate
        self.hidden_size = hidden_size

        # Fusion layer: [h_t; emb_{t+1}] -> hidden_size
        self.fusion_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        # Core sublayers (using DeltaNet for efficiency)
        attn = GatedDeltaNet(
            hidden_size=hidden_size,
            num_heads=attn_config.delta_v_heads,
            head_dim=attn_config.delta_head_dim,
            max_seq_len=config.max_position_embeddings,
            rope_base=pos_config.rope_theta,
            rope_original_max=pos_config.yarn_original_max_position,
            rope_scaling_factor=pos_config.rope_scaling_factor,
            conv_size=4,
            use_output_norm=True,
        )

        mlp = LightningMLP(
            hidden_size=hidden_size,
            intermediate_size=ffn_config.intermediate_size,
            num_experts=ffn_config.moe_num_experts,
            num_shared_experts=1,
            top_k=ffn_config.moe_num_experts_per_tok,
            data_sparsity=getattr(ffn_config, "moe_data_sparsity", 0.5),
            expert_intermediate_size=getattr(
                ffn_config, "moe_expert_intermediate_size", None
            ),
        )

        sinkhorn_iters = conn_config.mhc_sinkhorn_iters

        # mHC Wrappers
        self.attn_block = MHCSublayerV2(
            d_model=hidden_size,
            n_streams=self.n_streams,
            sublayer=attn,
            norm=RMSNorm(hidden_size),
            iters=sinkhorn_iters,
        )

        self.mlp_block = MHCSublayerV2(
            d_model=hidden_size,
            n_streams=self.n_streams,
            sublayer=mlp,
            norm=RMSNorm(hidden_size),
            iters=sinkhorn_iters,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (MoEFFN, MoEGate, MHCCoeffsV2)):
            return
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, h_t, next_emb, attention_mask=None):
        """
        Forward pass for MTP.

        Args:
            h_t: Hidden states from backbone (B, T, hidden_size)
            next_emb: Embeddings of next tokens (B, T, hidden_size)
            attention_mask: Optional

        Returns:
            x_out: Hidden states for MTP prediction (B, T, hidden_size)
        """
        batch_size, seq_len, _ = h_t.shape

        # Fuse hidden states with next-token embeddings
        x = torch.cat([h_t, next_emb], dim=-1)
        x = self.fusion_proj(x)

        # Sparse stream initialization (only stream 0 gets input)
        x_stream = torch.zeros(
            batch_size,
            seq_len,
            self.n_streams,
            self.hidden_size,
            device=x.device,
            dtype=x.dtype,
        )
        x_stream[:, :, 0, :] = x

        # mHC blocks (ignore aux_loss)
        x_stream, _ = self.attn_block(x_stream, attention_mask=attention_mask)
        x_stream, _ = self.mlp_block(x_stream, attention_mask=None)

        # Collapse streams by mean
        x_out = x_stream.mean(dim=2)

        return x_out
