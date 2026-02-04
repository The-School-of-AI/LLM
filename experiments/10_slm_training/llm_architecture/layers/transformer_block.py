"""
Transformer Block
==================

Modular transformer layer that combines:
- Attention (GQA, GSA, or DeepSeek Sparse)
- FFN (SwiGLU)
- Normalization (RMSNorm)
- Connections (Residual or mHC)

Configuration-driven architecture selection.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Any

import sys
sys.path.append('..')

# Import all attention variants
from components.attention.grouped_query_attention import GroupedQueryAttention
from components.attention.gated_sparse_attention import GatedSparseAttention
from components.attention.deepseek_gsa import DeepSeekGSA, DeepSeekGSAConfig
from components.attention.deepseek_sparse_attention import DeepSeekSparseAttention

# Import FFN
from components.ffn.swiglu_ffn import SwiGLUFFN

# Import normalization
from components.normalization.rms_norm import RMSNorm

# Import connections
from components.connections.mhc import (
    ManifoldHyperConnection,
    SimplifiedMHC,
    ResidualConnection
)

# Import config
from config.model_config import (
    ModelConfig,
    AttentionType,
    FFNType,
    ConnectionType
)


class TransformerBlock(nn.Module):
    """
    Single transformer layer with configurable components.
    
    Architecture:
        Pre-norm: norm -> attention -> connection -> norm -> ffn -> connection
        
    Components are selected based on config:
    - Attention: GQA, GSA, or DeepSeek Sparse
    - FFN: SwiGLU (standard)
    - Connection: Residual or mHC
    """
    
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Pre-attention normalization
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Attention - select based on config
        self.attention = self._create_attention(config, layer_idx)
        
        # Post-attention normalization
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # FFN
        self.ffn = self._create_ffn(config)
        
        # Connections - select based on config
        self.attn_connection = self._create_connection(config)
        self.ffn_connection = self._create_connection(config)
        
    def _create_attention(self, config: ModelConfig, layer_idx: int) -> nn.Module:
        """Create attention module based on config."""
        attn_config = config.attention
        pos_config = config.position
        
        common_args = {
            'hidden_size': config.hidden_size,
            'num_attention_heads': attn_config.num_attention_heads,
            'num_key_value_heads': attn_config.num_key_value_heads,
            'head_dim': attn_config.head_dim,
            'max_position_embeddings': config.max_position_embeddings,
            'rope_theta': pos_config.rope_theta,
            'attention_dropout': attn_config.attention_dropout,
            'attention_bias': attn_config.attention_bias,
            'layer_idx': layer_idx
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
            # DeepSeek-style GSA with corrected implementation
            # Determine if YaRN should be used based on position config
            from config.model_config import PositionEmbeddingType
            use_yarn = pos_config.position_type == PositionEmbeddingType.YARN

            gsa_config = DeepSeekGSAConfig(
                hidden_size=config.hidden_size,
                num_attention_heads=attn_config.num_attention_heads,
                num_key_value_heads=attn_config.num_key_value_heads,
                head_dim=attn_config.head_dim,
                indexer_dim=attn_config.gsa_indexer_dim,
                num_indexer_heads=attn_config.gsa_num_indexer_heads,
                indexer_activation=getattr(attn_config, 'gsa_indexer_activation', 'sigmoid'),
                k_base=attn_config.gsa_k_base,
                k_min=attn_config.gsa_k_min,
                k_max=attn_config.gsa_k_max,
                use_adaptive_k=getattr(attn_config, 'gsa_use_adaptive_k', True),
                adaptive_k_method=getattr(attn_config, 'gsa_adaptive_k_method', 'variance'),
                adaptive_k_temperature=getattr(attn_config, 'gsa_adaptive_k_temperature', 1.0),
                use_value_gate=getattr(attn_config, 'gsa_use_value_gate', True),
                use_output_gate=getattr(attn_config, 'gsa_use_output_gate', True),
                gate_activation=getattr(attn_config, 'gsa_gate_activation', 'sigmoid'),
                gate_bias_init=getattr(attn_config, 'gsa_gate_bias_init', 0.5),
                max_position_embeddings=config.max_position_embeddings,
                rope_theta=pos_config.rope_theta,
                # YaRN configuration - passed from position config
                use_yarn=use_yarn,
                yarn_scale=pos_config.yarn_scale,
                yarn_original_max_position=pos_config.yarn_original_max_position,
                yarn_beta_fast=pos_config.yarn_beta_fast,
                yarn_beta_slow=pos_config.yarn_beta_slow,
                yarn_mscale=pos_config.yarn_mscale,
                yarn_mscale_all_dim=pos_config.yarn_mscale_all_dim,
                use_dynamic_yarn=False,  # Can be made configurable if needed
                attention_dropout=attn_config.attention_dropout,
                attention_bias=attn_config.attention_bias,
                num_layers=config.num_hidden_layers,
                layer_idx=layer_idx,
                # Triton kernel optimization
                use_triton_kernels=getattr(attn_config, 'gsa_use_triton_kernels', True),
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
    
    def _create_ffn(self, config: ModelConfig) -> nn.Module:
        """Create FFN module based on config."""
        ffn_config = config.ffn
        
        if ffn_config.ffn_type == FFNType.SWIGLU:
            return SwiGLUFFN(
                hidden_size=config.hidden_size,
                intermediate_size=ffn_config.intermediate_size,
                bias=ffn_config.ffn_bias,
                dropout=ffn_config.ffn_dropout
            )
        else:
            # Default to SwiGLU
            return SwiGLUFFN(
                hidden_size=config.hidden_size,
                intermediate_size=ffn_config.intermediate_size,
                bias=ffn_config.ffn_bias,
                dropout=ffn_config.ffn_dropout
            )
    
    def _create_connection(self, config: ModelConfig) -> nn.Module:
        """Create connection module based on config."""
        conn_config = config.connection
        
        if conn_config.connection_type == ConnectionType.RESIDUAL:
            return ResidualConnection(dropout=config.hidden_dropout)
            
        elif conn_config.connection_type == ConnectionType.MHC:
            return ManifoldHyperConnection(
                hidden_size=config.hidden_size,
                expansion_rate=conn_config.mhc_expansion_rate,
                alpha_init=getattr(conn_config, "mhc_alpha_init", 0.01),
                sinkhorn_iters=getattr(conn_config, "mhc_sinkhorn_iters", 20),
                dropout=config.hidden_dropout
            )
            
        else:
            return ResidualConnection(dropout=config.hidden_dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through transformer block.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: KV cache
            output_attentions: Return attention weights
            use_cache: Return updated cache
            
        Returns:
            hidden_states: Output tensor
            attention_weights: Optional attention weights
            past_key_value: Optional updated cache
        """
        residual = hidden_states
        
        # Pre-norm
        hidden_states = self.input_layernorm(hidden_states)
        
        # Attention
        attn_output, attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs
        )
        
        # Post-attention connection
        hidden_states = self.attn_connection(residual, attn_output)
        
        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        ffn_output = self.ffn(hidden_states)
        
        # Post-FFN connection
        hidden_states = self.ffn_connection(residual, ffn_output)
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (attn_weights,)
            
        if use_cache:
            outputs += (present_key_value,)
            
        return outputs


class TransformerBlockList(nn.Module):
    """
    List of transformer blocks with layer-wise configuration support.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through all layers.
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_cache = () if use_cache else None
        
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_cache += (layer_outputs[-1],)
                
            if output_attentions:
                all_attentions += (layer_outputs[1],)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        return (hidden_states, next_cache, all_hidden_states, all_attentions)
