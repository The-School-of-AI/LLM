"""
MoE Transformer Model
=====================

Complete transformer model supporting both Dense and MoE configurations.

Supports all 4 stages of the growth cadence:
- Stage 1: 1B Dense (foundation)
- Stage 2: 3B MoE-8 (learn routing)
- Stage 3: 8B MoE-8 (scale dimensions)
- Stage 4: 70B MoE-64 (expand experts)

Architecture:
    Input IDs
        ↓
    Token Embedding
        ↓
    ┌─────────────────────────────────────┐
    │       Transformer Layer × N         │
    │  ┌─────────────────────────────┐   │
    │  │      RMSNorm + Attention    │   │
    │  │      (GQA with RoPE)        │   │
    │  └─────────────────────────────┘   │
    │              ↓                      │
    │  ┌─────────────────────────────┐   │
    │  │      RMSNorm + FFN          │   │
    │  │   (Dense or MoE Block)      │   │
    │  └─────────────────────────────┘   │
    └─────────────────────────────────────┘
        ↓
    RMSNorm
        ↓
    LM Head
        ↓
    Logits

Usage:
    from model.transformer import MoETransformer
    from configs.config_3b_moe import get_config
    
    config = get_config()
    model = MoETransformer(config)
    
    # Forward pass
    outputs = model(input_ids)
    logits = outputs['logits']
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Any
import math

from model.config import MoEModelConfig, ModelType
from model.attention import GQAttention, RMSNorm, create_causal_mask
from model.moe_block import MoEBlock, DenseFFN


class TransformerLayer(nn.Module):
    """
    Single transformer layer with attention and FFN/MoE.
    
    Architecture:
        x → RMSNorm → Attention → + → RMSNorm → FFN/MoE → +
        │_________________________↑   │_________________↑
                (residual)                 (residual)
    
    Args:
        config: Model configuration
        layer_idx: Layer index
        is_moe: Whether this layer uses MoE
    """
    
    def __init__(
        self,
        config: MoEModelConfig,
        layer_idx: int,
        is_moe: bool = True
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_moe = is_moe
        
        # Pre-attention normalization
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps
        )
        
        # Attention
        self.self_attn = GQAttention(config, layer_idx)
        
        # Post-attention normalization
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps
        )
        
        # FFN or MoE
        if is_moe and config.model_type == ModelType.MOE:
            self.ffn = MoEBlock(config, layer_idx)
        else:
            self.ffn = DenseFFN(config)
        
        # Dropout (usually 0)
        self.dropout = nn.Dropout(config.hidden_dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        token_ids: Optional[torch.Tensor] = None,
        return_router_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple], Optional[Dict]]:
        """
        Forward pass through transformer layer.
        
        Args:
            hidden_states: [batch, seq, hidden]
            attention_mask: Causal mask
            position_ids: Position IDs for RoPE
            past_key_value: KV cache
            use_cache: Return updated KV cache
            token_ids: Token IDs for null routing telemetry
            return_router_info: Return MoE routing information
            
        Returns:
            hidden_states: [batch, seq, hidden]
            past_key_value: Updated KV cache
            aux_info: MoE auxiliary information
        """
        residual = hidden_states
        
        # ============================================================
        # Attention Block
        # ============================================================
        
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # ============================================================
        # FFN/MoE Block
        # ============================================================
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        aux_info = None
        if self.is_moe and self.config.model_type == ModelType.MOE:
            hidden_states, aux_info = self.ffn(
                hidden_states,
                token_ids=token_ids,
                return_router_info=return_router_info
            )
        else:
            hidden_states = self.ffn(hidden_states)
        
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value, aux_info


class MoETransformer(nn.Module):
    """
    Complete MoE Transformer Model.
    
    Supports:
    - Dense configuration (Stage 1)
    - MoE configuration (Stages 2-4)
    - Configurable MoE layer frequency
    - KV caching for efficient inference
    - Comprehensive routing telemetry
    
    Args:
        config: Model configuration
    """
    
    def __init__(self, config: MoEModelConfig):
        super().__init__()
        self.config = config
        
        # ============================================================
        # Embeddings
        # ============================================================
        
        self.embed_tokens = nn.Embedding(
            config.tokenizer.vocab_size,
            config.hidden_size
        )
        
        # ============================================================
        # Transformer Layers
        # ============================================================
        
        self.layers = nn.ModuleList([
            TransformerLayer(
                config,
                layer_idx=i,
                is_moe=config.is_moe_layer(i)
            )
            for i in range(config.num_layers)
        ])
        
        # ============================================================
        # Final Normalization
        # ============================================================
        
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # ============================================================
        # Language Model Head
        # ============================================================
        
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.tokenizer.vocab_size,
            bias=False
        )
        
        # Optionally tie embeddings
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        # ============================================================
        # Tracking
        # ============================================================
        
        # Count MoE layers
        self.moe_layer_indices = [
            i for i in range(config.num_layers)
            if config.is_moe_layer(i) and config.model_type == ModelType.MOE
        ]
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Embeddings
        nn.init.normal_(self.embed_tokens.weight, std=self.config.initializer_range)
        
        # LM head
        if not self.config.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, std=self.config.initializer_range)
    
    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens
    
    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_router_info: bool = False,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: [batch, seq] input token IDs
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            past_key_values: KV cache from previous forward
            use_cache: Return KV cache for next forward
            return_router_info: Return MoE routing information
            labels: [batch, seq] labels for loss computation
            
        Returns:
            Dict with:
                - logits: [batch, seq, vocab_size]
                - loss: (if labels provided)
                - past_key_values: (if use_cache)
                - router_info: (if return_router_info)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # ============================================================
        # Embeddings
        # ============================================================
        
        hidden_states = self.embed_tokens(input_ids)
        
        # ============================================================
        # Attention Mask
        # ============================================================
        
        if attention_mask is None:
            # Create causal mask
            attention_mask = create_causal_mask(
                seq_len,
                device=device,
                dtype=hidden_states.dtype
            )
        
        # ============================================================
        # Position IDs
        # ============================================================
        
        if position_ids is None:
            if past_key_values is not None:
                # For cached generation, offset by cache length
                cache_len = past_key_values[0][0].shape[2]
                position_ids = torch.arange(
                    cache_len, cache_len + seq_len,
                    device=device
                ).unsqueeze(0).expand(batch_size, -1)
            else:
                position_ids = torch.arange(
                    seq_len, device=device
                ).unsqueeze(0).expand(batch_size, -1)
        
        # ============================================================
        # Transformer Layers
        # ============================================================
        
        all_router_info = [] if return_router_info else None
        present_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            
            hidden_states, present_kv, aux_info = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
                token_ids=input_ids,
                return_router_info=return_router_info
            )
            
            if use_cache:
                present_key_values.append(present_kv)
            
            if return_router_info and aux_info is not None:
                all_router_info.append(aux_info)
        
        # ============================================================
        # Final Normalization
        # ============================================================
        
        hidden_states = self.norm(hidden_states)
        
        # ============================================================
        # LM Head
        # ============================================================
        
        logits = self.lm_head(hidden_states)
        
        # ============================================================
        # Loss
        # ============================================================
        
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Cross entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        # ============================================================
        # Output
        # ============================================================
        
        output = {
            'logits': logits,
            'hidden_states': hidden_states
        }
        
        if loss is not None:
            output['loss'] = loss
        
        if use_cache:
            output['past_key_values'] = present_key_values
        
        if return_router_info:
            output['router_info'] = all_router_info
        
        return output
    
    def post_training_step(self) -> Dict[str, Any]:
        """
        Call after each training step.
        
        Updates MoE router biases and returns metrics.
        """
        metrics = {}
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer.ffn, 'post_training_step'):
                layer_metrics = layer.ffn.post_training_step()
                if layer_metrics.get('updated', False):
                    metrics[f'layer_{i}'] = layer_metrics
        
        return metrics
    
    def get_router_summary(self) -> str:
        """Get human-readable router summary."""
        if not self.moe_layer_indices:
            return "Dense model - no MoE layers"
        
        lines = [
            f"MoE Router Summary",
            f"=" * 40,
            f"Model: {self.config.model_name}",
            f"MoE Layers: {len(self.moe_layer_indices)}",
            f"Routed Experts: {self.config.num_routed_experts}",
            f"Shared Experts: {self.config.num_shared_experts}",
            f"Null Experts: {self.config.num_null_experts}",
            f"Top-K: {self.config.router.top_k}",
            ""
        ]
        
        # Get telemetry from first MoE layer
        if self.moe_layer_indices:
            layer_idx = self.moe_layer_indices[0]
            layer = self.layers[layer_idx]
            if hasattr(layer.ffn, 'telemetry'):
                lines.append(layer.ffn.telemetry.get_summary())
        
        return "\n".join(lines)
    
    @torch.no_grad()
    def init_from_dense(
        self,
        dense_model: 'MoETransformer',
        noise_std: float = 1e-4
    ):
        """
        Initialize MoE model from a dense model.
        
        Used for Stage 1 → Stage 2 (1B Dense → 3B MoE).
        
        Args:
            dense_model: Source dense model
            noise_std: Noise for symmetry breaking
        """
        # Copy embeddings
        self.embed_tokens.weight.data.copy_(dense_model.embed_tokens.weight.data)
        
        # Copy LM head (if not tied)
        if not self.config.tie_word_embeddings:
            self.lm_head.weight.data.copy_(dense_model.lm_head.weight.data)
        
        # Copy layers
        for i, (moe_layer, dense_layer) in enumerate(zip(self.layers, dense_model.layers)):
            # Copy attention
            moe_layer.self_attn.q_proj.weight.data.copy_(dense_layer.self_attn.q_proj.weight.data)
            moe_layer.self_attn.k_proj.weight.data.copy_(dense_layer.self_attn.k_proj.weight.data)
            moe_layer.self_attn.v_proj.weight.data.copy_(dense_layer.self_attn.v_proj.weight.data)
            moe_layer.self_attn.o_proj.weight.data.copy_(dense_layer.self_attn.o_proj.weight.data)
            
            # Copy norms
            moe_layer.input_layernorm.weight.data.copy_(dense_layer.input_layernorm.weight.data)
            moe_layer.post_attention_layernorm.weight.data.copy_(dense_layer.post_attention_layernorm.weight.data)
            
            # Initialize MoE from dense FFN
            if moe_layer.is_moe:
                moe_layer.ffn.init_from_dense(dense_layer.ffn, noise_std)
            else:
                # Copy dense FFN directly
                moe_layer.ffn.w1.weight.data.copy_(dense_layer.ffn.w1.weight.data)
                moe_layer.ffn.w2.weight.data.copy_(dense_layer.ffn.w2.weight.data)
                moe_layer.ffn.w3.weight.data.copy_(dense_layer.ffn.w3.weight.data)
        
        # Copy final norm
        self.norm.weight.data.copy_(dense_model.norm.weight.data)
    
    @torch.no_grad()
    def expand_experts(
        self,
        source_model: 'MoETransformer',
        children_per_parent: int = 8,
        noise_std: float = 1e-3
    ):
        """
        Expand experts from a smaller MoE model.
        
        Used for Stage 3 → Stage 4 (8B MoE-8 → 70B MoE-64).
        
        Args:
            source_model: Source MoE model with fewer experts
            children_per_parent: Number of children per parent expert
            noise_std: Noise for child divergence
        """
        # This is more complex as layer counts may differ
        # For now, handle the case where we have same number of layers
        
        if len(self.layers) != len(source_model.layers):
            raise NotImplementedError(
                "Expert expansion with different layer counts not yet implemented. "
                "Add layer interpolation logic."
            )
        
        # Copy embeddings
        self.embed_tokens.weight.data.copy_(source_model.embed_tokens.weight.data)
        
        # Copy LM head
        if not self.config.tie_word_embeddings:
            self.lm_head.weight.data.copy_(source_model.lm_head.weight.data)
        
        # Copy and expand layers
        for moe_layer, source_layer in zip(self.layers, source_model.layers):
            # Copy attention
            moe_layer.self_attn.q_proj.weight.data.copy_(source_layer.self_attn.q_proj.weight.data)
            moe_layer.self_attn.k_proj.weight.data.copy_(source_layer.self_attn.k_proj.weight.data)
            moe_layer.self_attn.v_proj.weight.data.copy_(source_layer.self_attn.v_proj.weight.data)
            moe_layer.self_attn.o_proj.weight.data.copy_(source_layer.self_attn.o_proj.weight.data)
            
            # Copy norms
            moe_layer.input_layernorm.weight.data.copy_(source_layer.input_layernorm.weight.data)
            moe_layer.post_attention_layernorm.weight.data.copy_(source_layer.post_attention_layernorm.weight.data)
            
            # Expand MoE
            if moe_layer.is_moe and source_layer.is_moe:
                moe_layer.ffn.expand_from_moe(
                    source_layer.ffn,
                    children_per_parent,
                    noise_std
                )
        
        # Copy final norm
        self.norm.weight.data.copy_(source_model.norm.weight.data)
    
    @property
    def num_parameters(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: MoEModelConfig) -> MoETransformer:
    """
    Create model from configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        MoETransformer model
    """
    return MoETransformer(config)


def load_model(
    checkpoint_path: str,
    config: Optional[MoEModelConfig] = None,
    device: str = 'cuda'
) -> MoETransformer:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        config: Optional config override
        device: Device to load to
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if config is None:
        config = MoEModelConfig.from_dict(checkpoint['config'])
    
    model = MoETransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model.to(device)
