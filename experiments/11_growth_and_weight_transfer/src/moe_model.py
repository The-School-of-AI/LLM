"""
MoE-enabled SmolLM2 Model

Same architecture as model.py but with:
- FFN (MLP) replaced with Mixture-of-Experts block
- Simple linear router with Top-k selection
- Load balancing auxiliary loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

from .model import (
    SmolLM2Config,
    RMSNorm,
    Attention,
    MLP,
)


@dataclass
class MoEConfig(SmolLM2Config):
    """Extended config for MoE model."""
    num_experts: int = 4
    num_experts_per_tok: int = 2  # Top-k
    router_aux_loss_coef: float = 0.01  # Load balancing loss coefficient


class Router(nn.Module):
    """Simple linear router for MoE."""
    
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Returns router logits of shape (batch, seq_len, num_experts)."""
        return self.gate(hidden_states)


class MoEBlock(nn.Module):
    """Mixture-of-Experts block replacing the standard MLP."""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.router_aux_loss_coef = config.router_aux_loss_coef
        
        self.router = Router(config.hidden_size, config.num_experts)
        self.experts = nn.ModuleList([
            MLP(config) for _ in range(config.num_experts)
        ])
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE block.
        
        Returns:
            output: Tensor of shape (batch, seq_len, hidden_size)
            aux_loss: Auxiliary load balancing loss (scalar)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Get router logits and probabilities
        router_logits = self.router(hidden_states)  # (batch, seq_len, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Renormalize top-k probs
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute output
        # For simplicity, we iterate over experts (can be optimized with expert parallelism)
        final_output = torch.zeros_like(hidden_states)
        
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # (batch, seq_len)
            
            if expert_mask.any():
                # Get the tokens for this expert
                expert_input = hidden_states[expert_mask]  # (num_tokens, hidden_dim)
                expert_output = self.experts[expert_idx](expert_input)
                
                # Get the routing weight for this expert
                # Shape: (batch, seq_len, top_k) -> find where this expert appears
                expert_weight_mask = (top_k_indices == expert_idx)  # (batch, seq_len, top_k)
                expert_weights = (top_k_probs * expert_weight_mask.float()).sum(dim=-1)  # (batch, seq_len)
                
                # Weight and accumulate
                weighted_output = expert_output * expert_weights[expert_mask].unsqueeze(-1)
                final_output[expert_mask] += weighted_output
        
        # Compute auxiliary load balancing loss
        aux_loss = self._compute_aux_loss(router_probs, top_k_indices)
        
        return final_output, aux_loss
    
    def _compute_aux_loss(self, router_probs: torch.Tensor, top_k_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss.
        
        Encourages equal distribution of tokens across experts.
        Loss = num_experts * sum(f_i * P_i) where:
        - f_i = fraction of tokens routed to expert i
        - P_i = average routing probability to expert i
        """
        num_tokens = router_probs.shape[0] * router_probs.shape[1]
        
        # Fraction of tokens routed to each expert
        expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
        tokens_per_expert = expert_mask.sum(dim=(0, 1, 2)) / num_tokens  # (num_experts,)
        
        # Average routing probability per expert
        avg_probs = router_probs.mean(dim=(0, 1))  # (num_experts,)
        
        aux_loss = self.num_experts * (tokens_per_expert * avg_probs).sum()
        
        return aux_loss * self.router_aux_loss_coef


class MoETransformerBlock(nn.Module):
    """Transformer block with MoE instead of standard MLP."""
    
    def __init__(self, config: MoEConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe = MoEBlock(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # MoE with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, aux_loss = self.moe(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, aux_loss


class SmolLM2MoE(nn.Module):
    """SmolLM2 with Mixture-of-Experts."""
    
    def __init__(self, config: Optional[MoEConfig] = None):
        super().__init__()
        self.config = config or MoEConfig()
        
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList([
            MoETransformerBlock(self.config, layer_idx)
            for layer_idx in range(self.config.num_hidden_layers)
        ])
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        
        # LM head (optionally tied with embeddings)
        if self.config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
    
    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        hidden_states = self.embed_tokens(input_ids)
        
        total_aux_loss = 0.0
        for layer in self.layers:
            hidden_states, aux_loss = layer(hidden_states, attention_mask)
            total_aux_loss = total_aux_loss + aux_loss
        
        hidden_states = self.norm(hidden_states)
        
        # Compute logits
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden_states, self.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = lm_loss + total_aux_loss
        
        return {
            "loss": loss,
            "logits": logits,
            "aux_loss": total_aux_loss,
        }
    
    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count total parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Test MoE model creation
    config = MoEConfig()
    model = SmolLM2MoE(config)
    
    print(f"MoE Model config: {config}")
    print(f"Total parameters: {model.num_parameters():,}")
    
    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    output = model(input_ids, labels=labels)
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"Aux Loss: {output['aux_loss']:.4f}")
    print(f"Logits shape: {output['logits'].shape}")
