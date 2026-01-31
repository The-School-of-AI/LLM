"""
Multi-Token Prediction Heads
=============================

Implementation of multi-token prediction as used in DeepSeek.

Key innovations:
1. Predict multiple future tokens simultaneously
2. Auxiliary training signal improves representations
3. Speculative decoding capability
4. Better sample efficiency

Reference: DeepSeek-V2/V3 architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any


class StandardLMHead(nn.Module):
    """
    Standard language model head.
    
    Single token prediction: hidden_states -> logits
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
        tie_weights: bool = True,
        embedding_weights: Optional[nn.Parameter] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tie_weights = tie_weights
        
        if tie_weights and embedding_weights is not None:
            # Share weights with embedding layer
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=bias)
            self.lm_head.weight = embedding_weights
        else:
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for next token prediction.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        return self.lm_head(hidden_states)


class MultiTokenPredictionHead(nn.Module):
    """
    Multi-Token Prediction (MTP) Head.
    
    Predicts multiple future tokens simultaneously using:
    1. Separate projection heads for each future position
    2. Shared hidden representations
    3. Position-aware prediction
    
    Benefits:
    - Improved representation learning
    - Auxiliary training signal
    - Enables speculative decoding
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_predict_tokens: int = 4,
        share_embeddings: bool = True,
        embedding_weights: Optional[nn.Parameter] = None,
        use_separate_heads: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_predict_tokens = num_predict_tokens
        self.use_separate_heads = use_separate_heads
        
        # Main prediction head (token t+1)
        self.main_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        if share_embeddings and embedding_weights is not None:
            self.main_head.weight = embedding_weights
        
        # Auxiliary heads for tokens t+2, t+3, ...
        if use_separate_heads:
            # Separate head for each future position
            self.aux_heads = nn.ModuleList([
                nn.Linear(hidden_size, vocab_size, bias=False)
                for _ in range(num_predict_tokens - 1)
            ])
            
            # Optionally share with main head
            if share_embeddings and embedding_weights is not None:
                for head in self.aux_heads:
                    head.weight = embedding_weights
        else:
            # Shared head with position embedding
            self.position_embed = nn.Embedding(num_predict_tokens, hidden_size)
            self.shared_aux_head = nn.Linear(hidden_size, vocab_size, bias=False)
            if share_embeddings and embedding_weights is not None:
                self.shared_aux_head.weight = embedding_weights
        
        # Transform from hidden to prediction space
        self.prediction_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        return_aux: bool = True
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Compute multi-token predictions.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            return_aux: Whether to return auxiliary predictions
            
        Returns:
            main_logits: [batch, seq_len, vocab_size] for t+1
            aux_logits: List of [batch, seq_len, vocab_size] for t+2, t+3, ...
        """
        # Main prediction (standard next token)
        main_logits = self.main_head(hidden_states)
        
        if not return_aux:
            return main_logits, None
        
        # Transform for auxiliary predictions
        transformed = self.prediction_transform(hidden_states)
        
        # Auxiliary predictions
        aux_logits = []
        
        if self.use_separate_heads:
            for head in self.aux_heads:
                aux_logits.append(head(transformed))
        else:
            # Use shared head with position-specific modifications
            for i in range(1, self.num_predict_tokens):
                pos_embed = self.position_embed.weight[i].unsqueeze(0).unsqueeze(0)
                pos_transformed = transformed + pos_embed
                aux_logits.append(self.shared_aux_head(pos_transformed))
        
        return main_logits, aux_logits


class MTPLoss(nn.Module):
    """
    Loss computation for Multi-Token Prediction.
    
    Combines main next-token loss with auxiliary future-token losses.
    """
    
    def __init__(
        self,
        num_predict_tokens: int = 4,
        aux_loss_weight: float = 0.3,
        aux_decay: float = 0.9,
        ignore_index: int = -100
    ):
        super().__init__()
        self.num_predict_tokens = num_predict_tokens
        self.aux_loss_weight = aux_loss_weight
        self.aux_decay = aux_decay
        self.ignore_index = ignore_index
        
    def forward(
        self,
        main_logits: torch.Tensor,
        aux_logits: Optional[List[torch.Tensor]],
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute MTP loss.
        
        Args:
            main_logits: [batch, seq_len, vocab_size]
            aux_logits: List of [batch, seq_len, vocab_size]
            labels: [batch, seq_len] - target token IDs
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        batch_size, seq_len = labels.shape
        
        # Main loss (t+1 prediction)
        # Shift: logits[:-1] predicts labels[1:]
        main_loss = F.cross_entropy(
            main_logits[:, :-1].contiguous().view(-1, main_logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=self.ignore_index
        )
        
        loss_dict = {'main_loss': main_loss}
        total_loss = main_loss
        
        # Auxiliary losses (t+2, t+3, ...)
        if aux_logits is not None:
            aux_total = 0.0
            
            for i, aux_log in enumerate(aux_logits):
                offset = i + 2  # t+2, t+3, ...
                
                if seq_len <= offset:
                    continue
                
                # Shift appropriately: logits[:-offset] predicts labels[offset:]
                aux_loss = F.cross_entropy(
                    aux_log[:, :-offset].contiguous().view(-1, aux_log.size(-1)),
                    labels[:, offset:].contiguous().view(-1),
                    ignore_index=self.ignore_index
                )
                
                # Decaying weight for further predictions
                weight = self.aux_loss_weight * (self.aux_decay ** i)
                aux_total += weight * aux_loss
                
                loss_dict[f'aux_loss_{offset}'] = aux_loss
            
            loss_dict['aux_total'] = aux_total
            total_loss = main_loss + aux_total
        
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict


class SpeculativeDecodingHead(nn.Module):
    """
    Head optimized for speculative decoding.
    
    Can predict multiple tokens in parallel for faster inference.
    Uses tree-based verification for acceptance.
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_speculative_tokens: int = 4,
        embedding_weights: Optional[nn.Parameter] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_speculative_tokens = num_speculative_tokens
        
        # Main LM head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if embedding_weights is not None:
            self.lm_head.weight = embedding_weights
        
        # Speculative prediction network
        self.spec_network = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Token embedding for speculative chain
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        speculative: bool = False,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional speculative tokens.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            speculative: Whether to generate speculative tokens
            temperature: Sampling temperature
            
        Returns:
            main_logits: Standard next-token logits
            spec_tokens: Speculative token IDs if speculative=True
        """
        main_logits = self.lm_head(hidden_states)
        
        if not speculative:
            return main_logits, None
        
        # Generate speculative tokens autoregressively
        batch_size = hidden_states.size(0)
        last_hidden = hidden_states[:, -1:, :]  # [batch, 1, hidden]
        
        spec_tokens = []
        current_hidden = last_hidden
        
        for _ in range(self.num_speculative_tokens):
            # Predict next token
            logits = self.lm_head(current_hidden[:, -1, :])  # [batch, vocab]
            
            # Sample (or argmax for greedy)
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)  # [batch, 1]
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            spec_tokens.append(next_token)
            
            # Embed token and update hidden
            token_embed = self.token_embed(next_token)  # [batch, 1, hidden]
            current_hidden, _ = self.spec_network(token_embed, current_hidden.transpose(0, 1))
            current_hidden = current_hidden.transpose(0, 1)
        
        spec_tokens = torch.cat(spec_tokens, dim=1)  # [batch, num_spec]
        
        return main_logits, spec_tokens
    
    def verify_speculative(
        self,
        hidden_states: torch.Tensor,
        spec_tokens: torch.Tensor,
        target_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        Verify speculative tokens against target model.
        
        Args:
            hidden_states: Hidden states from target model
            spec_tokens: Speculative token predictions
            target_logits: Logits from target model
            
        Returns:
            accepted_tokens: Verified tokens
            num_accepted: Number of accepted tokens
        """
        batch_size, num_spec = spec_tokens.shape
        
        # Get target predictions
        target_probs = F.softmax(target_logits, dim=-1)
        
        # Check each speculative token
        accepted = []
        for i in range(num_spec):
            spec_tok = spec_tokens[:, i]
            target_prob = target_probs[:, i].gather(1, spec_tok.unsqueeze(-1)).squeeze(-1)
            
            # Accept if probability is high enough (simplified acceptance)
            accept_mask = target_prob > 0.1
            
            if not accept_mask.all():
                break
            
            accepted.append(spec_tok)
        
        if accepted:
            accepted_tokens = torch.stack(accepted, dim=1)
            return accepted_tokens, len(accepted)
        else:
            return spec_tokens[:, :0], 0
