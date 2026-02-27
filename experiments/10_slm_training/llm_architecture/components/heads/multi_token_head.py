"""
Language Model Heads
====================

Implementation of LM heads following GSA paper (arXiv:2601.15305v1) and
DeepSeek 3.2 (arXiv:2512.02556v1) architecture.

Design Decisions:
1. **Untied Embeddings**: Input and output embeddings are separate (not shared).
   This follows the GSA reference implementation which uses untied weights by
   default. Since FFN layers will grow while LM head stays the same, untied
   weights provide consistent behavior across model sizes.

2. **No Bias**: Modern LLMs don't use bias in the output projection.

3. **Proper Initialization**: Uses normal initialization with configurable std.

Reference: GSA (arXiv:2601.15305v1), DeepSeek 3.2 (arXiv:2512.02556v1)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LMHead(nn.Module):
    """
    Language Model Head with untied embeddings.

    Projects hidden states to vocabulary logits using an independent
    output projection (not tied to input embeddings).

    This follows DeepSeek V3's design choice for better quality and
    consistent behavior as models scale.

    Args:
        hidden_size: Model hidden dimension
        vocab_size: Vocabulary size
        bias: Whether to use bias (default: False, following modern LLMs)
        init_std: Standard deviation for weight initialization
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Independent output projection (untied from input embeddings)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=bias)

        # Initialize weights
        self._init_weights(init_std)

    def _init_weights(self, std: float):
        """Initialize with normal distribution."""
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)

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
    Multi-Token Prediction (MTP) Head following DeepSeek V3.

    Predicts multiple future tokens simultaneously using:
    1. Main head for t+1 prediction
    2. Auxiliary heads for t+2, t+3, ... predictions
    3. Transformation network for auxiliary predictions

    All heads use untied embeddings (independent from input embeddings).

    Benefits:
    - Improved representation learning
    - Auxiliary training signal
    - Enables speculative decoding

    Args:
        hidden_size: Model hidden dimension
        vocab_size: Vocabulary size
        num_predict_tokens: Number of future tokens to predict (default: 4)
        init_std: Standard deviation for weight initialization
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_predict_tokens: int = 4,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_predict_tokens = num_predict_tokens
        self.init_std = init_std

        # Main prediction head (token t+1) - untied
        self.main_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self._init_linear(self.main_head)

        # Auxiliary heads for tokens t+2, t+3, ... - each untied
        self.aux_heads = nn.ModuleList(
            [
                nn.Linear(hidden_size, vocab_size, bias=False)
                for _ in range(num_predict_tokens - 1)
            ]
        )
        for head in self.aux_heads:
            self._init_linear(head)

        # Transform from hidden to prediction space for auxiliary tokens
        self.prediction_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _init_linear(self, module: nn.Linear):
        """Initialize linear layer with normal distribution."""
        nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self, hidden_states: torch.Tensor, return_aux: bool = True
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
        aux_logits = [head(transformed) for head in self.aux_heads]

        return main_logits, aux_logits


class MTPLoss(nn.Module):
    """
    Loss computation for Multi-Token Prediction.

    Combines main next-token loss with auxiliary future-token losses.
    Uses decaying weights for further predictions.
    """

    def __init__(
        self,
        num_predict_tokens: int = 4,
        aux_loss_weight: float = 0.3,
        aux_decay: float = 0.9,
        ignore_index: int = -100,
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
        labels: torch.Tensor,
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
            ignore_index=self.ignore_index,
        )

        loss_dict = {"main_loss": main_loss}
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
                    ignore_index=self.ignore_index,
                )

                # Decaying weight for further predictions
                weight = self.aux_loss_weight * (self.aux_decay**i)
                aux_total += weight * aux_loss

                loss_dict[f"aux_loss_{offset}"] = aux_loss

            loss_dict["aux_total"] = aux_total
            total_loss = main_loss + aux_total

        loss_dict["total_loss"] = total_loss

        return total_loss, loss_dict


# Backward compatibility aliases
StandardLMHead = LMHead
