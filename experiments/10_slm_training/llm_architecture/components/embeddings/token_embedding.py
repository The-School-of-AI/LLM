"""
Token Embeddings
================

Standard token embedding layer with optional scaling.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Token embedding layer.

    Converts token IDs to dense vectors.
    Optionally applies embedding scaling (sqrt(hidden_size)).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        padding_idx: Optional[int] = None,
        scale_embeddings: bool = False,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.scale_embeddings = scale_embeddings

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)

        # Initialize
        self._init_weights(initializer_range)

        # Scaling factor
        self.scale = math.sqrt(hidden_size) if scale_embeddings else 1.0

    def _init_weights(self, initializer_range: float):
        """Initialize embedding weights."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=initializer_range)
        if self.embedding.padding_idx is not None:
            self.embedding.weight.data[self.embedding.padding_idx].zero_()

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]

        Returns:
            Embeddings of shape [batch_size, seq_len, hidden_size]
        """
        embeddings = self.embedding(input_ids)

        if self.scale_embeddings:
            embeddings = embeddings * self.scale

        return embeddings

    @property
    def weight(self) -> torch.Tensor:
        """Get embedding weight matrix (for weight tying)."""
        return self.embedding.weight
