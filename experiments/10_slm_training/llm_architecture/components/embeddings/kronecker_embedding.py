"""
Kronecker Product Embeddings
==============================

Byte-level embeddings using Kronecker products of byte and position embeddings.

Encodes tokens as:
    PF(token) = (1/sqrt(L)) * vec(sum_{i=1..L} e_byte[b_i] x e_pos[i])

Configuration:
- CHAR_DIM=256: All bytes 0-255
- POS_DIM=32: Max 32 bytes per token
- D=8192: Total embedding dimension (256 x 32)

Note: Cannot tie with lm_head (D=8192 != hidden_size=4096).
Requires pf_to_model projection (8192 -> hidden_size) and embed_norm.

Reference: Test_Code/model_1b.py lines 38-346
"""

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn


@dataclass
class KroneckerConfig:
    """
    Configuration for Byte-Level Kronecker Product Embeddings.

    Parameters:
    - CHAR_DIM: 256 (bytes 0-255)
    - POS_DIM: 32 (max 32 bytes per token)
    - D: 32 x 256 = 8192 dimensions
    """

    CHAR_DIM: int = 256
    POS_DIM: int = 32
    D: int = 8192
    length_normalize: bool = True
    truncate_long_words: bool = True

    def __post_init__(self):
        assert self.CHAR_DIM == 256, "CHAR_DIM must be 256 for byte-level encoding"
        assert (
            self.D == self.CHAR_DIM * self.POS_DIM
        ), f"D ({self.D}) must equal CHAR_DIM x POS_DIM ({self.CHAR_DIM} x {self.POS_DIM})"


class KroneckerEmbeddings:
    """
    Byte-Level Kronecker Product Embeddings.

    Encodes tokens using Kronecker product of UTF-8 byte and position embeddings.
    This is a CPU-side encoder/decoder (uses numpy, not torch).

    Properties:
    - Invertible: Can decode back to original token
    - Length-normalized: 1/sqrt(L) scaling
    - Universal: 100% coverage of all UTF-8 text
    """

    def __init__(self, cfg: KroneckerConfig):
        self.cfg = cfg
        self.CHAR_DIM = cfg.CHAR_DIM
        self.POS_DIM = cfg.POS_DIM
        self.D = cfg.D
        self.E_char = np.eye(self.CHAR_DIM, dtype=np.float32)
        self.P_pos = np.eye(self.POS_DIM, dtype=np.float32)

    def _utf8_safe_truncate(self, byte_seq: bytes, max_bytes: int) -> bytes:
        """Truncate byte sequence without splitting UTF-8 multibyte characters."""
        if len(byte_seq) <= max_bytes:
            return byte_seq
        for end in range(max_bytes, max(max_bytes - 4, 0) - 1, -1):
            try:
                byte_seq[:end].decode("utf-8")
                return byte_seq[:end]
            except UnicodeDecodeError:
                continue
        return b""

    def encode_word(self, word: str) -> np.ndarray:
        """
        Encode a single token to Kronecker embedding.

        Args:
            word: Input token (Unicode string)

        Returns:
            Embedding vector of shape (D,) = (8192,)
        """
        if word is None or word == "":
            return np.zeros((self.D,), dtype=np.float32)

        byte_seq = word.encode("utf-8")

        if len(byte_seq) > self.POS_DIM:
            if self.cfg.truncate_long_words:
                byte_seq = self._utf8_safe_truncate(byte_seq, self.POS_DIM)
            else:
                raise ValueError(
                    f"Token byte length {len(byte_seq)} exceeds POS_DIM={self.POS_DIM}"
                )

        L = len(byte_seq)
        if L == 0:
            return np.zeros((self.D,), dtype=np.float32)

        M = np.zeros((self.CHAR_DIM, self.POS_DIM), dtype=np.float32)
        for i, byte_val in enumerate(byte_seq):
            M[byte_val, i] = 1.0

        if self.cfg.length_normalize:
            M *= 1.0 / math.sqrt(L)

        return M.reshape(self.D)

    def decode_word(self, pf_vec: np.ndarray, threshold: float = 1e-6) -> str:
        """
        Decode Kronecker embedding back to token.

        Args:
            pf_vec: Embedding vector of shape (D,)
            threshold: Minimum magnitude to consider a position active

        Returns:
            Decoded token string
        """
        if pf_vec.shape != (self.D,):
            raise ValueError(f"pf_vec must have shape ({self.D},), got {pf_vec.shape}")

        M = pf_vec.reshape(self.CHAR_DIM, self.POS_DIM)
        col_norms = np.linalg.norm(M, axis=0)
        positions = [i for i, cn in enumerate(col_norms) if cn > threshold]

        bytes_list = []
        for i in positions:
            byte_val = int(np.argmax(M[:, i]))
            bytes_list.append(byte_val)

        byte_seq = bytes(bytes_list)
        try:
            return byte_seq.decode("utf-8")
        except UnicodeDecodeError:
            return byte_seq.decode("utf-8", errors="replace")

    def encode_batch(self, words: List[str]) -> np.ndarray:
        """Encode a batch of words."""
        return np.stack([self.encode_word(w) for w in words], axis=0)

    def decode_batch(self, pf_mat: np.ndarray, threshold: float = 1e-6) -> List[str]:
        """Decode a batch of embeddings."""
        return [self.decode_word(pf_mat[i], threshold) for i in range(pf_mat.shape[0])]


# Backward compatibility aliases
PFCodec = KroneckerEmbeddings
PFConfig = KroneckerConfig


class PureHybridEmbeddingTorch(nn.Module):
    """
    Pure Kronecker Product Embedding (GPU module).

    Pre-computes PF(word) for the entire vocabulary and stores as a buffer.
    At runtime: fetches PF vector, normalizes per-token, returns D-dim embeddings.

    Note: Embedding tying NOT possible (D=8192 != hidden_size=4096).
    Requires external pf_to_model projection and embed_norm.
    """

    def __init__(self, vocab_words: List[str], pf_codec: KroneckerEmbeddings):
        super().__init__()
        PF_table = pf_codec.encode_batch(vocab_words)
        PF_np = PF_table.astype(np.float32)
        pf_tensor = torch.from_numpy(PF_np).to(torch.bfloat16)
        self.register_buffer("PF_table", pf_tensor, persistent=True)

    def forward(self, token_ids):
        """
        Fetch and normalize Kronecker embeddings.

        Args:
            token_ids: Token indices (B, T)

        Returns:
            Normalized embeddings (B, T, D=8192)
        """
        PF = self.PF_table[token_ids].to(dtype=torch.float32)
        PF_centered = PF - PF.mean(dim=-1, keepdim=True)
        PF_std = PF_centered.std(dim=-1, keepdim=True) + 1e-6
        PFn = PF_centered / PF_std
        return PFn

    def module(self):
        """Return self (compatibility method)."""
        return self
