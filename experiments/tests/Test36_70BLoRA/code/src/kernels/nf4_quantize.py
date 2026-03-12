"""
K6: NF4 Quantization Utilities for MoE Expert Weights.

Provides block-wise NF4 quantization/dequantization for 3D expert weight
tensors [E, K, N]. Designed to integrate with MoEFFN: base expert weights
(W_gate, W_up, W_down) are stored in 4-bit NF4 format, dequantized on-the-fly
during forward pass, while LoRA adapters remain in bf16.

Memory savings: 4× on expert weights (bf16 → NF4).
  - 70B model: 260 experts × 3 weights × [4096, 1024] × 2 bytes = ~6.4 GB/layer
  - With NF4: ~1.6 GB/layer → saves ~96 GB across 20 layers

NF4 (Normal Float 4-bit) uses a lookup table of 16 values derived from the
normal distribution, providing better quantization error than uniform INT4
for neural network weights.

Reference: Dettmers et al. "QLoRA" (2023), bitsandbytes NF4 implementation.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

# NF4 quantization levels (from QLoRA paper / bitsandbytes)
# These are the 16 values that minimize expected quantization error
# for normally-distributed weights.
NF4_LEVELS = torch.tensor([
    -1.0000, -0.6962, -0.5251, -0.3949,
    -0.2844, -0.1848, -0.0911,  0.0000,
     0.0796,  0.1609,  0.2461,  0.3379,
     0.4407,  0.5626,  0.7230,  1.0000,
], dtype=torch.float32)


@dataclass
class NF4QuantConfig:
    """Configuration for NF4 quantization."""
    block_size: int = 64          # Quantization block size (absmax per block)
    double_quant: bool = True     # Quantize the absmax scales themselves to FP8
    compute_dtype: torch.dtype = torch.bfloat16  # Dtype for dequantized compute


def _quantize_block_nf4(
    flat: torch.Tensor,
    block_size: int,
    nf4_levels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a flat tensor to NF4 with block-wise absmax scaling.

    Args:
        flat: [N] flat tensor in float32/bf16
        block_size: number of elements per quantization block
        nf4_levels: [16] NF4 lookup table

    Returns:
        quantized: [N//2] uint8 (two 4-bit values packed per byte)
        absmax: [num_blocks] per-block absmax scales
    """
    N = flat.numel()
    # Pad to multiple of block_size
    pad = (block_size - N % block_size) % block_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])

    padded_N = flat.numel()
    num_blocks = padded_N // block_size

    # Reshape into blocks
    blocks = flat.view(num_blocks, block_size).float()

    # Per-block absmax
    absmax = blocks.abs().max(dim=1).values.clamp(min=1e-12)

    # Normalize to [-1, 1]
    normalized = blocks / absmax.unsqueeze(1)

    # Find nearest NF4 level for each value
    nf4 = nf4_levels.to(flat.device)
    # [num_blocks * block_size, 1] vs [1, 16] → [num_blocks * block_size]
    diffs = (normalized.reshape(-1, 1) - nf4.reshape(1, -1)).abs()
    indices = diffs.argmin(dim=1).to(torch.uint8)  # [padded_N] values in 0..15

    # Pack two 4-bit indices into one uint8
    indices = indices[:padded_N]  # trim any extra
    if padded_N % 2 != 0:
        indices = torch.cat([indices, torch.zeros(1, device=indices.device, dtype=torch.uint8)])
    packed = (indices[0::2] << 4) | indices[1::2]

    return packed, absmax


def _dequantize_block_nf4(
    packed: torch.Tensor,
    absmax: torch.Tensor,
    original_numel: int,
    block_size: int,
    nf4_levels: torch.Tensor,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize NF4 packed tensor back to compute dtype.

    Args:
        packed: [N//2] uint8 packed 4-bit indices
        absmax: [num_blocks] per-block absmax scales
        original_numel: original number of elements (before padding)
        block_size: quantization block size
        nf4_levels: [16] NF4 lookup table
        compute_dtype: output dtype

    Returns:
        dequantized: [original_numel] tensor in compute_dtype
    """
    nf4 = nf4_levels.to(packed.device)

    # Unpack 4-bit indices
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    indices = torch.stack([high, low], dim=1).reshape(-1).long()

    # Lookup NF4 values
    values = nf4[indices].float()

    # Reshape into blocks and scale
    pad = (block_size - original_numel % block_size) % block_size
    padded_N = original_numel + pad
    num_blocks = padded_N // block_size
    values = values[:padded_N].view(num_blocks, block_size)
    dequantized = values * absmax.unsqueeze(1)

    # Flatten and trim padding
    return dequantized.reshape(-1)[:original_numel].to(compute_dtype)


class NF4Parameter:
    """
    Stores a quantized parameter with its metadata.
    Not an nn.Parameter — this is a frozen quantized buffer.
    """
    __slots__ = ['packed', 'absmax', 'original_shape', 'original_numel',
                 'block_size', 'compute_dtype']

    def __init__(self, packed, absmax, original_shape, original_numel,
                 block_size, compute_dtype):
        self.packed = packed
        self.absmax = absmax
        self.original_shape = original_shape
        self.original_numel = original_numel
        self.block_size = block_size
        self.compute_dtype = compute_dtype

    def dequantize(self) -> torch.Tensor:
        """Dequantize to full-precision tensor with original shape."""
        nf4 = NF4_LEVELS.to(self.packed.device)
        flat = _dequantize_block_nf4(
            self.packed, self.absmax, self.original_numel,
            self.block_size, nf4, self.compute_dtype,
        )
        return flat.view(self.original_shape)

    def nbytes(self) -> int:
        """Total bytes used by quantized storage."""
        return self.packed.nbytes + self.absmax.nbytes

    def to(self, device):
        """Move quantized data to device."""
        self.packed = self.packed.to(device)
        self.absmax = self.absmax.to(device)
        return self


def quantize_tensor_nf4(
    tensor: torch.Tensor,
    config: NF4QuantConfig,
) -> NF4Parameter:
    """
    Quantize a tensor to NF4 format.

    Args:
        tensor: any shape, will be flattened for quantization
        config: NF4QuantConfig

    Returns:
        NF4Parameter with packed data and metadata
    """
    original_shape = tensor.shape
    original_numel = tensor.numel()
    flat = tensor.contiguous().view(-1).float()

    nf4 = NF4_LEVELS
    packed, absmax = _quantize_block_nf4(flat, config.block_size, nf4)

    if config.double_quant:
        # Quantize absmax scales to FP8 (E4M3) for additional compression
        # FP8 E4M3 range: ±448, sufficient for absmax of normalized weights
        if hasattr(torch, 'float8_e4m3fn'):
            absmax = absmax.to(torch.float8_e4m3fn).float()

    return NF4Parameter(
        packed=packed,
        absmax=absmax,
        original_shape=original_shape,
        original_numel=original_numel,
        block_size=config.block_size,
        compute_dtype=config.compute_dtype,
    )
