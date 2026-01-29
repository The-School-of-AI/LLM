"""
MoE CUDA Kernels Package
========================

High-performance kernels for MoE operations:

- moe_kernels.py: Triton and PyTorch kernel implementations
  - Sigmoid gating
  - Top-k selection
  - Expert scatter/gather
  - Load balance bias update

The kernels automatically select the best implementation
based on available hardware (Triton for CUDA, PyTorch fallback for CPU).
"""

from .moe_kernels import (
    MoEKernels,
    MoEKernelsPyTorch,
    ExpertParallelExecutor,
    TRITON_AVAILABLE,
)

__all__ = [
    'MoEKernels',
    'MoEKernelsPyTorch',
    'ExpertParallelExecutor',
    'TRITON_AVAILABLE',
]
