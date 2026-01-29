"""
MoE CUDA Kernels
================

High-performance CUDA kernels for MoE operations, inspired by DeepSeek's
implementation and Triton-based optimizations.

Kernels provided:
1. expert_scatter_kernel - Scatter tokens to experts based on routing
2. expert_gather_kernel - Gather expert outputs back to token positions
3. gating_kernel - Efficient sigmoid gating computation
4. topk_gating_kernel - Fused top-k selection with gating
5. load_balance_kernel - Bias update for loss-free load balancing

Performance Considerations:
- Memory coalescing for expert access patterns
- Warp-level primitives for reductions
- Shared memory for routing indices
- Stream-based expert parallelism

References:
- DeepSeek-V3: MoE kernel optimizations
- Megablocks: Efficient sparse MoE
- Triton: Python-native GPU programming
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import math

# Check for Triton availability
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton not available. Using PyTorch fallback kernels.")


# =============================================================================
# Triton Kernels (High Performance)
# =============================================================================

if TRITON_AVAILABLE:
    
    @triton.jit
    def sigmoid_gating_kernel(
        input_ptr,          # Input scores [batch*seq, num_experts]
        output_ptr,         # Output gating [batch*seq, num_experts]
        bias_ptr,           # Expert bias [num_experts]
        num_experts: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused sigmoid gating kernel.
        
        Computes: output = sigmoid(input + bias)
        
        Fused for memory efficiency - avoids intermediate tensor.
        """
        # Program ID
        pid = tl.program_id(0)
        
        # Calculate offsets
        row_start = pid * num_experts
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_experts
        
        # Load input and bias
        input_vals = tl.load(input_ptr + row_start + offsets, mask=mask, other=0.0)
        bias_vals = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        
        # Compute sigmoid(input + bias)
        adjusted = input_vals + bias_vals
        output_vals = tl.sigmoid(adjusted)
        
        # Store output
        tl.store(output_ptr + row_start + offsets, output_vals, mask=mask)
    
    
    @triton.jit
    def topk_gating_kernel(
        scores_ptr,           # Input scores [batch*seq, num_experts]
        indices_ptr,          # Output indices [batch*seq, k]
        weights_ptr,          # Output weights [batch*seq, k]
        adjusted_scores_ptr,  # Adjusted scores for selection [batch*seq, num_experts]
        original_scores_ptr,  # Original scores for gating [batch*seq, num_experts]
        num_experts: tl.constexpr,
        k: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused top-k selection with gating weights.
        
        Key insight from GSA paper:
        - Selection uses adjusted_scores (with bias)
        - Gating weights use original_scores (without bias)
        
        This separates load balancing (selection) from contribution (gating).
        """
        # Program ID
        pid = tl.program_id(0)
        
        # Calculate row offset
        row_start = pid * num_experts
        out_start = pid * k
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_experts
        
        # Load adjusted scores for selection
        adjusted = tl.load(adjusted_scores_ptr + row_start + offsets, mask=mask, other=-float('inf'))
        
        # Load original scores for gating
        original = tl.load(original_scores_ptr + row_start + offsets, mask=mask, other=0.0)
        
        # Simple top-k via sorting (for small k)
        # Note: Triton doesn't have native top-k, so we use iterative selection
        # In production, use custom CUDA kernel for large num_experts
        
        selected_indices = tl.zeros([k], dtype=tl.int32)
        selected_weights = tl.zeros([k], dtype=tl.float32)
        
        # Iteratively find top-k
        for i in range(k):
            # Find max
            max_val = tl.max(adjusted, axis=0)
            max_idx = tl.argmax(adjusted, axis=0)
            
            # Store
            selected_indices = tl.where(tl.arange(0, k) == i, max_idx, selected_indices)
            selected_weights = tl.where(tl.arange(0, k) == i, original[max_idx], selected_weights)
            
            # Mask out selected
            adjusted = tl.where(offsets == max_idx, -float('inf'), adjusted)
        
        # Normalize weights
        weight_sum = tl.sum(selected_weights, axis=0) + 1e-9
        selected_weights = selected_weights / weight_sum
        
        # Store results
        out_offsets = tl.arange(0, k)
        tl.store(indices_ptr + out_start + out_offsets, selected_indices, mask=out_offsets < k)
        tl.store(weights_ptr + out_start + out_offsets, selected_weights, mask=out_offsets < k)
    
    
    @triton.jit
    def expert_scatter_kernel(
        input_ptr,            # Input tokens [batch*seq, hidden]
        output_ptr,           # Output per expert [total_tokens, hidden]
        indices_ptr,          # Expert indices [batch*seq, k]
        token_positions_ptr,  # Token positions per expert [total_tokens]
        hidden_size: tl.constexpr,
        k: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Scatter tokens to expert input buffers.
        
        Each token is copied to k positions (one per selected expert).
        This enables batched expert computation.
        """
        # Program ID
        pid = tl.program_id(0)
        
        # Calculate token and expert selection index
        token_idx = pid // k
        selection_idx = pid % k
        
        # Get expert index and output position
        expert_idx = tl.load(indices_ptr + token_idx * k + selection_idx)
        out_pos = tl.load(token_positions_ptr + pid)
        
        # Copy hidden states
        for offset in range(0, hidden_size, BLOCK_SIZE):
            block_offsets = offset + tl.arange(0, BLOCK_SIZE)
            mask = block_offsets < hidden_size
            
            input_vals = tl.load(
                input_ptr + token_idx * hidden_size + block_offsets,
                mask=mask,
                other=0.0
            )
            
            tl.store(
                output_ptr + out_pos * hidden_size + block_offsets,
                input_vals,
                mask=mask
            )
    
    
    @triton.jit
    def expert_gather_kernel(
        expert_output_ptr,    # Expert outputs [total_tokens, hidden]
        final_output_ptr,     # Final output [batch*seq, hidden]
        indices_ptr,          # Expert indices [batch*seq, k]
        weights_ptr,          # Gating weights [batch*seq, k]
        token_positions_ptr,  # Token positions per expert [total_tokens]
        hidden_size: tl.constexpr,
        k: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Gather and combine expert outputs with gating weights.
        
        output[t] = Σᵢ weight[t,i] × expert_output[position[t,i]]
        """
        # Program ID - one block per token
        pid = tl.program_id(0)
        
        # Process hidden dimensions in blocks
        for offset in range(0, hidden_size, BLOCK_SIZE):
            block_offsets = offset + tl.arange(0, BLOCK_SIZE)
            mask = block_offsets < hidden_size
            
            # Accumulate weighted outputs
            accumulated = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            
            for i in range(k):
                weight = tl.load(weights_ptr + pid * k + i)
                pos = tl.load(token_positions_ptr + pid * k + i)
                
                expert_vals = tl.load(
                    expert_output_ptr + pos * hidden_size + block_offsets,
                    mask=mask,
                    other=0.0
                )
                
                accumulated += weight * expert_vals
            
            # Store result
            tl.store(
                final_output_ptr + pid * hidden_size + block_offsets,
                accumulated,
                mask=mask
            )
    
    
    @triton.jit
    def load_balance_bias_update_kernel(
        expert_counts_ptr,    # Expert usage counts [num_experts]
        bias_ptr,             # Expert biases [num_experts]
        total_tokens,         # Total tokens processed
        num_experts: tl.constexpr,
        update_speed,         # γ from DeepSeek-V3
        bias_min,
        bias_max,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Loss-free load balancing via bias adjustment.
        
        Algorithm:
        - target = total_tokens / num_experts
        - if count > 1.1 × target: decrease bias
        - if count < 0.9 × target: increase bias
        - clamp to [bias_min, bias_max]
        """
        # Program ID
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_experts
        
        # Load counts and current bias
        counts = tl.load(expert_counts_ptr + offsets, mask=mask, other=0.0)
        current_bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        
        # Compute target and adjustment
        target = total_tokens / num_experts
        
        # Adjust bias based on utilization
        adjustment = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        adjustment = tl.where(counts > target * 1.1, -update_speed, adjustment)
        adjustment = tl.where(counts < target * 0.9, update_speed, adjustment)
        
        # Update and clamp
        new_bias = current_bias + adjustment
        new_bias = tl.minimum(tl.maximum(new_bias, bias_min), bias_max)
        
        # Store
        tl.store(bias_ptr + offsets, new_bias, mask=mask)


# =============================================================================
# PyTorch Fallback Kernels (for CPU or when Triton unavailable)
# =============================================================================

class MoEKernelsPyTorch:
    """
    Pure PyTorch implementations of MoE kernels.
    
    These serve as:
    1. Fallback when CUDA/Triton not available
    2. Reference implementation for testing
    3. CPU inference support
    """
    
    @staticmethod
    def sigmoid_gating(
        scores: torch.Tensor,
        bias: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute sigmoid gating with bias.
        
        Args:
            scores: [batch*seq, num_experts] raw routing scores
            bias: [num_experts] expert biases
            
        Returns:
            gated_scores: [batch*seq, num_experts] sigmoid(scores + bias)
        """
        return torch.sigmoid(scores + bias)
    
    @staticmethod
    def topk_gating(
        adjusted_scores: torch.Tensor,
        original_scores: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Top-k selection with separate gating weights.
        
        Key insight: Selection uses adjusted scores, gating uses original.
        
        Args:
            adjusted_scores: [batch*seq, num_experts] scores + bias
            original_scores: [batch*seq, num_experts] original scores
            k: number of experts to select
            
        Returns:
            indices: [batch*seq, k] selected expert indices
            weights: [batch*seq, k] normalized gating weights
        """
        # Select top-k using adjusted scores
        _, indices = torch.topk(adjusted_scores, k=k, dim=-1)
        
        # Get gating weights from ORIGINAL scores
        weights = torch.gather(original_scores, -1, indices)
        
        # Normalize weights to sum to 1
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        return indices, weights
    
    @staticmethod
    def expert_scatter(
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        num_experts: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Scatter tokens to expert groups for batched processing.
        
        Args:
            hidden_states: [batch, seq, hidden] input tokens
            expert_indices: [batch, seq, k] selected expert indices
            num_experts: total number of experts
            
        Returns:
            expert_inputs: dict mapping expert_id -> [num_tokens, hidden]
            token_positions: mapping back to original positions
            expert_counts: [num_experts] count per expert
        """
        batch, seq, hidden = hidden_states.shape
        k = expert_indices.shape[-1]
        
        # Flatten
        flat_hidden = hidden_states.view(-1, hidden)  # [batch*seq, hidden]
        flat_indices = expert_indices.view(-1, k)     # [batch*seq, k]
        
        # Group tokens by expert
        expert_inputs = {}
        token_positions = {}
        expert_counts = torch.zeros(num_experts, device=hidden_states.device)
        
        for expert_id in range(num_experts):
            # Find all (token, selection) pairs routed to this expert
            mask = (flat_indices == expert_id)  # [batch*seq, k]
            
            if mask.any():
                # Get token indices (which tokens selected this expert)
                token_indices = mask.any(dim=-1).nonzero().squeeze(-1)
                
                if token_indices.numel() > 0:
                    expert_inputs[expert_id] = flat_hidden[token_indices]
                    token_positions[expert_id] = token_indices
                    expert_counts[expert_id] = token_indices.numel()
        
        return expert_inputs, token_positions, expert_counts
    
    @staticmethod
    def expert_gather(
        expert_outputs: dict,
        token_positions: dict,
        gating_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        output_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Gather expert outputs and combine with gating weights.
        
        Args:
            expert_outputs: dict mapping expert_id -> [num_tokens, hidden]
            token_positions: dict mapping expert_id -> token indices
            gating_weights: [batch*seq, k] weights for each selection
            expert_indices: [batch*seq, k] selected expert IDs
            output_shape: (batch, seq, hidden) for output tensor
            
        Returns:
            output: [batch, seq, hidden] weighted combination of expert outputs
        """
        batch, seq, hidden = output_shape
        k = gating_weights.shape[-1]
        
        # Initialize output
        output = torch.zeros(batch * seq, hidden, 
                            device=gating_weights.device,
                            dtype=list(expert_outputs.values())[0].dtype if expert_outputs else torch.float32)
        
        flat_indices = expert_indices.view(-1, k)
        flat_weights = gating_weights.view(-1, k)
        
        # Accumulate weighted outputs
        for expert_id, expert_out in expert_outputs.items():
            positions = token_positions[expert_id]
            
            # Find which selection slot used this expert
            for sel_idx in range(k):
                mask = flat_indices[positions, sel_idx] == expert_id
                if mask.any():
                    valid_positions = positions[mask]
                    valid_outputs = expert_out[mask]
                    valid_weights = flat_weights[valid_positions, sel_idx:sel_idx+1]
                    
                    output[valid_positions] += valid_weights * valid_outputs
        
        return output.view(batch, seq, hidden)
    
    @staticmethod
    def update_load_balance_bias(
        expert_counts: torch.Tensor,
        bias: torch.Tensor,
        total_tokens: int,
        update_speed: float = 0.001,
        bias_min: float = -2.0,
        bias_max: float = 2.0
    ) -> Tuple[torch.Tensor, dict]:
        """
        Update expert biases for loss-free load balancing.
        
        Algorithm from DeepSeek-V3:
        - Decrease bias for overloaded experts
        - Increase bias for underloaded experts
        - Clamp to prevent extreme values
        
        Args:
            expert_counts: [num_experts] usage counts
            bias: [num_experts] current biases
            total_tokens: total tokens processed
            update_speed: γ adjustment rate
            bias_min/max: clamp bounds
            
        Returns:
            new_bias: [num_experts] updated biases
            metrics: dict with load balancing metrics
        """
        num_experts = len(expert_counts)
        target = total_tokens / num_experts
        
        # Compute utilization
        utilization = expert_counts / (total_tokens + 1e-9)
        
        # Compute adjustments
        adjustments = torch.zeros_like(bias)
        adjustments[expert_counts > target * 1.1] = -update_speed
        adjustments[expert_counts < target * 0.9] = update_speed
        
        # Update and clamp
        new_bias = (bias + adjustments).clamp(bias_min, bias_max)
        
        # Compute metrics
        metrics = {
            'utilization_mean': utilization.mean().item(),
            'utilization_std': utilization.std().item(),
            'utilization_max': utilization.max().item(),
            'utilization_min': utilization.min().item(),
            'dead_experts': (utilization < 0.01).sum().item(),
            'overloaded_experts': (utilization > 1.0 / num_experts * 2).sum().item(),
            'total_adjustment': adjustments.abs().sum().item(),
        }
        
        return new_bias, metrics


# =============================================================================
# Unified Kernel Interface
# =============================================================================

class MoEKernels:
    """
    Unified interface for MoE kernels.
    
    Automatically selects Triton or PyTorch implementation
    based on availability and device.
    
    Usage:
        kernels = MoEKernels(device='cuda')
        
        # Gating
        gated = kernels.sigmoid_gating(scores, bias)
        
        # Top-k selection
        indices, weights = kernels.topk_gating(adjusted, original, k=2)
        
        # Scatter-gather for expert computation
        expert_in, positions, counts = kernels.scatter(hidden, indices, num_experts)
        output = kernels.gather(expert_out, positions, weights, indices, shape)
        
        # Load balancing
        new_bias, metrics = kernels.update_bias(counts, bias, total, speed)
    """
    
    def __init__(self, device: str = 'cuda', use_triton: bool = True):
        self.device = device
        self.use_triton = use_triton and TRITON_AVAILABLE and device == 'cuda'
        
        # PyTorch fallback
        self._pytorch = MoEKernelsPyTorch()
    
    def sigmoid_gating(
        self,
        scores: torch.Tensor,
        bias: torch.Tensor
    ) -> torch.Tensor:
        """Compute sigmoid gating with bias."""
        if self.use_triton and scores.is_cuda:
            # Use Triton kernel
            output = torch.empty_like(scores)
            num_tokens, num_experts = scores.shape
            
            grid = (num_tokens,)
            BLOCK_SIZE = triton.next_power_of_2(num_experts)
            
            sigmoid_gating_kernel[grid](
                scores, output, bias,
                num_experts,
                BLOCK_SIZE=BLOCK_SIZE
            )
            return output
        else:
            return self._pytorch.sigmoid_gating(scores, bias)
    
    def topk_gating(
        self,
        adjusted_scores: torch.Tensor,
        original_scores: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-k selection with gating weights."""
        # Always use PyTorch for now (Triton top-k is complex)
        return self._pytorch.topk_gating(adjusted_scores, original_scores, k)
    
    def scatter(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        num_experts: int
    ) -> Tuple[dict, dict, torch.Tensor]:
        """Scatter tokens to expert groups."""
        return self._pytorch.expert_scatter(hidden_states, expert_indices, num_experts)
    
    def gather(
        self,
        expert_outputs: dict,
        token_positions: dict,
        gating_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        output_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Gather and combine expert outputs."""
        return self._pytorch.expert_gather(
            expert_outputs, token_positions, gating_weights, 
            expert_indices, output_shape
        )
    
    def update_bias(
        self,
        expert_counts: torch.Tensor,
        bias: torch.Tensor,
        total_tokens: int,
        update_speed: float = 0.001,
        bias_min: float = -2.0,
        bias_max: float = 2.0
    ) -> Tuple[torch.Tensor, dict]:
        """Update load balancing biases."""
        if self.use_triton and bias.is_cuda:
            # Use Triton kernel
            num_experts = len(bias)
            BLOCK_SIZE = triton.next_power_of_2(num_experts)
            grid = ((num_experts + BLOCK_SIZE - 1) // BLOCK_SIZE,)
            
            load_balance_bias_update_kernel[grid](
                expert_counts, bias,
                total_tokens, num_experts,
                update_speed, bias_min, bias_max,
                BLOCK_SIZE=BLOCK_SIZE
            )
            
            # Compute metrics separately (for logging)
            utilization = expert_counts / (total_tokens + 1e-9)
            metrics = {
                'utilization_mean': utilization.mean().item(),
                'utilization_std': utilization.std().item(),
            }
            return bias, metrics
        else:
            return self._pytorch.update_load_balance_bias(
                expert_counts, bias, total_tokens,
                update_speed, bias_min, bias_max
            )


# =============================================================================
# Optimized Expert Parallel Computation
# =============================================================================

class ExpertParallelExecutor:
    """
    Execute experts in parallel with optimized memory access.
    
    Strategies:
    1. Group tokens by expert for batched matmul
    2. Use CUDA streams for parallel expert execution
    3. Fuse scatter-compute-gather when possible
    
    This is critical for MoE performance at scale.
    """
    
    def __init__(self, num_experts: int, device: str = 'cuda'):
        self.num_experts = num_experts
        self.device = device
        self.kernels = MoEKernels(device)
        
        # Create CUDA streams for parallel expert execution
        if device == 'cuda':
            self.streams = [torch.cuda.Stream() for _ in range(min(num_experts, 8))]
        else:
            self.streams = None
    
    def execute_experts(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        gating_weights: torch.Tensor,
        experts: torch.nn.ModuleList,
        num_experts: int
    ) -> torch.Tensor:
        """
        Execute experts with optimized parallelism.
        
        Args:
            hidden_states: [batch, seq, hidden]
            expert_indices: [batch, seq, k]
            gating_weights: [batch, seq, k]
            experts: ModuleList of expert modules
            num_experts: total number of experts
            
        Returns:
            output: [batch, seq, hidden]
        """
        batch, seq, hidden = hidden_states.shape
        
        # Scatter tokens to experts
        expert_inputs, positions, counts = self.kernels.scatter(
            hidden_states, expert_indices, num_experts
        )
        
        # Execute experts (potentially in parallel)
        expert_outputs = {}
        
        if self.streams and hidden_states.is_cuda:
            # Parallel execution with streams
            for i, (expert_id, expert_input) in enumerate(expert_inputs.items()):
                stream_idx = i % len(self.streams)
                with torch.cuda.stream(self.streams[stream_idx]):
                    expert_outputs[expert_id] = experts[expert_id](expert_input)
            
            # Synchronize
            torch.cuda.synchronize()
        else:
            # Sequential execution
            for expert_id, expert_input in expert_inputs.items():
                expert_outputs[expert_id] = experts[expert_id](expert_input)
        
        # Gather outputs
        output = self.kernels.gather(
            expert_outputs, positions, gating_weights,
            expert_indices, (batch, seq, hidden)
        )
        
        return output


# =============================================================================
# Testing Utilities
# =============================================================================

def test_kernels():
    """Test kernel implementations."""
    print("Testing MoE Kernels...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kernels = MoEKernels(device)
    
    # Test parameters
    batch, seq, hidden = 2, 16, 256
    num_experts = 8
    k = 2
    
    # Create test data
    scores = torch.randn(batch * seq, num_experts, device=device)
    bias = torch.zeros(num_experts, device=device)
    
    # Test sigmoid gating
    gated = kernels.sigmoid_gating(scores, bias)
    assert gated.shape == scores.shape
    assert (gated >= 0).all() and (gated <= 1).all()
    print("✓ Sigmoid gating")
    
    # Test top-k gating
    adjusted = gated + bias
    indices, weights = kernels.topk_gating(adjusted, gated, k)
    assert indices.shape == (batch * seq, k)
    assert weights.shape == (batch * seq, k)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch * seq, device=device), atol=1e-5)
    print("✓ Top-k gating")
    
    # Test scatter-gather
    hidden_states = torch.randn(batch, seq, hidden, device=device)
    expert_inputs, positions, counts = kernels.scatter(
        hidden_states, indices.view(batch, seq, k), num_experts
    )
    print(f"✓ Scatter: {len(expert_inputs)} experts have tokens")
    
    # Simulate expert computation
    expert_outputs = {eid: inp * 2 for eid, inp in expert_inputs.items()}
    
    output = kernels.gather(
        expert_outputs, positions, weights.view(batch, seq, k),
        indices.view(batch, seq, k), (batch, seq, hidden)
    )
    assert output.shape == hidden_states.shape
    print("✓ Gather")
    
    # Test bias update
    new_bias, metrics = kernels.update_bias(counts, bias, batch * seq * k)
    print(f"✓ Bias update: {metrics}")
    
    print("\nAll kernel tests passed!")


if __name__ == "__main__":
    test_kernels()
