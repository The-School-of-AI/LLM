"""
Expert Module
=============

Expert implementations for MoE including:
1. GatedExpert: SwiGLU FFN with optional dual gating (G1+G2)
2. NullExpert: Zero-compute pathway for junk token absorption
3. SharedExpert: Always-active expert for common patterns

Dual Gating (from GSA paper Section 3.5):
- G2 (Input Gate): V' = V ⊙ σ(h W_V^g)
  Suppresses uninformative dimensions BEFORE computation.
  
- G1 (Output Gate): O^gated = O ⊙ σ(h W_O^g)  
  Provides "do nothing" pathway AFTER computation.
  Analogous to attention sinks - model can suppress output
  without parking probability on early tokens.

Key Insight from GSA Paper:
"The output gate provides a learned pathway for suppressing
attention outputs, reducing reliance on sink tokens."

For MoE: Dual gating prevents router collapse by giving experts
a way to "do nothing" gracefully without needing to collapse
all routing to a single expert.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from model.config import MoEModelConfig, ExpertConfig


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.
    
    Formula: SwiGLU(x) = Swish(xW1) ⊙ (xW3)
    
    Where Swish(x) = x * σ(x)
    
    This provides better gradient flow than ReLU/GELU
    and the gated structure (W1 * W3) allows the network
    to learn what information to pass through.
    
    Reference: Shazeer (2020) "GLU Variants Improve Transformer"
    """
    
    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """
        Compute SwiGLU activation.
        
        Args:
            gate: Output of gate projection (xW1)
            up: Output of up projection (xW3)
            
        Returns:
            SwiGLU activation
        """
        return F.silu(gate) * up


class GatedExpert(nn.Module):
    """
    Expert FFN with SwiGLU and optional dual gating.
    
    Architecture:
        Input (hidden_size)
            ↓
        [G2: Input Gate] (optional)
            ↓
        W1 (gate) + W3 (up) → SwiGLU
            ↓
        W2 (down)
            ↓
        [G1: Output Gate] (optional)
            ↓
        Output (hidden_size)
    
    Dual gating from GSA paper:
    - G2: Suppresses uninformative input dimensions
    - G1: Provides "do nothing" pathway for output
    
    This helps prevent router collapse by allowing experts
    to gracefully handle tokens they shouldn't process.
    
    Args:
        hidden_size: Model hidden dimension
        intermediate_size: FFN intermediate dimension
        config: Expert configuration
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        config: Optional[ExpertConfig] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.config = config or ExpertConfig()
        
        # ============================================================
        # SwiGLU FFN Projections
        # ============================================================
        
        # Gate projection (W1): hidden → intermediate
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        
        # Down projection (W2): intermediate → hidden
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        # Up projection (W3): hidden → intermediate
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        
        # SwiGLU activation
        self.act = SwiGLU()
        
        # ============================================================
        # Dual Gating (G1 + G2) from GSA Paper
        # ============================================================
        
        self.use_dual_gating = self.config.use_dual_gating
        
        if self.use_dual_gating:
            # G2: Input gate
            # Paper: V' = V ⊙ σ(h_s W_V^g)
            # Suppress uninformative dimensions before processing
            self.input_gate = nn.Linear(hidden_size, hidden_size, bias=True)
            
            # G1: Output gate
            # Paper: O^gated = O^sparse ⊙ σ(h_t W_O^g)
            # Provides "do nothing" pathway
            self.output_gate = nn.Linear(hidden_size, hidden_size, bias=True)
            
            # Initialize gate biases for σ(·) ≈ 0.5 at start
            # This ensures gradients flow initially while still
            # introducing non-linearity from the first step
            nn.init.zeros_(self.input_gate.weight)
            nn.init.constant_(self.input_gate.bias, self.config.gate_bias_init)
            nn.init.zeros_(self.output_gate.weight)
            nn.init.constant_(self.output_gate.bias, self.config.gate_bias_init)
        
        # Initialize FFN weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small random values."""
        std = self.config.expert_init_std
        nn.init.normal_(self.w1.weight, std=std)
        nn.init.normal_(self.w2.weight, std=std)
        nn.init.normal_(self.w3.weight, std=std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through gated expert.
        
        Args:
            x: Input tensor [batch, seq, hidden] or [num_tokens, hidden]
            
        Returns:
            Output tensor, same shape as input
        """
        # Store original input for G1 (uses original, not gated)
        x_original = x
        
        # ============================================================
        # G2: Input Gating
        # Suppress uninformative dimensions before processing
        # ============================================================
        if self.use_dual_gating:
            g2 = torch.sigmoid(self.input_gate(x))
            x = x * g2
        
        # ============================================================
        # SwiGLU FFN
        # ============================================================
        gate = self.w1(x)
        up = self.w3(x)
        hidden = self.act(gate, up)
        output = self.w2(hidden)
        
        # ============================================================
        # G1: Output Gating
        # Provides "do nothing" pathway
        # Note: Uses ORIGINAL input, not gated input
        # ============================================================
        if self.use_dual_gating:
            g1 = torch.sigmoid(self.output_gate(x_original))
            output = output * g1
        
        return output
    
    @torch.no_grad()
    def get_gate_stats(self, x: torch.Tensor) -> dict:
        """Get gate activation statistics for analysis."""
        if not self.use_dual_gating:
            return {}
        
        g2 = torch.sigmoid(self.input_gate(x))
        g1 = torch.sigmoid(self.output_gate(x))
        
        return {
            'g1_mean': g1.mean().item(),
            'g1_std': g1.std().item(),
            'g1_min': g1.min().item(),
            'g1_max': g1.max().item(),
            'g2_mean': g2.mean().item(),
            'g2_std': g2.std().item(),
            'g2_min': g2.min().item(),
            'g2_max': g2.max().item(),
            # Effective "activity" - how much output is kept
            'effective_activity': (g1 * g2).mean().item(),
        }


class NullExpert(nn.Module):
    """
    Null Expert: Zero-compute pathway for junk token absorption.
    
    Purpose (from GSA paper insight on attention sinks):
    "When g_O ≈ 0, the gated output vanishes irrespective of
    where attention mass falls. The model can therefore learn
    to 'do nothing' without parking probability on early positions."
    
    For MoE:
    - Null expert provides explicit "do nothing" that competes in routing
    - Tokens that don't need expert processing can route here
    - Saves compute on junk tokens (padding, punctuation, stopwords)
    - Prevents router collapse (junk doesn't overwhelm real experts)
    
    Target null routing rates (from spec):
    - Junk tokens: 60-80% should route to null
    - Signal tokens: <10% should route to null
    
    Implementation:
    - Returns near-zero output (tiny scale for gradient flow)
    - Competes in routing pool with real experts
    - Router learns which tokens should go here via gradient signal
    
    How router learns null routing:
    - Important tokens → null hurts loss → gradient pushes away from null
    - Junk tokens → null doesn't hurt loss → no gradient pressure
    - Over time: junk routes to null, signal routes to real experts
    """
    
    def __init__(self, hidden_size: int, learnable_scale: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        
        if learnable_scale:
            # Tiny learnable scale for gradient flow
            # Allows some learning signal to flow through
            self.scale = nn.Parameter(torch.tensor(0.001))
        else:
            self.register_buffer('scale', torch.tensor(0.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return near-zero output.
        
        The tiny scale factor:
        1. Allows gradient to flow (for router learning)
        2. Effectively zero for actual computation
        3. Prevents numerical issues
        """
        return x * (self.scale * 0.01)
    
    @property
    def num_params(self) -> int:
        """Null expert has essentially 0 parameters."""
        return 1 if hasattr(self, 'scale') and isinstance(self.scale, nn.Parameter) else 0


class SharedExpert(GatedExpert):
    """
    Shared Expert: Always-active expert for common patterns.
    
    Shared experts handle patterns that apply to most tokens:
    - Common syntax (articles, prepositions)
    - Formatting patterns
    - Language structure
    
    Benefits:
    - Consistent handling regardless of routing decisions
    - Reduces router burden (doesn't need to route common patterns)
    - Provides baseline capacity even if routing fails
    
    Architecture:
    - Same as GatedExpert but processed for ALL tokens
    - NOT selected by router, always active
    - Can use smaller intermediate size (simpler patterns)
    
    Mathematical rationale:
    - 3B MoE: 2 shared / (2 + 2 active) = 50% shared capacity
    - 70B MoE: 4 shared / (4 + 4 active) = 50% shared capacity
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        config: Optional[ExpertConfig] = None
    ):
        super().__init__(hidden_size, intermediate_size, config)
        self.is_shared = True


class ExpertContainer(nn.Module):
    """
    Container for all experts in a layer.
    
    Manages:
    - Routed experts (selected by router)
    - Shared experts (always active)
    - Null experts (zero-compute pathway)
    
    Provides efficient batched expert computation.
    """
    
    def __init__(self, config: MoEModelConfig):
        super().__init__()
        self.config = config
        
        # ============================================================
        # Create Experts
        # ============================================================
        
        # Routed experts
        self.routed_experts = nn.ModuleList([
            GatedExpert(
                hidden_size=config.hidden_size,
                intermediate_size=config.expert.intermediate_size,
                config=config.expert
            )
            for _ in range(config.num_routed_experts)
        ])
        
        # Shared experts (if any)
        self.shared_experts = nn.ModuleList([
            SharedExpert(
                hidden_size=config.hidden_size,
                intermediate_size=config.expert.intermediate_size,
                config=config.expert
            )
            for _ in range(config.num_shared_experts)
        ])
        
        # Null experts
        self.null_experts = nn.ModuleList([
            NullExpert(config.hidden_size)
            for _ in range(config.num_null_experts)
        ])
        
        # Expert index boundaries
        self.routed_start = 0
        self.routed_end = config.num_routed_experts
        self.null_start = self.routed_end
        self.null_end = self.null_start + config.num_null_experts
    
    def get_expert(self, idx: int) -> nn.Module:
        """Get expert by global index (routed or null)."""
        if idx < self.routed_end:
            return self.routed_experts[idx]
        else:
            return self.null_experts[idx - self.null_start]
    
    def forward_shared(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute output from shared experts.
        
        All shared experts process ALL tokens.
        Outputs are averaged.
        """
        if len(self.shared_experts) == 0:
            return torch.zeros_like(x)
        
        outputs = [expert(x) for expert in self.shared_experts]
        return sum(outputs) / len(self.shared_experts)
    
    def forward_routed(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        gating_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute output from routed experts.
        
        Args:
            x: [batch, seq, hidden] input
            expert_indices: [batch, seq, top_k] selected expert IDs
            gating_weights: [batch, seq, top_k] weights for each expert
            
        Returns:
            [batch, seq, hidden] weighted expert outputs
        """
        batch_size, seq_len, hidden = x.shape
        top_k = expert_indices.shape[-1]
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Flatten for efficient processing
        flat_x = x.view(-1, hidden)  # [batch*seq, hidden]
        flat_indices = expert_indices.view(-1, top_k)  # [batch*seq, top_k]
        flat_weights = gating_weights.view(-1, top_k)  # [batch*seq, top_k]
        flat_output = torch.zeros_like(flat_x)
        
        # Process each k position
        for k in range(top_k):
            expert_ids = flat_indices[:, k]  # [batch*seq]
            weights = flat_weights[:, k:k+1]  # [batch*seq, 1]
            
            # Group tokens by expert for efficient batching
            unique_experts = expert_ids.unique()
            
            for expert_id in unique_experts:
                expert_id = expert_id.item()
                mask = (expert_ids == expert_id)
                
                if not mask.any():
                    continue

                if expert_id >= self.routed_end:
                    # Null experts are zero-compute; skip
                    continue
                
                # Get expert
                expert = self.get_expert(expert_id)
                
                # Process tokens for this expert
                expert_input = flat_x[mask]
                expert_output = expert(expert_input)
                
                # Weighted accumulation
                flat_output[mask] += weights[mask] * expert_output
        
        return flat_output.view(batch_size, seq_len, hidden)
    
    @torch.no_grad()
    def init_from_dense(self, dense_ffn: nn.Module, noise_std: float = 1e-4):
        """
        Initialize all routed experts from a dense FFN.
        
        Used for "explosion" expansion (1 FFN → N experts).
        
        Args:
            dense_ffn: Source dense FFN to copy from
            noise_std: Noise for symmetry breaking
        """
        for expert in self.routed_experts:
            # Copy weights
            expert.w1.weight.data.copy_(dense_ffn.w1.weight.data)
            expert.w2.weight.data.copy_(dense_ffn.w2.weight.data)
            expert.w3.weight.data.copy_(dense_ffn.w3.weight.data)
            
            # Add symmetry-breaking noise
            expert.w1.weight.data.add_(
                torch.randn_like(expert.w1.weight) * noise_std
            )
            expert.w2.weight.data.add_(
                torch.randn_like(expert.w2.weight) * noise_std
            )
            expert.w3.weight.data.add_(
                torch.randn_like(expert.w3.weight) * noise_std
            )
        
        # Copy to shared experts too
        for shared in self.shared_experts:
            shared.w1.weight.data.copy_(dense_ffn.w1.weight.data)
            shared.w2.weight.data.copy_(dense_ffn.w2.weight.data)
            shared.w3.weight.data.copy_(dense_ffn.w3.weight.data)
    
    @torch.no_grad()
    def expand_experts(
        self,
        parent_experts: 'ExpertContainer',
        children_per_parent: int = 8,
        noise_std: float = 1e-3
    ):
        """
        Initialize from parent experts via expansion.
        
        Used for expert expansion (8 → 64).
        Each parent expert becomes N children.
        
        Args:
            parent_experts: Source ExpertContainer with fewer experts
            children_per_parent: Number of children per parent
            noise_std: Noise for children divergence
        """
        num_parents = len(parent_experts.routed_experts)
        expected_children = num_parents * children_per_parent
        
        assert len(self.routed_experts) == expected_children, \
            f"Expected {expected_children} experts, got {len(self.routed_experts)}"
        
        for parent_idx in range(num_parents):
            parent = parent_experts.routed_experts[parent_idx]
            
            for child_idx in range(children_per_parent):
                global_idx = parent_idx * children_per_parent + child_idx
                child = self.routed_experts[global_idx]
                
                # Copy parent weights
                child.w1.weight.data.copy_(parent.w1.weight.data)
                child.w2.weight.data.copy_(parent.w2.weight.data)
                child.w3.weight.data.copy_(parent.w3.weight.data)
                
                # Copy dual gating if present
                if hasattr(parent, 'input_gate') and hasattr(child, 'input_gate'):
                    child.input_gate.weight.data.copy_(parent.input_gate.weight.data)
                    child.input_gate.bias.data.copy_(parent.input_gate.bias.data)
                    child.output_gate.weight.data.copy_(parent.output_gate.weight.data)
                    child.output_gate.bias.data.copy_(parent.output_gate.bias.data)
                
                # Add divergence noise
                child.w1.weight.data.add_(
                    torch.randn_like(child.w1.weight) * noise_std
                )
                child.w2.weight.data.add_(
                    torch.randn_like(child.w2.weight) * noise_std
                )
                child.w3.weight.data.add_(
                    torch.randn_like(child.w3.weight) * noise_std
                )
