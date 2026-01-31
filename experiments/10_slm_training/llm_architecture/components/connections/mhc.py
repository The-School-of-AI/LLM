"""
Manifold-Constrained Hyper-Connections (mHC)
=============================================

Implementation based on paper: arXiv:2512.24880
"Manifold-Constrained Hyper-Connections for Efficient Deep Networks"

Key innovations:
1. Replaces standard residual connections
2. Dynamic weighting based on manifold constraints
3. Multi-path information flow
4. Improved gradient flow and representation learning

mHC achieves:
- Better gradient flow than residual connections
- Improved representation quality
- Minimal computational overhead
- Compatible with any transformer architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class ManifoldProjection(nn.Module):
    """
    Projects representations onto a learned manifold.
    
    The manifold constraint helps maintain geometric structure
    of representations during forward pass.
    """
    
    def __init__(
        self,
        hidden_size: int,
        manifold_dim: int,
        num_projections: int = 4
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.manifold_dim = manifold_dim
        self.num_projections = num_projections
        
        # Manifold basis vectors
        self.manifold_basis = nn.Parameter(
            torch.randn(num_projections, manifold_dim, hidden_size) * 0.02
        )
        
        # Projection scaling
        self.scale = nn.Parameter(torch.ones(num_projections))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input onto manifold and back.
        
        Args:
            x: Input tensor [..., hidden_size]
            
        Returns:
            Manifold-constrained tensor [..., hidden_size]
        """
        # Project onto each manifold basis
        # x: [..., H], basis: [P, M, H] -> projections: [..., P, M]
        projections = torch.einsum('...h,pmh->...pm', x, self.manifold_basis)
        
        # Project back with scaling
        # projections: [..., P, M], basis: [P, M, H] -> reconstructed: [..., H]
        reconstructed = torch.einsum('...pm,pmh,p->...h', 
                                     projections, 
                                     self.manifold_basis,
                                     self.scale)
        
        return reconstructed


class DynamicGating(nn.Module):
    """
    Learns dynamic weights for hyper-connections.
    
    Computes gating weights based on input features
    to adaptively route information.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_connections: int
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_connections = num_connections
        
        # Gate projection
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, num_connections)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gating weights.
        
        Args:
            x: Input tensor [..., hidden_size]
            
        Returns:
            Gate weights [..., num_connections] summing to 1
        """
        # Pool over sequence if present
        if x.dim() == 3:
            pooled = x.mean(dim=1)  # [batch, hidden]
        else:
            pooled = x
            
        gates = self.gate_proj(pooled)
        gates = F.softmax(gates, dim=-1)
        
        return gates


class ManifoldHyperConnection(nn.Module):
    """
    Manifold-Constrained Hyper-Connection (mHC).
    
    Replaces standard residual connection:
        Standard: output = x + f(x)
        mHC: output = g(x, f(x)) where g is manifold-constrained
    
    Architecture:
    1. Multiple parallel connection paths
    2. Dynamic gating based on content
    3. Manifold constraint for geometric preservation
    4. Learnable combination weights
    """
    
    def __init__(
        self,
        hidden_size: int,
        expansion_rate: float = 4.0,
        num_connections: int = 2,
        use_dynamic_weights: bool = True,
        manifold_dim: int = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.expansion_rate = expansion_rate
        self.num_connections = num_connections
        self.use_dynamic_weights = use_dynamic_weights
        
        # Connection paths
        expanded_dim = int(hidden_size * expansion_rate)
        
        self.connection_paths = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, expanded_dim),  # Takes [x, sublayer_output]
                nn.GELU(),
                nn.Linear(expanded_dim, hidden_size)
            )
            for _ in range(num_connections)
        ])
        
        # Dynamic gating
        if use_dynamic_weights:
            self.gating = DynamicGating(hidden_size, num_connections)
        else:
            # Static learnable weights
            self.static_weights = nn.Parameter(torch.ones(num_connections) / num_connections)
        
        # Manifold projection for constraint
        self.manifold_proj = ManifoldProjection(
            hidden_size=hidden_size,
            manifold_dim=manifold_dim,
            num_projections=4
        )
        
        # Residual weight (how much to keep original)
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
        # Output normalization
        self.out_norm = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(
        self,
        x: torch.Tensor,
        sublayer_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply manifold-constrained hyper-connection.
        
        Args:
            x: Original input (before sublayer)
            sublayer_output: Output from sublayer (attention or FFN)
            
        Returns:
            Combined output with manifold constraint
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Concatenate input and sublayer output
        combined = torch.cat([x, sublayer_output], dim=-1)  # [batch, seq, 2*hidden]
        
        # Compute connection path outputs
        path_outputs = []
        for path in self.connection_paths:
            path_out = path(combined)
            path_outputs.append(path_out)
        
        # Stack path outputs: [batch, seq, num_connections, hidden]
        path_outputs = torch.stack(path_outputs, dim=-2)
        
        # Get combination weights
        if self.use_dynamic_weights:
            # Dynamic gating based on input
            weights = self.gating(x)  # [batch, num_connections]
            weights = weights.unsqueeze(1).unsqueeze(-1)  # [batch, 1, num_connections, 1]
        else:
            weights = F.softmax(self.static_weights, dim=0)
            weights = weights.view(1, 1, -1, 1)
        
        # Weighted combination of paths
        combined_output = (path_outputs * weights).sum(dim=-2)  # [batch, seq, hidden]
        
        # Apply manifold constraint
        manifold_term = self.manifold_proj(combined_output)
        combined_output = combined_output + 0.1 * manifold_term  # Soft constraint
        
        # Mix with residual
        residual_w = torch.sigmoid(self.residual_weight)
        output = residual_w * x + (1 - residual_w) * combined_output
        
        # Apply dropout and normalize
        output = self.dropout(output)
        output = self.out_norm(output)
        
        return output


class SimplifiedMHC(nn.Module):
    """
    Simplified Manifold Hyper-Connection.
    
    More efficient version that maintains key benefits:
    - Multi-path information flow
    - Adaptive weighting
    - Better than standard residual
    
    Lower overhead for resource-constrained settings.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_connections: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_connections = num_connections
        
        # Simple linear transforms for each path
        self.transforms = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_connections)
        ])
        
        # Gating
        self.gate = nn.Linear(hidden_size, num_connections)
        
        # Output scaling
        self.alpha = nn.Parameter(torch.zeros(1))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(
        self,
        x: torch.Tensor,
        sublayer_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply simplified mHC.
        
        Args:
            x: Original input
            sublayer_output: Sublayer output
            
        Returns:
            Combined output
        """
        # Compute gate weights
        gate_weights = F.softmax(self.gate(x.mean(dim=1, keepdim=True)), dim=-1)
        gate_weights = gate_weights.unsqueeze(-1)  # [batch, 1, num_conn, 1]
        
        # Apply transforms to sublayer output
        transformed = []
        for transform in self.transforms:
            transformed.append(transform(sublayer_output))
        transformed = torch.stack(transformed, dim=-2)  # [batch, seq, num_conn, hidden]
        
        # Weighted combination
        combined = (transformed * gate_weights).sum(dim=-2)
        
        # Residual with learnable scaling
        alpha = torch.sigmoid(self.alpha)
        output = x + alpha * combined + (1 - alpha) * sublayer_output
        
        return self.dropout(output)


class ResidualConnection(nn.Module):
    """
    Standard residual connection for comparison.
    
    output = x + sublayer(x)
    """
    
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(
        self,
        x: torch.Tensor,
        sublayer_output: torch.Tensor
    ) -> torch.Tensor:
        return x + self.dropout(sublayer_output)


def create_connection(
    connection_type: str,
    hidden_size: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create connection module.
    
    Args:
        connection_type: "residual", "mhc", or "simplified_mhc"
        hidden_size: Model hidden dimension
        **kwargs: Additional arguments for specific connection types
        
    Returns:
        Connection module
    """
    if connection_type == "residual":
        return ResidualConnection(dropout=kwargs.get('dropout', 0.0))
    elif connection_type == "mhc":
        return ManifoldHyperConnection(
            hidden_size=hidden_size,
            expansion_rate=kwargs.get('expansion_rate', 4.0),
            num_connections=kwargs.get('num_connections', 2),
            use_dynamic_weights=kwargs.get('use_dynamic_weights', True),
            manifold_dim=kwargs.get('manifold_dim', 64),
            dropout=kwargs.get('dropout', 0.0)
        )
    elif connection_type == "simplified_mhc":
        return SimplifiedMHC(
            hidden_size=hidden_size,
            num_connections=kwargs.get('num_connections', 2),
            dropout=kwargs.get('dropout', 0.0)
        )
    else:
        raise ValueError(f"Unknown connection type: {connection_type}")
