"""
Growth utilities for Dense-to-MoE model transitions.

SVD-based weight compression and transfer from a trained dense model
to a Mixture-of-Experts model.

Key operations:
  - Joint SVD compression: Stacks [Wg.T, Wu.T, Wd] to find a shared subspace
    that preserves SwiGLU coordinate alignment across gate, up, and down projections.
  - Independent SVD compression: Per-matrix SVD candidates + consensus basis.
  - Orthogonal rotation: diversity injection for expert symmetry breaking.
  - Router bias: null-biased initialization for stable expert activation.

SVD modes:
  - "joint": Stacks all 3 SwiGLU weight matrices into one joint matrix M and
    performs a single SVD to find the best shared k directions in intermediate
    space. All weights are projected into this shared basis.
  - "independent": Each matrix independently nominates its best k directions
    via per-matrix SVD, then a consensus basis is formed from all candidates
    via a second SVD. This can surface directions that joint SVD misses if
    one matrix dominates the joint spectrum.

Based on spectral_moe_initializer.py by teammate (Rohan).
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


# =============================================================================
# SwiGLU-Aware SVD Compression
# =============================================================================

def svd_compress_swiglu(
    Wg: torch.Tensor,
    Wu: torch.Tensor,
    Wd: torch.Tensor,
    target_dim: int,
    mode: str = "joint",
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Compress SwiGLU FFN weights from source intermediate dim to target_dim,
    preserving coordinate alignment between gate, up, and down projections.

    SwiGLU computes: silu(x @ Wg.T) * (x @ Wu.T), then result @ Wd.T.
    The element-wise multiply REQUIRES gate and up to share the same coordinate
    system. Per-matrix SVD breaks this — use joint or independent instead.

    Args:
        Wg: Gate projection, shape (intermediate_dense, hidden_size)
        Wu: Up projection, shape (intermediate_dense, hidden_size)
        Wd: Down projection, shape (hidden_size, intermediate_dense)
        target_dim: Target intermediate dimension (k)
        mode: "joint" or "independent"
        verbose: Print diagnostics

    Returns:
        Wg_compressed: (target_dim, hidden_size)
        Wu_compressed: (target_dim, hidden_size)
        Wd_compressed: (hidden_size, target_dim)
        explained_variance: Fraction of signal energy retained
    """
    assert mode in ("joint", "independent"), \
        f"svd_mode must be 'joint' or 'independent', got '{mode}'"

    if mode == "joint":
        return _svd_compress_joint(Wg, Wu, Wd, target_dim, verbose)
    else:
        return _svd_compress_independent(Wg, Wu, Wd, target_dim, verbose)


def _svd_compress_joint(
    Wg: torch.Tensor,
    Wu: torch.Tensor,
    Wd: torch.Tensor,
    target_dim: int,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Joint SVD compression: finds the subspace that best preserves
    Gate, Up, AND Down projections simultaneously.

    Stacks [Wg.T, Wu.T, Wd] into a joint matrix M of shape
    (3*hidden_size, intermediate_dense), then takes SVD to find the
    top-k directions in intermediate space.

    Wg: (intermediate_dense, hidden_size)
    Wu: (intermediate_dense, hidden_size)
    Wd: (hidden_size, intermediate_dense)
    """
    k = target_dim
    Wg_f = Wg.float()
    Wu_f = Wu.float()
    Wd_f = Wd.float()

    # Stack: Wg.T (hidden, int), Wu.T (hidden, int), Wd (hidden, int)
    # M shape: (3*hidden, intermediate_dense)
    M = torch.cat([Wg_f.T, Wu_f.T, Wd_f], dim=0)

    # SVD on joint matrix — Vh rows are principal directions in intermediate space
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)

    # Explained variance: how much signal the top-k directions capture
    explained_variance = (S[:k] ** 2).sum() / (S ** 2).sum()
    ev = explained_variance.item()

    if verbose:
        print(f"    📊 Joint SVD ({Wg.shape[0]}→{k}): "
              f"Explained Variance = {ev:.4f}")
        if ev < 0.90:
            print(f"    ⚠️  WARNING: Aggressive compression! "
                  f"Retained only {ev*100:.1f}% of signal.")

    # Take top-k components
    V_k = Vh[:k, :]  # (k, intermediate_dense)

    # Project all weights into the shared subspace
    Wg_base = V_k @ Wg_f      # (k, hidden_size)
    Wu_base = V_k @ Wu_f      # (k, hidden_size)
    Wd_base = Wd_f @ V_k.T    # (hidden_size, k)

    return (
        Wg_base.to(Wg.dtype),
        Wu_base.to(Wu.dtype),
        Wd_base.to(Wd.dtype),
        ev,
    )


def _svd_compress_independent(
    Wg: torch.Tensor,
    Wu: torch.Tensor,
    Wd: torch.Tensor,
    target_dim: int,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Independent SVD with consensus basis.

    Each matrix independently nominates its best k directions in intermediate
    space, then a consensus basis is formed from all candidates via SVD.

    This can surface directions that joint SVD misses if one matrix dominates
    the joint spectrum, while still maintaining the shared coordinate system
    required by SwiGLU.

    Wg: (intermediate_dense, hidden_size)
    Wu: (intermediate_dense, hidden_size)
    Wd: (hidden_size, intermediate_dense)
    """
    k = target_dim
    int_dense = Wg.shape[0]
    Wg_f = Wg.float()
    Wu_f = Wu.float()
    Wd_f = Wd.float()

    # Per-matrix SVD to find each matrix's best intermediate directions
    # Wg: (int_dense, hidden) → left singular vectors span intermediate space
    U_g, S_g, _ = torch.linalg.svd(Wg_f, full_matrices=False)
    ev_g = (S_g[:k] ** 2).sum() / (S_g ** 2).sum()

    U_u, S_u, _ = torch.linalg.svd(Wu_f, full_matrices=False)
    ev_u = (S_u[:k] ** 2).sum() / (S_u ** 2).sum()

    # Wd: (hidden, int_dense) → right singular vectors span intermediate space
    _, S_d, Vh_d = torch.linalg.svd(Wd_f, full_matrices=False)
    ev_d = (S_d[:k] ** 2).sum() / (S_d ** 2).sum()

    if verbose:
        print(f"    📊 Independent SVD [Wg] ({int_dense}→{k}): "
              f"EV = {ev_g.item():.4f}")
        print(f"    📊 Independent SVD [Wu] ({int_dense}→{k}): "
              f"EV = {ev_u.item():.4f}")
        print(f"    📊 Independent SVD [Wd] ({int_dense}→{k}): "
              f"EV = {ev_d.item():.4f}")

    # Form consensus basis from all candidates
    # Each contributes top-k directions → (int_dense, 3k) candidates
    candidates = torch.cat([
        U_g[:, :k],       # (int_dense, k) — Wg's best directions
        U_u[:, :k],       # (int_dense, k) — Wu's best directions
        Vh_d[:k, :].T,    # (int_dense, k) — Wd's best directions
    ], dim=1)  # (int_dense, 3k)

    U_consensus, _, _ = torch.linalg.svd(candidates, full_matrices=False)
    V_k = U_consensus[:, :k].T  # (k, int_dense) — shared consensus basis

    # Project all weights into consensus basis
    Wg_base = V_k @ Wg_f      # (k, hidden_size)
    Wu_base = V_k @ Wu_f      # (k, hidden_size)
    Wd_base = Wd_f @ V_k.T    # (hidden_size, k)

    # Report consensus explained variance per matrix
    ev_consensus_g = (Wg_base.norm() ** 2) / (Wg_f.norm() ** 2)
    ev_consensus_u = (Wu_base.norm() ** 2) / (Wu_f.norm() ** 2)
    ev_consensus_d = (Wd_base.norm() ** 2) / (Wd_f.norm() ** 2)
    ev_mean = (ev_consensus_g + ev_consensus_u + ev_consensus_d).item() / 3

    if verbose:
        print(f"    📊 Consensus basis retained energy: "
              f"Wg={ev_consensus_g.item():.4f}, "
              f"Wu={ev_consensus_u.item():.4f}, "
              f"Wd={ev_consensus_d.item():.4f}")

    return (
        Wg_base.to(Wg.dtype),
        Wu_base.to(Wu.dtype),
        Wd_base.to(Wd.dtype),
        ev_mean,
    )


# =============================================================================
# Rotation for Expert Diversity
# =============================================================================

def random_small_rotation(dim: int, eps: float = 0.005, device: str = "cpu") -> torch.Tensor:
    """
    Generate a small orthogonal rotation matrix via skew-symmetric matrix exponential.

    Unlike additive noise, rotation preserves weight norms and spectral structure.
    Each expert gets a distinct rotation, breaking symmetry for specialization.

    Args:
        dim: Matrix dimension (intermediate_size of the expert)
        eps: Rotation magnitude (~0.005 = ~0.3° rotation)
        device: Target device

    Returns:
        R: (dim, dim) orthogonal matrix close to identity
    """
    A = torch.randn(dim, dim, device=device)
    A = A - A.T  # skew-symmetric: A^T = -A
    R = torch.matrix_exp(eps * A)  # R^T R = I (orthogonal)
    return R


# =============================================================================
# Weight Transfer: Global Components
# =============================================================================

def copy_global_components(model_src, model_dst, verbose: bool = True):
    """
    Copy all non-layer components from source to destination model.

    Transfers: embeddings, norms, lm_head, memory components.
    Both models must have the same hidden_size.
    """
    copied = []

    # Kronecker embeddings
    if hasattr(model_src, 'kronecker_embeddings') and hasattr(model_dst, 'kronecker_embeddings'):
        if model_src.kronecker_embeddings is not None and model_dst.kronecker_embeddings is not None:
            model_dst.kronecker_embeddings.load_state_dict(
                model_src.kronecker_embeddings.state_dict()
            )
            copied.append("kronecker_embeddings")

    # Standard token embedding fallback
    if hasattr(model_src, 'token_embed') and hasattr(model_dst, 'token_embed'):
        if model_src.token_embed is not None and model_dst.token_embed is not None:
            model_dst.token_embed.load_state_dict(model_src.token_embed.state_dict())
            copied.append("token_embed")

    # Embedding norm
    if hasattr(model_src, 'embed_norm') and hasattr(model_dst, 'embed_norm'):
        if model_src.embed_norm is not None and model_dst.embed_norm is not None:
            model_dst.embed_norm.load_state_dict(model_src.embed_norm.state_dict())
            copied.append("embed_norm")

    # Kronecker projection
    if hasattr(model_src, 'pf_to_model') and hasattr(model_dst, 'pf_to_model'):
        if model_src.pf_to_model is not None and model_dst.pf_to_model is not None:
            model_dst.pf_to_model.load_state_dict(model_src.pf_to_model.state_dict())
            copied.append("pf_to_model")

    # Final normalization
    model_dst.norm.load_state_dict(model_src.norm.state_dict())
    copied.append("norm")

    # LM head
    model_dst.lm_head.load_state_dict(model_src.lm_head.state_dict())
    copied.append("lm_head")

    # Memory components
    for name in ['lambda_r_raw', 'memory_ln', 'memory_gate_proj']:
        if hasattr(model_src, name) and hasattr(model_dst, name):
            src_component = getattr(model_src, name)
            dst_component = getattr(model_dst, name)
            if src_component is not None and dst_component is not None:
                if isinstance(src_component, nn.Parameter):
                    dst_component.data.copy_(src_component.data)
                elif isinstance(src_component, nn.Module):
                    dst_component.load_state_dict(src_component.state_dict())
                copied.append(name)

    if verbose:
        print(f"  ✓ Copied global components: {', '.join(copied)}")

    return copied


# =============================================================================
# Weight Transfer: Per-Layer Attention + MHC Coefficients
# =============================================================================

def copy_layer_attention_and_mhc(src_layer, dst_layer, layer_idx: int, verbose: bool = True):
    """
    Copy attention block and MHC coefficients from dense layer to MoE layer.

    Transfers:
      - attn_block (full state dict — identical architecture)
      - mlp_block.coeffs (MHC routing coefficients)
      - mlp_block.norm (pre-norm for MLP)
    """
    # Full attention block (DeltaNet or GSA — identical between 1B and 3B)
    dst_layer.attn_block.load_state_dict(src_layer.attn_block.state_dict())

    # MHC coefficients and norm for MLP block
    dst_layer.mlp_block.coeffs.load_state_dict(src_layer.mlp_block.coeffs.state_dict())
    dst_layer.mlp_block.norm.load_state_dict(src_layer.mlp_block.norm.state_dict())

    if verbose:
        print(f"  ✓ Layer {layer_idx}: attention + MHC coefficients copied")


# =============================================================================
# Weight Transfer: Dense FFN → MoE (SVD + Rotation)
# =============================================================================

def get_dense_ffn_weights(dense_mlp) -> Dict[str, torch.Tensor]:
    """
    Extract FFN weights from a dense MLP.

    Handles both DenseMLP (gate_proj/up_proj/down_proj)
    and MoEFFN in dense mode (shared_gate/shared_up/shared_down).
    """
    # DenseMLP wrapping LigerSwiGLUMLP: dense_mlp.mlp.{gate,up,down}_proj
    if hasattr(dense_mlp, 'mlp') and hasattr(dense_mlp.mlp, 'gate_proj'):
        return {
            'gate': dense_mlp.mlp.gate_proj.weight.data,  # (intermediate, hidden)
            'up': dense_mlp.mlp.up_proj.weight.data,      # (intermediate, hidden)
            'down': dense_mlp.mlp.down_proj.weight.data,   # (hidden, intermediate)
        }
    # Legacy DenseMLP with direct gate_proj/up_proj/down_proj
    elif hasattr(dense_mlp, 'gate_proj'):
        return {
            'gate': dense_mlp.gate_proj.weight.data,  # (intermediate, hidden)
            'up': dense_mlp.up_proj.weight.data,      # (intermediate, hidden)
            'down': dense_mlp.down_proj.weight.data,   # (hidden, intermediate)
        }
    # MoEFFN uses shared_gate/shared_up/shared_down
    elif hasattr(dense_mlp, 'shared_gate'):
        return {
            'gate': dense_mlp.shared_gate.weight.data,
            'up': dense_mlp.shared_up.weight.data,
            'down': dense_mlp.shared_down.weight.data,
        }
    else:
        raise ValueError(f"Cannot extract FFN weights from {type(dense_mlp).__name__}")


def clone_dense_to_moe_experts(
    dense_ffn_weights: Dict[str, torch.Tensor],
    moe_ffn,
    num_experts: int,
    target_intermediate: int,
    rotation_eps: float = 0.005,
    svd_mode: str = "joint",
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Transfer dense FFN weights to MoE layer via SVD compression + rotation.

    Steps:
      1. Copy dense weights → shared expert (direct, same shape)
      2. SVD-compress all 3 SwiGLU weights jointly to preserve coordinate alignment
      3. Apply orthogonal rotation to each expert for diversity

    Args:
        dense_ffn_weights: Dict with 'gate', 'up', 'down' weight tensors
        moe_ffn: MoEFFN module to populate
        num_experts: Number of routed experts
        target_intermediate: Expert intermediate size (e.g. 1024)
        rotation_eps: Rotation magnitude for expert diversity
        svd_mode: "joint" (shared subspace) or "independent" (consensus basis)
        device: computation device

    Returns:
        energy_report: Dict with SVD energy retention ratios
    """
    Wg = dense_ffn_weights['gate'].to(device)  # (2048, 4096)
    Wu = dense_ffn_weights['up'].to(device)    # (2048, 4096)
    Wd = dense_ffn_weights['down'].to(device)  # (4096, 2048)

    source_intermediate = Wg.shape[0]  # 2048

    # Step 1: Copy to shared expert (direct — same shape)
    moe_ffn.shared_gate.weight.data.copy_(Wg)
    moe_ffn.shared_up.weight.data.copy_(Wu)
    moe_ffn.shared_down.weight.data.copy_(Wd)

    # Step 2: SwiGLU-aware SVD compression
    # Uses joint or independent mode to maintain coordinate alignment
    # between gate and up projections (required for the element-wise multiply)
    needs_compression = (source_intermediate != target_intermediate)

    if needs_compression:
        Wg_c, Wu_c, Wd_c, explained_variance = svd_compress_swiglu(
            Wg, Wu, Wd, target_intermediate,
            mode=svd_mode, verbose=verbose,
        )
    else:
        # Same dimension — no compression needed
        Wg_c = Wg.clone()
        Wu_c = Wu.clone()
        Wd_c = Wd.clone()
        explained_variance = 1.0

    energy_report = {
        'explained_variance': explained_variance,
        'svd_mode': svd_mode,
    }

    # Step 3: Clone to each expert with orthogonal rotation for diversity
    # MoEFFN stores: W_gate[e] shape (d_model, d_hidden)
    # SVD gives us: Wg_c shape (target_intermediate, d_model)
    # So W_gate[e] = Wg_c.T = (d_model, target_intermediate)
    # Rotation R is applied in intermediate (k) space:
    #   gate/up: R @ W  (k,k) @ (k,h) = (k,h)
    #   down:    W @ R.T (h,k) @ (k,k) = (h,k)
    # This preserves SwiGLU: silu(x@Wg.T@R.T) * (x@Wu.T@R.T) @ R@Wd.T
    # For small eps, rotations approximately cancel → expert(x) ≈ base(x)

    for e in range(num_experts):
        R = random_small_rotation(target_intermediate, eps=rotation_eps, device=device)

        # Rotate in intermediate space
        Wg_e = R @ Wg_c.float()      # (target_intermediate, d_model)
        Wu_e = R @ Wu_c.float()      # (target_intermediate, d_model)
        Wd_e = Wd_c.float() @ R.T   # (d_model, target_intermediate)

        # Store in batched expert format: (d_model, d_hidden)
        moe_ffn.W_gate.data[e] = Wg_e.T.to(moe_ffn.W_gate.dtype)
        moe_ffn.W_up.data[e] = Wu_e.T.to(moe_ffn.W_up.dtype)
        moe_ffn.W_down.data[e] = Wd_e.T.to(moe_ffn.W_down.dtype)

    return energy_report


# =============================================================================
# Router Bias Initialization
# =============================================================================

def set_router_bias(model, logit_bias: float = -0.6, null_logit: float = 0.6, verbose: bool = True):
    """
    Set router biases for null-biased active routing.

    With gap = null_logit - logit_bias = 1.2, the null expert starts with
    ~75% selection probability, allowing gradient to gradually discover
    when routed experts help.

    Applies to both backbone layers and MTP block.
    """
    count = 0

    # Backbone layers
    if hasattr(model, 'layers'):
        for layer in model.layers:
            moe = _get_moe_from_layer(layer)
            if moe is not None and hasattr(moe, 'gate') and moe.gate is not None:
                _set_gate_bias(moe.gate, logit_bias, null_logit)
                count += 1

    # MTP block
    if hasattr(model, 'mtp_block') and model.mtp_block is not None:
        moe = _get_moe_from_mtp(model.mtp_block)
        if moe is not None and hasattr(moe, 'gate') and moe.gate is not None:
            _set_gate_bias(moe.gate, logit_bias, null_logit)
            count += 1

    if verbose:
        gap = null_logit - logit_bias
        print(f"  ✓ Router bias set on {count} layers "
              f"(bias={logit_bias}, null={null_logit}, gap={gap:.2f})")


def _set_gate_bias(gate, logit_bias: float, null_logit: float):
    """Set bias values on a MoEGate."""
    if hasattr(gate, 'logit_bias') and gate.logit_bias is not None:
        nn.init.constant_(gate.logit_bias, logit_bias)
    if hasattr(gate, 'null_logit') and gate.null_logit is not None:
        nn.init.constant_(gate.null_logit, null_logit)


# =============================================================================
# Helpers: Navigate model layer hierarchy
# =============================================================================

def _get_moe_from_layer(layer) -> Optional[nn.Module]:
    """Extract MoEFFN from a LightningDecoderLayer."""
    if hasattr(layer, 'mlp_block') and hasattr(layer.mlp_block, 'sublayer'):
        sublayer = layer.mlp_block.sublayer
        # 3B LightningMLP stores MoEFFN as .moe
        if hasattr(sublayer, 'moe'):
            return sublayer.moe
        # 1B LightningMLP stores DenseMLP/MoEFFN as .mlp
        if hasattr(sublayer, 'mlp'):
            return sublayer.mlp
    return None


def _get_moe_from_mtp(mtp_block) -> Optional[nn.Module]:
    """Extract MoEFFN from MTPTransformerBlock."""
    if hasattr(mtp_block, 'mlp_block') and hasattr(mtp_block.mlp_block, 'sublayer'):
        sublayer = mtp_block.mlp_block.sublayer
        if hasattr(sublayer, 'moe'):
            return sublayer.moe
        if hasattr(sublayer, 'mlp'):
            return sublayer.mlp
    # Fallback: direct .mlp attribute
    if hasattr(mtp_block, 'mlp'):
        mlp = mtp_block.mlp
        if hasattr(mlp, 'moe'):
            return mlp.moe
        if hasattr(mlp, 'mlp'):
            return mlp.mlp
    return None


def _get_dense_mlp_from_layer(layer) -> Optional[nn.Module]:
    """Extract DenseMLP from a 1B LightningDecoderLayer."""
    if hasattr(layer, 'mlp_block') and hasattr(layer.mlp_block, 'sublayer'):
        sublayer = layer.mlp_block.sublayer
        # 1B dense: LightningMLP.mlp → DenseMLP
        if hasattr(sublayer, 'mlp'):
            return sublayer.mlp
        # 1B with MoE (dense mode): LightningMLP.mlp → MoEFFN (is_dense=True)
        if hasattr(sublayer, 'moe'):
            return sublayer.moe
    return None


# =============================================================================
# Validation
# =============================================================================

def validate_expert_diversity(model, num_experts: int, verbose: bool = True) -> Dict[str, float]:
    """
    Check cosine similarity between expert weights.
    Good initialization: mean similarity ∈ [0.90, 0.99].
    """
    results = {}

    if not hasattr(model, 'layers'):
        return results

    for idx, layer in enumerate(model.layers):
        moe = _get_moe_from_layer(layer)
        if moe is None or not hasattr(moe, 'W_gate') or not isinstance(moe.W_gate, nn.Parameter):
            continue

        gate_flat = moe.W_gate.data.view(num_experts, -1)
        norms = gate_flat.norm(p=2, dim=1, keepdim=True)
        normalized = gate_flat / (norms + 1e-8)
        sim = torch.mm(normalized, normalized.t())
        mask = torch.triu(torch.ones_like(sim), diagonal=1).bool()
        mean_sim = sim[mask].mean().item()
        results[f"layer_{idx}"] = mean_sim

        if verbose:
            status = "✓" if 0.85 <= mean_sim <= 0.999 else "⚠"
            print(f"    {status} Layer {idx}: mean cosine sim = {mean_sim:.4f}")

    return results


def validate_growth(model_src, model_dst, sample_input: torch.Tensor,
                    force_null: bool = True, verbose: bool = True) -> Dict[str, float]:
    """
    Validate growth by checking loss equivalence with null routing.

    When routing is forced to null (only shared expert fires),
    the 3B MoE should produce identical loss to the 1B dense model.
    """
    import torch.nn.functional as F

    model_src.eval()
    model_dst.eval()

    # Optionally force null routing on dst
    if force_null:
        _force_null_routing(model_dst)

    with torch.no_grad():
        # Forward through both models
        out_src = model_src(sample_input[:, :-1])
        out_dst = model_dst(sample_input[:, :-1])

        # Handle different return types
        logits_src = out_src[0] if isinstance(out_src, tuple) else out_src
        logits_dst = out_dst[0] if isinstance(out_dst, tuple) else out_dst

        targets = sample_input[:, 1:logits_src.shape[1] + 1]

        loss_src = F.cross_entropy(
            logits_src.reshape(-1, logits_src.shape[-1]),
            targets.reshape(-1)
        ).item()

        loss_dst = F.cross_entropy(
            logits_dst.reshape(-1, logits_dst.shape[-1]),
            targets.reshape(-1)
        ).item()

    results = {
        'loss_src': loss_src,
        'loss_dst': loss_dst,
        'loss_diff': abs(loss_src - loss_dst),
    }

    if verbose:
        status = "✓" if results['loss_diff'] < 0.01 else "⚠"
        print(f"    {status} Source loss: {loss_src:.6f}, Grown loss: {loss_dst:.6f}, "
              f"Diff: {results['loss_diff']:.6f}")

    return results


def force_null_routing(model, logit_bias: float = -100.0, null_logit: float = 100.0):
    """
    Force all routers to select null experts by setting extreme biases.

    Useful for validation: with null routing, the 3B MoE should produce
    identical output to the 1B dense model (only shared expert fires).
    """
    set_router_bias(model, logit_bias=logit_bias, null_logit=null_logit, verbose=False)


# Keep old name for backwards compatibility
_force_null_routing = force_null_routing
