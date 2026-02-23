"""
IDFT Loss Functions
Team 18: SFT, RL-Style Alignment & Final Post-Training Benchmarks

Implements the IDFT (In-Distribution Fine-Tuning) loss from
"Towards On-Policy SFT" (arXiv:2602.12222, Feb 2026).

Also provides a standard SFT loss for baseline comparison.
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Standard SFT cross-entropy loss.

    L_SFT = -(1/L) * sum(log p_t(x_t))

    Args:
        logits: (batch, seq_len, vocab_size) raw model logits
        labels: (batch, seq_len) target token IDs
        attention_mask: (batch, seq_len) 1 for real tokens, 0 for padding

    Returns:
        Scalar loss tensor.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    masked = token_log_probs * attention_mask
    loss = -masked.sum() / attention_mask.sum()
    return loss


def idft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    clip_B: float = 5.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    IDFT loss from "Towards On-Policy SFT" (arXiv:2602.12222).

    L_IDFT = -(1/L) * sum(p_t(x_t)^gamma_t * log p_t(x_t))
    where gamma_t = exp(-phi_t) and phi_t = log p_t(x_t) + H[p_t]

    Args:
        logits: (batch, seq_len, vocab_size) raw model logits
        labels: (batch, seq_len) target token IDs
        attention_mask: (batch, seq_len) 1 for real tokens, 0 for padding
        clip_B: Clipping bound for phi. Paper recommends 3-10, default 5.

    Returns:
        Tuple of (loss, diagnostics_dict).
        loss: scalar tensor (differentiable).
        diagnostics_dict: dict with phi/gamma statistics (detached).
    """
    # Step 1: log probabilities and probabilities
    log_probs = F.log_softmax(logits, dim=-1)  # (B, L, V)
    probs = log_probs.exp()  # (B, L, V)

    # Step 2: log p_t(x_t) for target tokens
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, L)

    # Step 3: entropy H[p_t] = -sum(p(v) * log p(v))
    entropy = -(probs * log_probs).sum(dim=-1)  # (B, L)

    # Step 4: phi_t = log p_t(x_t) + H[p_t]  (CLL discriminant)
    phi = token_log_probs + entropy  # (B, L)

    # Step 5: clip phi for numerical stability
    phi_clipped = phi.clamp(-clip_B, clip_B)

    # Step 6: gamma_t = exp(-phi_t)
    gamma = torch.exp(-phi_clipped)  # (B, L)

    # Step 7: IDFT loss in log-space for stability: p^gamma = exp(gamma * log p)
    weighted_factor = torch.exp(gamma * token_log_probs)  # p_t^gamma_t
    per_token_loss = -weighted_factor * token_log_probs  # -p_t^gamma_t * log p_t

    # Step 8: mask and average
    masked_loss = per_token_loss * attention_mask
    loss = masked_loss.sum() / attention_mask.sum()

    # Diagnostics (detached, no grad)
    with torch.no_grad():
        valid_phi = phi_clipped[attention_mask.bool()]
        valid_gamma = gamma[attention_mask.bool()]
        diagnostics = {
            "phi_mean": valid_phi.mean().item(),
            "phi_std": valid_phi.std().item(),
            "phi_below_neg1_pct": (valid_phi < -1).float().mean().item() * 100,
            "phi_below_neg3_pct": (valid_phi < -3).float().mean().item() * 100,
            "phi_below_neg5_pct": (valid_phi < -5).float().mean().item() * 100,
            "gamma_mean": valid_gamma.mean().item(),
            "gamma_max": valid_gamma.max().item(),
        }

    return loss, diagnostics
