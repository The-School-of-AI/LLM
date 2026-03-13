"""
Cut Cross Entropy — V-tiled Fused Linear + CE that never materializes [BT, V] logits.

For V=131,072 vocab: 32x memory reduction vs. full logit materialization.

Algorithm (forward):
    1. For each tile of vocab [v_start, v_end):
       - Compute logit_tile = hidden @ W[v_start:v_end].T  → [BT, V_TILE] only
       - Update online LSE: (max, sum_exp) accumulation
       - Extract correct-class logit if target falls in this tile
    2. loss = LSE - logit_y (per sample)

Algorithm (backward):
    1. Reload hidden states + saved LSE
    2. For each tile of vocab:
       - Recompute logit_tile = hidden @ W[v_start:v_end].T
       - grad_tile = exp(logit_tile - LSE) - one_hot(target)
       - Accumulate: grad_hidden += grad_tile @ W_tile, grad_W_tile += grad_tile.T @ hidden
    3. Never stores [BT, V] — recomputation trades FLOPs for memory

Reference: Apple ml-cross-entropy, Unsloth cut_cross_entropy
"""

import torch
import torch.nn as nn


class CutCrossEntropyFn(torch.autograd.Function):
    """
    Memory-efficient cross entropy that tiles over vocabulary.

    Saves for backward: hidden_states [BT, H], target [BT], lse [BT].
    Does NOT save grad_input or grad_weight (recomputed in backward).
    """

    @staticmethod
    def forward(ctx, hidden_states, weight, target, ignore_index, reduction,
                v_tile_size, softcap):
        BT, H = hidden_states.shape
        V = weight.shape[0]
        device = hidden_states.device

        n_non_ignore = max(int((target != ignore_index).sum().item()), 1)
        valid_mask = (target != ignore_index)

        # Online LSE accumulation + correct-class logit extraction
        max_logit = torch.full((BT,), float('-inf'), device=device, dtype=torch.float32)
        sum_exp = torch.zeros(BT, device=device, dtype=torch.float32)
        logit_y = torch.zeros(BT, device=device, dtype=torch.float32)

        for v_start in range(0, V, v_tile_size):
            v_end = min(v_start + v_tile_size, V)

            # Small matmul: [BT, H] @ [V_TILE, H].T = [BT, V_TILE]
            logit_tile = hidden_states @ weight[v_start:v_end].t()

            if softcap > 0:
                logit_tile = softcap * torch.tanh(logit_tile.float() / softcap)
            else:
                logit_tile = logit_tile.float()

            # Extract correct-class logits that fall in this tile
            in_tile = (target >= v_start) & (target < v_end) & valid_mask
            if in_tile.any():
                rows = torch.where(in_tile)[0]
                cols = (target[rows] - v_start).long()
                logit_y[rows] = logit_tile[rows, cols]

            # Online max + sum_exp update
            tile_max = logit_tile.max(dim=-1).values  # [BT]
            new_max = torch.maximum(max_logit, tile_max)
            sum_exp = (sum_exp * torch.exp(max_logit - new_max) +
                       torch.exp(logit_tile - new_max.unsqueeze(-1)).sum(dim=-1))
            max_logit = new_max

        # Finalize LSE
        lse = max_logit + torch.log(sum_exp)  # [BT]

        # Loss = LSE - logit_y
        loss_per_sample = lse - logit_y  # [BT]
        loss_per_sample = loss_per_sample * valid_mask.float()

        if reduction == "mean":
            loss = loss_per_sample.sum() / n_non_ignore
        elif reduction == "sum":
            loss = loss_per_sample.sum()
        else:
            loss = loss_per_sample

        # Save for backward — small tensors only, NO [BT, V] anywhere
        ctx.save_for_backward(hidden_states, target, lse)
        ctx.weight = weight  # reference, not a copy
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.n_non_ignore = n_non_ignore
        ctx.v_tile_size = v_tile_size
        ctx.softcap = softcap

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, target, lse = ctx.saved_tensors
        weight = ctx.weight
        ignore_index = ctx.ignore_index
        reduction = ctx.reduction
        n_non_ignore = ctx.n_non_ignore
        v_tile_size = ctx.v_tile_size
        softcap = ctx.softcap

        BT, H = hidden_states.shape
        V = weight.shape[0]
        device = hidden_states.device
        valid_mask = (target != ignore_index)

        # Scale factor
        if reduction == "mean":
            scale = grad_output / n_non_ignore
        elif reduction == "sum":
            scale = grad_output
        else:
            scale = grad_output  # [BT]

        grad_hidden = torch.zeros_like(hidden_states)
        grad_weight = torch.zeros_like(weight)

        for v_start in range(0, V, v_tile_size):
            v_end = min(v_start + v_tile_size, V)
            w_tile = weight[v_start:v_end]  # [V_TILE, H]

            # Recompute logit tile
            logit_tile = hidden_states @ w_tile.t()  # [BT, V_TILE]

            if softcap > 0:
                logit_tile_f = logit_tile.float()
                tanh_val = torch.tanh(logit_tile_f / softcap)
                sc_tile = softcap * tanh_val
                chain_factor = 1.0 - tanh_val * tanh_val
                # Softmax probabilities from softcapped logits
                prob_tile = torch.exp(sc_tile - lse.unsqueeze(-1))
            else:
                logit_tile_f = logit_tile.float()
                prob_tile = torch.exp(logit_tile_f - lse.unsqueeze(-1))

            # Subtract 1 at correct-class positions in this tile
            in_tile = (target >= v_start) & (target < v_end) & valid_mask
            if in_tile.any():
                rows = torch.where(in_tile)[0]
                cols = (target[rows] - v_start).long()
                prob_tile[rows, cols] -= 1.0

            if softcap > 0:
                prob_tile = prob_tile * chain_factor

            # Zero out ignored positions
            prob_tile = prob_tile * valid_mask.float().unsqueeze(-1)

            # Apply scale
            if scale.dim() == 0:
                prob_tile = prob_tile * scale
            else:
                prob_tile = prob_tile * scale.unsqueeze(-1)

            # Accumulate gradients
            prob_tile = prob_tile.to(hidden_states.dtype)
            grad_hidden.addmm_(prob_tile, w_tile)  # [BT, H] += [BT, V_TILE] @ [V_TILE, H]
            grad_weight[v_start:v_end].addmm_(prob_tile.t(), hidden_states)  # [V_TILE, H]

        return grad_hidden, grad_weight, None, None, None, None, None


class CutCrossEntropyLoss(nn.Module):
    """
    Drop-in replacement for FusedLinearCrossEntropyLoss using Cut CE.

    Never materializes [BT, V] logits. Peak memory is [BT, V_TILE] per tile
    where V_TILE defaults to 4096 (configurable).

    For V=131,072 and V_TILE=4096: 32x memory reduction.

    Args:
        ignore_index: target value to ignore (-100 default)
        reduction: "mean" or "sum"
        v_tile_size: vocabulary tile size (default 4096)
        softcap: softcap value (0.0 = disabled)
    """

    def __init__(self, ignore_index=-100, reduction="mean", v_tile_size=4096, softcap=0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.v_tile_size = v_tile_size
        self.softcap = softcap

    def forward(self, hidden_states, weight, target):
        """
        Args:
            hidden_states: [BT, H] — last hidden states (flattened batch*seq)
            weight: [V, H] — output embedding / lm_head weight
            target: [BT] — target token indices

        Returns: scalar loss
        """
        return CutCrossEntropyFn.apply(
            hidden_states, weight, target,
            self.ignore_index, self.reduction,
            self.v_tile_size, self.softcap
        )
