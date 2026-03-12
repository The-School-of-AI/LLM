"""
LoRA-aware Fused Linear Cross-Entropy — ZeRO-3 safe.

Problem: ZeRO-3 partitions parameters into 1D flat buffers. Standard autograd
through lora_B @ lora_A crashes in backward() because parameters are ungathered
(RuntimeError: self must be a matrix).

Solution: Custom autograd Function that pre-computes ALL gradients in forward()
using raw .data tensors (no autograd graph to ZeRO-3 parameters). backward()
returns pre-computed gradients without referencing live parameters.

Gradient decomposition (forward-time):
  W_eff = cat([weight[:head], weight[head:] + lora_B @ lora_A])
  loss, grad_h, grad_W_eff = fused_linear_ce_forward(h, W_eff, target)
  grad_weight[:head] = grad_W_eff[:head]    (head trains fully)
  grad_weight[head:] = 0                     (tail base frozen)
  grad_lora_A = lora_B^T @ grad_W_eff[head:]
  grad_lora_B = grad_W_eff[head:] @ lora_A^T

Memory optimization: Only saves grad_weight_head [8192, H] (~64MB) instead
of full grad_weight [131072, H] (~1GB). Reconstructs full tensor in backward.
"""
import torch
from .triton_cross_entropy import _fused_linear_ce_forward


class _LoRAFusedCEFunction(torch.autograd.Function):
    """
    Fused Linear CE with LoRA gradient decomposition.
    All gradient computation happens in forward() — backward() is a lookup.

    Inputs to .apply():
      h:         [B*T, H]  hidden states (requires_grad=True)
      weight:    [V, H]    full lm_head weight Parameter
      lora_A:    [rank, H] LoRA down-projection Parameter
      lora_B:    [tail, rank] LoRA up-projection Parameter
      target:    [B*T]     token IDs
      head_size: int       number of full-rank head tokens
      ignore_index, reduction, max_chunk_gb, softcap: CE config
    """

    @staticmethod
    def forward(ctx, h, weight, lora_A, lora_B, target,
                head_size, ignore_index, reduction, max_chunk_gb, softcap):
        # ── Build effective weight from raw .data (no autograd graph) ───────
        W_eff = weight.data.clone()
        lora_A_data = lora_A.data
        lora_B_data = lora_B.data
        lora_delta = lora_B_data @ lora_A_data          # [tail, H]
        W_eff[head_size:].add_(lora_delta)

        # Force requires_grad so _fused_linear_ce_forward computes both gradients
        # Inside autograd.Function.forward(), inputs have requires_grad=False
        h_ce = h.detach().requires_grad_(True)
        W_eff.requires_grad_(True)

        # ── Run the fused kernel ────────────────────────────────────────────
        max_bytes = int(max_chunk_gb * 1024 * 1024 * 1024)
        loss, grad_h, grad_W_eff = _fused_linear_ce_forward(
            h_ce, W_eff, target, ignore_index, reduction, max_bytes, softcap
        )

        # ── Decompose grad_W_eff into component gradients ──────────────────
        # Head: full gradient (head tokens train at full rank)
        # Only save head portion — saves ~930MB vs full [V, H]
        grad_weight_head = grad_W_eff[:head_size].clone()

        # Tail: LoRA chain rule (tail base is frozen, only adapters train)
        grad_W_tail = grad_W_eff[head_size:]             # [tail, H]
        grad_lora_A = lora_B_data.T @ grad_W_tail        # [rank, H]
        grad_lora_B = grad_W_tail @ lora_A_data.T        # [tail, rank]

        # ── Save for backward (regular tensors, not ZeRO-3 params) ─────────
        ctx.save_for_backward(grad_h, grad_weight_head, grad_lora_A, grad_lora_B)
        ctx.head_size = head_size
        ctx.vocab_size = weight.shape[0]
        ctx.in_features = weight.shape[1]
        ctx.weight_dtype = weight.dtype

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_h, grad_weight_head, grad_lora_A, grad_lora_B = ctx.saved_tensors

        # Scale by grad_output if not 1.0 (e.g., MTP weight factor)
        s = grad_output
        if s.numel() == 1:
            sv = s.item()
            if sv != 1.0:
                if grad_h is not None:
                    grad_h = grad_h * sv
                grad_weight_head = grad_weight_head * sv
                grad_lora_A = grad_lora_A * sv
                grad_lora_B = grad_lora_B * sv

        # Reconstruct full grad_weight: head gets gradient, tail gets zeros
        grad_weight = torch.zeros(
            ctx.vocab_size, ctx.in_features,
            dtype=ctx.weight_dtype, device=grad_weight_head.device
        )
        grad_weight[:ctx.head_size] = grad_weight_head

        # Returns match forward inputs:
        # h, weight, lora_A, lora_B, target, head_size, ignore_index,
        # reduction, max_chunk_gb, softcap
        return (grad_h, grad_weight, grad_lora_A, grad_lora_B,
                None, None, None, None, None, None)


class LoRAFusedLinearCrossEntropyLoss(torch.nn.Module):
    """
    Drop-in replacement for FusedLinearCrossEntropyLoss when LoRA is active.

    Usage:
        lora_ce = LoRAFusedLinearCrossEntropyLoss(head_size=8192, softcap=15.0)
        loss = lora_ce(hidden_states, lm_head.weight, lm_head.lora_A, lm_head.lora_B, target)
    """

    def __init__(self, head_size, ignore_index=-100, reduction="mean",
                 max_chunk_gb=8.0, softcap=0.0):
        super().__init__()
        self.head_size = head_size
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.max_chunk_gb = max_chunk_gb
        self.softcap = softcap

    def forward(self, hidden_states, weight, lora_A, lora_B, target):
        return _LoRAFusedCEFunction.apply(
            hidden_states, weight, lora_A, lora_B, target,
            self.head_size, self.ignore_index, self.reduction,
            self.max_chunk_gb, self.softcap
        )
