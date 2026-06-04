"""
Expert-Parallel TQP MoE Wrapper.

Drop-in replacement for TurboQuantPretrainingMoEWrapper that uses expert parallelism
to distribute 460 experts across 8 GPUs. Each GPU holds ~58 experts' TQ indices
on local GPU memory, enabling the fused Triton kernel path.

Usage in main.py:
    from lightninglm.tqp.tqp_moe_expert_parallel import EPTurboQuantPretrainingMoEWrapper, ExpertParallelContext
    ep_ctx = ExpertParallelContext(num_experts=460)
    # After TQP wrapping:
    for layer in model.layers:
        if hasattr(layer, 'moe') and isinstance(layer.moe, TurboQuantPretrainingMoEWrapper):
            layer.moe = EPTurboQuantPretrainingMoEWrapper.from_tq_wrapper(layer.moe, ep_ctx)
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightninglm.tqp.expert_parallel import ExpertParallelContext, shard_tq_weights


class EPTurboQuantPretrainingMoEWrapper(nn.Module):
    """
    Expert-Parallel version of TurboQuantPretrainingMoEWrapper.

    Key differences from the base class:
    1. Each GPU only holds local experts' TQ indices on GPU (not all 460)
    2. Forward uses all-to-all to dispatch tokens to expert-owning GPUs
    3. Local expert compute uses the fused Triton kernel (indices are on GPU)
    4. Results are gathered back via all-to-all
    """

    def __init__(self, tq_wrapper, ep_ctx: ExpertParallelContext):
        super().__init__()
        self.ep_ctx = ep_ctx
        self.global_num_experts = tq_wrapper.num_experts
        self.top_k = tq_wrapper.top_k
        self.dropout = tq_wrapper.dropout

        # Gate stays replicated (all GPUs run the same gate)
        self.gate = tq_wrapper.gate

        # Shared expert stays replicated (always active, not part of EP)
        self.shared_gate = tq_wrapper.shared_gate
        self.shared_up = tq_wrapper.shared_up
        self.shared_down = tq_wrapper.shared_down
        self.quantize_shared = tq_wrapper.quantize_shared

        # Shard expert weights: each GPU keeps only its local experts on GPU
        device = next(tq_wrapper.gate.parameters()).device
        self.tq_gate_weights = shard_tq_weights(
            tq_wrapper.tq_gate_weights, ep_ctx, device
        )
        self.tq_up = shard_tq_weights(tq_wrapper.tq_up, ep_ctx, device)
        self.tq_down = shard_tq_weights(tq_wrapper.tq_down, ep_ctx, device)

        # Local expert count
        self.local_num_experts = ep_ctx.local_num_experts
        self.num_experts = self.global_num_experts  # Keep global for gate compatibility

    @classmethod
    def from_tq_wrapper(cls, tq_wrapper, ep_ctx: ExpertParallelContext):
        """Convert an existing TurboQuantPretrainingMoEWrapper to EP version."""
        return cls(tq_wrapper, ep_ctx)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        EP forward:
        1. Shared expert (replicated, same as before)
        2. Gate (replicated) → get expert assignments
        3. Dispatch tokens to expert-owning GPUs via all-to-all
        4. Local expert compute (fused Triton, indices on GPU)
        5. Combine results via all-to-all
        6. Weighted sum + output
        """
        B, T, D = x.shape
        N = B * T
        K = self.top_k
        E = self.global_num_experts
        device, dtype = x.device, x.dtype

        # 1. Shared expert (dense, replicated)
        shared_out = self.shared_down(F.silu(self.shared_gate(x)) * self.shared_up(x))

        # 2. Gate (replicated)
        topk_idx, topk_weight, is_null, aux_loss = self.gate(x)
        flat_x = x.view(N, D)
        flat_idx = topk_idx.view(N, K)
        flat_weight = topk_weight.view(N, K)
        flat_is_null = is_null.view(N, K)
        real_mask = ~flat_is_null

        token_indices = torch.arange(N, device=device).unsqueeze(1).expand(N, K)
        real_token_indices = token_indices[real_mask]
        real_expert_indices = flat_idx[real_mask]
        real_weights = flat_weight[real_mask]

        M = real_token_indices.size(0)

        # 3. Dispatch: send tokens to expert-owning GPUs
        tokens_to_send = flat_x[real_token_indices]  # [M, D]
        recv_tokens, recv_local_eids, recv_weights, meta = self.ep_ctx.dispatch(
            tokens_to_send, real_expert_indices, real_weights
        )

        # 4. Local expert compute (fused Triton path — indices are on GPU!)
        M_local = recv_tokens.size(0)
        local_E = self.local_num_experts

        if M_local > 0:
            # Sort by local expert for grouped processing
            sort_idx = recv_local_eids.argsort()
            sorted_tokens = recv_tokens[sort_idx]
            sorted_eids = recv_local_eids[sort_idx]
            sorted_weights_local = recv_weights[sort_idx]

            expert_counts = torch.bincount(sorted_eids.long(), minlength=local_E)
            offsets = expert_counts.cumsum(0)
            expert_offsets = torch.cat(
                [
                    torch.zeros(1, device=device, dtype=torch.int64),
                    offsets.to(torch.int64),
                ]
            )

            # Use fused Triton kernel (indices are on GPU now!)
            try:
                # USE_BF16_BASE force-fallback: in bf16 mode, Triton kernel uses dead-weight
                # codebook/indices/row_norms. Raise so the existing except-fallback (per-expert
                # loop using compute_expert_chunk) runs instead.
                import os as _os_for_bf16_fast

                if _os_for_bf16_fast.environ.get("USE_BF16_BASE", "0") == "1":
                    raise RuntimeError(
                        "USE_BF16_BASE: forcing per-expert fallback (Triton path uses dead-weight codebook in bf16 mode)"
                    )
                from lightninglm.tqp.triton_tqp_grouped import (
                    grouped_dequant_matmul,
                    grouped_tqp,
                )

                def _fused_local(x_in, offsets, tq_w):
                    R = tq_w.rotation_matrix.to(dtype=x_in.dtype)
                    x_rot = x_in @ R.t()
                    with torch.no_grad():
                        nf = tq_w.row_norms.float()
                        cb = tq_w.weight_codebook.float()
                        y_base = grouped_dequant_matmul(
                            x_rot, offsets, tq_w.weight_indices, nf, cb
                        )
                        y_base = torch.nan_to_num(
                            y_base, nan=0.0, posinf=60000.0, neginf=-60000.0
                        )
                        y_base = y_base.clamp(-60000.0, 60000.0)
                    y_tqp = grouped_tqp(x_rot, offsets, tq_w.tqp_A, tq_w.tqp_B)
                    return y_base + y_tqp

                h_gate = _fused_local(
                    sorted_tokens, expert_offsets, self.tq_gate_weights
                ).to(dtype)
                h_up = _fused_local(sorted_tokens, expert_offsets, self.tq_up).to(dtype)
                hidden = F.silu(h_gate) * h_up
                if self.dropout > 0 and self.training:
                    hidden = F.dropout(hidden, p=self.dropout, training=True)
                sorted_out = _fused_local(hidden, expert_offsets, self.tq_down).to(
                    dtype
                )

            except Exception:
                # Fallback: per-expert loop (should not happen with EP)
                sorted_out = torch.empty(M_local, D, device=device, dtype=dtype)
                start = 0
                for e in range(local_E):
                    end = offsets[e].item()
                    if end > start:
                        chunk = sorted_tokens[start:end]
                        h_g = self.tq_gate_weights.compute_expert_chunk(chunk, e).to(
                            dtype
                        )
                        h_u = self.tq_up.compute_expert_chunk(chunk, e).to(dtype)
                        h = F.silu(h_g) * h_u
                        if self.dropout > 0 and self.training:
                            h = F.dropout(h, p=self.dropout, training=True)
                        sorted_out[start:end] = self.tq_down.compute_expert_chunk(
                            h, e
                        ).to(dtype)
                    start = end

            # Unsort local results
            local_output = torch.zeros(M_local, D, device=device, dtype=dtype)
            local_output[sort_idx] = sorted_out
        else:
            local_output = torch.empty(0, D, device=device, dtype=dtype)

        # 5. Combine: send results back to originating GPUs
        combined_output = self.ep_ctx.combine(local_output, meta)  # [M, D]

        # 6. Weighted sum + scatter back
        weighted_out = combined_output * real_weights.to(dtype).unsqueeze(-1)
        routed_out = torch.zeros(N, D, device=device, dtype=dtype)
        routed_out.scatter_add_(
            0,
            real_token_indices.unsqueeze(-1).expand(-1, D),
            weighted_out.to(dtype),
        )
        routed_out = routed_out.view(B, T, D)

        output = shared_out + routed_out
        if self.dropout > 0 and self.training:
            output = F.dropout(output, p=self.dropout, training=True)

        return output, aux_loss

    def flush_all(self):
        """Flush local experts' TQP adapters."""
        stats_list = []
        for name, mod in [
            ("tq_gate", self.tq_gate_weights),
            ("tq_up", self.tq_up),
            ("tq_down", self.tq_down),
        ]:
            s = mod.flush()
            s["module"] = name
            stats_list.append(s)

        if self.quantize_shared:
            for name, mod in [
                ("shared_gate", self.shared_gate),
                ("shared_up", self.shared_up),
                ("shared_down", self.shared_down),
            ]:
                if hasattr(mod, "flush"):
                    s = mod.flush()
                    s["module"] = name
                    stats_list.append(s)

        return {
            "AB_norm_mean": sum(s["AB_norm"] for s in stats_list) / len(stats_list),
            "AB_norm_max": max(s["AB_norm"] for s in stats_list),
            "w_change_mean": sum(s["w_change_frac"] for s in stats_list)
            / len(stats_list),
            "w_change_max": max(s["w_change_frac"] for s in stats_list),
            "per_module": stats_list,
        }

    def total_memory_bytes(self):
        total = 0
        for mod in [self.tq_gate_weights, self.tq_up, self.tq_down]:
            total += mod.memory_bytes()["total"]
        if self.quantize_shared:
            for mod in [self.shared_gate, self.shared_up, self.shared_down]:
                if hasattr(mod, "memory_bytes"):
                    total += mod.memory_bytes()["total"]
        return total
