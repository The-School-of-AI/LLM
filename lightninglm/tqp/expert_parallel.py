"""
Expert Parallelism (EP) for TQP MoE -- Optimized v3.

Distributes 460 experts across 8 GPUs. Each GPU owns ~58 experts and their
TQ weight indices on local GPU memory. Tokens are routed to the correct GPU
via all-to-all communication.

Optimizations (v3 -- EP1 patch):
  - dispatch/combine wrapped in @torch.no_grad() (same as v2)
  - Eager del of intermediate tensors (same as v2)
  - index_copy_ in combine (same as v2)
  - NEW: torch.bincount replaces per-rank for-loop for send_counts
  - NEW: Pack tokens+expert_ids+weights into single buffer, one all-to-all
  - NEW: async count exchange overlapped with sort
  - NEW: Eliminated redundant .tolist() host syncs
"""

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist


class _AllToAllVariable(torch.autograd.Function):
    """Differentiable all-to-all with variable per-rank splits.

    Forward:  send_tensor (size = sum(send_counts)) -> recv_tensor (size = sum(recv_counts))
    Backward: grad_recv -> grad_send via inverse all-to-all (send/recv counts swapped).

    The actual NCCL all_to_all is communication-only with no math, so backward is
    just the same comm with counts swapped — gradients route back to the originating
    rank in exactly the same order they arrived.
    """

    @staticmethod
    def forward(ctx, send_tensor, send_counts, recv_counts, group):
        # send_counts / recv_counts are python lists of ints (already host-side,
        # consistent with the existing dispatch/combine call sites).
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.group = group
        ctx.send_shape_tail = tuple(send_tensor.shape[1:])

        total_recv = int(sum(recv_counts))
        recv_tensor = torch.empty(
            total_recv,
            *send_tensor.shape[1:],
            dtype=send_tensor.dtype,
            device=send_tensor.device,
        )

        send_splits = list(torch.split(send_tensor.contiguous(), send_counts, dim=0))
        recv_splits = list(torch.split(recv_tensor, recv_counts, dim=0))
        dist.all_to_all(recv_splits, send_splits, group=group)
        return recv_tensor

    @staticmethod
    def backward(ctx, grad_recv):
        # Inverse comm: tensor sized recv_counts goes back as tensor sized send_counts.
        total_send = int(sum(ctx.send_counts))
        grad_send = torch.empty(
            total_send,
            *ctx.send_shape_tail,
            dtype=grad_recv.dtype,
            device=grad_recv.device,
        )
        send_splits = list(torch.split(grad_recv.contiguous(), ctx.recv_counts, dim=0))
        recv_splits = list(torch.split(grad_send, ctx.send_counts, dim=0))
        dist.all_to_all(recv_splits, send_splits, group=ctx.group)
        return grad_send, None, None, None


def _all_to_all_diff(send_tensor, send_counts, recv_counts, group):
    """Convenience wrapper."""
    return _AllToAllVariable.apply(send_tensor, send_counts, recv_counts, group)


class ExpertParallelContext:
    """
    Manages expert-to-GPU assignment and provides dispatch/combine primitives.
    """

    def __init__(
        self,
        num_experts: int,
        ep_group: Optional[dist.ProcessGroup] = None,
    ):
        self.num_experts = num_experts
        self.ep_group = ep_group or dist.group.WORLD
        self.ep_size = dist.get_world_size(self.ep_group)
        self.ep_rank = dist.get_rank(self.ep_group)

        self.expert_to_rank = torch.zeros(num_experts, dtype=torch.int32)
        self.local_expert_ids: List[int] = []
        self.local_expert_start = 0
        self.local_num_experts = 0

        # Contiguous assignment: rank r owns experts [start_r, start_r + count_r)
        experts_per_rank = num_experts // self.ep_size
        remainder = num_experts % self.ep_size
        start = 0
        for r in range(self.ep_size):
            count = experts_per_rank + (1 if r < remainder else 0)
            self.expert_to_rank[start : start + count] = r
            if r == self.ep_rank:
                self.local_expert_start = start
                self.local_num_experts = count
                self.local_expert_ids = list(range(start, start + count))
            start += count

        self._expert_to_rank_gpu = None

    def _ensure_gpu(self, device):
        if (
            self._expert_to_rank_gpu is None
            or self._expert_to_rank_gpu.device != device
        ):
            self._expert_to_rank_gpu = self.expert_to_rank.to(device)

    def dispatch(
        self,
        tokens: torch.Tensor,  # [M, D]
        expert_ids: torch.Tensor,  # [M]
        weights: torch.Tensor,  # [M]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Send tokens to the GPU that owns their assigned expert.
        Optimized: single packed all-to-all, bincount for counts, async overlap.
        """
        device = tokens.device
        self._ensure_gpu(device)
        M, D = tokens.shape

        # Determine destination rank for each token
        token_dest_rank = self._expert_to_rank_gpu[expert_ids.long()]

        # Count tokens per destination rank — single bincount, no per-rank loop
        send_counts = torch.bincount(token_dest_rank.int(), minlength=self.ep_size).to(
            torch.int64
        )

        # Async count exchange — overlaps with the sort below
        recv_counts = torch.zeros(self.ep_size, dtype=torch.int64, device=device)
        count_work = dist.all_to_all_single(
            recv_counts, send_counts, group=self.ep_group, async_op=True
        )

        # Sort tokens by destination rank for contiguous sends (runs while count exchange is in flight)
        sort_idx = token_dest_rank.argsort(stable=True)
        sorted_tokens = tokens[sort_idx]
        sorted_expert_ids = expert_ids[sort_idx]
        sorted_weights = weights[sort_idx]

        # Wait for count exchange to finish
        count_work.wait()

        send_counts_list = send_counts.tolist()
        recv_counts_list = recv_counts.tolist()
        total_recv = sum(recv_counts_list)

        # --- Pack tokens + weights in bf16 buffer, expert_ids separate as int32 ---
        # CRITICAL: use torch.cat (not torch.empty + slice-assign) so autograd graph
        # propagates from sorted_tokens through send_packed.  In-place assignment to
        # a freshly-allocated empty buffer detaches the gradient flow.
        pack_cols = D + 1
        send_packed = torch.cat(
            [sorted_tokens, sorted_weights.to(tokens.dtype).unsqueeze(-1)],
            dim=-1,
        )
        del sorted_tokens, sorted_weights

        # All-to-all 1: packed tokens + weights — DIFFERENTIABLE.
        # Backward routes grad on recv_tokens (D cols) back to sorted_tokens via the
        # inverse all-to-all. Weights column gradient is unused downstream.
        recv_packed = _all_to_all_diff(
            send_packed, send_counts_list, recv_counts_list, self.ep_group
        )
        del send_packed

        # All-to-all 2: expert IDs as int32 — non-differentiable (indices have no grad).
        with torch.no_grad():
            send_eids = sorted_expert_ids.to(torch.int32)
            recv_eids = torch.empty(total_recv, dtype=torch.int32, device=device)
            send_eid_splits = list(torch.split(send_eids, send_counts_list, dim=0))
            recv_eid_splits = list(torch.split(recv_eids, recv_counts_list, dim=0))
            dist.all_to_all(recv_eid_splits, send_eid_splits, group=self.ep_group)
            del send_eid_splits, send_eids, sorted_expert_ids

        # Unpack
        recv_tokens = recv_packed[:, :D].contiguous()
        recv_weights_out = recv_packed[:, D].to(weights.dtype)
        del recv_packed
        recv_expert_ids = recv_eids.to(expert_ids.dtype)
        del recv_eids

        # Convert global expert IDs to local (0-based)
        recv_local_expert_ids = recv_expert_ids - self.local_expert_start

        meta = {
            "sort_idx": sort_idx,
            "send_counts": send_counts_list,
            "recv_counts": recv_counts_list,
            "original_M": M,
            "D": D,
        }

        return recv_tokens, recv_local_expert_ids, recv_weights_out, meta

    def combine(
        self,
        local_output: torch.Tensor,  # [M_local, D]
        meta: dict,
    ) -> torch.Tensor:
        """
        Send expert outputs back to the originating GPU.
        Memory-optimized: index_copy_ instead of fancy indexing.
        """
        device = local_output.device
        D = meta["D"]
        M = meta["original_M"]
        send_counts = meta["recv_counts"]  # what we received is what we send back
        recv_counts = meta["send_counts"]  # what we sent is what we receive back

        # Differentiable all-to-all — gradient on the returned output flows back to local_output.
        recv_output = _all_to_all_diff(
            local_output, send_counts, recv_counts, self.ep_group
        )

        # Unsort via inverse permutation — autograd-safe (in-place index_copy_ would
        # break the gradient graph, preventing TQP grads from flowing back).
        sort_idx = meta["sort_idx"]
        inv_sort = sort_idx.argsort()
        output = recv_output[inv_sort]
        del recv_output, sort_idx, inv_sort

        return output


def shard_tq_weights(tq_expert_weights, ep_ctx, device, max_local_experts=None):
    """Shard expert weights to local subset. Pads TQP params to max_local_experts
    for uniform param sizes across ranks (required for DeepSpeed BF16 optimizer
    which flattens params into a single buffer — all ranks must have same shape)."""
    # USE_BF16_BASE shard: if module has base_weight (bf16 mode), shard + pad it like
    # the tqp_A/tqp_B params do below. The codebook/indices/row_norms get sliced too
    # by the existing code path; they're dead weight in this mode but harmless.
    import os as _os_for_bf16_shard

    if _os_for_bf16_shard.environ.get("USE_BF16_BASE", "0") == "1" and hasattr(
        tq_expert_weights, "base_weight"
    ):
        _start = ep_ctx.local_expert_start
        _end = _start + ep_ctx.local_num_experts
        _local_count = ep_ctx.local_num_experts
        with torch.no_grad():
            _bw = (
                tq_expert_weights.base_weight.data[_start:_end]
                .clone()
                .contiguous()
                .to(device)
            )
            if max_local_experts is not None and _local_count < max_local_experts:
                _pad = max_local_experts - _local_count
                _bw_pad = torch.zeros(
                    _pad, *_bw.shape[1:], dtype=_bw.dtype, device=_bw.device
                )
                _bw = torch.cat([_bw, _bw_pad], dim=0)
            tq_expert_weights.base_weight = torch.nn.Parameter(_bw, requires_grad=False)

    start = ep_ctx.local_expert_start
    end = start + ep_ctx.local_num_experts
    local_count = ep_ctx.local_num_experts
    # CRITICAL: clone() so each rank owns its slice and the unsharded original
    # storage gets GC'd.  Without clone(), slicing creates a VIEW that pins the
    # full unsharded tensor alive per-rank → 8 ranks × 119B BF16 OOMs the host.
    if (
        hasattr(tq_expert_weights, "_lazy_init_done")
        and not tq_expert_weights._lazy_init_done
    ):
        if tq_expert_weights._pending_weight is not None:
            tq_expert_weights._pending_weight = (
                tq_expert_weights._pending_weight[start:end].clone().contiguous()
            )
        tq_expert_weights.weight_indices = (
            tq_expert_weights.weight_indices[start:end].clone().contiguous()
        )
        tq_expert_weights.row_norms = (
            tq_expert_weights.row_norms[start:end].clone().contiguous()
        )
    else:
        with torch.no_grad():
            tq_expert_weights.weight_indices = (
                tq_expert_weights.weight_indices[start:end]
                .clone()
                .contiguous()
                .to(device)
            )
            tq_expert_weights.row_norms = (
                tq_expert_weights.row_norms[start:end].clone().contiguous().to(device)
            )

    # Shard TQP params and pad to uniform size across ranks
    with torch.no_grad():
        local_A = tq_expert_weights.tqp_A.data[start:end].to(device)
        local_B = tq_expert_weights.tqp_B.data[start:end].to(device)
        if max_local_experts is not None and local_count < max_local_experts:
            pad = max_local_experts - local_count
            local_A = torch.cat(
                [
                    local_A,
                    torch.zeros(
                        pad, *local_A.shape[1:], dtype=local_A.dtype, device=device
                    ),
                ]
            )
            local_B = torch.cat(
                [
                    local_B,
                    torch.zeros(
                        pad, *local_B.shape[1:], dtype=local_B.dtype, device=device
                    ),
                ]
            )
        tq_expert_weights.tqp_A = torch.nn.Parameter(local_A)
        tq_expert_weights.tqp_B = torch.nn.Parameter(local_B)

    tq_expert_weights.num_experts = local_count
    if tq_expert_weights.rotation_matrix.numel() > 0:
        tq_expert_weights.rotation_matrix = tq_expert_weights.rotation_matrix.to(device)
    if tq_expert_weights.weight_codebook.numel() > 0:
        tq_expert_weights.weight_codebook = tq_expert_weights.weight_codebook.to(device)
    return tq_expert_weights
