"""
TQP Integration for Production MoE Training.

Provides:
  - apply_tqp(model): wrap MoE expert weights with TQP
  - rebuild_reversible_cached_keys(model): fix reversible stack after wrapping
  - get_tqp_param_groups(model, base_lr, adapter_lr_mult): two AdamW param groups
  - flush_tqp(model, optimizer, step): periodic flush + logging
"""

import os
import time

import torch

from lightninglm.models.reversible_ops_midpoint import ReversibleMidpointStack
from lightninglm.tqp.turboquant_pretraining_linear import (
    TurboQuantPretrainingExpertWeights,
    TurboQuantPretrainingMoEWrapper,
)
from lightninglm.utils.utils import print_rank_0


def _find_moe_modules(model):
    """
    Walk model.layers (and model.mtp_block if present) to find MoEFFN modules.

    Layer structure: layer.mlp_block.sublayer.moe (LightningMLP wraps MoEFFN)
    Also handles: layer.mlp_block.sublayer (if sublayer IS the MoEFFN directly)

    MTP block has the same internal structure (mlp_block.sublayer.moe) and its
    own dense W_gate/W_up/W_down which must be wrapped to avoid OOM during
    reversible-recompute backward.
    """
    containers = list(model.layers)
    mtp = getattr(model, "mtp_block", None)
    if mtp is not None:
        containers.append(mtp)

    results = []
    for i, layer in enumerate(containers):
        mlp_block = getattr(layer, "mlp_block", None)
        if mlp_block is None:
            continue
        sublayer = getattr(mlp_block, "sublayer", None)
        if sublayer is None:
            continue
        moe = getattr(sublayer, "moe", None)
        if moe is not None and hasattr(moe, "W_gate"):
            results.append((i, sublayer, "moe", moe))
        elif hasattr(sublayer, "W_gate"):
            results.append((i, layer.mlp_block, "sublayer", sublayer))
    return results


def apply_tqp(model, weight_bits=4, rank=16, flush_every=25, skip_materialize=False):
    """
    Wrap every MoE FFN layer's expert weights with TQP.

    Expert weights (W_gate, W_up, W_down) become:
      - int8 weight_indices (frozen between flushes)
      - fp32 tqp_A, tqp_B (trained via AdamW)
      - Periodic flush absorbs A@B into base weights

    Shared experts (nn.Linear) are NOT wrapped — they stay as-is.
    """
    moe_locations = _find_moe_modules(model)
    n_wrapped = 0
    total_tqp_params = 0

    for i, parent, attr_name, moe_module in moe_locations:
        wrapper = TurboQuantPretrainingMoEWrapper(
            moe_module,
            weight_bits=weight_bits,
            rank=rank,
            rotation_seed=42 + i,
            quantize_shared=False,
            lazy=True,
        )

        setattr(parent, attr_name, wrapper)

        for name, p in wrapper.named_parameters():
            if "tqp_A" in name or "tqp_B" in name:
                total_tqp_params += p.numel()

        n_wrapped += 1
        print_rank_0(
            f"  [TQP] Layer {i}: wrapped MoE ({moe_module.num_experts} experts)"
        )

    print_rank_0(f"  [TQP] Wrapped {n_wrapped} MoE layers total")
    print_rank_0(f"  [TQP] TQP adapter params: {total_tqp_params / 1e6:.1f}M")

    # Materialize all lazy wrappers using GPU for fast rotation + quantization.
    # When EP is enabled (skip_materialize=True), defer to after EP sharding
    # so each GPU only materializes its local ~58 experts instead of all 460 (8x faster).
    import gc

    gc.collect()

    if skip_materialize:
        print_rank_0("  [TQP] Skipping materialization (will run after EP sharding)")
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpu_device = torch.device(f"cuda:{local_rank}")
        print_rank_0(f"  [TQP] Materializing quantization on GPU {local_rank}...")
        import time as _time

        _t0 = _time.time()
        _materialized = 0
        for module in model.modules():
            if isinstance(module, TurboQuantPretrainingExpertWeights):
                if hasattr(module, "_lazy_init_done") and not module._lazy_init_done:
                    module._materialize_lazy_gpu(gpu_device)
                    _materialized += 1
                    if _materialized % 10 == 0:
                        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        _dt = _time.time() - _t0
        print_rank_0(
            f"  [TQP] Materialized {_materialized} expert weight sets in {_dt:.1f}s"
        )

    # Move TQP buffers strategically:
    # - weight_indices (65 GB for 260 experts): KEEP ON CPU, load per-expert on demand
    # - row_norms: KEEP ON CPU (loaded per-expert in _dequant_single_expert)
    # - rotation_matrix, weight_codebook: MOVE TO GPU (small, shared across experts)
    _ref_param = next(model.parameters())
    _target_device = _ref_param.device
    _indices_kept_cpu = 0
    _bufs_moved_gpu = 0
    for module in model.modules():
        if isinstance(module, TurboQuantPretrainingExpertWeights):
            for name, buf in module.named_buffers():
                # Move ALL buffers to GPU (b300/p5en has enough memory)
                if buf.device != _target_device:
                    setattr(module, name, buf.to(_target_device))
                _bufs_moved_gpu += buf.numel() * buf.element_size()
    print_rank_0(
        f"  [TQP] Buffers on CPU: {_indices_kept_cpu / 1e9:.1f} GB "
        f"(indices+norms), GPU: {_bufs_moved_gpu / 1e9:.2f} GB (rotation+codebook)"
    )

    rebuild_reversible_cached_keys(model)

    return model


def rebuild_reversible_cached_keys(model):
    """
    After swapping MoE wrappers, the reversible stack's MidpointBlock.param_keys
    and buffer_keys caches are stale. Rebuild them from the current module tree.
    """
    if not hasattr(model, "stack"):
        print_rank_0("  [TQP] No stack attribute found, skipping key rebuild")
        return
    if not isinstance(model.stack, ReversibleMidpointStack):
        print_rank_0(
            f"  [TQP] Stack is {type(model.stack).__name__}, not ReversibleMidpointStack — skipping"
        )
        return

    for mid_block in model.stack.mid_layers:
        block = mid_block.block
        # INCLUDE all params (including tqp_A/tqp_B) in param_keys.
        # functional_call reparametrizes them, so compute_expert_chunk's
        # self.tqp_A[idx] accesses the reparametrized clone.
        # torch.autograd.grad then returns gradients for tqp params.
        #
        # CRITICAL: param_keys MUST align with list(block.parameters()).
        # Filtering keys without filtering values causes a fatal mismatch
        # where wrong tensors map to wrong keys and tqp grads are lost.
        mid_block.param_keys = list(dict(block.named_parameters()).keys())
        mid_block.buffer_keys = list(dict(block.named_buffers()).keys())

    print_rank_0(
        f"  [TQP] Rebuilt reversible stack cached keys for "
        f"{len(model.stack.mid_layers)} mid_layers"
    )


def get_tqp_param_groups(
    model, base_lr, adapter_lr_mult=10.0, router_rest_lr_ratio=1.0
):
    """
    Split model parameters into 3 groups with different LRs:
      1. Router params (gate.gate.weight, gate.gate.bias) at base_lr
      2. Rest (backbone, shared experts, etc.) at base_lr * router_rest_lr_ratio
      3. TQP adapter params (tqp_A, tqp_B) at base_lr * adapter_lr_mult

    When router_rest_lr_ratio=0.01 (Stage 1), backbone runs at 1/100 of router LR.
    """
    router_params = []
    rest_params = []
    tqp_params = []
    router_count = 0
    rest_count = 0
    tqp_count = 0

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "tqp_A" in name or "tqp_B" in name:
            tqp_params.append(p)
            tqp_count += p.numel()
        elif "gate.gate.weight" in name or "gate.gate.bias" in name:
            router_params.append(p)
            router_count += p.numel()
        else:
            rest_params.append(p)
            rest_count += p.numel()

    router_lr = base_lr
    rest_lr = base_lr * router_rest_lr_ratio
    adapter_lr = base_lr * adapter_lr_mult

    print_rank_0("  [TQP] Param groups (3-way split):")
    print_rank_0(
        f"    Router: {router_count / 1e6:.1f}M params @ lr={router_lr:.2e} (1.0x)"
    )
    print_rank_0(
        f"    Rest:   {rest_count / 1e6:.1f}M params @ lr={rest_lr:.2e} ({router_rest_lr_ratio}x)"
    )
    print_rank_0(
        f"    TQP:   {tqp_count / 1e6:.1f}M params @ lr={adapter_lr:.2e} ({adapter_lr_mult}x)"
    )

    return [
        {"params": router_params, "lr": router_lr, "name": "router"},
        {"params": rest_params, "lr": rest_lr, "name": "rest"},
        {"params": tqp_params, "lr": adapter_lr, "name": "tqp"},
    ]


def flush_tqp(model, optimizer, step, rank=0):
    """
    Flush all TQP layers: absorb tqp_A @ tqp_B into quantized base weights.
    Reset TQP params and clear optimizer state for them.

    Call this every flush_every steps.
    """
    t0 = time.time()
    n_flushed = 0
    total_ab_norm = 0.0
    total_w_change = 0.0

    for module in model.modules():
        if isinstance(module, TurboQuantPretrainingMoEWrapper):
            stats = module.flush_all()
            n_flushed += 1
            total_ab_norm += stats.get("AB_norm_mean", 0.0)
            total_w_change += stats.get("w_change_mean", 0.0) * 100

    if n_flushed == 0:
        return

    for name, param in model.named_parameters():
        if "tqp_A" in name or "tqp_B" in name:
            if param in optimizer.state:
                optimizer.state[param] = {}

    dt = time.time() - t0
    avg_ab = total_ab_norm / max(n_flushed, 1)
    avg_w_pct = total_w_change / max(n_flushed, 1)

    if rank == 0:
        print_rank_0(
            f"  [TQP FLUSH] step={step} | layers={n_flushed} | "
            f"avg_||AB||={avg_ab:.3f} | avg_w%={avg_w_pct:.1f}% | "
            f"time={dt:.2f}s"
        )

    rebuild_reversible_cached_keys(model)
