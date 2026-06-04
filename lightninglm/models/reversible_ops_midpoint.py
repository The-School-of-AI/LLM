import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint as grad_checkpoint

_REV_CLONE_PARAMS = os.getenv("T19_REV_CLONE_PARAMS", "0") == "1"
_REV_CLONE_BUFFERS = os.getenv("T19_REV_CLONE_BUFFERS", "0") == "1"
_REV_ZERO3_BACKEND = os.getenv("T19_REV_ZERO3_BACKEND", "1") == "1"
# Default ON for ZeRO-3 to reduce hook-driven gather bursts in reversible recompute.
_REV_ZERO3_MANUAL_GATHER = os.getenv("T19_REV_ZERO3_MANUAL_GATHER", "1") == "1"
# Bootstrap checkpointing can amplify ZeRO-3 gather pressure; allow disabling in that mode.
_REV_ZERO3_BOOTSTRAP_CKPT = os.getenv("T19_REV_ZERO3_BOOTSTRAP_CKPT", "0") == "1"
# Optional: switch checkpoint reentrant mode for runtime-retention diagnostics.
_REV_CKPT_USE_REENTRANT = os.getenv("T19_REV_CKPT_USE_REENTRANT", "0") == "1"


def _zero3_gather_ctx(params, fwd_module=None):
    """
    Return DeepSpeed ZeRO gathered-parameter context when parameters are partitioned.
    Falls back to a null context outside ZeRO.
    """
    if not _REV_ZERO3_BACKEND:
        return nullcontext()
    # Let ZeRO's module hooks manage parameter lifecycle by default.
    # Manual gather is available only as an opt-in diagnostic switch.
    if not _REV_ZERO3_MANUAL_GATHER:
        return nullcontext()
    try:
        import deepspeed  # type: ignore
    except Exception:
        return nullcontext()
    zero_mod = getattr(deepspeed, "zero", None)
    gp = getattr(zero_mod, "GatheredParameters", None) if zero_mod is not None else None
    if gp is None:
        return nullcontext()
    # ZeRO params carry ds_id attribute after DeepSpeed initialize.
    if any(hasattr(p, "ds_id") for p in params):
        # ZeRO-3 is sensitive to parameter lifecycle. Use modifier_rank=0 and
        # pass fwd_module when available so DeepSpeed tracks active submodules.
        try:
            return gp(params, modifier_rank=0, fwd_module=fwd_module, enabled=True)
        except TypeError:
            return gp(params, modifier_rank=0, enabled=True)
    return nullcontext()


class _ForceWrapper(nn.Module):
    """
    Wraps a Transformer block so functional_call(module, ...) calls force(x),
    not forward(x). This is CRITICAL for midpoint/leapfrog, where f(x)=delta.
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, x, attention_mask=None):
        # must return (delta, aux)
        return self.layer.force(x, attention_mask=attention_mask)


class MidpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        p_prev,
        p_cur,
        two_h,
        a,
        module,
        attention_mask,
        param_keys,
        buffer_keys,
        *flat_tensors,
    ):
        """
        Implements generalized reversible midpoint:
            p_next = a*p_prev + (1-a)*p_cur + two_h * f(p_cur)
        where f(p_cur) = delta returned by layer.force(p_cur).

        Notes:
        - a=1 gives pure leapfrog: p_next = p_prev + two_h*f(p_cur)
        - a<1 adds a stabilizing blend toward p_cur (still reversible if a!=0)
        """
        n_params = len(param_keys)

        # IMPORTANT: module is a _ForceWrapper, so param/buffer names must be prefixed with "layer."
        params = {f"layer.{k}": v for k, v in zip(param_keys, flat_tensors[:n_params])}
        buffers = {
            f"layer.{k}": v for k, v in zip(buffer_keys, flat_tensors[n_params:])
        }

        # CRITICAL FIX: Save buffer tensors for backward pass
        # This prevents race conditions with NCCL gradient sync in distributed training
        buffer_tensors = flat_tensors[
            n_params:
        ]  # Extract buffer tensors from flat_tensors

        # Save what we truly need for backward (including buffer tensors now!)
        ctx.save_for_backward(p_prev, p_cur, *flat_tensors[:n_params], *buffer_tensors)
        ctx.two_h = float(two_h)
        ctx.a = float(a)
        ctx.module = module
        ctx.attention_mask = attention_mask
        ctx.param_keys = param_keys
        ctx.buffer_keys = buffer_keys
        ctx.n_params = n_params
        ctx.n_buffers = len(buffer_keys)  # Track number of buffers

        with torch.no_grad():
            delta, aux = functional_call(
                module, (params, buffers), (p_cur, attention_mask), tie_weights=True
            )
            p_next = (ctx.a * p_prev) + ((1.0 - ctx.a) * p_cur) + (ctx.two_h * delta)

        return p_next, aux

    @staticmethod
    def backward(ctx, grad_p_next, grad_aux):
        # CRITICAL FIX: Retrieve saved tensors including buffers (no live module access!)
        saved_tensors = ctx.saved_tensors
        n_params = ctx.n_params
        n_buffers = ctx.n_buffers

        p_prev = saved_tensors[0]
        p_cur = saved_tensors[1]
        param_tensors = saved_tensors[2 : 2 + n_params]
        buffer_tensors = saved_tensors[2 + n_params : 2 + n_params + n_buffers]

        # Rebuild params/buffers for functional_call using SAVED buffers
        params = {f"layer.{k}": v for k, v in zip(ctx.param_keys, param_tensors)}
        buffers = {f"layer.{k}": v for k, v in zip(ctx.buffer_keys, buffer_tensors)}

        # Direct paths:
        # p_next = a*p_prev + (1-a)*p_cur + two_h*delta(p_cur)
        grad_p_prev = grad_p_next * ctx.a
        grad_p_cur_direct = grad_p_next * (1.0 - ctx.a)

        # CRITICAL FIX: Ensure CUDA synchronization before recomputation in distributed training
        if torch.cuda.is_available() and dist.is_initialized():
            torch.cuda.synchronize()

        with torch.enable_grad():
            # Keep state clone optional (default off) for strict low-memory operation.
            if _REV_CLONE_PARAMS:
                p_cur_req = p_cur.detach().clone().requires_grad_(True)
            else:
                p_cur_req = p_cur.detach().requires_grad_(True)

            # IMPORTANT for MoE memory: do NOT clone full parameter tensors by default.
            # Cloning expert tensors in backward can dominate VRAM and defeat reversibility.
            if _REV_CLONE_PARAMS:
                param_req = [
                    t.detach().clone().requires_grad_(True) for t in param_tensors
                ]
            else:
                param_req = [t for t in param_tensors]
            params_req = {f"layer.{k}": v for k, v in zip(ctx.param_keys, param_req)}

            if _REV_CLONE_BUFFERS:
                buffers_req = {
                    k: v.detach().clone() if v is not None else None
                    for k, v in buffers.items()
                }
            else:
                buffers_req = buffers

            use_amp = p_cur_req.is_cuda and p_cur_req.dtype in (
                torch.float16,
                torch.bfloat16,
            )
            amp_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=p_cur_req.dtype)
                if use_amp
                else nullcontext()
            )
            with amp_ctx:
                delta, aux = functional_call(
                    ctx.module,
                    (params_req, buffers_req),
                    (p_cur_req, ctx.attention_mask),
                    tie_weights=False,
                )

            if grad_aux is None:
                # aux may be scalar or tensor
                grad_aux = torch.zeros_like(aux)

            grad_delta = grad_p_next * ctx.two_h

            grads = torch.autograd.grad(
                outputs=(delta, aux),
                inputs=(p_cur_req, *param_req),
                grad_outputs=(grad_delta.to(delta.dtype), grad_aux.to(aux.dtype)),
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

        grad_p_cur_through_f = (
            grads[0] if grads[0] is not None else torch.zeros_like(p_cur)
        )
        grad_p_cur = grad_p_cur_direct + grad_p_cur_through_f

        grad_params = grads[1:]
        grad_params = [
            g if g is not None else torch.zeros_like(t)
            for g, t in zip(grad_params, param_tensors)
        ]

        # Return grads for (p_prev, p_cur, two_h, a, module, param_keys, buffer_keys, *flat_tensors)
        # Non-tensor args -> None
        grad_two_h = None
        grad_a = None
        grad_module = None
        grad_attention_mask = None
        grad_param_keys = None
        grad_buffer_keys = None

        # buffers are non-diff
        grad_buffers = (None,) * n_buffers

        return (
            grad_p_prev,
            grad_p_cur,
            grad_two_h,
            grad_a,
            grad_module,
            grad_attention_mask,
            grad_param_keys,
            grad_buffer_keys,
            *grad_params,
            *grad_buffers,
        )


class MidpointFunctionZero3(torch.autograd.Function):
    """
    ZeRO-3 compatible reversible midpoint rule.

    Difference vs MidpointFunction:
    - does NOT pass model parameters as explicit autograd.Function inputs.
    - gathers sharded params only around recompute call, and accumulates param grads
      directly via autograd.backward on real module params.
    """

    @staticmethod
    def forward(ctx, p_prev, p_cur, two_h, a, module, attention_mask):
        ctx.save_for_backward(p_prev, p_cur)
        ctx.two_h = float(two_h)
        ctx.a = float(a)
        ctx.module = module
        ctx.attention_mask = attention_mask

        with torch.no_grad():
            params = [p for p in module.layer.parameters() if p.requires_grad]
            with _zero3_gather_ctx(params, fwd_module=module.layer):
                delta, aux = module(p_cur, attention_mask)
            p_next = (ctx.a * p_prev) + ((1.0 - ctx.a) * p_cur) + (ctx.two_h * delta)
        return p_next, aux

    @staticmethod
    def backward(ctx, grad_p_next, grad_aux):
        p_prev, p_cur = ctx.saved_tensors

        grad_p_prev = grad_p_next * ctx.a
        grad_p_cur_direct = grad_p_next * (1.0 - ctx.a)

        if torch.cuda.is_available() and dist.is_initialized():
            torch.cuda.synchronize()

        with torch.enable_grad():
            if _REV_CLONE_PARAMS:
                p_cur_req = p_cur.detach().clone().requires_grad_(True)
            else:
                p_cur_req = p_cur.detach().requires_grad_(True)

            params = [p for p in ctx.module.layer.parameters() if p.requires_grad]
            with _zero3_gather_ctx(params, fwd_module=ctx.module.layer):
                use_amp = p_cur_req.is_cuda and p_cur_req.dtype in (
                    torch.float16,
                    torch.bfloat16,
                )
                amp_ctx = (
                    torch.amp.autocast(device_type="cuda", dtype=p_cur_req.dtype)
                    if use_amp
                    else nullcontext()
                )
                with amp_ctx:
                    delta, aux = ctx.module(p_cur_req, ctx.attention_mask)

                if grad_aux is None:
                    grad_aux = torch.zeros_like(aux)
                grad_delta = grad_p_next * ctx.two_h

                # Accumulate grads directly onto real module params (ZeRO-safe).
                torch.autograd.backward(
                    (delta, aux),
                    (grad_delta.to(delta.dtype), grad_aux.to(aux.dtype)),
                    retain_graph=False,
                    create_graph=False,
                )

            grad_p_cur_through_f = (
                p_cur_req.grad
                if p_cur_req.grad is not None
                else torch.zeros_like(p_cur)
            )

        grad_p_cur = grad_p_cur_direct + grad_p_cur_through_f
        grad_two_h = None
        grad_a = None
        grad_module = None
        grad_attention_mask = None
        return (
            grad_p_prev,
            grad_p_cur,
            grad_two_h,
            grad_a,
            grad_module,
            grad_attention_mask,
        )


class MidpointBlock(nn.Module):
    def __init__(self, block: nn.Module, step_size: float, a: float):
        super().__init__()
        self.block = block
        self.wrapper = _ForceWrapper(block)

        # two_h corresponds to 2h in the leapfrog form
        self.two_h = float(2.0 * step_size)
        self.a = float(a)

        # Cache keys (from original block) so mapping is stable
        self.param_keys = list(dict(block.named_parameters()).keys())
        self.buffer_keys = list(dict(block.named_buffers()).keys())

    def _use_zero3_backend(self) -> bool:
        if not _REV_ZERO3_BACKEND:
            return False
        try:
            return any(hasattr(p, "ds_id") for p in self.block.parameters())
        except Exception:
            return False

    def forward(self, p_prev, p_cur, attention_mask=None):
        if self._use_zero3_backend():
            return MidpointFunctionZero3.apply(
                p_prev,
                p_cur,
                self.two_h,
                self.a,
                self.wrapper,
                attention_mask,
            )

        param_values = [p for p in self.block.parameters()]
        buffer_values = [b for b in self.block.buffers()]
        return MidpointFunction.apply(
            p_prev,
            p_cur,
            self.two_h,
            self.a,
            self.wrapper,
            attention_mask,
            self.param_keys,
            self.buffer_keys,
            *param_values,
            *buffer_values,
        )


class ReversibleMidpointStack(nn.Module):
    """
    Forward-only stack that implements:
        bootstrap to create (p_prev, p_cur)
        then midpoint recurrence for subsequent layers.

    Key knobs:
    - step_size: h
    - a: stabilizing blend coefficient (a=1 pure leapfrog; 0.85–0.98 often helps)
    - bootstrap: "no_kick" or "euler"
    - noise_eps: optional noise to delta during training
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        step_size: float = 0.05,
        a: float = 0.95,
        noise_eps: float = 0.0,
        bootstrap: str = "no_kick",
    ):
        super().__init__()
        assert 0.0 <= a <= 1.0, "a must be in [0,1]"
        assert bootstrap in (
            "no_kick",
            "euler",
        ), "bootstrap must be 'no_kick' or 'euler'"

        self.blocks = blocks
        self.h = float(step_size)
        self.a = float(a)
        self.noise_eps = float(noise_eps)
        self.bootstrap = bootstrap

        self.bootstrap_layer = blocks[0]
        self.mid_layers = nn.ModuleList(
            [MidpointBlock(b, step_size=self.h, a=self.a) for b in blocks[1:]]
        )

        self.step_count = 0

    def _use_zero3_runtime(self) -> bool:
        if not _REV_ZERO3_BACKEND:
            return False
        try:
            return any(hasattr(p, "ds_id") for p in self.blocks.parameters())
        except Exception:
            return False

    def forward(self, x, attention_mask=None):
        # Bootstrap creates two states (p_prev, p_cur)
        p_prev = x

        if self.bootstrap == "no_kick":
            # Baseline-aligned start: p_cur = p_prev (no Euler kick)
            p_cur = p_prev
            if self._use_zero3_runtime() and (not _REV_ZERO3_BOOTSTRAP_CKPT):
                # In ZeRO-3, checkpoint re-entry can trigger extra gather/prefetch bursts.
                # Prefer direct call here; reversibility still avoids storing full stack activations.
                delta0, aux0 = self.bootstrap_layer.force(
                    p_cur, attention_mask=attention_mask
                )
            else:
                # Gradient checkpointing: bootstrap runs WITH autograd; without checkpoint, a Python
                # for-loop over T tokens retains ~160 MB per step (v_outer, k_outer, S, etc.) → T*160MB OOM.
                # See MEMORY_OOM_REPORT in docs/ or scripts/diagnose_memory.py.
                delta0, aux0 = grad_checkpoint(
                    lambda p: self.bootstrap_layer.force(
                        p, attention_mask=attention_mask
                    ),
                    p_cur,
                    use_reentrant=_REV_CKPT_USE_REENTRANT,
                )
        # else:
        #     # Euler kick start: p_cur = p_prev + h*delta(p_prev)
        #     delta0, aux0 = self.bootstrap_layer.force(p_prev)
        #     if self.training and self.noise_eps > 0:
        #         delta0 = delta0 + self.noise_eps * torch.randn_like(delta0)
        #     p_cur = p_prev + (self.h * delta0)
        else:
            # HALF-STEP Euler bootstrap (paper-consistent + stable for h=0.25, a=0.5)
            # Gradient checkpointing: same as no_kick — avoids T-step autograd retention (see above).
            if self._use_zero3_runtime() and (not _REV_ZERO3_BOOTSTRAP_CKPT):
                delta0, aux0 = self.bootstrap_layer.force(
                    p_prev, attention_mask=attention_mask
                )
            else:
                delta0, aux0 = grad_checkpoint(
                    lambda p: self.bootstrap_layer.force(
                        p, attention_mask=attention_mask
                    ),
                    p_prev,
                    use_reentrant=_REV_CKPT_USE_REENTRANT,
                )
            if self.training and self.noise_eps > 0:
                delta0 = delta0 + self.noise_eps * torch.randn_like(delta0)

            # critical change: half-step, NOT full h
            p_cur = p_prev + (0.5 * self.h * delta0)

        total_aux = (
            aux0
            if aux0 is not None
            else torch.tensor(0.0, device=x.device, dtype=torch.float32)
        )

        # Midpoint / leapfrog recurrence
        for layer in self.mid_layers:
            p_next, aux = layer(p_prev, p_cur, attention_mask=attention_mask)
            if aux is not None:
                total_aux = total_aux + aux
            p_prev, p_cur = p_cur, p_next

        # Scale aux so total magnitude matches 8-layer (3B) baseline regardless of depth.
        # 3B (8 layers) → total_aux ≈ 0.3 (stable, ~2.4% of NTP loss).
        # Without scaling, deeper models accumulate proportionally more aux gradient,
        # destabilizing routers. Factor = 8/N keeps the operating point constant.
        num_layers = 1 + len(self.mid_layers)  # bootstrap + midpoint layers
        if num_layers > 8:
            total_aux = total_aux * (8.0 / num_layers)

        if self.training:
            self.step_count += 1

        return p_cur, total_aux
