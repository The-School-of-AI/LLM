import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint as grad_checkpoint


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
    """
    ZeRO-3 compatible reversible midpoint integration.

    Previous implementation used functional_call() to call module layers,
    which bypasses DeepSpeed ZeRO-3's per-module parameter-gathering hooks
    (allgather before forward, release after).  This rewrite calls
    module(p_cur) directly so ZeRO-3 hooks fire naturally.

    Parameter gradients are accumulated via torch.autograd.backward() inside
    this Function's backward() rather than being returned as explicit outputs.
    This is mathematically identical for all ZeRO stages (0/1/2/3).

    Backward compatibility: when ZeRO-3 is not active, behaviour is unchanged.
    """

    @staticmethod
    def forward(ctx, p_prev, p_cur, two_h, a, module, attention_mask):
        """
        Implements generalized reversible midpoint:
            p_next = a*p_prev + (1-a)*p_cur + two_h * f(p_cur)
        where f(p_cur) = delta returned by layer.force(p_cur).

        Notes:
        - a=1 gives pure leapfrog: p_next = p_prev + two_h*f(p_cur)
        - a<1 adds a stabilizing blend toward p_cur (still reversible if a!=0)
        """
        ctx.two_h = float(two_h)
        ctx.a = float(a)
        ctx.module = module
        ctx.attention_mask = attention_mask

        # Save only activations — NOT parameters or buffers.
        # Parameters are accessed from ctx.module in backward;
        # ZeRO-3 hooks gather them on-demand during the recomputation call.
        ctx.save_for_backward(p_prev, p_cur)

        with torch.no_grad():
            # Direct module call: ZeRO-3 pre-forward hooks fire here,
            # gathering partitioned parameters before each submodule,
            # and releasing them after.
            delta, aux = module(p_cur, attention_mask=attention_mask)
            p_next = (a * p_prev) + ((1.0 - a) * p_cur) + (two_h * delta)

        return p_next, aux

    @staticmethod
    def backward(ctx, grad_p_next, grad_aux):
        p_prev, p_cur = ctx.saved_tensors

        # Direct paths from:
        # p_next = a*p_prev + (1-a)*p_cur + two_h*delta(p_cur)
        grad_p_prev = grad_p_next * ctx.a
        grad_p_cur_direct = grad_p_next * (1.0 - ctx.a)

        # CRITICAL FIX: Ensure CUDA synchronization before recomputation
        # in distributed training to prevent race conditions with NCCL.
        if torch.cuda.is_available() and dist.is_initialized():
            torch.cuda.synchronize()

        # ── Recompute f(p_cur) with grad enabled ────────────────────────
        # Direct module call ensures ZeRO-3 hooks fire for parameter
        # gathering. Parameter gradients accumulate naturally via
        # torch.autograd.backward — no need to return them explicitly
        # from this Function.
        with torch.enable_grad():
            # Clone and detach to prevent shared memory with forward pass
            p_cur_req = p_cur.detach().clone().requires_grad_(True)

            # Recompute: ZeRO-3 pre-forward hooks gather params; post-forward
            # hooks release them.  Backward hooks then re-gather for gradient
            # computation and partition the resulting gradients.
            delta, aux = ctx.module(p_cur_req, attention_mask=ctx.attention_mask)

            if grad_aux is None:
                # aux may be scalar or tensor
                grad_aux = torch.zeros_like(aux)

            grad_delta = grad_p_next * ctx.two_h

            # Backward through BOTH delta and aux so that:
            #   - param grads include contributions from aux (e.g. MoE router)
            #   - p_cur_req.grad includes contributions from both paths
            torch.autograd.backward(
                [delta, aux],
                [grad_delta.to(delta.dtype), grad_aux.to(aux.dtype)],
            )

        grad_p_cur_through_f = (
            p_cur_req.grad if p_cur_req.grad is not None
            else torch.zeros_like(p_cur)
        )
        grad_p_cur = grad_p_cur_direct + grad_p_cur_through_f

        # Return grads for (p_prev, p_cur, two_h, a, module, attention_mask)
        # Non-tensor / non-differentiable args → None
        return grad_p_prev, grad_p_cur, None, None, None, None


class MidpointBlock(nn.Module):
    def __init__(self, block: nn.Module, step_size: float, a: float):
        super().__init__()
        self.block = block
        self.wrapper = _ForceWrapper(block)

        # two_h corresponds to 2h in the leapfrog form
        self.two_h = float(2.0 * step_size)
        self.a = float(a)

    def forward(self, p_prev, p_cur, attention_mask=None):
        return MidpointFunction.apply(
            p_prev,
            p_cur,
            self.two_h,
            self.a,
            self.wrapper,
            attention_mask,
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
        assert bootstrap in ("no_kick", "euler"), "bootstrap must be 'no_kick' or 'euler'"

        self.blocks = blocks
        self.h = float(step_size)
        self.a = float(a)
        self.noise_eps = float(noise_eps)
        self.bootstrap = bootstrap

        self.bootstrap_layer = blocks[0]
        self.mid_layers = nn.ModuleList([MidpointBlock(b, step_size=self.h, a=self.a) for b in blocks[1:]])

        self.step_count = 0

    def forward(self, x, attention_mask=None):
        # Bootstrap creates two states (p_prev, p_cur)
        p_prev = x

        if self.bootstrap == "no_kick":
            # Baseline-aligned start: p_cur = p_prev (no Euler kick)
            p_cur = p_prev
            # Gradient checkpointing: bootstrap runs WITH autograd; without checkpoint, a Python
            # for-loop over T tokens retains ~160 MB per step (v_outer, k_outer, S, etc.) → T*160MB OOM.
            # See MEMORY_OOM_REPORT in docs/ or scripts/diagnose_memory.py.
            delta0, aux0 = grad_checkpoint(
                lambda p: self.bootstrap_layer.force(p, attention_mask=attention_mask), p_cur, use_reentrant=False
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
            delta0, aux0 = grad_checkpoint(
                lambda p: self.bootstrap_layer.force(p, attention_mask=attention_mask), p_prev, use_reentrant=False
            )
            if self.training and self.noise_eps > 0:
                delta0 = delta0 + self.noise_eps * torch.randn_like(delta0)

            # critical change: half-step, NOT full h
            p_cur = p_prev + (0.5 * self.h * delta0)

        total_aux = aux0 if aux0 is not None else torch.tensor(0.0, device=x.device, dtype=torch.float32)

        # Midpoint / leapfrog recurrence
        for layer in self.mid_layers:
            p_next, aux = layer(p_prev, p_cur, attention_mask=attention_mask)
            if aux is not None:
                total_aux = total_aux + aux
            p_prev, p_cur = p_cur, p_next

        if self.training:
            self.step_count += 1

        return p_cur, total_aux