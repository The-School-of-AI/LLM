import torch
import torch.nn as nn
from torch.func import functional_call


class _ForceWrapper(nn.Module):
    """
    Wraps a Transformer block so functional_call(module, ...) calls force(x),
    not forward(x). This is CRITICAL for midpoint/leapfrog, where f(x)=delta.
    """
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        # must return (delta, aux)
        return self.layer.force(x)


class MidpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p_prev, p_cur, two_h, a, module, param_keys, buffer_keys, *flat_tensors):
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
        buffers = {f"layer.{k}": v for k, v in zip(buffer_keys, flat_tensors[n_params:])}

        # CRITICAL FIX: Save buffer tensors for backward pass
        # This prevents race conditions with NCCL gradient sync in distributed training
        buffer_tensors = flat_tensors[n_params:]  # Extract buffer tensors from flat_tensors
        
        # Save what we truly need for backward (including buffer tensors now!)
        ctx.save_for_backward(p_prev, p_cur, *flat_tensors[:n_params], *buffer_tensors)
        ctx.two_h = float(two_h)
        ctx.a = float(a)
        ctx.module = module
        ctx.param_keys = param_keys
        ctx.buffer_keys = buffer_keys
        ctx.n_params = n_params
        ctx.n_buffers = len(buffer_keys)  # Track number of buffers

        with torch.no_grad():
            delta, aux = functional_call(module, (params, buffers), (p_cur,), tie_weights=True)
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
        param_tensors = saved_tensors[2:2+n_params]
        buffer_tensors = saved_tensors[2+n_params:2+n_params+n_buffers]

        # Rebuild params/buffers for functional_call using SAVED buffers
        params = {f"layer.{k}": v for k, v in zip(ctx.param_keys, param_tensors)}
        buffers = {f"layer.{k}": v for k, v in zip(ctx.buffer_keys, buffer_tensors)}

        # Direct paths:
        # p_next = a*p_prev + (1-a)*p_cur + two_h*delta(p_cur)
        grad_p_prev = grad_p_next * ctx.a
        grad_p_cur_direct = grad_p_next * (1.0 - ctx.a)

        # NOTE: No torch.cuda.synchronize() needed here.
        # The backward runs entirely on the default CUDA stream. All operations
        # below (clone, functional_call, autograd.grad) are properly ordered via
        # intra-stream serialization. NCCL allreduce (via overlap_comm) runs on
        # a separate stream but only touches .grad tensors — never the parameter
        # *value* tensors we read here. Additionally, we clone all tensors below,
        # creating independent memory. PyTorch's NCCL backend manages its own
        # stream synchronization automatically (same pattern as torch.utils.checkpoint).

        with torch.enable_grad():
            # Clone and detach to ensure no shared memory with forward pass
            p_cur_req = p_cur.detach().clone().requires_grad_(True)

            # Need param tensors to require_grad for autograd.grad to produce param grads
            # Clone parameters to avoid any shared memory issues
            param_req = [t.detach().clone().requires_grad_(True) for t in param_tensors]
            params_req = {f"layer.{k}": v for k, v in zip(ctx.param_keys, param_req)}
            
            # Clone buffers to ensure thread safety
            buffers_cloned = {k: v.detach().clone() if v is not None else None for k, v in buffers.items()}

            delta, aux = functional_call(ctx.module, (params_req, buffers_cloned), (p_cur_req,), tie_weights=False)

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

        grad_p_cur_through_f = grads[0] if grads[0] is not None else torch.zeros_like(p_cur)
        grad_p_cur = grad_p_cur_direct + grad_p_cur_through_f

        grad_params = grads[1:]
        grad_params = [g if g is not None else torch.zeros_like(t) for g, t in zip(grad_params, param_tensors)]

        # Return grads for (p_prev, p_cur, two_h, a, module, param_keys, buffer_keys, *flat_tensors)
        # Non-tensor args -> None
        grad_two_h = None
        grad_a = None
        grad_module = None
        grad_param_keys = None
        grad_buffer_keys = None

        # buffers are non-diff
        grad_buffers = (None,) * n_buffers

        return (grad_p_prev, grad_p_cur, grad_two_h, grad_a, grad_module, grad_param_keys, grad_buffer_keys, *grad_params, *grad_buffers)


class BootstrapFunction(torch.autograd.Function):
    """Runs the bootstrap layer under torch.no_grad() and recomputes in backward.

    This is the same save-and-recompute pattern as MidpointFunction but for the
    Euler half-step: ``p_cur = p_prev + scale * delta(p_prev)``.
    Without this wrapper the bootstrap layer's full autograd graph is retained,
    consuming ~2 GB for a single layer.
    """

    @staticmethod
    def forward(ctx, p_prev, scale, module, param_keys, buffer_keys, *flat_tensors):
        n_params = len(param_keys)
        params = {f"layer.{k}": v for k, v in zip(param_keys, flat_tensors[:n_params])}
        buffers = {f"layer.{k}": v for k, v in zip(buffer_keys, flat_tensors[n_params:])}
        buffer_tensors = flat_tensors[n_params:]

        ctx.save_for_backward(p_prev, *flat_tensors[:n_params], *buffer_tensors)
        ctx.scale = float(scale)
        ctx.module = module
        ctx.param_keys = param_keys
        ctx.buffer_keys = buffer_keys
        ctx.n_params = n_params
        ctx.n_buffers = len(buffer_keys)

        with torch.no_grad():
            delta, aux = functional_call(module, (params, buffers), (p_prev,), tie_weights=True)
            p_cur = p_prev + (ctx.scale * delta)

        return p_cur, aux

    @staticmethod
    def backward(ctx, grad_p_cur, grad_aux):
        saved = ctx.saved_tensors
        n_params = ctx.n_params
        n_buffers = ctx.n_buffers

        p_prev = saved[0]
        param_tensors = saved[1:1 + n_params]
        buffer_tensors = saved[1 + n_params:1 + n_params + n_buffers]

        buffers = {f"layer.{k}": v for k, v in zip(ctx.buffer_keys, buffer_tensors)}

        # grad through p_cur = p_prev + scale * delta(p_prev)
        # dp_prev_direct = grad_p_cur (identity path)
        grad_p_prev_direct = grad_p_cur

        with torch.enable_grad():
            p_prev_req = p_prev.detach().clone().requires_grad_(True)
            param_req = [t.detach().clone().requires_grad_(True) for t in param_tensors]
            params_req = {f"layer.{k}": v for k, v in zip(ctx.param_keys, param_req)}
            buffers_cloned = {k: v.detach().clone() if v is not None else None for k, v in buffers.items()}

            delta, aux = functional_call(ctx.module, (params_req, buffers_cloned), (p_prev_req,), tie_weights=False)

            if grad_aux is None:
                grad_aux = torch.zeros_like(aux)

            grad_delta = grad_p_cur * ctx.scale

            grads = torch.autograd.grad(
                outputs=(delta, aux),
                inputs=(p_prev_req, *param_req),
                grad_outputs=(grad_delta.to(delta.dtype), grad_aux.to(aux.dtype)),
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

        grad_p_prev_through_f = grads[0] if grads[0] is not None else torch.zeros_like(p_prev)
        grad_p_prev = grad_p_prev_direct + grad_p_prev_through_f

        grad_params = grads[1:]
        grad_params = [g if g is not None else torch.zeros_like(t) for g, t in zip(grad_params, param_tensors)]

        grad_buffers = (None,) * n_buffers
        return (grad_p_prev, None, None, None, None, *grad_params, *grad_buffers)


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

    def forward(self, p_prev, p_cur):
        param_values = [p for p in self.block.parameters()]
        buffer_values = [b for b in self.block.buffers()]
        return MidpointFunction.apply(
            p_prev,
            p_cur,
            self.two_h,
            self.a,
            self.wrapper,
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
        assert bootstrap in ("no_kick", "euler"), "bootstrap must be 'no_kick' or 'euler'"

        self.blocks = blocks
        self.h = float(step_size)
        self.a = float(a)
        self.noise_eps = float(noise_eps)
        self.bootstrap = bootstrap

        self.bootstrap_layer = blocks[0]
        self.bootstrap_wrapper = _ForceWrapper(blocks[0])
        self.bootstrap_param_keys = list(dict(blocks[0].named_parameters()).keys())
        self.bootstrap_buffer_keys = list(dict(blocks[0].named_buffers()).keys())
        self.mid_layers = nn.ModuleList([MidpointBlock(b, step_size=self.h, a=self.a) for b in blocks[1:]])

        self.step_count = 0

    def _bootstrap_via_function(self, p_prev, scale):
        """Run bootstrap layer through BootstrapFunction (no_grad forward, recompute backward)."""
        param_values = list(self.bootstrap_layer.parameters())
        buffer_values = list(self.bootstrap_layer.buffers())
        return BootstrapFunction.apply(
            p_prev,
            scale,
            self.bootstrap_wrapper,
            self.bootstrap_param_keys,
            self.bootstrap_buffer_keys,
            *param_values,
            *buffer_values,
        )

    def forward(self, x):
        # Bootstrap creates two states (p_prev, p_cur)
        p_prev = x

        if self.bootstrap == "no_kick":
            # Baseline-aligned start: p_cur = p_prev (no Euler kick)
            # scale=0 means p_cur = p_prev + 0*delta = p_prev
            # We still run through BootstrapFunction to get aux and param grads.
            # The delta is computed but scaled by 0 for p_cur.
            p_cur = p_prev
            # For no_kick we only need aux and param grads from the bootstrap.
            # Use scale=0 so p_cur stays equal to p_prev.
            _, aux0 = self._bootstrap_via_function(p_prev, 0.0)
        else:
            # HALF-STEP Euler bootstrap (paper-consistent + stable for h=0.25, a=0.5)
            # p_cur = p_prev + 0.5*h * delta(p_prev)
            p_cur, aux0 = self._bootstrap_via_function(p_prev, 0.5 * self.h)

        total_aux = aux0 if aux0 is not None else torch.tensor(0.0, device=x.device, dtype=torch.float32)

        # Midpoint / leapfrog recurrence
        for layer in self.mid_layers:
            p_next, aux = layer(p_prev, p_cur)
            if aux is not None:
                total_aux = total_aux + aux
            p_prev, p_cur = p_cur, p_next

        if self.training:
            self.step_count += 1

        return p_cur, total_aux