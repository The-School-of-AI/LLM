import torch
import torch.nn as nn
from torch.func import functional_call
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
    @staticmethod
    def forward(ctx, p_prev, p_cur, attention_mask, two_h, a, module, param_keys, buffer_keys, *flat_tensors):
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

        # Save what we truly need for backward
        ctx.save_for_backward(p_prev, p_cur, *flat_tensors[:n_params])
        ctx.two_h = float(two_h)
        ctx.a = float(a)
        ctx.module = module
        ctx.param_keys = param_keys
        ctx.buffer_keys = buffer_keys
        ctx.n_params = n_params
        ctx.attention_mask = attention_mask

        with torch.no_grad():
            delta, aux = functional_call(module, (params, buffers), (p_cur, attention_mask), tie_weights=True)
            p_next = (ctx.a * p_prev) + ((1.0 - ctx.a) * p_cur) + (ctx.two_h * delta)

        return p_next, aux

    @staticmethod
    def backward(ctx, grad_p_next, grad_aux):
        p_prev, p_cur, *param_tensors = ctx.saved_tensors
        n_params = ctx.n_params

        # Rebuild params/buffers for functional_call
        params = {f"layer.{k}": v for k, v in zip(ctx.param_keys, param_tensors)}
        # buffers are non-diff; we still need them for correct forward recompute
        # they come from the original module at runtime via named_buffers
        # so we recreate them here from the live module's buffers:
        live_buffers = dict(ctx.module.named_buffers())
        buffers = {f"layer.{k}": live_buffers.get(f"layer.{k}", None) for k in ctx.buffer_keys}
        # Remove None entries (some layers may have no buffers)
        buffers = {k: v for k, v in buffers.items() if v is not None}

        # Direct paths:
        # p_next = a*p_prev + (1-a)*p_cur + two_h*delta(p_cur)
        grad_p_prev = grad_p_next * ctx.a
        grad_p_cur_direct = grad_p_next * (1.0 - ctx.a)

        with torch.enable_grad():
            p_cur_req = p_cur.detach().requires_grad_(True)

            # Need param tensors to require_grad for autograd.grad to produce param grads
            param_req = [t.detach().requires_grad_(True) for t in param_tensors]
            params_req = {f"layer.{k}": v for k, v in zip(ctx.param_keys, param_req)}

            delta, aux = functional_call(
                ctx.module,
                (params_req, buffers),
                (p_cur_req, ctx.attention_mask),
                tie_weights=True,
            )

            if grad_aux is None:
                # aux may be scalar or tensor
                grad_aux = torch.zeros_like(aux)

            grad_delta = grad_p_next * ctx.two_h

            # Only include aux in autograd.grad if it has a grad_fn
            # (dense FFN returns constant 0.0 with no grad_fn)
            if aux.grad_fn is not None:
                grads = torch.autograd.grad(
                    outputs=(delta, aux),
                    inputs=(p_cur_req, *param_req),
                    grad_outputs=(grad_delta.to(delta.dtype), grad_aux.to(aux.dtype)),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
            else:
                grads = torch.autograd.grad(
                    outputs=(delta,),
                    inputs=(p_cur_req, *param_req),
                    grad_outputs=(grad_delta.to(delta.dtype),),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )

        grad_p_cur_through_f = grads[0] if grads[0] is not None else torch.zeros_like(p_cur)
        grad_p_cur = grad_p_cur_direct + grad_p_cur_through_f

        grad_params = grads[1:]
        grad_params = [g if g is not None else torch.zeros_like(t) for g, t in zip(grad_params, param_tensors)]

        # Return grads for (p_prev, p_cur, attention_mask, two_h, a, module, param_keys, buffer_keys, *flat_tensors)
        # Non-tensor args -> None
        grad_attention_mask = None
        grad_two_h = None
        grad_a = None
        grad_module = None
        grad_param_keys = None
        grad_buffer_keys = None

        # buffers are non-diff
        grad_buffers = (None,) * len(ctx.buffer_keys)

        return (
            grad_p_prev,
            grad_p_cur,
            grad_attention_mask,
            grad_two_h,
            grad_a,
            grad_module,
            grad_param_keys,
            grad_buffer_keys,
            *grad_params,
            *grad_buffers,
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

    def forward(self, p_prev, p_cur, attention_mask=None):
        param_values = [p for p in self.block.parameters()]
        buffer_values = [b for b in self.block.buffers()]
        return MidpointFunction.apply(
            p_prev,
            p_cur,
            attention_mask,
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
        self.mid_layers = nn.ModuleList([MidpointBlock(b, step_size=self.h, a=self.a) for b in blocks[1:]])

        self.step_count = 0

    def forward(self, x, attention_mask=None):
        # Bootstrap creates two states (p_prev, p_cur)
        p_prev = x

        if self.bootstrap == "no_kick":
            # Baseline-aligned start: p_cur = p_prev (no Euler kick)
            p_cur = p_prev
            if attention_mask is None:
                delta0, aux0 = grad_checkpoint(
                    self.bootstrap_layer.force, p_cur, use_reentrant=False
                )
            else:
                delta0, aux0 = grad_checkpoint(
                    self.bootstrap_layer.force, p_cur, attention_mask, use_reentrant=False
                )
        else:
            # HALF-STEP Euler bootstrap (paper-consistent + stable for h=0.25, a=0.5)
            if attention_mask is None:
                delta0, aux0 = grad_checkpoint(
                    self.bootstrap_layer.force, p_prev, use_reentrant=False
                )
            else:
                delta0, aux0 = grad_checkpoint(
                    self.bootstrap_layer.force, p_prev, attention_mask, use_reentrant=False
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
