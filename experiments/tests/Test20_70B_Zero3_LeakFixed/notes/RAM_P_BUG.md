Summary: Root Cause Found ✅
Here's exactly what happened:

The Bug
In MidpointFunctionZero3.backward, line 233:

python
params = [p for p in ctx.module.layer.parameters() if p.requires_grad]
with _zero3_gather_ctx(params, fwd_module=ctx.module.layer):   # ← PROBLEM
    delta, aux = ctx.module(p_cur_req, ...)
    torch.autograd.backward(...)   # ← runs INSIDE the gather context
When torch.autograd.backward() runs inside the GatheredParameters context, the backward pass through sub-layers (RMSNorm, mHC Sinkhorn, etc.) triggers ZeRO-3's own gather hooks for those nested submodules. ZeRO-3 then tries to free those params at the end of the context, but they still show as active_sub_modules: {262} from the outer manual context — causing both the "Cannot free" error AND a per-step memory accumulation, because ZeRO-3 can't complete its cleanup.

The Fix (already applied to 

reversible_ops_midpoint.py
)
Remove 

_zero3_gather_ctx
 entirely from backward. ZeRO-3's standard module hooks handle all gathering automatically during torch.autograd.backward(). The manual gather in backward was redundant AND destructive.

The fix is live in your local 

reversible_ops_midpoint.py
. You can now commit and test on the actual training server.


BUG ON AWS IS BACK
Now the version mismatch. Here's what GCP had vs P4de:

Library	|GCP (verified working)	|P4de (leaking)
---------|-------------------------|----------------
PyTorch	|2.7.1+cu128	|2.10.0+cu128
Triton	|3.3.1	|3.6.0
FLA	|0.4.2 @ 2f18f7d	|0.4.2 @ d792da6
DeepSpeed	|0.18.6	|0.18.6
Python	|3.10.12	|3.12.12
