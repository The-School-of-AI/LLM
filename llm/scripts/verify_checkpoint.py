import torch

ckpt_path = "/Users/yash/Downloads/mp_rank_00_model_states.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

print("=" * 50)
print("CHECKPOINT INSPECTION REPORT")
print("=" * 50)

# 1. Step number
step = ckpt.get("global_steps") or ckpt.get("iteration") or ckpt.get("step")
print(f"\n{'✅' if step else '❌'} Step Number: {step if step else 'Not found'}")

# 2. Model weights
model_state = ckpt.get("module") or ckpt.get("model") or ckpt.get("model_state_dict")
if model_state:
    nan_layers = [k for k, v in model_state.items() if torch.is_tensor(v) and v.is_floating_point() and torch.isnan(v).any()]
    inf_layers = [k for k, v in model_state.items() if torch.is_tensor(v) and v.is_floating_point() and torch.isinf(v).any()]
    total_params = sum(v.numel() for v in model_state.values() if torch.is_tensor(v))
    print(f"\n✅ Model Weights: {len(model_state)} tensors | {total_params/1e6:.1f}M params")
    print(f"   NaN layers : {nan_layers if nan_layers else 'None ✅'}")
    print(f"   Inf layers : {inf_layers if inf_layers else 'None ✅'}")
else:
    print(f"\n❌ Model Weights: Not found")

# 3. Optimizer state
opt = ckpt.get("optimizer")
if opt:
    print(f"\n✅ Optimizer State: found")
    for i, pg in enumerate(opt.get("param_groups", [])):
        print(f"   Group {i}: lr={pg.get('lr')}, weight_decay={pg.get('weight_decay')}")
else:
    print(f"\n❌ Optimizer State: Not found")

# 4. Scheduler state
sched = ckpt.get("lr_scheduler")
if sched:
    print(f"\n✅ Scheduler State: found")
    print(f"   Last epoch : {sched.get('last_epoch')}")
    print(f"   Last LR    : {sched.get('_last_lr')}")
else:
    print(f"\n❌ Scheduler State: Not found")

# 5. RNG state
rng_keys = ["random_rng_state", "np_rng_state", "torch_rng_state", "cuda_rng_state", "rng_state"]
found_rng = {k: k in ckpt for k in rng_keys}
if any(found_rng.values()):
    print(f"\n✅ RNG State: {[k for k, v in found_rng.items() if v]}")
else:
    print(f"\n❌ RNG State: Not found")

# Summary
print("\n" + "=" * 50)
print("RAW TOP-LEVEL KEYS:", list(ckpt.keys()))
print("=" * 50)