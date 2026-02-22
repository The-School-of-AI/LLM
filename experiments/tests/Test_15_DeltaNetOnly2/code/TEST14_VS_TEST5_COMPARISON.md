# Test 14 vs Test 5 (base) — Comparison

Test 5 was the base code (no fused kernels in the original sense; both now use fused paths where available). This doc summarizes parameter alignment, deviations, risks, and anything that could have been missed.

---

## 1. ModelConfig — Parameter alignment

### Matching (identical values)

| Parameter | Test 5 | Test 14 |
|-----------|--------|---------|
| vocab_size | 131072 | 131072 |
| hidden_size | 4096 | 4096 |
| num_layers | 8 | 8 |
| num_deltanet_layers | 6 | 6 |
| num_gsa_layers | 2 | 2 |
| delta_v_heads | 32 | 32 |
| delta_head_dim | 128 | 128 |
| delta_gate_dim | 384 | 384 |
| gsa_num_heads | 16 | 16 |
| gsa_head_dim | 256 | 256 |
| gsa_k_base | 512 | 512 |
| gsa_k_min | 32 | 32 |
| gsa_k_max | 1024 | 1024 |
| gsa_indexer_heads | 4 | 4 |
| enable_mtp | True | True |
| mtp_num_predictions | 2 | 2 |
| n_streams | 4 | 4 |
| sinkhorn_iters | 20 | 20 |
| max_seq_len | 262144 | 262144 |
| rope_base | 10000 | 10000 |
| rope_original_max_position | 8192 | 8192 |
| rope_scaling_factor | 32.0 | 32.0 |
| dropout | 0.0 | 0.0 |
| require_fused_deltanet_kernel | True | True |
| require_fused_gsa_kernel | True | True |
| num_real_experts | 0 | 0 *(added for parity)* |
| num_null_experts | 0 | 0 *(added for parity)* |
| total_expert_slots | 0 | 0 *(added for parity)* |
| top_k | 0 | 0 *(added for parity)* |
| expert_intermediate_size | 1024 | 1024 *(added for parity)* |
| shared_expert_intermediate_size | 2048 | 2048 |
| data_sparsity | 0.0 | 0.0 *(added for parity)* |

All config numbers are now aligned. The MoE-related attributes were added to Test 14 so that any script or `__main__` that reads them (e.g. weight_calculator, future MoE branch) does not see deviations or AttributeErrors.

---

## 2. Architectural / behavioral differences

| Aspect | Test 5 | Test 14 |
|--------|--------|---------|
| **Layer pattern** | DDDGDDDG (6 DeltaNet, 2 GSA) | DDDGDDDG (same) |
| **DeltaNet backend** | fla `chunk_gated_delta_rule` (with Python fallback that *raises* in T5) | fla `chunk_gated_delta_rule` only (no Python fallback) |
| **GSA backend** | Triton sparse attn + fused indexer (when available) | Same (Test 14 Triton kernels) |
| **MLP** | LightningMLP with full MoE/dense args (DenseMLP when num_experts=0) | LightningMLP(config) → DenseMLP only (same effective dense path) |
| **RoPE** | Standard RoPE (YaRN params in signature but not applied) | Standard RoPE (same) |
| **Fused CE** | Has LigerFusedLinearCrossEntropyLoss in liger_ops; train can use use_fused_ce | No fused CE: liger_ops has no CE class; train always uses logits + F.cross_entropy |
| **Memory recurrence** | recurrence_stream_idx=3, lambda_r_raw, memory_ln, memory_gate_proj | Same |

No parameter deviations; only implementation details differ (kernels, no fused CE in T14).

---

## 3. Risks and things to watch

1. **DeltaNet dependency (fla)**  
   Both Test 5 and Test 14 require `fla` for DeltaNet when `require_fused_deltanet_kernel=True`. Test 5’s `_delta_rule_python` is a stub that raises; there is no real Python fallback in either. So Test 14 is not stricter than Test 5 here. **Risk:** If `fla` is not installed, both fail at runtime.

2. **weight_calculator import**  
   Both use `from weight_calculator import LightningCalculator, LightningConfig` in `if __name__ == "__main__"`. That module is not under the test codebase; if it’s missing from the path, `python recurrence_model_1b.py` fails in both. **Risk:** Same for T5 and T14.

3. **Checkpoint compatibility**  
   Test 14 has extra parameters (DeltaNet layers, fla-based DeltaNet) and no fused CE. Checkpoints from Test 5 (same DDDGDDDG + dense) should match in layer layout and size; loading T5 → T14 or T14 → T5 may still need a strict state_dict key match and same config. **Risk:** Validate load/save when reusing checkpoints across tests.

4. **Numerics**  
   Test 14 uses the same RoPE, same alpha/beta and L2 norm in DeltaNet, and same GSA indexer/attention design. Small differences can still come from Triton vs PyTorch paths (e.g. GSA) or bf16/fp32 usage. **Risk:** If you need bit‑identical reproducibility to Test 5, run short parity checks (fixed seed, same inputs).

---

## 4. What was in Test 5 and is now in Test 14

- **ModelConfig:** All Test 5 config attributes are present in Test 14, including MoE-related (num_real_experts, top_k, data_sparsity, etc.) for parity and to avoid AttributeErrors.
- **DDDGDDDG:** Same 8-layer pattern (every 4th layer GSA).
- **Memory stream recurrence:** Same (stream 3, lambda_r_raw, memory_ln, memory_gate_proj).
- **MTP block:** Same (GSA + MLP, no DeltaNet in MTP).
- **Reversible midpoint stack:** Same (step_size=0.25, a=0.5, etc.).
- **Kronecker embeddings:** Same (PureHybridEmbeddingTorch, pf_to_model, embed_norm).
- **Forward contract:** Same (logits_ntp, logits_mtp, aux_loss); Test 14 never returns fused-CE scalar losses.

---

## 5. Intended differences (Test 14 vs Test 5)

- **No fused CE:** Test 14 has no `LigerFusedLinearCrossEntropyLoss`; training always uses logits and `F.cross_entropy` in train.py. Test 5 can use fused CE when enabled.
- **Kernels:** Test 14 uses its own Triton GSA/indexer/sinkhorn and fla_deltanet wrapper; Test 5 uses its own kernel set. Same *roles*, different *implementations* under the same config.
- **Liger ops:** Test 14 liger_ops only has SwiGLU + RoPE helpers (no CE). Test 5 liger_ops includes the fused CE class.

---

## 6. Fix applied during comparison

- **Duplicate/wrong print in Test 14:** Removed the extra line that said “GSA … (100%)” in the model init summary so it only prints DeltaNet % and GSA % once and correctly.
- **Missing ModelConfig in Test 14:** Added `num_real_experts`, `num_null_experts`, `total_expert_slots`, `top_k`, `expert_intermediate_size`, and `data_sparsity` so Test 14 matches Test 5 and scripts that touch these do not break.

No other parameter deviations or missing pieces were found; the two tests are aligned for the same “base” behavior aside from the intended no–fused-CE and kernel choices in Test 14.
