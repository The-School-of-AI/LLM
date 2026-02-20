# Test 11 Fused-MoE Run (Expected vs Got)

## Expected
- 1000-step run completes without NaN/inf.
- Per-step logging is present (`log_interval: 1`).
- Init metadata confirms `DDDGDDDG`, MoE `20/20, top_k=2`, and fused MoE backend=`grouped_gemm`.

## Got
- init checkpoint sha256: TODO
- final step line: TODO
- final loss: TODO
- final loss2: TODO
- final r_loss: TODO
- avg tok/sec: TODO

## Gate
- PASS / FAIL: TODO
- rationale: TODO
