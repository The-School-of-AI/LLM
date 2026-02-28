# T3 Coreset Engineering — Production Approval Summary  
**Decision: GO**

---

## 1. Execution Scope (Validated)

**Source:** C4 (post-EMR exact dedup)  
**Stages:** 1B → 3B → 8B → 70B  
**Architecture:** Streaming, sharded, batch processing  
**Determinism:** Fixed seed + checkpointed RNG state  
**Fault tolerance:** Checkpoint/resume tested under ~20 Spot interruptions  

---

## 2. Input / Output Accounting (C4 Run)

| Metric | Value |
|--------|--------|
| Input tokens (post-dedup) | **136,932,109,554** |
| Selected tokens (all stages) | **81,560,691,927** |
| Input chunks | 896,870,901 |
| Selected chunks | **112,244,531** |
| Chunk reduction | **87.5%** |
| Single-pass compression | **1.68×** |
| Stage-exposure compression | **5.62×** |

Stage-wise token accounting reconciles exactly (cumulative exposure: 458.7B).  
No arithmetic inconsistencies detected.

**Note**: C4 tests (~23GB curriculum metadata, no raw text) done on Macbook-Pro M4   
Shards: 14, Batch size: 80K, Runtime: 95 mins   

---

## 3. Engineering Validation Results

### Determinism
- Seed fixed (42)
- Config + curriculum hashes embedded in manifest
- RNG state serialized in checkpoints
- Resume compatibility guards enforced (shard_id, shard_count, stage_target)

**Result:** Bitwise-stable restarts verified.

---

### Non-Overlap Guarantees
- Persistent SQLite used-chunk store per shard
- Cross-stage exclusion enforced
- Stage overlap verification tool available

**Result:** No cross-stage duplication observed.

---

### Fault Tolerance
- ~20 Spot interruptions during test runs
- All restarts resumed correctly from checkpoints
- No token leakage or duplication observed

**Result:** Operationally resilient under interruption.

---

## 4. Curriculum & Band Behavior

C4 is web-only (domain=web, language=en).

Curriculum policy does not allow `web` in B4/B5. Therefore:

- B4/B5 = 0% in C4 output (expected, not pipeline failure)
- B3 limited by availability
- Redistribution occurred only within eligible bands

Pipeline behavior matches curriculum constraints.

**Note**: NCERT small-dataset validation executed on EC2 to test B4/B5 bands

---

## 5. Deduplication

- Exact dedup performed upstream (EMR)
- In-pipeline batch-level dedup active
- Near-dedup intentionally excluded (scale constraint)

Chunk reduction (87.5%) primarily driven by budget-constrained selection.

---

## 6. Tooling & Controls

| Control | Status |
|----------|--------|
| Output validation tool | ✅ |
| Stage overlap checker | ✅ |
| Determinism checker | ✅ |
| Infra validation script | ✅ |
| Deployment automation | ✅ |
| Resume documentation | ✅ |

Production scripts: `commands.sh`, `monitor.sh`, `validate_coreset_outputs.py`.

---

## 7. Known Limitations (Non-Blocking)

1. C4-only validation (with smaller multi-source validation with Ncert)
2. Near-dedup deferred
3. Convergence/benchmark comparison owned by training team
4. On-demand infra provisioning pending approval

None invalidate engineering correctness.

---

## 8. Risk Profile

| Risk | Status |
|-------|--------|
| Determinism drift | Mitigated |
| Token accounting error | Mitigated |
| Data overlap | Mitigated |
| Infra instability (Spot) | Avoidable via on-demand |
| Learning-quality validation | Pending (training team) |

Overall risk: **Controlled / Acceptable for production.**

---

# Final Recommendation

The T3 Coreset Engineering pipeline:

- Is numerically correct  
- Is architecturally sound  
- Is deterministically reproducible  
- Has validated fault tolerance  
- Meets its chartered scope  

**Recommendation: Proceed with production execution (GO).**