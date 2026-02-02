# Experiment 11: Growth & Weight Transfer

This directory contains multiple experiments exploring different approaches to model growth and weight transfer.

## Experiments

| Folder | Approach | Description |
|--------|----------|-------------|
| `01-5phase-growth-pipeline/` | ADD HEADS + Ghost Layers + YaRN | 5-phase growth with RoPE-safe dimension scaling |
| `02-gstack-bilateral/` | GStack + Bilateral Growth | Depth stacking + weight tiling (TBD) |

---

## 01-5phase-growth-pipeline

Our primary implementation with **5 function-preserving growth phases**:

1. **Phase 1**: Dense model training (~70M params)
2. **Phase 2**: Dense → MoE conversion (+0.25 spike)
3. **Phase 3**: +Layers (ghost init) + Scale Dim (ADD HEADS) (+0.07 spike)
4. **Phase 4**: +Experts (4→8) (+0.13 spike)
5. **Phase 5**: YaRN Context Extension (256→1024) (+0.37 spike)

**Key Discovery**: Can't change `head_dim` when scaling - must ADD HEADS to preserve RoPE.

**Run it**:
```bash
cd 01-5phase-growth-pipeline
pip install torch pyyaml datasets
python run_growth_experiment.py
```

---

## References

- [Issue #229](https://github.com/The-School-of-AI/LLM/issues/229)
- [YaRN Paper](https://arxiv.org/abs/2309.00071)
- [Net2Net Paper](https://arxiv.org/abs/1511.05641)
- [GStack Paper](https://arxiv.org/abs/2405.15319)
