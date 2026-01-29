# Team 9 Tests: Training Stack Optimization & Cost Governor

This folder lists verification checks for Team 9 deliverables.

## Required Checks
- Training configs validate with DeepSpeed parser
- Metrics schema emits tokens/sec and data-wait time
- Cost-per-token drift calculation matches budget envelope
- HALT triggers: NaN, throughput collapse, starvation, cost drift
- HALT action: checkpoint written + training stop + incident report

## Evidence Artifacts
- Metrics JSONL samples
- Cost governor incident logs
- Checkpoint directories written on forced halt
