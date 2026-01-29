# HALT System

This is a minimal, file-based cost governor and HALT controller.

## Files
- `governor.py`: reads metrics JSONL + budget YAML, decides whether to halt
- `budget.yaml`: per-stage budget and trigger thresholds
- `metrics.jsonl`: rolling metrics stream (written by training loop)
- `incidents/`: incident reports written on HALT
- `checkpoints/`: target dir for HALT checkpoints

## How It Works
- `governor.py` tails `metrics.jsonl` and evaluates triggers
- On trigger: write incident report, emit `HALT` file, and exit
- Training loop should check for `HALT` file every step and checkpoint

## Required Metrics Keys
- ts (ISO timestamp)
- stage (1-4)
- step
- tokens
- tokens_per_s
- data_wait_s
- step_time_s
- loss
- nan (bool) optional

## Example Run
```
python governor.py --metrics metrics.jsonl --budget budget.yaml
```

## HALT File Location
By default, `HALT` file is written to this directory. Training loop must read it.
