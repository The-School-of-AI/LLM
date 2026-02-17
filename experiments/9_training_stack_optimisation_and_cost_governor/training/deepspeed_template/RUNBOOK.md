# AWS Training Runbook

Quick-reference commands for running training after SSHing into the AWS instance.

---

## 1. SSH into the Instance

```bash
ssh -i ~/.ssh/your-key.pem ubuntu@<instance-ip>
```

## 2. Activate Environment

```bash
cd /home/ubuntu/LLM
source venv/bin/activate
```

## 3. Pull Latest Code

```bash
git fetch origin
git checkout p9/feat/dense-hardening
git pull origin p9/feat/dense-hardening
```

## 4. Install/Update Dependencies

```bash
pip install -r experiments/9_training_stack_optimisation_and_cost_governor/training/deepspeed_template/requirements.txt
```

## 5. Navigate to Training Directory

```bash
cd experiments/9_training_stack_optimisation_and_cost_governor/training/deepspeed_template
```

## 6. Run Training

### Quick Test (20 steps, with profiling)

`config_profile.yaml` ships with `max_train_steps: 20` — good for a quick smoke test:

```bash
deepspeed main.py --config config_profile.yaml 2>&1 | tee training.log
```

### Full Training (all GPUs)

Edit `config_profile.yaml` first — set `max_train_steps: null` and `max_eval_steps: null`, then:

```bash
deepspeed main.py --config config_profile.yaml 2>&1 | tee training_full.log
```

### Specify GPU Count
```bash
deepspeed --num_gpus=8 main.py --config config_profile.yaml 2>&1 | tee training.log
```

> The `2>&1 | tee training.log` captures both stdout and stderr to `training.log` while still showing output in the terminal.

## 7. Monitor Training (from another terminal)

### Watch GPU utilization
```bash
watch -n 1 nvidia-smi
```

### Tail the training log
```bash
tail -f training.log
```

### Check structured metrics
```bash
tail -f logs/metrics.jsonl | python -m json.tool --no-ensure-ascii
```

## 8. After Training

### View final metrics
```bash
cat logs/metrics.jsonl | tail -5
```

### Check disk usage
```bash
df -h
du -sh checkpoints/
```

---

## Config Quick Reference

| Setting | File | What it does |
|---|---|---|
| `chunked_ce_loss_size: 1024` | `config_profile.yaml` | Memory-efficient CE (saves ~3-15GB) |
| `require_fused_kernels: true` | `config_profile.yaml` | Crash if Triton/FLA missing |
| `enable_system_metrics: true` | `config_profile.yaml` | GPU util/memory in terminal |
| `metrics_jsonl_path: "./logs/metrics.jsonl"` | `config_profile.yaml` | Structured metrics to file |
| `max_train_steps: 20` | `config_profile.yaml` | Quick test (set `null` for full) |

---

## Troubleshooting

### OOM (Out of Memory)
```bash
# Reduce sequence length in config_profile.yaml
# data.max_length: 2048  (down from 4096)

# Or enable chunked CE
# training.chunked_ce_loss_size: 1024
```

### Training hangs
```bash
# Kill all python processes
pkill -f deepspeed
pkill -f python

# Check if GPUs are still occupied
nvidia-smi
```

### Permission denied on SSH key
```bash
chmod 400 ~/.ssh/your-key.pem
```
