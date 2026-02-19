cd /Users/rohanshravan/TSAI/ERAV4/LLM/experiments/9_training_stack_optimisation_and_cost_governor/training/deepspeed_template/dense_hardened

python scripts/benchmark_deltanet.py \
  --seq-lengths 4096,8192,16384 \
  --batch-size 1 \
  --dtype bf16 \
  --backward \
  --json-out ./logs/bench_deltanet.json

python scripts/benchmark_gsa.py \
  --seq-lengths 4096,8192,16384 \
  --batch-size 1 \
  --dtype bf16 \
  --backward \
  --json-out ./logs/bench_gsa.json
