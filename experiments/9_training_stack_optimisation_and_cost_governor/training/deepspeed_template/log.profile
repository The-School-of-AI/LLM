/home/ubuntu/LLM/experiments/9_training_stack_optimisation_and_cost_governor/training/deepspeed_template/.venv/lib/python3.12/site-packages/torch/cuda/__init__.py:65: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
[2026-02-10 17:35:25,525] [WARNING] [runner.py:232:fetch_hostfile] Unable to find hostfile, will proceed with training with local resources only.
[2026-02-10 17:35:25,526] [INFO] [runner.py:630:main] cmd = /home/ubuntu/LLM/experiments/9_training_stack_optimisation_and_cost_governor/training/deepspeed_template/.venv/bin/python -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMCwgMSwgMiwgM119 --master_addr=127.0.0.1 --master_port=29500 --enable_each_rank_log=None --log_level=info main.py
/home/ubuntu/LLM/experiments/9_training_stack_optimisation_and_cost_governor/training/deepspeed_template/.venv/lib/python3.12/site-packages/torch/cuda/__init__.py:65: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
[2026-02-10 17:35:31,096] [INFO] [launch.py:162:main] WORLD INFO DICT: {'localhost': [0, 1, 2, 3]}
[2026-02-10 17:35:31,096] [INFO] [launch.py:168:main] nnodes=1, num_local_procs=4, node_rank=0
[2026-02-10 17:35:31,096] [INFO] [launch.py:179:main] global_rank_mapping=defaultdict(<class 'list'>, {'localhost': [0, 1, 2, 3]})
[2026-02-10 17:35:31,096] [INFO] [launch.py:180:main] dist_world_size=4
[2026-02-10 17:35:31,096] [INFO] [launch.py:184:main] Setting CUDA_VISIBLE_DEVICES=0,1,2,3
[2026-02-10 17:35:31,097] [INFO] [launch.py:272:main] process 46738 spawned with command: ['/home/ubuntu/LLM/experiments/9_training_stack_optimisation_and_cost_governor/training/deepspeed_template/.venv/bin/python', '-u', 'main.py', '--local_rank=0']
[2026-02-10 17:35:31,097] [INFO] [launch.py:272:main] process 46739 spawned with command: ['/home/ubuntu/LLM/experiments/9_training_stack_optimisation_and_cost_governor/training/deepspeed_template/.venv/bin/python', '-u', 'main.py', '--local_rank=1']
[2026-02-10 17:35:31,097] [INFO] [launch.py:272:main] process 46740 spawned with command: ['/home/ubuntu/LLM/experiments/9_training_stack_optimisation_and_cost_governor/training/deepspeed_template/.venv/bin/python', '-u', 'main.py', '--local_rank=2']
[2026-02-10 17:35:31,098] [INFO] [launch.py:272:main] process 46741 spawned with command: ['/home/ubuntu/LLM/experiments/9_training_stack_optimisation_and_cost_governor/training/deepspeed_template/.venv/bin/python', '-u', 'main.py', '--local_rank=3']
/home/ubuntu/LLM/experiments/9_training_stack_optimisation_and_cost_governor/training/deepspeed_template/.venv/lib/python3.12/site-packages/torch/cuda/__init__.py:65: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
/home/ubuntu/LLM/experiments/9_training_stack_optimisation_and_cost_governor/training/deepspeed_template/.venv/lib/python3.12/site-packages/torch/cuda/__init__.py:65: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
/home/ubuntu/LLM/experiments/9_training_stack_optimisation_and_cost_governor/training/deepspeed_template/.venv/lib/python3.12/site-packages/torch/cuda/__init__.py:65: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
/home/ubuntu/LLM/experiments/9_training_stack_optimisation_and_cost_governor/training/deepspeed_template/.venv/lib/python3.12/site-packages/torch/cuda/__init__.py:65: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
================================================================================
DeepSpeed Training Template
================================================================================
Configuration File: config.yaml
DeepSpeed Version: 0.18.5
PyTorch Version: 2.10.0+cu128
CUDA Available: True
CUDA Devices: 4

Configuration:
  Dataset: wikitext/wikitext-2-raw-v1
  DeepSpeed Config: deepspeed/zero-2-moe.json
  Batch Size: 32
  Max Length: 128
  Epochs: 1
  Checkpoint Interval: Every 100 steps
  Output Directory: ./checkpoints
  Random Seed: 42
================================================================================

[1/5] Loading data...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
================================================================================
DeepSpeed Training Template
================================================================================
Configuration File: config.yaml
DeepSpeed Version: 0.18.5
PyTorch Version: 2.10.0+cu128
CUDA Available: True
CUDA Devices: 4

Configuration:
  Dataset: wikitext/wikitext-2-raw-v1
  DeepSpeed Config: deepspeed/zero-2-moe.json
  Batch Size: 32
  Max Length: 128
  Epochs: 1
  Checkpoint Interval: Every 100 steps
  Output Directory: ./checkpoints
  Random Seed: 42
================================================================================

[1/5] Loading data...
================================================================================
DeepSpeed Training Template
================================================================================
Configuration File: config.yaml
DeepSpeed Version: 0.18.5
PyTorch Version: 2.10.0+cu128
CUDA Available: True
CUDA Devices: 4

Configuration:
  Dataset: wikitext/wikitext-2-raw-v1
  DeepSpeed Config: deepspeed/zero-2-moe.json
  Batch Size: 32
================================================================================  Max Length: 128

  Epochs: 1
DeepSpeed Training Template  Checkpoint Interval: Every 100 steps

================================================================================  Output Directory: ./checkpoints

  Random Seed: 42Configuration File: config.yaml

================================================================================
DeepSpeed Version: 0.18.5

[1/5] Loading data...
PyTorch Version: 2.10.0+cu128
CUDA Available: True
CUDA Devices: 4

Configuration:
  Dataset: wikitext/wikitext-2-raw-v1
  DeepSpeed Config: deepspeed/zero-2-moe.json
  Batch Size: 32
  Max Length: 128
  Epochs: 1
  Checkpoint Interval: Every 100 steps
  Output Directory: ./checkpoints
  Random Seed: 42
================================================================================

[1/5] Loading data...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading dataset: wikitext (wikitext-2-raw-v1)
Loading dataset: wikitext (wikitext-2-raw-v1)
Loading dataset: wikitext (wikitext-2-raw-v1)
Loading dataset: wikitext (wikitext-2-raw-v1)
Tokenizing dataset...
Tokenizing dataset...
Tokenizing dataset...
Tokenizing dataset...
  Train batches: 743
  Eval batches: 77
  Test batches: 91

[2/5] Loading model...
  Creating Qwen2 MoE model from scratch...
  Train batches: 743
  Eval batches: 77
  Test batches: 91

[2/5] Loading model...
  Creating Qwen2 MoE model from scratch...
  Train batches: 743
  Eval batches: 77
  Test batches: 91

[2/5] Loading model...
  Creating Qwen2 MoE model from scratch...
  Train batches: 743
  Eval batches: 77
  Test batches: 91

[2/5] Loading model...
  Creating Qwen2 MoE model from scratch...
  Model created: Qwen2 MoE
  Configuration:
    - Hidden size: 512
    - Layers: 12
    - Attention heads: 8
    - KV heads: 2
    - MoE experts: 8
    - Active experts per token: 2
  Total parameters: 503,262,720
  Trainable parameters: 503,262,720
  Gradient checkpointing: Enabled

[3/5] Initializing DeepSpeed...
  Model created: Qwen2 MoE
  Configuration:
    - Hidden size: 512
    - Layers: 12
    - Attention heads: 8
    - KV heads: 2
    - MoE experts: 8
    - Active experts per token: 2
  Total parameters: 503,262,720
  Trainable parameters: 503,262,720
  Gradient checkpointing: Enabled

[3/5] Initializing DeepSpeed...
  Model created: Qwen2 MoE
  Configuration:
    - Hidden size: 512
    - Layers: 12
    - Attention heads: 8
    - KV heads: 2
    - MoE experts: 8
    - Active experts per token: 2
  Total parameters: 503,262,720
  Trainable parameters: 503,262,720
  Gradient checkpointing: Enabled

[3/5] Initializing DeepSpeed...
  Model created: Qwen2 MoE
  Configuration:
    - Hidden size: 512
    - Layers: 12
    - Attention heads: 8
    - KV heads: 2
    - MoE experts: 8
    - Active experts per token: 2
  Total parameters: 503,262,720
  Trainable parameters: 503,262,720
  Gradient checkpointing: Enabled

[3/5] Initializing DeepSpeed...
Installed CUDA version 12.0 does not match the version torch was compiled with 12.8 but since the APIs are compatible, accepting this combination
Installed CUDA version 12.0 does not match the version torch was compiled with 12.8 but since the APIs are compatible, accepting this combination
Installed CUDA version 12.0 does not match the version torch was compiled with 12.8 but since the APIs are compatible, accepting this combination
Installed CUDA version 12.0 does not match the version torch was compiled with 12.8 but since the APIs are compatible, accepting this combination
Before initializing optimizer states
MA 1.08 GB         Max_MA 1.17 GB         CA 1.17 GB         Max_CA 1 GB 
CPU Virtual Memory:  used = 16.53 GB, percent = 8.9%
After initializing optimizer states
MA 1.08 GB         Max_MA 1.08 GB         CA 1.17 GB         Max_CA 1 GB 
CPU Virtual Memory:  used = 20.53 GB, percent = 11.0%
[2026-02-10 17:35:57,961] [WARNING] [lr_schedules.py:690:get_lr] Attempting to get learning rate from scheduler before it has started
[2026-02-10 17:35:57,988] [WARNING] [lr_schedules.py:690:get_lr] Attempting to get learning rate from scheduler before it has started
[2026-02-10 17:35:57,991] [WARNING] [lr_schedules.py:690:get_lr] Attempting to get learning rate from scheduler before it has started
After initializing ZeRO optimizer
MA 1.08 GB         Max_MA 1.08 GB         CA 1.17 GB         Max_CA 1 GB 
CPU Virtual Memory:  used = 19.82 GB, percent = 10.6%
[2026-02-10 17:35:58,101] [WARNING] [lr_schedules.py:690:get_lr] Attempting to get learning rate from scheduler before it has started

[4/5] Training...
Checkpoint interval: Every 100 steps
Starting from epoch 0, global step 0

================================================================================
Epoch 1/1
================================================================================
Epoch 0:   0%|          | 0/743 [00:00<?, ?it/s][Rank 0] time (ms) | fwd_microstep: 1136.68 | bwd_microstep: 2443.19 | bwd_inner_microstep: 2265.61 | bwd_allreduce_microstep: 177.41 | step_microstep: 0.05
Epoch 0:   0%|          | 0/743 [00:03<?, ?it/s, loss=12.0024, global_step=1, toks/s=2480.0, gpu_util=99%, gpu_mem=10.5G, cpu_util=15%, cpu_mem=21.7G]Epoch 0, Step 0, Global Step 1, Loss: 12.0024, Tokens/s: 2480.0, Tokens: 8884, GPU Util: 99%, GPU Mem: 10.5G/15.0G, CPU Util: 15%, CPU Mem: 21.7G/186.7G

GPU Utilization / Memory (all devices):
GPU  Util(%)  Mem(GB Used/Total)
--------------------------------
0        99%   10.5G/ 15.0G
1        97%   10.2G/ 15.0G
2        99%   10.2G/ 15.0G
3        89%   10.2G/ 15.0G
Epoch 0:   0%|          | 1/743 [00:03<44:41,  3.61s/it, loss=12.0024, global_step=1, toks/s=2480.0, gpu_util=99%, gpu_mem=10.5G, cpu_util=15%, cpu_mem=21.7G][Rank 0] time (ms) | fwd_microstep: 654.24 | bwd_microstep: 1825.14 | bwd_inner_microstep: 1698.97 | bwd_allreduce_microstep: 126.12 | step_microstep: 0.04
Epoch 0:   0%|          | 1/743 [00:06<44:41,  3.61s/it, loss=11.9413, global_step=2, toks/s=3930.4, gpu_util=100%, gpu_mem=12.8G, cpu_util=9%, cpu_mem=21.4G]Epoch 0, Step 1, Global Step 2, Loss: 11.9413, Tokens/s: 3930.4, Tokens: 9748, GPU Util: 100%, GPU Mem: 12.8G/15.0G, CPU Util: 9%, CPU Mem: 21.4G/186.7G

GPU Utilization / Memory (all devices):
GPU  Util(%)  Mem(GB Used/Total)
--------------------------------
0       100%   12.8G/ 15.0G
1       100%   12.5G/ 15.0G
2       100%   12.5G/ 15.0G
3       100%   12.5G/ 15.0G
Epoch 0:   0%|          | 2/743 [00:06<36:27,  2.95s/it, loss=11.9413, global_step=2, toks/s=3930.4, gpu_util=100%, gpu_mem=12.8G, cpu_util=9%, cpu_mem=21.4G][Rank 0] time (ms) | fwd_microstep: 653.33 | bwd_microstep: 1828.33 | bwd_inner_microstep: 1702.43 | bwd_allreduce_microstep: 125.85 | step_microstep: 0.04
Epoch 0:   0%|          | 2/743 [00:08<36:27,  2.95s/it, loss=11.9925, global_step=3, toks/s=4052.5, gpu_util=100%, gpu_mem=12.8G, cpu_util=9%, cpu_mem=21.4G]Epoch 0, Step 2, Global Step 3, Loss: 11.9925, Tokens/s: 4052.5, Tokens: 10060, GPU Util: 100%, GPU Mem: 12.8G/15.0G, CPU Util: 9%, CPU Mem: 21.4G/186.7G

GPU Utilization / Memory (all devices):
GPU  Util(%)  Mem(GB Used/Total)
--------------------------------
0       100%   12.8G/ 15.0G
1       100%   12.5G/ 15.0G
2       100%   12.5G/ 15.0G
3        99%   12.5G/ 15.0G
Epoch 0:   0%|          | 3/743 [00:08<33:48,  2.74s/it, loss=11.9925, global_step=3, toks/s=4052.5, gpu_util=100%, gpu_mem=12.8G, cpu_util=9%, cpu_mem=21.4G][Rank 0] time (ms) | fwd_microstep: 654.82 | bwd_microstep: 1827.09 | bwd_inner_microstep: 1701.27 | bwd_allreduce_microstep: 125.77 | step_microstep: 0.04
Epoch 0:   0%|          | 3/743 [00:11<33:48,  2.74s/it, loss=11.9233, global_step=4, toks/s=3538.0, gpu_util=100%, gpu_mem=12.8G, cpu_util=9%, cpu_mem=21.3G]Epoch 0, Step 3, Global Step 4, Loss: 11.9233, Tokens/s: 3538.0, Tokens: 8784, GPU Util: 100%, GPU Mem: 12.8G/15.0G, CPU Util: 9%, CPU Mem: 21.3G/186.7G

GPU Utilization / Memory (all devices):
GPU  Util(%)  Mem(GB Used/Total)
--------------------------------
0       100%   12.8G/ 15.0G
1       100%   12.5G/ 15.0G
2       100%   12.5G/ 15.0G
3       100%   12.5G/ 15.0G
Epoch 0:   1%|          | 4/743 [00:11<32:32,  2.64s/it, loss=11.9233, global_step=4, toks/s=3538.0, gpu_util=100%, gpu_mem=12.8G, cpu_util=9%, cpu_mem=21.3G][Rank 0] time (ms) | fwd_microstep: 655.65 | bwd_microstep: 1832.33 | bwd_inner_microstep: 1706.43 | bwd_allreduce_microstep: 125.86 | step_microstep: 0.04
Epoch 0:   1%|          | 4/743 [00:13<32:32,  2.64s/it, loss=11.9457, global_step=5, toks/s=3763.9, gpu_util=100%, gpu_mem=12.8G, cpu_util=9%, cpu_mem=21.2G]Epoch 0, Step 4, Global Step 5, Loss: 11.9457, Tokens/s: 3763.9, Tokens: 9368, GPU Util: 100%, GPU Mem: 12.8G/15.0G, CPU Util: 9%, CPU Mem: 21.2G/186.7G

GPU Utilization / Memory (all devices):
GPU  Util(%)  Mem(GB Used/Total)
--------------------------------
0       100%   12.8G/ 15.0G
1       100%   12.5G/ 15.0G
2       100%   12.5G/ 15.0G
3       100%   12.5G/ 15.0G
Epoch 0:   1%|          | 5/743 [00:13<31:51,  2.59s/it, loss=11.9457, global_step=5, toks/s=3763.9, gpu_util=100%, gpu_mem=12.8G, cpu_util=9%, cpu_mem=21.2G][Rank 0] time (ms) | fwd_microstep: 658.43 | bwd_microstep: 1831.65 | bwd_inner_microstep: 1705.81 | bwd_allreduce_microstep: 125.79 | step_microstep: 0.04
Epoch 0:   1%|          | 5/743 [00:16<31:51,  2.59s/it, loss=11.9632, global_step=6, toks/s=3470.0, gpu_util=100%, gpu_mem=12.8G, cpu_util=9%, cpu_mem=21.1G]Epoch 0, Step 5, Global Step 6, Loss: 11.9632, Tokens/s: 3470.0, Tokens: 8644, GPU Util: 100%, GPU Mem: 12.8G/15.0G, CPU Util: 9%, CPU Mem: 21.1G/186.7G

GPU Utilization / Memory (all devices):
GPU  Util(%)  Mem(GB Used/Total)
--------------------------------
0       100%   12.8G/ 15.0G
1        99%   12.5G/ 15.0G
2       100%   12.5G/ 15.0G
3       100%   12.5G/ 15.0G
Epoch 0:   1%|          | 6/743 [00:16<31:26,  2.56s/it, loss=11.9632, global_step=6, toks/s=3470.0, gpu_util=100%, gpu_mem=12.8G, cpu_util=9%, cpu_mem=21.1G][Rank 0] time (ms) | fwd_microstep: 655.08 | bwd_microstep: 1830.74 | bwd_inner_microstep: 1704.87 | bwd_allreduce_microstep: 125.82 | step_microstep: 0.04
Epoch 0:   1%|          | 6/743 [00:18<31:26,  2.56s/it, loss=12.0042, global_step=7, toks/s=4013.1, gpu_util=100%, gpu_mem=12.8G, cpu_util=9%, cpu_mem=21.0G]Epoch 0, Step 6, Global Step 7, Loss: 12.0042, Tokens/s: 4013.1, Tokens: 9980, GPU Util: 100%, GPU Mem: 12.8G/15.0G, CPU Util: 9%, CPU Mem: 21.0G/186.7G

GPU Utilization / Memory (all devices):
GPU  Util(%)  Mem(GB Used/Total)
--------------------------------
0       100%   12.8G/ 15.0G
1       100%   12.5G/ 15.0G
2       100%   12.5G/ 15.0G
3       100%   12.5G/ 15.0G
Epoch 0:   1%|          | 7/743 [00:18<31:07,  2.54s/it, loss=12.0042, global_step=7, toks/s=4013.1, gpu_util=100%, gpu_mem=12.8G, cpu_util=9%, cpu_mem=21.0G][Rank 0] time (ms) | optimizer_allgather: 170.78 | optimizer_gradients: 48.65 | optimizer_step: 361.10
[Rank 0] time (ms) | fwd_microstep: 654.38 | bwd_microstep: 2688.62 | bwd_inner_microstep: 2544.27 | bwd_allreduce_microstep: 144.26 | step_microstep: 581.75
[Rank 0] time (ms) | fwd: 5722.49 | bwd: 16107.19 | bwd_inner: 15029.70 | bwd_allreduce: 1077.03 | step: 582.04
Epoch 0:   1%|          | 7/743 [00:22<31:07,  2.54s/it, loss=11.9652, global_step=8, toks/s=2651.6, gpu_util=95%, gpu_mem=13.8G, cpu_util=18%, cpu_mem=24.3G]Epoch 0, Step 7, Global Step 8, Loss: 11.9652, Tokens/s: 2651.6, Tokens: 10412, GPU Util: 95%, GPU Mem: 13.8G/15.0G, CPU Util: 18%, CPU Mem: 24.3G/186.7G

GPU Utilization / Memory (all devices):
GPU  Util(%)  Mem(GB Used/Total)
--------------------------------
0        95%   13.8G/ 15.0G
1        17%   12.7G/ 15.0G
2         6%   12.8G/ 15.0G
3        28%   13.5G/ 15.0G
Epoch 0:   1%|          | 8/743 [00:22<36:32,  2.98s/it, loss=11.9652, global_step=8, toks/s=2651.6, gpu_util=95%, gpu_mem=13.8G, cpu_util=18%, cpu_mem=24.3G][Rank 0] time (ms) | fwd_microstep: 655.10 | bwd_microstep: 1821.84 | bwd_inner_microstep: 1700.97 | bwd_allreduce_microstep: 120.83 | step_microstep: 0.04
Epoch 0:   1%|          | 8/743 [00:24<36:32,  2.98s/it, loss=11.8778, global_step=9, toks/s=3383.9, gpu_util=100%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=24.3G]Epoch 0, Step 8, Global Step 9, Loss: 11.8778, Tokens/s: 3383.9, Tokens: 8384, GPU Util: 100%, GPU Mem: 13.8G/15.0G, CPU Util: 9%, CPU Mem: 24.3G/186.7G

GPU Utilization / Memory (all devices):
GPU  Util(%)  Mem(GB Used/Total)
--------------------------------
0       100%   13.8G/ 15.0G
1       100%   12.7G/ 15.0G
2       100%   12.8G/ 15.0G
3        99%   13.5G/ 15.0G
Epoch 0:   1%|          | 9/743 [00:24<34:35,  2.83s/it, loss=11.8778, global_step=9, toks/s=3383.9, gpu_util=100%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=24.3G][Rank 0] time (ms) | fwd_microstep: 656.87 | bwd_microstep: 1825.10 | bwd_inner_microstep: 1698.73 | bwd_allreduce_microstep: 126.32 | step_microstep: 0.04
Profile started
Profile started
Profile started
Epoch 0:   1%|          | 9/743 [00:27<34:35,  2.83s/it, loss=11.9752, global_step=10, toks/s=3209.4, gpu_util=100%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=24.3G]Epoch 0, Step 9, Global Step 10, Loss: 11.9752, Tokens/s: 3209.4, Tokens: 7968, GPU Util: 100%, GPU Mem: 13.8G/15.0G, CPU Util: 9%, CPU Mem: 24.3G/186.7G

GPU Utilization / Memory (all devices):
GPU  Util(%)  Mem(GB Used/Total)
--------------------------------
0       100%   13.8G/ 15.0G
1       100%   12.7G/ 15.0G
2       100%   12.8G/ 15.0G
3       100%   13.5G/ 15.0G
Epoch 0:   1%|▏         | 10/743 [00:27<33:16,  2.72s/it, loss=11.9752, global_step=10, toks/s=3209.4, gpu_util=100%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=24.3G]Profile started
[Rank 0] time (ms) | fwd_microstep: 674.63 | bwd_microstep: 1930.12 | bwd_inner_microstep: 1793.70 | bwd_allreduce_microstep: 136.30 | step_microstep: 0.04
Profile stoped
Profile stoped
Profile stoped




-------------------------- DeepSpeed Flops Profiler --------------------------
Profile Summary at step 10:
Notations:
data parallel size (dp_size), model parallel size(mp_size),
number of parameters (params), number of multiply-accumulate operations(MACs),
number of floating-point operations (flops), floating-point operations per second (FLOPS),
fwd latency (forward propagation latency), bwd latency (backward propagation latency),
step (weights update latency), iter latency (sum of fwd, bwd and step latency)


-------------------------- DeepSpeed Flops Profiler --------------------------
Profile Summary at step 10:
params per GPU:                                                         503.26 M
Notations:
data parallel size (dp_size), model parallel size(mp_size),
number of parameters (params), number of multiply-accumulate operations(MACs),
number of floating-point operations (flops), floating-point operations per second (FLOPS),
fwd latency (forward propagation latency), bwd latency (backward propagation latency),
step (weights update latency), iter latency (sum of fwd, bwd and step latency)

params of model = params per GPU * mp_size:                             0       
fwd MACs per GPU:                                                       705.63 GMACs
fwd flops per GPU:                                                      1.41 T  params per GPU:                                                         503.26 M

fwd flops of model = fwd flops per GPU * mp_size:                       1.41 T  
params of model = params per GPU * mp_size:                             0       
fwd latency:                                                            653.07 ms

-------------------------- DeepSpeed Flops Profiler --------------------------fwd MACs per GPU:                                                       705.63 GMACs

fwd FLOPS per GPU = fwd flops per GPU / fwd latency:                    2.16 TFLOPS
Profile Summary at step 10:fwd flops per GPU:                                                      1.41 T  

Notations:
data parallel size (dp_size), model parallel size(mp_size),
number of parameters (params), number of multiply-accumulate operations(MACs),
number of floating-point operations (flops), floating-point operations per second (FLOPS),
fwd latency (forward propagation latency), bwd latency (backward propagation latency),
step (weights update latency), iter latency (sum of fwd, bwd and step latency)

fwd flops of model = fwd flops per GPU * mp_size:                       1.41 T  
fwd latency:                                                            670.23 ms
params per GPU:                                                         503.26 M
fwd FLOPS per GPU = fwd flops per GPU / fwd latency:                    2.11 TFLOPS
params of model = params per GPU * mp_size:                             0       
fwd MACs per GPU:                                                       705.63 GMACs
fwd flops per GPU:                                                      1.41 T  
fwd flops of model = fwd flops per GPU * mp_size:                       1.41 T  
fwd latency:                                                            667.98 ms
fwd FLOPS per GPU = fwd flops per GPU / fwd latency:                    2.11 TFLOPS

----------------------------- Aggregated Profile per GPU -----------------------------

----------------------------- Aggregated Profile per GPU -----------------------------

----------------------------- Aggregated Profile per GPU -----------------------------
Top 1 modules in terms of params, MACs or fwd latency at different model depths:
depth 0:
    params      - {'DeepSpeedEngine': '503.26 M'}
    MACs        - {'DeepSpeedEngine': '705.63 GMACs'}
    fwd latency - {'DeepSpeedEngine': '653.07 ms'}
depth 1:
    params      - {'Qwen2MoeForCausalLM': '503.26 M'}
    MACs        - {'Qwen2MoeForCausalLM': '705.63 GMACs'}
    fwd latency - {'Qwen2MoeForCausalLM': '653.07 ms'}
depth 2:
    params      - {'Qwen2MoeModel': '425.47 M'}
    MACs        - {'Qwen2MoeModel': '387 GMACs'}
Top 1 modules in terms of params, MACs or fwd latency at different model depths:    fwd latency - {'Qwen2MoeModel': '382.29 ms'}

depth 3:
    params      - {'ModuleList': '347.68 M'}
    MACs        - {'ModuleList': '387 GMACs'}
    fwd latency - {'ModuleList': '376.43 ms'}
depth 0:
    params      - {'DeepSpeedEngine': '503.26 M'}
depth 4:
    MACs        - {'DeepSpeedEngine': '705.63 GMACs'}    params      - {'Qwen2MoeDecoderLayer': '347.68 M'}

    fwd latency - {'DeepSpeedEngine': '670.23 ms'}    MACs        - {'Qwen2MoeDecoderLayer': '387 GMACs'}

    fwd latency - {'Qwen2MoeDecoderLayer': '376.43 ms'}
depth 5:depth 1:

    params      - {'Qwen2MoeSparseMoeBlock': '339.79 M'}
    params      - {'Qwen2MoeForCausalLM': '503.26 M'}
    MACs        - {'Qwen2MoeSparseMoeBlock': '309.69 GMACs'}
    MACs        - {'Qwen2MoeForCausalLM': '705.63 GMACs'}
    fwd latency - {'Qwen2MoeSparseMoeBlock': '314.78 ms'}
    fwd latency - {'Qwen2MoeForCausalLM': '670.23 ms'}
depth 6:depth 2:

    params      - {'Qwen2MoeExperts': '301.99 M'}
    params      - {'Qwen2MoeModel': '425.47 M'}
    MACs        - {'Qwen2MoeMLP': '309.24 GMACs'}
    MACs        - {'Qwen2MoeModel': '387 GMACs'}    fwd latency - {'Qwen2MoeExperts': '416.6 ms'}

    fwd latency - {'Qwen2MoeModel': '393.9 ms'}

------------------------------ Detailed Profile per GPU ------------------------------
Each module profile is listed after its name in the following order: 
params, percentage of total params, MACs, percentage of total MACs, fwd latency, percentage of total fwd latency, fwd FLOPS

Note: 1. A module can have torch.nn.module or torch.nn.functional to compute logits (e.g. CrossEntropyLoss). They are not counted as submodules, thus not to be printed out. However they make up the difference between a parent's MACs (or latency) and the sum of its submodules'.
2. Number of floating-point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.
3. The fwd latency listed in the top module's profile is directly captured at the module forward function in PyTorch, thus it's less than the fwd latency shown above which is captured in DeepSpeed.

depth 3:
    params      - {'ModuleList': '347.68 M'}
    MACs        - {'ModuleList': '387 GMACs'}
    fwd latency - {'ModuleList': '387.3 ms'}
depth 4:
    params      - {'Qwen2MoeDecoderLayer': '347.68 M'}
    MACs        - {'Qwen2MoeDecoderLayer': '387 GMACs'}
    fwd latency - {'Qwen2MoeDecoderLayer': '387.3 ms'}
depth 5:
    params      - {'Qwen2MoeSparseMoeBlock': '339.79 M'}
    MACs        - {'Qwen2MoeSparseMoeBlock': '309.69 GMACs'}
    fwd latency - {'Qwen2MoeSparseMoeBlock': '322.08 ms'}
Top 1 modules in terms of params, MACs or fwd latency at different model depths:depth 6:

    params      - {'Qwen2MoeExperts': '301.99 M'}
    MACs        - {'Qwen2MoeMLP': '309.24 GMACs'}
    fwd latency - {'Qwen2MoeExperts': '422.2 ms'}

------------------------------ Detailed Profile per GPU ------------------------------
Each module profile is listed after its name in the following order: 
params, percentage of total params, MACs, percentage of total MACs, fwd latency, percentage of total fwd latency, fwd FLOPS

Note: 1. A module can have torch.nn.module or torch.nn.functional to compute logits (e.g. CrossEntropyLoss). They are not counted as submodules, thus not to be printed out. However they make up the difference between a parent's MACs (or latency) and the sum of its submodules'.
2. Number of floating-point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.
3. The fwd latency listed in the top module's profile is directly captured at the module forward function in PyTorch, thus it's less than the fwd latency shown above which is captured in DeepSpeed.

depth 0:
    params      - {'DeepSpeedEngine': '503.26 M'}
    MACs        - {'DeepSpeedEngine': '705.63 GMACs'}
    fwd latency - {'DeepSpeedEngine': '667.98 ms'}
depth 1:
    params      - {'Qwen2MoeForCausalLM': '503.26 M'}
    MACs        - {'Qwen2MoeForCausalLM': '705.63 GMACs'}
    fwd latency - {'Qwen2MoeForCausalLM': '667.98 ms'}
depth 2:
    params      - {'Qwen2MoeModel': '425.47 M'}
    MACs        - {'Qwen2MoeModel': '387 GMACs'}
    fwd latency - {'Qwen2MoeModel': '391.48 ms'}
depth 3:
    params      - {'ModuleList': '347.68 M'}
    MACs        - {'ModuleList': '387 GMACs'}
    fwd latency - {'ModuleList': '385.91 ms'}
depth 4:
    params      - {'Qwen2MoeDecoderLayer': '347.68 M'}
    MACs        - {'Qwen2MoeDecoderLayer': '387 GMACs'}
    fwd latency - {'Qwen2MoeDecoderLayer': '385.91 ms'}
depth 5:
    params      - {'Qwen2MoeSparseMoeBlock': '339.79 M'}
    MACs        - {'Qwen2MoeSparseMoeBlock': '309.69 GMACs'}
    fwd latency - {'Qwen2MoeSparseMoeBlock': '323.35 ms'}
depth 6:
    params      - {'Qwen2MoeExperts': '301.99 M'}
    MACs        - {'Qwen2MoeMLP': '309.24 GMACs'}
    fwd latency - {'Qwen2MoeExperts': '429.92 ms'}

------------------------------ Detailed Profile per GPU ------------------------------
Each module profile is listed after its name in the following order: 
params, percentage of total params, MACs, percentage of total MACs, fwd latency, percentage of total fwd latency, fwd FLOPS

Note: 1. A module can have torch.nn.module or torch.nn.functional to compute logits (e.g. CrossEntropyLoss). They are not counted as submodules, thus not to be printed out. However they make up the difference between a parent's MACs (or latency) and the sum of its submodules'.
2. Number of floating-point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.
3. The fwd latency listed in the top module's profile is directly captured at the module forward function in PyTorch, thus it's less than the fwd latency shown above which is captured in DeepSpeed.

DeepSpeedEngine(
  503.26 M = 100% Params, 705.63 GMACs = 100% MACs, 653.07 ms = 100% latency, 2.16 TFLOPS
  (module): Qwen2MoeForCausalLM(
    503.26 M = 100% Params, 705.63 GMACs = 100% MACs, 653.07 ms = 100% latency, 2.16 TFLOPS
    (model): Qwen2MoeModel(
      425.47 M = 84.54% Params, 387 GMACs = 54.84% MACs, 382.29 ms = 58.54% latency, 2.03 TFLOPS
      (embed_tokens): Embedding(77.79 M = 15.46% Params, 0 MACs = 0% MACs, 126.12 us = 0.02% latency, 0 FLOPS, 151936, 512)
      (layers): ModuleList(
        (0): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 27.82 ms = 4.26% latency, 2.32 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.03 ms = 1.23% latency, 1.6 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.35 ms = 0.21% latency, 3.18 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 512.36 us = 0.08% latency, 2.1 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 510.93 us = 0.08% latency, 2.1 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.27 ms = 0.19% latency, 3.38 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 22.89 ms = 3.5% latency, 2.26 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 658.04 us = 0.1% latency, 102.08 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 31.79 ms = 4.87% latency, 1.06 GFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 787.73 us = 0.12% latency, 42.6 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 14.17 ms = 2.17% latency, 3.64 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.35 ms = 0.67% latency, 3.95 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.35 ms = 0.67% latency, 3.95 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.37 ms = 0.67% latency, 3.93 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 373.13 us = 0.06% latency, 44.96 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 231.74 us = 0.04% latency, 36.2 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 814.2 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 765.32 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (1): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 31 ms = 4.75% latency, 2.08 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 7.91 ms = 1.21% latency, 1.63 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.32 ms = 0.2% latency, 3.27 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 489 us = 0.07% latency, 2.2 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 492.57 us = 0.08% latency, 2.18 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.31 ms = 0.2% latency, 3.29 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 26.1 ms = 4% latency, 1.98 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 647.78 us = 0.1% latency, 103.7 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 34.85 ms = 5.34% latency, 962.76 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 808.48 us = 0.12% latency, 41.5 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 15.19 ms = 2.33% latency, 3.39 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.67 ms = 0.72% latency, 3.68 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.7 ms = 0.72% latency, 3.66 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.71 ms = 0.72% latency, 3.65 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 377.66 us = 0.06% latency, 44.42 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 231.03 us = 0.04% latency, 36.31 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 790.83 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 767.95 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (2): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 32.23 ms = 4.94% latency, 2 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.36 ms = 1.28% latency, 1.54 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.45 ms = 0.22% latency, 2.96 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 572.2 us = 0.09% latency, 1.88 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 530.24 us = 0.08% latency, 2.02 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.45 ms = 0.22% latency, 2.97 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 27.05 ms = 4.14% latency, 1.91 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 647.31 us = 0.1% latency, 103.78 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 36.5 ms = 5.59% latency, 919.38 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 838.28 us = 0.13% latency, 40.03 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16.12 ms = 2.47% latency, 3.2 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.03 ms = 0.77% latency, 3.41 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.97 ms = 0.76% latency, 3.45 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.04 ms = 0.77% latency, 3.41 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 369.07 us = 0.06% latency, 45.46 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 232.46 us = 0.04% latency, 36.09 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 787.97 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 770.57 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (3): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 31.95 ms = 4.89% latency, 2.02 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.33 ms = 1.27% latency, 1.55 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.44 ms = 0.22% latency, 2.98 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 548.6 us = 0.08% latency, 1.96 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 523.09 us = 0.08% latency, 2.05 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.44 ms = 0.22% latency, 2.98 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 26.79 ms = 4.1% latency, 1.93 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 628.47 us = 0.1% latency, 106.89 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 35.6 ms = 5.45% latency, 942.64 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 827.07 us = 0.13% latency, 40.57 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16.13 ms = 2.47% latency, 3.2 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.01 ms = 0.77% latency, 3.43 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.99 ms = 0.76% latency, 3.44 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.03 ms = 0.77% latency, 3.41 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 372.41 us = 0.06% latency, 45.05 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 231.03 us = 0.04% latency, 36.31 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 792.5 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 771.76 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (4): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 31.14 ms = 4.77% latency, 2.07 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.21 ms = 1.26% latency, 1.57 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.4 ms = 0.21% latency, 3.06 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 535.25 us = 0.08% latency, 2.01 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 529.05 us = 0.08% latency, 2.03 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.39 ms = 0.21% latency, 3.08 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 25.99 ms = 3.98% latency, 1.99 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 654.7 us = 0.1% latency, 102.6 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 34.34 ms = 5.26% latency, 977.19 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 818.01 us = 0.13% latency, 41.02 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 15.64 ms = 2.4% latency, 3.3 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.83 ms = 0.74% latency, 3.56 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.8 ms = 0.73% latency, 3.58 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.89 ms = 0.75% latency, 3.51 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 383.38 us = 0.06% latency, 43.76 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 233.17 us = 0.04% latency, 35.98 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 790.12 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 771.76 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (5): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 32.42 ms = 4.96% latency, 1.99 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.43 ms = 1.29% latency, 1.53 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.48 ms = 0.23% latency, 2.9 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 551.46 us = 0.08% latency, 1.95 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 530.24 us = 0.08% latency, 2.02 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.47 ms = 0.22% latency, 2.93 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 27.19 ms = 4.16% latency, 1.9 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 625.13 us = 0.1% latency, 107.46 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 36.29 ms = 5.56% latency, 924.54 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 837.33 us = 0.13% latency, 40.07 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16.48 ms = 2.52% latency, 3.13 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.13 ms = 0.79% latency, 3.35 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.11 ms = 0.78% latency, 3.36 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.14 ms = 0.79% latency, 3.34 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 367.16 us = 0.06% latency, 45.69 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 235.8 us = 0.04% latency, 35.58 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 785.11 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 773.67 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (6): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 31.79 ms = 4.87% latency, 2.03 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.25 ms = 1.26% latency, 1.56 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.45 ms = 0.22% latency, 2.95 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 532.63 us = 0.08% latency, 2.02 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 534.77 us = 0.08% latency, 2.01 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.37 ms = 0.21% latency, 3.13 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 26.59 ms = 4.07% latency, 1.94 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 638.25 us = 0.1% latency, 105.25 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 34.5 ms = 5.28% latency, 972.59 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 813.48 us = 0.12% latency, 41.25 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 15.22 ms = 2.33% latency, 3.39 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.7 ms = 0.72% latency, 3.66 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.7 ms = 0.72% latency, 3.66 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.73 ms = 0.72% latency, 3.63 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 374.08 us = 0.06% latency, 44.85 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 246.05 us = 0.04% latency, 34.09 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 791.79 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 768.18 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (7): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 31.84 ms = 4.88% latency, 2.03 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.22 ms = 1.26% latency, 1.57 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.42 ms = 0.22% latency, 3.03 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 532.63 us = 0.08% latency, 2.02 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 515.46 us = 0.08% latency, 2.08 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.39 ms = 0.21% latency, 3.08 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 26.66 ms = 4.08% latency, 1.94 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 639.68 us = 0.1% latency, 105.01 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 34.36 ms = 5.26% latency, 976.44 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 822.31 us = 0.13% latency, 40.81 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 15.63 ms = 2.39% latency, 3.3 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.84 ms = 0.74% latency, 3.55 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.83 ms = 0.74% latency, 3.56 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.88 ms = 0.75% latency, 3.52 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 366.93 us = 0.06% latency, 45.72 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 244.62 us = 0.04% latency, 34.29 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 790.36 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 768.18 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (8): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 31.95 ms = 4.89% latency, 2.02 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.38 ms = 1.28% latency, 1.54 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.45 ms = 0.22% latency, 2.97 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 535.49 us = 0.08% latency, 2.01 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 595.81 us = 0.09% latency, 1.8 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.46 ms = 0.22% latency, 2.94 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 26.75 ms = 4.1% latency, 1.93 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 642.06 us = 0.1% latency, 104.62 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 36.01 ms = 5.51% latency, 931.92 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 851.63 us = 0.13% latency, 39.4 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16.3 ms = 2.5% latency, 3.16 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.08 ms = 0.78% latency, 3.38 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.06 ms = 0.78% latency, 3.39 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.08 ms = 0.78% latency, 3.38 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 367.16 us = 0.06% latency, 45.69 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 256.06 us = 0.04% latency, 32.76 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 786.3 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 771.05 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (9): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 31.39 ms = 4.81% latency, 2.06 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.16 ms = 1.25% latency, 1.58 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.38 ms = 0.21% latency, 3.12 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 525 us = 0.08% latency, 2.05 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 503.54 us = 0.08% latency, 2.13 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.37 ms = 0.21% latency, 3.13 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 26.22 ms = 4.01% latency, 1.97 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 635.15 us = 0.1% latency, 105.76 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 33.42 ms = 5.12% latency, 1 GFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 799.89 us = 0.12% latency, 41.95 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 15.33 ms = 2.35% latency, 3.36 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.75 ms = 0.73% latency, 3.61 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.74 ms = 0.73% latency, 3.62 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.75 ms = 0.73% latency, 3.62 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 370.74 us = 0.06% latency, 45.25 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 231.27 us = 0.04% latency, 36.27 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 790.36 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 770.57 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (10): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 31.53 ms = 4.83% latency, 2.05 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.39 ms = 1.29% latency, 1.54 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.46 ms = 0.22% latency, 2.94 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 546.93 us = 0.08% latency, 1.96 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 557.9 us = 0.09% latency, 1.92 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.47 ms = 0.22% latency, 2.93 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 26.32 ms = 4.03% latency, 1.96 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 638.25 us = 0.1% latency, 105.25 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 35.37 ms = 5.42% latency, 948.56 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 829.94 us = 0.13% latency, 40.43 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16.46 ms = 2.52% latency, 3.13 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.11 ms = 0.78% latency, 3.36 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.1 ms = 0.78% latency, 3.37 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.16 ms = 0.79% latency, 3.33 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 374.08 us = 0.06% latency, 44.85 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 231.03 us = 0.04% latency, 36.31 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 806.09 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 775.58 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (11): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 31.36 ms = 4.8% latency, 2.06 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8 ms = 1.22% latency, 1.61 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.33 ms = 0.2% latency, 3.23 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 499.73 us = 0.08% latency, 2.15 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 494.48 us = 0.08% latency, 2.17 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.31 ms = 0.2% latency, 3.28 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 26.24 ms = 4.02% latency, 1.97 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 626.56 us = 0.1% latency, 107.21 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 33.56 ms = 5.14% latency, 999.7 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 806.81 us = 0.12% latency, 41.59 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 14.71 ms = 2.25% latency, 3.5 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.52 ms = 0.69% latency, 3.8 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.55 ms = 0.7% latency, 3.78 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.55 ms = 0.7% latency, 3.78 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 376.94 us = 0.06% latency, 44.51 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 231.03 us = 0.04% latency, 36.31 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 792.5 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 767.23 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
      )
      (norm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 386 us = 0.06% latency, 0 FLOPS, (512,), eps=1e-06)
      (rotary_emb): Qwen2MoeRotaryEmbedding(0 = 0% Params, 4.1 KMACs = 0% MACs, 344.28 us = 0.05% latency, 23.79 MFLOPS)
    )
    (lm_head): Linear(77.79 M = 15.46% Params, 318.63 GMACs = 45.16% MACs, 212.58 ms = 32.55% latency, 3 TFLOPS, in_features=512, out_features=151936, bias=False)
  )
)
DeepSpeedEngine(
  503.26 M = 100% Params, 705.63 GMACs = 100% MACs, 670.23 ms = 100% latency, 2.11 TFLOPS
  (module): Qwen2MoeForCausalLM(
    503.26 M = 100% Params, 705.63 GMACs = 100% MACs, 670.23 ms = 100% latency, 2.11 TFLOPS
    (model): Qwen2MoeModel(
      425.47 M = 84.54% Params, 387 GMACs = 54.84% MACs, 393.9 ms = 58.77% latency, 1.97 TFLOPS
      (embed_tokens): Embedding(77.79 M = 15.46% Params, 0 MACs = 0% MACs, 138.04 us = 0.02% latency, 0 FLOPS, 151936, 512)
      (layers): ModuleList(
        (0): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 28.49 ms = 4.25% latency, 2.27 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.35 ms = 1.25% latency, 1.54 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.38 ms = 0.21% latency, 3.12 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 532.15 us = 0.08% latency, 2.02 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 514.75 us = 0.08% latency, 2.09 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.36 ms = 0.2% latency, 3.16 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 23.31 ms = 3.48% latency, 2.22 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 696.66 us = 0.1% latency, 96.42 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 32.44 ms = 4.84% latency, 1.03 GFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 804.9 us = 0.12% latency, 41.69 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 14.31 ms = 2.13% latency, 3.6 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.38 ms = 0.65% latency, 3.92 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.37 ms = 0.65% latency, 3.93 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.42 ms = 0.66% latency, 3.88 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 380.75 us = 0.06% latency, 44.06 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 262.74 us = 0.04% latency, 31.93 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 815.87 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 774.38 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (1): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 32.12 ms = 4.79% latency, 2.01 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.38 ms = 1.25% latency, 1.54 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.4 ms = 0.21% latency, 3.06 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 533.82 us = 0.08% latency, 2.01 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 548.6 us = 0.08% latency, 1.96 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.39 ms = 0.21% latency, 3.08 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 26.76 ms = 3.99% latency, 1.93 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 684.26 us = 0.1% latency, 98.17 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 35.64 ms = 5.32% latency, 941.54 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 836.37 us = 0.12% latency, 40.12 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 15.54 ms = 2.32% latency, 3.32 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.79 ms = 0.71% latency, 3.59 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.79 ms = 0.71% latency, 3.59 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.82 ms = 0.72% latency, 3.56 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 375.99 us = 0.06% latency, 44.62 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 245.57 us = 0.04% latency, 34.16 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 791.79 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 772.95 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (2): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 32.72 ms = 4.88% latency, 1.97 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.61 ms = 1.28% latency, 1.5 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.49 ms = 0.22% latency, 2.88 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 559.09 us = 0.08% latency, 1.92 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 553.37 us = 0.08% latency, 1.94 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.48 ms = 0.22% latency, 2.9 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 27.29 ms = 4.07% latency, 1.89 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 696.66 us = 0.1% latency, 96.42 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 36.87 ms = 5.5% latency, 910.13 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 846.62 us = 0.13% latency, 39.63 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16.49 ms = 2.46% latency, 3.13 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.11 ms = 0.76% latency, 3.36 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.12 ms = 0.76% latency, 3.36 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.16 ms = 0.77% latency, 3.33 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 370.03 us = 0.06% latency, 45.34 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 248.91 us = 0.04% latency, 33.7 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 799.42 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 777.01 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (3): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 32.53 ms = 4.85% latency, 1.98 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.55 ms = 1.28% latency, 1.51 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.46 ms = 0.22% latency, 2.93 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 556.23 us = 0.08% latency, 1.93 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 560.28 us = 0.08% latency, 1.92 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.46 ms = 0.22% latency, 2.94 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 27.07 ms = 4.04% latency, 1.91 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 707.39 us = 0.11% latency, 94.96 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 34.52 ms = 5.15% latency, 972.15 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 819.21 us = 0.12% latency, 40.96 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 15.57 ms = 2.32% latency, 3.31 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.83 ms = 0.72% latency, 3.56 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.79 ms = 0.71% latency, 3.59 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.82 ms = 0.72% latency, 3.56 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 376.22 us = 0.06% latency, 44.59 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 262.98 us = 0.04% latency, 31.9 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 805.14 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 778.91 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (4): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 32.24 ms = 4.81% latency, 2 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.95 ms = 1.33% latency, 1.44 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.47 ms = 0.22% latency, 2.92 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 604.39 us = 0.09% latency, 1.78 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 576.26 us = 0.09% latency, 1.86 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.44 ms = 0.21% latency, 2.99 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 26.78 ms = 4% latency, 1.93 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 728.13 us = 0.11% latency, 92.26 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 35.16 ms = 5.25% latency, 954.33 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 832.32 us = 0.12% latency, 40.31 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 15.93 ms = 2.38% latency, 3.24 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.91 ms = 0.73% latency, 3.5 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.89 ms = 0.73% latency, 3.51 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5 ms = 0.75% latency, 3.44 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 378.85 us = 0.06% latency, 44.28 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 247.72 us = 0.04% latency, 33.86 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 862.12 us = 0.13% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 781.3 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (5): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 31.7 ms = 4.73% latency, 2.04 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.48 ms = 1.27% latency, 1.52 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.43 ms = 0.21% latency, 2.99 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 547.89 us = 0.08% latency, 1.96 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 529.53 us = 0.08% latency, 2.03 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.43 ms = 0.21% latency, 3 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 26.3 ms = 3.92% latency, 1.96 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 697.85 us = 0.1% latency, 96.26 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 35.8 ms = 5.34% latency, 937.24 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 842.57 us = 0.13% latency, 39.82 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 15.94 ms = 2.38% latency, 3.23 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.95 ms = 0.74% latency, 3.47 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.92 ms = 0.73% latency, 3.49 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.96 ms = 0.74% latency, 3.47 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 375.51 us = 0.06% latency, 44.68 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 294.45 us = 0.04% latency, 28.49 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 789.88 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 793.93 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (6): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 33.42 ms = 4.99% latency, 1.93 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.49 ms = 1.27% latency, 1.52 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.46 ms = 0.22% latency, 2.95 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 547.65 us = 0.08% latency, 1.96 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 538.59 us = 0.08% latency, 1.99 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.44 ms = 0.22% latency, 2.97 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 27.93 ms = 4.17% latency, 1.85 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 709.06 us = 0.11% latency, 94.74 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 35.15 ms = 5.24% latency, 954.68 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 824.21 us = 0.12% latency, 40.71 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16 ms = 2.39% latency, 3.22 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.99 ms = 0.74% latency, 3.44 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.97 ms = 0.74% latency, 3.46 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.92 ms = 0.73% latency, 3.49 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 375.51 us = 0.06% latency, 44.68 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 260.83 us = 0.04% latency, 32.16 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 801.09 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 775.1 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (7): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 33.52 ms = 5% latency, 1.93 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.52 ms = 1.27% latency, 1.51 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.46 ms = 0.22% latency, 2.94 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 543.83 us = 0.08% latency, 1.97 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 545.98 us = 0.08% latency, 1.97 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.44 ms = 0.22% latency, 2.98 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 28.05 ms = 4.19% latency, 1.84 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 700 us = 0.1% latency, 95.96 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 36.21 ms = 5.4% latency, 926.67 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 840.66 us = 0.13% latency, 39.91 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16.13 ms = 2.41% latency, 3.2 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.98 ms = 0.74% latency, 3.45 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.98 ms = 0.74% latency, 3.45 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.04 ms = 0.75% latency, 3.41 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 371.93 us = 0.06% latency, 45.11 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 262.02 us = 0.04% latency, 32.01 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 799.89 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 773.43 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (8): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 32.32 ms = 4.82% latency, 2 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.64 ms = 1.29% latency, 1.49 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.48 ms = 0.22% latency, 2.91 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 556.71 us = 0.08% latency, 1.93 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 617.27 us = 0.09% latency, 1.74 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.46 ms = 0.22% latency, 2.94 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 26.75 ms = 3.99% latency, 1.93 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 682.83 us = 0.1% latency, 98.38 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 35.88 ms = 5.35% latency, 935.28 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 844 us = 0.13% latency, 39.76 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16.23 ms = 2.42% latency, 3.18 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.06 ms = 0.76% latency, 3.39 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.04 ms = 0.75% latency, 3.41 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.02 ms = 0.75% latency, 3.42 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 371.22 us = 0.06% latency, 45.2 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 242.23 us = 0.04% latency, 34.63 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 808.95 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 804.66 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (9): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 33.19 ms = 4.95% latency, 1.95 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.44 ms = 1.26% latency, 1.53 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.43 ms = 0.21% latency, 3.01 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 545.02 us = 0.08% latency, 1.97 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 527.14 us = 0.08% latency, 2.04 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.43 ms = 0.21% latency, 3 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 27.71 ms = 4.14% latency, 1.86 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 696.42 us = 0.1% latency, 96.46 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 34.48 ms = 5.14% latency, 973.16 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 820.4 us = 0.12% latency, 40.9 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 15.87 ms = 2.37% latency, 3.25 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.91 ms = 0.73% latency, 3.5 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.89 ms = 0.73% latency, 3.52 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.96 ms = 0.74% latency, 3.47 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 377.66 us = 0.06% latency, 44.42 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 254.63 us = 0.04% latency, 32.94 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 804.9 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 776.77 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (10): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 32.39 ms = 4.83% latency, 1.99 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.65 ms = 1.29% latency, 1.49 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.49 ms = 0.22% latency, 2.87 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 580.79 us = 0.09% latency, 1.85 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 571.49 us = 0.09% latency, 1.88 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.49 ms = 0.22% latency, 2.87 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 26.88 ms = 4.01% latency, 1.92 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 683.31 us = 0.1% latency, 98.31 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 35.99 ms = 5.37% latency, 932.35 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 847.34 us = 0.13% latency, 39.6 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16.43 ms = 2.45% latency, 3.14 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.13 ms = 0.77% latency, 3.35 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.12 ms = 0.76% latency, 3.36 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.08 ms = 0.76% latency, 3.38 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 372.41 us = 0.06% latency, 45.05 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 246.29 us = 0.04% latency, 34.06 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 841.86 us = 0.13% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 780.34 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (11): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 32.67 ms = 4.87% latency, 1.98 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.29 ms = 1.24% latency, 1.55 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.38 ms = 0.21% latency, 3.11 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 527.38 us = 0.08% latency, 2.04 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 537.16 us = 0.08% latency, 2 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.37 ms = 0.2% latency, 3.13 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 27.23 ms = 4.06% latency, 1.9 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 706.91 us = 0.11% latency, 95.03 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 34.07 ms = 5.08% latency, 984.98 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 814.2 us = 0.12% latency, 41.21 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 15.24 ms = 2.27% latency, 3.38 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.71 ms = 0.7% latency, 3.65 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.7 ms = 0.7% latency, 3.66 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.71 ms = 0.7% latency, 3.64 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 380.04 us = 0.06% latency, 44.15 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 242.71 us = 0.04% latency, 34.56 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 797.51 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 773.91 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
      )
      (norm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 392.44 us = 0.06% latency, 0 FLOPS, (512,), eps=1e-06)
      (rotary_emb): Qwen2MoeRotaryEmbedding(0 = 0% Params, 4.1 KMACs = 0% MACs, 427.01 us = 0.06% latency, 19.18 MFLOPS)
    )
    (lm_head): Linear(77.79 M = 15.46% Params, 318.63 GMACs = 45.16% MACs, 217.94 ms = 32.52% latency, 2.92 TFLOPS, in_features=512, out_features=151936, bias=False)
  )
)
DeepSpeedEngine(
  503.26 M = 100% Params, 705.63 GMACs = 100% MACs, 667.98 ms = 100% latency, 2.11 TFLOPS
  (module): Qwen2MoeForCausalLM(
    503.26 M = 100% Params, 705.63 GMACs = 100% MACs, 667.98 ms = 100% latency, 2.11 TFLOPS
    (model): Qwen2MoeModel(
      425.47 M = 84.54% Params, 387 GMACs = 54.84% MACs, 391.48 ms = 58.61% latency, 1.98 TFLOPS
      (embed_tokens): Embedding(77.79 M = 15.46% Params, 0 MACs = 0% MACs, 126.36 us = 0.02% latency, 0 FLOPS, 151936, 512)
      (layers): ModuleList(
        (0): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 27.89 ms = 4.18% latency, 2.31 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.14 ms = 1.22% latency, 1.58 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.36 ms = 0.2% latency, 3.16 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 518.8 us = 0.08% latency, 2.07 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 497.58 us = 0.07% latency, 2.16 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.36 ms = 0.2% latency, 3.16 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 22.95 ms = 3.44% latency, 2.25 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 648.74 us = 0.1% latency, 103.55 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 32.72 ms = 4.9% latency, 1.03 GFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 797.27 us = 0.12% latency, 42.09 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 14.49 ms = 2.17% latency, 3.56 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.48 ms = 0.67% latency, 3.84 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.45 ms = 0.67% latency, 3.86 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.48 ms = 0.67% latency, 3.84 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 372.41 us = 0.06% latency, 45.05 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 231.98 us = 0.03% latency, 36.16 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 791.31 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 766.52 us = 0.11% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (1): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 30.51 ms = 4.57% latency, 2.12 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 7.93 ms = 1.19% latency, 1.62 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.3 ms = 0.2% latency, 3.29 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 512.6 us = 0.08% latency, 2.09 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 482.32 us = 0.07% latency, 2.23 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.3 ms = 0.19% latency, 3.31 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 25.62 ms = 3.84% latency, 2.02 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 624.66 us = 0.09% latency, 107.54 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 35.12 ms = 5.26% latency, 955.43 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 823.02 us = 0.12% latency, 40.77 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 15.02 ms = 2.25% latency, 3.43 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.5 ms = 0.67% latency, 3.82 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.68 ms = 0.7% latency, 3.67 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.73 ms = 0.71% latency, 3.63 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 367.64 us = 0.06% latency, 45.63 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 229.12 us = 0.03% latency, 36.61 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 792.26 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 762.94 us = 0.11% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (2): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 33.84 ms = 5.07% latency, 1.91 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.56 ms = 1.28% latency, 1.5 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.52 ms = 0.23% latency, 2.82 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 576.97 us = 0.09% latency, 1.86 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 549.79 us = 0.08% latency, 1.95 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.52 ms = 0.23% latency, 2.82 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 28.53 ms = 4.27% latency, 1.81 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 660.9 us = 0.1% latency, 101.64 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 38 ms = 5.69% latency, 883.04 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 858.55 us = 0.13% latency, 39.08 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16.96 ms = 2.54% latency, 3.04 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.32 ms = 0.8% latency, 3.23 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.29 ms = 0.79% latency, 3.25 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.27 ms = 0.79% latency, 3.26 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 367.4 us = 0.06% latency, 45.66 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 234.13 us = 0.04% latency, 35.83 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 808.95 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 775.1 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (3): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 33.08 ms = 4.95% latency, 1.95 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.49 ms = 1.27% latency, 1.52 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.5 ms = 0.22% latency, 2.86 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 559.33 us = 0.08% latency, 1.92 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 544.07 us = 0.08% latency, 1.97 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.5 ms = 0.22% latency, 2.86 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 27.8 ms = 4.16% latency, 1.86 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 633 us = 0.09% latency, 106.12 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 36.92 ms = 5.53% latency, 908.94 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 846.62 us = 0.13% latency, 39.63 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16.5 ms = 2.47% latency, 3.12 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.11 ms = 0.77% latency, 3.36 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.11 ms = 0.76% latency, 3.36 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.18 ms = 0.77% latency, 3.32 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 368.12 us = 0.06% latency, 45.58 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 230.79 us = 0.03% latency, 36.35 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 797.27 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 789.4 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (4): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 32.39 ms = 4.85% latency, 1.99 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.39 ms = 1.26% latency, 1.54 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.45 ms = 0.22% latency, 2.96 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 563.14 us = 0.08% latency, 1.91 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 551.7 us = 0.08% latency, 1.95 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.45 ms = 0.22% latency, 2.96 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 27.12 ms = 4.06% latency, 1.91 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 694.04 us = 0.1% latency, 96.79 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 35.72 ms = 5.35% latency, 939.3 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 844.96 us = 0.13% latency, 39.71 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16.15 ms = 2.42% latency, 3.19 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.01 ms = 0.75% latency, 3.43 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.97 ms = 0.74% latency, 3.46 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.08 ms = 0.76% latency, 3.38 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 370.74 us = 0.06% latency, 45.25 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 232.22 us = 0.03% latency, 36.12 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 787.02 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 771.76 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (5): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 32.71 ms = 4.9% latency, 1.97 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.46 ms = 1.27% latency, 1.52 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.48 ms = 0.22% latency, 2.9 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 556.95 us = 0.08% latency, 1.93 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 538.11 us = 0.08% latency, 2 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.5 ms = 0.22% latency, 2.87 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 27.43 ms = 4.11% latency, 1.88 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 632.29 us = 0.09% latency, 106.24 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 36.88 ms = 5.52% latency, 909.85 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 848.53 us = 0.13% latency, 39.54 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16.68 ms = 2.5% latency, 3.09 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.19 ms = 0.78% latency, 3.31 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.17 ms = 0.77% latency, 3.32 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.23 ms = 0.78% latency, 3.29 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 373.13 us = 0.06% latency, 44.96 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 248.19 us = 0.04% latency, 33.8 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 790.12 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 772 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (6): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 32.62 ms = 4.88% latency, 1.98 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.41 ms = 1.26% latency, 1.53 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.48 ms = 0.22% latency, 2.9 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 553.13 us = 0.08% latency, 1.94 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 532.87 us = 0.08% latency, 2.02 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.47 ms = 0.22% latency, 2.92 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 27.4 ms = 4.1% latency, 1.89 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 620.6 us = 0.09% latency, 108.24 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 36.6 ms = 5.48% latency, 916.74 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 844.96 us = 0.13% latency, 39.71 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16.49 ms = 2.47% latency, 3.13 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.13 ms = 0.77% latency, 3.35 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.12 ms = 0.77% latency, 3.35 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.15 ms = 0.77% latency, 3.34 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 369.31 us = 0.06% latency, 45.43 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 232.22 us = 0.03% latency, 36.12 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 794.65 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 774.15 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (7): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 32.17 ms = 4.82% latency, 2.01 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.28 ms = 1.24% latency, 1.56 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.43 ms = 0.21% latency, 3 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 537.4 us = 0.08% latency, 2 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 519.99 us = 0.08% latency, 2.06 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.43 ms = 0.21% latency, 3.01 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 26.94 ms = 4.03% latency, 1.92 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 629.43 us = 0.09% latency, 106.72 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 35.54 ms = 5.32% latency, 944.18 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 825.88 us = 0.12% latency, 40.63 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 15.88 ms = 2.38% latency, 3.25 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.95 ms = 0.74% latency, 3.47 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.93 ms = 0.74% latency, 3.48 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.92 ms = 0.74% latency, 3.49 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 365.02 us = 0.05% latency, 45.96 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 229.12 us = 0.03% latency, 36.61 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 784.4 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 769.14 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (8): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 32.86 ms = 4.92% latency, 1.96 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.56 ms = 1.28% latency, 1.51 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.49 ms = 0.22% latency, 2.87 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 554.08 us = 0.08% latency, 1.94 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 632.05 us = 0.09% latency, 1.7 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.5 ms = 0.22% latency, 2.86 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 27.51 ms = 4.12% latency, 1.88 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 665.43 us = 0.1% latency, 100.95 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 36.72 ms = 5.5% latency, 913.75 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 856.16 us = 0.13% latency, 39.19 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16.71 ms = 2.5% latency, 3.08 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.2 ms = 0.78% latency, 3.31 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.19 ms = 0.78% latency, 3.31 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.25 ms = 0.79% latency, 3.27 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 366.93 us = 0.05% latency, 45.72 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 251.29 us = 0.04% latency, 33.38 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 787.73 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 770.09 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (9): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 32.33 ms = 4.84% latency, 2 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.42 ms = 1.26% latency, 1.53 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.5 ms = 0.22% latency, 2.87 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 549.08 us = 0.08% latency, 1.96 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 555.52 us = 0.08% latency, 1.93 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.47 ms = 0.22% latency, 2.93 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 27.07 ms = 4.05% latency, 1.91 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 636.58 us = 0.1% latency, 105.52 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 34.94 ms = 5.23% latency, 960.29 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 818.25 us = 0.12% latency, 41.01 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 15.62 ms = 2.34% latency, 3.3 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.96 ms = 0.74% latency, 3.47 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.78 ms = 0.72% latency, 3.59 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.81 ms = 0.72% latency, 3.57 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 361.68 us = 0.05% latency, 46.39 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 230.07 us = 0.03% latency, 36.46 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 791.79 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 771.28 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (10): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 32.7 ms = 4.89% latency, 1.97 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.49 ms = 1.27% latency, 1.52 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.5 ms = 0.22% latency, 2.86 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 558.61 us = 0.08% latency, 1.92 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 544.79 us = 0.08% latency, 1.97 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.51 ms = 0.23% latency, 2.85 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 27.4 ms = 4.1% latency, 1.89 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 635.86 us = 0.1% latency, 105.64 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 36.54 ms = 5.47% latency, 918.36 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 858.55 us = 0.13% latency, 39.08 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16.83 ms = 2.52% latency, 3.06 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.26 ms = 0.79% latency, 3.27 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.27 ms = 0.79% latency, 3.26 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.23 ms = 0.78% latency, 3.29 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 362.63 us = 0.05% latency, 46.26 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 227.69 us = 0.03% latency, 36.84 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 834.94 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 776.53 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (11): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 32.8 ms = 4.91% latency, 1.97 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.07 ms = 1.21% latency, 1.6 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.37 ms = 0.21% latency, 3.13 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 509.5 us = 0.08% latency, 2.11 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 513.32 us = 0.08% latency, 2.09 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.35 ms = 0.2% latency, 3.18 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 27.58 ms = 4.13% latency, 1.87 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 621.08 us = 0.09% latency, 108.16 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 34.22 ms = 5.12% latency, 980.56 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 815.15 us = 0.12% latency, 41.16 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 15.05 ms = 2.25% latency, 3.42 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.65 ms = 0.7% latency, 3.7 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.64 ms = 0.69% latency, 3.7 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.69 ms = 0.7% latency, 3.67 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 370.74 us = 0.06% latency, 45.25 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 232.46 us = 0.03% latency, 36.09 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 789.88 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 767.47 us = 0.11% latency, 0 FLOPS, (512,), eps=1e-06)
        )
      )
      (norm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 383.62 us = 0.06% latency, 0 FLOPS, (512,), eps=1e-06)
      (rotary_emb): Qwen2MoeRotaryEmbedding(0 = 0% Params, 4.1 KMACs = 0% MACs, 347.14 us = 0.05% latency, 23.6 MFLOPS)
    )
    (lm_head): Linear(77.79 M = 15.46% Params, 318.63 GMACs = 45.16% MACs, 218.07 ms = 32.65% latency, 2.92 TFLOPS, in_features=512, out_features=151936, bias=False)
  )
)
------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------
Profile stoped


-------------------------- DeepSpeed Flops Profiler --------------------------
Profile Summary at step 10:
Notations:
data parallel size (dp_size), model parallel size(mp_size),
number of parameters (params), number of multiply-accumulate operations(MACs),
number of floating-point operations (flops), floating-point operations per second (FLOPS),
fwd latency (forward propagation latency), bwd latency (backward propagation latency),
step (weights update latency), iter latency (sum of fwd, bwd and step latency)

params per GPU:                                                         503.26 M
params of model = params per GPU * mp_size:                             0       
fwd MACs per GPU:                                                       705.63 GMACs
fwd flops per GPU:                                                      1.41 T  
fwd flops of model = fwd flops per GPU * mp_size:                       1.41 T  
fwd latency:                                                            674.58 ms
fwd FLOPS per GPU = fwd flops per GPU / fwd latency:                    2.09 TFLOPS

----------------------------- Aggregated Profile per GPU -----------------------------
Top 1 modules in terms of params, MACs or fwd latency at different model depths:
depth 0:
    params      - {'DeepSpeedEngine': '503.26 M'}
    MACs        - {'DeepSpeedEngine': '705.63 GMACs'}
    fwd latency - {'DeepSpeedEngine': '674.58 ms'}
depth 1:
    params      - {'Qwen2MoeForCausalLM': '503.26 M'}
    MACs        - {'Qwen2MoeForCausalLM': '705.63 GMACs'}
    fwd latency - {'Qwen2MoeForCausalLM': '674.58 ms'}
depth 2:
    params      - {'Qwen2MoeModel': '425.47 M'}
    MACs        - {'Qwen2MoeModel': '387 GMACs'}
    fwd latency - {'Qwen2MoeModel': '394.2 ms'}
depth 3:
    params      - {'ModuleList': '347.68 M'}
    MACs        - {'ModuleList': '387 GMACs'}
    fwd latency - {'ModuleList': '388.66 ms'}
depth 4:
    params      - {'Qwen2MoeDecoderLayer': '347.68 M'}
    MACs        - {'Qwen2MoeDecoderLayer': '387 GMACs'}
    fwd latency - {'Qwen2MoeDecoderLayer': '388.66 ms'}
depth 5:
    params      - {'Qwen2MoeSparseMoeBlock': '339.79 M'}
    MACs        - {'Qwen2MoeSparseMoeBlock': '309.69 GMACs'}
    fwd latency - {'Qwen2MoeSparseMoeBlock': '325.48 ms'}
depth 6:
    params      - {'Qwen2MoeExperts': '301.99 M'}
    MACs        - {'Qwen2MoeMLP': '309.24 GMACs'}
    fwd latency - {'Qwen2MoeExperts': '430.28 ms'}

------------------------------ Detailed Profile per GPU ------------------------------
Each module profile is listed after its name in the following order: 
params, percentage of total params, MACs, percentage of total MACs, fwd latency, percentage of total fwd latency, fwd FLOPS

Note: 1. A module can have torch.nn.module or torch.nn.functional to compute logits (e.g. CrossEntropyLoss). They are not counted as submodules, thus not to be printed out. However they make up the difference between a parent's MACs (or latency) and the sum of its submodules'.
2. Number of floating-point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.
3. The fwd latency listed in the top module's profile is directly captured at the module forward function in PyTorch, thus it's less than the fwd latency shown above which is captured in DeepSpeed.

DeepSpeedEngine(
  503.26 M = 100% Params, 705.63 GMACs = 100% MACs, 674.58 ms = 100% latency, 2.09 TFLOPS
  (module): Qwen2MoeForCausalLM(
    503.26 M = 100% Params, 705.63 GMACs = 100% MACs, 674.58 ms = 100% latency, 2.09 TFLOPS
    (model): Qwen2MoeModel(
      425.47 M = 84.54% Params, 387 GMACs = 54.84% MACs, 394.2 ms = 58.44% latency, 1.96 TFLOPS
      (embed_tokens): Embedding(77.79 M = 15.46% Params, 0 MACs = 0% MACs, 157.83 us = 0.02% latency, 0 FLOPS, 151936, 512)
      (layers): ModuleList(
        (0): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 27.95 ms = 4.14% latency, 2.31 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.1 ms = 1.2% latency, 1.59 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.37 ms = 0.2% latency, 3.14 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 514.98 us = 0.08% latency, 2.08 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 503.54 us = 0.07% latency, 2.13 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.36 ms = 0.2% latency, 3.15 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 23.02 ms = 3.41% latency, 2.24 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 627.52 us = 0.09% latency, 107.05 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 32.27 ms = 4.78% latency, 1.04 GFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 790.6 us = 0.12% latency, 42.44 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 14.92 ms = 2.21% latency, 3.46 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.64 ms = 0.69% latency, 3.7 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.63 ms = 0.69% latency, 3.71 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.55 ms = 0.67% latency, 3.78 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 372.17 us = 0.06% latency, 45.08 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 226.74 us = 0.03% latency, 37 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 794.41 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 768.42 us = 0.11% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (1): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 29.56 ms = 4.38% latency, 2.18 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 7.97 ms = 1.18% latency, 1.62 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.32 ms = 0.2% latency, 3.24 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 518.8 us = 0.08% latency, 2.07 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 490.67 us = 0.07% latency, 2.19 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.31 ms = 0.19% latency, 3.27 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 24.69 ms = 3.66% latency, 2.09 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 633 us = 0.09% latency, 106.12 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 34.92 ms = 5.18% latency, 960.86 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 817.3 us = 0.12% latency, 41.06 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 14.72 ms = 2.18% latency, 3.5 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.53 ms = 0.67% latency, 3.8 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.52 ms = 0.67% latency, 3.8 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.56 ms = 0.68% latency, 3.77 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 370.98 us = 0.05% latency, 45.22 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 238.9 us = 0.04% latency, 35.11 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 794.41 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 762.7 us = 0.11% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (2): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 33.85 ms = 5.02% latency, 1.91 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.25 ms = 1.22% latency, 1.56 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.4 ms = 0.21% latency, 3.07 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 522.38 us = 0.08% latency, 2.06 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 513.32 us = 0.08% latency, 2.09 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.5 ms = 0.22% latency, 2.86 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 28.71 ms = 4.26% latency, 1.8 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 644.92 us = 0.1% latency, 104.16 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 38.16 ms = 5.66% latency, 879.32 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 863.79 us = 0.13% latency, 38.85 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16.78 ms = 2.49% latency, 3.07 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.21 ms = 0.77% latency, 3.3 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.2 ms = 0.77% latency, 3.3 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.28 ms = 0.78% latency, 3.25 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 363.59 us = 0.05% latency, 46.14 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 233.17 us = 0.03% latency, 35.98 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 782.49 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 774.15 us = 0.11% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (3): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 33.62 ms = 4.98% latency, 1.92 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.57 ms = 1.27% latency, 1.5 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.54 ms = 0.23% latency, 2.79 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 571.01 us = 0.08% latency, 1.88 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 566.01 us = 0.08% latency, 1.9 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.51 ms = 0.22% latency, 2.85 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 28.27 ms = 4.19% latency, 1.83 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 653.98 us = 0.1% latency, 102.72 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 37.09 ms = 5.5% latency, 904.59 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 847.82 us = 0.13% latency, 39.58 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16.88 ms = 2.5% latency, 3.05 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.32 ms = 0.79% latency, 3.23 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.2 ms = 0.77% latency, 3.3 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.27 ms = 0.78% latency, 3.26 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 368.12 us = 0.05% latency, 45.58 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 239.85 us = 0.04% latency, 34.97 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 798.23 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 775.81 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (4): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 32.91 ms = 4.88% latency, 1.96 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.4 ms = 1.24% latency, 1.53 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.47 ms = 0.22% latency, 2.92 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 536.2 us = 0.08% latency, 2 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 530.24 us = 0.08% latency, 2.02 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.45 ms = 0.22% latency, 2.96 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 27.64 ms = 4.1% latency, 1.87 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 643.73 us = 0.1% latency, 104.35 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 35.5 ms = 5.26% latency, 945.25 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 837.09 us = 0.12% latency, 40.08 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16.21 ms = 2.4% latency, 3.18 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.03 ms = 0.75% latency, 3.41 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.03 ms = 0.74% latency, 3.42 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.06 ms = 0.75% latency, 3.4 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 372.65 us = 0.06% latency, 45.02 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 229.6 us = 0.03% latency, 36.54 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 801.56 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 775.1 us = 0.11% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (5): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 32.78 ms = 4.86% latency, 1.97 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.54 ms = 1.27% latency, 1.51 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.5 ms = 0.22% latency, 2.86 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 555.04 us = 0.08% latency, 1.93 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 553.13 us = 0.08% latency, 1.94 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.5 ms = 0.22% latency, 2.86 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 27.53 ms = 4.08% latency, 1.88 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 635.62 us = 0.09% latency, 105.68 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 37.56 ms = 5.57% latency, 893.33 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 856.16 us = 0.13% latency, 39.19 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16.8 ms = 2.49% latency, 3.07 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.21 ms = 0.77% latency, 3.3 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.2 ms = 0.77% latency, 3.3 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.29 ms = 0.78% latency, 3.25 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 370.74 us = 0.05% latency, 45.25 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 256.3 us = 0.04% latency, 32.73 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 789.88 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 773.19 us = 0.11% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (6): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 32.52 ms = 4.82% latency, 1.98 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.43 ms = 1.25% latency, 1.53 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.49 ms = 0.22% latency, 2.87 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 551.7 us = 0.08% latency, 1.95 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 531.67 us = 0.08% latency, 2.02 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.47 ms = 0.22% latency, 2.92 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 27.28 ms = 4.04% latency, 1.89 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 637.05 us = 0.09% latency, 105.45 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 35.07 ms = 5.2% latency, 956.65 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 820.16 us = 0.12% latency, 40.91 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16.03 ms = 2.38% latency, 3.22 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.1 ms = 0.76% latency, 3.37 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.97 ms = 0.74% latency, 3.46 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.82 ms = 0.72% latency, 3.56 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 371.46 us = 0.06% latency, 45.17 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 273.94 us = 0.04% latency, 30.62 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 795.84 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 771.28 us = 0.11% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (7): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 34.07 ms = 5.05% latency, 1.89 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.98 ms = 1.33% latency, 1.43 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.52 ms = 0.23% latency, 2.83 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 548.84 us = 0.08% latency, 1.96 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 538.35 us = 0.08% latency, 1.99 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.47 ms = 0.22% latency, 2.92 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 28.13 ms = 4.17% latency, 1.84 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 631.57 us = 0.09% latency, 106.36 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 36.56 ms = 5.42% latency, 917.72 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 840.9 us = 0.12% latency, 39.9 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16.44 ms = 2.44% latency, 3.14 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.09 ms = 0.76% latency, 3.37 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.09 ms = 0.75% latency, 3.38 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.17 ms = 0.77% latency, 3.32 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 366.69 us = 0.05% latency, 45.75 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 226.74 us = 0.03% latency, 37 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 789.17 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 774.62 us = 0.11% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (8): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 32.61 ms = 4.83% latency, 1.98 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.45 ms = 1.25% latency, 1.52 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.48 ms = 0.22% latency, 2.9 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 555.28 us = 0.08% latency, 1.93 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 535.25 us = 0.08% latency, 2.01 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.48 ms = 0.22% latency, 2.9 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 27.36 ms = 4.06% latency, 1.89 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 642.3 us = 0.1% latency, 104.58 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 36.79 ms = 5.45% latency, 912.02 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 846.86 us = 0.13% latency, 39.62 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 16.59 ms = 2.46% latency, 3.11 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.16 ms = 0.76% latency, 3.33 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.15 ms = 0.76% latency, 3.33 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.21 ms = 0.77% latency, 3.3 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 365.02 us = 0.05% latency, 45.96 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 224.83 us = 0.03% latency, 37.31 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 785.11 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 785.11 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (9): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 32.3 ms = 4.79% latency, 2 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.24 ms = 1.22% latency, 1.56 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.42 ms = 0.21% latency, 3.02 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 535.25 us = 0.08% latency, 2.01 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 513.32 us = 0.08% latency, 2.09 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.4 ms = 0.21% latency, 3.06 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 27.04 ms = 4.01% latency, 1.91 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 606.3 us = 0.09% latency, 110.79 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 34.14 ms = 5.06% latency, 982.87 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 806.33 us = 0.12% latency, 41.61 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 15.5 ms = 2.3% latency, 3.33 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.88 ms = 0.72% latency, 3.52 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.8 ms = 0.71% latency, 3.58 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.72 ms = 0.7% latency, 3.64 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 365.5 us = 0.05% latency, 45.9 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 224.83 us = 0.03% latency, 37.31 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 791.31 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 770.09 us = 0.11% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (10): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 33.47 ms = 4.96% latency, 1.93 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.64 ms = 1.28% latency, 1.49 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.54 ms = 0.23% latency, 2.78 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 573.16 us = 0.08% latency, 1.87 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 573.87 us = 0.09% latency, 1.87 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.55 ms = 0.23% latency, 2.77 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 28.11 ms = 4.17% latency, 1.84 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 659.47 us = 0.1% latency, 101.86 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 37.48 ms = 5.56% latency, 895.29 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 867.13 us = 0.13% latency, 38.7 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 17.12 ms = 2.54% latency, 3.01 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.39 ms = 0.8% latency, 3.19 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.29 ms = 0.78% latency, 3.25 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 5.36 ms = 0.79% latency, 3.21 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 363.83 us = 0.05% latency, 46.11 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 243.19 us = 0.04% latency, 34.49 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 802.04 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 777.01 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
        )
        (11): Qwen2MoeDecoderLayer(
          28.97 M = 5.76% Params, 32.25 GMACs = 4.57% MACs, 33.02 ms = 4.89% latency, 1.96 TFLOPS
          (self_attn): Qwen2MoeAttention(
            656.13 K = 0.13% Params, 6.44 GMACs = 0.91% MACs, 8.22 ms = 1.22% latency, 1.57 TFLOPS
            (q_proj): Linear(262.66 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.41 ms = 0.21% latency, 3.05 TFLOPS, in_features=512, out_features=512, bias=True)
            (k_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 525.24 us = 0.08% latency, 2.04 TFLOPS, in_features=512, out_features=128, bias=True)
            (v_proj): Linear(65.66 K = 0.01% Params, 536.87 MMACs = 0.08% MACs, 518.8 us = 0.08% latency, 2.07 TFLOPS, in_features=512, out_features=128, bias=True)
            (o_proj): Linear(262.14 K = 0.05% Params, 2.15 GMACs = 0.3% MACs, 1.41 ms = 0.21% latency, 3.04 TFLOPS, in_features=512, out_features=512, bias=False)
          )
          (mlp): Qwen2MoeSparseMoeBlock(
            28.32 M = 5.63% Params, 25.81 GMACs = 3.66% MACs, 27.7 ms = 4.11% latency, 1.87 TFLOPS
            (gate): Qwen2MoeTopKRouter(4.1 K = 0% Params, 33.55 MMACs = 0% MACs, 639.2 us = 0.09% latency, 105.09 GFLOPS)
            (experts): Qwen2MoeExperts(
              25.17 M = 5% Params, 0 MACs = 0% MACs, 34.73 ms = 5.15% latency, 966.1 MFLOPS
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 812.77 us = 0.12% latency, 41.28 GFLOPS)
            )
            (shared_expert): Qwen2MoeMLP(
              3.15 M = 0.63% Params, 25.77 GMACs = 3.65% MACs, 15.5 ms = 2.3% latency, 3.33 TFLOPS
              (gate_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.79 ms = 0.71% latency, 3.59 TFLOPS, in_features=512, out_features=2048, bias=False)
              (up_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.78 ms = 0.71% latency, 3.59 TFLOPS, in_features=512, out_features=2048, bias=False)
              (down_proj): Linear(1.05 M = 0.21% Params, 8.59 GMACs = 1.22% MACs, 4.84 ms = 0.72% latency, 3.55 TFLOPS, in_features=2048, out_features=512, bias=False)
              (act_fn): SiLUActivation(0 = 0% Params, 0 MACs = 0% MACs, 372.89 us = 0.06% latency, 44.99 GFLOPS)
            )
            (shared_expert_gate): Linear(512 = 0% Params, 4.19 MMACs = 0% MACs, 231.98 us = 0.03% latency, 36.16 GFLOPS, in_features=512, out_features=1, bias=False)
          )
          (input_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 795.6 us = 0.12% latency, 0 FLOPS, (512,), eps=1e-06)
          (post_attention_layernorm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 767.95 us = 0.11% latency, 0 FLOPS, (512,), eps=1e-06)
        )
      )
      (norm): Qwen2MoeRMSNorm(512 = 0% Params, 0 MACs = 0% MACs, 388.38 us = 0.06% latency, 0 FLOPS, (512,), eps=1e-06)
      (rotary_emb): Qwen2MoeRotaryEmbedding(0 = 0% Params, 4.1 KMACs = 0% MACs, 363.11 us = 0.05% latency, 22.56 MFLOPS)
    )
    (lm_head): Linear(77.79 M = 15.46% Params, 318.63 GMACs = 45.16% MACs, 221.89 ms = 32.89% latency, 2.87 TFLOPS, in_features=512, out_features=151936, bias=False)
  )
)
------------------------------------------------------------------------------
Epoch 0:   1%|▏         | 10/743 [00:30<33:16,  2.72s/it, loss=11.9581, global_step=11, toks/s=3334.7, gpu_util=96%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=24.2G] Epoch 0, Step 10, Global Step 11, Loss: 11.9581, Tokens/s: 3334.7, Tokens: 8692, GPU Util: 96%, GPU Mem: 13.8G/15.0G, CPU Util: 9%, CPU Mem: 24.2G/186.7G

GPU Utilization / Memory (all devices):
GPU  Util(%)  Mem(GB Used/Total)
--------------------------------
0        96%   13.8G/ 15.0G
1        97%   12.7G/ 15.0G
2        88%   12.8G/ 15.0G
3        87%   13.5G/ 15.0G
Epoch 0:   1%|▏         | 11/743 [00:30<33:59,  2.79s/it, loss=11.9581, global_step=11, toks/s=3334.7, gpu_util=96%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=24.2G][Rank 0] time (ms) | fwd_microstep: 658.51 | bwd_microstep: 1846.37 | bwd_inner_microstep: 1720.13 | bwd_allreduce_microstep: 126.19 | step_microstep: 0.04
Epoch 0:   1%|▏         | 11/743 [00:32<33:59,  2.79s/it, loss=11.9859, global_step=12, toks/s=3416.2, gpu_util=99%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=24.2G]Epoch 0, Step 11, Global Step 12, Loss: 11.9859, Tokens/s: 3416.2, Tokens: 8560, GPU Util: 99%, GPU Mem: 13.8G/15.0G, CPU Util: 9%, CPU Mem: 24.2G/186.7G

GPU Utilization / Memory (all devices):
GPU  Util(%)  Mem(GB Used/Total)
--------------------------------
0        99%   13.8G/ 15.0G
1       100%   12.7G/ 15.0G
2       100%   12.8G/ 15.0G
3       100%   13.5G/ 15.0G
Epoch 0:   2%|▏         | 12/743 [00:32<32:56,  2.70s/it, loss=11.9859, global_step=12, toks/s=3416.2, gpu_util=99%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=24.2G][Rank 0] time (ms) | fwd_microstep: 657.98 | bwd_microstep: 1836.72 | bwd_inner_microstep: 1710.39 | bwd_allreduce_microstep: 126.28 | step_microstep: 0.05
Epoch 0:   2%|▏         | 12/743 [00:35<32:56,  2.70s/it, loss=11.9409, global_step=13, toks/s=3806.7, gpu_util=100%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=24.1G]Epoch 0, Step 12, Global Step 13, Loss: 11.9409, Tokens/s: 3806.7, Tokens: 9500, GPU Util: 100%, GPU Mem: 13.8G/15.0G, CPU Util: 9%, CPU Mem: 24.1G/186.7G

GPU Utilization / Memory (all devices):
GPU  Util(%)  Mem(GB Used/Total)
--------------------------------
0       100%   13.8G/ 15.0G
1       100%   12.7G/ 15.0G
2       100%   12.8G/ 15.0G
3       100%   13.5G/ 15.0G
Epoch 0:   2%|▏         | 13/743 [00:35<32:09,  2.64s/it, loss=11.9409, global_step=13, toks/s=3806.7, gpu_util=100%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=24.1G][Rank 0] time (ms) | fwd_microstep: 660.06 | bwd_microstep: 1837.78 | bwd_inner_microstep: 1711.83 | bwd_allreduce_microstep: 125.90 | step_microstep: 0.04
Epoch 0:   2%|▏         | 13/743 [00:37<32:09,  2.64s/it, loss=11.9725, global_step=14, toks/s=3616.1, gpu_util=100%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=24.0G]Epoch 0, Step 13, Global Step 14, Loss: 11.9725, Tokens/s: 3616.1, Tokens: 9036, GPU Util: 100%, GPU Mem: 13.8G/15.0G, CPU Util: 9%, CPU Mem: 24.0G/186.7G

GPU Utilization / Memory (all devices):
GPU  Util(%)  Mem(GB Used/Total)
--------------------------------
0       100%   13.8G/ 15.0G
1       100%   12.7G/ 15.0G
2       100%   12.8G/ 15.0G
3        99%   13.5G/ 15.0G
Epoch 0:   2%|▏         | 14/743 [00:37<31:36,  2.60s/it, loss=11.9725, global_step=14, toks/s=3616.1, gpu_util=100%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=24.0G][Rank 0] time (ms) | fwd_microstep: 661.87 | bwd_microstep: 1848.76 | bwd_inner_microstep: 1722.94 | bwd_allreduce_microstep: 125.78 | step_microstep: 0.04
Epoch 0:   2%|▏         | 14/743 [00:40<31:36,  2.60s/it, loss=11.9704, global_step=15, toks/s=4675.8, gpu_util=100%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=23.9G]Epoch 0, Step 14, Global Step 15, Loss: 11.9704, Tokens/s: 4675.8, Tokens: 11744, GPU Util: 100%, GPU Mem: 13.8G/15.0G, CPU Util: 9%, CPU Mem: 23.9G/186.7G

GPU Utilization / Memory (all devices):
GPU  Util(%)  Mem(GB Used/Total)
--------------------------------
0       100%   13.8G/ 15.0G
1       100%   12.7G/ 15.0G
2       100%   12.8G/ 15.0G
3       100%   13.5G/ 15.0G
Epoch 0:   2%|▏         | 15/743 [00:40<31:16,  2.58s/it, loss=11.9704, global_step=15, toks/s=4675.8, gpu_util=100%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=23.9G][Rank 0] time (ms) | optimizer_allgather: 166.81 | optimizer_gradients: 41.10 | optimizer_step: 227.74
[Rank 0] time (ms) | fwd_microstep: 661.57 | bwd_microstep: 2522.57 | bwd_inner_microstep: 2378.47 | bwd_allreduce_microstep: 144.02 | step_microstep: 436.76
[Rank 0] time (ms) | fwd: 5286.45 | bwd: 15469.35 | bwd_inner: 14437.15 | bwd_allreduce: 1031.78 | step: 437.03
Epoch 0:   2%|▏         | 15/743 [00:44<31:16,  2.58s/it, loss=12.0282, global_step=16, toks/s=2926.1, gpu_util=20%, gpu_mem=13.8G, cpu_util=15%, cpu_mem=23.9G]Epoch 0, Step 15, Global Step 16, Loss: 12.0282, Tokens/s: 2926.1, Tokens: 10600, GPU Util: 20%, GPU Mem: 13.8G/15.0G, CPU Util: 15%, CPU Mem: 23.9G/186.7G

GPU Utilization / Memory (all devices):
GPU  Util(%)  Mem(GB Used/Total)
--------------------------------
0        20%   13.8G/ 15.0G
1        44%   12.7G/ 15.0G
2        33%   12.8G/ 15.0G
3        51%   13.5G/ 15.0G
Epoch 0:   2%|▏         | 16/743 [00:44<35:04,  2.89s/it, loss=12.0282, global_step=16, toks/s=2926.1, gpu_util=20%, gpu_mem=13.8G, cpu_util=15%, cpu_mem=23.9G][Rank 0] time (ms) | fwd_microstep: 663.30 | bwd_microstep: 1830.14 | bwd_inner_microstep: 1709.42 | bwd_allreduce_microstep: 120.68 | step_microstep: 0.08
Epoch 0:   2%|▏         | 16/743 [00:46<35:04,  2.89s/it, loss=11.9270, global_step=17, toks/s=3951.6, gpu_util=100%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=23.8G]Epoch 0, Step 16, Global Step 17, Loss: 11.9270, Tokens/s: 3951.6, Tokens: 9856, GPU Util: 100%, GPU Mem: 13.8G/15.0G, CPU Util: 9%, CPU Mem: 23.8G/186.7G

GPU Utilization / Memory (all devices):
GPU  Util(%)  Mem(GB Used/Total)
--------------------------------
0       100%   13.8G/ 15.0G
1       100%   12.7G/ 15.0G
2       100%   12.8G/ 15.0G
3       100%   13.5G/ 15.0G
Epoch 0:   2%|▏         | 17/743 [00:46<33:35,  2.78s/it, loss=11.9270, global_step=17, toks/s=3951.6, gpu_util=100%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=23.8G][Rank 0] time (ms) | fwd_microstep: 661.31 | bwd_microstep: 1847.97 | bwd_inner_microstep: 1721.69 | bwd_allreduce_microstep: 126.24 | step_microstep: 0.08
Epoch 0:   2%|▏         | 17/743 [00:49<33:35,  2.78s/it, loss=12.0117, global_step=18, toks/s=3469.3, gpu_util=100%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=23.8G]Epoch 0, Step 17, Global Step 18, Loss: 12.0117, Tokens/s: 3469.3, Tokens: 8708, GPU Util: 100%, GPU Mem: 13.8G/15.0G, CPU Util: 9%, CPU Mem: 23.8G/186.7G

GPU Utilization / Memory (all devices):
GPU  Util(%)  Mem(GB Used/Total)
--------------------------------
0       100%   13.8G/ 15.0G
1       100%   12.7G/ 15.0G
2       100%   12.8G/ 15.0G
3       100%   13.5G/ 15.0G
Epoch 0:   2%|▏         | 18/743 [00:49<32:36,  2.70s/it, loss=12.0117, global_step=18, toks/s=3469.3, gpu_util=100%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=23.8G][Rank 0] time (ms) | fwd_microstep: 660.58 | bwd_microstep: 1843.06 | bwd_inner_microstep: 1717.07 | bwd_allreduce_microstep: 125.94 | step_microstep: 0.07
Epoch 0:   2%|▏         | 18/743 [00:51<32:36,  2.70s/it, loss=11.9141, global_step=19, toks/s=3109.6, gpu_util=100%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=23.7G]Epoch 0, Step 18, Global Step 19, Loss: 11.9141, Tokens/s: 3109.6, Tokens: 7788, GPU Util: 100%, GPU Mem: 13.8G/15.0G, CPU Util: 9%, CPU Mem: 23.7G/186.7G

GPU Utilization / Memory (all devices):
GPU  Util(%)  Mem(GB Used/Total)
--------------------------------
0       100%   13.8G/ 15.0G
1       100%   12.7G/ 15.0G
2       100%   12.8G/ 15.0G
3       100%   13.5G/ 15.0G
Epoch 0:   3%|▎         | 19/743 [00:51<31:53,  2.64s/it, loss=11.9141, global_step=19, toks/s=3109.6, gpu_util=100%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=23.7G][Rank 0] time (ms) | fwd_microstep: 662.56 | bwd_microstep: 1845.58 | bwd_inner_microstep: 1719.12 | bwd_allreduce_microstep: 126.42 | step_microstep: 0.07
Epoch 0:   3%|▎         | 19/743 [00:54<31:53,  2.64s/it, loss=12.0021, global_step=20, toks/s=3998.4, gpu_util=100%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=23.7G]Epoch 0, Step 19, Global Step 20, Loss: 12.0021, Tokens/s: 3998.4, Tokens: 10032, GPU Util: 100%, GPU Mem: 13.8G/15.0G, CPU Util: 9%, CPU Mem: 23.7G/186.7G

GPU Utilization / Memory (all devices):
GPU  Util(%)  Mem(GB Used/Total)
--------------------------------
0       100%   13.8G/ 15.0G
1       100%   12.7G/ 15.0G
2       100%   12.8G/ 15.0G
3       100%   13.5G/ 15.0G
Epoch 0:   3%|▎         | 20/743 [00:54<31:23,  2.61s/it, loss=12.0021, global_step=20, toks/s=3998.4, gpu_util=100%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=23.7G][Rank 0] time (ms) | fwd_microstep: 661.01 | bwd_microstep: 1845.13 | bwd_inner_microstep: 1718.75 | bwd_allreduce_microstep: 126.34 | step_microstep: 0.07
Epoch 0:   3%|▎         | 20/743 [00:56<31:23,  2.61s/it, loss=11.9422, global_step=21, toks/s=2977.2, gpu_util=100%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=23.6G]Epoch 0, Step 20, Global Step 21, Loss: 11.9422, Tokens/s: 2977.2, Tokens: 7464, GPU Util: 100%, GPU Mem: 13.8G/15.0G, CPU Util: 9%, CPU Mem: 23.6G/186.7G

GPU Utilization / Memory (all devices):
GPU  Util(%)  Mem(GB Used/Total)
--------------------------------
0       100%   13.8G/ 15.0G
1       100%   12.7G/ 15.0G
2       100%   12.8G/ 15.0G
3       100%   13.5G/ 15.0G
Epoch 0:   3%|▎         | 20/743 [00:56<34:08,  2.83s/it, loss=11.9422, global_step=21, toks/s=2977.2, gpu_util=100%, gpu_mem=13.8G, cpu_util=9%, cpu_mem=23.6G]
Epoch 0 - Training Average Loss: 11.9640

Evaluating on validation set...
Validation:   0%|          | 0/77 [00:00<?, ?it/s]Validation:   0%|          | 0/77 [00:00<?, ?it/s, loss=11.9534]Validation:   1%|▏         | 1/77 [00:00<00:50,  1.50it/s, loss=11.9534]Validation:   1%|▏         | 1/77 [00:01<00:50,  1.50it/s, loss=11.9436]Validation:   3%|▎         | 2/77 [00:01<00:50,  1.49it/s, loss=11.9436]Validation:   3%|▎         | 2/77 [00:02<00:50,  1.49it/s, loss=12.0057]Validation:   3%|▎         | 2/77 [00:02<01:15,  1.01s/it, loss=12.0057]
Validation - Avg Loss: 11.9676, Avg Perplexity: 157623.9271

[5/5] Final Evaluation...

Evaluating on test set...
Test:   0%|          | 0/91 [00:00<?, ?it/s]Test:   0%|          | 0/91 [00:00<?, ?it/s, loss=11.9612]Test:   1%|          | 1/91 [00:00<00:59,  1.51it/s, loss=11.9612]Test:   1%|          | 1/91 [00:01<00:59,  1.51it/s, loss=11.9670]Test:   2%|▏         | 2/91 [00:01<00:59,  1.50it/s, loss=11.9670]Test:   2%|▏         | 2/91 [00:02<00:59,  1.50it/s, loss=11.9878]Test:   2%|▏         | 2/91 [00:02<01:29,  1.00s/it, loss=11.9878]
Test - Avg Loss: 11.9720, Avg Perplexity: 158271.2865

================================================================================
Training Complete!
================================================================================
Final Test Loss: 11.9720
Final Test Perplexity: 158271.2865
Total Global Steps: 21
================================================================================
[2026-02-10 17:37:02,112] [INFO] [launch.py:367:main] Process 46741 exits successfully.
[2026-02-10 17:37:03,112] [INFO] [launch.py:367:main] Process 46739 exits successfully.
[2026-02-10 17:37:03,113] [INFO] [launch.py:367:main] Process 46738 exits successfully.
[2026-02-10 17:37:03,113] [INFO] [launch.py:367:main] Process 46740 exits successfully.
