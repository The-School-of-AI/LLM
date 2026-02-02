# Profiling Memory and Performance in LLM Training

PyTorch utilities for profiling **memory usage** and **runtime performance** during LLM training.


## Overview

The Memory Profiler helps you understand:

* Where GPU/CPU memory is used
* Which operations are slow
* How memory and time change across training steps

It is designed to plug into existing training loops with minimal changes.

## Features

### Memory Profiling

Tracks:

* CUDA allocated and reserved memory
* CPU memory usage
  Useful for diagnosing OOM errors and memory leaks.

### Performance Profiling

Measures:

* Operator execution time
* Kernel-level timing
  Helps identify slow layers or steps.


### Configurable Profiling Window

Control:

* When profiling starts
* How many steps are profiled
  Avoids overhead during full training.

### Summary Statistics

Provides:

* Per-op time breakdown
* Memory usage summary
* Top memory and time consumers


## Basic Use

### Setup

```python
from profiler import MemoryProfiler, ProfilerConfig

config = ProfilerConfig(
    output_dir="./profiler_logs",
    profile_memory=True,
    active_steps=10
)

profiler = MemoryProfiler(config)
```

### Training Loop Integration

```python
profiler.start()

for step, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    profiler.step()

    if profiler.should_stop(step):
        break

profiler.stop()
profiler.print_summary()
profiler.export_chrome_trace()
```

## Trainer Integration Example

### Initialization

```python
from profiler import MemoryProfiler, ProfilerConfig

if enable_profiling:
    profiler_config = ProfilerConfig(
        output_dir="./profiler_logs",
        profile_memory=True,
        active_steps=10
    )
    self.profiler = MemoryProfiler(profiler_config)
else:
    self.profiler = None
```

### Training Loop

```python
if self.profiler:
    self.profiler.start()

for batch in dataloader:
    # training step

    if self.profiler:
        self.profiler.step()

        if self.profiler.should_stop(self.global_step):
            self.profiler.stop()
            self.profiler.print_summary()
            self.profiler.export_chrome_trace()
            self.profiler = None  # Disable after profiling
```

## Profiler Configuration

`ProfilerConfig` defines how and when profiling is performed. It controls output locations, profiling scope, scheduling, and summary behavior.


## Configuration Class

```python
@dataclass
class ProfilerConfig:
    # Output settings
    output_dir: str = "./profiler_logs"
    tensorboard_dir: Optional[str] = None
    chrome_trace_file: str = "memory_profile.json"

    # Profiling targets
    profile_cpu: bool = True
    profile_cuda: bool = True

    # Memory profiling options
    profile_memory: bool = True
    record_shapes: bool = True
    with_stack: bool = True

    # Scheduling
    wait_steps: int = 5
    warmup_steps: int = 5
    active_steps: int = 10
    repeat: int = 1

    # Summary output
    sort_by: str = "cuda_time_total"
    row_limit: int = 20

    # Optional metrics
    with_flops: bool = False
    with_modules: bool = False
```

## Output Settings

### `output_dir`

Directory where all profiler outputs are written.

### `tensorboard_dir`

Optional directory for TensorBoard logs.
If not set, TensorBoard logging is disabled.

### `chrome_trace_file`

Filename for the exported Chrome trace.


## Profiling Targets

### `profile_cpu`

Enable CPU-side profiling.

### `profile_cuda`

Enable CUDA kernel profiling.


## Memory Profiling Options

### `profile_memory`

Track CPU and CUDA memory usage.

### `record_shapes`

Record tensor shapes for each operation.

### `with_stack`

Capture Python stack traces for each op.
Useful for mapping ops back to source code.

## Scheduling Configuration

Controls **when** profiling runs during training.

### `wait_steps`

Number of initial steps to skip.
Used to avoid profiling unstable early training.

### `warmup_steps`

Profiler runs but does not record data.
Used to stabilize profiler overhead.

### `active_steps`

Steps where profiling data is actively recorded.

### `repeat`

Number of times the schedule is repeated.

## Scheduling Formula

```
total_steps = (wait_steps + warmup_steps + active_steps) * repeat
```

### Example

```
wait_steps   = 5
warmup_steps = 5
active_steps = 10
repeat       = 1
```

**Total profiled steps:** `20`

## Summary Output Settings

### `sort_by`

Metric used to sort profiler results
(e.g. `cuda_time_total`, `cpu_time_total`).

### `row_limit`

Maximum number of rows shown in the summary.

## Optional Metrics

### `with_flops`

Enable FLOPs estimation per operation.

### `with_modules`

Attribute operations to PyTorch modules.


## Viewing Profiling Results

The profiler supports multiple ways to inspect results, depending on the level of detail needed.


## TensorBoard View

TensorBoard provides a high-level view of time and memory usage across steps.

### Start TensorBoard

```bash
tensorboard --logdir=./profiler_logs/tensorboard
```

### Access the Profiler UI

1. Open a browser at `http://localhost:6006`
2. Navigate to the **PYTORCH_PROFILER** tab

This view shows:

* Operator timelines
* CUDA and CPU time breakdown
* Memory usage over steps

## Chrome Trace View

Chrome trace provides a low-level, event-based timeline.

### Steps

1. Open the Chrome browser
2. Go to `chrome://tracing`
3. Click **Load**
4. Select `profiler_logs/memory_profile.json`

This view is useful for:

* Kernel-level timing
* CUDA stream analysis
* CPU–GPU overlap inspection

## Console Summary

The profiler prints a text summary to the console.

The summary includes:

* Top operations by total CUDA time
* Top operations by total CPU time
* Top operations by memory usage

This is useful for quick inspection without external tools.


## Profiling Specific Training Phases

You can explicitly profile different parts of the training step.

## Forward and Backward Sections

```python
with profiler.profile_section("forward_pass"):
    outputs = model(inputs)

with profiler.profile_section("backward_pass"):
    loss.backward()
```

This allows:

* Separate timing for forward and backward passes
* Clear attribution in TensorBoard and summaries


## Stack Trace Export

```python
profiler.export_stacks("stack_trace.txt")
```

Exports Python stack traces for recorded operations.
Useful for tracing expensive ops back to source code.

## Custom Summary Output

```python
profiler.print_summary(
    sort_by="cuda_memory_usage",
    row_limit=50
)
```

Controls:

* Sorting metric for the summary
* Number of operations displayed



## Example Output

```
✓ Memory profiler started
  Output: profiler_logs/base_training_detailed
  TensorBoard: profiler_logs/base_training_detailed/tensorboard

  with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
Step     10/10000 | Loss: 44.8910 | LR: 6.00e-06 | Tok/s: 12,006 | Grad: 2.87 | ETA: 2.4h
Step     20/10000 | Loss: 44.9202 | LR: 1.20e-05 | Tok/s: 11,802 | Grad: 2.73 | ETA: 2.2h
Step     30/10000 | Loss: 44.9663 | LR: 1.80e-05 | Tok/s: 11,620 | Grad: 2.80 | ETA: 2.2h
Step     40/10000 | Loss: 44.9328 | LR: 2.40e-05 | Tok/s: 11,673 | Grad: 2.84 | ETA: 2.2h
✓ Memory profiler stopped after 40 steps

================================================================================
PROFILING SUMMARY
================================================================================

Top 20 operations by CUDA time:
--------------------------------------------------------------------------------
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                          ProfilerStep*         0.00%       0.000us         0.00%       0.000us       0.000us       15.099s       104.04%       15.099s     754.955ms           0 B           0 B           0 B           0 B            20  
                                          ProfilerStep*        40.86%        8.684s        71.37%       15.168s     758.423ms       0.000us         0.00%        5.810s     290.520ms           0 B          16 B           0 B   -1651.32 GB            20  
                                               aten::mm         3.12%     662.593ms         4.56%     969.276ms      23.897us        3.812s        26.27%        3.813s      93.999us           0 B           0 B     301.48 GB     301.47 GB         40560  
                                           aten::linear         0.40%      85.155ms         8.73%        1.856s      68.644us       0.000us         0.00%        2.944s     108.860us           0 B           0 B     369.39 GB      -8.00 MB         27040  
                                            aten::copy_         2.32%     493.622ms         5.47%        1.162s      11.756us        2.733s        18.83%        2.733s      27.639us           0 B           0 B           0 B           0 B         98880  
       autograd::engine::evaluate_function: MmBackward0         0.59%     126.430ms         4.81%        1.022s      75.563us       0.000us         0.00%        2.548s     188.461us           0 B           0 B     -97.81 GB    -301.35 GB         13520  
                                            MmBackward0         0.56%     120.025ms         4.21%     895.177ms      66.211us       0.000us         0.00%        2.548s     188.461us           0 B           0 B     203.54 GB           0 B         13520  
                                           aten::matmul         0.60%     127.891ms         5.72%        1.215s      57.312us       0.000us         0.00%        2.385s     112.506us           0 B           0 B     535.40 GB     -12.56 GB         21200  
                                               aten::to         0.46%      98.329ms         7.27%        1.545s      15.508us       0.000us         0.00%        2.363s      23.720us           0 B           0 B    1204.04 GB           0 B         99640  
                                         aten::_to_copy         1.50%     318.755ms         6.81%        1.447s      20.185us       0.000us         0.00%        2.363s      32.972us           0 B           0 B    1204.04 GB           0 B         71680  
                                              aten::mul         2.66%     564.636ms         4.39%     932.261ms      14.104us        2.235s        15.40%        2.235s      33.810us         712 B         712 B    1165.08 GB    1165.06 GB         66100  
autograd::engine::evaluate_function: SoftmaxBackward...         0.05%      10.928ms         0.32%      68.319ms      35.583us       0.000us         0.00%        1.681s     875.648us           0 B           0 B    -240.00 GB    -480.00 GB          1920  
                                       SoftmaxBackward0         0.03%       7.126ms         0.27%      57.391ms      29.891us       0.000us         0.00%        1.681s     875.648us           0 B           0 B     240.00 GB           0 B          1920  
                           aten::_softmax_backward_data         0.09%      18.413ms         0.24%      50.265ms      26.179us     839.602ms         5.79%        1.681s     875.648us           0 B           0 B     240.00 GB           0 B          1920  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us        1.417s         9.77%        1.417s      33.368us           0 B           0 B           0 B           0 B         42480  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us        1.160s         8.00%        1.160s      65.936us           0 B           0 B           0 B           0 B         17600  
                              Optimizer.step#AdamW.step         0.00%       0.000us         0.00%       0.000us       0.000us        1.148s         7.91%        1.148s      57.398ms           0 B           0 B           0 B           0 B            20  
                              Optimizer.step#AdamW.step         0.30%      63.196ms         0.67%     142.600ms       7.130ms       0.000us         0.00%        1.144s      57.195ms           0 B        -160 B           0 B     -48.95 GB            20  
ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_3...         0.00%       0.000us         0.00%       0.000us       0.000us        1.068s         7.36%        1.068s      79.444us           0 B           0 B           0 B           0 B         13440  
autograd::engine::evaluate_function: ToCopyBackward0...         0.75%     158.771ms         4.20%     892.339ms      30.728us       0.000us         0.00%        1.059s      36.459us           0 B           0 B     218.16 GB    -444.21 GB         29040  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 21.252s
Self CUDA time total: 14.512s


Top 20 operations by CUDA memory:
--------------------------------------------------------------------------------
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                    aten::empty_strided         1.55%     328.448ms         1.55%     328.451ms       4.315us       0.000us         0.00%       0.000us       0.000us           0 B           0 B    1252.99 GB    1252.99 GB         76120  
                                               aten::to         0.46%      98.329ms         7.27%        1.545s      15.508us       0.000us         0.00%        2.363s      23.720us           0 B           0 B    1204.04 GB           0 B         99640  
                                         aten::_to_copy         1.50%     318.755ms         6.81%        1.447s      20.185us       0.000us         0.00%        2.363s      32.972us           0 B           0 B    1204.04 GB           0 B         71680  
                                              aten::mul         2.66%     564.636ms         4.39%     932.261ms      14.104us        2.235s        15.40%        2.235s      33.810us         712 B         712 B    1165.08 GB    1165.06 GB         66100  
                                        ToCopyBackward0         0.20%      43.258ms         3.19%     678.522ms      23.365us       0.000us         0.00%     945.099ms      32.545us           0 B           0 B     662.37 GB           0 B         29040  
                                           aten::matmul         0.60%     127.891ms         5.72%        1.215s      57.312us       0.000us         0.00%        2.385s     112.506us           0 B           0 B     535.40 GB     -12.56 GB         21200  
                                           MulBackward0         0.32%      68.938ms         2.10%     445.297ms      23.001us       0.000us         0.00%     781.645ms      40.374us           0 B           0 B     470.00 GB           0 B         19360  
                                           aten::linear         0.40%      85.155ms         8.73%        1.856s      68.644us       0.000us         0.00%        2.944s     108.860us           0 B           0 B     369.39 GB      -8.00 MB         27040  
                                              aten::add         0.65%     138.130ms         1.06%     225.538ms      16.657us     619.993ms         4.27%     619.995ms      45.790us           0 B           0 B     337.53 GB     337.53 GB         13540  
                                               aten::mm         3.12%     662.593ms         4.56%     969.276ms      23.897us        3.812s        26.27%        3.813s      93.999us           0 B           0 B     301.48 GB     301.47 GB         40560  
                                              aten::bmm         1.02%     217.233ms         1.40%     297.770ms      25.848us     981.916ms         6.77%     981.916ms      85.236us           0 B           0 B     300.00 GB     300.00 GB         11520  
                                          aten::softmax         0.05%      10.452ms         0.19%      41.373ms      21.548us       0.000us         0.00%     566.200ms     294.896us           0 B           0 B     240.00 GB           0 B          1920  
                                         aten::_softmax         0.09%      19.141ms         0.14%      30.377ms      15.821us     566.200ms         3.90%     566.200ms     294.896us           0 B           0 B     240.00 GB     240.00 GB          1920  
                                       SoftmaxBackward0         0.03%       7.126ms         0.27%      57.391ms      29.891us       0.000us         0.00%        1.681s     875.648us           0 B           0 B     240.00 GB           0 B          1920  
                           aten::_softmax_backward_data         0.09%      18.413ms         0.24%      50.265ms      26.179us     839.602ms         5.79%        1.681s     875.648us           0 B           0 B     240.00 GB           0 B          1920  
autograd::engine::evaluate_function: ToCopyBackward0...         0.75%     158.771ms         4.20%     892.339ms      30.728us       0.000us         0.00%        1.059s      36.459us           0 B           0 B     218.16 GB    -444.21 GB         29040  
                                            aten::empty         0.49%     104.843ms         0.49%     104.852ms       4.432us       0.000us         0.00%       0.000us       0.000us         160 B         160 B     211.82 GB     211.79 GB         23660  
                                            MmBackward0         0.56%     120.025ms         4.21%     895.177ms      66.211us       0.000us         0.00%        2.548s     188.461us           0 B           0 B     203.54 GB           0 B         13520  
                                           BmmBackward0         0.14%      28.711ms         1.14%     242.352ms      63.112us       0.000us         0.00%     670.139ms     174.515us           0 B           0 B     165.00 GB           0 B          3840  
                                       aten::empty_like         0.13%      28.449ms         0.51%     109.378ms       7.011us       0.000us         0.00%       0.000us       0.000us           0 B           0 B     127.84 GB           0 B         15600  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 21.252s
Self CUDA time total: 14.512s

================================================================================

✓ Stack traces exported to: profiler_logs/base_training_detailed/stack_trace.txt
```
