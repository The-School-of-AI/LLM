# P18 SFT Method Decision for the 70B MoE Model

Date: 2026-03-11
Scope: Team 18 SFT stage for the grown-and-expanded 70B MoE
Hardware assumption: AWS `p4de.24xlarge`

## Executive Decision

For p18, we should **not** start with full fine-tuning.

The recommended strategy is:

1. **Primary recommendation: LoRA on a BF16 base model**
2. **Fallback / pressure-release option: QLoRA**
3. **Escalation only after evidence: full fine-tune**

In short: **prefer LoRA over QLoRA on p4de.24xlarge, and prefer both over full fine-tuning unless adapter-based SFT clearly plateaus for the wrong reasons**.

## Final Thesis

The 70B MoE has already gone through capability-building stages: `1B -> 3B MoE -> 7B MoE -> 70B MoE`. That means p18 is not primarily trying to create new base-model competence. It is trying to shape instruction following, response behavior, safety, formatting, and post-training usability.

That is exactly the regime where parameter-efficient fine-tuning is strongest.

The main reason to reject full fine-tuning as the default is not ideology. It is that, on `p4de.24xlarge`, full FT is the worst point on the quality-speed-risk frontier unless we have evidence that:

- the SFT target requires broad representational change rather than behavioral steering,
- adapter-based methods have already plateaued after a fair sweep,
- and the remaining quality gap is large enough to justify much higher systems and operations burden.

The main reason to prefer **LoRA over QLoRA** on this specific hardware is that `p4de.24xlarge` gives us enough memory headroom to avoid quantizing the frozen base unless we need that headroom for sequence length or batch size. Since this is a **70B MoE**, avoiding unnecessary quantization is attractive because MoE systems are more brittle around expert specialization, routing balance, and growth/expansion artifacts than dense models.

So the best first-line answer for p18 is:

- **LoRA first**
- **QLoRA if memory/throughput constraints force it**
- **Full FT only if the evidence says the frozen-base assumption is the real bottleneck**

## Why This Decision Fits P18 Specifically

### 1. The local repo is already structurally optimized for QLoRA/LoRA

The current p18 codebase is not a method-agnostic SFT stack. It is a PEFT-first stack:

- [`README.md`](./README.md) defines the experiment as a QLoRA-based post-training pipeline.
- [`02_sft_training/train_qlora.py`](./02_sft_training/train_qlora.py) always applies PEFT adapters.
- [`02_sft_training/qlora_config.py`](./02_sft_training/qlora_config.py) exposes quantization and LoRA configuration, not a full-FT branch.
- [`03_evaluation/validate_quantization.py`](./03_evaluation/validate_quantization.py) validates the quantized adapter path, not full-model fine-tuning readiness.

This matters because full FT is not just a different hyperparameter choice here. It is a pipeline expansion with new failure modes.

### 2. P18 is a post-training shaping stage, not a second pretraining stage

If the base model were weak or undertrained, full FT would be easier to justify. But the stated growth path implies the opposite: the 70B MoE is supposed to already contain transferred capability from earlier stages. SFT should mostly expose and steer that capability.

That biases the decision toward PEFT.

### 3. MoE models make "possible" and "wise" diverge more sharply

A full fine-tune can in principle update everything: experts, shared layers, and routing-related behavior. That is exactly why it is dangerous as a default for a MoE:

- it can destabilize already-specialized experts,
- it can amplify router imbalance or expert collapse,
- it can turn a post-training run into a broad model retuning event.

LoRA and QLoRA give a narrower control surface. That is often a feature, not a bug.

## Hardware and Memory Analysis for `p4de.24xlarge`

AWS documents `p4de.24xlarge` as providing **8 NVIDIA A100 80GB GPUs** with **640 GB total GPU memory** and high-bandwidth networking. Source: AWS EC2 P4 Instances page.

This changes the LoRA-vs-QLoRA decision materially.

### What this means in practice

- A BF16 70B base model is roughly `70B * 2 bytes ~= 140 GB` just for weights.
- With frozen-base LoRA, the optimizer state is only for adapters, not the full base.
- That makes **plain LoRA on BF16 base weights feasible in principle** on 8x80GB hardware, subject to sequence length, activation memory, routing overhead, and implementation details.

By contrast, **full FT** has to deal with:

- full model weights,
- gradients for the full model,
- optimizer state for the full model,
- activations,
- and distributed sharding complexity.

Using standard mixed-precision Adam-style accounting, full FT on 70B is an **inference from standard optimizer memory formulas** that lands well above 640 GB once parameters, gradients, and optimizer states are counted, before comfortable activation headroom. That means full FT is not merely "large"; it is a **heavy sharding problem** that would require ZeRO-3/FSDP-style partitioning and still be much less forgiving operationally than adapter-based training.

So the hardware conclusion is:

- **p4de is strong enough that we do not need QLoRA by default**
- **p4de is still constrained enough that full FT remains expensive and fragile**

That is why LoRA is the right first choice on this machine class.

## Method-by-Method Evaluation

## Option A: Full Fine-Tune

### Advantages

- Maximum update freedom across all experts and shared layers.
- Best chance of fixing deep representational problems.
- Best chance of changing router-sensitive or expert-allocation-sensitive behavior if those are truly broken.
- Produces a single directly fine-tuned model artifact.

### Disadvantages

- Highest memory and optimizer-state burden.
- Slowest iteration cycle.
- Largest checkpoint and resume burden.
- Highest distributed training fragility.
- Greatest risk of damaging useful expert specialization learned during growth/expansion.
- Requires new pipeline work in p18 before it is even a supported training mode.

### MoE-specific concern

Full FT is the only option that can directly and broadly reshape all expert weights and any routing-adjacent behavior. That is an advantage only if the problem actually demands it. If the issue is instruction following, formatting, helpfulness, or safety behavior, full FT is usually overkill.

### Decision

**Do not use full FT as the default p18 SFT method.**

Treat it as an escalation path only.

## Option B: LoRA

### Advantages

- Keeps the frozen base in higher fidelity than QLoRA.
- Much cheaper and safer than full FT.
- Faster iteration than full FT.
- Better fit for p18's actual objective: behavioral reshaping after pretraining.
- Lower risk of quantization-related distortions in a MoE setting.
- On `p4de.24xlarge`, likely enough memory headroom to be practical without 4-bit base quantization.

### Disadvantages

- Still cannot globally rewrite all model internals.
- If routing/expert specialization itself is wrong, LoRA may not be enough.
- May offer less memory headroom than QLoRA for long contexts, larger batches, or aggressive parallelism.

### MoE-specific view

LoRA is attractive for MoE because it can steer many linear projections while leaving the underlying expert system stable. That is especially important when the 70B checkpoint comes from growth/expansion, where preserving learned specialization may matter more than broad model movement.

### Decision

**This is the recommended first-line method for p18.**

## Option C: QLoRA

### Advantages

- Lowest memory footprint among the three options.
- Highest headroom for larger sequence length, larger effective batch, or safer cluster packing.
- Strong empirical pedigree for very large models.
- Already aligned with the current p18 codebase and documentation.

### Disadvantages

- Quantizes the frozen base, which can introduce some fidelity loss or optimization noise.
- More attractive when memory is scarce than when memory is merely valuable.
- For MoE systems, quantizing many expert-heavy linear layers may be a less conservative move than keeping a BF16 frozen base.

### MoE-specific view

QLoRA is still much safer than full FT for p18, but it is not automatically better than plain LoRA on `p4de.24xlarge`. If the machine already gives enough room for BF16 frozen weights plus adapters, quantization becomes an optimization choice rather than a necessity.

### Decision

**Use QLoRA when memory headroom, context length, or throughput needs justify it.**

It is the correct fallback if LoRA is too memory-hungry for the actual run configuration.

## Why LoRA Beats QLoRA Here

This is the most important non-obvious part of the decision.

If we were on smaller hardware, QLoRA would be the obvious default. But on `p4de.24xlarge`, the hardware gives enough headroom that the memory advantage of QLoRA is no longer automatically worth paying for.

That changes the calculus:

- **QLoRA wins when memory is the gating factor**
- **LoRA wins when memory is sufficient and preserving a cleaner frozen base matters**

For a 70B MoE grown through expansion stages, preserving a cleaner frozen base matters because:

- expert specialization may already be fragile,
- routing patterns may already be hard-won,
- and the SFT objective is more likely to be behavioral than deeply representational.

So the right default is:

- **LoRA unless proven memory-bound**

not:

- **QLoRA unless proven quality-bound**

## Why Full FT Does Not Win the Default Decision

The strongest argument for full FT is this:

> A 70B MoE may need global movement across experts and routing-sensitive behavior, and low-rank adapters may underfit that need.

This is a serious argument, but it still loses as the default because the evidence burden is not met yet.

To justify full FT, we would need to see all of the following:

1. **Adapter-based SFT has been run seriously**
   - LoRA rank sweep
   - QLoRA comparison if needed
   - fair LR sweep
   - same checkpoint
   - same data
   - same evaluation protocol

2. **The residual gap is structurally important**
   - not small benchmark noise,
   - not formatting mistakes,
   - not data/template issues,
   - not prompt-loss masking issues,
   - not checkpoint-choice issues.

3. **The failure pattern points to frozen-base limitations**
   - deep domain adaptation failure,
   - broad semantic mismatch,
   - persistent MoE routing or expert specialization problems,
   - or strong evidence that internal representation changes are needed.

Until then, full FT is paying the highest cost before proving that the extra freedom is necessary.

## MoE-Specific Risk Analysis

## Expert Specialization

Growth-and-expansion pipelines try to preserve or bootstrap specialization across scaling stages. Full FT can disrupt that specialization globally. LoRA and QLoRA are narrower interventions and therefore safer when the goal is to preserve pretrained expert structure while adapting downstream behavior.

## Router Behavior

If the final 70B MoE has router pathologies or expert imbalance, full FT has the best theoretical ability to repair them. But if routing is mostly sound and the problem is user-facing behavior, full FT is too broad.

This yields a clean rule:

- **Router/expert pathology proven** -> consider escalation beyond LoRA
- **Router/expert pathology not proven** -> stay adapter-based

## Growth/Expansion Artifacts

The growth path `1B -> 3B MoE -> 7B MoE -> 70B MoE` creates one special caution:

- some apparent SFT failures may actually be inherited expansion artifacts.

That is the best case for full FT.

But it is still not enough to start there, because:

- checkpoint selection might matter more,
- SFT data quality might matter more,
- masking/template correctness might matter more,
- and adapter rank or target modules might matter more.

In other words: the MoE growth history increases the plausibility of eventually needing more than LoRA, but it does not justify skipping LoRA.

## Practical Decision Rule

Use the following staged policy for p18:

### Stage 1: Default path

Run **LoRA** on the best pretraining checkpoint selected by [`03_evaluation/select_pretrain_checkpoint.py`](./03_evaluation/select_pretrain_checkpoint.py).

Suggested starting posture:

- BF16 base weights
- LoRA on `all-linear` or a deliberately chosen expert/attention subset
- conservative LR sweep
- rank sweep around current defaults

### Stage 2: If memory becomes the binding constraint

Switch to **QLoRA** if one or more of the following happens:

- LoRA cannot sustain the target context length
- effective batch size is too small
- activation + routing overhead causes repeated OOMs
- experiment velocity is materially limited by memory pressure

### Stage 3: Escalation gate for full FT

Only consider **full FT** if:

- LoRA and QLoRA both plateau after fair sweeps,
- the remaining gap is large on the actual target metrics,
- error analysis points to deep model-internal limitations,
- and the expected quality gain justifies the systems cost.

Recommended evidence threshold before opening the full-FT path:

- at least one strong adapter baseline with tuned rank and LR,
- repeatability across seeds or reruns,
- a material gain target, such as a few absolute points on the primary metric or a clear multi-benchmark win,
- and a clear explanation of why the failure is not due to data or evaluation setup.

## What Would Make Me Reverse This Decision

I would revisit the recommendation and consider full FT if any of the following becomes true:

- adapter-based SFT saturates below target despite serious tuning,
- failures cluster around deep domain competence rather than behavior/style,
- MoE router or expert diagnostics show structural imbalance that PEFT cannot correct,
- or the final deliverable absolutely requires a single deeply retuned checkpoint and the cost is acceptable.

I would revisit the LoRA-over-QLoRA preference if:

- real runs show LoRA is memory-constrained on the target context/batch regime,
- QLoRA delivers similar quality with much better throughput,
- or the deployment plan strongly rewards smaller training footprints over fidelity.

## Recommended P18 Plan

1. Keep the current p18 PEFT-first architecture.
2. Choose the best pretraining checkpoint using the checkpoint selector.
3. Run **LoRA-first SFT sweeps** on `p4de.24xlarge`.
4. Use **QLoRA only if memory headroom becomes the limiting factor**.
5. Do not fund full FT until adapter baselines fail for reasons that genuinely implicate frozen-base limits.

## Sources

### External sources consulted

1. AWS EC2 P4 Instances
   - https://aws.amazon.com/ec2/instance-types/p4/
   - Used for `p4de.24xlarge` hardware characteristics: 8x A100 80GB, 640GB aggregate GPU memory.

2. LoRA: Low-Rank Adaptation of Large Language Models
   - https://arxiv.org/abs/2106.09685
   - Used for the core PEFT rationale: large reduction in trainable parameters and memory burden.

3. QLoRA: Efficient Finetuning of Quantized LLMs
   - https://arxiv.org/abs/2305.14314
   - Used for the case that 4-bit adapter-based tuning can preserve strong quality while dramatically reducing memory.

4. Hugging Face Transformers documentation, bitsandbytes quantization
   - https://huggingface.co/docs/transformers/quantization/bitsandbytes
   - Used for current practical guidance on 4-bit/8-bit training constraints and PEFT integration.

5. Hugging Face PEFT documentation
   - https://huggingface.co/docs/peft/index
   - Used for current adapter workflow and deployment/merge considerations.

6. Hugging Face TRL PEFT integration docs
   - https://huggingface.co/docs/trl/peft_integration
   - Used for current SFT training integration assumptions.

### Repo-local sources consulted

- [`README.md`](./README.md)
- [`02_sft_training/README.md`](./02_sft_training/README.md)
- [`02_sft_training/train_qlora.py`](./02_sft_training/train_qlora.py)
- [`02_sft_training/qlora_config.py`](./02_sft_training/qlora_config.py)
- [`02_sft_training/default_config.yaml`](./02_sft_training/default_config.yaml)
- [`03_evaluation/select_pretrain_checkpoint.py`](./03_evaluation/select_pretrain_checkpoint.py)
- [`archive/old_docs/QLORA_QUANTIZATION_APPROACH.md`](./archive/old_docs/QLORA_QUANTIZATION_APPROACH.md)

## One-Sentence Conclusion

For p18 on `p4de.24xlarge`, **LoRA is the best default SFT method for the 70B MoE; QLoRA is the memory-saving fallback; full fine-tuning is an evidence-gated escalation path, not the starting point**.
