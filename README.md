<p align="center">
  <img src="./assets/lightninglm-logo.png" alt="LightningLM" width="220" />
</p>

<h1 align="center">LightningLM</h1>

<p align="center">
  <a href="./LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python ≥3.11" src="https://img.shields.io/badge/python-%E2%89%A53.11-blue.svg"></a>
  <a href="#paper"><img alt="Paper" src="https://img.shields.io/badge/arXiv-TBD-b31b1b.svg"></a>
  <a href="https://huggingface.co/theschoolofai/LightningLM-0.1V-120B-MoE"><img alt="Model on HF" src="https://img.shields.io/badge/%F0%9F%A4%97-LightningLM--0.1V--120B--MoE-yellow.svg"></a>
  <a href="https://github.com/The-School-of-AI/LLM"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-The--School--of--AI%2FLLM-181717.svg?logo=github"></a>
</p>

**Reference training pipeline for the LightningLM 0.1V model family.** One architecture, four growth stages: a 2B dense seed grown to 5B MoE, 9B MoE, and a 120B sparse mixture-of-experts trained through TurboQuant-PreTraining (TQP) on a single eight-GPU node.

The 120B model is publicly released on Hugging Face: [**LightningLM-0.1V-120B-MoE**](https://huggingface.co/theschoolofai/LightningLM-0.1V-120B-MoE).

---

## Paper

> **Reversible Foundations: Training a 120B Sparse MoE through State-Preserving Scaling**
> Rohan Shravan. *arXiv preprint arXiv:TBD*, 2026.

The paper is a systems and experience report describing the full training pipeline this repository implements. It documents the three disciplines the work is organized around — reversibility, state-preserving growth, and single-node economics — and the failure modes the recipe is shaped to avoid.

## Released model

| Stage | Parameters (stored / active) | Checkpoint |
|---|---|---|
| 120B sparse MoE | 118.67B / 5.93B (top-12 of 460 routed experts) | [`LightningLM-0.1V-120B-MoE`](https://huggingface.co/theschoolofai/LightningLM-0.1V-120B-MoE) |

The 5B-MoE, 9B-MoE, and 2B-dense intermediate checkpoints from the same training lineage are also planned for public release.

## Quickstart

Install dependencies, run the health check, and launch the 2B seed stage:

```bash
bash scripts/setup_stable.sh
python3 scripts/doctor.py
NUM_GPUS=8 bash scripts/run_2b_stage.sh
```

Grow through 5B, 9B, and launch the 120B TQP stage:

```bash
python3 -m lightninglm.growth.dense_to_moe \
  --src results/2b/checkpoint.pt \
  --dst results/5b/init_from_2b.pt \
  --strategy partition
NUM_GPUS=8 bash scripts/run_5b_stage.sh

python3 -m lightninglm.growth.depth_map \
  --src results/5b/checkpoint.pt \
  --dst results/9b/init_from_5b.pt \
  --mapping lightninglm_5b_to_9b
NUM_GPUS=8 bash scripts/run_9b_stage.sh

python3 scripts/build_120b_init.py \
  --src results/9b/checkpoint.pt \
  --dst results/120b/120b_init.pt \
  --config configs/train_120b_tqp.yaml \
  --ratio 0.5 --router_sigma 0.05 --seed 1337
NUM_GPUS=8 bash scripts/run_120b_tqp.sh
```

The full stage-by-stage workflow lives in [docs/cookbook.md](./docs/cookbook.md).

## Documentation

- [Training cookbook](./docs/cookbook.md) — end-to-end stage-by-stage walkthrough
- [Data pipeline](./docs/data_pipeline.md) — shard preparation, tokenization, manifest generation
- [Tokenizer pipeline](./docs/tokenizer_pipeline.md) — building or adapting the included tokenizer
- [Runtime hot-config](./docs/runtime_hotconfig.md) — operator-side controls for router balance, AON continuation, and 120B bring-up
- [Apache 2.0 license](./LICENSE)

## Repository layout

```text
lightninglm/      model code, training loop, data loading, OPUS, TQP, kernels, growth utilities
configs/          per-stage training and curriculum YAML configs
deepspeed/        DeepSpeed ZeRO configs (zero-1 for 120B TQP, zero-3 for smaller stages)
scripts/          launch scripts, setup, doctor, data and tokenizer tooling, 120B init, tensor hashing
manifests/        curriculum shard manifests (D1-D4 bulk pools, AON guaranteed pools)
tokenizer/        BrahmicTokenizer-131K artifacts and byte-level analysis tools
docs/             training cookbook, data pipeline, tokenizer pipeline, runtime hot-config
data/             local mount points for shard directories (.gitkept placeholders only)
requirements/     pinned dependency manifests
aws/              AWS-specific helpers
experiments/      per-team experiment history (preserved from the project's development)
tests/            test suite for the release pipeline
```

## Companion papers

The LightningLM 0.1V family relies on two companion papers, both implemented in this repository:

- **BrahmicTokenizer-131K** (`./tokenizer/`) - the 131K tokenizer covering English and the major Brahmic scripts. [arXiv:2605.29379](https://arxiv.org/abs/2605.29379).
- **Kronecker Embeddings** (`./lightninglm/models/`) - byte-level structured embeddings that replace the standard 537M-parameter embedding table with a 33.6M Kronecker construction. [arXiv:2605.29459](https://arxiv.org/abs/2605.29459).

## Citation

```bibtex
@article{shravan2026reversible,
  title  = {Reversible Foundations: Training a 120B Sparse MoE through State-Preserving Scaling},
  author = {Shravan, Rohan},
  journal = {arXiv preprint arXiv:TBD},
  year   = {2026},
  url    = {https://github.com/The-School-of-AI/LLM}
}
```

## License

Apache 2.0 - see [LICENSE](./LICENSE). Copyright 2026 Rohan Shravan and The School of AI.

## Contact

- Issues and pull requests: [github.com/The-School-of-AI/LLM/issues](https://github.com/The-School-of-AI/LLM/issues)
- Email: `rshravan@theschoolofai.in`
