# MILU-IN Evaluation

## What is MILU-IN?

MILU (Multi-task Indic Language Understanding) is a benchmark by AI4Bharat for evaluating LLMs on Indian languages. MILU-IN refers to the Indic-only portion — excluding English.

It contains ~80,000 multiple-choice questions spanning 8 domains and 41 subjects, sourced from regional and state-level Indian examinations covering history, science, law, arts, culture, and more.

Dataset: https://huggingface.co/datasets/ai4bharat/MILU

---

## Languages & Test Samples

| Language   | Test Samples |
|------------|-------------|
| Bengali    | 6,637       |
| Gujarati   | 7,089       |
| Hindi      | 14,831      |
| Kannada    | 7,370       |
| Malayalam  | 7,278       |
| Marathi    | 8,197       |
| Odia       | 6,492       |
| Punjabi    | 6,320       |
| Tamil      | 9,007       |
| Telugu     | 7,304       |
| **Total**  | **~80,525** |

---

## Dataset Splits

| Split      | Samples | Purpose                           |
|------------|---------|-----------------------------------|
| test       | ~80,000 | Evaluation — what we use          |
| validation | 8,933   | Few-shot examples (not used here) |

> Note: There is no train split. MILU is evaluation-only.

---

## Repo Structure

```
milu-in/
├── eval.py          ← full pipeline in one file
├── pyproject.toml   ← dependencies
├── README.md
└── results/
    ├── Hindi.json       ← per language: metrics + wrong predictions
    ├── Bengali.json
    ├── ...
    └── summary.json     ← all languages combined, main metrics only
```

---

## Results

Each language JSON contains:
- accuracy, total, correct, avg_latency
- per_subject breakdown
- wrong_predictions (questions the model got wrong)

`summary.json` contains:
- overall accuracy across all 10 languages
- per language: total, correct, accuracy, latency

---

## Setup

**Machine:** AWS g4dn.xlarge — 1x NVIDIA T4 (16GB VRAM), 4 vCPUs, 16GB RAM, 30GB EBS

**Time:** ~2 hours for full MILU-IN (all 10 languages, ~80k questions)

**Requirements:** Python 3.12+, uv

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Setup project
cd milu-in
uv venv
source .venv/bin/activate
uv sync

# Set HuggingFace token (request access at huggingface.co/datasets/ai4bharat/MILU)
echo 'export HF_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

---

## How to Run

```bash
# Smoke test — 50 questions, Hindi only
python eval.py --model Qwen/Qwen2.5-0.5B-Instruct --language Hindi --limit 50

# Single language
python eval.py --model Qwen/Qwen2.5-0.5B-Instruct --language Hindi

# Full MILU-IN — all 10 languages (~2 hours on g4dn.xlarge)
python eval.py --model Qwen/Qwen2.5-0.5B-Instruct

# Your own model
python eval.py --model your-org/your-model-name
```

Run in tmux to avoid SSH disconnects:
```bash
tmux new -s milu
python eval.py --model Qwen/Qwen2.5-0.5B-Instruct
# Detach: Ctrl+B then D
# Reattach: tmux attach -t milu
```