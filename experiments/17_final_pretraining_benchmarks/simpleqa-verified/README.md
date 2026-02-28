# SimpleQA Verified Benchmark Evaluation

This script evaluates an LLM (such as Google Gemma 3) against the `google/simpleqa-verified` dataset. It uses **Hugging Face Transformers** for inference and **Google Gemini** as an LLM judge to grade the answers.

## Prerequisites

1.  **Hugging Face Access**: You must accept the model's license agreement on Hugging Face (e.g., [Gemma 3 1B IT](https://huggingface.co/google/gemma-3-1b-it)).
2.  **API Keys**:
    *   **HF_TOKEN**: A Hugging Face access token with read permissions.
    *   **GEMINI_API_KEY**: A Google Gemini API key (obtain from [Google AI Studio](https://aistudio.google.com/app/apikey)).

## Installation

Ensure you have `uv` installed, then install the dependencies:

```bash
uv pip install -r requirements.txt
uv pip install torch transformers accelerate datasets google-genai
```

## Usage

### Dry Run (5 examples)
To quickly verify your setup, run the script on 5 examples:

```bash
HF_TOKEN="your_hf_token" 
GEMINI_API_KEY="your_gemini_key" 
uv run python evaluate.py --model google/gemma-3-1b-it --dry-run --batch-size 1
```

### Full Evaluation
To run the full benchmark (1,000 examples):

```bash
HF_TOKEN="your_hf_token" 
GEMINI_API_KEY="your_gemini_key" 
uv run python evaluate.py --model google/gemma-3-1b-it --output results.json
```

## Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--model` | HuggingFace model ID or local path | (Required) |
| `--device` | Device to use (e.g., `cuda`, `cpu`) | `cuda` (if available) |
| `--gemini-model` | Gemini model used for grading | `gemini-2.5-flash` |
| `--max-samples` | Limit evaluation to N samples | `None` (all) |
| `--batch-size` | Inference batch size | `32` |
| `--max-new-tokens`| Max tokens generated per answer | `128` |
| `--temperature` | Sampling temperature | `0.0` (greedy) |
| `--output` | Path to save JSON results | `results.json` |
| `--dry-run` | Run on 5 examples only | `False` |

## Notes for Tesla T4 Users
This script is optimized for Tesla T4 GPUs by using the Hugging Face backend with `float32`. If you encounter Out-of-Memory (OOM) errors, reduce the `--batch-size` to `1`.
