# Get token counts for Hugging Face datasets


## Datasets and their source URLs

1. https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
2. https://huggingface.co/datasets/cais/mmlu
3. https://huggingface.co/datasets/mandarjoshi/trivia_qa
4. https://huggingface.co/datasets/fingertap/GPQA-Diamond
5. https://huggingface.co/datasets/openai/gsm8k
6. https://huggingface.co/datasets/allenai/ai2_arc
7. https://huggingface.co/datasets/EleutherAI/hendrycks_math
8. https://huggingface.co/datasets/google/IFEval
9. https://huggingface.co/datasets/google/simpleqa-verified
10. https://huggingface.co/datasets/openai/openai_humaneval
11. https://huggingface.co/datasets/opencompass/AIME2025 
12. https://huggingface.co/datasets/nyu-mll/blimp
13. https://huggingface.co/datasets/ai4bharat/indic_glue
14. https://huggingface.co/datasets/ai4bharat/Indic-Bias

## Token counts
Output file is `token_counts.csv` and it is stored in the same directory as `count_hf_tokens.py`

Current one : https://docs.google.com/spreadsheets/d/1QDDvT3BjJWEpIzEBG3kDUAPoj_jOidCg7OqnSaOtI1k/edit?gid=665508499#gid=665508499

## How to run

```bash
uv run count_hf_tokens.py https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
```