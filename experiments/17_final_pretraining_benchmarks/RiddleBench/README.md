RiddleBench Evaluation Script
==============================
Dataset:  https://huggingface.co/datasets/ai4bharat/RiddleBench
Paper:    https://arxiv.org/abs/2510.24932

Real schema (from HF viewer):
  - id       int64
  - type     string  (4 category values, e.g. "sequence tasks")
  - question string  (already contains full task instructions + the puzzle)
  - answer   string  (free-text, e.g. "5435", "B", "126")

Notes:
  - Only one split: "train" (1,737 rows)
  - No multiple-choice options column — answer is a raw value
  - The question text already embeds the instructions, so we pass it directly

Usage
-----
pip install transformers datasets torch accelerate

python riddlebench_eval.py
python riddlebench_eval.py --model Qwen/Qwen2.5-0.5B-Instruct
python riddlebench_eval.py --limit 50 --verbose
python riddlebench_eval.py --type "sequence tasks" --limit 20
-----

## Examples

==============================================================
  RiddleBench -- Qwen/Qwen2.5-0.5B-Instruct
  1737 items  |  26 correct  |  1.5% accuracy
  Avg latency: 1.09s / item
==============================================================
  Type                         Correct   Total      Acc
  ---------------------------- -------  ------  -------
  blood relations                    0     146     0.0%
  coding and decoding sum            3     169     1.8%
  seating task                       0     432     0.0%
  sequence tasks                    23     990     2.3%
==============================================================
