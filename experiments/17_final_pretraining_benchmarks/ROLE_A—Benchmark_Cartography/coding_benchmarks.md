# Top-5 Coding Benchmarks (from Excel)
_Generated on 2026-01-29 from `top5_coding_benchmarks.xlsx`._
## Summary (compact)
| Dataset | Metric | Example model scores | Hugging Face | Primary spec / paper |
|---|---|---|---|---|
| SWE-bench Verified | Resolved % (higher is better) | GPT‑5.1‑Codex‑Max: 77.9%; Gemini 3 Pro: 76.2%; Claude Sonnet 4.5: 71.2%; GPT‑5.1: 69.7% | https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified | SWE-bench paper: https://arxiv.org/abs/2310.06770 ; OpenAI SWE-bench Verified intro: https://openai.com/index/introducing-swe-bench-verified/ |
| Terminal-Bench 2.0 | Success rate / accuracy (higher is better) | GPT‑5.1‑Codex‑Max: 58.1%; Gemini 3 Pro: 54.2%; Claude Sonnet 4.5: 35.5%; GPT‑5.1: 30.2% | https://huggingface.co/datasets/penfever/terminal-bench-2 | Terminal-Bench paper: https://arxiv.org/abs/2601.11868 (also OpenReview PDF: https://openreview.net/pdf?id=a7Qa4CcHak) ; Benchmark site: https://www.tbench.ai/ |
| LiveCodeBench (code_generation) | Pass@1 (Easy/Medium/Hard/Total) | From LiveCodeBench paper (code generation, Table 3): GPT‑4‑Turbo‑1106 Total 35.8 (Easy 81.9 / Med 24.5 / Hard 1.1); Claude‑3‑Opus Total 32.5; Google‑Gemini‑Pro Total 19.6 | https://huggingface.co/datasets/livecodebench/code_generation | LiveCodeBench paper: https://arxiv.org/abs/2403.07974 ; Project: https://livecodebench.github.io/ |
| HumanEval | Pass@1 (often reported; sometimes HumanEval+) | From CodeEval‑Pro Table 2 (HumanEval): o1‑mini 97.6; GPT‑4o 90.2; Claude‑3.5‑Sonnet 92.1; DeepSeek‑V2.5 90.2 | https://huggingface.co/datasets/openai/openai_humaneval | HumanEval paper (Codex): https://arxiv.org/abs/2107.03374 ; CodeEval-Pro paper using it: https://aclanthology.org/2025.findings-acl.686.pdf |
| MBPP (Mostly Basic Python Problems) | Pass@1 (often MBPP / MBPP+ variants) | From CodeEval‑Pro Table 2 (MBPP): o1‑mini 93.9; GPT‑4o 86.8; Claude‑3.5‑Sonnet 91.0; DeepSeek‑V2.5 87.6 | https://huggingface.co/datasets/Muennighoff/mbpp | MBPP paper: https://arxiv.org/abs/2108.07732 ; CodeEval‑Pro paper using it: https://aclanthology.org/2025.findings-acl.686.pdf |

---
## Full details (mirrors the Excel sheet)
### SWE-bench Verified
| Field | Value |
|---|---|
| HF dataset link | https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified |
| Primary spec / paper | SWE-bench paper: https://arxiv.org/abs/2310.06770 ; OpenAI SWE-bench Verified intro: https://openai.com/index/introducing-swe-bench-verified/ |
| What it contains | 500 human-validated real GitHub issues (repo snapshots + tests); task is to produce patches that make CI/tests pass. |
| Difficulty (qual + why) | Very hard: even frontier agentic coding models are ~70–80% resolved on Verified (real repos, multi-file edits, tests). |
| Applicable to & why | Primary: agentic software engineering eval (issue → patch passing tests). Secondary: can be used for tool-using SFT/RL on trajectories (high leakage risk). |
| Metric (reported) | Resolved % (higher is better) |
| Example model scores | GPT‑5.1‑Codex‑Max: 77.9%; Gemini 3 Pro: 76.2%; Claude Sonnet 4.5: 71.2%; GPT‑5.1: 69.7% |
| Score source(s) | OpenAI GPT‑5.1‑Codex‑Max eval table: https://openai.com/index/gpt-5-1-codex-max/ ; Gemini 3 Pro eval page: https://deepmind.google/models/gemini/gemini-3/ ; Benchmark intro: https://openai.com/index/introducing-swe-bench-verified/ |
| Where mentioned + justification (exact) | OpenAI: 'human‑validated subset ... more reliably evaluates' (SWE‑bench Verified intro). OpenAI reports SWE‑bench Verified in GPT‑5.1‑Codex‑Max appendix (Model evaluations). DeepMind reports SWE‑bench Verified under 'Agentic coding' for Gemini 3 Pro. |
| Notes / caveats | Keep test split private on official leaderboard; avoid training on it if you want clean evaluation; use sb-cli for submissions. |

**Training / usage validity**

| Pretraining Valid | SFT Valid | RLHF Valid | Agentic Valid | Requires Instruction Tuning |
|---|---|---|---|---|
| No (too small; benchmark; contamination risk) | Yes (patch demonstrations / trajectories) | Yes (reward = tests pass / issue resolved) | Yes (designed for repo+tools workflows) | Usually yes (chat/agent scaffolding helps) |

### Terminal-Bench 2.0
| Field | Value |
|---|---|
| HF dataset link | https://huggingface.co/datasets/penfever/terminal-bench-2 |
| Primary spec / paper | Terminal-Bench paper: https://arxiv.org/abs/2601.11868 (also OpenReview PDF: https://openreview.net/pdf?id=a7Qa4CcHak) ; Benchmark site: https://www.tbench.ai/ |
| What it contains | 82 command-line tasks run in real Linux/Docker-like environments (install/build, debug, security, data tasks). Typically requires multi-step plans + tool use. |
| Difficulty (qual + why) | Extremely hard: long-horizon, environment-dependent tasks; top models ~30–60% success. |
| Applicable to & why | Primary: agentic terminal evaluation (CLI + file edits + running tests). Secondary: RL rollouts & agent training in sandboxed envs. |
| Metric (reported) | Success rate / accuracy (higher is better) |
| Example model scores | GPT‑5.1‑Codex‑Max: 58.1%; Gemini 3 Pro: 54.2%; Claude Sonnet 4.5: 35.5%; GPT‑5.1: 30.2% |
| Score source(s) | OpenAI GPT‑5.1‑Codex‑Max eval table: https://openai.com/index/gpt-5-1-codex-max/ ; Gemini 3 Pro eval page: https://deepmind.google/models/gemini/gemini-3/ ; Dataset card: https://huggingface.co/datasets/penfever/terminal-bench-2 ; Paper: https://openreview.net/pdf?id=a7Qa4CcHak |
| Where mentioned + justification (exact) | Terminal‑Bench paper: evaluates agents on 'hard, realistic tasks in command line interfaces' and lists task types (legacy systems, reimplementing papers, software engineering). OpenAI and DeepMind both report Terminal‑Bench 2.0 in their public model eval tables/pages. |
| Notes / caveats | Terminal‑Bench warns benchmark data should not appear in training corpora; ensure clean separation if used for training. |

**Training / usage validity**

| Pretraining Valid | SFT Valid | RLHF Valid | Agentic Valid | Requires Instruction Tuning |
|---|---|---|---|---|
| No (benchmark; small; environment-specific) | Maybe (if you have expert trajectories; heavy infra) | Yes (natural for RL: binary/graded success) | Yes (explicitly agent/terminal oriented) | Yes (needs tool-using instruction format) |

### LiveCodeBench (code_generation)
| Field | Value |
|---|---|
| HF dataset link | https://huggingface.co/datasets/livecodebench/code_generation |
| Primary spec / paper | LiveCodeBench paper: https://arxiv.org/abs/2403.07974 ; Project: https://livecodebench.github.io/ |
| What it contains | Continuously collected competitive-programming problems (LeetCode/AtCoder/Codeforces) with release dates; supports multiple scenarios (gen, self-repair, exec, test-output). |
| Difficulty (qual + why) | Hard: realistic contest problems; Pass@1 totals are far below HumanEval/MBPP and include hard split headroom. |
| Applicable to & why | Primary: contamination-resistant coding evaluation across time windows; Secondary: SFT/RL for competitive-programming style codegen + repair. |
| Metric (reported) | Pass@1 (Easy/Medium/Hard/Total) |
| Example model scores | From LiveCodeBench paper (code generation, Table 3): GPT‑4‑Turbo‑1106 Total 35.8 (Easy 81.9 / Med 24.5 / Hard 1.1); Claude‑3‑Opus Total 32.5; Google‑Gemini‑Pro Total 19.6 |
| Score source(s) | LiveCodeBench paper HTML (Appendix D, Table 3): https://ar5iv.labs.arxiv.org/html/2403.07974 ; Dataset card: https://huggingface.co/datasets/livecodebench/code_generation |
| Where mentioned + justification (exact) | Paper motivation: existing benchmarks (HumanEval/MBPP) are insufficient and can be contaminated/overfit; Live updates + release dates enable evaluation post cutoff. Appendix D provides the multi-model Pass@1 tables (incl. GPT/Claude/Gemini). |
| Notes / caveats | Leaderboard site is dynamic; ensure you pick the same release window/settings when comparing scores. |

**Training / usage validity**

| Pretraining Valid | SFT Valid | RLHF Valid | Agentic Valid | Requires Instruction Tuning |
|---|---|---|---|---|
| No (benchmark-sized; not broad corpus) | Yes (solutions as supervised targets) | Maybe (reward from unit tests/judging; requires harness) | Maybe (self-repair/execution scenarios resemble tool loops) | Not required (works for base), but helps for chat format |

### HumanEval
| Field | Value |
|---|---|
| HF dataset link | https://huggingface.co/datasets/openai/openai_humaneval |
| Primary spec / paper | HumanEval paper (Codex): https://arxiv.org/abs/2107.03374 ; CodeEval-Pro paper using it: https://aclanthology.org/2025.findings-acl.686.pdf |
| What it contains | 164 Python function-completion problems with unit tests; classic pass@k code-generation eval. |
| Difficulty (qual + why) | Medium (now saturated): frontier models score ~90%+ pass@1; still useful for quick regression. |
| Applicable to & why | Primary: fast regression / sanity check for codegen; Secondary: can be used for SFT/verification datasets, but contamination risk is high. |
| Metric (reported) | Pass@1 (often reported; sometimes HumanEval+) |
| Example model scores | From CodeEval‑Pro Table 2 (HumanEval): o1‑mini 97.6; GPT‑4o 90.2; Claude‑3.5‑Sonnet 92.1; DeepSeek‑V2.5 90.2 |
| Score source(s) | CodeEval‑Pro (ACL Findings 2025) PDF: https://aclanthology.org/2025.findings-acl.686.pdf ; Dataset card: https://huggingface.co/datasets/openai/openai_humaneval |
| Where mentioned + justification (exact) | CodeEval‑Pro explicitly treats HumanEval as a 'fundamental' code-generation benchmark; Table 2 reports pass@1 for many models (incl. GPT‑4o, Claude‑3.5‑Sonnet, DeepSeek‑V2.5, o1‑mini). |
| Notes / caveats | Be careful with training contamination; consider HumanEval+ for stricter tests. |

**Training / usage validity**

| Pretraining Valid | SFT Valid | RLHF Valid | Agentic Valid | Requires Instruction Tuning |
|---|---|---|---|---|
| No (benchmark; tiny; contamination risk) | Maybe (not recommended for clean eval; but used in practice) | No (RLHF not typical here) | No (single-turn completion; not an agent benchmark) | No (completion setting works without instruction tuning) |

### MBPP (Mostly Basic Python Problems)
| Field | Value |
|---|---|
| HF dataset link | https://huggingface.co/datasets/Muennighoff/mbpp |
| Primary spec / paper | MBPP paper: https://arxiv.org/abs/2108.07732 ; CodeEval‑Pro paper using it: https://aclanthology.org/2025.findings-acl.686.pdf |
| What it contains | ~1,000 crowd-sourced Python problems (entry-level) with reference solutions + test cases; often used with 'sanitized' subset. |
| Difficulty (qual + why) | Easy→Medium: less complex than LiveCodeBench/SWE-bench; but still catches basic reasoning & library usage. |
| Applicable to & why | Primary: broad baseline for Python codegen; Secondary: SFT-style training set (but keep held-out split). |
| Metric (reported) | Pass@1 (often MBPP / MBPP+ variants) |
| Example model scores | From CodeEval‑Pro Table 2 (MBPP): o1‑mini 93.9; GPT‑4o 86.8; Claude‑3.5‑Sonnet 91.0; DeepSeek‑V2.5 87.6 |
| Score source(s) | CodeEval‑Pro (ACL Findings 2025) PDF: https://aclanthology.org/2025.findings-acl.686.pdf ; Dataset card: https://huggingface.co/datasets/Muennighoff/mbpp ; Original repo: https://github.com/google-research/google-research/blob/master/mbpp/README.md |
| Where mentioned + justification (exact) | CodeEval‑Pro frames MBPP as a foundational benchmark alongside HumanEval and reports multi-model pass@1 in Table 2. The original MBPP README describes the dataset as ~1k crowd-sourced entry-level problems with tests. |
| Notes / caveats | Use 'sanitized' split for cleaner eval; MBPP can be overfit via public solution leakage. |

**Training / usage validity**

| Pretraining Valid | SFT Valid | RLHF Valid | Agentic Valid | Requires Instruction Tuning |
|---|---|---|---|---|
| No (benchmark-sized) | Yes (commonly used as SFT mix; watch leakage) | Maybe (reward from unit tests possible) | No (single-turn codegen) | Not required (completion setting works) |

---
## Evidence snippets (from the workbook)
| Dataset | Source URL | Where (page/section/lines) | Snippet (<=25 words) | Why it matters |
|---|---|---|---|---|
| SWE-bench Verified | https://openai.com/index/introducing-swe-bench-verified/ | Intro section | human-validated subset ... more reliably evaluates AI models’ ability to solve real-world software issues. | Justification for using Verified (higher reliability). |
| SWE-bench Verified | https://openai.com/index/gpt-5-1-codex-max/ | Appendix: Model evaluations | SWE-bench Verified (n=500) ... 77.9% | OpenAI reports score on this benchmark for Codex-Max. |
| Terminal-Bench 2.0 | https://openreview.net/pdf?id=a7Qa4CcHak | Abstract/Intro | evaluate agents on real tasks in command line interfaces ... configuring legacy systems ... reimplementing research papers. | Defines benchmark scope & task realism. |
| Terminal-Bench 2.0 | https://openai.com/index/gpt-5-1-codex-max/ | Appendix: Model evaluations | Terminal-Bench 2.0 ... 58.1% | OpenAI reports score. |
| LiveCodeBench | https://ar5iv.labs.arxiv.org/html/2403.07974 | Appendix D, Table 3 | Code Generation Performances ... GPT-4-Turbo-1106 ... Total 35.8 | Provides comparable Pass@1 for GPT/Claude/Gemini. |
| HumanEval/MBPP | https://aclanthology.org/2025.findings-acl.686.pdf | Page 4, Table 2 | o1-mini ... HumanEval ... 97.6 ... MBPP ... 93.9 | Multi-model pass@1 for HumanEval & MBPP. |
