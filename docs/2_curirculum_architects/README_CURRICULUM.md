# This document outlines the strategies and methodologies for arriving at final curriculum

## 1) Bucket Classification Strategies:
The below sections define the various bucket classification strategies which would be used

### Text Feature based methods

| Signal                                    | Strongly correlates with |
| ----------------------------------------- | ------------------------ |
| Avg sentence length                       | Higher difficulty        |
| Paragraph length                          | Higher difficulty        |
| % rare tokens                             | Higher difficulty        |
| Presence of code blocks                   | B3+                      |
| Presence of math symbols                  | B4+                      |
| Markdown structure                        | B2+                      |
| Bullet depth                              | Structured content (B2+) |
| Presence of “step-by-step”, “first…then…” | reasoning → B3+          |
| Presence of JSON/tool schemas             | agentic → B5             |
| URL/domain (e.g. arxiv.org)               | B4/B5                    |
| GitHub repo                               | B3+                      |
| Children's corpora                        | B0                       |
| Flesch-Kincaid grade                      | B0–B2 predictor          |

Example rule sketch:
```python
if contains_tool_schema(text):
    band = B5
elif has_math_symbols and avg_sentence_len > 25:
    band = B4
elif contains_code_blocks:
    band = B3
elif readability_grade > 10:
    band = B2
elif readability_grade > 6:
    band = B1
else:
    band = B0
```

### Dataset Prior Based Methods

| Dataset Type          | Default Band Prior |
| --------------------- | ------------------ |
| Children books        | B0                 |
| Wikipedia intro       | B1                 |
| Wikipedia full        | B2                 |
| StackOverflow answers | B2–B3              |
| GitHub repos          | B3–B5              |
| ArXiv papers          | B4–B5              |
| Math Olympiad         | B5                 |
| Tool-use synthetic    | B5                 |
| News articles         | B1–B2              |

### Reference Implementation

```python
import re
import math
from collections import Counter

# -----------------------------
# Utilities
# -----------------------------

def tokenize_words(text):
    return re.findall(r"\b\w+\b", text.lower())

def split_sentences(text):
    return re.split(r"[.!?]+", text)

# -----------------------------
# Readability (Flesch-Kincaid approx)
# -----------------------------

def estimate_syllables(word):
    return max(1, len(re.findall(r"[aeiouy]+", word.lower())))

def flesch_kincaid_grade(text):
    words = tokenize_words(text)
    sentences = split_sentences(text)
    
    if len(words) < 10 or len(sentences) < 1:
        return 0.0
    
    syllables = sum(estimate_syllables(w) for w in words)
    
    return (
        0.39 * (len(words) / max(1, len(sentences))) +
        11.8 * (syllables / len(words)) -
        15.59
    )

# -----------------------------
# Modality Detection
# -----------------------------

def detect_code(text):
    patterns = [
        r"```",                 # fenced code
        r"\bdef\s+\w+\(",       # Python
        r"\bclass\s+\w+",       # class
        r";\s*$",               # semicolon lines
        r"#include\s+<",        # C/C++
        r"function\s+\w+\(",    # JS
    ]
    return any(re.search(p, text) for p in patterns)

def detect_cot(text):
    cot_markers = [
        "let's think step by step",
        "step by step",
        "first, we",
        "second, we",
        "therefore",
        "reasoning:",
        "chain of thought",
        "thought process"
    ]
    t = text.lower()
    return any(marker in t for marker in cot_markers)

def detect_agentic(text):
    agent_patterns = [
        r'"tool"\s*:',
        r'"action"\s*:',
        r'"observation"\s*:',
        r"Thought:",
        r"Action:",
        r"Observation:",
        r"\{.*\"arguments\".*\}"
    ]
    return any(re.search(p, text) for p in agent_patterns)

# -----------------------------
# Structural Features
# -----------------------------

def avg_sentence_length(text):
    words = tokenize_words(text)
    sentences = split_sentences(text)
    if not sentences:
        return 0
    return len(words) / max(1, len(sentences))

def rare_word_ratio(text):
    words = tokenize_words(text)
    if not words:
        return 0
    
    freq = Counter(words)
    rare = [w for w, c in freq.items() if c == 1]
    return len(rare) / len(words)

def has_math_symbols(text):
    return bool(re.search(r"[∑∫√≈≠≤≥→∞]", text))

# -----------------------------
# Dataset-level priors
# -----------------------------

DATASET_PRIORS = {
    "children_books": "B0",
    "simple_dialogue": "B0",
    "wiki_intro": "B1",
    "news_articles": "B1",
    "textbook": "B2",
    "stackoverflow": "B2",
    "github_code": "B3",
    "api_docs": "B3",
    "arxiv_papers": "B5",
    "math_olympiad": "B5",
}

BAND_ORDER = ["B0", "B1", "B2", "B3", "B4", "B5"]

def band_to_int(b): return BAND_ORDER.index(b)
def int_to_band(i): return BAND_ORDER[max(0, min(5, i))]

# -----------------------------
# Core Band Classifier
# -----------------------------

def classify_band(text, dataset_id=None):
    grade = flesch_kincaid_grade(text)
    sent_len = avg_sentence_length(text)
    rare_ratio = rare_word_ratio(text)

    code = detect_code(text)
    cot = detect_cot(text)
    agent = detect_agentic(text)
    mathy = has_math_symbols(text)

    # Heuristic difficulty
    if agent:
        heuristic = "B5"
    elif mathy and sent_len > 25:
        heuristic = "B4"
    elif code:
        heuristic = "B3"
    elif grade > 10:
        heuristic = "B2"
    elif grade > 6:
        heuristic = "B1"
    else:
        heuristic = "B0"

    # Dataset prior blending
    if dataset_id in DATASET_PRIORS:
        prior = DATASET_PRIORS[dataset_id]
        blended = round((band_to_int(prior) + band_to_int(heuristic)) / 2)
        final_band = int_to_band(blended)
    else:
        final_band = heuristic

    return {
        "band": final_band,
        "contains_code": code,
        "contains_cot": cot,
        "contains_agentic": agent,
        "readability_grade": round(grade, 2),
        "avg_sentence_length": round(sent_len, 2),
        "rare_word_ratio": round(rare_ratio, 3)
    }

# -----------------------------
# Example Run
# -----------------------------

if __name__ == "__main__":
    text = """
    Let's think step by step. First, we define the function.
    ```python
    def add(a, b):
        return a + b
    ```
    Therefore, the complexity is O(n).
    """

    result = classify_band(text, dataset_id="github_code")
    print(result)
```

**Reference papers**:
* Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning: https://arxiv.org/html/2506.11300


## 2) Bucket proportion estimation strategies for Training stages

### A) Model based approach:

Ye et al. (2024) — Data Mixing Laws discovered that model performance is predictable based on data mixture proportions.

**Core idea**: Train small models on a few sample mixtures, measure their performance, then fit mathematical functions (scaling laws) that predict performance on any mixture — without actually training on it.

How to use their approach:

Sample mixtures — Create 10-20 different band ratio combinations (e.g., 30% B0 / 40% B1 / 30% B2, or 20% B0 / 30% B1 / 50% B2, etc.)

Train small proxies — Train tiny models (~100M params) on each mixture for a fixed token budget

Fit scaling functions — Use their regression approach to learn how performance changes with mixture ratios

Predict optimal ratios — Use the fitted function to find the best mixture for your 1B → 3B → 8B → 70B stages before full training

### B) Model Free Approach

**Theoretical Heuristic: Capacity-Difficulty Alignment**

The idea: at each stage, the **median difficulty** of training data should match the model's **absorption capacity**.

**Rule of thumb from literature:**
- Small models (1B): focus on B0-B2, with B3+ as "stretch goals" (5-10%)
- Mid models (3B-8B): shift center to B2-B3, grow B4-B5 gradually
- Large models (70B): can handle full spectrum, but still need B0-B2 for stability

**Proposed formula:**

For stage *s* with capacity *C*, set band *b* weight as:

```
weight(b, s) = base(b) × decay(b, s) × floor_constraint(b, s)
```

Where:
- `base(b)` = natural corpus distribution of band *b*
- `decay(b, s)` = exponential decay for easy bands as *s* increases
- `floor_constraint(b, s)` = minimum exposure to prevent capability gaps

this specific **formula structure** is a synthesis I created based on common patterns across multiple papers, not a single source:

---

**Component origins:**

1. **`base(b)` — natural corpus distribution**
   - From **Xie et al. (DoReMi, 2023)** and **Peng et al. (Topic Over Source, 2025)**: start with empirical data distribution, then adjust

2. **`decay(b, s)` — exponential decay for easy bands**
   - From **Bengio et al. (2009)** curriculum learning: gradually increase difficulty over training
   - Also **Kim et al. (2024)** strategic data ordering: shift from simple to complex

3. **`floor_constraint(b, s)` — minimum exposure**
   - From **arxiv 2511.18903** (LR Decay & Curriculum): warns against completely abandoning easier data too early
   - Also implicit in **OLMo 2 / Phi-4** multi-stage approach: maintain some base distribution

---
---
**Detailed Methodology**

**Capacity-Aligned Growth Model**

Our curriculum uses a **mathematically grounded approach** to determine how data difficulty bands evolve across model scales (1B → 3B → 8B → 70B).

#### 1. Difficulty Quantile Assignment

Each band represents a percentile range of corpus difficulty:

| Band | Percentile | Difficulty Centroid (d_b) | Description |
|------|-----------|---------------------------|-------------|
| B0 | 0–15% | 0.10 | Nursery |
| B1 | 15–30% | 0.225 | Primary |
| B2 | 30–50% | 0.40 | High School |
| B3 | 50–70% | 0.60 | Undergraduate |
| B4 | 70–85% | 0.775 | Graduate |
| B5 | 85–100% | 0.925 | PhD |

#### 2. Model Capacity Scaling

Capacity grows logarithmically with parameters:

```
capacity(stage) = [log(params) - log(1B)] / [log(70B) - log(1B)]
```

| Stage | Capacity |
|-------|----------|
| 1B | 0.00 |
| 3B | 0.26 |
| 8B | 0.49 |
| 70B | 1.00 |

#### 3. Growth Factor Formula

Band sampling weights evolve based on difficulty-capacity alignment:

```
growth_factor(band, stage) = exp(k × (difficulty - capacity))
```

Where k = 3.0 (controls aggressiveness)

**Principle:** When difficulty exceeds capacity → upweight; when below → downweight

#### 4. Derived Decay Rates

Per-stage multipliers computed from the growth model:

| Band | Rate | Effect |
|------|------|--------|
| B0 | 0.75 | decay |
| B1 | 0.82 | decay |
| B2 | 0.92 | mild decay |
| B3 | 1.03 | stable |
| B4 | 1.22 | growth |
| B5 | 1.40 | growth |

#### 5. Constraints

- **Floor minimums** prevent capability gaps (e.g., B0 ≥ 10%)
- **Normalization** ensures weights sum to 100% per stage
- **Guardrails** cap CoT (6%), agentic (3%), Hindi (8%)

---

**Optional proportion refinement based on off the shelf frozen LLMs**

Here's an **objective adjustment formula** based on perplexity measurements:

#### **Step 1: Measure perplexity for each (model_size, band) pair**

Create a matrix:

| Model Size | B0  | B1  | B2  | B3  | B4  | B5  |
|------------|-----|-----|-----|-----|-----|-----|
| 135M       | 15  | 25  | 40  | 65  | 95  | 130 |
| 1.5B       | 8   | 12  | 20  | 35  | 55  | 80  |
| 3B         | 6   | 9   | 14  | 24  | 38  | 58  |
| 8B         | 5   | 7   | 10  | 17  | 27  | 42  |

*(example values)*

---

#### **Step 2: Compute "absorption score" for each band at each stage**

```
absorption(b, s) = 1 - (perplexity(b, s) / max_perplexity(s))
```

Higher score = band is easier for that model size = less weight needed.

---

#### **Step 3: Adjust band weights**

```
adjusted_weight(b, s) = base_weight(b) × (1 - α × absorption(b, s))
```

Where α = 0.3–0.5 (tuning parameter for adjustment strength)

Then normalize to sum to 1.0.

**Example: Adjusting B0 weight for 3B stage**

**Given data:**
- B0 perplexity at 3B model: 6
- Max perplexity across all bands at 3B: 58 (from B5)
- Base weight for B0 (from theory): 22%
- α (adjustment strength): 0.4

**Step 1: Compute absorption score**
```
absorption(B0, 3B) = 1 - (6 / 58) = 1 - 0.10 = 0.90
```

Interpretation: The 3B-sized model has "absorbed" 90% of B0's difficulty — it's very easy for this capacity.

**Step 2: Adjust weight**
```
adjusted_weight(B0, 3B) = 22% × (1 - 0.4 × 0.90)
                        = 22% × (1 - 0.36)
                        = 22% × 0.64
                        = 14.08%
```

**Step 3: After normalization**
When all bands are adjusted and normalized to sum to 100%, B0 might end up at ~18-20% instead of 22%.

---

**Recommended model suite:**

1. **SmolLM2-135M** → proxy for 1B stage
2. **SmolLM2-360M** → proxy for 3B stage  
3. **SmolLM2-1.7B** → proxy for 8B stage
4. **Qwen2.5-3B or Llama-3.2-3B** → proxy for 70B stage

---

**Theoretical Justification**

---

This methodology is grounded in established research on **curriculum learning**, which shows that training models on easier examples before gradually introducing harder ones improves optimization and generalization (Bengio et al., ICML 2009). Bengio et al. connect curriculum learning to **continuation methods**, where learning progresses smoothly from simpler to more complex objectives rather than via abrupt shifts.

Building on this, **competence-based curriculum learning** formalizes training as a function of the relationship between **model competence (capacity)** and **example difficulty** (Platanios et al., ICML 2019). In this framework, data selection is explicitly conditioned on how difficulty compares to current model competence, motivating schedules of the form:

> exposure ∝ f(difficulty − capacity)

Recent theoretical analysis further shows that such structured curricula can improve convergence and generalization under well-defined conditions (Arora, Wang, Zhang, 2025).

**Relationship to Our Growth-Factor Formulation**

Our growth factor follows this principle directly:

> growth_factor(b, s) ∝ exp(k · (d_b − c_s))

where difficulty (d_b) and capacity (c_s) are continuous quantities. The exponential form is a **soft, smooth weighting mechanism** analogous to softmax-based sampling commonly used in ML, and operationalizes the continuation-style progression advocated by Bengio et al. (2009) while respecting the competence–difficulty alignment formalized by Platanios et al. (2019). Although the exact functional form is heuristic, it is a **principled instantiation** of these established theoretical ideas.

---

### Rejection Strategy

Some of the data rejection strategy that can be considered based on the heuristics and also followed by OLMo, DeepSeek, Qwen, and LLaMA:
- **Length**: Reject if <50 words or >100K words (prevents fragments and scraped dumps)
- **Language**: FastText language detection with confidence >0.8 (OLMo/DeepSeek standard)
- **Repetition**: Reject if gzip compression ratio <0.4
- **Token Threshold**: Minimum 200 tokens (~4K characters)



