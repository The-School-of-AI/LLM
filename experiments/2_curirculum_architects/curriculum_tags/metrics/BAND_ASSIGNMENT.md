# Band Assignment Logic

The `BandAssignmentMetric` is the **final decision maker** in the curriculum tagging pipeline. It aggregates signals from all other metrics (Difficulty, Modality, Readability, Entropy, Diversity, COT Scanner) to assign a definitive curriculum band (**B0** to **B5**).

## Decision Hierarchy

The metric follows a strict hierarchical logic to ensure safety and alignment with `curriculum.yaml`.

### 1. Modality Overrides (Highest Priority)
Certain modalities force specific bands regardless of text complexity.

| Signal | Target Band | Reason |
| :--- | :--- | :--- |
| **Agentic Traces** | **B5** | Agentic planning is restricted to PhD level. |
| **Research Papers** | **B4 / B5** | B5 if highly complex (Grade > 16), otherwise B4. |
| **Code / Math** | **B2 - B5** | B5 if Diff > 0.8, B4 if Diff > 0.6, B3 if Diff > 0.4. |

### 2. Quality Floors (Safety Nets)
We apply "floors" to prevents complex concepts from being misclassified as simple due to simple vocabulary.

*   **COT Floor**: If `cot_scanner` detects Chain-of-Thought traces, the band is forced to be **at least B3**.
    *   *Example*: A simple "Let's think step by step" explanation written in B1-level English will be bumped to B3.

### 3. Constraint-Based Classification (The Core Logic)
For general text, we use a **Multi-Constraint Matching** approach. We check if a sample fits the criteria for *any* band, and then apply a **Resolution Policy**.

#### Step A: Find Candidate Bands
A sample is a candidate for a band if it meets ALL of the following:

1.  **Difficulty Level**: The sample's L-level (e.g., L2) is in the band's `allowed_difficulty_levels`.
2.  **Metric Ranges**: The sample's Readability, Difficulty Score, Entropy, and Diversity fall within the band's defined `(min, max)` ranges.

**Default Constraints:**

| Band | Allowed Levels | Readability Range (FK) | Difficulty Score Range | Entropy Range | Diversity Range |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B0** | L0, L1 | 0.0 - 6.0 | 0.0 - 0.30 | 0.0 - 4.5 | 0.00 - 0.15 |
| **B1** | L1, L2, L3 | 4.0 - 10.0 | 0.20 - 0.50 | 3.5 - 5.5 | 0.10 - 0.25 |
| **B2** | L2, L3, L4 | 8.0 - 14.0 | 0.40 - 0.70 | 4.0 - 6.0 | 0.15 - 0.35 |
| **B3** | L3, L4 | 12.0 - Inf | 0.60 - 0.85 | 4.5 - Inf | 0.20 - Inf |
| **B4** | L4, L5 | 14.0 - Inf | 0.75 - Inf | 5.0 - Inf | 0.25 - Inf |
| **B5** | L5 | 16.0 - Inf | 0.85 - Inf | 5.5 - Inf | 0.30 - Inf |

#### Step B: Policy Resolution
If a sample qualifies for multiple bands (e.g., both B1 and B2), we use the configured `overlap_policy`.

*   **Highest (Default)**: Assign the highest band (e.g., B2).
*   **Lowest**: Assign the lowest band (e.g., B1).

#### Example
Sample: **L2**, FK **9.0**, Diff **0.42**.
*   **B1 Check**: L2 allowed. FK 9.0 in [4,10]. Diff 0.42 in [0.2, 0.5]. -> **PASS**
*   **B2 Check**: L2 allowed. FK 9.0 in [8,14]. Diff 0.42 in [0.4, 0.7]. -> **PASS**
*   **B3 Check**: L3 allowed? No. -> **FAIL**
*   **Result**: Candidates [B1, B2]. Policy Highest -> **B2**.

## Configuration

Logic is defined in `BandAssignmentConfig` within `band_assignment.py`. You can override these thresholds via `curriculum_tags.config` if needed, but defaults are hardcoded to match the Curriculum Constitution.

## Output Format

The metric adds a `band_assignment` field to the tags:

```json
"band_assignment": {
    "band": "B4",
    "reason": "Mapped from L4 + validated stats"
}
```
