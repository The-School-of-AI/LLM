# Band Assignment Logic

The `BandAssignmentMetric` is the **final decision maker** in the curriculum tagging pipeline. It aggregates signals from all other metrics (Difficulty, Modality, Readability, Entropy, Diversity, Structural Density, Tokenizer Level) to assign a definitive curriculum band (**B0** to **B5**).

## Decision Hierarchy

The metric follows a strict hierarchical logic to ensure safety and alignment with the project's complexity rules.

### 1. Hard Overrides (Highest Priority)
Certain signals force specific bands regardless of other metrics to ensure high-stakes content is always correctly categorized.

| Signal | Target Band | Reason / Condition |
| :--- | :--- | :--- |
| **Agentic Traces** | **B5** | Agentic planning and execution traces are always PhD level. |
| **Research Papers** | **B4 / B5** | **B5** if Grade > 16 or Difficulty > 0.8, otherwise **B4**. |
| **Code / Math** | **B2 - B5** | **B5** if Diff > 0.8 or Diversity > 0.4. **B4** if Diff > 0.6. **B3** if Diff > 0.4. Else **B2**. |

### 2. Quality Floors (Safety Nets)
We apply "floors" to prevent complex reasoning from being misclassified as simple due to plain language.

*   **Reasoning Floor**: If `has_reasoning` or `has_cot` is detected in the `modality` tags, the band is forced to be **at least B3**.
    *   *Result*: Any sample with Chain-of-Thought (COT) traces is disqualified from B0, B1, and B2.

### 3. Constraint-Based Classification (The Core Logic)
For general text, we use a **Multi-Constraint Matching** approach based on thresholds defined in `band_assignment.yaml`.

#### Step A: Find Candidate Bands
A sample is a candidate for a band if it meets ALL of the following:

1.  **Difficulty Level**: The sample's L-level (e.g., L2) is in the band's `allowed_difficulty_levels`.
2.  **Modality Check**: The sample's `primary_modality` is in the band's `allowed_modalities` (if defined).
3.  **Metric Ranges**: The sample's stats fall within the band's defined `(min, max)` ranges.
4.  **Tokenizer Level**: If restricted, the `tokenizer_level` must match.

**Current Thresholds (from `band_assignment.yaml`):**

| Band | Allowed Levels | Readability (FK) | Difficulty Score | Entropy | Diversity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B0** | L0, L1 | 0.0 - 6.0 | 0.00 - 0.30 | 0.0 - 4.5 | 0.00 - 0.15 |
| **B1** | L1, L2, L3 | 2.0 - 14.0 | 0.10 - 0.60 | 1.5 - 6.0 | 0.00 - 0.40 |
| **B2** | L2, L3, L4 | 8.0 - 14.0 | 0.40 - 0.70 | 2.5 - 6.0 | 0.10 - 0.35 |
| **B3** | L3, L4 | 12.0 - Inf | 0.60 - 0.85 | 3.0 - Inf | 0.15 - Inf |
| **B4** | L4, L5 | 14.0 - Inf | 0.75 - Inf | 3.5 - Inf | 0.20 - Inf |
| **B5** | L5 | 16.0 - Inf | 0.85 - Inf | 4.0 - Inf | 0.25 - Inf |

> [!NOTE]
> Values above are the defaults. The system also supports `structural_density_range` and `allowed_tokenizer_levels` constraints if enabled in configuration.

#### Step B: Policy Resolution
If a sample qualifies for multiple bands (common between B1 and B2), we use the `overlap_policy`:

*   **Highest (Default)**: Assign the highest qualifying band (e.g., Candidates [B1, B2] -> **B2**).
*   **Lowest**: Assign the lowest qualifying band.

## Configuration

Logic is defined in `BandAssignmentConfig`. Thresholds are primarily loaded from `band_assignment.yaml` located in the project root.

## Output Format

The metric adds a `band_assignment` field to the `curriculum_tags`:

```json
"band_assignment": {
    "band": "B4",
    "reason": "Constraints met: ['B3', 'B4']"
}
```
