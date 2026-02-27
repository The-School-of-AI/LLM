# Walkthrough: Metadata-Aware Difficulty Scoring

I have successfully updated the curriculum metrics calculator to capture and leverage explicit metadata information for more accurate difficulty scoring.

## Changes Made

### [t2_metrics_calculator_v5.py](file:///Users/hemanthk/Desktop/llms/capstone/LLM/experiments/2_curirculum_architects/glue_jobs/t2_metrics_calculator_v5.py)

- **Metadata Extraction**: Implemented logic to extract "Difficulty", "Grade", and "Student Level" from both the JSON [metadata](file:///Users/hemanthk/Desktop/llms/capstone/LLM/experiments/2_curirculum_architects/glue_jobs/t2_metrics_calculator_v5.py#217-222) column and embedded text markers (e.g., `### Metadata: Difficulty: Hard`).
- **Structural Signals**: Added a boost for educational structural markers (`### Explanation:`, `### Question:`, `### Answer:`) in the `cot_score`.
- **Score Mapping**: Created a robust mapping for qualitative signals:
  - `Hard` -> 0.8
  - `Grade 11+` -> 0.8
  - `Expert/Advanced` -> 0.95
- **Weighted Blending**: Updated the [difficulty_score](file:///Users/hemanthk/Desktop/llms/capstone/LLM/experiments/2_curirculum_architects/glue_jobs/t2_metrics_calculator_v5.py#593-677) to blend metadata-based signals (70% weight) with existing heuristics (30% weight) when metadata is present.

## Verification Results

I verified the fix using the provided Quantum Mechanics NCERT example.

| Metric | Before Fix | After Fix |
| :--- | :--- | :--- |
| **Heuristic Score** | 0.3571 | 0.4071 (Boosted) |
| **Metadata Signal** | N/A (Ignored) | 0.8000 |
| **Final Difficulty Score** | **0.3571** | **0.6821** |
| **Projected Band** | B2 | **B4** |

> [!NOTE]
> The text is now correctly categorized in the **B4 (Advanced)** range, aligning with its "Grade 11" and "Hard" qualitative labels.

### Code Snippet: New Extraction Logic

```python
# Extraction from text block (embedded)
df = df.withColumn("meta_diff_text", F.lower(F.regexp_extract(F.col("text"), METADATA_DIFFICULTY_PATTERN, 1)))
df = df.withColumn("meta_grade_text", F.regexp_extract(F.col("text"), METADATA_GRADE_PATTERN, 1))

# Final Blend: If metadata exists, it carries 70% weight
df = df.withColumn("difficulty_score",
    F.when(F.col("_metadata_base").isNotNull(), 
           (F.lit(0.7) * F.col("_metadata_base") + F.lit(0.3) * F.col("_heuristic_score")))
    .otherwise(F.col("_heuristic_score"))
)
```
