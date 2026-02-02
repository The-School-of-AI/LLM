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

### 3. General Text Classification (The Core Logic)
For general text, we use a **Map-and-Validate** approach.

#### Step A: Primary Mapping
We take the **Difficulty Level (L0-L5)** from the `DifficultyMetric` and map it to a candidate Band.

| Difficulty Level | Candidate Band |
| :--- | :--- |
| **L0**, **L1** | **B0** (Nursery) |
| **L2** | **B1** (Primary) |
| **L3** | **B2** (High School) |
| **L4** | **B4** (Graduate) |
| **L5** | **B5** (PhD) |

#### Step B: Secondary Validation
The Candidate Band is NOT guaranteed. The sample must **prove** it belongs there by meeting minimum thresholds for **Entropy**, **Diversity**, and **Readability**.

If a sample fails the validation for its Candidate Band, we demote it to the next lower band and check again, recursively.

**Validation Thresholds:**

| Band | Min Grade (FK) | Min Entropy | Min Diversity |
| :--- | :--- | :--- | :--- |
| **B5** | 16.0 | 5.5 | 0.30 |
| **B4** | 14.0 | 5.0 | 0.25 |
| **B3** | 12.0 | 4.5 | 0.20 |
| **B2** | 8.0 | 4.0 | 0.15 |
| **B1** | 4.0 | 3.5 | 0.10 |
| **B0** | 0.0 | 0.0 | 0.00 |

*   *Example*: Text is **L4** (Candidate B4).
    *   It has high difficulty words, BUT it is repetitive (Low Diversity: 0.10).
    *   Check B4: Fails Diversity > 0.25.
    *   Check B3: Fails Diversity > 0.20.
    *   Check B2: Fails Diversity > 0.15.
    *   Check B1: Passes Diversity > 0.10.
    *   **Final Result**: **B1**.

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
