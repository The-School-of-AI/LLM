# Curriculum Architects — Output Schema (CSV)

Canonical schema for the **parsable flat output** and **rejected-files log** produced by the tagging pipeline.

**Schema version:** `v1`

---

## 1. Main output CSV (parsable file)

**File naming:** `{output_prefix}.csv` (e.g. `output.csv`, or same base name as Parquet with `.csv`).

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `uuid` | string (UUID) | yes | Unique per row (e.g. UUID4). |
| `id` | string | yes | Document/sample ID from input (Team 1). |
| `file_path` | string | yes | S3 or local path of source file (or row origin). |
| `band` | string | yes | Curriculum band, e.g. B0–B5 (from band_assignment). |
| `band_reason` | string | no | Short reason for band (from band_assignment.reason). |
| `difficulty_level` | string | no | L0–L5 (from difficulty.level). |
| `difficulty_score` | float | no | 0.0–1.0 (from difficulty.score). |
| `readability_fk_grade` | float | no | Flesch-Kincaid grade (from readability). |
| `primary_modality` | string | no | text, code, math, etc. (from modality). |
| `tokenizer_level` | string | no | T0–T5 (from tokenizer_difficulty.level). |
| `entropy_score` | float | no | From entropy.score. |
| `structural_density` | float | no | From structural_density.structural_density. |
| `has_cot` | bool | no | From cot_scanner.has_cot. |
| `has_agentic` | bool | no | From cot_scanner.has_agentic. |
| `checksum` | string | yes | SHA-256 of normalized text (hex). |
| `minhash` | string | no | Reserved; empty for v1. |
| `optional_1` | string | no | Reserved for future metrics. |
| `optional_2` | string | no | Reserved. |
| `optional_3` | string | no | Reserved. |
| `schema_version` | string | yes | e.g. `v1`. |

**Example row:**

```csv
uuid,id,file_path,band,band_reason,difficulty_level,difficulty_score,readability_fk_grade,primary_modality,tokenizer_level,entropy_score,structural_density,has_cot,has_agentic,checksum,minhash,optional_1,optional_2,optional_3,schema_version
a1b2c3d4-e5f6-7890-abcd-ef1234567890,sample_0,path/to/input.parquet,B1,constraint_match,L1,0.25,5.2,text,T1,3.8,0.02,false,false,a3f2b1c...,,,,v1
```

---

## 2. Rejected-files log CSV

**File naming:** `{output_prefix}_rejected.csv`.

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `uuid` | string (UUID) | yes | Unique per rejected row. |
| `id` | string | yes | Document ID from input (empty if unknown). |
| `file_path` | string | yes | Source file path. |
| `reason` | string | yes | One of: `parse_error`, `empty_text`, `metric_failed`, `band_assignment_failed`. |
| `details` | string | no | Full error message or stack trace. |
| `schema_version` | string | yes | e.g. `v1`. |

**Example row:**

```csv
uuid,id,file_path,reason,details,schema_version
c3d4e5f6-a7b8-9012-cdef-123456789012,sample_42,path/to/input.parquet,metric_failed,"TokenizerDifficultyMetric: tokenizer not found",v1
```

---

## 3. Rejection reasons (enum)

| Value | When used |
|-------|-----------|
| `parse_error` | Missing/invalid required field (e.g. no `text`). |
| `empty_text` | Text empty or too short. |
| `metric_failed` | A metric plugin raised or returned error. |
| `band_assignment_failed` | Band assignment missing or error. |

---

## 4. Optional slots

New metrics that are not yet in the fixed schema can use `optional_1`, `optional_2`, `optional_3` until the schema is extended. When adding a new fixed column, bump schema version and document in this file.
