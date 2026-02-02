
# Steps: Using data with the curriculum generator

This walkthrough shows the minimal workflow to generate a schema-aligned `curriculum.yaml`
using your real data (FineWeb/Dolma/etc.) and the provided tools.

---

## 0) Files you have

- `curriculum_tools.py`  
  - Computes difficulty scores, quantile edges, maps docs -> bands, and computes stage band weights using
    capacity + quantile anchoring + KL regularization.

- `curriculum_yaml_generator.py`  
  - Generates `curriculum.yaml` in the standard schema.
  - Can either use the default stage profiles OR compute band weights from a base distribution.

- `curriculum_validator.py`  
  - Validates the YAML against guardrails.

- `spike_simulator.py`  
  - Simulates rolling-window domain spikes (useful once domain tags are available).

---

## 1) Prepare a calibration sample (cheap)

Goal: compute score quantile edges used to map scores -> bands.

### Recommended
- Sample ~100k documents across all candidate datasets (balanced across sources).
- Keep only the raw `text` field (and optionally `dataset_id`, `domain`, `lang`).

Output: `calibration_sample.jsonl`

Each line:
```json
{"text": "...", "dataset_id": "fineweb_edu", "domain":"general_web", "lang":"en"}
```

---

## 2) Compute quantile edges

Run a small script:

```python
import json
from curriculum_tools import map_doc_to_band, compute_quantile_edges

scores = []
with open("calibration_sample.jsonl","r",encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        out = map_doc_to_band(row, text_key="text", edges=None)
        scores.append(out["score"])

edges = compute_quantile_edges(scores)  # [q15,q30,q50,q70,q85]
print("edges:", edges)
```

Save edges to a file (`band_edges.json`):
```json
{"edges":[...]}
```

---

## 3) Map your full corpus to band + token counts (streaming)

Now run a banding pass on large data (distributed):

Map step (per record):
- load `edges`
- compute band + tokens:
  ```python
  out = map_doc_to_band(sample, edges=edges)
  # out = {"band":"B2","tokens":1234,"modalities":{...}}
  ```

Reduce step:
- aggregate token totals per band (and optionally by dataset_id/domain/lang)
- normalize to compute `base_distribution`

Final output example:
```json
{"B0":0.31,"B1":0.27,"B2":0.20,"B3":0.14,"B4":0.06,"B5":0.02}
```

---

## 4) Generate curriculum.yaml from base distribution

Option A (fast): use default stage profiles.
```bash
python curriculum_yaml_generator.py
# produces curriculum.yaml (edit path in __main__ or call build_curriculum_yaml)
```

Option B (recommended): compute stage band weights from your base distribution
```python
import datetime
from curriculum_yaml_generator import build_curriculum_yaml, dump_yaml

base = {"B0":0.31,"B1":0.27,"B2":0.20,"B3":0.14,"B4":0.06,"B5":0.02}

curr = build_curriculum_yaml(
    frozen_on=datetime.date.today().isoformat(),
    compute_profiles_from_base=True,
    base_distribution=base,
)

dump_yaml(curr, "curriculum.yaml")
```

This uses:
- target median + p75 anchoring
- KL(w || base) regularization
- conservative floors/caps per stage

---

## 5) Validate curriculum.yaml (must pass before training)

```bash
python curriculum_validator.py curriculum.yaml
```

---

## 6) (Optional) Run domain spike simulation

Once your dataloader logs domain tags, either:
- feed the logs to `check_spikes()`, or
- run the built-in synthetic example:

```bash
python spike_simulator.py --window-tokens 2000000 --max-domain-share 0.25
```

---

## Notes for Coreset Teams
- Curriculum provides:
  - stage profiles (band weights + modality weights)
  - floors/caps and earliest-stage constraints
- Coreset team implements:
  - dedup, decontam, and representative selection to meet those targets
