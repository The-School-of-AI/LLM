# Band–Domain Policy and Band Weights: Selection Rationale

**Document:** Band-domain policy and stage band weights for T3-aligned curriculum  
**Source data:** `T3StatsFromT2.txt` (complete T3 pool, 2026-03)  
**Curriculum:** `config/curriculum_t3_aligned.yaml`  
**Audience:** Lead / stakeholders / curriculum architects

---

## 1. Executive summary

This report documents the selection of **(1) band–domain policy** (which domains are allowed per difficulty band) and **(2) band weights per stage** (how much of each band to select at 1B, 3B, 8B, 70B). Both are defined in `config/curriculum_t3_aligned.yaml` and are aligned with the T3 pool so that:

- **Band–domain policy:** Every (band, domain) pair observed in the T3 pool is allowed; no in-pool data is excluded by policy. The domain `general_knowledge` is not present in the pool and is omitted entirely from the curriculum.
- **Band weights:** Progression from easier (1B: more B1/B2) to harder (70B: more B3/B4/B5), with all stage profiles summing to 1.0 and respecting the global band distribution in the pool.

---

## 2. Band–domain policy

### 2.1 Principle

For each difficulty band (B0–B5), **allowed_domains** (and `domains.band_domain_policy`) define which domains may be selected. If a (band, domain) pair exists in the pool but is not in the policy, those chunks are never selected and the pool is underused. The policy is therefore set so that:

- Every (band, domain) pair present in `T3StatsFromT2.txt` (section **PER-SOURCE PER-BAND PER-DOMAIN**) is allowed for that band.
- Domains that do **not** appear in the pool for a given band are not included in that band’s allowed list (strict T3Stats alignment).

### 2.2 Allowed domains per band (current policy)

| Band | Name        | Allowed domains |
|------|-------------|-----------------|
| **B0** | Nursery     | web, social, qa, education, language_literacy, conversation, translation, news |
| **B1** | Primary     | web, encyclopedia, news, social, qa, education, language_literacy, conversation, translation, literature, code |
| **B2** | HighSchool  | web, encyclopedia, news, social, qa, education, literature, conversation, translation, code, science, math |
| **B3** | Undergraduate | web, encyclopedia, news, social, qa, education, code, literature, conversation, translation, instruction, preference, science, math |
| **B4** | Graduate    | science, math, code, instruction, education, literature, encyclopedia |
| **B5** | PhD         | instruction, science, math, code, education, preference |

**Notes:**

- **B0:** `language_literacy` appears only in B0 in the pool (erav4_lang_* sources); `news` is present (cc_news B0–B5).
- **general_knowledge:** This domain does not appear in the T3 pool (T3StatsFromT2.txt). It has been removed entirely from the curriculum (all bands and domain_groups).
- **B4:** No `translation`, `social`, `news`, `qa`, or `preference` — the pool has no (B4, domain) pairs for those domains in the stats.
- **B5:** Narrow set (instruction, science, math, code, education, preference) matching pool presence.

### 2.3 Data backup

- **Single source of truth:** `T3StatsFromT2.txt` in the repo.
- **Relevant section:** **PER-SOURCE PER-BAND PER-DOMAIN** — each line is `B<0–5>  <domain>:  docs=<N>  tokens=<M>`.
- Domains observed in the pool (by band): web, news, social, qa, education, language_literacy (B0 only), conversation, translation (B0–B3), literature, code, encyclopedia, instruction, science, math, preference (B3–B5).

---

## 3. Band weights per stage

### 3.1 Principle

- **Progression:** Early stages (1B, 3B) emphasize easier bands (B1, B2); later stages (8B, 70B) shift toward harder bands (B3, B4, B5).
- **Feasibility:** Weights are within or close to the global pool distribution (B0 0.34%, B1 16.99%, B2 41.68%, B3 21.13%, B4 11.87%, B5 7.99%) so selection can meet targets without excluding in-pool data.
- **Normalization:** Each stage profile’s band_weights sum to 1.0.

### 3.2 Band distribution in the T3 pool (reference)

From `T3StatsFromT2.txt` (global, token-weighted):

| Band | Pool share |
|------|------------|
| B0   | 0.34%  |
| B1   | 16.99% |
| B2   | 41.68% |
| B3   | 21.13% |
| B4   | 11.87% |
| B5   | 7.99%  |

### 3.3 Band weights by stage (curriculum)

| Band | 1B (20B tokens) | 3B (40B tokens) | 8B (100B tokens) | 70B (240B tokens) |
|------|-----------------|-----------------|------------------|-------------------|
| B0   | 0.003 | 0.003 | 0.003 | 0.003 |
| B1   | 0.17  | 0.16  | 0.14  | 0.12  |
| B2   | 0.417 | 0.41  | 0.38  | 0.357 |
| B3   | 0.21  | 0.22  | 0.24  | 0.21  |
| B4   | 0.12  | 0.12  | 0.13  | 0.14  |
| B5   | 0.08  | 0.087 | 0.097 | 0.09  |
| **Sum** | **1.0** | **1.0** | **1.0** | **1.0** |

**Stage profiles in config:** `base` (1B), `harder_shift_1` (3B), `harder_shift_2` (8B), `final_adaptive` (70B).

**Rationale:**

- **B0:** Fixed at 0.003 (~0.3%) in all stages; pool has 0.34%, so target is feasible.
- **B1/B2:** Decrease from 1B → 70B (0.17→0.12, 0.417→0.357) to shift capacity toward harder content.
- **B3/B4/B5:** Increase or hold from 1B → 8B (e.g. B3 0.21→0.24, B5 0.08→0.097); 70B slightly reduces B3 (0.21) and raises B4 (0.14) and B5 (0.09) for a harder end profile.
- All weights remain within or close to pool shares so selection can achieve them.

---

## 4. Summary table for lead

| Item | Detail |
|------|--------|
| **Band–domain policy** | Allowed domains per band match (band, domain) pairs observed in T3StatsFromT2.txt; `general_knowledge` omitted entirely (not in pool). |
| **Band weights** | Four stage profiles (1B, 3B, 8B, 70B); progression from more B1/B2 early to more B3/B4/B5 later; each row sums to 1.0. |
| **Source data** | `T3StatsFromT2.txt` — PER-SOURCE PER-BAND PER-DOMAIN and global band distribution. |
| **Config** | `config/curriculum_t3_aligned.yaml`: `difficulty_system.bands.*.allowed_domains`, `growth_schedule.stage_profiles.*.band_weights`, `domains.band_domain_policy`. |
| **Risk** | Low: policy does not exclude in-pool data; weights are feasible given pool distribution. |

---

## 5. References

- **Curriculum:** `config/curriculum_t3_aligned.yaml`
- **Pool stats:** `T3StatsFromT2.txt` (global totals, per-source, per-band, per-source-per-band-per-domain)
- **Related report:** `T3_BAND_DOMAIN_POLICY_REPORT.md` (conversation in B1–B5 and policy rationale)
