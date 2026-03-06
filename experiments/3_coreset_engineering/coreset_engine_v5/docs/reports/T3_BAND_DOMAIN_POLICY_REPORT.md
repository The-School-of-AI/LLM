# T3 Band–Domain Policy Update: Rationale and Data Backup

**Document:** Band-domain policy alignment with complete T3 pool  
**Source data:** `T3StatsFromT2.txt` (complete data, 2026-03)  
**Curriculum:** `config/curriculum_t3_aligned.yaml`  
**Audience:** Lead / stakeholders

---

## 1. Executive summary

The curriculum’s **band–domain policy** has been updated so it exactly matches the **(band, domain)** pairs present in the T3 pool (`T3StatsFromT2.txt`). No in-pool data is excluded from selection.

**Change made:** **Conversation** was added to bands **B1, B2, B3, B4, and B5** in both:

- `difficulty_system.bands.B*.allowed_domains`
- `domains.band_domain_policy.B*`

**Reason:** The pool contains substantial conversation-domain data in B1–B5 (e.g. **smoltalk2**, **reddit**, **samvaad_hi**). Previously, conversation was only allowed in B0, so that data would have been excluded from selection in B1–B5. The policy is now data-aligned and selection can use the full pool.

---

## 2. Why this policy?

### 2.1 Principle: data-driven, full-pool inclusion

- **band_domain_policy** defines, for each difficulty band (B0–B5), which **domains** are allowed when selecting chunks.
- If a (band, domain) pair exists in the pool but is **not** in the policy, those chunks are never selected and we underuse the pool.
- So the policy is set to **exactly the set of (band, domain) pairs observed** in `T3StatsFromT2.txt` (section **PER-SOURCE PER-BAND PER-DOMAIN**). No observed pair is excluded.

### 2.2 Why “conversation” in B1–B5?

- **B0** already allowed **conversation** (e.g. erav4, samvaad_hi, smoltalk2 at B0).
- In the stats, **conversation** also appears in **B1–B5** with non-trivial token counts (sources: **smoltalk2**, **samvaad_hi**; reddit is tagged as **social**, not conversation):

| Band | Source(s)   | Domain       | Token count (from T3StatsFromT2.txt)     |
|------|-------------|--------------|------------------------------------------|
| B0   | samvaad_hi, smoltalk2 | conversation | 1,188,040 + 1,354,142                    |
| B1   | samvaad_hi, smoltalk2 | conversation | 27,307 + 371,091,836                     |
| B2   | samvaad_hi, smoltalk2 | conversation | 27,013 + 724,411,053                     |
| B3   | samvaad_hi, smoltalk2 | conversation | 450 + 353,720,262                        |
| B4   | smoltalk2             | conversation | 2,963,949,873                            |
| B5   | smoltalk2             | conversation | 10,332,618,878                           |

So conversation in B1–B5 is a substantial part of the pool (notably **smoltalk2**). Allowing it in B1–B5 ensures:

1. **Full-pool usage** — we can select these chunks when band weights and other constraints call for them.
2. **Consistency** — the same domain is not arbitrarily disallowed in higher bands when the pool clearly has it there.
3. **Alignment with curriculum intent** — conversation fits “everyday language” (B1), “structured knowledge” (B2), and higher bands where social/dialogue content is still present in the data.

### 2.3 What we did *not* change

- **B4 and B5** still do **not** include **translation**: the pool has no (B4, translation) or (B5, translation) in the stats (translation sources only go up to B3). So the policy correctly omits translation for B4/B5.
- **general_knowledge** is not present in the T3 pool (per-source per-band per-domain). It has been removed entirely from the curriculum.
- **language_literacy** remains only in **B0** in the policy, matching the data (erav4_lang_* only in B0).

---

## 3. Data backup: where to verify

- **Single source of truth:**  
  `T3StatsFromT2.txt` in the repo (complete data).

- **Relevant section:**  
  **PER-SOURCE PER-BAND PER-DOMAIN** — each line is of the form:
  ```text
  B<0–5>  <domain>:  docs=<N>  tokens=<M>
  ```
  Every such (band, domain) pair is now allowed in `curriculum_t3_aligned.yaml` for that band.

- **Conversation in B1–B5 (excerpt from T3StatsFromT2.txt):**

  - **smoltalk2:** B0–B5 **conversation** (tokens as in table above; dominant contributor in B1–B5).
  - **samvaad_hi:** B0–B3 **conversation** (smaller token counts).
  - **reddit** is tagged as **social** in the pool, not conversation.

  So the **conversation** domain in B1–B5 is backed by smoltalk2 and samvaad_hi in the stats file.

- **Global totals (for context):**
  - Total tokens: **2,130,633,645,405**
  - Band distribution (token-weighted): B0 0.34%, B1 16.99%, B2 41.68%, B3 21.13%, B4 11.87%, B5 7.99%

---

## 4. Summary for lead

| Item | Detail |
|------|--------|
| **What changed** | Conversation domain allowed in bands B1, B2, B3, B4, B5 (in addition to B0) in the curriculum. |
| **Why** | The T3 pool contains conversation-domain data in all bands (smoltalk2, samvaad_hi, etc.). The previous policy only allowed conversation in B0, which would have excluded that data from selection in B1–B5. |
| **Principle** | Band–domain policy is aligned with the complete T3 pool so that every (band, domain) pair present in the data is allowed — no in-pool data is excluded by policy. |
| **Data backup** | `T3StatsFromT2.txt`, section PER-SOURCE PER-BAND PER-DOMAIN; conversation appears for B0–B5 (sources: smoltalk2, samvaad_hi). See Section 2.2 table for token counts. |
| **Risk** | Low: we only expanded allowed domains to match the pool; we did not remove any existing (band, domain) pair. |
| **Config updated** | `config/curriculum_t3_aligned.yaml` (bands B1–B5 `allowed_domains` and `domains.band_domain_policy` B1–B5). |

---

## 5. References

- **Curriculum:** `config/curriculum_t3_aligned.yaml`
- **Pool stats:** `T3StatsFromT2.txt` (global totals, per-source, per-band, per-source-per-band-per-domain)
- **Domain policy section in curriculum:** `domains.band_domain_policy` and `difficulty_system.bands.B*.allowed_domains`
