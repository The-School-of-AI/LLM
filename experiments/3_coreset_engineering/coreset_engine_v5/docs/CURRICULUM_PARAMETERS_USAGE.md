# Curriculum parameters: what the pipeline uses

This doc summarizes which parts of `config/curriculum_t3_aligned.yaml` are **actually used** by the coreset selection pipeline vs only loaded for schema/documentation.

**Source:** `coreset_builder.py`, `src/selection/engine.py`, `src/selection/engine_batched.py`, `src/curriculum/loader.py`.

---

## Used in selection / builder

| Curriculum section / field | Where used | Purpose |
|----------------------------|------------|--------|
| **difficulty_system.bands.*.allowed_domains** | Selection engine, coreset_builder | Domain eligibility per band: chunk is selected only if `domain in allowed_domains` for its band. |
| **domains.band_domain_policy** | Loader → `get_allowed_domains_for_band()` | Fallback source for allowed domains per band. |
| **growth_schedule.stage_profiles.*.band_weights** | Coreset builder, selection | Target token share per band per stage (1B, 3B, 8B, 70B). Replaces band_ratios when using stage profiles. |
| **growth_schedule.stage_profiles.*.total_tokens** | Coreset builder | Stage target size (e.g. 20B for 1B). |
| **language_and_context.language_policy** | Selection engine | `get_allowed_languages_for_stage()`, explicitly_excluded; chunks dropped if language not allowed. |
| **guardrails.rolling_window** | Selection engine | `max_band_delta`, `max_domain_delta`, `window_tokens` for smoothness constraints. |
| **difficulty_system.difficulty_centroids** | Coreset builder | Band inference: map difficulty score → band when band is missing. |
| **dataset_interface.input_from_team1.required_fields** | Schema / validation | Declares required input fields (id, text, domain, language, etc.). |

---

## Loaded but not used in selection

| Curriculum section / field | Status |
|----------------------------|--------|
| **difficulty_system.bands.*.allowed_modalities** | Parsed and stored on `BandDefinition`; **not used** for filtering or weighting. Selection does not check modality. |
| **growth_schedule.stage_profiles.*.modality_weights** | Parsed and stored on `StageSpec`; **not used** in selection or builder. No modality-based targets. |
| **modalities** (top-level) | Loaded for schema; **not used** in selection. |
| **difficulty_system.floors** | Loaded; **not used** in current selection logic. |
| **reasoning_policy**, **constraints** (per band) | Stored on band definition; **not used** in selection (reserved for future / other teams). |

So: **modality** (allowed_modalities per band, modality_weights per stage, and the `modalities` block) is **not** used by the pipeline. It is effectively reserved for future use (e.g. modality-stratified selection or reporting). No need to change it for current behavior; you can leave it as-is for documentation or trim it later if you want a minimal config.

---

## Do we need to update anything else?

- **Band–domain policy and band weights:** Already aligned with T3StatsFromT2 and your chosen progression (see `docs/reports/BAND_DOMAIN_POLICY_AND_WEIGHTS_REPORT.md`).
- **Modality:** No update needed for selection; it isn’t used. If you want the YAML to reflect that, add a comment that modality is reserved for future use (see below).
- **Floors:** Not used; safe to leave or adjust for future use.
- **Language policy, rolling window, difficulty_centroids:** In use; update only if you change language caps, smoothness rules, or band inference.

Optional: add a one-line comment in the curriculum above the `modalities` and/or `modality_weights` sections noting they are reserved for future use and not used in selection.
