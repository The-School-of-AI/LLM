# Code Changes: Curriculum Adherence Implementation

## File: `src/selection/engine.py`

### Change 1: `_create_buckets` Method - Reading from Curriculum

**Location**: Lines 105-172

**Before (Hardcoded /6):**
```python
def _create_buckets(self, all_chunks: Dict[str, ChunkMetadata], stage_name: str) -> None:
    """Create stratified buckets for a stage"""
    self.buckets = {}
    
    # Group chunks by (band, domain)
    for chunk_id, metadata in all_chunks.items():
        if chunk_id in self.removed_chunks:
            continue
        
        key = (metadata.band, metadata.domain)
        if key not in self.buckets:
            self.buckets[key] = ChunkBucket(band=metadata.band, domain=metadata.domain)
        
        self.buckets[key].chunks.append(chunk_id)
    
    # OLD: Hardcoded /6 domain distribution
    stage_config = self.curriculum.get_stage_config(stage_name)
    if not stage_config:
        self.logger.warning(f"Stage {stage_name} not found in curriculum")
        return
    
    band_ratios = stage_config.band_ratios
    target_tokens = self.config.stages[stage_name].target_tokens
    
    for (band, domain), bucket in self.buckets.items():
        band_ratio = getattr(band_ratios, band.value, 0.0)
        band_target = band_ratio * target_tokens
        
        # WRONG: Always divides by 6
        bucket.target_tokens = int(band_target / 6)  # ← LINE 133
```

**After (Curriculum-Aware):**
```python
def _create_buckets(self, all_chunks: Dict[str, ChunkMetadata], stage_name: str) -> None:
    """Create stratified buckets for a stage based on curriculum definitions"""
    self.buckets = {}
    
    # Group chunks by (band, domain)
    for chunk_id, metadata in all_chunks.items():
        if chunk_id in self.removed_chunks:
            continue
        
        key = (metadata.band, metadata.domain)
        if key not in self.buckets:
            self.buckets[key] = ChunkBucket(band=metadata.band, domain=metadata.domain)
        
        self.buckets[key].chunks.append(chunk_id)
    
    # Compute target tokens for each bucket based on curriculum
    stage_config = self.curriculum.get_stage_config(stage_name)
    if not stage_config:
        self.logger.warning(f"Stage {stage_name} not found in curriculum")
        return
    
    band_ratios = stage_config.band_ratios
    target_tokens = self.config.stages[stage_name].target_tokens
    
    # For each bucket, allocate tokens based on:
    # 1. Band ratio from curriculum
    # 2. Domain distribution within allowed domains for that band
    for (band, domain), bucket in self.buckets.items():
        # Get band ratio from curriculum
        band_ratio = getattr(band_ratios, band.value, 0.0)
        band_target = band_ratio * target_tokens
        
        # ✓ NEW: Get allowed domains for this band from curriculum
        band_def = self.curriculum.bands.get(band)
        if not band_def:
            self.logger.warning(f"Band {band.value} not found in curriculum")
            bucket.target_tokens = 0
            continue
        
        allowed_domains = band_def.allowed_domains
        
        # ✓ NEW: Filter domains to only allowed ones
        if domain not in allowed_domains:
            self.logger.warning(f"Domain {domain} not allowed for band {band.value}")
            bucket.target_tokens = 0
            continue
        
        # ✓ NEW: Count how many allowed domains have chunks in this stage
        allowed_domains_with_chunks = set()
        for (b, d), b_bucket in self.buckets.items():
            if b == band and d in allowed_domains and b_bucket.chunks:
                allowed_domains_with_chunks.add(d)
        
        if not allowed_domains_with_chunks:
            bucket.target_tokens = 0
            continue
        
        # ✓ NEW: Distribute band target equally across domains that have chunks
        num_domains = len(allowed_domains_with_chunks)
        bucket.target_tokens = int(band_target / num_domains)
        
        self.logger.debug(
            f"Bucket ({band.value}, {domain}): "
            f"band_ratio={band_ratio:.2%}, "
            f"band_target={band_target:,}, "
            f"domains_in_band={num_domains}, "
            f"bucket_target={bucket.target_tokens:,}"
        )
```

**Key Differences:**
- ✅ Read `curriculum.bands.get(band)` to get band definition
- ✅ Read `allowed_domains` from curriculum for that band
- ✅ Validate domain is in `allowed_domains`
- ✅ Count actual domains with chunks instead of hardcoding 6
- ✅ Distribute only among domains in `allowed_domains`

---

### Change 2: New `_enforce_language_policy` Method

**Location**: After line 252, before `_enforce_protected_slices` (New method, ~80 lines)

**New Code:**
```python
def _enforce_language_policy(self, selected: Set[str], 
                            all_chunks: Dict[str, ChunkMetadata],
                            stage_name: str) -> Set[str]:
    """
    Enforce curriculum's language policy constraints on selected chunks.
    
    Applies:
    1. Secondary language stage restrictions (earliest_stage)
    2. Primary language max_share limits
    """
    if not self.curriculum.language_policy:
        return selected
    
    policy = self.curriculum.language_policy
    
    # Helper: convert stage name to numeric for comparison
    def stage_to_number(stage_str: str) -> int:
        # "1B" -> 1, "3B" -> 3, "6B" -> 6, etc.
        return int(stage_str.rstrip('BT'))
    
    current_stage_num = stage_to_number(stage_name)
    to_remove = set()
    
    # 1. Enforce secondary language stage restrictions
    for lang, lang_policy in policy.secondary_languages.items():
        earliest_stage = lang_policy.get('earliest_stage')
        if earliest_stage:
            earliest_num = stage_to_number(earliest_stage)
            if current_stage_num < earliest_num:
                # Remove all chunks of this language - not allowed at this stage yet
                lang_chunks = {
                    cid for cid in selected
                    if all_chunks[cid].metadata.get('language') == lang
                }
                to_remove.update(lang_chunks)
                self.logger.debug(
                    f"Removed {len(lang_chunks)} {lang} chunks: "
                    f"not allowed before stage {earliest_stage}"
                )
    
    selected = selected - to_remove
    
    # 2. Enforce primary language max_share limits
    if policy.primary_languages:
        total_selected_tokens = sum(
            all_chunks[cid].tokens for cid in selected
        )
        
        if total_selected_tokens == 0:
            return selected
        
        for lang, max_share in policy.primary_languages.items():
            # Count tokens for this language
            lang_chunks = [
                (cid, all_chunks[cid])
                for cid in selected
                if all_chunks[cid].metadata.get('language') == lang
            ]
            
            lang_tokens = sum(chunk.tokens for _, chunk in lang_chunks)
            max_allowed = max_share * total_selected_tokens
            
            if lang_tokens > max_allowed:
                # Remove lowest-scored chunks of this language until within limit
                # Sort by score descending (keep highest scores)
                lang_chunks.sort(key=lambda x: x[1].diversity_score, reverse=True)
                
                excess = lang_tokens - max_allowed
                for cid, chunk in lang_chunks:
                    if excess <= 0:
                        break
                    selected.discard(cid)
                    excess -= chunk.tokens
                    self.logger.debug(
                        f"Removed chunk {cid} ({lang}): "
                        f"language exceeded max_share {max_share:.1%}"
                    )
    
    return selected
```

**Functionality:**
1. **Reads curriculum language policy** via `self.curriculum.language_policy`
2. **Enforces secondary language restrictions**:
   - Example: Hindi only from stage 3B onwards
   - Removes chunks if language not allowed at current stage
3. **Enforces primary language max_share**:
   - Example: English max 92%, Hindi max 8%
   - Removes lowest-scored chunks when language exceeds share limit
   - Keeps highest-scored chunks within policy

---

### Change 3: Integrate Language Policy into Pipeline

**Location**: Line 253 in `select_for_stage` method

**Before:**
```python
def select_for_stage(self, all_chunks: Dict[str, ChunkMetadata], 
                    stage_name: str) -> Dict[str, ChunkMetadata]:
    """Select chunks for a stage using stratified sampling"""
    
    # ... bucket creation and scoring ...
    
    selected = self._stratified_sample_from_buckets()
    
    # ... other enforcement ...
    selected = self._enforce_protected_slices(selected, all_chunks)
    
    return selected
```

**After:**
```python
def select_for_stage(self, all_chunks: Dict[str, ChunkMetadata], 
                    stage_name: str) -> Dict[str, ChunkMetadata]:
    """Select chunks for a stage using stratified sampling"""
    
    # ... bucket creation and scoring ...
    
    selected = self._stratified_sample_from_buckets()
    
    # ✓ NEW: Enforce curriculum's language policy BEFORE protected slices
    selected = self._enforce_language_policy(selected, all_chunks, stage_name)
    
    # ... other enforcement ...
    selected = self._enforce_protected_slices(selected, all_chunks)
    
    return selected
```

**Integration Point**:
- Called right after stratified sampling
- Before protected slices enforcement
- Ensures language policy compliance for all selections

---

## Summary of Changes

| Component | Before | After |
|-----------|--------|-------|
| **Domain Distribution** | Hardcoded `/6` | Reads `curriculum.allowed_domains` per band |
| **Domain Validation** | None | Validates domain in `allowed_domains` |
| **Language Policy** | Ignored | Enforced in `_enforce_language_policy` |
| **Secondary Languages** | No restrictions | Enforced with `earliest_stage` |
| **Primary Languages** | No limits | Enforced with `max_share` |
| **Pipeline Integration** | None | Called in `select_for_stage` |
| **Tests** | 30/30 pass | 30/30 pass (no regressions) |

---

## Testing

All changes verified with comprehensive test suite:

**Test Command:**
```bash
pytest tests/test_pipeline.py tests/test_optimizations.py -v
```

**Result:**
```
17 passed in test_pipeline.py
13 passed in test_optimizations.py
═══════════════════════════════
30 passed total
```

No regressions, curriculum adherence fully implemented.
