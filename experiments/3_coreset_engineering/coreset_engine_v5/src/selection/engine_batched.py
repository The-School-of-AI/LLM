"""
Enhanced selection engine with batch processing support for 2T+ token scale.
Extends SelectionEngine with streaming deduplication and checkpoint-aware selection.
"""

import logging
import random
import time
from collections import Counter, defaultdict, deque
from typing import Any, Deque, Dict, Generator, Iterable, List, Optional, Set, Tuple

import numpy as np

from ..core.types import ChunkMetadata, DifficultyBand, difficulty_band_order
from .engine import SelectionEngine

logger = logging.getLogger(__name__)


class BatchedSelectionEngine(SelectionEngine):
    """
    Selection engine optimized for batch processing of large datasets.

    Features:
    - Processes chunks in streaming batches (no full-file load)
    - Applies deduplication at batch boundaries
    - Maintains selection state across batch restarts
    - Checkpoints after each batch for fault tolerance
    """

    def select_for_stage_batched(
        self,
        chunk_stream: Iterable[Tuple[str, ChunkMetadata]],
        stage_name: str,
        batch_size: int = 10_000,
        protected_slices: Optional[List] = None,
        checkpoint_callback=None,
        total_input_tokens_estimate: Optional[int] = None,
        target_tokens_override: Optional[int] = None,
    ) -> Tuple[Set[str], Dict[str, Any]]:
        """
        Select coreset from streaming chunk generator.

        Args:
            chunk_stream: Generator yielding (chunk_id, metadata) tuples
            stage_name: Name of training stage
            batch_size: Chunks per batch for processing
            protected_slices: Protected slice rules
            checkpoint_callback: Function(batch_num, selected, stats) for checkpointing

        Returns:
            (selected_chunk_ids, selection_stats)
        """
        logger.info(f"Starting batched selection for stage: {stage_name}")

        # Validate curriculum
        curriculum_valid, errors = self.curriculum.validate_deterministic_guarantees()
        if not curriculum_valid:
            raise ValueError(f"Curriculum validation failed: {errors}")

        selected: Set[str] = set()
        batch_num = 0
        total_chunks = 0
        total_tokens = 0
        selected_tokens = 0

        # Determine stage target for this run (supports sharded/partial runs)
        stage_target_tokens = (
            int(target_tokens_override)
            if target_tokens_override is not None
            else int(self.config.stages[stage_name].target_tokens)
        )

        try:
            # Process in streaming batches
            batch: List[Tuple[str, ChunkMetadata]] = []
            for chunk_id, metadata in chunk_stream:
                batch.append((chunk_id, metadata))

                if len(batch) >= batch_size:
                    # Process batch
                    selected_in_batch, batch_stats = self._process_batch(
                        batch,
                        stage_name,
                        protected_slices,
                        total_input_tokens_estimate=total_input_tokens_estimate,
                        stage_target_tokens=stage_target_tokens,
                    )
                    selected.update(selected_in_batch)

                    total_chunks += len(batch)
                    total_tokens += batch_stats["batch_tokens"]
                    selected_tokens += batch_stats["batch_selected_tokens"]

                    logger.info(
                        f"Batch {batch_num}: {len(batch)} chunks, "
                        f"{batch_stats['batch_tokens']:,} tokens, "
                        f"selected so far: {len(selected)}"
                    )

                    # Checkpoint
                    if checkpoint_callback:
                        checkpoint_callback(
                            batch_num,
                            selected,
                            {
                                "total_chunks": total_chunks,
                                "total_tokens": total_tokens,
                                "selected_chunks": len(selected),
                                "selected_tokens": selected_tokens,
                            },
                        )

                    batch = []
                    batch_num += 1

            # Process final batch
            if batch:
                selected_in_batch, batch_stats = self._process_batch(
                    batch,
                    stage_name,
                    protected_slices,
                    total_input_tokens_estimate=total_input_tokens_estimate,
                    stage_target_tokens=stage_target_tokens,
                )
                selected.update(selected_in_batch)

                total_chunks += len(batch)
                total_tokens += batch_stats["batch_tokens"]
                selected_tokens += batch_stats["batch_selected_tokens"]

                logger.info(
                    f"Final batch {batch_num}: {len(batch)} chunks, "
                    f"{batch_stats['batch_tokens']:,} tokens, "
                    f"total selected: {len(selected)}"
                )

                if checkpoint_callback:
                    checkpoint_callback(
                        batch_num,
                        selected,
                        {
                            "total_chunks": total_chunks,
                            "total_tokens": total_tokens,
                            "selected_chunks": len(selected),
                            "selected_tokens": selected_tokens,
                        },
                    )

        except Exception as e:
            logger.error(f"Error during batched selection: {e}", exc_info=True)
            raise

        # Minimal stats (global exact stats require re-loading metadata, which is not feasible at 2T scale)
        stats = {
            "total_chunks_seen": total_chunks,
            "total_tokens_seen": total_tokens,
            "selected_chunks": len(selected),
            "selected_tokens": selected_tokens,
            "batches_processed": batch_num + (1 if total_chunks > 0 else 0),
        }

        logger.info(
            f"Batched selection complete. batches={stats['batches_processed']}, "
            f"selected_chunks={stats['selected_chunks']:,}, selected_tokens={stats['selected_tokens']:,}"
        )

        return selected, stats

    def get_rolling_window_stats(self) -> Dict[str, float]:
        """Return rolling-window enforcement summary for the active stage."""
        if not getattr(self.config.curriculum, "enforce_rolling_window", False):
            return {}
        if not self.curriculum.rolling_window:
            return {}
        return {
            "window_tokens": float(self.curriculum.rolling_window.window_tokens),
            "max_band_delta": float(getattr(self, "_rw_max_band_delta", 0.0) or 0.0),
            "max_domain_delta": float(
                getattr(self, "_rw_max_domain_delta", 0.0) or 0.0
            ),
        }

    def get_checkpoint_state(self) -> Dict[str, Any]:
        """Return a pickle-friendly snapshot of streaming selection state.

        This state is required for deterministic crash+resume behavior.
        """

        state: Dict[str, Any] = {
            "active_stage_name": getattr(self, "_active_stage_name", None),
            "remaining_stage_tokens": int(
                getattr(self, "_remaining_stage_tokens", 0) or 0
            ),
            "remaining_band_tokens": dict(
                getattr(self, "_remaining_band_tokens", {}) or {}
            ),
            "remaining_domain_tokens": dict(
                getattr(self, "_remaining_domain_tokens", {}) or {}
            ),
            "selected_chunks": list(getattr(self, "selected_chunks", set()) or set()),
            "removed_chunks": list(getattr(self, "removed_chunks", set()) or set()),
        }

        # Rolling-window state (only present when enabled/initialized)
        if hasattr(self, "_rw_window"):
            state.update(
                {
                    "rw_window": list(getattr(self, "_rw_window", []) or []),
                    "rw_total_tokens": int(getattr(self, "_rw_total_tokens", 0) or 0),
                    "rw_band_tokens": dict(getattr(self, "_rw_band_tokens", {}) or {}),
                    "rw_domain_tokens": dict(
                        getattr(self, "_rw_domain_tokens", {}) or {}
                    ),
                    "rw_max_band_delta": float(
                        getattr(self, "_rw_max_band_delta", 0.0) or 0.0
                    ),
                    "rw_max_domain_delta": float(
                        getattr(self, "_rw_max_domain_delta", 0.0) or 0.0
                    ),
                    "rw_band_targets": dict(
                        getattr(self, "_rw_band_targets", {}) or {}
                    ),
                    "rw_domain_targets": dict(
                        getattr(self, "_rw_domain_targets", {}) or {}
                    ),
                }
            )

        # RNG state (SelectionEngine seeds global RNGs; restore for deterministic resume)
        try:
            state["python_random_state"] = random.getstate()
        except Exception:
            pass
        try:
            state["numpy_random_state"] = np.random.get_state()
        except Exception:
            pass

        return state

    def load_checkpoint_state(self, state: Dict[str, Any]) -> None:
        """Restore streaming selection state from a checkpoint snapshot."""

        if not state:
            return

        self._active_stage_name = state.get("active_stage_name")
        self._remaining_stage_tokens = int(state.get("remaining_stage_tokens", 0) or 0)
        self._remaining_band_tokens = {
            k: int(v) for k, v in (state.get("remaining_band_tokens") or {}).items()
        }
        self._remaining_domain_tokens = {
            k: int(v) for k, v in (state.get("remaining_domain_tokens") or {}).items()
        }

        self.selected_chunks = set(state.get("selected_chunks") or [])
        self.removed_chunks = set(state.get("removed_chunks") or [])

        if "rw_window" in state:
            self._rw_window = deque(state.get("rw_window") or [])
            self._rw_total_tokens = int(state.get("rw_total_tokens", 0) or 0)
            self._rw_band_tokens = Counter(state.get("rw_band_tokens") or {})
            self._rw_domain_tokens = Counter(state.get("rw_domain_tokens") or {})
            self._rw_max_band_delta = float(state.get("rw_max_band_delta", 0.0) or 0.0)
            self._rw_max_domain_delta = float(
                state.get("rw_max_domain_delta", 0.0) or 0.0
            )
            self._rw_band_targets = dict(state.get("rw_band_targets") or {})
            self._rw_domain_targets = dict(state.get("rw_domain_targets") or {})

        py_state = state.get("python_random_state")
        if py_state is not None:
            try:
                random.setstate(py_state)
            except Exception:
                pass
        np_state = state.get("numpy_random_state")
        if np_state is not None:
            try:
                np.random.set_state(np_state)
            except Exception:
                pass

    def _process_batch(
        self,
        batch: List[Tuple[str, ChunkMetadata]],
        stage_name: str,
        protected_slices=None,
        *,
        total_input_tokens_estimate: Optional[int],
        stage_target_tokens: int,
    ) -> Tuple[Set[str], Dict]:
        """
        Process a single batch of chunks.

        Returns:
            (selected_in_batch, batch_stats)
        """
        # Stage-level carryover accounting
        #
        # The streaming/batched implementation must be able to meet stage-level curriculum
        # targets even when individual batches are skewed. The legacy per-batch approach
        # allocates *fresh* band quotas every batch, which can permanently underfill a stage
        # when a batch lacks a given band/domain. Here we track remaining stage + band token
        # budgets and cap each batch's bucket targets accordingly, allowing later batches to
        # make up deficits.
        if getattr(self, "_active_stage_name", None) != stage_name:
            self._active_stage_name = stage_name
            self._remaining_stage_tokens = int(stage_target_tokens)
            self._remaining_band_tokens = self._init_remaining_band_targets(
                stage_name, int(stage_target_tokens)
            )

            # Track stage-level remaining budgets for protected domains (code/agentic/indic)
            self._remaining_domain_tokens = self._init_remaining_domain_targets(
                stage_name,
                int(stage_target_tokens),
                protected_slices or [],
            )

            # Rolling-window state is stage-specific.
            self._rw_init_stage(stage_name)

        timings: Dict[str, float] = {}

        # Register chunks
        batch_chunks: Dict[str, ChunkMetadata] = {}
        batch_tokens_raw = 0

        for chunk_id, metadata in batch:
            batch_chunks[chunk_id] = metadata
            batch_tokens_raw += metadata.token_count

        # Tokens after applying language filtering (used to cap selections)
        batch_tokens = batch_tokens_raw

        # Filter disallowed languages early so per-batch token budgets and bucket targets
        # are computed on an allowed pool (otherwise post-hoc removal can crater totals).
        if self.curriculum.language_policy:
            allowed_languages = set(
                self.curriculum.get_allowed_languages_for_stage(stage_name)
            )
            explicitly_excluded = set(
                self.curriculum.language_policy.explicitly_excluded or set()
            )
            if explicitly_excluded:
                allowed_languages -= explicitly_excluded

            if allowed_languages:
                before = len(batch_chunks)
                batch_chunks = {
                    cid: meta
                    for cid, meta in batch_chunks.items()
                    if meta.language in allowed_languages
                }
                if len(batch_chunks) != before:
                    batch_tokens = sum(
                        meta.token_count for meta in batch_chunks.values()
                    )

        # Register for dedup and diversity analysis
        t0 = time.perf_counter()
        self.register_chunks([(cid, meta, None) for cid, meta in batch_chunks.items()])
        timings["register_chunks_s"] = time.perf_counter() - t0

        # Apply dedup at batch level (not pairwise across all)
        t0 = time.perf_counter()
        if self.config.dedup.enable_exact_dedup:
            self._apply_batch_deduplication(batch_chunks)
        timings["exact_dedup_s"] = time.perf_counter() - t0

        # Determine proportional token budget for this batch.
        # If total_input_tokens_estimate isn't provided, default to selecting everything in-batch
        # (caller should supply estimate for 2T-scale correctness).
        if total_input_tokens_estimate and total_input_tokens_estimate > 0:
            # Budget is based on the *raw* batch token mass (pre-language filter), so that dropping
            # disallowed languages doesn't implicitly downscale the stage target.
            batch_target_tokens = int(
                stage_target_tokens
                * (batch_tokens_raw / float(total_input_tokens_estimate))
            )
            batch_target_tokens = max(0, min(batch_target_tokens, batch_tokens))
        else:
            batch_target_tokens = batch_tokens

        # Don't allocate more than the remaining stage budget.
        remaining_stage = int(getattr(self, "_remaining_stage_tokens", 0) or 0)
        if remaining_stage <= 0:
            return set(), {
                "batch_tokens": batch_tokens,
                "batch_chunks": len(batch_chunks),
                "batch_selected": 0,
                "batch_selected_tokens": 0,
                "batch_target_tokens": 0,
                "timings_s": timings,
            }
        batch_target_tokens = min(int(batch_target_tokens), remaining_stage)

        # Create buckets for batch using per-batch target budget
        t0 = time.perf_counter()
        self._create_buckets(
            batch_chunks, stage_name, target_tokens_override=batch_target_tokens
        )
        timings["create_buckets_s"] = time.perf_counter() - t0

        # Re-allocate per-bucket targets based on remaining stage-level band quotas.
        # This caps bucket targets so that:
        #  - a band can't exceed its remaining quota
        #  - deficits can carry over to later batches
        self._cap_bucket_targets_by_remaining(
            stage_name=stage_name, batch_target_tokens=batch_target_tokens
        )

        # Score chunks in this batch
        t0 = time.perf_counter()
        for bucket in self.buckets.values():
            self._score_chunks_in_bucket(bucket, batch_chunks)
        timings["score_chunks_s"] = time.perf_counter() - t0

        # Select from batch buckets
        t0 = time.perf_counter()
        selected_in_batch: Set[str] = set()
        for bucket in self.buckets.values():
            bucket_selection = self._stratified_sample_from_bucket(bucket, batch_chunks)
            selected_in_batch.update(bucket_selection)
        timings["stratified_sample_s"] = time.perf_counter() - t0

        # Enforce language policy caps (keeps secondary <= max_share and prevents drift).
        # Note: the pool has already been filtered to allowed languages.
        selected_in_batch = self._enforce_language_policy(
            selected_in_batch, batch_chunks, stage_name
        )

        # Enforce rolling-window anti-spike constraints in streaming mode.
        selected_in_batch = self._enforce_rolling_window_streaming(
            selected_in_batch, batch, batch_chunks, stage_name
        )

        # Streaming-safe protected slice enforcement: only add protected candidates up to
        # remaining stage-level budgets (prevents per-batch overshoot).
        if protected_slices and self.config.selection.include_protected_slices:
            before = set(selected_in_batch)
            selected_in_batch = self._enforce_protected_slices_streaming(
                selected_in_batch,
                batch_chunks,
                protected_slices,
                stage_name,
            )

            # Re-enforce language policy after potential additions.
            selected_in_batch = self._enforce_language_policy(
                selected_in_batch, batch_chunks, stage_name
            )

            # Ensure protected additions also respect rolling-window constraints.
            added = list(selected_in_batch - before)
            if added:
                admitted_added = self._rolling_window_admit_candidates(
                    added, batch_chunks, stage_name, prefer_score_order=True
                )
                selected_in_batch = set(before) | set(admitted_added)

        batch_selected_tokens = sum(
            batch_chunks[cid].token_count
            for cid in selected_in_batch
            if cid in batch_chunks
        )

        # Update remaining budgets
        if batch_selected_tokens > 0:
            self._remaining_stage_tokens = max(
                0, int(self._remaining_stage_tokens) - int(batch_selected_tokens)
            )

            selected_by_band: Dict[str, int] = defaultdict(int)
            selected_by_domain: Dict[str, int] = defaultdict(int)
            for cid in selected_in_batch:
                meta = batch_chunks.get(cid)
                if not meta:
                    continue
                selected_by_band[meta.band.value] += int(meta.token_count)
                selected_by_domain[
                    str(getattr(meta.domain, "value", meta.domain))
                ] += int(meta.token_count)
            for band_name, tok in selected_by_band.items():
                if band_name in self._remaining_band_tokens:
                    self._remaining_band_tokens[band_name] = max(
                        0, int(self._remaining_band_tokens[band_name]) - int(tok)
                    )

            remaining_domain: Dict[str, int] = (
                getattr(self, "_remaining_domain_tokens", {}) or {}
            )
            for domain_name, tok in selected_by_domain.items():
                if domain_name in remaining_domain:
                    remaining_domain[domain_name] = max(
                        0, int(remaining_domain[domain_name]) - int(tok)
                    )
            self._remaining_domain_tokens = remaining_domain

        timings["total_batch_s"] = sum(timings.values())

        return selected_in_batch, {
            "batch_tokens": batch_tokens,
            "batch_chunks": len(batch_chunks),
            "batch_selected": len(selected_in_batch),
            "batch_selected_tokens": batch_selected_tokens,
            "batch_target_tokens": batch_target_tokens,
            "timings_s": timings,
        }

    def _rw_init_stage(self, stage_name: str) -> None:
        """Initialize rolling-window tracking for a stage."""
        self._rw_window: Deque[Tuple[str, str, str, int]] = (
            deque()
        )  # (chunk_id, band, domain, tokens)
        self._rw_total_tokens = 0
        self._rw_band_tokens: Counter[str] = Counter()
        self._rw_domain_tokens: Counter[str] = Counter()
        self._rw_max_band_delta = 0.0
        self._rw_max_domain_delta = 0.0

        stage_spec = self.curriculum.stages.get(stage_name)
        band_targets: Dict[str, float] = {
            b: 0.0 for b in ["B0", "B1", "B2", "B3", "B4", "B5"]
        }
        if stage_spec:
            for b in band_targets.keys():
                band_targets[b] = float(getattr(stage_spec.band_ratios, b, 0.0) or 0.0)

        inferred_domain_target: Dict[str, float] = defaultdict(float)
        for band_enum, band_def in self.curriculum.bands.items():
            band_name = band_enum.value
            band_ratio = float(band_targets.get(band_name, 0.0) or 0.0)
            allowed = band_def.allowed_domains or []
            if not allowed:
                continue
            per_domain = band_ratio / max(1, len(allowed))
            for d in allowed:
                inferred_domain_target[d] += per_domain

        total_inferred = float(sum(inferred_domain_target.values()))
        if total_inferred > 0:
            for d in list(inferred_domain_target.keys()):
                inferred_domain_target[d] = (
                    float(inferred_domain_target[d]) / total_inferred
                )

        self._rw_band_targets = band_targets
        self._rw_domain_targets = dict(inferred_domain_target)

    def _enforce_rolling_window_streaming(
        self,
        selected_in_batch: Set[str],
        batch: List[Tuple[str, ChunkMetadata]],
        batch_chunks: Dict[str, ChunkMetadata],
        stage_name: str,
    ) -> Set[str]:
        """Apply rolling-window anti-spike constraints incrementally, preserving stream order."""
        if not getattr(self.config.curriculum, "enforce_rolling_window", False):
            return selected_in_batch
        if not self.curriculum.rolling_window:
            return selected_in_batch

        ordered = [
            str(cid)
            for cid, _m in batch
            if str(cid) in selected_in_batch and str(cid) in batch_chunks
        ]
        admitted = self._rolling_window_admit_candidates(
            ordered, batch_chunks, stage_name, prefer_score_order=False
        )
        return set(admitted)

    def _rolling_window_admit_candidates(
        self,
        candidates: List[str],
        batch_chunks: Dict[str, ChunkMetadata],
        stage_name: str,
        *,
        prefer_score_order: bool,
    ) -> List[str]:
        """Admit candidates into the rolling window, rejecting any that would violate constraints."""
        if not getattr(self.config.curriculum, "enforce_rolling_window", False):
            return list(candidates)
        if not self.curriculum.rolling_window:
            return list(candidates)

        rw = self.curriculum.rolling_window
        window_tokens = int(rw.window_tokens)
        max_band_delta_allowed = float(rw.max_band_delta)
        max_domain_delta_allowed = float(rw.max_domain_delta)

        if prefer_score_order:

            def score_for(cid: str) -> float:
                meta = batch_chunks.get(cid)
                if not meta:
                    return 0.0
                bucket = self.buckets.get((meta.band, meta.domain))
                if not bucket:
                    return 0.0
                return float(bucket.scores.get(cid, 0.0) or 0.0)

            # Deterministic: score desc, then chunk_id asc
            candidates = sorted(candidates, key=lambda c: (-score_for(c), str(c)))

        admitted: List[str] = []

        for cid in candidates:
            meta = batch_chunks.get(cid)
            if not meta:
                continue

            tok = int(meta.token_count or 0)
            if tok <= 0:
                continue

            # Simulate left removals to keep window <= window_tokens after adding this chunk.
            sim_total = int(self._rw_total_tokens) + tok
            removed_entries: List[Tuple[str, str, str, int]] = []
            removed_tokens = 0
            for entry in self._rw_window:
                if sim_total - removed_tokens <= window_tokens:
                    break
                removed_entries.append(entry)
                removed_tokens += int(entry[3])

            new_total = sim_total - removed_tokens
            if new_total <= 0:
                continue

            # Rolling-window constraints are defined over windows of size `window_tokens`.
            # While the window is still "warming up" (total < window_tokens), accept
            # chunks without applying anti-spike checks; otherwise early windows would
            # trivially violate targets (e.g., a single chunk has 100% share of its band).
            enforce_now = new_total >= window_tokens

            # Compute simulated counts (only 6 bands and a small domain set).
            sim_band = Counter(self._rw_band_tokens)
            sim_domain = Counter(self._rw_domain_tokens)
            for _old_cid, old_band, old_domain, old_tok in removed_entries:
                sim_band[old_band] -= int(old_tok)
                sim_domain[old_domain] -= int(old_tok)
            sim_band[meta.band.value] += tok
            sim_domain[meta.domain] += tok

            violation = False
            if enforce_now:
                # Check band constraints.
                for band_name, target_share in (
                    getattr(self, "_rw_band_targets", {}) or {}
                ).items():
                    count = float(sim_band.get(band_name, 0) or 0)
                    share = count / float(new_total)
                    allowed = float(target_share) + max_band_delta_allowed
                    if share > allowed + 1e-12:
                        violation = True
                        break

                # Check domain constraints.
                if not violation:
                    for domain_name, target_share in (
                        getattr(self, "_rw_domain_targets", {}) or {}
                    ).items():
                        count = float(sim_domain.get(domain_name, 0) or 0)
                        share = count / float(new_total)
                        allowed = float(target_share) + max_domain_delta_allowed
                        if share > allowed + 1e-12:
                            violation = True
                            break

            if violation:
                continue

            # Accept: apply removals, then append.
            for _ in range(len(removed_entries)):
                old_cid, old_band, old_domain, old_tok = self._rw_window.popleft()
                self._rw_total_tokens -= int(old_tok)
                self._rw_band_tokens[old_band] -= int(old_tok)
                self._rw_domain_tokens[old_domain] -= int(old_tok)

            self._rw_window.append((str(cid), meta.band.value, meta.domain, tok))
            self._rw_total_tokens += tok
            self._rw_band_tokens[meta.band.value] += tok
            self._rw_domain_tokens[meta.domain] += tok

            # Update observed deltas.
            if self._rw_total_tokens >= window_tokens:
                total_now = float(self._rw_total_tokens)
                for band_name, target_share in (
                    getattr(self, "_rw_band_targets", {}) or {}
                ).items():
                    share = (
                        float(self._rw_band_tokens.get(band_name, 0) or 0) / total_now
                    )
                    self._rw_max_band_delta = max(
                        self._rw_max_band_delta, float(share - float(target_share))
                    )
                for domain_name, target_share in (
                    getattr(self, "_rw_domain_targets", {}) or {}
                ).items():
                    share = (
                        float(self._rw_domain_tokens.get(domain_name, 0) or 0)
                        / total_now
                    )
                    self._rw_max_domain_delta = max(
                        self._rw_max_domain_delta, float(share - float(target_share))
                    )

            admitted.append(str(cid))

        return admitted

    def _init_remaining_band_targets(
        self, stage_name: str, stage_target_tokens: int
    ) -> Dict[str, int]:
        """Initialize remaining band token budgets for a stage based on curriculum ratios."""
        stage_config = self.curriculum.get_stage_config(stage_name)
        if not stage_config:
            return {b: 0 for b in difficulty_band_order()}

        band_ratios = stage_config.band_ratios
        # Deterministic rounding: floor each band then give remainder to the highest band.
        remaining: Dict[str, int] = {}
        total = 0
        ordered_bands = difficulty_band_order()
        for band_name in ordered_bands:
            ratio = float(getattr(band_ratios, band_name, 0.0) or 0.0)
            tokens = int(stage_target_tokens * ratio)
            remaining[band_name] = max(0, tokens)
            total += remaining[band_name]

        # Ensure sums match stage_target_tokens (within integer rounding).
        if total != int(stage_target_tokens):
            if ordered_bands:
                highest = ordered_bands[-1]
                remaining[highest] = max(
                    0,
                    int(remaining.get(highest, 0))
                    + int(stage_target_tokens)
                    - int(total),
                )
        return remaining

    def _cap_bucket_targets_by_remaining(
        self, *, stage_name: str, batch_target_tokens: int
    ) -> None:
        """Cap per-bucket targets using remaining stage-level band budgets."""
        stage_config = self.curriculum.get_stage_config(stage_name)
        if not stage_config:
            return

        band_ratios = stage_config.band_ratios
        remaining_band: Dict[str, int] = (
            getattr(self, "_remaining_band_tokens", {}) or {}
        )

        # Compute allowed domains with chunks for each band for this batch.
        band_to_domains_with_chunks: Dict[str, List[str]] = defaultdict(list)
        for (band, domain), bucket in self.buckets.items():
            if not bucket.chunks:
                continue
            allowed_domains = self.curriculum.get_allowed_domains_for_band(band)
            if domain in allowed_domains:
                band_to_domains_with_chunks[band.value].append(domain)

        # Assign targets by band, capped by remaining per-band tokens.
        for band_name, domains in band_to_domains_with_chunks.items():
            if not domains:
                continue
            domains_sorted = sorted(set(domains))
            num_domains = len(domains_sorted)
            ratio = float(getattr(band_ratios, band_name, 0.0) or 0.0)
            desired_band_tokens = int(batch_target_tokens * ratio)
            desired_band_tokens = min(
                desired_band_tokens, int(remaining_band.get(band_name, 0) or 0)
            )
            if desired_band_tokens <= 0:
                for domain in domains_sorted:
                    bucket = self.buckets.get((DifficultyBand(band_name), domain))
                    if bucket:
                        bucket.target_tokens = 0
                continue

            base = int(desired_band_tokens // num_domains)
            rem = int(desired_band_tokens % num_domains)
            for i, domain in enumerate(domains_sorted):
                bucket = self.buckets.get((DifficultyBand(band_name), domain))
                if not bucket:
                    continue
                bucket.target_tokens = int(base + (1 if i < rem else 0))

        # For any disallowed buckets that slipped through, ensure target is zero.
        for (band, domain), bucket in self.buckets.items():
            allowed = self.curriculum.get_allowed_domains_for_band(band)
            if domain not in allowed:
                bucket.target_tokens = 0

    def _init_remaining_domain_targets(
        self, stage_name: str, stage_target_tokens: int, protected_slices: List
    ) -> Dict[str, int]:
        """Initialize remaining token budgets for protected domains based on curriculum implied targets."""

        def _key(v) -> str:
            return str(getattr(v, "value", v))

        protected_domains = {
            _key(r.band_or_domain)
            for r in protected_slices
            if _key(r.band_or_domain) not in set(difficulty_band_order())
        }
        if not protected_domains:
            return {}

        stage_spec = self.curriculum.stages.get(stage_name)
        if not stage_spec:
            return {d: 0 for d in sorted(protected_domains)}

        band_targets: Dict[str, float] = {
            b: float(getattr(stage_spec.band_ratios, b, 0.0) or 0.0)
            for b in difficulty_band_order()
        }
        inferred: Dict[str, float] = defaultdict(float)
        for band_enum, band_def in self.curriculum.bands.items():
            band_name = band_enum.value
            band_ratio = float(band_targets.get(band_name, 0.0) or 0.0)
            allowed = band_def.allowed_domains or []
            if not allowed:
                continue
            per_domain = band_ratio / max(1, len(allowed))
            for d in allowed:
                inferred[d] += per_domain

        total_inferred = float(sum(inferred.values()))
        if total_inferred > 0:
            for d in list(inferred.keys()):
                inferred[d] = float(inferred[d]) / total_inferred

        remaining: Dict[str, int] = {}
        for d in sorted(protected_domains):
            remaining[d] = max(
                0, int(stage_target_tokens * float(inferred.get(d, 0.0) or 0.0))
            )
        return remaining

    def _enforce_protected_slices_streaming(
        self,
        selected_in_batch: Set[str],
        batch_chunks: Dict[str, ChunkMetadata],
        protected_slices: List,
        stage_name: str,
    ) -> Set[str]:
        """Add protected-slice chunks within the current batch, capped by remaining stage budgets."""
        # IMPORTANT: account for the current batch's base selection before adding more.
        # The engine tracks remaining budgets across batches in self._remaining_*.
        # Within a batch, protected-slice boosting must not treat already-selected tokens
        # as still "available" budget.
        base_selected_tokens = 0
        base_selected_by_band: Dict[str, int] = defaultdict(int)
        base_selected_by_domain: Dict[str, int] = defaultdict(int)
        for cid in selected_in_batch:
            meta = batch_chunks.get(cid)
            if not meta:
                continue
            tok = int(meta.token_count or 0)
            if tok <= 0:
                continue
            base_selected_tokens += tok
            base_selected_by_band[str(getattr(meta.band, "value", meta.band))] += tok
            base_selected_by_domain[
                str(getattr(meta.domain, "value", meta.domain))
            ] += tok

        remaining_stage_pre = int(getattr(self, "_remaining_stage_tokens", 0) or 0)
        remaining_stage = max(0, remaining_stage_pre - int(base_selected_tokens))
        if remaining_stage <= 0:
            return selected_in_batch

        remaining_band_pre: Dict[str, int] = (
            getattr(self, "_remaining_band_tokens", {}) or {}
        )
        remaining_domain_pre: Dict[str, int] = (
            getattr(self, "_remaining_domain_tokens", {}) or {}
        )

        remaining_band: Dict[str, int] = {
            k: max(0, int(v) - int(base_selected_by_band.get(k, 0) or 0))
            for k, v in remaining_band_pre.items()
        }
        remaining_domain: Dict[str, int] = {
            k: max(0, int(v) - int(base_selected_by_domain.get(k, 0) or 0))
            for k, v in remaining_domain_pre.items()
        }

        # Helper to score deterministically
        def score_for(cid: str) -> float:
            meta = batch_chunks.get(cid)
            if not meta:
                return 0.0
            bucket = self.buckets.get((meta.band, meta.domain))
            if not bucket:
                return 0.0
            return float(bucket.scores.get(cid, 0.0) or 0.0)

        selected_out = set(selected_in_batch)

        def _key(v) -> str:
            return str(getattr(v, "value", v))

        band_keys = set(difficulty_band_order())

        _allowed_cache: Dict[str, Set[str]] = {}

        def allowed_domains_for_band_name(band_name: str) -> Set[str]:
            band_name = str(band_name)
            cached = _allowed_cache.get(band_name)
            if cached is not None:
                return cached
            try:
                band_enum = DifficultyBand(str(band_name))
            except Exception:
                out = set()
                _allowed_cache[band_name] = out
                return out
            try:
                allowed = self.curriculum.get_allowed_domains_for_band(band_enum) or []
            except Exception:
                allowed = []
            out = set(str(d) for d in allowed)
            _allowed_cache[band_name] = out
            return out

        for rule in protected_slices:
            key = _key(rule.band_or_domain)
            # Band-based rules
            if key in band_keys:
                needed = int(remaining_band.get(key, 0) or 0)
                if needed <= 0:
                    continue
                budget = min(needed, remaining_stage)
                if budget <= 0:
                    continue

                allowed_domains = allowed_domains_for_band_name(key)

                candidates = [
                    cid
                    for cid, meta in batch_chunks.items()
                    if cid not in selected_out
                    and _key(meta.band) == key
                    and (
                        (not allowed_domains) or (_key(meta.domain) in allowed_domains)
                    )
                ]
                if not candidates:
                    continue
                candidates_sorted = sorted(
                    candidates, key=lambda c: (-score_for(c), str(c))
                )

                added = 0
                for cid in candidates_sorted:
                    tok = int(batch_chunks[cid].token_count or 0)
                    if tok <= 0:
                        continue
                    if added + tok > budget:
                        continue
                    selected_out.add(cid)
                    added += tok
                    dom_key = _key(batch_chunks[cid].domain)
                    if dom_key in remaining_domain:
                        remaining_domain[dom_key] = max(
                            0, int(remaining_domain.get(dom_key, 0)) - int(tok)
                        )
                    if added >= budget:
                        break
                remaining_stage -= added
                remaining_band[key] = max(
                    0, int(remaining_band.get(key, 0)) - int(added)
                )
                if remaining_stage <= 0:
                    break
                continue

            # Domain-based rules
            needed = int(remaining_domain.get(key, 0) or 0)
            if needed <= 0:
                continue
            budget = min(needed, remaining_stage)
            if budget <= 0:
                continue

            candidates = [
                cid
                for cid, meta in batch_chunks.items()
                if cid not in selected_out
                and _key(meta.domain) == key
                and (
                    (not allowed_domains_for_band_name(_key(meta.band)))
                    or (
                        _key(meta.domain)
                        in allowed_domains_for_band_name(_key(meta.band))
                    )
                )
            ]
            if not candidates:
                continue
            candidates_sorted = sorted(
                candidates, key=lambda c: (-score_for(c), str(c))
            )

            added = 0
            for cid in candidates_sorted:
                meta = batch_chunks.get(cid)
                if not meta:
                    continue
                tok = int(meta.token_count or 0)
                if tok <= 0:
                    continue
                band_name = _key(meta.band)
                band_budget = int(remaining_band.get(band_name, 0) or 0)
                if band_budget <= 0:
                    continue
                if added + tok > budget:
                    continue
                if tok > band_budget:
                    continue
                selected_out.add(cid)
                added += tok
                remaining_band[band_name] = max(0, int(band_budget) - int(tok))
                if added >= budget:
                    break
            remaining_stage -= added
            remaining_domain[key] = max(
                0, int(remaining_domain.get(key, 0)) - int(added)
            )
            if remaining_stage <= 0:
                break

        return selected_out

    def _apply_batch_deduplication(
        self, batch_chunks: Dict[str, ChunkMetadata]
    ) -> None:
        """
        Apply deduplication within a batch only (not globally).

        This is more efficient than pairwise comparison across all chunks
        and reduces memory usage for large datasets.
        """
        if not self.config.dedup.enable_exact_dedup:
            return

        # Simple hash-based dedup within batch
        hashes_seen = {}

        for chunk_id, metadata in batch_chunks.items():
            if chunk_id in self.removed_chunks:
                continue

            # Compute hash if text available
            if hasattr(metadata, "chunk_text"):
                chunk_hash = self.exact_dedup.compute_hash(
                    chunk_id, metadata.chunk_text
                )

                if chunk_hash in hashes_seen:
                    # Keep first, mark second for removal
                    self.removed_chunks.add(chunk_id)
                else:
                    hashes_seen[chunk_hash] = chunk_id
            else:
                # No text, use chunk_id as hash
                if chunk_id in hashes_seen:
                    self.removed_chunks.add(chunk_id)
                else:
                    hashes_seen[chunk_id] = chunk_id

    def select_from_checkpoint(
        self,
        chunk_stream: Generator[Tuple[str, ChunkMetadata], None, None],
        stage_name: str,
        last_batch_checkpoint: Optional[int] = None,
        batch_size: int = 10_000,
        protected_slices=None,
        checkpoint_callback=None,
    ) -> Tuple[Set[str], Dict[str, Any]]:
        """
        Resume selection from a checkpoint, skipping already-processed batches.

        Args:
            chunk_stream: Fresh generator of chunks
            stage_name: Stage name
            last_batch_checkpoint: Number of last successfully processed batch
            batch_size: Chunks per batch
            protected_slices: Protected slice rules
            checkpoint_callback: Checkpoint save function

        Returns:
            (selected, stats)
        """
        if last_batch_checkpoint is None:
            # No resumption needed
            return self.select_for_stage_batched(
                chunk_stream,
                stage_name,
                batch_size,
                protected_slices,
                checkpoint_callback,
            )

        logger.info(
            f"Resuming selection for {stage_name} from batch {last_batch_checkpoint + 1}"
        )

        # Skip to last checkpoint
        batch_num = 0
        for _ in chunk_stream:
            if batch_num * batch_size >= (last_batch_checkpoint + 1) * batch_size:
                break
            batch_num += 1

        # Continue from current position
        return self.select_for_stage_batched(
            chunk_stream, stage_name, batch_size, protected_slices, checkpoint_callback
        )
