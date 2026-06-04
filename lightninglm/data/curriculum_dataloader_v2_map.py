"""
Curriculum Dataloader v2 — Map-style (random-access) variant.

Solves the checkpoint determinism problem by pre-computing the full
sequence of (pool, shard, seq_offset) tuples upfront.  Workers are
stateless — they just call __getitem__(idx) — so there's no prefetch
buffer state mismatch.  Checkpoint state is a single integer: the
global index.

Usage:
    dataset, loader = build_curriculum_v2_map_dataloader(
        shard_root="data/training_shards_8k",
        manifest_dir="manifests",
        curriculum_path="configs/curriculum_v2.yaml",
        stage="1B",
        seq_len=4096,
        rank=0, world_size=8,
        batch_size=4,
        num_workers=4,
    )
    for batch in loader:
        # batch["input_ids"], batch["labels"], batch["_pool"], batch["_global_idx"]
        ...

Determinism test:
    Run N steps → save index → resume from index → compare against
    a fresh run of N+M steps.  They must match exactly.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Sampler

from lightninglm.data.bin_idx_dataloader import SHARD_BLOCK_SIZE

logger = logging.getLogger(__name__)


def _count_sequences_in_shard(bin_path: str, dtype: np.dtype, seq_len: int) -> int:
    """Count how many seq_len sequences fit in a shard file without reading it."""
    if not os.path.exists(bin_path):
        return 0
    file_size = os.path.getsize(bin_path)
    total_tokens = file_size // dtype.itemsize
    return total_tokens // seq_len


def _read_sequence_at(
    bin_path: str, dtype: np.dtype, seq_len: int, seq_idx: int
) -> torch.Tensor:
    """Random-access read of a single sequence from a binary shard file.

    Each sequence is seq_len tokens of dtype (uint32 = 4 bytes each).
    Sequence i starts at byte offset i * seq_len * itemsize.
    """
    itemsize = dtype.itemsize
    offset = seq_idx * seq_len * itemsize
    nbytes = seq_len * itemsize

    with open(bin_path, "rb") as f:
        f.seek(offset)
        raw = f.read(nbytes)

    if len(raw) < nbytes:
        raise RuntimeError(
            f"Short read in {bin_path}: wanted {nbytes} bytes at offset {offset}, "
            f"got {len(raw)}. File may be truncated."
        )

    arr = np.frombuffer(raw, dtype=dtype)
    return torch.from_numpy(arr.astype(np.int64)).to(torch.long)


# ---------------------------------------------------------------------------
# Pre-computed index: list of (pool_name, shard_path, seq_idx) tuples
# ---------------------------------------------------------------------------


def _build_sequence_index(
    pool_shards: Dict[str, List[str]],
    pool_names: List[str],
    pool_weights: List[float],
    shard_root: str,
    dtype: np.dtype,
    seq_len: int,
    seed: int,
    max_sequences: Optional[int] = None,
    indic_language_shards: Optional[Dict[str, List[str]]] = None,
    exclusion_set: Optional[set] = None,
    recycle_pools: Optional[set] = None,
    prefix_phases: Optional[List[Tuple[List[float], int]]] = None,
) -> List[Tuple[str, str, int]]:
    """Pre-compute the full training sequence order.

    Uses weighted random sampling (same RNG logic as the IterableDataset)
    to pick pools, then iterates through each pool's shards in a
    *shuffled* order (seeded per-rank) so that shard boundaries are
    staggered across ranks.  Without this, all ranks hit shard transitions
    at roughly the same step, causing a coordinated content shift that
    spikes the loss.

    For AON_indic with language annotations (indic_language_shards),
    uses a multi-language round-robin: holds one cursor per language and
    cycles through them, so every N consecutive AON_indic samples cover
    all N languages.  This gives true per-sample language mixing instead
    of reading one language for ~976 steps straight.

    Multi-phase prefix support (prefix_phases):
        When resuming after multiple weight changes, prefix_phases is a
        list of (weights, count) tuples.  Each tuple specifies the
        normalized weight distribution and the number of index positions
        to build with those weights.  After all prefix phases are
        exhausted, the main pool_weights (the NEW distribution) is used.
        This ensures the index exactly replays the historical weight
        sequence so no data is skipped or repeated.

    Returns a list of (pool_name, shard_rel_path, seq_index_within_shard).
    """
    rng = random.Random(seed)

    # Multi-phase prefix: list of (weights, count) for historical replay.
    # After all phases, switches to pool_weights (the new distribution).
    _use_phases = prefix_phases is not None and len(prefix_phases) > 0
    _phase_boundaries = []  # cumulative index boundaries
    _phase_weights = []  # normalized weight lists per phase
    if _use_phases:
        _cum = 0
        for i, (pw, pc) in enumerate(prefix_phases):
            assert len(pw) == len(
                pool_weights
            ), f"prefix_phases[{i}] weights length {len(pw)} != pool_weights length {len(pool_weights)}"
            _pw_total = sum(pw)
            _norm = [w / _pw_total for w in pw] if _pw_total > 0 else list(pool_weights)
            _phase_weights.append(_norm)
            _cum += pc
            _phase_boundaries.append(_cum)
            print(
                f"[multi-phase] phase {i}: count={pc}, cumulative={_cum}, "
                f"weights={[f'{w:.4f}' for w in _norm]}",
                flush=True,
            )
        print(
            f"[multi-phase] final phase (post-resume): "
            f"weights={[f'{w:.4f}' for w in pool_weights]}",
            flush=True,
        )
    _current_phase = 0
    _post_weights_norm = list(pool_weights)

    # Build per-pool state: list of (shard_path, num_sequences) + cursor
    class PoolState:
        def __init__(self, name: str, shards: List[str], excl: Optional[set] = None):
            self.name = name
            self._excl = excl or set()
            self.shards = []  # (rel_path, num_seqs)
            for s in shards:
                bin_path = os.path.join(shard_root, s, "tokens.bin")
                n = _count_sequences_in_shard(bin_path, dtype, seq_len)
                if n > 0:
                    self.shards.append((s, n))
            # Shuffle shard order within each pool so that different ranks
            # (which get different seeds) hit shard boundaries at different
            # steps.  This prevents the coordinated shard transition problem
            # that caused the loss spike at step ~16K.
            # Use a deterministic hash (not Python's hash() which is
            # randomized per-process via PYTHONHASHSEED) so shard order
            # is reproducible across runs/machines.
            _name_hash = int.from_bytes(name.encode("utf-8"), "little") % (2**31)
            _pool_rng = random.Random(seed + _name_hash)
            _pool_rng.shuffle(self.shards)
            self.shard_cursor = 0
            self.seq_cursor = 0
            self.exhausted = len(self.shards) == 0

        def next_entry(self) -> Optional[Tuple[str, str, int]]:
            while not self.exhausted:
                if self.shard_cursor < len(self.shards):
                    rel_path, n_seqs = self.shards[self.shard_cursor]
                    if self.seq_cursor < n_seqs:
                        cur = self.seq_cursor
                        self.seq_cursor += 1
                        # Skip excluded sequences internally so the
                        # weighted pool selection isn't wasted
                        if self._excl and (rel_path, cur) in self._excl:
                            continue
                        return (self.name, rel_path, cur)
                    else:
                        self.shard_cursor += 1
                        self.seq_cursor = 0
                else:
                    self.exhausted = True
            return None

    class MultiLangPoolState:
        """AON_indic pool with per-language sub-cursors and round-robin mixing.

        Holds one PoolState per language.  Each call to next_entry() advances
        a round-robin pointer, pulling one sample from the next language.
        When a language exhausts all its shards, it's removed from the
        rotation.  The pool is exhausted only when ALL languages are done.
        """

        def __init__(self, name: str, lang_shards: Dict[str, List[str]]):
            self.name = name
            # Sort languages alphabetically for determinism
            self._languages = sorted(lang_shards.keys())
            self._sub_pools: Dict[str, PoolState] = {}
            for lang in self._languages:
                # Each language gets its own PoolState with its own shard shuffle
                sub = PoolState(f"{name}_{lang}", lang_shards[lang], excl=exclusion_set)
                # Override the name back to AON_indic so the index entries
                # are tagged correctly (pool name stays "AON_indic")
                sub.name = name
                self._sub_pools[lang] = sub

            self._active_langs = [
                l for l in self._languages if not self._sub_pools[l].exhausted
            ]
            self._rr_cursor = 0
            self.exhausted = len(self._active_langs) == 0

            _total = sum(len(sp.shards) for sp in self._sub_pools.values())
            print(
                f"[MultiLangPool] {name}: {len(self._languages)} languages, "
                f"{_total} total shards",
                flush=True,
            )
            for lang in self._languages:
                sp = self._sub_pools[lang]
                print(f"  {lang}: {len(sp.shards)} shards", flush=True)

        def next_entry(self) -> Optional[Tuple[str, str, int]]:
            """Round-robin across languages, pulling one sample per call."""
            if self.exhausted:
                return None

            # Try up to len(active_langs) times to find a non-exhausted language
            attempts = 0
            while attempts < len(self._active_langs):
                lang = self._active_langs[self._rr_cursor % len(self._active_langs)]
                sub = self._sub_pools[lang]
                entry = sub.next_entry()

                if entry is not None:
                    # Advance round-robin to next language
                    self._rr_cursor = (self._rr_cursor + 1) % len(self._active_langs)
                    return entry

                # This language is exhausted — remove from rotation
                self._active_langs.remove(lang)
                if not self._active_langs:
                    self.exhausted = True
                    return None
                # Adjust cursor to stay in bounds
                self._rr_cursor = self._rr_cursor % len(self._active_langs)
                attempts += 1

            self.exhausted = True
            return None

    # Build states for all pools
    states: Dict[str, Any] = {}
    for name in pool_names:
        if name == "AON_indic" and indic_language_shards is not None:
            states[name] = MultiLangPoolState(name, indic_language_shards)
        else:
            states[name] = PoolState(name, pool_shards[name], excl=exclusion_set)

    active_names = list(pool_names)
    # Start with first phase weights if multi-phase mode is active
    if _use_phases and _phase_weights:
        active_weights = list(_phase_weights[0])
    else:
        active_weights = list(pool_weights)
    index: List[Tuple[str, str, int]] = []

    while active_names:
        if max_sequences is not None and len(index) >= max_sequences:
            break

        # Multi-phase: check if we need to advance to the next phase
        if _use_phases and _current_phase < len(_phase_boundaries):
            if len(index) >= _phase_boundaries[_current_phase]:
                _current_phase += 1
                # Pick the right weight source for this phase
                if _current_phase < len(_phase_boundaries):
                    _src = _phase_weights[_current_phase]
                else:
                    _src = _post_weights_norm  # final phase = new weights
                _new_weights = []
                for name in active_names:
                    _orig_idx = pool_names.index(name)
                    _new_weights.append(_src[_orig_idx])
                _nw_total = sum(_new_weights)
                if _nw_total > 0:
                    active_weights = [w / _nw_total for w in _new_weights]
                _phase_label = (
                    f"phase {_current_phase}"
                    if _current_phase < len(_phase_boundaries)
                    else "post-resume"
                )
                print(
                    f"[multi-phase] Switched to {_phase_label} at index {len(index)}, "
                    f"active_pools={active_names}, "
                    f"weights={[f'{w:.4f}' for w in active_weights]}",
                    flush=True,
                )

        chosen = rng.choices(active_names, weights=active_weights, k=1)[0]
        entry = states[chosen].next_entry()

        if entry is None:
            # Pool exhausted — recycle or remove
            if recycle_pools and chosen in recycle_pools:
                # Reset cursor: re-shuffle shards and start over
                state = states[chosen]
                if hasattr(state, "shard_cursor"):
                    _pool_rng_r = random.Random(seed + len(index))
                    _pool_rng_r.shuffle(state.shards)
                    state.shard_cursor = 0
                    state.seq_cursor = 0
                    state.exhausted = False
                    if len(index) % 100000 < 10:
                        print(
                            f"[recycle] {chosen} recycled at index {len(index)}",
                            flush=True,
                        )
                    continue
                # MultiLangPoolState — reset all sub-pools
                elif hasattr(state, "_sub_pools"):
                    for lang, sub in state._sub_pools.items():
                        _pool_rng_r = random.Random(
                            seed + len(index) + hash(lang) % 10000
                        )
                        _pool_rng_r.shuffle(sub.shards)
                        sub.shard_cursor = 0
                        sub.seq_cursor = 0
                        sub.exhausted = False
                    state._active_langs = sorted(state._sub_pools.keys())
                    state._rr_cursor = 0
                    state.exhausted = False
                    if len(index) % 100000 < 10:
                        print(
                            f"[recycle] {chosen} (multi-lang) recycled at index {len(index)}",
                            flush=True,
                        )
                    continue

            # Non-recycled pool: remove and renormalize
            idx = active_names.index(chosen)
            active_names.pop(idx)
            active_weights.pop(idx)
            # Remove any zero-weight pools (e.g. AON_bench with bench_train=0.0)
            _to_remove = [i for i, w in enumerate(active_weights) if w <= 0.0]
            for i in reversed(_to_remove):
                active_names.pop(i)
                active_weights.pop(i)
            if active_names:
                # After removing a pool, rebuild weights from the correct
                # source (prefix or post) to avoid drift from repeated
                # renormalization of already-normalized values.
                if _use_phases and _current_phase < len(_phase_weights):
                    _src = _phase_weights[_current_phase]
                else:
                    _src = _post_weights_norm
                _new_w = []
                for name in active_names:
                    _orig_idx = pool_names.index(name)
                    _new_w.append(_src[_orig_idx])
                _nw_total = sum(_new_w)
                if _nw_total > 0:
                    active_weights = [w / _nw_total for w in _new_w]
                else:
                    break  # all remaining pools have zero weight
            continue

        # Exclusions are now handled inside PoolState.next_entry()
        # so weighted pool selection is never wasted on excluded sequences.
        index.append(entry)

    return index


# ---------------------------------------------------------------------------
# Map-style Dataset
# ---------------------------------------------------------------------------


class CurriculumDatasetV2Map(Dataset):
    """
    Map-style curriculum dataset with pre-computed sequence order.

    The entire sequence of (pool, shard, offset) is computed at init time
    using the same weighted-random pool selection as the IterableDataset.
    __getitem__ does a single random-access read from the shard file.

    Checkpoint state = one integer (the current index into the sequence).
    Resume = create a SequentialSampler starting from that index.
    """

    OPUS_POOLS = ("D1", "D2", "D3", "D4", "D5")
    # Pools that recycle (reset cursor) instead of exhausting.
    # D5 has only 548 shards but 44.5% weight — without recycling
    # it exhausts in the first ~60K dataloader steps and the model
    # gets zero Wikipedia for the remaining ~90% of training.
    RECYCLE_POOLS = {"D5"}
    AON_SUB_POOLS = ("AON_bench", "AON_indic")

    def __init__(
        self,
        shard_root: str,
        manifest_dir: str,
        curriculum_path: str,
        stage: str,
        seq_len: int = SHARD_BLOCK_SIZE,
        rank: int = 0,
        world_size: int = 1,
        mode: str = "combined",
        seed: int = 42,
        dtype: str = "uint32",
        max_sequences: Optional[int] = None,
        exclusion_file: Optional[str] = None,
        prefix_band_weights: Optional[Dict[str, float]] = None,
        prefix_count: int = 0,
    ) -> None:
        super().__init__()
        self.shard_root = shard_root
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self._dtype = np.dtype(dtype)
        self._consumed_index: int = 0  # tracks last consumed batch for checkpointing

        # Load curriculum config
        with open(curriculum_path, "r") as f:
            curriculum = yaml.safe_load(f)

        stage_cfg = curriculum["stages"].get(stage)
        if stage_cfg is None:
            raise ValueError(
                f"Unknown stage '{stage}'. Available: {list(curriculum['stages'].keys())}"
            )
        weights = stage_cfg["band_weights"]

        # Multi-phase prefix: check YAML for prefix_phases list
        # Format: list of {weights: {D1: w, ...}, count: N}
        # Falls back to old prefix_band_weights (single dict) for backward compat
        _yaml_phases = stage_cfg.get("prefix_phases")
        if _yaml_phases and prefix_count > 0:
            print(
                f"[MapDataset] Loaded prefix_phases from YAML: {len(_yaml_phases)} phases",
                flush=True,
            )
        elif prefix_band_weights is None and prefix_count > 0:
            # Backward compat: single prefix_band_weights -> 1-phase list
            _yaml_prefix = stage_cfg.get("prefix_band_weights")
            if _yaml_prefix:
                _yaml_phases = [{"weights": _yaml_prefix, "count": prefix_count}]
                print(
                    "[MapDataset] Converted prefix_band_weights to single-phase prefix_phases",
                    flush=True,
                )

        # Load manifest
        manifest_path = os.path.join(manifest_dir, "curriculum_v2_manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Load shard lists (rank-striped)
        pool_shards: Dict[str, List[str]] = {}
        manifest_pools = manifest["pools"]

        for pool_name in self.OPUS_POOLS:
            pool_def = manifest_pools[pool_name]
            pool_shards[pool_name] = self._load_shards(
                os.path.join(manifest_dir, pool_def["shard_list_file"])
            )

        aon_def = manifest_pools["AON"]["sub_pools"]
        pool_shards["AON_bench"] = self._load_shards(
            os.path.join(manifest_dir, aon_def["bench_train"]["shard_list_file"])
        )
        pool_shards["AON_indic"] = self._load_shards(
            os.path.join(manifest_dir, aon_def["indic_guaranteed"]["shard_list_file"])
        )

        # Build pool names + weights (same logic as IterableDataset)
        pool_names: List[str] = []
        pool_weights: List[float] = []

        if mode == "combined":
            # AON pools are NOT in combined index.
            # They come via train.py AON injection only.
            # Including them here causes double Indic.
            for name in self.OPUS_POOLS:
                pool_names.append(name)
                pool_weights.append(weights[name])
        elif mode == "opus_candidates":
            for name in self.OPUS_POOLS:
                pool_names.append(name)
                pool_weights.append(weights[name])
        elif mode == "always_on":
            aon_split = curriculum["aon_config"]["internal_split"]
            pool_names.append("AON_bench")
            pool_weights.append(aon_split["bench_train"])
            pool_names.append("AON_indic")
            pool_weights.append(aon_split["indic_guaranteed"])
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        # Normalize weights
        total = sum(pool_weights)
        pool_weights = [w / total for w in pool_weights]

        self._pool_names = pool_names

        # Load Indic language map for per-sample mixing (if available)
        indic_language_shards = self._load_indic_language_map(
            manifest_dir, pool_shards.get("AON_indic", [])
        )

        # Load exclusion set (sequences consumed by previous stage)
        _exclusion_set = None
        if exclusion_file and os.path.exists(exclusion_file):
            with open(exclusion_file, "r") as f:
                excl_data = json.load(f)
            _exclusion_set = set()
            for _shard_rel, _indices in excl_data["exclusions"].items():
                for _idx in _indices:
                    _exclusion_set.add((_shard_rel, _idx))
            print(
                f"[MapDataset] Loaded exclusion set: {len(_exclusion_set)} sequences "
                f"from {exclusion_file}",
                flush=True,
            )

        # Load v2 supplemental exclusion file (dict format: shard_rel -> [idx, ...])
        _v2_path = "/tmp/8b_data_exclusions_v2.json"
        if os.path.exists(_v2_path):
            with open(_v2_path, "r") as f:
                _v2_data = json.load(f)
            _v2_excl = _v2_data.get("exclusions", {})
            _v2_set = set()
            for _shard_rel, _indices in _v2_excl.items():
                for _idx in _indices:
                    _v2_set.add((_shard_rel, _idx))
            if _exclusion_set is None:
                _exclusion_set = _v2_set
            else:
                _exclusion_set = _exclusion_set | _v2_set
            print(
                f"[MapDataset] Loaded v2 exclusion set: {len(_v2_set)} sequences "
                f"from {_v2_path} (total: {len(_exclusion_set)})",
                flush=True,
            )

        # Build multi-phase prefix for index building.
        # Each phase = (normalized_weight_list, count)
        _prefix_phases_list = None
        if _yaml_phases and prefix_count > 0 and mode == "combined":
            _prefix_phases_list = []
            for phase_cfg in _yaml_phases:
                _pw_dict = phase_cfg["weights"]
                _pc = phase_cfg["count"]
                _pw_list = []
                for name in pool_names:
                    _pw_list.append(_pw_dict.get(name, 0.0))
                _pw_sum = sum(_pw_list)
                if _pw_sum > 0:
                    _pw_list = [w / _pw_sum for w in _pw_list]
                _prefix_phases_list.append((_pw_list, _pc))
                print(
                    f"[MapDataset] Phase: count={_pc}, "
                    f"weights={dict(zip(pool_names, _pw_list))}",
                    flush=True,
                )
            # Verify total prefix count matches resume_index
            _total_prefix = sum(pc for _, pc in _prefix_phases_list)
            if _total_prefix != prefix_count:
                print(
                    f"[MapDataset] WARNING: prefix_phases total count ({_total_prefix}) "
                    f"!= prefix_count ({prefix_count}). Using prefix_phases counts.",
                    flush=True,
                )

        # Pre-compute the full sequence index
        print(
            f"[MapDataset] Building sequence index (rank={rank}, "
            f"world_size={world_size}, seed={seed})...",
            flush=True,
        )
        self._index = _build_sequence_index(
            pool_shards=pool_shards,
            pool_names=pool_names,
            pool_weights=pool_weights,
            shard_root=shard_root,
            dtype=self._dtype,
            seq_len=seq_len,
            seed=seed + rank,  # different seed per rank
            max_sequences=max_sequences,
            indic_language_shards=indic_language_shards,
            exclusion_set=_exclusion_set,
            recycle_pools=self.RECYCLE_POOLS,
            prefix_phases=_prefix_phases_list,
        )
        print(f"[MapDataset] Index built: {len(self._index)} sequences", flush=True)

    def _load_indic_language_map(
        self, manifest_dir: str, indic_shards: List[str]
    ) -> Optional[Dict[str, List[str]]]:
        """Load indic_language_map.json and group shards by language.

        Returns a dict of {language: [shard_rel_paths]} for the shards
        assigned to this rank, or None if the language map doesn't exist
        (falls back to sequential reading).
        """
        lang_map_path = os.path.join(manifest_dir, "indic_language_map.json")
        if not os.path.exists(lang_map_path):
            print(
                "[MapDataset] No indic_language_map.json found — "
                "AON_indic will use sequential reading",
                flush=True,
            )
            return None

        with open(lang_map_path, "r") as f:
            shard_to_lang: Dict[str, str] = json.load(f)

        # Group this rank's indic shards by language
        # indic_shards are already rank-striped, format: "band_indic_numerals/shard_XXXXXX"
        # or "band_D5_wiki_XX/shard_XXXXXX" for D5 Indic wikis.
        # Language map keys: old indic_numerals use bare "shard_XXXXXX",
        # D5 wikis use full "band_D5_wiki_XX/shard_XXXXXX".
        # Try full path first, then fall back to bare shard name.
        lang_groups: Dict[str, List[str]] = {}
        mapped = 0
        unmapped = 0
        for shard_rel in indic_shards:
            lang = shard_to_lang.get(shard_rel)
            if lang is None:
                shard_name = shard_rel.split("/")[-1] if "/" in shard_rel else shard_rel
                lang = shard_to_lang.get(shard_name)
            if lang is None:
                # No annotation — put in "unknown" bucket
                lang = "unknown"
                unmapped += 1
            else:
                mapped += 1
            lang_groups.setdefault(lang, []).append(shard_rel)

        if not lang_groups:
            return None

        n_langs = len(lang_groups)
        print(
            f"[MapDataset] Indic language map loaded: {mapped} mapped, "
            f"{unmapped} unmapped, {n_langs} languages for rank {self.rank}",
            flush=True,
        )

        return lang_groups

    def _load_shards(self, path: str) -> List[str]:
        """Load shard paths from manifest, stripe by rank."""
        with open(path, "r") as f:
            all_shards = [line.strip() for line in f if line.strip()]
        return all_shards[self.rank :: self.world_size]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pool_name, shard_rel, seq_idx = self._index[idx]
        bin_path = os.path.join(self.shard_root, shard_rel, "tokens.bin")

        tokens = _read_sequence_at(bin_path, self._dtype, self.seq_len, seq_idx)

        return {
            "input_ids": tokens,
            "labels": tokens.clone(),
            "_pool": pool_name,
            "_global_idx": torch.tensor(idx, dtype=torch.long),
        }

    def lookup_step(self, global_step: int, batch_size: int) -> List[Dict[str, Any]]:
        """Look up which (pool, shard, seq_idx) were in a given training step.

        Args:
            global_step: The training step number.
            batch_size: Micro-batch size per GPU (sequences per step per rank).

        Returns:
            List of dicts with pool, shard, seq_idx, and index for each
            sequence in the batch at that step.
        """
        start_idx = global_step * batch_size
        end_idx = min(start_idx + batch_size, len(self._index))
        results = []
        for idx in range(start_idx, end_idx):
            pool_name, shard_rel, seq_idx = self._index[idx]
            results.append(
                {
                    "index": idx,
                    "pool": pool_name,
                    "shard": shard_rel,
                    "seq_idx": seq_idx,
                    "bin_path": os.path.join(self.shard_root, shard_rel, "tokens.bin"),
                }
            )
        return results

    def mark_batch_consumed(self, batch: Dict[str, Any]) -> None:
        """Track the last consumed batch index for checkpointing.

        Called by train_epoch after each step. The _global_idx field in the
        batch tells us exactly where we are in the pre-computed index.
        """
        if "_global_idx" in batch:
            # _global_idx is [batch_size] tensor; take the max (last in batch)
            self._consumed_index = int(batch["_global_idx"].max().item()) + 1

    def get_shard_state(self, use_consumed: bool = False) -> Dict[str, Any]:
        """Return checkpoint state — just the resume index.

        Compatible with the curriculum_v2 iterable dataset interface so
        train.py's checkpoint logic works without changes.
        """
        return {
            "loader_type": "curriculum_v2_map",
            "resume_index": self._consumed_index,
            "total_sequences": len(self._index),
        }


# ---------------------------------------------------------------------------
# Sampler that starts from a given offset (for resume)
# ---------------------------------------------------------------------------


class ResumableSequentialSampler(Sampler):
    """Sequential sampler that starts from a given offset."""

    def __init__(self, data_source, start_idx: int = 0):
        self.data_source = data_source
        self.start_idx = start_idx

    def __iter__(self):
        return iter(range(self.start_idx, len(self.data_source)))

    def __len__(self):
        return len(self.data_source) - self.start_idx


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_curriculum_v2_map_dataloader(
    shard_root: str,
    manifest_dir: str,
    curriculum_path: str,
    stage: str,
    batch_size: int = 1,
    seq_len: int = SHARD_BLOCK_SIZE,
    rank: int = 0,
    world_size: int = 1,
    mode: str = "combined",
    seed: int = 42,
    num_workers: int = 0,
    resume_index: int = 0,
    max_sequences: Optional[int] = None,
    exclusion_file: Optional[str] = None,
    prefix_band_weights: Optional[Dict[str, float]] = None,
    prefix_count: int = 0,
) -> Tuple[CurriculumDatasetV2Map, DataLoader]:
    """Build a Map-style DataLoader for curriculum training.

    Returns (dataset, dataloader) so the caller can access dataset.__len__()
    and the current index for checkpointing.

    Dual-weight support:
        prefix_band_weights: dict of {pool_name: weight} for the OLD
            distribution (used for index positions 0..prefix_count-1).
        prefix_count: number of index positions to build with old weights.
            Typically set to the resume_index so the prefix exactly
            replays the consumed data distribution.
    """
    dataset = CurriculumDatasetV2Map(
        shard_root=shard_root,
        manifest_dir=manifest_dir,
        curriculum_path=curriculum_path,
        stage=stage,
        seq_len=seq_len,
        rank=rank,
        world_size=world_size,
        mode=mode,
        seed=seed,
        max_sequences=max_sequences,
        exclusion_file=exclusion_file,
        prefix_band_weights=prefix_band_weights,
        prefix_count=prefix_count,
    )

    sampler = ResumableSequentialSampler(dataset, start_idx=resume_index)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataset, loader
