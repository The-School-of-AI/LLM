
"""
curriculum_tools.py

A lightweight, dataset-agnostic toolkit for:
1) Extracting cheap difficulty + modality signals from text
2) Mapping documents to difficulty bands B0..B5 (quantile-based)
3) Computing stage-wise band proportions using:
   - Capacity–difficulty alignment (target median/quantile anchoring)
   - KL-divergence regularization to a base distribution
   - Floors/Caps constraints
4) Map/Reduce helpers for trillion-token style pipelines

Design goals:
- Stateless per-sample map functions
- Single-pass text processing where possible
- No model training required
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


# -----------------------------
# Bands & centroids
# -----------------------------

BANDS: List[str] = ["B0", "B1", "B2", "B3", "B4", "B5"]

# Difficulty centroids (0..1) for alignment/targets (tune as you learn)
BAND_DIFFICULTY: Dict[str, float] = {
    "B0": 0.10,
    "B1": 0.225,
    "B2": 0.40,
    "B3": 0.60,
    "B4": 0.775,
    "B5": 0.925,
}


# -----------------------------
# Fast tokenization / sentence counting
# -----------------------------

_WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)

def tokenize_words(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())

def rough_sentence_count(text: str) -> int:
    """
    Cheap sentence count that avoids splitting too aggressively on code/URLs.
    Not perfect, but stable + fast.
    """
    parts = re.split(r"[.!?]+\s+(?=[A-Za-z0-9])", text.strip())
    return max(1, len([p for p in parts if p.strip()]))


# -----------------------------
# Modality detection (tags)
# -----------------------------

RE_CODE_FENCE = re.compile(r"```")
RE_CODE = re.compile(
    r"```|\bdef\s+\w+\(|\bclass\s+\w+|#include\s+<|\bfunction\s+\w+\(|\bSELECT\b.+\bFROM\b",
    re.IGNORECASE | re.MULTILINE,
)
RE_MATH = re.compile(r"[∑∫√≈≠≤≥→∞]|\\(frac|sum|int|sqrt|begin\{equation\})", re.IGNORECASE)
RE_AGENTIC = re.compile(
    r'"\s*(tool|action|observation|arguments)\s*"\s*:|\bThought:\b|\bAction:\b|\bObservation:\b',
    re.IGNORECASE,
)
RE_COT = re.compile(r"let's think step by step|step by step|therefore|reasoning:", re.IGNORECASE)

def detect_modalities(text: str) -> Dict[str, bool]:
    """
    Tags are orthogonal to difficulty. Do NOT map these directly to B3/B5.
    """
    return {
        "code": bool(RE_CODE.search(text)),
        "math": bool(RE_MATH.search(text)),
        "agentic_traces": bool(RE_AGENTIC.search(text)),
        "cot_reasoning": bool(RE_COT.search(text)),
    }


# -----------------------------
# Cheap difficulty features
# -----------------------------

@lru_cache(maxsize=200_000)
def _syllables(word: str) -> int:
    return max(1, len(re.findall(r"[aeiouy]+", word)))

def flesch_kincaid_grade(tokens: List[str], sentence_count: int) -> float:
    """
    FK grade is English-centric; treat as a weak signal.
    """
    if len(tokens) < 10:
        return 0.0
    syllables = sum(_syllables(t) for t in tokens)
    return 0.39 * (len(tokens) / max(1, sentence_count)) + 11.8 * (syllables / len(tokens)) - 15.59

def char_entropy(text: str, max_chars: int = 4000) -> float:
    """
    Cheap proxy for technicality/noise: code/math/symbol-heavy text often higher entropy.
    """
    s = text[:max_chars]
    if not s:
        return 0.0
    freq: Dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    ent = 0.0
    for c in freq.values():
        p = c / n
        ent -= p * math.log(p + 1e-12)
    return ent

def rare_ratio(tokens: List[str]) -> float:
    """
    Notebook version counts tokens with frequency==1. Keep as a weak signal.
    For multilingual, consider replacing with subword stats later.
    """
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    rare = sum(1 for t in tokens if freq[t] == 1)
    return rare / len(tokens)

@dataclass
class DocSignals:
    num_tokens: int
    avg_sentence_len: float
    fk_grade: float
    entropy: float
    rare_ratio: float
    modalities: Dict[str, bool]

def extract_signals(text: str, *, min_tokens_early_b0: int = 40) -> Tuple[Optional[str], DocSignals]:
    """
    Single entry point for signal extraction.
    Returns (early_band, signals).
    """
    t = (text or "").strip()
    tokens = tokenize_words(t)
    n_tok = len(tokens)
    s_count = rough_sentence_count(t)

    # Early exit: short texts behave like B0/B1; keep conservative
    if n_tok < min_tokens_early_b0:
        sig = DocSignals(
            num_tokens=n_tok,
            avg_sentence_len=n_tok / max(1, s_count),
            fk_grade=0.0,
            entropy=char_entropy(t),
            rare_ratio=0.0,
            modalities=detect_modalities(t),
        )
        return "B0", sig

    mods = detect_modalities(t)
    avg_sent = n_tok / max(1, s_count)

    # Compute FK only when potentially informative (keeps notebook’s “signals-first” intent)
    fk = 0.0
    if mods["code"] or mods["math"] or avg_sent > 18:
        fk = flesch_kincaid_grade(tokens, s_count)

    sig = DocSignals(
        num_tokens=n_tok,
        avg_sentence_len=avg_sent,
        fk_grade=fk,
        entropy=char_entropy(t),
        rare_ratio=rare_ratio(tokens),
        modalities=mods,
    )
    return None, sig

def difficulty_score(sig: DocSignals) -> float:
    """
    Continuous score. Modalities are soft bumps (NOT hard band jumps).
    """
    score = 0.0
    score += 0.55 * sig.fk_grade
    score += 0.03 * sig.avg_sentence_len
    score += 0.40 * sig.entropy
    score += 0.80 * sig.rare_ratio

    # soft modality bumps
    if sig.modalities.get("code"): score += 1.0
    if sig.modalities.get("math"): score += 1.3
    if sig.modalities.get("cot_reasoning"): score += 0.4
    if sig.modalities.get("agentic_traces"): score += 1.8

    return float(score)


# -----------------------------
# Score -> Band (quantile-based)
# -----------------------------

def compute_quantile_edges(scores: List[float],
                           cutpoints: List[float] = [0.15, 0.30, 0.50, 0.70, 0.85]) -> List[float]:
    if not scores:
        return [0, 0, 0, 0, 0]
    s = sorted(scores)
    edges = []
    for p in cutpoints:
        idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
        edges.append(s[idx])
    return edges

def assign_band_from_edges(score: float, edges: List[float]) -> str:
    for i, e in enumerate(edges):
        if score <= e:
            return BANDS[i]
    return "B5"


# -----------------------------
# Map/Reduce helpers (dataset-agnostic)
# -----------------------------

def map_doc_to_band(sample: dict,
                    *,
                    text_key: str = "text",
                    edges: Optional[List[float]] = None) -> dict:
    """
    Stateless map function. Requires quantile edges computed from a calibration sample.
    If edges is None, returns score + signals (for calibration pass).
    """
    text = sample.get(text_key, "")
    early, sig = extract_signals(text)
    score = difficulty_score(sig)

    if edges is None:
        return {"score": score, "tokens": sig.num_tokens, "modalities": sig.modalities, "early_band": early}

    band = early if early is not None else assign_band_from_edges(score, edges)
    return {"band": band, "tokens": sig.num_tokens, "modalities": sig.modalities}

def reduce_band_distribution(rows: Iterable[dict]) -> Dict[str, float]:
    """
    Aggregate token proportions per band. Always returns all B0..B5 keys (missing => 0).
    """
    totals = defaultdict(int)
    total_tokens = 0

    for r in rows:
        b = r.get("band")
        t = int(r.get("tokens", 0))
        if b is None:
            continue
        totals[b] += t
        total_tokens += t

    if total_tokens <= 0:
        raise ValueError("Total token count is zero; cannot form distribution.")

    dist = {b: totals.get(b, 0) / total_tokens for b in BANDS}
    return dist


# -----------------------------
# Capacity -> target distribution (median/quantile anchored)
# -----------------------------

def model_capacity(params: float, min_params: float = 1e9, max_params: float = 70e9) -> float:
    """
    Normalized capacity in [0,1] using log scaling.
    """
    p = max(min_params, min(max_params, params))
    return (math.log(p) - math.log(min_params)) / (math.log(max_params) - math.log(min_params))

def target_median_difficulty(capacity: float) -> float:
    """
    Conservative: median moves from ~0.25 (B1-ish) to ~0.75 (B4-ish) as capacity grows.
    Tune if you observe faster/slower difficulty absorption.
    """
    return 0.25 + 0.50 * capacity

def capacity_target_distribution(capacity: float, sharpness: float = 10.0) -> Dict[str, float]:
    """
    Create a target distribution peaked around target median difficulty.
    """
    m = target_median_difficulty(capacity)
    raw = {b: math.exp(-sharpness * abs(BAND_DIFFICULTY[b] - m)) for b in BANDS}
    z = sum(raw.values()) + 1e-12
    return {b: raw[b] / z for b in BANDS}

def distribution_quantile(dist: Dict[str, float], q: float) -> float:
    items = sorted([(BAND_DIFFICULTY[b], dist.get(b, 0.0)) for b in BANDS], key=lambda x: x[0])
    cum = 0.0
    for d, w in items:
        cum += w
        if cum >= q:
            return d
    return items[-1][0]

def distribution_mean(dist: Dict[str, float]) -> float:
    return sum(dist.get(b, 0.0) * BAND_DIFFICULTY[b] for b in BANDS)


# -----------------------------
# KL-regularized mixture optimization
# -----------------------------

def kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    eps = 1e-12
    out = 0.0
    for b in BANDS:
        pb = max(eps, p.get(b, 0.0))
        qb = max(eps, q.get(b, 0.0))
        out += pb * math.log(pb / qb)
    return out

def apply_floors_caps(w: Dict[str, float],
                      floors: Optional[Dict[str, float]] = None,
                      caps: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    floors = floors or {}
    caps = caps or {}
    x = {b: float(w.get(b, 0.0)) for b in BANDS}

    for b in BANDS:
        if b in floors:
            x[b] = max(x[b], floors[b])
        if b in caps:
            x[b] = min(x[b], caps[b])

    s = sum(x.values())
    if s <= 0:
        return {b: 1.0 / len(BANDS) for b in BANDS}
    return {b: x[b] / s for b in BANDS}

def optimize_band_weights(base: Dict[str, float],
                          target: Dict[str, float],
                          *,
                          floors: Optional[Dict[str, float]] = None,
                          caps: Optional[Dict[str, float]] = None,
                          lambda_kl: float = 0.4,
                          q_anchors: Tuple[float, ...] = (0.50,),
                          q_weight: float = 3.0,
                          mean_weight: float = 1.0,
                          steps: int = 200,
                          lr: float = 0.08) -> Dict[str, float]:
    """
    Optimize w to:
    - match target median/quantile(s) difficulty
    - match target mean difficulty
    - stay close to base via KL(w || base)
    - satisfy floors/caps

    This avoids training proxy models while still being principled.
    """
    w = apply_floors_caps(base, floors, caps)

    target_q = {q: distribution_quantile(target, q) for q in q_anchors}
    target_mean = distribution_mean(target)

    for _ in range(steps):
        w_mean = distribution_mean(w)
        dm = (w_mean - target_mean)

        dq = {q: (distribution_quantile(w, q) - target_q[q]) for q in q_anchors}

        grad = {}
        for b in BANDS:
            wb = max(1e-12, w.get(b, 0.0))
            bb = max(1e-12, base.get(b, 0.0))

            # KL gradient ~ log(w/base) + 1
            g_kl = (math.log(wb / bb) + 1.0)

            # Mean mismatch pushes mass toward lower/higher centroids
            g_mean = dm * (BAND_DIFFICULTY[b] - target_mean)

            # Quantile mismatch (coarse directional)
            g_q = 0.0
            for q in q_anchors:
                g_q += dq[q] * (BAND_DIFFICULTY[b] - target_q[q])

            grad[b] = lambda_kl * g_kl + mean_weight * g_mean + q_weight * g_q

        # multiplicative update (keeps positivity)
        new_w = {b: w[b] * math.exp(-lr * grad[b]) for b in BANDS}
        s = sum(new_w.values()) + 1e-12
        new_w = {b: new_w[b] / s for b in BANDS}

        w = apply_floors_caps(new_w, floors, caps)

    return w

def compute_stage_band_weights(base_distribution: Dict[str, float],
                               *,
                               params: float,
                               floors: Optional[Dict[str, float]] = None,
                               caps: Optional[Dict[str, float]] = None,
                               lambda_kl: float = 0.4,
                               q_anchors: Tuple[float, ...] = (0.50,),
                               sharpness: float = 10.0) -> Dict[str, float]:
    cap = model_capacity(params)
    target = capacity_target_distribution(capacity=cap, sharpness=sharpness)
    return optimize_band_weights(
        base=base_distribution,
        target=target,
        floors=floors,
        caps=caps,
        lambda_kl=lambda_kl,
        q_anchors=q_anchors,
    )
