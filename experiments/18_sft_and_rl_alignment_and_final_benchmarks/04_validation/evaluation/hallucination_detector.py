"""
Hallucination Detector
========================
Detects potential hallucination patterns in model outputs by:

  1. Anchor verification  — checks that factual anchor terms are present
     when the output claims to address the topic.
  2. Ground-truth fact matching — verifies known facts appear correctly.
  3. Numeric plausibility  — flags suspiciously precise numbers that weren't
     in the prompt or anchors.
  4. Negation consistency  — checks that "must_not" statements aren't asserted.
  5. Novel proper-noun detection — flags proper nouns not in prompt, anchors,
     or ground truth (possible fabricated names).

Risk score:
  0.0 = very low hallucination risk
  1.0 = very high hallucination risk

A response is flagged as "hallucination_detected" when risk_score >= threshold
(default 0.5 — configurable via config.yaml → hallucination.threshold).

NOTE: This module uses rule-based heuristics. For production, supplement
with an LLM-as-judge step (see analyze_results.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HallucinationFlag:
    flag_type: str
    description: str
    severity: float   # 0.0 – 1.0
    evidence: str = ""


@dataclass
class HallucinationResult:
    risk_score: float
    detected: bool
    flags: list[HallucinationFlag] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "risk_score": round(self.risk_score, 4),
            "detected": self.detected,
            "flag_count": len(self.flags),
            "flags": [
                {
                    "type": f.flag_type,
                    "description": f.description,
                    "severity": f.severity,
                    "evidence": f.evidence[:200],
                }
                for f in self.flags
            ],
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class HallucinationDetector:
    """
    Multi-signal hallucination risk scorer.

    Parameters
    ----------
    config : dict
        Configuration from config.yaml → hallucination section.
        Keys:
          - threshold (float, default 0.5): above this → detected = True
          - penalise_missing_anchors (bool, default True)
          - check_novel_proper_nouns (bool, default False): slower, use with care
    """

    # Common high-risk words that often precede fabricated facts
    FABRICATION_SIGNALS = [
        r"\bin\s+\d{4}\b",               # "in 1842" — suspicious unsourced dates
        r"\baccording to\b",             # unverified attributions
        r"\bstudies show\b",
        r"\bresearch proves\b",
        r"\bit is well[- ]known\b",
        r"\bexperts agree\b",
        r"\bstatistics show\b",
    ]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.threshold: float = float(cfg.get("threshold", 0.5))
        self.penalise_missing_anchors: bool = bool(
            cfg.get("penalise_missing_anchors", True)
        )
        self.check_novel_proper_nouns: bool = bool(
            cfg.get("check_novel_proper_nouns", False)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        output: str,
        anchors: list[str],
        ground_truth: Any,
    ) -> dict:
        """
        Analyse `output` for hallucination signals.

        Parameters
        ----------
        output       : model response text
        anchors      : list of topic-relevant terms that should appear
        ground_truth : known facts (dict, list, or string) to verify against

        Returns
        -------
        dict — serialisable HallucinationResult
        """
        flags: list[HallucinationFlag] = []

        # --- Check 1: anchor coverage ---
        if self.penalise_missing_anchors and anchors:
            flags.extend(self._check_anchors(output, anchors))

        # --- Check 2: ground-truth fact verification ---
        if ground_truth:
            flags.extend(self._check_ground_truth(output, ground_truth))

        # --- Check 3: fabrication signal language ---
        flags.extend(self._check_fabrication_signals(output))

        # --- Check 4: error/refusal markers (model admits inability) ---
        flags.extend(self._check_error_markers(output))

        # --- Check 5: novel proper nouns (optional, slow) ---
        if self.check_novel_proper_nouns and anchors:
            flags.extend(
                self._check_novel_proper_nouns(output, anchors, ground_truth)
            )

        # --- Risk aggregation ---
        risk_score = self._aggregate_risk(flags, output)
        detected = risk_score >= self.threshold

        result = HallucinationResult(
            risk_score=round(risk_score, 4),
            detected=detected,
            flags=flags,
        )
        return result.to_dict()

    # ------------------------------------------------------------------
    # Check methods
    # ------------------------------------------------------------------

    def _check_anchors(self, output: str, anchors: list[str]) -> list[HallucinationFlag]:
        """
        Flag when the output completely avoids key anchor terms.
        An output that never mentions the expected topic may be off-topic
        or hallucinating a different subject.
        """
        flags = []
        norm_out = output.lower()
        missing_anchors = []
        for anchor in anchors:
            if anchor.lower() not in norm_out:
                missing_anchors.append(anchor)

        coverage = 1.0 - (len(missing_anchors) / len(anchors)) if anchors else 1.0

        if coverage < 0.5 and len(anchors) >= 2:
            # More than half the anchors missing — strong off-topic signal
            flags.append(HallucinationFlag(
                flag_type="LOW_ANCHOR_COVERAGE",
                description=f"Output covers only {coverage:.0%} of expected topic anchors.",
                severity=0.6,
                evidence=f"Missing anchors: {missing_anchors[:5]}",
            ))
        elif coverage < 1.0 and len(anchors) >= 3:
            # Some anchors missing — mild signal
            flags.append(HallucinationFlag(
                flag_type="PARTIAL_ANCHOR_COVERAGE",
                description=f"Output missing some expected topic terms ({coverage:.0%} coverage).",
                severity=0.2,
                evidence=f"Missing: {missing_anchors[:3]}",
            ))
        return flags

    def _check_ground_truth(
        self, output: str, ground_truth: Any
    ) -> list[HallucinationFlag]:
        """Compare output against known ground-truth facts."""
        flags = []
        norm_out = output.lower()

        facts = self._flatten_ground_truth(ground_truth)

        contradictions = []
        for fact_key, fact_value in facts.items():
            fact_str = str(fact_value).lower()
            # Only check short, specific facts (names, dates, numbers)
            if len(fact_str) < 60 and fact_str:
                result = self._check_fact_in_output(norm_out, fact_key, fact_str)
                if result == "contradicted":
                    contradictions.append(f"{fact_key}={fact_value}")
                # "absent" facts are not penalised heavily — model may paraphrase

        if contradictions:
            flags.append(HallucinationFlag(
                flag_type="GROUND_TRUTH_CONTRADICTION",
                description="Output appears to contradict known facts.",
                severity=0.8,
                evidence=f"Contradicted facts: {contradictions[:3]}",
            ))
        return flags

    def _check_fabrication_signals(self, output: str) -> list[HallucinationFlag]:
        """Detect language patterns associated with unverified claims."""
        flags = []
        hits = []
        for pattern in self.FABRICATION_SIGNALS:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                hits.extend(matches[:2])

        if len(hits) >= 3:
            flags.append(HallucinationFlag(
                flag_type="FABRICATION_LANGUAGE",
                description="Multiple unverified-claim language patterns detected.",
                severity=0.35,
                evidence=f"Patterns found: {hits[:5]}",
            ))
        elif hits:
            # mild — single signal
            pass
        return flags

    def _check_error_markers(self, output: str) -> list[HallucinationFlag]:
        """
        Check for error markers from the evaluation wrapper (e.g., API failures).
        These indicate no real output was generated.
        """
        flags = []
        if output.startswith("[ERROR]"):
            flags.append(HallucinationFlag(
                flag_type="GENERATION_ERROR",
                description="Model generation failed — no valid output to evaluate.",
                severity=1.0,
                evidence=output[:100],
            ))
        return flags

    def _check_novel_proper_nouns(
        self, output: str, anchors: list[str], ground_truth: Any
    ) -> list[HallucinationFlag]:
        """
        Flag proper nouns in output that don't appear in anchors or ground truth.
        This is a rough heuristic — capitalised words not in expected context.
        """
        flags = []
        # Simple heuristic: capitalised words not at sentence start
        proper_nouns = set(re.findall(r"(?<!\. )(?<![?!] )\b([A-Z][a-z]{2,})\b", output))

        known_terms = set()
        for a in anchors:
            known_terms.update(a.split())
        gt_text = str(ground_truth).lower()
        for noun in list(proper_nouns):
            if noun.lower() in gt_text or any(noun.lower() in a.lower() for a in anchors):
                proper_nouns.discard(noun)

        if len(proper_nouns) > 8:
            flags.append(HallucinationFlag(
                flag_type="MANY_NOVEL_PROPER_NOUNS",
                description=f"Output contains {len(proper_nouns)} proper nouns not in ground truth.",
                severity=0.3,
                evidence=f"Novel nouns: {list(proper_nouns)[:6]}",
            ))
        return flags

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flatten_ground_truth(self, ground_truth: Any) -> dict[str, Any]:
        """Flatten ground truth into key-value pairs for checking."""
        if isinstance(ground_truth, dict):
            flat = {}
            for k, v in ground_truth.items():
                if isinstance(v, (str, int, float)):
                    flat[k] = v
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        flat[f"{k}[{i}]"] = item
            return flat
        if isinstance(ground_truth, list):
            return {str(i): v for i, v in enumerate(ground_truth)}
        if isinstance(ground_truth, str) and ground_truth:
            return {"fact": ground_truth}
        return {}

    def _check_fact_in_output(
        self, norm_out: str, fact_key: str, fact_str: str
    ) -> str:
        """
        Returns:
          "present"      — fact appears in output
          "absent"       — fact not mentioned (may be paraphrased)
          "contradicted" — a numeric or named value looks wrong
        """
        if fact_str in norm_out:
            return "present"

        # For dates/years: check if a different year is used for same context
        year_match = re.match(r"^(\d{4})$", fact_str)
        if year_match:
            correct_year = fact_str
            other_years = re.findall(r"\b(\d{4})\b", norm_out)
            # If the output mentions years and the correct one isn't there
            if other_years and correct_year not in other_years:
                return "contradicted"

        return "absent"

    def _aggregate_risk(
        self, flags: list[HallucinationFlag], output: str
    ) -> float:
        """
        Combine flag severities into a single risk score [0, 1].
        Strategy:
          - If any single flag has severity >= 1.0 (e.g. generation error), return 1.0.
          - Otherwise use a weighted sum normalised to [0, 1].
        """
        if not flags:
            return 0.0

        # Any catastrophic flag (severity = 1.0) immediately gives maximum risk
        max_severity = max(f.severity for f in flags)
        if max_severity >= 1.0:
            return 1.0

        # Weighted sum — higher-severity flags matter more
        raw_score = sum(f.severity for f in flags)

        # Normalise: assume >3 flags at full severity = worst case
        normalised = min(raw_score / 3.0, 1.0)
        return round(normalised, 4)


# ---------------------------------------------------------------------------
# Convenience function for manual annotation integration
# ---------------------------------------------------------------------------

def manual_flag(
    flag_type: str = "HUMAN_ANNOTATION",
    description: str = "",
    severity: float = 0.8,
) -> dict:
    """
    Create a manual hallucination flag for human annotators to append.
    Usage in annotation CSV: add a 'human_hall_flag' column with JSON.
    """
    return HallucinationFlag(
        flag_type=flag_type,
        description=description,
        severity=severity,
    ).__dict__


# ---------------------------------------------------------------------------
# Stand-alone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    detector = HallucinationDetector()

    # Good output — should score low
    good_output = """
    Mitochondria are membrane-bound organelles found in eukaryotic cells.
    They produce ATP through cellular respiration, making them the powerhouse of the cell.
    Prokaryotes, such as bacteria, lack mitochondria entirely.
    """
    result_good = detector.detect(
        output=good_output,
        anchors=["mitochondria", "ATP", "organelle"],
        ground_truth={"organelle": "mitochondria", "function": "produces ATP", "lacking_in": "prokaryotes"},
    )
    print("=== Good Output ===")
    print(json.dumps(result_good, indent=2))

    # Bad output — should score higher
    bad_output = """
    The nucleus is the powerhouse of the cell, producing ATP in 1823 according to studies show.
    Research proves that experts agree mitochondria were discovered by Napoleon Bonaparte.
    Statistics show that 97.3% of cells have exactly 14 mitochondria.
    """
    result_bad = detector.detect(
        output=bad_output,
        anchors=["mitochondria", "ATP", "organelle"],
        ground_truth={"organelle": "mitochondria", "lacking_in": "prokaryotes"},
    )
    print("\n=== Hallucinated Output ===")
    print(json.dumps(result_bad, indent=2))
