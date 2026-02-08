"""
synth_adapter.py — Adapter for PleIAs/SYNTH Dataset

Converts SYNTH dataset format to our internal format for use in the pipeline.

SYNTH Dataset: https://huggingface.co/datasets/PleIAs/SYNTH
- 79M samples with CoT reasoning
- Multilingual (35 languages)
- Based on Wikipedia article amplification

Field Mapping:
    SYNTH               ->  Our Format
    -----                   ----------
    query               ->  question
    synthetic_reasoning ->  think_view
    synthetic_answer    ->  distilled_view
    language            ->  language
    exercise            ->  skill_bucket (inferred)
    -                   ->  band (inferred from content)

Usage:
    from integration.synth_adapter import SynthAdapter

    adapter = SynthAdapter()
    samples = adapter.load_samples(num_samples=1000, language="en")
    adapter.save_to_bank(samples, output_dir="./synth_data_bank")
"""

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator

# Add parent to path
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))


# Mapping from SYNTH exercise types to our skill buckets
EXERCISE_TO_SKILL = {
    "memorization": "KNOW-FACTUAL",
    "reasoning": "RSN-LOGIC",
    "math": "RSN-ARITHMETIC",
    "algebra": "RSN-ALGEBRA",
    "coding": "CODE-COMPLETION",
    "code": "CODE-COMPLETION",
    "translation": "LANG-GRAMMAR",
    "grammar": "LANG-GRAMMAR",
    "summarization": "LANG-COHERENCE",
    "explanation": "CODE-EXPLAIN",
    "science": "KNOW-SCIENCE",
    "qa": "KNOW-COMMONSENSE",
}

# Keywords for skill inference if exercise type is unclear
SKILL_KEYWORDS = {
    "RSN-ARITHMETIC": ["calculate", "sum", "multiply", "divide", "add", "subtract", "number"],
    "RSN-ALGEBRA": ["equation", "solve for x", "variable", "polynomial", "factor"],
    "RSN-LOGIC": ["therefore", "conclude", "implies", "if-then", "logical"],
    "RSN-CAUSAL": ["because", "caused by", "leads to", "result of", "effect"],
    "CODE-COMPLETION": ["def ", "function", "class ", "import ", "return "],
    "CODE-DEBUG": ["error", "bug", "fix", "exception", "traceback"],
    "CODE-ALGO": ["algorithm", "sort", "search", "complexity", "O(n)"],
    "LANG-GRAMMAR": ["grammar", "syntax", "sentence", "correct form"],
    "KNOW-SCIENCE": ["physics", "chemistry", "biology", "science", "experiment"],
    "KNOW-FACTUAL": ["who", "when", "where", "what year", "capital of"],
}

# Band inference based on content complexity
def infer_band(text: str, reasoning: str = "") -> str:
    """Infer difficulty band from content."""
    word_count = len(text.split())
    reasoning_len = len(reasoning) if reasoning else 0

    # Check for code (usually higher bands)
    if re.search(r'def |class |function |import |```', text):
        if reasoning_len > 1000:
            return "B5"
        return "B4"

    # Check for math/equations
    if re.search(r'[=+\-*/^].*\d|solve|equation', text.lower()):
        if reasoning_len > 500:
            return "B4"
        return "B3"

    # Based on word count and reasoning length
    if reasoning_len > 1500:
        return "B5"
    elif reasoning_len > 500:
        return "B4"
    elif reasoning_len > 200:
        return "B3"
    elif word_count > 100:
        return "B2"
    elif word_count > 30:
        return "B1"
    else:
        return "B0"


def infer_skill(exercise: str, query: str, answer: str = "") -> str:
    """Infer skill bucket from exercise type and content."""
    # First try exercise type mapping
    exercise_lower = exercise.lower() if exercise else ""
    for key, skill in EXERCISE_TO_SKILL.items():
        if key in exercise_lower:
            return skill

    # Fallback to keyword matching
    combined = f"{query} {answer}".lower()
    best_skill = "KNOW-FACTUAL"  # Default
    best_score = 0

    for skill, keywords in SKILL_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in combined)
        if score > best_score:
            best_score = score
            best_skill = skill

    return best_skill


@dataclass
class SynthSample:
    """Converted SYNTH sample in our format."""
    id: str
    question: str
    answer: str
    distilled_view: str
    think_view: Optional[str]
    skill_bucket: str
    band: str
    language: str
    source: str = "synth"
    verified: bool = False
    verification_score: Optional[float] = None
    hard_negative: Optional[dict] = None
    error_correction: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "distilled_view": self.distilled_view,
            "think_view": self.think_view,
            "skill_bucket": self.skill_bucket,
            "band": self.band,
            "language": self.language,
            "source": self.source,
            "verified": self.verified,
            "verification_score": self.verification_score,
            "hard_negative": self.hard_negative,
            "error_correction": self.error_correction,
        }


class SynthAdapter:
    """Adapter for loading and converting PleIAs/SYNTH dataset."""

    def __init__(self, cache_dir: str = "./.synth_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self, streaming: bool = True):
        """Load the SYNTH dataset from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install 'datasets' library: pip install datasets")

        print("[SYNTH] Loading dataset from HuggingFace (streaming)...")
        ds = load_dataset(
            "PleIAs/SYNTH",
            split="train",
            streaming=streaming,
        )
        return ds

    def convert_row(self, row: dict, index: int = 0) -> SynthSample:
        """Convert a SYNTH row to our format."""
        query = row.get("query", "")
        reasoning = row.get("synthetic_reasoning", "")
        answer = row.get("synthetic_answer", "")
        exercise = row.get("exercise", "")
        language = row.get("language", "en")
        synth_id = row.get("synth_id", f"synth_{index}")

        # Infer skill and band
        skill = infer_skill(exercise, query, answer)
        band = infer_band(query, reasoning)

        # Create distilled view (short answer)
        distilled = answer
        if len(answer) > 500:
            # Truncate long answers but keep first paragraph
            first_para = answer.split('\n\n')[0]
            if len(first_para) > 500:
                distilled = first_para[:500] + "..."
            else:
                distilled = first_para

        return SynthSample(
            id=synth_id,
            question=query,
            answer=answer[:200] if answer else "",  # Short answer extract
            distilled_view=distilled,
            think_view=reasoning if reasoning else None,
            skill_bucket=skill,
            band=band,
            language=language,
        )

    def load_samples(
        self,
        num_samples: int = 1000,
        language: str = None,
        skills: list[str] = None,
        bands: list[str] = None,
        min_reasoning_length: int = 100,
    ) -> list[SynthSample]:
        """Load and filter samples from SYNTH dataset.

        Args:
            num_samples: Maximum number of samples to load
            language: Filter by language (e.g., "en", "de")
            skills: Filter by skill buckets
            bands: Filter by bands
            min_reasoning_length: Minimum length of synthetic_reasoning

        Returns:
            List of converted samples
        """
        ds = self.load_dataset(streaming=True)

        samples = []
        seen = 0
        skipped = 0

        print(f"[SYNTH] Loading up to {num_samples} samples...")
        if language:
            print(f"  Language filter: {language}")
        if skills:
            print(f"  Skill filter: {skills}")
        if bands:
            print(f"  Band filter: {bands}")

        for i, row in enumerate(ds):
            seen += 1

            # Language filter
            if language and row.get("language", "") != language:
                skipped += 1
                continue

            # Reasoning length filter
            reasoning = row.get("synthetic_reasoning", "")
            if len(reasoning) < min_reasoning_length:
                skipped += 1
                continue

            # Convert sample
            sample = self.convert_row(row, i)

            # Skill filter
            if skills and sample.skill_bucket not in skills:
                skipped += 1
                continue

            # Band filter
            if bands and sample.band not in bands:
                skipped += 1
                continue

            samples.append(sample)

            if len(samples) % 100 == 0:
                print(f"  [{len(samples)}/{num_samples}] seen={seen}, skipped={skipped}", end="\r")

            if len(samples) >= num_samples:
                break

        print(f"\n[Done] Loaded {len(samples)} samples (seen={seen}, skipped={skipped})")
        return samples

    def save_to_bank(
        self,
        samples: list[SynthSample],
        output_dir: str = "./synth_data_bank",
        skill_grouping: bool = True,
    ):
        """Save samples to the data bank format.

        Args:
            samples: List of converted samples
            output_dir: Output directory for shards
            skill_grouping: If True, group by skill into separate files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if skill_grouping:
            # Group by skill
            by_skill = {}
            for s in samples:
                if s.skill_bucket not in by_skill:
                    by_skill[s.skill_bucket] = []
                by_skill[s.skill_bucket].append(s)

            # Save each skill shard
            for skill, skill_samples in by_skill.items():
                shard_path = output_path / f"{skill}_synth.jsonl"
                with open(shard_path, "w", encoding="utf-8") as f:
                    for s in skill_samples:
                        f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
                print(f"  [Saved] {skill}: {len(skill_samples)} samples -> {shard_path.name}")

            # Update manifest
            self._update_manifest(output_path, by_skill)
        else:
            # Single file
            shard_path = output_path / "synth_external.jsonl"
            with open(shard_path, "w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
            print(f"[Saved] {len(samples)} samples -> {shard_path}")

    def _update_manifest(self, bank_dir: Path, by_skill: dict):
        """Update bank manifest with SYNTH samples."""
        manifest_path = bank_dir / "manifest.json"

        # Load existing manifest
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
        else:
            manifest = {"created": datetime.now().isoformat(), "model": "synth", "skills": {}}

        manifest["updated"] = datetime.now().isoformat()
        manifest["synth_added"] = datetime.now().isoformat()

        # Add SYNTH skill entries
        from common import cot_allowed_for_band, get_skill_bucket

        for skill, samples in by_skill.items():
            # Get primary band from samples
            bands = [s.band for s in samples]
            primary_band = max(set(bands), key=bands.count)

            # OLD: used raw skill name from EXERCISE_TO_SKILL (e.g. "KNOW-FACTUAL")
            #      → manifest got legacy keys like "KNOW-FACTUAL_synth"
            # NEW: resolve to canonical ID before writing to manifest
            try:
                canonical_skill = get_skill_bucket(skill).id
            except ValueError:
                canonical_skill = skill  # keep as-is if unresolvable

            entry_key = f"{canonical_skill}_synth"
            manifest["skills"][entry_key] = {
                "samples": len(samples),
                "band": primary_band,
                "cot_allowed": cot_allowed_for_band(primary_band),
                "with_hard_negatives": 0,
                # OLD: shard_file used raw skill name
                # NEW: shard_file still uses original name (file on disk unchanged)
                "shard_file": f"{skill}_synth.jsonl",
                "priority": "medium",
                "source": "PleIAs/SYNTH",
            }

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"[Manifest] Updated with {len(by_skill)} SYNTH skill groups")


def convert_synth_to_bank(
    num_samples: int = 1000,
    output_dir: str = "./synth_data_bank",
    language: str = "en",
    skills: list[str] = None,
    bands: list[str] = None,
):
    """Convenience function to load and save SYNTH data.

    Args:
        num_samples: Number of samples to load
        output_dir: Output directory
        language: Language filter
        skills: Skill bucket filter
        bands: Band filter
    """
    adapter = SynthAdapter()
    samples = adapter.load_samples(
        num_samples=num_samples,
        language=language,
        skills=skills,
        bands=bands,
    )
    adapter.save_to_bank(samples, output_dir)
    return len(samples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Import SYNTH dataset")
    parser.add_argument("--num", "-n", type=int, default=1000)
    parser.add_argument("--output", "-o", default="./synth_data_bank")
    parser.add_argument("--language", "-l", default="en")
    parser.add_argument("--skills", nargs="*")
    parser.add_argument("--bands", nargs="*")

    args = parser.parse_args()
    convert_synth_to_bank(
        num_samples=args.num,
        output_dir=args.output,
        language=args.language,
        skills=args.skills,
        bands=args.bands,
    )
