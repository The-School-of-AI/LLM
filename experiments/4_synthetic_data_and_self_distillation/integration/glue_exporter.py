"""
glue_exporter.py — Export synth_data_bank to Glue Normalized Format

Converts synth_data_bank JSONL samples to the 11-column normalized document schema
(same as Dolma, Sangraha, NCERT) and writes as zstd-compressed Parquet.

Field Mapping:
    Synth Data Bank      ->  Normalized Schema
    ---------------          ----------------
    id                   ->  id
    (sha256 of text)     ->  hash
    "synthetic"          ->  dataset
    (from skill cat)     ->  domain
    "synth_{cat}_{model}"->  source
    (formatted markdown) ->  text
    language             ->  language
    (json metadata)      ->  metadata
    (export timestamp)   ->  added
    None                 ->  created
    "1.0"                ->  version

Output: zstd Parquet, partitioned by source (Hive-style: source=value)

Usage:
    from integration.glue_exporter import GlueExporter

    exporter = GlueExporter(bank_dir="./synth_data_bank")
    exporter.export(output_path="./normalized_output")

    # Or via CLI:
    python run_pipeline.py export-glue --output ./normalized_output
"""

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent to path
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from common.skills import SKILL_ALIASES, SKILL_BUCKETS, SkillCategory  # noqa: E402

# ================================================================
# CONSTANTS
# ================================================================

VERSION = "1.0"
DATASET_NAME = "synthetic"
OUTPUT_BASE_S3 = "s3://t1-dataacquisition-datasets/processed_dataset/normalized_data"

SKILL_CATEGORY_TO_DOMAIN = {
    SkillCategory.FOUNDATION: "foundation",
    SkillCategory.REASONING: "reasoning",
    SkillCategory.CODE: "code",
    SkillCategory.LANGUAGE: "language",
    SkillCategory.ALIGNMENT: "alignment",
    SkillCategory.PRODUCTION: "production",
    SkillCategory.INDIC: "indic",
}


# ================================================================
# HELPERS
# ================================================================


def sanitize_model_name(model: str) -> str:
    """Sanitize model name for use in source field.

    Converts characters like : and / to underscores.
    Examples:
        qwen3:8b      -> qwen3_8b
        Qwen/Qwen2.5  -> Qwen_Qwen2.5
    """
    return re.sub(r"[:/\\]", "_", model)


def resolve_domain_source(skill_bucket: str, model: str = "unknown") -> tuple[str, str]:
    """Resolve domain and source values from a skill_bucket ID and model name.

    Args:
        skill_bucket: Skill ID (e.g., "RSN-ARITHMETIC" or "RSN-ARITH")
        model: Model name used for generation

    Returns:
        (domain, source) tuple
    """
    canonical_id = SKILL_ALIASES.get(skill_bucket, skill_bucket)
    model_slug = sanitize_model_name(model)

    if canonical_id in SKILL_BUCKETS:
        skill = SKILL_BUCKETS[canonical_id]
        domain = SKILL_CATEGORY_TO_DOMAIN.get(skill.category, "synthetic")
        source = f"synth_{skill.category.value}_{model_slug}"
    else:
        domain = "synthetic"
        source = f"synth_unknown_{model_slug}"

    return domain, source


def format_synth_text(sample: dict) -> str:
    """Format a synth_data_bank sample into structured text.

    Follows the NCERTProcessing.py pattern of markdown-structured fields.
    Hard negatives are excluded (intentionally wrong answers).
    """
    parts = []

    skill_bucket = sample.get("skill_bucket", "")
    band = sample.get("band", "")
    if skill_bucket:
        parts.append(f"### Skill: {skill_bucket}")
    if band:
        parts.append(f"### Band: {band}")

    question = sample.get("question", "")
    if question:
        parts.append(f"\n### Question:\n{question}")

    answer = sample.get("answer", "")
    if answer:
        parts.append(f"\n### Answer:\n{answer}")

    distilled_view = sample.get("distilled_view", "")
    if distilled_view:
        parts.append(f"\n### Distilled View:\n{distilled_view}")

    think_view = sample.get("think_view")
    if think_view:
        parts.append(f"\n### Reasoning:\n{think_view}")

    error_correction = sample.get("error_correction")
    if error_correction and isinstance(error_correction, dict):
        ec_parts = []
        if error_correction.get("error_identified"):
            ec_parts.append(f"Error: {error_correction['error_identified']}")
        if error_correction.get("explanation"):
            ec_parts.append(f"Explanation: {error_correction['explanation']}")
        if error_correction.get("correct_solution"):
            ec_parts.append(f"Correction: {error_correction['correct_solution']}")
        if ec_parts:
            parts.append("\n### Error Correction:\n" + "\n".join(ec_parts))

    return "\n".join(parts)


def build_synth_metadata(sample: dict, model: str = "unknown") -> str:
    """Build metadata JSON string for a synth sample.

    Args:
        sample: Raw synth_data_bank sample dict
        model: Model name used for generation

    Returns:
        JSON string with metadata fields
    """
    metadata = {
        "skill_bucket": sample.get("skill_bucket", ""),
        "band": sample.get("band", ""),
        "stage": sample.get("stage", ""),
        "verified": sample.get("verified", False),
        "verification_score": sample.get("verification_score"),
        "source_type": "synthetic",
        "model": model,
    }

    hard_neg = sample.get("hard_negative")
    if hard_neg and isinstance(hard_neg, dict):
        metadata["has_hard_negative"] = True
        metadata["hard_negative_error_type"] = hard_neg.get("error_type", "")
    else:
        metadata["has_hard_negative"] = False

    metadata["has_error_correction"] = bool(sample.get("error_correction"))

    return json.dumps(metadata, ensure_ascii=False)


# ================================================================
# DATACLASS
# ================================================================


@dataclass
class NormalizedRecord:
    """A single record in the Glue normalized schema."""

    id: str
    hash: str
    dataset: str
    domain: str
    source: str
    text: str
    language: str
    metadata: str  # JSON string
    added: Optional[datetime]
    created: Optional[datetime]
    version: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "hash": self.hash,
            "dataset": self.dataset,
            "domain": self.domain,
            "source": self.source,
            "text": self.text,
            "language": self.language,
            "metadata": self.metadata,
            "added": self.added,
            "created": self.created,
            "version": self.version,
        }


# ================================================================
# EXPORTER CLASS
# ================================================================


class GlueExporter:
    """Export synth_data_bank to Glue normalized Parquet format."""

    def __init__(self, bank_dir: str = "./synth_data_bank", model: str = None):
        """Initialize exporter.

        Args:
            bank_dir: Path to synth_data_bank directory
            model: Model name override. If None, reads from manifest.
        """
        self.bank_dir = Path(bank_dir)
        self._model_override = model

    def load_manifest(self) -> dict:
        """Load the bank manifest."""
        manifest_path = self.bank_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest at {manifest_path}")
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)

    def get_model_name(self) -> str:
        """Get model name from override or manifest."""
        if self._model_override:
            return self._model_override
        manifest = self.load_manifest()
        return manifest.get("model", "unknown")

    def load_samples(
        self,
        skills: list[str] | None = None,
        languages: list[str] | None = None,
    ) -> list[dict]:
        """Load raw samples from bank JSONL files.

        Args:
            skills: Filter by skill IDs. None = all skills.
            languages: Filter by language codes. None = all languages.
        """
        manifest = self.load_manifest()
        all_samples = []

        for skill_id, info in manifest.get("skills", {}).items():
            if skills and skill_id not in skills:
                continue

            shard_path = self.bank_dir / info["shard_file"]
            if not shard_path.exists():
                print(f"  [WARN] Shard not found: {shard_path}")
                continue

            with open(shard_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    sample = json.loads(line)

                    if languages and sample.get("language", "en") not in languages:
                        continue

                    all_samples.append(sample)

        return all_samples

    def convert_sample(self, sample: dict) -> NormalizedRecord:
        """Convert a single synth_data_bank sample to normalized schema."""
        model = self.get_model_name()

        text = format_synth_text(sample)
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        domain, source = resolve_domain_source(sample.get("skill_bucket", ""), model)
        metadata_json = build_synth_metadata(sample, model)

        return NormalizedRecord(
            id=sample.get("id", ""),
            hash=text_hash,
            dataset=DATASET_NAME,
            domain=domain,
            source=source,
            text=text,
            language=sample.get("language", "en"),
            metadata=metadata_json,
            added=datetime.now(),
            created=None,
            version=VERSION,
        )

    def export(
        self,
        output_path: str | None = None,
        skills: list[str] | None = None,
        languages: list[str] | None = None,
        partition_by_source: bool = True,
    ) -> dict:
        """Export synth_data_bank to normalized Parquet.

        Args:
            output_path: Local directory or S3 path.
            skills: Filter by skill IDs.
            languages: Filter by language codes.
            partition_by_source: Write separate files per source partition.

        Returns:
            Export summary dict.
        """
        import pandas as pd

        output_path = output_path or "./normalized_output"

        print("\n" + "=" * 60)
        print("  GLUE NORMALIZED EXPORT")
        print("=" * 60)
        print(f"  Bank: {self.bank_dir}")
        print(f"  Output: {output_path}")
        print(f"  Model: {self.get_model_name()}")

        samples = self.load_samples(skills=skills, languages=languages)
        print(f"  Loaded: {len(samples)} samples")

        if not samples:
            print("  [WARN] No samples to export")
            return {"exported": 0}

        records = []
        for sample in samples:
            try:
                record = self.convert_sample(sample)
                records.append(record.to_dict())
            except Exception as e:
                print(f"  [ERROR] Converting {sample.get('id', '?')}: {e}")

        print(f"  Converted: {len(records)} records")

        df = pd.DataFrame(records)

        # Ensure correct column order matching Glue schema
        column_order = [
            "id",
            "hash",
            "dataset",
            "domain",
            "source",
            "text",
            "language",
            "metadata",
            "added",
            "created",
            "version",
        ]
        df = df[column_order]

        df["added"] = pd.to_datetime(df["added"])
        df["created"] = pd.to_datetime(df["created"])

        # Define a fixed Arrow schema so all partition files have
        # identical column types (prevents ArrowTypeError on read).
        import pyarrow as pa
        import pyarrow.parquet as pq

        arrow_schema = pa.schema(
            [
                ("id", pa.string()),
                ("hash", pa.string()),
                ("dataset", pa.string()),
                ("domain", pa.string()),
                ("source", pa.string()),
                ("text", pa.string()),
                ("language", pa.string()),
                ("metadata", pa.string()),
                ("added", pa.timestamp("us")),
                ("created", pa.timestamp("us")),
                ("version", pa.string()),
            ]
        )

        if partition_by_source:
            summary_by_source = {}
            for source_val, group_df in df.groupby("source"):
                source_dir = Path(output_path) / f"source={source_val}"
                source_dir.mkdir(parents=True, exist_ok=True)
                parquet_path = source_dir / "data.parquet"

                # Drop source column from data — pyarrow reconstructs
                # it from the Hive partition directory name on read.
                write_df = group_df.drop(columns=["source"])
                write_schema = pa.schema(
                    [f for f in arrow_schema if f.name != "source"]
                )
                table = pa.Table.from_pandas(write_df, schema=write_schema)
                pq.write_table(
                    table,
                    str(parquet_path),
                    compression="zstd",
                    use_dictionary=False,
                )

                summary_by_source[source_val] = len(group_df)
                print(f"    [OK] {source_val}: {len(group_df)} records")

            summary = {
                "exported": len(records),
                "by_source": summary_by_source,
                "output_path": str(output_path),
            }
        else:
            out_dir = Path(output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "synth_normalized.parquet"

            table = pa.Table.from_pandas(df, schema=arrow_schema)
            pq.write_table(
                table,
                str(out_file),
                compression="zstd",
                use_dictionary=False,
            )

            summary = {
                "exported": len(records),
                "output_file": str(out_file),
            }

        print(f"\n  [Done] Exported {len(records)} records to Glue format")
        return summary


# ================================================================
# CONVENIENCE FUNCTION
# ================================================================


def export_bank_to_glue(
    bank_dir: str = "./synth_data_bank",
    output_path: str = "./normalized_output",
    skills: list[str] | None = None,
    languages: list[str] | None = None,
    model: str | None = None,
) -> dict:
    """Convenience function to export bank to Glue normalized format.

    Args:
        bank_dir: Path to synth_data_bank directory
        output_path: Output directory for Parquet files
        skills: Filter by skill IDs
        languages: Filter by language codes
        model: Model name override (default: from manifest)

    Returns:
        Export summary dict
    """
    exporter = GlueExporter(bank_dir=bank_dir, model=model)
    return exporter.export(
        output_path=output_path,
        skills=skills,
        languages=languages,
    )


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export synth_data_bank to Glue normalized Parquet"
    )
    parser.add_argument(
        "--bank-dir", default="./synth_data_bank", help="Bank directory"
    )
    parser.add_argument(
        "--output", "-o", default="./normalized_output", help="Output path"
    )
    parser.add_argument("--skills", nargs="*", help="Filter by skill IDs")
    parser.add_argument("--languages", nargs="*", help="Filter by language codes")
    parser.add_argument(
        "--model", "-m", help="Model name override (default: from manifest)"
    )

    args = parser.parse_args()
    export_bank_to_glue(
        bank_dir=args.bank_dir,
        output_path=args.output,
        skills=args.skills,
        languages=args.languages,
        model=args.model,
    )
