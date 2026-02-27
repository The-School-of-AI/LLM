"""
Curriculum Data Validator for Coreset Building
================================================
Validates curriculum data from the curriculum team before coreset selection.
"""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


class CurriculumValidator:
    """Validates curriculum data for coreset building readiness."""

    REQUIRED_COLUMNS = [
        "uuid",
        "id",
        "text",
        "source",
        "domain",
        "hash",
        "language",
        "band_p_B0",
        "band_p_B1",
        "band_p_B2",
        "band_p_B3",
        "band_p_B4",
        "band_p_B5",
        "difficulty_score",
        "byte_length",
        "word_count",
        "token_count_estimate",
    ]

    EXPECTED_BANDS = {"B0", "B1", "B2", "B3", "B4", "B5"}

    MODALITY_COLUMNS = [
        "agentic_score",
        "cot_score",
        "reasoning_score",
        "code_score",
        "math_score",
        "table_score",
    ]

    INDIC_LANGUAGES = ["hi", "te", "ta", "kn", "ml", "bn", "gu", "mr", "pa", "or"]

    STAGE_BUDGETS = {
        "1B": 20_000_000_000,
        "3B": 40_000_000_000,
        "8B": 100_000_000_000,
        "70B": 240_000_000_000,
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}

    def validate_all(self) -> Dict:
        """Run all validation checks."""
        print("=" * 60)
        print("CURRICULUM DATA VALIDATION REPORT")
        print("=" * 60)

        self.check_schema()
        self.check_data_integrity()
        # self.check_band_distribution()
        # self.check_difficulty_scores()
        self.check_token_lengths()
        self.check_modality_scores()
        self.check_domain_language_diversity()
        self.check_stage_feasibility()

        print("\n" + "=" * 60)
        print("VALIDATION COMPLETE")
        print("=" * 60)

        return self.results

    # -------------------------------------------------------------------------
    # 1. Schema & Data Integrity
    # -------------------------------------------------------------------------

    def check_schema(self) -> Dict:
        """Check for required columns."""
        print("\n📋 SCHEMA VALIDATION")
        print("-" * 40)

        missing = [c for c in self.REQUIRED_COLUMNS if c not in self.df.columns]
        self.results["missing_columns"] = missing

        if missing:
            print(f"❌ Missing columns: {missing}")
        else:
            print(f"✓ All required columns present ({len(self.REQUIRED_COLUMNS)})")

        return {"missing_columns": missing}

    def check_data_integrity(self) -> Dict:
        """Check for nulls and duplicates."""
        print("\n🔍 DATA INTEGRITY")
        print("-" * 40)

        # Null checks
        critical_cols = ["uuid", "text", "hash", "band", "difficulty_score"]
        null_counts = {
            col: self.df[col].isnull().sum()
            for col in critical_cols
            if col in self.df.columns
        }

        for col, count in null_counts.items():
            status = "❌" if count > 0 else "✓"
            print(f"  {col}: {count:,} nulls {status}")

        # Duplicate UUIDs
        dup_uuids = (
            self.df["uuid"].duplicated().sum() if "uuid" in self.df.columns else 0
        )
        print(f"  Duplicate UUIDs: {dup_uuids:,} {'❌' if dup_uuids > 0 else '✓'}")

        # Duplicate hashes
        dup_hashes = (
            self.df["hash"].duplicated().sum() if "hash" in self.df.columns else 0
        )
        print(
            f"  Duplicate hashes: {dup_hashes:,} {'⚠️ (exact dupes)' if dup_hashes > 0 else '✓'}"
        )

        self.results["null_counts"] = null_counts
        self.results["duplicate_uuids"] = dup_uuids
        self.results["duplicate_hashes"] = dup_hashes

        return self.results

    # -------------------------------------------------------------------------
    # 2. Band Distribution
    # -------------------------------------------------------------------------

    def check_band_distribution(self, plot: bool = True) -> Dict:
        """Validate band coverage and distribution."""
        print("\n📊 BAND DISTRIBUTION")
        print("-" * 40)

        if "band" not in self.df.columns:
            print("❌ 'band' column not found")
            return {}

        band_dist = self.df["band"].value_counts().sort_index()

        for band in self.EXPECTED_BANDS:
            count = band_dist.get(band, 0)
            pct = count / len(self.df) * 100
            print(f"  {band}: {count:>10,} ({pct:5.1f}%)")

        # Missing bands
        actual = set(self.df["band"].unique())
        missing = self.EXPECTED_BANDS - actual

        if missing:
            print(f"\n⚠️ Missing bands: {missing}")
        else:
            print("\n✓ All bands present")

        print(f"  Total: {len(self.df):,}")

        self.results["band_distribution"] = band_dist.to_dict()
        self.results["missing_bands"] = list(missing)

        if plot:
            self._plot_band_distribution(band_dist)

        return self.results

    def _plot_band_distribution(self, band_dist: pd.Series):
        """Plot band distribution."""
        plt.figure(figsize=(10, 5))
        band_dist.plot(kind="bar", color="steelblue", edgecolor="black")
        plt.title("Document Distribution by Curriculum Band")
        plt.xlabel("Band")
        plt.ylabel("Document Count")
        plt.xticks(rotation=0)
        for i, v in enumerate(band_dist.values):
            plt.text(i, v + len(self.df) * 0.01, f"{v:,}", ha="center", fontsize=9)
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # 3. Difficulty Score Validation
    # -------------------------------------------------------------------------

    def check_difficulty_scores(self) -> Dict:
        """Validate difficulty scores."""
        print("\n📈 DIFFICULTY SCORES")
        print("-" * 40)

        if "difficulty_score" not in self.df.columns:
            print("❌ 'difficulty_score' column not found")
            return {}

        # Range check
        invalid = self.df[
            (self.df["difficulty_score"] < 0) | (self.df["difficulty_score"] > 1)
        ]
        print(
            f"  Invalid range (not 0-1): {len(invalid):,} {'❌' if len(invalid) > 0 else '✓'}"
        )

        # Mean by band
        print("\n  Mean difficulty by band:")
        band_means = self.df.groupby("band")["difficulty_score"].mean().sort_index()
        for band, mean in band_means.items():
            print(f"    {band}: {mean:.3f}")

        # Monotonicity check
        sorted_means = band_means.sort_index()
        is_monotonic = all(
            sorted_means.iloc[i] <= sorted_means.iloc[i + 1]
            for i in range(len(sorted_means) - 1)
        )
        print(f"\n  Monotonic (B0 < B1 < ... < B5): {'✓' if is_monotonic else '❌'}")

        self.results["invalid_difficulty_count"] = len(invalid)
        self.results["difficulty_by_band"] = band_means.to_dict()
        self.results["difficulty_is_monotonic"] = is_monotonic

        return self.results

    # -------------------------------------------------------------------------
    # 4. Token/Length Validation
    # -------------------------------------------------------------------------

    def check_token_lengths(self) -> Dict:
        """Validate token counts for coreset budgeting."""
        print("\n📏 TOKEN LENGTH VALIDATION")
        print("-" * 40)

        if "token_count_estimate" not in self.df.columns:
            print("❌ 'token_count_estimate' column not found")
            return {}

        # Stats by band
        stats = self.df.groupby("band")["token_count_estimate"].agg(
            ["sum", "mean", "min", "max"]
        )
        print("\n  Token stats by band:")
        print(stats.to_string())

        # Anomalies
        too_short = self.df[self.df["token_count_estimate"] < 10]
        too_long = self.df[self.df["token_count_estimate"] > 100000]

        print(f"\n  Too short (<10 tokens): {len(too_short):,}")
        print(f"  Too long (>100k tokens): {len(too_long):,}")

        # Total tokens
        total = self.df["token_count_estimate"].sum()
        print(f"\n  Total tokens available: {total:,}")

        self.results["token_stats_by_band"] = stats.to_dict()
        self.results["too_short_count"] = len(too_short)
        self.results["too_long_count"] = len(too_long)
        self.results["total_tokens"] = total

        return self.results

    # -------------------------------------------------------------------------
    # 5. Modality Score Validation
    # -------------------------------------------------------------------------

    def check_modality_scores(self) -> Dict:
        """Validate modality scores for protected slice identification."""
        print("\n🎯 MODALITY SCORES (Protected Slices)")
        print("-" * 40)

        modality_stats = {}

        for col in self.MODALITY_COLUMNS:
            if col in self.df.columns:
                non_zero = (self.df[col] > 0).sum()
                high_score = (self.df[col] >= 5).sum()
                pct = non_zero / len(self.df) * 100
                print(
                    f"  {col}: {non_zero:,} non-zero ({pct:.1f}%), {high_score:,} high (≥5)"
                )
                modality_stats[col] = {"non_zero": non_zero, "high_score": high_score}

        # Protected content thresholds based on pipeline.yaml
        protected = {}
        if "code_score" in self.df.columns:
            protected["high_code"] = len(self.df[self.df["code_score"] >= 10])
        if "agentic_score" in self.df.columns:
            protected["high_agentic"] = len(self.df[self.df["agentic_score"] >= 5])
        if "reasoning_score" in self.df.columns:
            protected["high_reasoning"] = len(self.df[self.df["reasoning_score"] >= 6])
        if "math_score" in self.df.columns:
            protected["high_math"] = len(self.df[self.df["math_score"] >= 8])

        print("\n  Protected content counts:")
        for k, v in protected.items():
            print(f"    {k}: {v:,}")

        self.results["modality_stats"] = modality_stats
        self.results["protected_counts"] = protected

        return self.results

    # -------------------------------------------------------------------------
    # 6. Domain/Language Diversity
    # -------------------------------------------------------------------------

    def check_domain_language_diversity(self) -> Dict:
        """Check domain and language diversity."""
        print("\n🌐 DOMAIN & LANGUAGE DIVERSITY")
        print("-" * 40)

        # Domain
        if "domain" in self.df.columns:
            domain_counts = self.df["domain"].value_counts()
            print("\n  Domain distribution:")
            for domain, count in domain_counts.head(10).items():
                pct = count / len(self.df) * 100
                print(f"    {domain}: {count:,} ({pct:.1f}%)")
            if len(domain_counts) > 10:
                print(f"    ... and {len(domain_counts) - 10} more domains")
            self.results["domain_distribution"] = domain_counts.to_dict()

        # Language
        if "language" in self.df.columns:
            lang_counts = self.df["language"].value_counts()
            print("\n  Language distribution:")
            for lang, count in lang_counts.head(10).items():
                pct = count / len(self.df) * 100
                print(f"    {lang}: {count:,} ({pct:.1f}%)")

            # Indic check
            indic_mask = self.df["language"].isin(self.INDIC_LANGUAGES)
            indic_count = indic_mask.sum()
            indic_pct = indic_count / len(self.df) * 100
            print(
                f"\n  Indic languages (protected): {indic_count:,} ({indic_pct:.1f}%)"
            )

            self.results["language_distribution"] = lang_counts.to_dict()
            self.results["indic_count"] = indic_count

        return self.results

    # -------------------------------------------------------------------------
    # 7. Stage Budget Feasibility
    # -------------------------------------------------------------------------

    def check_stage_feasibility(self) -> Dict:
        """Check if data can support training stages."""
        print("\n💰 STAGE BUDGET FEASIBILITY")
        print("-" * 40)

        if "token_count_estimate" not in self.df.columns:
            print("❌ Cannot check - 'token_count_estimate' missing")
            return {}

        available = self.df["token_count_estimate"].sum()
        print(f"  Available tokens: {available:,}")
        print()

        feasibility = {}
        for stage, budget in self.STAGE_BUDGETS.items():
            pct = (available / budget) * 100
            status = "✓" if pct >= 100 else "⚠️"
            print(f"  {stage}: covers {pct:.2f}% of target ({budget:,}) {status}")
            feasibility[stage] = {"budget": budget, "coverage_pct": pct}

        self.results["stage_feasibility"] = feasibility

        return self.results

    # -------------------------------------------------------------------------
    # Spot Check
    # -------------------------------------------------------------------------

    def spot_check(self, n: int = 3, band: Optional[str] = None):
        """Sample random documents for manual inspection."""
        print("\n🔬 SPOT CHECK - Sample Documents")
        print("-" * 40)

        subset = self.df[self.df["band"] == band] if band else self.df
        sample = subset.sample(min(n, len(subset)))

        for _, row in sample.iterrows():
            print(f"\n{'='*60}")
            print(f"Band: {row['band']} | Difficulty: {row['difficulty_score']:.3f}")
            print(
                f"Domain: {row.get('domain', 'N/A')} | Language: {row.get('language', 'N/A')}"
            )
            print(f"Tokens: {row.get('token_count_estimate', 'N/A'):,}")

            scores = []
            for col in self.MODALITY_COLUMNS:
                if col in row and row[col] > 0:
                    scores.append(f"{col.replace('_score', '')}={row[col]}")
            if scores:
                print(f"Modalities: {', '.join(scores)}")

            text = row.get("text", "")
            print(f"\nText Preview:\n{text[:400]}...")


def validate_curriculum_data(df: pd.DataFrame, plot: bool = True) -> Dict:
    """Convenience function to run all validations."""
    validator = CurriculumValidator(df)
    return validator.validate_all()
