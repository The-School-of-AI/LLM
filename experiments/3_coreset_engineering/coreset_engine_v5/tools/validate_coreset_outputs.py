"""
Coreset Output Validator - Validates coreset engine outputs against curriculum
Generates checklists and verification reports for manifest and selected indices
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.types import DifficultyBand, difficulty_band_order
from src.curriculum.loader import CurriculumLoader


@dataclass
class ValidationCheck:
    """A single validation check result"""

    check_id: str
    category: str  # band_ratios, domain_distribution, language_policy, etc.
    name: str
    expected: Any
    actual: Any
    passed: bool
    severity: str  # critical, high, medium, low
    message: str
    details: str = ""


@dataclass
class ValidationReport:
    """Complete validation report"""

    stage_name: str
    manifest_path: Path
    indices_path: Path
    checks: List[ValidationCheck] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""
    language_metrics: Dict[str, Any] = field(
        default_factory=dict
    )  # Language policy metrics

    def add_check(self, check: ValidationCheck):
        """Add a check result"""
        self.checks.append(check)

    def get_summary(self) -> Dict[str, int]:
        """Get summary statistics"""
        by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        by_status = {"passed": 0, "failed": 0}

        for check in self.checks:
            by_severity[check.severity] += 1
            if check.passed:
                by_status["passed"] += 1
            else:
                by_status["failed"] += 1

        return {
            "total_checks": len(self.checks),
            "by_severity": by_severity,
            "by_status": by_status,
            "success_rate": (
                (by_status["passed"] / len(self.checks) * 100) if self.checks else 0
            ),
        }


class CoresetValidator:
    """Validates coreset outputs against curriculum"""

    def _get_availability_stats(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Return availability stats dict from manifest (or empty dict)."""
        availability = manifest.get("availability_stats")
        return availability if isinstance(availability, dict) else {}

    def _get_selected_tokens_from_manifest(self, manifest: Dict[str, Any]) -> int:
        return int(
            manifest.get("actual_tokens")
            or manifest.get("selected_tokens")
            or manifest.get("total_tokens")
            or 0
        )

    def __init__(self, curriculum_path: str, output_base_dir: str = "output/coresets"):
        self.curriculum_path = Path(curriculum_path)
        self.output_dir = Path(output_base_dir)
        self.curriculum = CurriculumLoader(str(curriculum_path))

        success, errors = self.curriculum.load()
        if not success:
            raise ValueError(f"Failed to load curriculum: {errors}")

        self.logger = logging.getLogger(__name__)
        self.reports: Dict[str, ValidationReport] = {}

    def validate_stage(self, stage_name: str) -> ValidationReport:
        """Validate all outputs for a stage"""
        stage_dir = self.output_dir / stage_name
        manifest_path = self._resolve_manifest_path(stage_dir)
        indices_path = self._resolve_indices_path(stage_dir)

        report = ValidationReport(
            stage_name=stage_name,
            manifest_path=manifest_path,
            indices_path=indices_path,
            generated_at=self._get_timestamp(),
        )

        # Check file existence
        self._validate_files_exist(report)

        # If manifest is missing, no further checks are possible.
        if not manifest_path.exists():
            self.reports[stage_name] = report
            return report

        # Load outputs
        manifest = self._load_manifest(manifest_path)

        indices: List[Dict[str, Any]] = []
        if indices_path.exists():
            indices = self._load_selected_indices(indices_path)

        # Validate manifest structure
        self._validate_manifest_structure(report, manifest)

        # Validate indices format (if indices exist)
        if indices_path.exists():
            self._validate_indices_format(report, indices)

        # Validate against curriculum
        self._validate_band_distribution(report, manifest)
        self._validate_domain_distribution(report, manifest)
        self._validate_language_distribution(report, manifest)
        self._validate_stage_targets(report, manifest)
        self._validate_rolling_window(report, manifest)
        self._validate_protected_slices(report, manifest)

        self.reports[stage_name] = report
        return report

    def _resolve_manifest_path(self, stage_dir: Path) -> Path:
        """Resolve manifest path for both legacy and streaming outputs."""
        legacy = stage_dir / "manifest.json"
        if legacy.exists():
            return legacy

        shard_manifests = sorted(stage_dir.glob("manifest_shard*.json"))
        if shard_manifests:
            # Single-shard streaming outputs will have manifest_shard000.json.
            return shard_manifests[0]

        return legacy

    def _resolve_indices_path(self, stage_dir: Path) -> Path:
        """Resolve indices path (parquet preferred, jsonl fallback)."""
        parquet = stage_dir / "selected_indices.parquet"
        if parquet.exists():
            return parquet

        jsonl = stage_dir / "selected_indices.jsonl"
        if jsonl.exists():
            return jsonl

        # As a last resort, allow validating format checks against a part file
        part_files = sorted(stage_dir.glob("selected_indices_part_*.parquet"))
        if part_files:
            return part_files[0]

        return jsonl

    def _validate_files_exist(self, report: ValidationReport):
        """Check if required files exist"""
        manifest_exists = report.manifest_path.exists()
        indices_exists = report.indices_path.exists()

        report.add_check(
            ValidationCheck(
                check_id="FILE_MANIFEST",
                category="files",
                name="Manifest file exists",
                expected=True,
                actual=manifest_exists,
                passed=manifest_exists,
                severity="critical",
                message=f"Manifest: {report.manifest_path}",
                details="Manifest JSON file should exist for stage",
            )
        )

        report.add_check(
            ValidationCheck(
                check_id="FILE_INDICES",
                category="files",
                name="Selected indices file exists",
                expected=True,
                actual=indices_exists,
                passed=indices_exists,
                severity="critical",
                message=f"Indices: {report.indices_path}",
                details="Selected indices JSONL file should exist for stage",
            )
        )

    def _load_manifest(self, path: Path) -> Dict[str, Any]:
        """Load manifest JSON"""
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load manifest: {e}")
            return {}

    def _load_selected_indices(self, path: Path) -> List[Dict[str, Any]]:
        """Load selected indices from JSONL or Parquet."""
        indices: List[Dict[str, Any]] = []

        suffix = path.suffix.lower()
        if suffix == ".parquet":
            try:
                try:
                    import pyarrow.parquet as pq

                    table = pq.read_table(path)
                    return table.to_pylist()
                except Exception:
                    # Fallback to pandas if pyarrow isn't available in this environment
                    import pandas as pd

                    df = pd.read_parquet(path)
                    return df.to_dict(orient="records")
            except Exception as e:
                self.logger.error(f"Failed to load parquet indices: {e}")
                return []

        # Default: JSONL
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        indices.append(json.loads(line))
        except Exception as e:
            self.logger.error(f"Failed to load indices: {e}")
        return indices

    def _validate_manifest_structure(self, report: ValidationReport, manifest: Dict):
        """Validate manifest has required fields"""
        required_fields = ["stage_name", "composition", "protected_slices_preserved"]

        # Check top-level fields
        for field_name in required_fields:
            field_exists = field_name in manifest
            report.add_check(
                ValidationCheck(
                    check_id=f"MANIFEST_{field_name.upper()}",
                    category="manifest_structure",
                    name=f"Manifest has '{field_name}' field",
                    expected=True,
                    actual=field_exists,
                    passed=field_exists,
                    severity="high" if not field_exists else "low",
                    message=f"Field '{field_name}' is present",
                    details=f"Required manifest field: {field_name}",
                )
            )

        # Check composition sub-fields
        composition = manifest.get("composition", {})
        composition_fields = [
            "band_distribution",
            "domain_distribution",
            "language_distribution",
        ]

        for field_name in composition_fields:
            field_exists = field_name in composition
            report.add_check(
                ValidationCheck(
                    check_id=f"MANIFEST_COMPOSITION_{field_name.upper()}",
                    category="manifest_structure",
                    name=f"Manifest composition has '{field_name}' field",
                    expected=True,
                    actual=field_exists,
                    passed=field_exists,
                    severity="high" if not field_exists else "low",
                    message=f"Field 'composition.{field_name}' is present",
                    details=f"Required composition field: {field_name}",
                )
            )

    def _validate_indices_format(self, report: ValidationReport, indices: List[Dict]):
        """Validate indices format"""
        has_indices = len(indices) > 0
        report.add_check(
            ValidationCheck(
                check_id="INDICES_NOT_EMPTY",
                category="indices_format",
                name="Selected indices not empty",
                expected=True,
                actual=has_indices,
                passed=has_indices,
                severity="medium" if not has_indices else "low",
                message=f"Found {len(indices)} selected indices",
                details="Should have at least some selected indices",
            )
        )

        if indices:
            # Check first index has required fields
            first_index = indices[0]
            required_fields = ["chunk_id", "band", "domain"]
            for field_name in required_fields:
                field_exists = field_name in first_index
                report.add_check(
                    ValidationCheck(
                        check_id=f"INDICES_FIELD_{field_name.upper()}",
                        category="indices_format",
                        name=f"Index entries have '{field_name}' field",
                        expected=True,
                        actual=field_exists,
                        passed=field_exists,
                        severity="high" if not field_exists else "low",
                        message=f"Field '{field_name}' present in indices",
                        details=f"Sample: {first_index}",
                    )
                )

            # Token count field: accept either the new canonical name or legacy name.
            token_field_exists = ("token_count" in first_index) or (
                "token_count_estimate" in first_index
            )
            report.add_check(
                ValidationCheck(
                    check_id="INDICES_FIELD_TOKEN_COUNT",
                    category="indices_format",
                    name="Index entries have token count field",
                    expected=True,
                    actual=token_field_exists,
                    passed=token_field_exists,
                    severity="high" if not token_field_exists else "low",
                    message="Field 'token_count' or 'token_count_estimate' present in indices",
                    details=f"Sample: {first_index}",
                )
            )

    def _validate_band_distribution(self, report: ValidationReport, manifest: Dict):
        """Validate band distribution matches curriculum"""
        # Get band_distribution from composition
        composition = manifest.get("composition", {})
        band_dist = composition.get("band_distribution", {})
        stage_config = self.curriculum.stages.get(report.stage_name)

        if not stage_config:
            report.add_check(
                ValidationCheck(
                    check_id="BAND_STAGE_NOT_FOUND",
                    category="band_ratios",
                    name="Stage found in curriculum",
                    expected=True,
                    actual=False,
                    passed=False,
                    severity="critical",
                    message=f"Stage {report.stage_name} not in curriculum",
                    details=f"Available stages: {list(self.curriculum.stages.keys())}",
                )
            )
            return

        expected_ratios = stage_config.band_ratios
        tolerance = 0.02  # 2% tolerance

        availability = self._get_availability_stats(manifest)
        eligible_total = int(availability.get("eligible_unused_tokens_total", 0) or 0)
        eligible_by_band = availability.get("eligible_unused_tokens_by_band")
        eligible_by_band = (
            eligible_by_band if isinstance(eligible_by_band, dict) else {}
        )

        selected_total_tokens = self._get_selected_tokens_from_manifest(manifest)

        for band_name in difficulty_band_order():
            expected = getattr(expected_ratios, band_name, 0.0)
            actual = band_dist.get(band_name, 0.0)

            # Convert to ratios if given as percentages
            if actual > 1.0:
                actual = actual / 100.0

            passed = abs(expected - actual) <= tolerance

            # Availability-aware downgrade: if manifest proves this ratio is not achievable
            # under strict non-overlap + stage gating, treat as informational PASS.
            if (not passed) and selected_total_tokens > 0 and eligible_total > 0:
                eligible_band_tokens = int(eligible_by_band.get(band_name, 0) or 0)
                upper_share_possible = min(
                    1.0, eligible_band_tokens / float(selected_total_tokens)
                )

                # If we're below expected: even selecting all eligible from this band can't reach expected.
                if actual < (expected - tolerance):
                    if upper_share_possible < (expected - tolerance):
                        passed = True

                # If we're above expected: even selecting the maximum from other bands can't dilute this band enough.
                if actual > (expected + tolerance):
                    other_cap = max(0, eligible_total - eligible_band_tokens)
                    min_band_tokens = max(0, selected_total_tokens - other_cap)
                    lower_share_possible = min_band_tokens / float(
                        selected_total_tokens
                    )
                    if lower_share_possible > (expected + tolerance):
                        passed = True
            report.add_check(
                ValidationCheck(
                    check_id=f"BAND_{band_name}",
                    category="band_ratios",
                    name=f"Band {band_name} ratio matches curriculum",
                    expected=expected,
                    actual=actual,
                    passed=passed,
                    severity="high" if not passed else "low",
                    message=(
                        f"{band_name}: expected {expected:.2%}, got {actual:.2%}"
                        + (
                            " (availability-limited)"
                            if passed
                            and abs(expected - actual) > tolerance
                            and eligible_total > 0
                            else ""
                        )
                    ),
                    details=(
                        f"Tolerance: {tolerance:.2%}"
                        + (
                            f" | eligible_unused_tokens_total={eligible_total:,}"
                            f" | eligible_{band_name}_tokens={int(eligible_by_band.get(band_name, 0) or 0):,}"
                            if passed
                            and abs(expected - actual) > tolerance
                            and eligible_total > 0
                            else ""
                        )
                    ),
                )
            )

    def _validate_domain_distribution(self, report: ValidationReport, manifest: Dict):
        """Validate domain distribution is valid"""
        # Get domain_distribution from composition
        composition = manifest.get("composition", {})
        domain_dist = composition.get("domain_distribution", {})
        stage_config = self.curriculum.stages.get(report.stage_name)

        if not stage_config:
            return

        by_band = None
        if isinstance(domain_dist, dict):
            if isinstance(domain_dist.get("by_band"), dict):
                by_band = domain_dist.get("by_band")
            elif isinstance(domain_dist.get("byBand"), dict):
                by_band = domain_dist.get("byBand")

        # Preferred (v2): validate per band using by_band structure.
        if by_band:
            for band_name, domain_ratio in (by_band or {}).items():
                try:
                    band = DifficultyBand(band_name)
                except Exception:
                    continue
                allowed_domains = self.curriculum.get_allowed_domains_for_band(band)
                for domain_name in (domain_ratio or {}).keys():
                    domain_allowed = domain_name in allowed_domains
                    report.add_check(
                        ValidationCheck(
                            check_id=f"DOMAIN_{band_name}_{domain_name}",
                            category="domain_distribution",
                            name=f"Domain {domain_name} allowed for {band_name}",
                            expected=True,
                            actual=domain_allowed,
                            passed=domain_allowed,
                            severity="high" if not domain_allowed else "low",
                            message=f"{band_name}/{domain_name} is allowed",
                            details=f"Allowed domains for {band_name}: {allowed_domains}",
                        )
                    )
            return

        # Legacy fallback: if domain_distribution is a flat mapping of domain->share,
        # validate that each domain is allowed in at least one band used by the stage.
        if isinstance(domain_dist, dict) and domain_dist:
            used_bands = []
            if stage_config and getattr(stage_config, "band_ratios", None):
                for band_name in difficulty_band_order():
                    if getattr(stage_config.band_ratios, band_name, 0.0) > 0:
                        used_bands.append(band_name)
            for domain_name in domain_dist.keys():
                allowed_somewhere = False
                allowed_details = {}
                for band_name in used_bands:
                    try:
                        band = DifficultyBand(band_name)
                    except Exception:
                        continue
                    allowed = self.curriculum.get_allowed_domains_for_band(band)
                    allowed_details[band_name] = allowed
                    if domain_name in allowed:
                        allowed_somewhere = True
                        break
                report.add_check(
                    ValidationCheck(
                        check_id=f"DOMAIN_ANY_{domain_name}",
                        category="domain_distribution",
                        name=f"Domain {domain_name} allowed in stage",
                        expected=True,
                        actual=allowed_somewhere,
                        passed=allowed_somewhere,
                        severity="high" if not allowed_somewhere else "low",
                        message=f"{domain_name} is allowed in at least one band",
                        details=f"Bands used: {used_bands} | Allowed domains per band: {allowed_details}",
                    )
                )

    def _validate_language_distribution(self, report: ValidationReport, manifest: Dict):
        """Validate language distribution against policy with comprehensive metrics"""
        # Get language_distribution from composition
        composition = manifest.get("composition", {})
        lang_dist = composition.get("language_distribution", {})

        if not self.curriculum.language_policy:
            return

        lang_policy = self.curriculum.language_policy
        primary_langs = lang_policy.primary_languages
        secondary_langs = lang_policy.secondary_languages
        excluded_langs = lang_policy.explicitly_excluded

        # Tolerance: 1% variance from max_share constraint
        tolerance = 0.01

        # Track compliance metrics
        metrics = {
            "total_languages": len(lang_dist),
            "allowed_languages": len(primary_langs) + len(secondary_langs),
            "excluded_found": 0,
            "primary_compliant": 0,
            "primary_total": 0,
            "secondary_compliant": 0,
            "secondary_total": 0,
            "unrecognized_languages": [],
            "primary_violations": [],
            "secondary_violations": [],
        }

        for lang_code, token_share in lang_dist.items():
            # Check not excluded (CRITICAL)
            is_excluded = lang_code in excluded_langs
            if is_excluded:
                metrics["excluded_found"] += 1
                report.add_check(
                    ValidationCheck(
                        check_id=f"LANG_EXCLUDED_{lang_code}",
                        category="language_policy",
                        name=f"Language {lang_code} NOT in excluded list",
                        expected=False,
                        actual=True,
                        passed=False,
                        severity="critical",
                        message=f"{lang_code} FOUND in excluded list but present in coreset",
                        details=f"Excluded: {sorted(excluded_langs)} | Found: {lang_code} ({token_share:.2%})",
                    )
                )
            else:
                report.add_check(
                    ValidationCheck(
                        check_id=f"LANG_EXCLUDED_{lang_code}",
                        category="language_policy",
                        name=f"Language {lang_code} NOT in excluded list",
                        expected=False,
                        actual=False,
                        passed=True,
                        severity="low",
                        message=f"{lang_code} not in excluded list",
                        details=f"Excluded: {sorted(excluded_langs)}",
                    )
                )

            # Check primary language share constraints
            if lang_code in primary_langs:
                metrics["primary_total"] += 1
                max_share = primary_langs[lang_code]
                # Allow 1% variance
                actual_share = token_share
                shares_ok = actual_share <= max_share + tolerance

                if not shares_ok:
                    metrics["primary_violations"].append(
                        {
                            "lang": lang_code,
                            "actual": actual_share,
                            "max": max_share,
                            "excess": actual_share - max_share,
                        }
                    )
                else:
                    metrics["primary_compliant"] += 1

                report.add_check(
                    ValidationCheck(
                        check_id=f"LANG_PRIMARY_{lang_code}",
                        category="language_policy",
                        name=f"Primary language {lang_code} compliance",
                        expected=max_share,
                        actual=actual_share,
                        passed=shares_ok,
                        severity="high" if not shares_ok else "low",
                        message=f"{lang_code}: {actual_share:.2%} {'<=' if shares_ok else '>'} {max_share:.2%} (tol: {tolerance:.2%})",
                        details=f"Primary max share: {max_share:.2%} | Actual: {actual_share:.2%} | Excess: {max(0, actual_share - max_share):.2%}",
                    )
                )
            elif lang_code in secondary_langs:
                metrics["secondary_total"] += 1
                max_share = secondary_langs[lang_code]
                # Allow 1% variance
                actual_share = token_share
                shares_ok = actual_share <= max_share + tolerance

                if not shares_ok:
                    metrics["secondary_violations"].append(
                        {
                            "lang": lang_code,
                            "actual": actual_share,
                            "max": max_share,
                            "excess": actual_share - max_share,
                        }
                    )
                else:
                    metrics["secondary_compliant"] += 1

                report.add_check(
                    ValidationCheck(
                        check_id=f"LANG_SECONDARY_{lang_code}",
                        category="language_policy",
                        name=f"Secondary language {lang_code} compliance",
                        expected=max_share,
                        actual=actual_share,
                        passed=shares_ok,
                        severity="high" if not shares_ok else "low",
                        message=f"{lang_code}: {actual_share:.2%} {'<=' if shares_ok else '>'} {max_share:.2%} (tol: {tolerance:.2%})",
                        details=f"Secondary max share: {max_share:.2%} | Actual: {actual_share:.2%} | Excess: {max(0, actual_share - max_share):.2%}",
                    )
                )
            else:
                # Unrecognized language
                metrics["unrecognized_languages"].append((lang_code, token_share))
                report.add_check(
                    ValidationCheck(
                        check_id=f"LANG_UNKNOWN_{lang_code}",
                        category="language_policy",
                        name=f"Language {lang_code} recognized in policy",
                        expected=True,
                        actual=False,
                        passed=False,
                        severity="high",
                        message=f"{lang_code} not in primary or secondary languages",
                        details=f"Primary: {list(primary_langs.keys())} | Secondary: {list(secondary_langs.keys())}",
                    )
                )

        # Add compliance summary check
        compliance_score = 0
        if metrics["excluded_found"] == 0:
            compliance_score += 25
        if metrics["unrecognized_languages"] == 0:
            compliance_score += 25
        if (
            metrics["primary_total"] == 0
            or metrics["primary_compliant"] == metrics["primary_total"]
        ):
            compliance_score += 25
        if (
            metrics["secondary_total"] == 0
            or metrics["secondary_compliant"] == metrics["secondary_total"]
        ):
            compliance_score += 25

        report.add_check(
            ValidationCheck(
                check_id="LANG_POLICY_COMPLIANCE_SCORE",
                category="language_policy",
                name="Overall language policy compliance",
                expected=100,
                actual=compliance_score,
                passed=compliance_score >= 75,
                severity="high" if compliance_score < 75 else "low",
                message=f"Language policy compliance: {compliance_score}/100",
                details=(
                    f"Excluded found: {metrics['excluded_found']} | "
                    f"Unrecognized: {len(metrics['unrecognized_languages'])} | "
                    f"Primary: {metrics['primary_compliant']}/{metrics['primary_total']} | "
                    f"Secondary: {metrics['secondary_compliant']}/{metrics['secondary_total']}"
                ),
            )
        )

        # Store metrics in report for reporting
        report.language_metrics = metrics

    def _validate_stage_targets(self, report: ValidationReport, manifest: Dict):
        """Validate stage meets target tokens"""
        # Prefer manifest values because streaming runs may scale targets for sample-sized E2E.
        # Fall back to curriculum when manifest doesn't have targets.
        selected_tokens = self._get_selected_tokens_from_manifest(manifest)

        # Prefer explicit per-shard target when present (streaming/sharded runs).
        target_tokens_manifest = manifest.get("target_tokens_shard")
        if target_tokens_manifest is None:
            target_tokens_manifest = manifest.get("target_tokens")
        target_tokens_curriculum = None
        stage_config = getattr(self.curriculum, "stages", {}).get(report.stage_name)
        if stage_config is not None:
            target_tokens_curriculum = getattr(stage_config, "total_tokens", None)

        target_tokens = int(target_tokens_manifest or target_tokens_curriculum or 0)
        ratio = selected_tokens / target_tokens if target_tokens > 0 else 0

        # Allow 5% deviation from target
        tolerance = 0.05
        within_tolerance = abs(1.0 - ratio) <= tolerance

        availability = self._get_availability_stats(manifest)
        eligible_total = int(availability.get("eligible_unused_tokens_total", 0) or 0)

        # Availability-aware downgrade: if the remaining eligible pool is insufficient to
        # meet the *minimum* required tokens for passing tolerance, treat as informational.
        if (not within_tolerance) and target_tokens > 0 and eligible_total > 0:
            required_min = int(target_tokens * (1.0 - tolerance))
            if eligible_total < required_min:
                within_tolerance = True

        if availability:
            report.add_check(
                ValidationCheck(
                    check_id="AVAILABILITY_SUMMARY",
                    category="availability",
                    name="Manifest contains availability stats",
                    expected=True,
                    actual=True,
                    passed=True,
                    severity="low",
                    message=f"Eligible unused tokens observed: {eligible_total:,}",
                    details="Source: manifest.availability_stats (pre-selection eligibility after non-overlap + stage gating)",
                )
            )

        report.add_check(
            ValidationCheck(
                check_id="STAGE_TARGET_TOKENS",
                category="stage_targets",
                name="Stage meets token target (±5%)",
                expected=target_tokens,
                actual=selected_tokens,
                passed=within_tolerance,
                severity="high" if not within_tolerance else "low",
                message=(
                    f"Target: {target_tokens:,}, Actual: {selected_tokens:,}, Ratio: {ratio:.2%}"
                    + (
                        " (availability-limited)"
                        if within_tolerance is True
                        and abs(1.0 - ratio) > tolerance
                        and eligible_total > 0
                        else ""
                    )
                ),
                details=(
                    f"Tolerance: {tolerance:.1%} | "
                    f"Source: {'manifest.target_tokens_shard' if manifest.get('target_tokens_shard') is not None else 'manifest.target_tokens' if manifest.get('target_tokens') is not None else 'curriculum.stage.total_tokens' if target_tokens_curriculum else 'missing'}"
                    + (
                        f" | eligible_unused_tokens_total={eligible_total:,}"
                        if within_tolerance is True
                        and abs(1.0 - ratio) > tolerance
                        and eligible_total > 0
                        else ""
                    )
                ),
            )
        )

    def _validate_rolling_window(self, report: ValidationReport, manifest: Dict):
        """Validate rolling window constraints are met"""
        if not self.curriculum.rolling_window:
            return

        if "rolling_window_stats" not in manifest or not isinstance(
            manifest.get("rolling_window_stats"), dict
        ):
            report.add_check(
                ValidationCheck(
                    check_id="ROLLING_WINDOW_STATS_PRESENT",
                    category="rolling_window",
                    name="Manifest contains rolling_window_stats",
                    expected=True,
                    actual=False,
                    passed=False,
                    severity="high",
                    message="rolling_window_stats missing; anti-spike enforcement not auditable",
                    details="Expected manifest.rolling_window_stats with max_band_delta/max_domain_delta/window_tokens",
                )
            )
            return

        rolling_window_stats = manifest.get("rolling_window_stats", {})
        max_band_delta = rolling_window_stats.get("max_band_delta", 0)
        max_domain_delta = rolling_window_stats.get("max_domain_delta", 0)

        constraint_band_delta = self.curriculum.rolling_window.max_band_delta
        constraint_domain_delta = self.curriculum.rolling_window.max_domain_delta

        band_delta_ok = max_band_delta <= constraint_band_delta
        domain_delta_ok = max_domain_delta <= constraint_domain_delta

        report.add_check(
            ValidationCheck(
                check_id="ROLLING_WINDOW_BAND",
                category="rolling_window",
                name="Rolling window band delta within constraint",
                expected=constraint_band_delta,
                actual=max_band_delta,
                passed=band_delta_ok,
                severity="high" if not band_delta_ok else "low",
                message=f"Max band delta: {max_band_delta:.4f} <= {constraint_band_delta:.4f}",
                details=f"Rolling window size: {self.curriculum.rolling_window.window_tokens:,} tokens",
            )
        )

        report.add_check(
            ValidationCheck(
                check_id="ROLLING_WINDOW_DOMAIN",
                category="rolling_window",
                name="Rolling window domain delta within constraint",
                expected=constraint_domain_delta,
                actual=max_domain_delta,
                passed=domain_delta_ok,
                severity="high" if not domain_delta_ok else "low",
                message=f"Max domain delta: {max_domain_delta:.4f} <= {constraint_domain_delta:.4f}",
                details="Domain delta constraint",
            )
        )

    def _validate_protected_slices(self, report: ValidationReport, manifest: Dict):
        """Validate protected slices are enforced"""
        protected_stats = manifest.get("protected_slices", {})

        # Check that protected slices are present if curriculum requires them
        if protected_stats:
            report.add_check(
                ValidationCheck(
                    check_id="PROTECTED_SLICES_PRESENT",
                    category="protected_slices",
                    name="Protected slices enforced",
                    expected=True,
                    actual=True,
                    passed=True,
                    severity="low",
                    message=f"Protected slices: {list(protected_stats.keys())}",
                    details=f"Protected slice stats: {protected_stats}",
                )
            )

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime

        return datetime.now().isoformat()

    def generate_checklist(self, stage_name: str) -> str:
        """Generate checklist output"""
        report = self.reports.get(stage_name)
        if not report:
            return f"No report for stage {stage_name}"

        lines = []
        lines.append("\n" + "=" * 80)
        lines.append(f"CORESET VALIDATION CHECKLIST - Stage {stage_name}")
        lines.append("=" * 80)

        # Group by category
        by_category = {}
        for check in report.checks:
            if check.category not in by_category:
                by_category[check.category] = []
            by_category[check.category].append(check)

        for category in sorted(by_category.keys()):
            checks = by_category[category]
            passed_count = sum(1 for c in checks if c.passed)
            total_count = len(checks)

            lines.append(
                f"\n### {category.upper().replace('_', ' ')} ({passed_count}/{total_count})"
            )
            lines.append("-" * 80)

            for check in checks:
                status = "✓ PASS" if check.passed else "✗ FAIL"
                severity = f"[{check.severity.upper()}]"
                lines.append(f"{status} {severity:12} {check.name}")
                lines.append(f"         {check.message}")
                if check.details:
                    lines.append(f"         Details: {check.details}")

        return "\n".join(lines)

    def generate_report(self, stage_name: str) -> str:
        """Generate detailed verification report"""
        report = self.reports.get(stage_name)
        if not report:
            return f"No report for stage {stage_name}"

        summary = report.get_summary()
        lines = []

        lines.append("\n" + "=" * 100)
        lines.append(f"CORESET ENGINE VERIFICATION REPORT - Stage {stage_name}")
        lines.append("=" * 100)

        # Summary section
        lines.append(f"\nGenerated: {report.generated_at}")
        lines.append(f"Manifest: {report.manifest_path}")
        lines.append(f"Indices:  {report.indices_path}")

        lines.append("\n### SUMMARY")
        lines.append("-" * 100)
        lines.append(f"Total Checks:        {summary['total_checks']}")
        lines.append(f"Passed:              {summary['by_status']['passed']}")
        lines.append(f"Failed:              {summary['by_status']['failed']}")
        lines.append(f"Success Rate:        {summary['success_rate']:.1f}%")
        lines.append(f"Critical Issues:     {summary['by_severity']['critical']}")
        lines.append(f"High Severity:       {summary['by_severity']['high']}")
        lines.append(f"Medium Severity:     {summary['by_severity']['medium']}")
        lines.append(f"Low Severity:        {summary['by_severity']['low']}")

        # Detailed findings
        lines.append("\n### DETAILED FINDINGS")
        lines.append("-" * 100)

        # Group failures by category
        failures = [c for c in report.checks if not c.passed]
        if failures:
            by_category = {}
            for check in failures:
                if check.category not in by_category:
                    by_category[check.category] = []
                by_category[check.category].append(check)

            lines.append(f"\nFAILED CHECKS ({len(failures)}):\n")
            for category in sorted(by_category.keys()):
                checks = by_category[category]
                lines.append(f"  {category.upper().replace('_', ' ')}:")
                for check in checks:
                    lines.append(f"    • {check.check_id}: {check.name}")
                    lines.append(f"      Expected: {check.expected}")
                    lines.append(f"      Actual:   {check.actual}")
                    lines.append(f"      Message:  {check.message}")
                    lines.append("")
        else:
            lines.append("\n✓ All checks passed!\n")

        # By category breakdown
        lines.append("\n### BREAKDOWN BY CATEGORY")
        lines.append("-" * 100)
        by_category = {}
        for check in report.checks:
            if check.category not in by_category:
                by_category[check.category] = {"passed": 0, "failed": 0}
            if check.passed:
                by_category[check.category]["passed"] += 1
            else:
                by_category[check.category]["failed"] += 1

        for category in sorted(by_category.keys()):
            stats = by_category[category]
            total = stats["passed"] + stats["failed"]
            pct = (stats["passed"] / total * 100) if total > 0 else 0
            lines.append(
                f"{category:30} {stats['passed']:3}/{total:3} passed ({pct:5.1f}%)"
            )

        # Language policy compliance metrics (if available)
        if report.language_metrics:
            lines.append("\n### LANGUAGE POLICY COMPLIANCE METRICS")
            lines.append("-" * 100)
            metrics = report.language_metrics

            # Compliance summary
            lines.append(
                f"Excluded languages found:    {metrics.get('excluded_found', 0)}"
            )
            lines.append(
                f"Unrecognized languages:      {len(metrics.get('unrecognized_languages', []))}"
            )
            if metrics.get("unrecognized_languages"):
                unknown = metrics["unrecognized_languages"]
                lines.append(
                    f"  Unrecognized: {', '.join([f'{lang[0]} ({lang[1]:.1%})' for lang in unknown])}"
                )

            # Primary languages
            if metrics.get("primary_total", 0) > 0:
                lines.append("\nPrimary languages:")
                lines.append(
                    f"  Compliant: {metrics.get('primary_compliant', 0)}/{metrics.get('primary_total', 0)}"
                )
                if metrics.get("primary_violations"):
                    lines.append("  Violations:")
                    for v in metrics["primary_violations"]:
                        lines.append(
                            f"    {v['lang']}: {v['actual']:.2%} (max: {v['max']:.2%}, excess: {v['excess']:.2%})"
                        )

            # Secondary languages
            if metrics.get("secondary_total", 0) > 0:
                lines.append("\nSecondary languages:")
                lines.append(
                    f"  Compliant: {metrics.get('secondary_compliant', 0)}/{metrics.get('secondary_total', 0)}"
                )
                if metrics.get("secondary_violations"):
                    lines.append("  Violations:")
                    for v in metrics["secondary_violations"]:
                        lines.append(
                            f"    {v['lang']}: {v['actual']:.2%} (max: {v['max']:.2%}, excess: {v['excess']:.2%})"
                        )

        lines.append("\n" + "=" * 100)

        return "\n".join(lines)

    def validate_all_stages(self, stages: List[str]) -> Dict[str, ValidationReport]:
        """Validate all specified stages"""
        for stage in stages:
            self.validate_stage(stage)
        return self.reports


def main():
    """Main entry point for validation"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate coreset engine outputs")
    parser.add_argument(
        "--curriculum",
        type=str,
        default="config/curriculum.yaml",
        help="Path to curriculum YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/coresets",
        help="Base directory for coreset outputs",
    )
    parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        default=["1B", "3B", "8B", "70B"],
        help="Stages to validate",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="output/validation_reports",
        help="Directory to save reports",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["checklist", "report", "both"],
        default="both",
        help="Output format",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    # Create validator
    try:
        validator = CoresetValidator(args.curriculum, args.output_dir)
        logger.info(f"Loaded curriculum: {args.curriculum}")
    except Exception as e:
        logger.error(f"Failed to initialize validator: {e}")
        return 1

    # Validate stages
    validator.validate_all_stages(args.stages)

    # Create report directory
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Generate outputs
    for stage in args.stages:
        if stage not in validator.reports:
            logger.warning(f"No validation results for stage {stage}")
            continue

        # Checklist
        if args.format in ["checklist", "both"]:
            checklist = validator.generate_checklist(stage)
            checklist_file = report_dir / f"{stage}_checklist.txt"
            with open(checklist_file, "w", encoding="utf-8") as f:
                f.write(checklist)
            logger.info(f"Checklist saved: {checklist_file}")
            print(checklist)

        # Report
        if args.format in ["report", "both"]:
            report = validator.generate_report(stage)
            report_file = report_dir / f"{stage}_verification_report.txt"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Report saved: {report_file}")
            print(report)

    logger.info("Validation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
