"""
Coreset Output Validator v2 - Validates coreset engine outputs against curriculum.
Supports both local filesystem and S3 URIs (s3://bucket/path).
Generates checklists and verification reports for manifest and selected indices.
"""

import io
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.types import DifficultyBand, difficulty_band_order
from src.curriculum.loader import CurriculumLoader

try:
    import boto3
    from botocore.exceptions import ClientError

    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False


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
    manifest_path: str  # Path or S3 URI
    indices_path: str  # Path or S3 URI
    checks: List[ValidationCheck] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""
    language_metrics: Dict[str, Any] = field(default_factory=dict)

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
    """Validates coreset outputs against curriculum. Supports local and S3."""

    def __init__(self, curriculum_path: str, output_base_dir: str = "output/coresets"):
        self.curriculum_path = Path(curriculum_path)
        self.output_base_dir = output_base_dir
        self.is_s3 = output_base_dir.startswith("s3://")

        if self.is_s3 and not S3_AVAILABLE:
            raise ImportError("boto3 is required for S3 support but not found.")

        self.curriculum = CurriculumLoader(str(curriculum_path))
        success, errors = self.curriculum.load()
        if not success:
            raise ValueError(f"Failed to load curriculum: {errors}")

        self.logger = logging.getLogger(__name__)
        self.reports: Dict[str, ValidationReport] = {}

        self.s3_client = None
        if self.is_s3:
            self.s3_client = boto3.client("s3")
            match = re.match(r"s3://([^/]+)/(.*)", output_base_dir)
            if not match:
                raise ValueError(f"Invalid S3 URI: {output_base_dir}")
            self.bucket = match.group(1)
            self.prefix = match.group(2).rstrip("/")

    def _path_exists(self, path: str) -> bool:
        """Check if path or S3 key exists."""
        if path.startswith("s3://"):
            match = re.match(r"s3://([^/]+)/(.*)", path)
            if not match:
                return False
            bucket, key = match.group(1), match.group(2)
            try:
                self.s3_client.head_object(Bucket=bucket, Key=key)
                return True
            except ClientError:
                # Try as a prefix (directory)
                response = self.s3_client.list_objects_v2(
                    Bucket=bucket, Prefix=key, MaxKeys=1
                )
                return "Contents" in response or "CommonPrefixes" in response
        else:
            return Path(path).exists()

    def _list_files(self, directory: str, pattern: str) -> List[str]:
        """List files in directory matching pattern (supports glob-ish patterns for S3)."""
        if directory.startswith("s3://"):
            match = re.match(r"s3://([^/]+)/(.*)", directory)
            if not match:
                return []
            bucket, prefix = match.group(1), match.group(2).rstrip("/")
            if prefix:
                prefix += "/"

            # Simple regex conversion for glob patterns like "manifest_shard*.json"
            regex_pattern = pattern.replace(".", "\\.").replace("*", ".*")
            regex = re.compile(f"^{re.escape(prefix)}{regex_pattern}$")

            results = []
            paginator = self.s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if regex.match(key):
                        results.append(f"s3://{bucket}/{key}")
            return sorted(results)
        else:
            p = Path(directory)
            return [str(f) for f in sorted(p.glob(pattern))]

    def _resolve_manifest_path(self, stage_dir: str) -> str:
        """Resolve manifest path for both legacy and streaming outputs."""
        if not self._path_exists(stage_dir):
            return ""

        legacy = f"{stage_dir.rstrip('/')}/manifest.json"
        if self._path_exists(legacy):
            return legacy

        shard_manifests = self._list_files(stage_dir, "manifest_shard*.json")
        if shard_manifests:
            return shard_manifests[0]

        return legacy

    def _resolve_indices_path(self, stage_dir: str) -> str:
        """Resolve indices path (parquet preferred, jsonl fallback)."""
        if not self._path_exists(stage_dir):
            return ""

        parquet = f"{stage_dir.rstrip('/')}/selected_indices.parquet"
        if self._path_exists(parquet):
            return parquet

        jsonl = f"{stage_dir.rstrip('/')}/selected_indices.jsonl"
        if self._path_exists(jsonl):
            return jsonl

        # Try part files
        part_files = self._list_files(stage_dir, "selected_indices_part_*.parquet")
        if part_files:
            return part_files[0]

        return jsonl

    def _load_manifest(self, path: str) -> Dict[str, Any]:
        """Load manifest JSON from local or S3."""
        if not path:
            return {}
        try:
            if path.startswith("s3://"):
                match = re.match(r"s3://([^/]+)/(.*)", path)
                bucket, key = match.group(1), match.group(2)
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                return json.loads(response["Body"].read().decode("utf-8"))
            else:
                with open(path, "r") as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load manifest at {path}: {e}")
            return {}

    def _load_selected_indices(self, path: str) -> List[Dict[str, Any]]:
        """Load selected indices from JSONL or Parquet (local or S3)."""
        if not path:
            return []
        indices: List[Dict[str, Any]] = []
        is_parquet = path.lower().endswith(".parquet")

        if path.startswith("s3://"):
            if is_parquet:
                try:
                    import pandas as pd

                    # pandas.read_parquet supports s3:// if s3fs/fsspec is installed
                    df = pd.read_parquet(path)
                    return df.to_dict(orient="records")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load S3 parquet with pandas: {e}. Trying pyarrow/boto3 buffer..."
                    )
                    try:
                        import pyarrow.parquet as pq

                        match = re.match(r"s3://([^/]+)/(.*)", path)
                        bucket, key = match.group(1), match.group(2)
                        response = self.s3_client.get_object(Bucket=bucket, Key=key)
                        buffer = io.BytesIO(response["Body"].read())
                        table = pq.read_table(buffer)
                        return table.to_pylist()
                    except Exception as e2:
                        self.logger.error(
                            f"Failed all S3 parquet loading methods for {path}: {e2}"
                        )
                        return []
            else:
                # S3 JSONL
                try:
                    match = re.match(r"s3://([^/]+)/(.*)", path)
                    bucket, key = match.group(1), match.group(2)
                    response = self.s3_client.get_object(Bucket=bucket, Key=key)
                    for line in response["Body"].iter_lines():
                        if line:
                            indices.append(json.loads(line.decode("utf-8")))
                    return indices
                except Exception as e:
                    self.logger.error(f"Failed to load S3 JSONL {path}: {e}")
                    return []
        else:
            # Local loading logic (existing)
            if is_parquet:
                try:
                    import pyarrow.parquet as pq

                    table = pq.read_table(path)
                    return table.to_pylist()
                except Exception:
                    import pandas as pd

                    df = pd.read_parquet(path)
                    return df.to_dict(orient="records")
            else:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                indices.append(json.loads(line))
                except Exception as e:
                    self.logger.error(f"Failed to load local indices {path}: {e}")
            return indices

    def validate_stage(self, stage_name: str) -> ValidationReport:
        """Validate all outputs for a stage. Handles flat, split, and double-slash layouts."""
        manifest_path = ""
        indices_path = ""

        # 1. Prepare search directories
        search_dirs = []
        if self.is_s3:
            # Standard flat stage dir
            search_dirs.append(f"s3://{self.bucket}/{self.prefix}/{stage_name}")
            # Split manifests/coresets dirs
            search_dirs.append(
                f"s3://{self.bucket}/{self.prefix}/coresets/{stage_name}"
            )
            search_dirs.append(
                f"s3://{self.bucket}/{self.prefix}/manifests/{stage_name}"
            )
            # Double-slash variants (observed in user environment)
            search_dirs.append(
                f"s3://{self.bucket}/{self.prefix}//coresets/{stage_name}"
            )
            search_dirs.append(
                f"s3://{self.bucket}/{self.prefix}//manifests/{stage_name}"
            )
        else:
            p_base = Path(self.output_base_dir)
            search_dirs.append(str(p_base / stage_name))
            search_dirs.append(str(p_base / "coresets" / stage_name))
            search_dirs.append(str(p_base / "manifests" / stage_name))

        # 2. Resolve Manifest (JSON)
        for d in search_dirs:
            manifest_path = self._resolve_manifest_path(d)
            if manifest_path:
                self.logger.info(f"Resolved manifest: {manifest_path}")
                break

        # 3. Resolve Indices (Parquet/JSONL)
        for d in search_dirs:
            indices_path = self._resolve_indices_path(d)
            if indices_path:
                self.logger.info(f"Resolved indices: {indices_path}")
                break

        report = ValidationReport(
            stage_name=stage_name,
            manifest_path=manifest_path or "NOT_FOUND",
            indices_path=indices_path or "NOT_FOUND",
            generated_at=self._get_timestamp(),
        )

        # Check file existence
        self._validate_files_exist(report)

        # If manifest is missing, no further checks are possible.
        if not manifest_path or not self._path_exists(manifest_path):
            self.reports[stage_name] = report
            return report

        # Load outputs
        manifest = self._load_manifest(manifest_path)

        indices: List[Dict[str, Any]] = []
        if indices_path and self._path_exists(indices_path):
            indices = self._load_selected_indices(indices_path)

        # Validate manifest structure
        self._validate_manifest_structure(report, manifest)

        # Validate indices format (if indices exist)
        if self._path_exists(indices_path):
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

    def _validate_files_exist(self, report: ValidationReport):
        """Check if required files exist"""
        manifest_exists = self._path_exists(report.manifest_path)
        indices_exists = self._path_exists(report.indices_path)

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
                details="Selected indices file (Parquet/JSONL) should exist for stage",
            )
        )

    # --- Inherited / Re-implemented Validation Logic ---

    def _get_availability_stats(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        availability = manifest.get("availability_stats")
        return availability if isinstance(availability, dict) else {}

    def _get_selected_tokens_from_manifest(self, manifest: Dict[str, Any]) -> int:
        return int(
            manifest.get("actual_tokens")
            or manifest.get("selected_tokens")
            or manifest.get("total_tokens")
            or 0
        )

    def _validate_manifest_structure(self, report: ValidationReport, manifest: Dict):
        required_fields = ["stage_name", "composition", "protected_slices_preserved"]
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
        tolerance = 0.02
        availability = self._get_availability_stats(manifest)
        eligible_total = int(availability.get("eligible_unused_tokens_total", 0) or 0)
        eligible_by_band = availability.get("eligible_unused_tokens_by_band", {})
        selected_total_tokens = self._get_selected_tokens_from_manifest(manifest)

        for band_name in difficulty_band_order():
            expected = getattr(expected_ratios, band_name, 0.0)
            actual = band_dist.get(band_name, 0.0)
            if actual > 1.0:
                actual /= 100.0
            passed = abs(expected - actual) <= tolerance

            if (not passed) and selected_total_tokens > 0 and eligible_total > 0:
                eligible_band_tokens = int(eligible_by_band.get(band_name, 0) or 0)
                upper_share_possible = min(
                    1.0, eligible_band_tokens / float(selected_total_tokens)
                )
                if actual < (expected - tolerance) and upper_share_possible < (
                    expected - tolerance
                ):
                    passed = True
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
                    message=f"{band_name}: expected {expected:.2%}, got {actual:.2%}"
                    + (
                        " (availability-limited)"
                        if passed and abs(expected - actual) > tolerance
                        else ""
                    ),
                    details=f"Tolerance: {tolerance:.2%}",
                )
            )

    def _validate_domain_distribution(self, report: ValidationReport, manifest: Dict):
        composition = manifest.get("composition", {})
        domain_dist = composition.get("domain_distribution", {})
        stage_config = self.curriculum.stages.get(report.stage_name)
        if not stage_config:
            return

        by_band = domain_dist.get("by_band") or domain_dist.get("byBand")
        if by_band:
            for band_name, domain_ratio in by_band.items():
                try:
                    band = DifficultyBand(band_name)
                except Exception:
                    continue
                allowed = self.curriculum.get_allowed_domains_for_band(band)
                for domain_name in domain_ratio.keys():
                    domain_allowed = domain_name in allowed
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
                            details=f"Allowed domains for {band_name}: {allowed}",
                        )
                    )
            return

        if isinstance(domain_dist, dict) and domain_dist:
            used_bands = [
                b
                for b in difficulty_band_order()
                if getattr(stage_config.band_ratios, b, 0.0) > 0
            ]
            for domain_name in domain_dist.keys():
                allowed_somewhere = False
                for band_name in used_bands:
                    try:
                        band = DifficultyBand(band_name)
                    except Exception:
                        continue
                    if domain_name in self.curriculum.get_allowed_domains_for_band(
                        band
                    ):
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
                        message=f"{domain_name} is allowed in stage context",
                    )
                )

    def _validate_language_distribution(self, report: ValidationReport, manifest: Dict):
        composition = manifest.get("composition", {})
        lang_dist = composition.get("language_distribution", {})
        if not self.curriculum.language_policy:
            return

        policy = self.curriculum.language_policy
        primary = policy.primary_languages
        secondary = policy.secondary_languages
        excluded = policy.explicitly_excluded
        tolerance = 0.01

        metrics = {
            "excluded_found": 0,
            "primary_total": 0,
            "primary_compliant": 0,
            "secondary_total": 0,
            "secondary_compliant": 0,
            "unrecognized_languages": [],
        }

        for lang_code, token_share in lang_dist.items():
            is_excluded = lang_code in excluded
            report.add_check(
                ValidationCheck(
                    check_id=f"LANG_EXCLUDED_{lang_code}",
                    category="language_policy",
                    name=f"Language {lang_code} NOT excluded",
                    expected=False,
                    actual=is_excluded,
                    passed=not is_excluded,
                    severity="critical" if is_excluded else "low",
                    message=f"{lang_code} found status: {'EXCLUDED' if is_excluded else 'OK'}",
                )
            )
            if is_excluded:
                metrics["excluded_found"] += 1

            if lang_code in primary:
                metrics["primary_total"] += 1
                max_s = primary[lang_code]
                ok = token_share <= max_s + tolerance
                if ok:
                    metrics["primary_compliant"] += 1
                report.add_check(
                    ValidationCheck(
                        check_id=f"LANG_PRIMARY_{lang_code}",
                        category="language_policy",
                        name=f"Primary {lang_code} compliance",
                        expected=max_s,
                        actual=token_share,
                        passed=ok,
                        severity="high" if not ok else "low",
                        message=f"{lang_code}: {token_share:.2%} <= {max_s:.2%}",
                    )
                )
            elif lang_code in secondary:
                metrics["secondary_total"] += 1
                max_s = secondary[lang_code]
                ok = token_share <= max_s + tolerance
                if ok:
                    metrics["secondary_compliant"] += 1
                report.add_check(
                    ValidationCheck(
                        check_id=f"LANG_SECONDARY_{lang_code}",
                        category="language_policy",
                        name=f"Secondary {lang_code} compliance",
                        expected=max_s,
                        actual=token_share,
                        passed=ok,
                        severity="high" if not ok else "low",
                        message=f"{lang_code}: {token_share:.2%} <= {max_s:.2%}",
                    )
                )
            else:
                metrics["unrecognized_languages"].append((lang_code, token_share))
                report.add_check(
                    ValidationCheck(
                        check_id=f"LANG_UNKNOWN_{lang_code}",
                        category="language_policy",
                        name=f"Language {lang_code} recognized",
                        expected=True,
                        actual=False,
                        passed=False,
                        severity="high",
                        message=f"{lang_code} not in policy",
                    )
                )

        comp_score = 0
        if metrics["excluded_found"] == 0:
            comp_score += 25
        if not metrics["unrecognized_languages"]:
            comp_score += 25
        if (
            metrics["primary_total"] == 0
            or metrics["primary_compliant"] == metrics["primary_total"]
        ):
            comp_score += 25
        if (
            metrics["secondary_total"] == 0
            or metrics["secondary_compliant"] == metrics["secondary_total"]
        ):
            comp_score += 25
        report.add_check(
            ValidationCheck(
                check_id="LANG_POLICY_COMPLIANCE_SCORE",
                category="language_policy",
                name="Overall language policy compliance",
                expected=100,
                actual=comp_score,
                passed=comp_score >= 75,
                severity="high" if comp_score < 75 else "low",
                message=f"Compliance Score: {comp_score}/100",
            )
        )
        report.language_metrics = metrics

    def _validate_stage_targets(self, report: ValidationReport, manifest: Dict):
        selected = self._get_selected_tokens_from_manifest(manifest)
        target = int(
            manifest.get("target_tokens_shard") or manifest.get("target_tokens") or 0
        )
        if target == 0:
            stage_config = self.curriculum.stages.get(report.stage_name)
            if stage_config:
                target = int(getattr(stage_config, "total_tokens", 0))

        ratio = selected / target if target > 0 else 0
        tolerance = 0.05
        within_tolerance = abs(1.0 - ratio) <= tolerance

        availability = self._get_availability_stats(manifest)
        eligible_total = int(availability.get("eligible_unused_tokens_total", 0) or 0)
        if (not within_tolerance) and target > 0 and eligible_total > 0:
            if eligible_total < int(target * (1.0 - tolerance)):
                within_tolerance = True

        report.add_check(
            ValidationCheck(
                check_id="STAGE_TARGET_TOKENS",
                category="stage_targets",
                name="Stage meets token target (±5%)",
                expected=target,
                actual=selected,
                passed=within_tolerance,
                severity="high" if not within_tolerance else "low",
                message=f"Ratio: {ratio:.2%} (Target: {target:,}, Actual: {selected:,})"
                + (
                    " (availability-limited)"
                    if within_tolerance and abs(1 - ratio) > tolerance
                    else ""
                ),
            )
        )

    def _validate_rolling_window(self, report: ValidationReport, manifest: Dict):
        if not self.curriculum.rolling_window or "rolling_window_stats" not in manifest:
            return
        stats = manifest.get("rolling_window_stats", {})
        max_b_delta = stats.get("max_band_delta", 0)
        max_d_delta = stats.get("max_domain_delta", 0)
        limit_b = self.curriculum.rolling_window.max_band_delta
        limit_d = self.curriculum.rolling_window.max_domain_delta
        report.add_check(
            ValidationCheck(
                check_id="ROLLING_WINDOW_BAND",
                category="rolling_window",
                name="Rolling window band delta within constraint",
                expected=limit_b,
                actual=max_b_delta,
                passed=max_b_delta <= limit_b,
                severity="high" if max_b_delta > limit_b else "low",
                message=f"Max band delta: {max_b_delta:.4f} <= {limit_b:.4f}",
            )
        )
        report.add_check(
            ValidationCheck(
                check_id="ROLLING_WINDOW_DOMAIN",
                category="rolling_window",
                name="Rolling window domain delta within constraint",
                expected=limit_d,
                actual=max_d_delta,
                passed=max_d_delta <= limit_d,
                severity="high" if max_d_delta > limit_d else "low",
                message=f"Max domain delta: {max_d_delta:.4f} <= {limit_d:.4f}",
            )
        )

    def _validate_protected_slices(self, report: ValidationReport, manifest: Dict):
        protected = manifest.get("protected_slices", {})
        if protected:
            report.add_check(
                ValidationCheck(
                    check_id="PROTECTED_SLICES_PRESENT",
                    category="protected_slices",
                    name="Protected slices enforced",
                    expected=True,
                    actual=True,
                    passed=True,
                    severity="low",
                    message=f"Slices: {list(protected.keys())}",
                )
            )

    def _get_timestamp(self) -> str:
        from datetime import datetime

        return datetime.now().isoformat()

    def generate_checklist(self, stage_name: str) -> str:
        report = self.reports.get(stage_name)
        if not report:
            return f"No report for {stage_name}"
        lines = [
            "=" * 80,
            f"CORESET VALIDATION CHECKLIST - Stage {stage_name}",
            "=" * 80,
        ]
        by_cat = {}
        for c in report.checks:
            by_cat.setdefault(c.category, []).append(c)
        for cat in sorted(by_cat.keys()):
            checks = by_cat[cat]
            psd = sum(1 for c in checks if c.passed)
            lines.append(f"\n### {cat.upper().replace('_', ' ')} ({psd}/{len(checks)})")
            lines.append("-" * 80)
            for c in checks:
                status = "✓ PASS" if c.passed else "✗ FAIL"
                lines.append(f"{status} [{c.severity.upper():8}] {c.name}")
                lines.append(f"         {c.message}")
        return "\n".join(lines)

    def generate_report(self, stage_name: str) -> str:
        report = self.reports.get(stage_name)
        if not report:
            return f"No report for {stage_name}"
        sm = report.get_summary()
        lines = [
            "=" * 100,
            f"CORESET ENGINE VERIFICATION REPORT - Stage {stage_name}",
            "=" * 100,
        ]
        lines.append(f"\nGenerated: {report.generated_at}")
        lines.append(f"Manifest:  {report.manifest_path}")
        lines.append(f"Indices:   {report.indices_path}")
        lines.append("\n### SUMMARY")
        lines.append(
            f"Success Rate: {sm['success_rate']:.1f}% ({sm['by_status']['passed']}/{sm['total_checks']} passed)"
        )
        lines.append(
            f"Severity Breakdown: Critical={sm['by_severity']['critical']}, High={sm['by_severity']['high']}"
        )

        failures = [c for c in report.checks if not c.passed]
        if failures:
            lines.append("\n### FAILED CHECKS")
            for c in failures:
                lines.append(f"  [{c.severity.upper()}] {c.check_id}: {c.name}")
                lines.append(f"    Expected: {c.expected}")
                lines.append(f"    Actual:   {c.actual}")
                lines.append(f"    Message:  {c.message}\n")
        else:
            lines.append("\n✓ All checks passed!")

        if report.language_metrics:
            lines.append("\n### LANGUAGE POLICY")
            m = report.language_metrics
            lines.append(f"  Excluded found: {m['excluded_found']}")
            lines.append(
                f"  Primary compliant: {m['primary_compliant']}/{m['primary_total']}"
            )
            lines.append(
                f"  Secondary compliant: {m['secondary_compliant']}/{m['secondary_total']}"
            )

        return "\n".join(lines)

    def validate_all_stages(self, stages: List[str]) -> Dict[str, ValidationReport]:
        for s in stages:
            self.validate_stage(s)
        return self.reports


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate coreset engine outputs (v2 S3 support)"
    )
    parser.add_argument("--curriculum", type=str, default="config/curriculum.yaml")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/coresets",
        help="Base dir or s3:// URI",
    )
    parser.add_argument(
        "--stages", type=str, nargs="+", default=["1B", "3B", "8B", "70B"]
    )
    parser.add_argument("--report-dir", type=str, default="output/validation_reports")
    parser.add_argument(
        "--format", type=str, choices=["checklist", "report", "both"], default="both"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    try:
        validator = CoresetValidator(args.curriculum, args.output_dir)
        logger.info(f"Initialized validator for: {args.output_dir}")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return 1

    validator.validate_all_stages(args.stages)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    for stage in args.stages:
        if stage not in validator.reports:
            continue
        if args.format in ["checklist", "both"]:
            txt = validator.generate_checklist(stage)
            with open(report_dir / f"{stage}_checklist.txt", "w") as f:
                f.write(txt)
            print(txt)
        if args.format in ["report", "both"]:
            txt = validator.generate_report(stage)
            with open(report_dir / f"{stage}_verification_report.txt", "w") as f:
                f.write(txt)
            print(txt)

    logger.info("Validation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
