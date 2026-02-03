"""
AWS Lambda function for coreset manifests validation and health checks.

Triggers:
- Post-deployment validation in ECS
- Scheduled manifest integrity checks
- Manual invocation via CI/CD pipeline
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")
cloudwatch = boto3.client("cloudwatch")


def lambda_handler(event, context):
    """
    Main Lambda handler for coreset validation.

    Event structure:
    {
        "environment": "staging" | "production",
        "bucket": "llm-coreset-artifacts-...",
        "manifest_path": "manifests/1B/manifest.json",
        "check_type": "quick" | "comprehensive"
    }
    """

    try:
        environment = event.get("environment", "staging")
        bucket = event.get(
            "bucket",
            f'llm-coreset-artifacts-{context.invoked_function_arn.split(":")[4]}-{environment}',
        )
        manifest_path = event.get("manifest_path", "manifests/1B/manifest.json")
        check_type = event.get("check_type", "quick")

        logger.info(f"Starting validation for {environment}: {manifest_path}")

        # Load manifest from S3
        manifest = load_manifest(bucket, manifest_path)

        # Run validation checks
        validation_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "environment": environment,
            "manifest_path": manifest_path,
            "checks": {},
        }

        # Basic structure validation
        validation_results["checks"]["schema"] = validate_manifest_schema(manifest)

        # Quick checks
        validation_results["checks"]["completeness"] = validate_completeness(manifest)
        validation_results["checks"]["consistency"] = validate_consistency(manifest)

        # Comprehensive checks
        if check_type == "comprehensive":
            validation_results["checks"]["distribution"] = validate_distributions(
                manifest
            )
            validation_results["checks"]["curriculum_compliance"] = (
                validate_curriculum_compliance(manifest)
            )

        # Overall status
        validation_results["status"] = (
            "PASS"
            if all(
                check.get("passed", False)
                for check in validation_results["checks"].values()
            )
            else "FAIL"
        )

        # Log metrics to CloudWatch
        log_metrics(validation_results)

        # Log results
        logger.info(json.dumps(validation_results, indent=2))

        return {
            "statusCode": 200 if validation_results["status"] == "PASS" else 400,
            "body": json.dumps(validation_results),
        }

    except Exception as e:
        logger.error(f"Validation failed: {str(e)}", exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


def load_manifest(bucket: str, key: str) -> Dict[str, Any]:
    """Load manifest JSON from S3."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        manifest = json.loads(response["Body"].read().decode("utf-8"))
        return manifest
    except Exception as e:
        logger.error(f"Failed to load manifest from s3://{bucket}/{key}: {str(e)}")
        raise


def validate_manifest_schema(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Validate manifest schema against expected structure."""
    required_fields = ["metadata", "stages"]
    required_stage_fields = ["indices", "statistics", "metadata"]
    required_stats_fields = [
        "total_samples",
        "band_distribution",
        "modality_distribution",
    ]

    issues = []

    # Check top-level structure
    for field in required_fields:
        if field not in manifest:
            issues.append(f"Missing top-level field: {field}")

    # Check stage structure
    if "stages" in manifest:
        for stage_name, stage_data in manifest["stages"].items():
            for field in required_stage_fields:
                if field not in stage_data:
                    issues.append(f"Stage '{stage_name}' missing field: {field}")

            # Check statistics
            if "statistics" in stage_data:
                for field in required_stats_fields:
                    if field not in stage_data["statistics"]:
                        issues.append(
                            f"Stage '{stage_name}' statistics missing: {field}"
                        )

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "checked_fields": required_fields
        + required_stage_fields
        + required_stats_fields,
    }


def validate_completeness(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that all required stages are present."""
    required_stages = ["1B", "3B", "8B", "70B"]
    issues = []

    if "stages" in manifest:
        present_stages = list(manifest["stages"].keys())
        for stage in required_stages:
            if stage not in present_stages:
                issues.append(f"Missing stage: {stage}")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "expected_stages": required_stages,
    }


def validate_consistency(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Validate consistency across stages (e.g., stage progression)."""
    issues = []

    if "stages" not in manifest:
        return {"passed": False, "issues": ["No stages found in manifest"]}

    stages = manifest["stages"]
    stage_order = ["1B", "3B", "8B", "70B"]

    # Convert indices to sets for each stage
    stage_indices = {}
    for stage_name in stage_order:
        if stage_name in stages:
            indices = stages[stage_name].get("indices", [])
            stage_indices[stage_name] = (
                set(indices) if isinstance(indices, list) else set()
            )

    # Verify progression: each stage should contain previous stages' data
    for i in range(len(stage_order) - 1):
        curr_stage = stage_order[i]
        next_stage = stage_order[i + 1]

        if curr_stage in stage_indices and next_stage in stage_indices:
            if not stage_indices[curr_stage].issubset(stage_indices[next_stage]):
                issues.append(
                    f"Stage progression violation: {curr_stage} indices not subset of {next_stage}"
                )

    return {"passed": len(issues) == 0, "issues": issues, "stages_checked": stage_order}


def validate_distributions(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Validate band and modality distributions."""
    issues = []

    if "stages" not in manifest:
        return {"passed": False, "issues": ["No stages found"]}

    for stage_name, stage_data in manifest["stages"].items():
        if "statistics" not in stage_data:
            continue

        stats = stage_data["statistics"]

        # Check band distribution sums to 1.0 (with tolerance)
        if "band_distribution" in stats:
            band_sum = sum(stats["band_distribution"].values())
            if not (0.99 <= band_sum <= 1.01):
                issues.append(
                    f"Stage '{stage_name}' band distribution sums to {band_sum}, expected ~1.0"
                )

        # Check modality distribution sums to 1.0 (with tolerance)
        if "modality_distribution" in stats:
            modality_sum = sum(stats["modality_distribution"].values())
            if not (0.99 <= modality_sum <= 1.01):
                issues.append(
                    f"Stage '{stage_name}' modality distribution sums to {modality_sum}, expected ~1.0"
                )

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "stages_validated": list(manifest.get("stages", {}).keys()),
    }


def validate_curriculum_compliance(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Validate adherence to curriculum constraints."""
    issues = []

    # Expected curriculum profiles
    expected_profiles = {
        "1B": {"target_tokens": 1_000_000_000},
        "3B": {"target_tokens": 3_000_000_000},
        "8B": {"target_tokens": 8_000_000_000},
        "70B": {"target_tokens": 70_000_000_000},
    }

    if "stages" in manifest:
        for stage_name, stage_config in expected_profiles.items():
            if stage_name not in manifest["stages"]:
                issues.append(f"Stage '{stage_name}' missing")
                continue

            stage = manifest["stages"][stage_name]

            # Check that stage has required metadata
            if "metadata" not in stage:
                issues.append(f"Stage '{stage_name}' missing metadata")

            # Check statistics exist
            if "statistics" not in stage:
                issues.append(f"Stage '{stage_name}' missing statistics")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "expected_stages": list(expected_profiles.keys()),
    }


def log_metrics(validation_results: Dict[str, Any]):
    """Log validation metrics to CloudWatch."""
    try:
        environment = validation_results["environment"]
        status = 1 if validation_results["status"] == "PASS" else 0

        cloudwatch.put_metric_data(
            Namespace="CoresetEngineering",
            MetricData=[
                {
                    "MetricName": "ValidationStatus",
                    "Value": status,
                    "Unit": "Count",
                    "Dimensions": [
                        {"Name": "Environment", "Value": environment},
                        {
                            "Name": "ManifestPath",
                            "Value": validation_results["manifest_path"],
                        },
                    ],
                }
            ],
        )

        logger.info(f"Metrics logged to CloudWatch for {environment}")
    except Exception as e:
        logger.warning(f"Failed to log metrics: {str(e)}")
