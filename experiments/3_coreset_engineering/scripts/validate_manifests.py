#!/usr/bin/env python3
"""
CI reproducibility validation script.

Run this in CI/CD pipeline to ensure manifests meet reproducibility standards:
- Seed correctness (must be 42)
- JSON schema compliance
- Fingerprint validity
- Determinism markers in audit trail

Usage:
    python scripts/validate_manifests.py --manifest-dir output/manifests/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coreset_engine.reproducibility import ReproducibilityValidator


def validate_manifest_file(manifest_path: str) -> Tuple[bool, List[str]]:
    """
    Validate a single manifest file.

    Args:
        manifest_path: Path to manifest.json

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    except FileNotFoundError:
        return False, [f"File not found: {manifest_path}"]

    # Run validation
    validator = ReproducibilityValidator(manifest)

    if not validator.validate_all():
        # Collect specific failures
        for check_name, result in validator.validation_results.items():
            if not result:
                issues.append(f"✗ {check_name} failed")

    return len(issues) == 0, issues


def validate_manifest_directory(manifest_dir: str) -> Tuple[bool, Dict]:
    """
    Validate all manifests in a directory.

    Args:
        manifest_dir: Directory containing manifest.json files

    Returns:
        (all_valid, summary_dict)
    """
    manifest_dir = Path(manifest_dir)

    if not manifest_dir.exists():
        print(f"✗ Directory not found: {manifest_dir}")
        return False, {}

    summary = {
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "files": {},
    }

    # Find all manifest.json files
    manifest_files = list(manifest_dir.rglob("manifest.json"))

    if not manifest_files:
        print(f"⚠ No manifest.json files found in {manifest_dir}")
        return False, summary

    print(f"Found {len(manifest_files)} manifest file(s)")

    for manifest_file in sorted(manifest_files):
        is_valid, issues = validate_manifest_file(str(manifest_file))

        summary["total_files"] += 1
        if is_valid:
            summary["valid_files"] += 1
            status = "✓ VALID"
        else:
            summary["invalid_files"] += 1
            status = "✗ INVALID"

        rel_path = manifest_file.relative_to(manifest_dir.parent)
        print(f"  {status}: {rel_path}")

        summary["files"][str(rel_path)] = {
            "valid": is_valid,
            "issues": issues,
        }

        if issues:
            for issue in issues:
                print(f"      {issue}")

    all_valid = summary["invalid_files"] == 0
    return all_valid, summary


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate manifests for reproducibility compliance"
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="output/manifests/",
        help="Directory containing manifest.json files (default: output/manifests/)",
    )
    parser.add_argument(
        "--manifest-file", type=str, help="Single manifest file to validate (optional)"
    )
    parser.add_argument(
        "--json-output", type=str, help="Output results as JSON to file (optional)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Manifest Reproducibility Validation")
    print("=" * 70)

    # Validate single file or directory
    if args.manifest_file:
        is_valid, issues = validate_manifest_file(args.manifest_file)
        summary = {
            "valid": is_valid,
            "issues": issues,
        }
    else:
        is_valid, summary = validate_manifest_directory(args.manifest_dir)

    # Print summary
    print("\n" + "=" * 70)
    if is_valid:
        print("✓ All manifests passed reproducibility validation!")
        exit_code = 0
    else:
        print("✗ Some manifests failed reproducibility validation")
        exit_code = 1

    # Output JSON if requested
    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results written to: {args.json_output}")

    print("=" * 70)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
