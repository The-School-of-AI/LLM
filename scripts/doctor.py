#!/usr/bin/env python3
"""Validate a LightningLM training checkout.

This script intentionally avoids third-party dependencies so it can run before a
CUDA/PyTorch environment is installed.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

REQUIRED_FILES = [
    "README.md",
    "pyproject.toml",
    "docs/cookbook.md",
    "docs/data_pipeline.md",
    "docs/runtime_hotconfig.md",
    "docs/tokenizer_pipeline.md",
    "configs/curriculum_v2.yaml",
    "configs/train_2b.yaml",
    "configs/train_5b.yaml",
    "configs/train_9b.yaml",
    "configs/train_120b_tqp.yaml",
    "scripts/run_training.sh",
    "scripts/train_entrypoint.py",
    "scripts/create_curriculum_test_shards.py",
    "scripts/build_120b_init.py",
    "scripts/data/process.py",
    "scripts/data/verify.py",
    "scripts/data/test_cleaning_smoke.py",
    "scripts/tokenizer/build_tokenizer.py",
    "scripts/hash_tensors.py",
    "lightninglm/__init__.py",
    "lightninglm/data/curriculum_dataloader_v2.py",
    "lightninglm/growth/dense_to_moe.py",
    "lightninglm/tqp/tqp_integration.py",
    "lightninglm/growth/depth_map.py",
    "tokenizer/tokenizer_reordered.json",
    "tokenizer/token_permutation.npy",
    "tokenizer/token_inv_permutation.npy",
]

MANIFEST_COUNTS = {
    "D1_shards.txt": 4894,
    "D2_shards.txt": 18710,
    "D3_shards.txt": 5933,
    "D4_shards.txt": 1464,
    "AON_bench_train_shards.txt": 356,
    "AON_indic_shards.txt": 1996,
    "GP_shards.txt": 11,
    "DROPPED_B2_shards.txt": 934,
}

_DEPRECATED_TURBOQUANT_NAMES = [
    "TQ-" + "Lo" + "RA",
    "TQ" + "Lo" + "RA",
    "T" + "Q" + "L",
    "tq_" + "lo" + "ra",
    "tq" + "lo" + "ra",
]

FORBIDDEN_PATTERNS = [
    (re.compile(r"\bfrom\s+src\.|\bimport\s+src\b"), "internal src import"),
    (re.compile(r"\bfrom aws\.|\bimport aws\b"), "top-level aws import"),
    (
        re.compile(r"\bfrom components\.|\bimport components\b"),
        "top-level components import",
    ),
    (
        re.compile("|".join(re.escape(name) for name in _DEPRECATED_TURBOQUANT_NAMES)),
        "deprecated TurboQuant adapter name",
    ),
]

PATH_KEYS = {
    "tokenizer_dir",
    "curriculum_config_path",
    "manifest_dir",
    "lr_schedule_path",
    "config_path",
}

OPTIONAL_PATH_KEYS = {
    "init_model_path",
    "shard_dir",
    "eval_shard_dir",
    "metrics_jsonl_path",
    "output_dir",
    "proxy_token_path",
}


@dataclass
class Finding:
    level: str
    message: str


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def iter_text_files(root: Path):
    suffixes = {".py", ".sh", ".yaml", ".yml", ".md", ".json", ".txt", ".toml"}
    skip_dirs = {"__pycache__", ".git"}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        yield path


def scan_forbidden(root: Path, findings: list[Finding]) -> None:
    for path in iter_text_files(root):
        if "tokenizer" in path.parts or "manifests" in path.parts:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for pattern, label in FORBIDDEN_PATTERNS:
                if pattern.search(line):
                    findings.append(
                        Finding(
                            "error",
                            f"{rel(path, root)}:{lineno}: {label}: {line.strip()}",
                        )
                    )


def parse_simple_yaml_paths(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip().strip("'\"")
        if not key or val in {"", "null", "Null", "None"}:
            continue
        if key in PATH_KEYS or key in OPTIONAL_PATH_KEYS:
            values[key] = val
    return values


def resolve_config_path(config: Path, value: str) -> Path:
    value = value.replace("$REPO_ROOT", str(config.parents[1]))
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (config.parent / candidate).resolve()


def check_config_paths(
    root: Path, findings: list[Finding], strict_public: bool
) -> None:
    for config in sorted((root / "configs").glob("*.yaml")):
        values = parse_simple_yaml_paths(config)
        for key, value in values.items():
            resolved = resolve_config_path(config, value)
            is_machine_path = value.startswith("/mnt/") or value.startswith("/Users/")
            if key in PATH_KEYS and not resolved.exists() and not is_machine_path:
                findings.append(
                    Finding(
                        "error",
                        f"{rel(config, root)}: {key} points to missing path: {value}",
                    )
                )
            if is_machine_path:
                level = "error" if strict_public else "warn"
                findings.append(
                    Finding(
                        level,
                        f"{rel(config, root)}: {key} is machine-specific: {value}",
                    )
                )
            elif key in OPTIONAL_PATH_KEYS and not resolved.exists():
                continue


def check_required_files(root: Path, findings: list[Finding]) -> None:
    for name in REQUIRED_FILES:
        if not (root / name).exists():
            findings.append(Finding("error", f"missing required file: {name}"))


def check_manifest_counts(root: Path, findings: list[Finding]) -> None:
    manifest_dir = root / "manifests"
    for filename, expected in MANIFEST_COUNTS.items():
        path = manifest_dir / filename
        if not path.exists():
            findings.append(Finding("error", f"missing manifest: manifests/{filename}"))
            continue
        count = sum(1 for line in path.read_text().splitlines() if line.strip())
        if count != expected:
            findings.append(
                Finding(
                    "error",
                    f"manifest count mismatch for {filename}: expected {expected}, got {count}",
                )
            )


def check_tqp_layout(root: Path, findings: list[Finding]) -> None:
    tqp_dir = root / "lightninglm" / "tqp"
    expected = [
        "turboquant_pretraining_linear.py",
        "tqp_integration.py",
        "tqp_moe_expert_parallel.py",
        "triton_tqp_grouped.py",
        "codebook.py",
        "stochastic_round.py",
    ]
    for name in expected:
        if not (tqp_dir / name).exists():
            findings.append(
                Finding("error", f"missing TQP module: lightninglm/tqp/{name}")
            )


def check_no_runtime_artifacts(root: Path, findings: list[Finding]) -> None:
    for path in root.rglob("__pycache__"):
        if path.is_dir():
            findings.append(
                Finding("warn", f"runtime artifact present: {rel(path, root)}")
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--strict-public",
        action="store_true",
        help="treat machine-specific paths as errors",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    findings: list[Finding] = []
    check_required_files(root, findings)
    check_manifest_counts(root, findings)
    check_tqp_layout(root, findings)
    check_no_runtime_artifacts(root, findings)
    scan_forbidden(root, findings)
    check_config_paths(root, findings, strict_public=args.strict_public)

    errors = [f for f in findings if f.level == "error"]
    warnings = [f for f in findings if f.level == "warn"]

    for finding in errors + warnings:
        print(f"[{finding.level}] {finding.message}")

    if errors:
        print(f"\nFAILED: {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1

    print(f"OK: no errors ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
