"""
run_pipeline.py — Synthetic Data Pipeline for LLM Training

Architecture (Two-Phase):

  PHASE 1: PREPARATION (Before Training)
  --------------------------------------
  generate-bank -> Pre-generate data for all skills
                   Store as ready-to-inject shards
  
  PHASE 2: INJECTION (During Training)  
  ------------------------------------
  diagnose -> Run tests, identify weak skills
  inject   -> Select shards from bank based on weakness
  validate -> Measure before/after delta

Commands:
  python run_pipeline.py generate-bank --all
  python run_pipeline.py diagnose --model qwen3:4b
  python run_pipeline.py inject --skills RSN-ARITHMETIC
  python run_pipeline.py validate synthetic.jsonl
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    SKILL_BUCKETS,
    STAGE_CONFIGS,
    Band,
    Stage,
    cot_allowed_for_band,
    get_injection_cap,
    get_skill_bucket,
)

DEFAULT_BANK_DIR = Path("./synth_data_bank")


def cmd_generate_bank(args):
    """Pre-generate synthetic data bank."""
    from generation.dual_view_generator import DualViewGenerator
    from generation.seed_generator import SeedGenerator, get_builtin_seeds

    bank_dir = Path(args.bank_dir)
    bank_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        skills = list(SKILL_BUCKETS.keys())
    elif args.skills:
        skills = args.skills
    else:
        skills = [
            s
            for s, spec in SKILL_BUCKETS.items()
            if spec.priority in ("critical", "high")
        ]

    tracker = args._tracker

    print("\n" + "=" * 60)
    print("  PHASE 1: PRE-GENERATING SYNTHETIC DATA BANK")
    print("=" * 60)
    print(f"  Bank directory: {bank_dir}")
    print(f"  Skills: {len(skills)}")
    print(f"  Samples per skill: {args.num}")
    tracker.log(
        "Phase 1: Pre-generating synthetic data bank",
        data={"bank_dir": str(bank_dir), "skills_count": len(skills), "num": args.num},
    )

    seed_gen = SeedGenerator(model=args.model)
    dual_gen = DualViewGenerator(model=args.model)

    # Load existing manifest for incremental updates (don't lose previous skills)
    manifest_path = bank_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["updated"] = datetime.now().isoformat()
        print(f"  Loaded existing manifest: {len(manifest.get('skills', {}))} skills")
    else:
        manifest = {
            "created": datetime.now().isoformat(),
            "model": args.model,
            "skills": {},
        }

    for skill_id in skills:
        skill = get_skill_bucket(skill_id)
        shard_path = bank_dir / f"{skill_id}.jsonl"

        print(f"\n[{skill_id}] Generating {args.num} samples...")

        if args.builtin_seeds:
            seeds = get_builtin_seeds(skill_id, args.num)
        else:
            seeds = seed_gen.generate(skill_id, args.num, args.difficulty)

        samples = []
        for i, seed in enumerate(seeds):
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(seeds)}]", end="\r")
            try:
                sample = dual_gen.generate(
                    question=seed["question"],
                    skill_bucket=skill_id,
                    band=skill.primary_band.value,
                    generate_hard_negative=args.hard_negatives,
                    generate_error_correction=args.error_correction,
                    sample_id=seed.get("id"),
                )
                samples.append(sample)
            except Exception as e:
                print(f"\n  [ERROR] {i}: {e}")
                tracker.log(
                    f"Sample generation failed: {e}",
                    level="ERROR",
                    data={"skill": skill_id, "sample_index": i},
                )

        print(f"  Generated {len(samples)} samples")
        tracker.log(f"Skill complete: {skill_id}", data={"samples": len(samples)})

        with open(shard_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
        tracker.record_output(shard_path)

        manifest["skills"][skill_id] = {
            "samples": len(samples),
            "band": skill.primary_band.value,
            "cot_allowed": cot_allowed_for_band(skill.primary_band),
            "with_hard_negatives": sum(1 for s in samples if s.hard_negative),
            "shard_file": str(shard_path.name),
            "priority": skill.priority,
            "last_run_id": tracker.run_id,
        }

    # Record last run in manifest
    manifest["last_run"] = {
        "run_id": tracker.run_id,
        "command": "generate-bank",
        "config_hash": tracker.config_hash,
        "timestamp": datetime.now().isoformat(),
    }

    with open(bank_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    tracker.record_output(bank_dir / "manifest.json")

    total = sum(m["samples"] for m in manifest["skills"].values())
    print(f"\n[Done] Created bank with {total} total samples")
    tracker.log("Bank generation complete", data={"total_samples": total})

    if args.glue_export:
        from integration.glue_exporter import export_bank_to_glue

        export_bank_to_glue(
            bank_dir=str(bank_dir),
            output_path=args.glue_output,
            model=args.model,
        )

    return manifest


def cmd_status(args):
    """Show data bank status."""
    from common.skills import resolve_skill_alias

    bank_dir = Path(args.bank_dir)
    manifest_path = bank_dir / "manifest.json"

    print("\n" + "=" * 60)
    print("  SYNTHETIC DATA BANK STATUS")
    print("=" * 60)

    if not manifest_path.exists():
        print(f"  Bank not found at: {bank_dir}")
        print("  Create with: python run_pipeline.py generate-bank")
        return

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"  Created: {manifest.get('created')}")
    print(f"  Model: {manifest.get('model')}")
    print("\n  Shards:")

    total = 0
    covered_canonical = set()

    for skill_id, info in manifest.get("skills", {}).items():
        total += info["samples"]
        cot = "Y" if info.get("cot_allowed") else "N"

        # Check if this is an alias
        canonical = resolve_skill_alias(skill_id)
        alias_marker = ""
        if canonical != skill_id:
            alias_marker = f" -> {canonical}"
            covered_canonical.add(canonical)
        elif skill_id in SKILL_BUCKETS:
            covered_canonical.add(skill_id)

        print(
            f"    {skill_id:20s} {info['samples']:4d} samples  COT:{cot}  {info['priority']}{alias_marker}"
        )

    print(f"\n  Total: {total} samples")

    # Calculate missing based on canonical skills
    missing = set(SKILL_BUCKETS.keys()) - covered_canonical
    if missing:
        print(f"\n  Missing: {len(missing)} skills (run with --verbose to list)")
        if args.verbose:
            for m in sorted(missing):
                print(f"    - {m}")


def cmd_rebuild_manifest(args):
    """Rebuild manifest from existing shards."""
    bank_dir = Path(args.bank_dir)

    print("\n" + "=" * 60)
    print("  REBUILDING MANIFEST FROM SHARDS")
    print("=" * 60)

    if not bank_dir.exists():
        print(f"  Bank not found at: {bank_dir}")
        return

    # Scan all JSONL files
    shards = list(bank_dir.glob("*.jsonl"))
    print(f"  Found {len(shards)} shard files")

    manifest = {
        "created": datetime.now().isoformat(),
        "model": args.model or "unknown",
        "skills": {},
    }

    for shard_path in shards:
        skill_id = shard_path.stem  # e.g., "RSN-ARITHMETIC" from "RSN-ARITHMETIC.jsonl"

        # Check if this is a known skill
        if skill_id not in SKILL_BUCKETS:
            print(f"  [SKIP] {skill_id} - not a known skill")
            continue

        skill = get_skill_bucket(skill_id)

        # Count samples
        with open(shard_path, encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]

        hard_neg_count = sum(1 for s in samples if s.get("hard_negative"))

        manifest["skills"][skill_id] = {
            "samples": len(samples),
            "band": skill.primary_band.value,
            "cot_allowed": cot_allowed_for_band(skill.primary_band),
            "with_hard_negatives": hard_neg_count,
            "shard_file": shard_path.name,
            "priority": skill.priority,
        }
        print(f"  [OK] {skill_id}: {len(samples)} samples")

    # Save manifest
    manifest_path = bank_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total = sum(m["samples"] for m in manifest["skills"].values())
    print(
        f"\n[Done] Rebuilt manifest: {len(manifest['skills'])} skills, {total} total samples"
    )
    return manifest


def cmd_seeds(args):
    """Generate seed questions."""
    from generation.seed_generator import SeedGenerator, get_builtin_seeds

    all_seeds = []

    if args.builtin:
        skills = list(SKILL_BUCKETS.keys()) if args.all else [args.skill]
        for skill_id in skills:
            seeds = get_builtin_seeds(skill_id, args.num)
            all_seeds.extend(seeds)
            print(f"[Builtin] {skill_id}: {len(seeds)} seeds")
    else:
        generator = SeedGenerator(model=args.model)
        if args.all:
            result = generator.generate_all(args.num, difficulty=args.difficulty)
            for seeds in result.values():
                all_seeds.extend(seeds)
        elif args.skill:
            all_seeds = generator.generate(args.skill, args.num, args.difficulty)

    output_path = args.output or f"seeds_{datetime.now():%Y%m%d_%H%M}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for seed in all_seeds:
            f.write(json.dumps(seed, ensure_ascii=False) + "\n")
    args._tracker.record_output(output_path)

    print(f"[Done] Saved {len(all_seeds)} seeds to: {output_path}")


def cmd_generate(args):
    """Generate for single skill."""
    from generation.dual_view_generator import DualViewGenerator
    from generation.seed_generator import get_builtin_seeds

    skill = get_skill_bucket(args.skill)

    if args.seeds:
        with open(args.seeds, encoding="utf-8") as f:
            seeds = [json.loads(line) for line in f if line.strip()][: args.num]
    else:
        seeds = get_builtin_seeds(args.skill, args.num)

    generator = DualViewGenerator(model=args.model)
    samples = []

    for i, seed in enumerate(seeds):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(seeds)}]", end="\r")
        try:
            sample = generator.generate(
                question=seed["question"],
                skill_bucket=args.skill,
                band=skill.primary_band.value,
                generate_hard_negative=args.hard_negatives,
                generate_error_correction=args.error_correction,
            )
            samples.append(sample)
        except Exception as e:
            print(f"  [ERROR] {i}: {e}")

    output_path = (
        args.output or f"synthetic_{args.skill}_{datetime.now():%Y%m%d_%H%M}.jsonl"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    args._tracker.record_output(output_path)

    print(f"\n[Done] Saved {len(samples)} samples to: {output_path}")


def cmd_diagnose(args):
    """Run diagnostic tests."""
    from diagnostics.diagnostic_tests import DIAGNOSTIC_TESTS, get_tests_for_skill
    from diagnostics.run_diagnostics import check_ollama, run_all_tests

    if not check_ollama():
        print("ERROR: Ollama not running")
        sys.exit(1)

    tests = DIAGNOSTIC_TESTS
    if args.skill:
        tests = get_tests_for_skill(args.skill)
    if args.band:
        tests = [t for t in tests if t.band == args.band]

    print("\n" + "=" * 60)
    print("  PHASE 2: DIAGNOSTIC TESTING")
    print("=" * 60)

    results = run_all_tests(model=args.model, tests=tests, verbose=not args.quiet)

    weak_skills = []
    for skill, stats in results.get("by_skill", {}).items():
        total = stats["passed"] + stats["failed"]
        rate = stats["passed"] / total if total > 0 else 0
        if rate < args.threshold:
            weak_skills.append({"skill": skill, "pass_rate": rate})

    weak_skills.sort(key=lambda x: x["pass_rate"])

    bank_dir = Path(args.bank_dir)
    manifest_path = bank_dir / "manifest.json"
    bank_available = {}
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            bank_available = json.load(f).get("skills", {})

    print("\n" + "=" * 60)
    print("  WEAKNESS ANALYSIS")
    print("=" * 60)
    print(f"  Threshold: {args.threshold*100:.0f}%")
    print(f"  Weak skills: {len(weak_skills)}")

    if weak_skills:
        print("\n  Recommended injections:")
        for w in weak_skills:
            avail = bank_available.get(w["skill"], {}).get("samples", 0)
            status = f"[OK] {avail} ready" if avail else "[MISSING] NOT IN BANK"
            print(f"    {w['skill']:20s} {w['pass_rate']*100:5.1f}% -> {status}")

    if args.output:
        results["weak_skills"] = weak_skills
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        args._tracker.record_output(args.output)
        print(f"\nSaved to: {args.output}")

    args._tracker.log(
        "Diagnosis complete",
        data={"weak_skills_count": len(weak_skills), "threshold": args.threshold},
    )
    return {"results": results, "weak_skills": weak_skills}


def cmd_inject(args):
    """Prepare injection from bank."""
    bank_dir = Path(args.bank_dir)
    manifest_path = bank_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"ERROR: Bank not found at {bank_dir}")
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    if args.skills:
        skills_to_inject = args.skills
    elif args.weakness_report:
        with open(args.weakness_report, encoding="utf-8") as f:
            report = json.load(f)
        skills_to_inject = [w["skill"] for w in report.get("weak_skills", [])]
    else:
        print("ERROR: Specify --skills or --weakness-report")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  INJECTION PREPARATION")
    print("=" * 60)
    print(f"  Skills: {skills_to_inject}")
    print(f"  Stage: {args.stage}")

    # Get allowed bands for the stage
    stage_enum = Stage[args.stage]
    allowed_bands = STAGE_CONFIGS[stage_enum].bands_allowed
    allowed_band_names = {b.name for b in allowed_bands}
    print(f"  Allowed bands for {args.stage}: {sorted(allowed_band_names)}")

    all_samples = []
    skipped_by_band = 0
    for skill_id in skills_to_inject:
        if skill_id not in manifest["skills"]:
            print(f"  [SKIP] {skill_id} not in bank")
            continue

        shard_path = bank_dir / manifest["skills"][skill_id]["shard_file"]
        with open(shard_path, encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]

        # Filter by allowed bands for stage
        original_count = len(samples)
        samples = [s for s in samples if s.get("band", "B3") in allowed_band_names]
        if len(samples) < original_count:
            skipped = original_count - len(samples)
            skipped_by_band += skipped
            print(
                f"  [FILTER] {skill_id}: {skipped} samples excluded (band not in {args.stage})"
            )

        max_samples = int(len(samples) * args.max_pct / 100)
        samples = samples[:max_samples]
        all_samples.extend(samples)
        print(f"  [OK] {skill_id}: {len(samples)} samples")

    if skipped_by_band > 0:
        print(f"\n  [INFO] Total skipped by band restriction: {skipped_by_band}")

    # Format for training
    formatted = []
    for s in all_samples:
        cot = cot_allowed_for_band(s.get("band", "B3"))
        use_cot = cot and args.stage in ("SFT", "DPO")

        item = {
            "id": s.get("id"),
            "instruction": s.get("question"),
            "output": (
                s.get("think_view")
                if use_cot and s.get("think_view")
                else s.get("distilled_view")
            ),
            "skill_bucket": s.get("skill_bucket"),
            "band": s.get("band"),
            "stage": args.stage,
        }

        if args.stage == "DPO" and s.get("hard_negative"):
            item["rejected"] = s["hard_negative"].get("reasoning", "")

        formatted.append(item)

    output_path = (
        args.output or f"injection_{args.stage}_{datetime.now():%Y%m%d_%H%M}.jsonl"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        for item in formatted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    args._tracker.record_output(output_path)

    # Summary with band distribution and injection caps
    band_counts = {}
    for item in formatted:
        b = item.get("band", "unknown")
        band_counts[b] = band_counts.get(b, 0) + 1

    print(f"\n[Done] {len(formatted)} samples -> {output_path}")
    print("\n  Band Distribution & Injection Caps:")
    for band_name in sorted(band_counts.keys()):
        count = band_counts[band_name]
        try:
            band_enum = Band[band_name]
            cap_pct = get_injection_cap(stage_enum, band_enum) * 100
            print(
                f"    {band_name}: {count} samples (cap: {cap_pct:.0f}% of training mix)"
            )
        except (KeyError, ValueError):
            print(f"    {band_name}: {count} samples")


def cmd_validate(args):
    """Run proxy validation."""
    from validation.proxy_validation import ValidationConfig, run_validation

    config = ValidationConfig(
        proxy_model=args.model,
        synthetic_data_path=args.synthetic_data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        lora_r=args.lora_r,
    )

    run_validation(config, baseline_ollama_model=args.baseline_model)


def cmd_check_contamination(args):
    """Check data for benchmark contamination."""
    from validation.contamination import run_contamination_check

    benchmarks = args.benchmarks if args.benchmarks else ["gsm8k", "math", "mmlu"]
    report = run_contamination_check(
        data_path=args.data,
        benchmarks=benchmarks,
        output_path=args.output if args.filter else None,
        filter_contaminated=args.filter,
    )

    # Save report
    if args.report:
        import json

        with open(args.report, "w") as f:
            json.dump(
                {
                    "total_samples": report.total_samples,
                    "contaminated_samples": report.contaminated_samples,
                    "contamination_rate": report.contamination_rate,
                    "exact_matches": report.exact_matches,
                    "high_similarity_matches": report.high_similarity_matches,
                    "partial_matches": report.partial_matches,
                    "by_benchmark": report.by_benchmark,
                    "flagged_samples": report.flagged_samples,
                },
                f,
                indent=2,
            )
        print(f"[Report] {args.report}")


def cmd_verify(args):
    """Run verification pipeline on samples."""
    from validation.verification import run_verification

    run_verification(
        data_path=args.data,
        output_path=args.output,
        teacher_model=args.teacher,
        student_model=args.student,
        update_samples=True,
    )


def cmd_import_synth(args):
    """Import samples from PleIAs/SYNTH dataset."""
    from integration.synth_adapter import convert_synth_to_bank

    count = convert_synth_to_bank(
        num_samples=args.num,
        output_dir=args.bank_dir,
        language=args.language,
        skills=args.skills,
        bands=args.bands,
    )
    print(f"\n[Done] Imported {count} samples from SYNTH dataset")


def cmd_export_glue(args):
    """Export synth_data_bank to Glue normalized Parquet format."""
    from integration.glue_exporter import export_bank_to_glue

    summary = export_bank_to_glue(
        bank_dir=args.bank_dir,
        output_path=args.output,
        skills=args.skills,
        languages=args.languages,
        model=args.model,
    )

    if summary.get("by_source"):
        print("\n  Source breakdown:")
        for source, count in summary["by_source"].items():
            print(f"    {source}: {count} records")

    # Hash all output parquet files
    output_dir = Path(args.output)
    if output_dir.exists():
        for pq_file in output_dir.rglob("*.parquet"):
            args._tracker.record_output(pq_file)

    print(f"\n[Done] Exported {summary.get('exported', 0)} records to Glue format")


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WORKFLOW:
  Phase 1 (Before Training):
    python run_pipeline.py generate-bank --all
    python run_pipeline.py status
  
  Phase 2 (During Training):
    python run_pipeline.py diagnose --model qwen3:4b
    python run_pipeline.py inject --skills RSN-ARITHMETIC
    python run_pipeline.py validate injection.jsonl
        """,
    )

    sub = parser.add_subparsers(dest="command")

    # generate-bank
    p = sub.add_parser("generate-bank", help="Pre-generate data bank (Phase 1)")
    p.add_argument("--all", action="store_true")
    p.add_argument("--skills", nargs="*")
    p.add_argument("--num", "-n", type=int, default=30)
    p.add_argument("--model", "-m", default="qwen3:8b")
    p.add_argument("--bank-dir", default="./synth_data_bank")
    p.add_argument(
        "--difficulty", default="mixed", choices=["easy", "medium", "hard", "mixed"]
    )
    p.add_argument("--builtin-seeds", action="store_true")
    p.add_argument("--no-hard-negatives", dest="hard_negatives", action="store_false")
    p.add_argument(
        "--no-error-correction", dest="error_correction", action="store_false"
    )
    p.add_argument(
        "--glue-export",
        action="store_true",
        help="Also export to Glue normalized Parquet",
    )
    p.add_argument(
        "--glue-output",
        default="./normalized_output",
        help="Glue export output path",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducibility (Ollama + Python)",
    )

    # status
    p = sub.add_parser("status", help="Show bank status")
    p.add_argument("--bank-dir", default="./synth_data_bank")
    p.add_argument("--verbose", "-v", action="store_true", help="List missing skills")

    # rebuild-manifest
    p = sub.add_parser("rebuild-manifest", help="Rebuild manifest from existing shards")
    p.add_argument("--bank-dir", default="./synth_data_bank")
    p.add_argument(
        "--model", "-m", default=None, help="Model name to record in manifest"
    )

    # seeds
    p = sub.add_parser("seeds", help="Generate seeds only")
    p.add_argument("--skill", "-s")
    p.add_argument("--all", action="store_true")
    p.add_argument("--num", "-n", type=int, default=20)
    p.add_argument("--model", "-m", default="qwen3:8b")
    p.add_argument("--difficulty", default="mixed")
    p.add_argument("--output", "-o")
    p.add_argument("--builtin", action="store_true")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducibility (Ollama + Python)",
    )

    # generate
    p = sub.add_parser("generate", help="Generate for single skill")
    p.add_argument("--skill", "-s", required=True)
    p.add_argument("--num", "-n", type=int, default=10)
    p.add_argument("--model", "-m", default="qwen3:8b")
    p.add_argument("--seeds")
    p.add_argument("--output", "-o")
    p.add_argument("--no-hard-negatives", dest="hard_negatives", action="store_false")
    p.add_argument(
        "--no-error-correction", dest="error_correction", action="store_false"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducibility (Ollama + Python)",
    )

    # diagnose
    p = sub.add_parser("diagnose", help="Run diagnostics (Phase 2)")
    p.add_argument("--model", "-m", default="qwen3:4b")
    p.add_argument("--skill", "-s")
    p.add_argument("--band", "-b")
    p.add_argument("--threshold", "-t", type=float, default=0.7)
    p.add_argument("--bank-dir", default="./synth_data_bank")
    p.add_argument("--output", "-o")
    p.add_argument("--quiet", "-q", action="store_true")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducibility (Ollama + Python)",
    )

    # inject
    p = sub.add_parser("inject", help="Prepare injection from bank")
    p.add_argument("--skills", nargs="*")
    p.add_argument("--weakness-report")
    p.add_argument("--stage", default="SFT", choices=["PRE", "SFT", "DPO", "RLHF"])
    p.add_argument("--max-pct", type=float, default=100.0)
    p.add_argument("--bank-dir", default="./synth_data_bank")
    p.add_argument("--output", "-o")

    # validate
    p = sub.add_parser("validate", help="Validate with proxy model")
    p.add_argument("synthetic_data")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--output-dir", default="./validation_output")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--baseline-model", default="qwen2.5:1.5b")

    # check-contamination
    p = sub.add_parser(
        "check-contamination", help="Check data for benchmark contamination"
    )
    p.add_argument("data", help="Path to JSONL data file")
    p.add_argument(
        "--benchmarks",
        nargs="*",
        help="Benchmarks to check (default: gsm8k, math, mmlu)",
    )
    p.add_argument(
        "--filter", action="store_true", help="Filter out contaminated samples"
    )
    p.add_argument("--output", "-o", help="Output path for filtered data")
    p.add_argument("--report", "-r", help="Path to save contamination report JSON")

    # verify
    p = sub.add_parser("verify", help="Run verification pipeline on samples")
    p.add_argument("data", help="Path to JSONL data file")
    p.add_argument(
        "--teacher", "-t", default="qwen3:8b", help="Teacher model for verification"
    )
    p.add_argument(
        "--student", "-s", default="qwen3:4b", help="Student model for re-solve"
    )
    p.add_argument("--output", "-o", help="Output path for verified data")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducibility (Ollama + Python)",
    )

    # import-synth
    p = sub.add_parser("import-synth", help="Import samples from PleIAs/SYNTH dataset")
    p.add_argument(
        "--num", "-n", type=int, default=1000, help="Number of samples to import"
    )
    p.add_argument("--bank-dir", default="./synth_data_bank", help="Output directory")
    p.add_argument("--language", "-l", default="en", help="Language filter")
    p.add_argument("--skills", nargs="*", help="Filter by skill buckets")
    p.add_argument("--bands", nargs="*", help="Filter by bands")

    # export-glue
    p = sub.add_parser(
        "export-glue", help="Export bank to Glue normalized Parquet format"
    )
    p.add_argument("--bank-dir", default="./synth_data_bank", help="Bank directory")
    p.add_argument("--output", "-o", default="./normalized_output", help="Output path")
    p.add_argument("--skills", nargs="*", help="Filter by skill IDs")
    p.add_argument("--languages", nargs="*", help="Filter by language codes")
    p.add_argument("--model", "-m", help="Model name override (default: from manifest)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize run tracker for reproducibility
    from common.ollama_client import set_ollama_seed  # noqa: E402
    from common.run_tracker import RunTracker  # noqa: E402

    tracker = RunTracker(command_name=args.command, args=args)
    args._tracker = tracker

    # Set seeds if provided
    seed = getattr(args, "seed", None)
    if seed is not None:
        set_ollama_seed(seed)
        random.seed(seed)

    cmds = {
        "generate-bank": cmd_generate_bank,
        "status": cmd_status,
        "rebuild-manifest": cmd_rebuild_manifest,
        "seeds": cmd_seeds,
        "generate": cmd_generate,
        "diagnose": cmd_diagnose,
        "inject": cmd_inject,
        "validate": cmd_validate,
        "check-contamination": cmd_check_contamination,
        "verify": cmd_verify,
        "import-synth": cmd_import_synth,
        "export-glue": cmd_export_glue,
    }

    try:
        cmds[args.command](args)
        tracker.finish(status="success")
    except KeyboardInterrupt:
        tracker.log("Run interrupted by user", level="WARN")
        tracker.finish(status="interrupted")
        sys.exit(130)
    except SystemExit:
        tracker.finish(status="error")
        raise
    except Exception as e:
        tracker.log(f"Run failed: {e}", level="ERROR")
        tracker.finish(status="error")
        raise


if __name__ == "__main__":
    main()
