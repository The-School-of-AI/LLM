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
import logging
import os
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
from common.logging_config import setup_logging, get_logger

logger = get_logger("run_pipeline")

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
        skills = [s for s, spec in SKILL_BUCKETS.items() if spec.priority in ("critical", "high")]
    
    logger.info("=" * 60)
    logger.info("  PHASE 1: PRE-GENERATING SYNTHETIC DATA BANK")
    logger.info("=" * 60)
    logger.info("  Bank directory: %s", bank_dir.resolve())
    logger.info("  Model: %s", args.model)
    logger.info("  Skills to generate: %d", len(skills))
    logger.info("  Samples per skill: %d", args.num)
    logger.info("  Builtin seeds: %s | Hard negatives: %s | Error correction: %s",
                args.builtin_seeds, args.hard_negatives, args.error_correction)
    logger.debug("  Skill list: %s", skills)
    
    seed_gen = SeedGenerator(model=args.model)
    dual_gen = DualViewGenerator(model=args.model)

    # Load existing manifest for incremental updates (don't lose previous skills)
    manifest_path = bank_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["updated"] = datetime.now().isoformat()
        logger.info("  Loaded existing manifest: %d skills", len(manifest.get('skills', {})))
    else:
        manifest = {"created": datetime.now().isoformat(), "model": args.model, "skills": {}}
        logger.info("  Creating new manifest")
    
    for skill_idx, raw_skill_id in enumerate(skills):
        skill = get_skill_bucket(raw_skill_id)
        skill_id = skill.id  # canonical form
        shard_path = bank_dir / f"{skill_id}.jsonl"
        
        if raw_skill_id != skill_id:
            logger.info("[%d/%d] %s -> %s | Generating %d samples (band=%s, priority=%s)",
                        skill_idx + 1, len(skills), raw_skill_id, skill_id, args.num,
                        skill.primary_band.value, skill.priority)
        else:
            logger.info("[%d/%d] %s | Generating %d samples (band=%s, priority=%s)",
                        skill_idx + 1, len(skills), skill_id, args.num,
                        skill.primary_band.value, skill.priority)
        
        if args.builtin_seeds:
            seeds = get_builtin_seeds(skill_id, args.num)
            logger.debug("  Using %d builtin seeds for %s", len(seeds), skill_id)
        else:
            seeds = seed_gen.generate(skill_id, args.num, args.difficulty)
            logger.debug("  Generated %d LLM seeds for %s (difficulty=%s)", len(seeds), skill_id, args.difficulty)
        
        samples = []
        for i, seed in enumerate(seeds):
            seed_language = seed.get("language", skill.languages[0] if skill.languages else "en")
            logger.debug("  [%d/%d] Seed: %s (lang=%s)", i + 1, len(seeds), seed["question"][:60], seed_language)
            try:
                sample = dual_gen.generate(
                    question=seed["question"],
                    skill_bucket=skill_id,
                    band=skill.primary_band.value,
                    generate_hard_negative=args.hard_negatives,
                    generate_error_correction=args.error_correction,
                    sample_id=seed.get("id"),
                    language=seed_language,
                )
                samples.append(sample)
                logger.debug("    -> answer=%s | dist=%d chars | think=%s chars | hn=%s | ec=%s",
                             repr(sample.answer[:50]),
                             len(sample.distilled_view),
                             len(sample.think_view) if sample.think_view else 0,
                             "yes" if sample.hard_negative else "no",
                             "yes" if sample.error_correction else "no")
            except Exception as e:
                logger.error("  [%d/%d] FAILED: %s", i + 1, len(seeds), e, exc_info=True)
        
        logger.info("  -> Generated %d/%d samples for %s", len(samples), len(seeds), skill_id)
        
        with open(shard_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
        logger.debug("  Saved shard: %s", shard_path)
        
        manifest["skills"][skill_id] = {
            "samples": len(samples),
            "band": skill.primary_band.value,
            "cot_allowed": cot_allowed_for_band(skill.primary_band),
            "with_hard_negatives": sum(1 for s in samples if s.hard_negative),
            "shard_file": str(shard_path.name),
            "priority": skill.priority,
        }

    with open(bank_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total = sum(m["samples"] for m in manifest["skills"].values())
    logger.info("=" * 60)
    logger.info("  DONE: Bank created with %d total samples across %d skills", total, len(manifest["skills"]))
    logger.info("=" * 60)
    return manifest


def cmd_status(args):
    """Show data bank status."""
    from common.skills import resolve_skill_alias

    bank_dir = Path(args.bank_dir)
    manifest_path = bank_dir / "manifest.json"

    logger.info("=" * 60)
    logger.info("  SYNTHETIC DATA BANK STATUS")
    logger.info("=" * 60)

    if not manifest_path.exists():
        logger.warning("  Bank not found at: %s", bank_dir)
        logger.info("  Create with: python run_pipeline.py generate-bank")
        return

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    logger.info("  Created: %s", manifest.get('created'))
    logger.info("  Model: %s", manifest.get('model'))
    logger.info("  Shards:")

    total = 0
    covered_canonical = set()

    for skill_id, info in manifest.get("skills", {}).items():
        total += info["samples"]
        cot = "Y" if info.get("cot_allowed") else "N"

        canonical = resolve_skill_alias(skill_id)
        alias_marker = ""
        if canonical != skill_id:
            alias_marker = f" -> {canonical}"
            covered_canonical.add(canonical)
        elif skill_id in SKILL_BUCKETS:
            covered_canonical.add(skill_id)

        logger.info("    %s %4d samples  COT:%s  %s%s", skill_id.ljust(20), info['samples'], cot, info['priority'], alias_marker)

    logger.info("  Total: %d samples", total)

    missing = set(SKILL_BUCKETS.keys()) - covered_canonical
    if missing:
        logger.info("  Missing: %d skills (run with --verbose to list)", len(missing))
        if args.verbose:
            for m in sorted(missing):
                logger.info("    - %s", m)


def cmd_rebuild_manifest(args):
    """Rebuild manifest from existing shards."""
    bank_dir = Path(args.bank_dir)

    logger.info("=" * 60)
    logger.info("  REBUILDING MANIFEST FROM SHARDS")
    logger.info("=" * 60)

    if not bank_dir.exists():
        logger.error("  Bank not found at: %s", bank_dir)
        return

    shards = list(bank_dir.glob("*.jsonl"))
    logger.info("  Found %d shard files", len(shards))

    manifest = {
        "created": datetime.now().isoformat(),
        "model": args.model or "unknown",
        "skills": {},
    }

    for shard_path in shards:
        raw_name = shard_path.stem  # e.g., "RSN-ARITHMETIC" from "RSN-ARITHMETIC.jsonl"

        # OLD: only checked raw_name against SKILL_BUCKETS (canonical keys)
        #      — legacy-named files like RSN-ARITHMETIC.jsonl were SKIPPED
        # NEW: try to resolve the filename as a skill ID (canonical or alias)
        try:
            skill = get_skill_bucket(raw_name)  # handles aliases via SKILL_ALIASES
            skill_id = skill.id  # canonical form
        except ValueError:
            # Not a known skill or alias (e.g., "KNOW-FACTUAL_synth")
            logger.warning("  [SKIP] %s - not a known skill or alias", raw_name)
            continue

        if raw_name != skill_id:
            logger.info("  [ALIAS] %s -> %s", raw_name, skill_id)

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
        logger.info("  [OK] %s: %d samples", skill_id, len(samples))

    manifest_path = bank_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total = sum(m["samples"] for m in manifest["skills"].values())
    logger.info("  DONE: Rebuilt manifest: %d skills, %d total samples", len(manifest['skills']), total)
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
            logger.info("[Builtin] %s: %d seeds", skill_id, len(seeds))
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
    
    logger.info("Saved %d seeds to: %s", len(all_seeds), output_path)


def cmd_generate(args):
    """Generate for single skill."""
    from generation.dual_view_generator import DualViewGenerator
    from generation.seed_generator import get_builtin_seeds

    skill = get_skill_bucket(args.skill)
    logger.info("Generating for skill: %s (band=%s, model=%s)", args.skill, skill.primary_band.value, args.model)
    
    if args.seeds:
        with open(args.seeds, encoding="utf-8") as f:
            seeds = [json.loads(l) for l in f if l.strip()][:args.num]
        logger.info("Loaded %d seeds from file: %s", len(seeds), args.seeds)
    else:
        seeds = get_builtin_seeds(args.skill, args.num)
        logger.info("Using %d builtin seeds", len(seeds))
    
    generator = DualViewGenerator(model=args.model)
    samples = []

    for i, seed in enumerate(seeds):
        seed_language = seed.get("language", skill.languages[0] if skill.languages else "en")
        logger.debug("[%d/%d] Processing seed: %s (lang=%s)", i + 1, len(seeds), seed["question"][:60], seed_language)
        try:
            sample = generator.generate(
                question=seed["question"],
                skill_bucket=args.skill,
                band=skill.primary_band.value,
                generate_hard_negative=args.hard_negatives,
                generate_error_correction=args.error_correction,
                language=seed_language,
            )
            samples.append(sample)
            logger.debug("  -> OK: answer=%s", repr(sample.answer[:50]))
        except Exception as e:
            logger.error("[%d/%d] FAILED: %s", i + 1, len(seeds), e, exc_info=True)
    
    output_path = args.output or f"synthetic_{args.skill}_{datetime.now():%Y%m%d_%H%M}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    
    logger.info("Saved %d samples to: %s", len(samples), output_path)


def cmd_diagnose(args):
    """Run diagnostic tests."""
    from diagnostics.diagnostic_tests import DIAGNOSTIC_TESTS, get_tests_for_skill
    from diagnostics.run_diagnostics import check_ollama, run_all_tests

    if not check_ollama():
        logger.error("Ollama is not running! Start with: ollama serve")
        sys.exit(1)

    tests = DIAGNOSTIC_TESTS
    if args.skill:
        tests = get_tests_for_skill(args.skill)
    if args.band:
        tests = [t for t in tests if t.band == args.band]
    
    logger.info("=" * 60)
    logger.info("  PHASE 2: DIAGNOSTIC TESTING")
    logger.info("=" * 60)
    logger.info("  Model: %s | Tests: %d | Threshold: %.0f%%", args.model, len(tests), args.threshold * 100)
    
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
    
    logger.info("=" * 60)
    logger.info("  WEAKNESS ANALYSIS")
    logger.info("=" * 60)
    logger.info("  Threshold: %.0f%% | Weak skills: %d", args.threshold * 100, len(weak_skills))
    
    if weak_skills:
        logger.info("  Recommended injections:")
        for w in weak_skills:
            avail = bank_available.get(w["skill"], {}).get("samples", 0)
            status = f"[OK] {avail} ready" if avail else "[MISSING] NOT IN BANK"
            logger.info("    %s %5.1f%% -> %s", w['skill'].ljust(20), w['pass_rate'] * 100, status)
    
    if args.output:
        results["weak_skills"] = weak_skills
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved diagnostics to: %s", args.output)
    
    return {"results": results, "weak_skills": weak_skills}


def cmd_inject(args):
    """Prepare injection from bank."""
    bank_dir = Path(args.bank_dir)
    manifest_path = bank_dir / "manifest.json"

    if not manifest_path.exists():
        logger.error("Bank not found at %s", bank_dir)
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
        logger.error("Specify --skills or --weakness-report")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("  INJECTION PREPARATION")
    logger.info("=" * 60)
    logger.info("  Skills: %s", skills_to_inject)
    logger.info("  Stage: %s", args.stage)
    
    stage_enum = Stage[args.stage]
    allowed_bands = STAGE_CONFIGS[stage_enum].bands_allowed
    allowed_band_names = {b.name for b in allowed_bands}
    logger.info("  Allowed bands for %s: %s", args.stage, sorted(allowed_band_names))

    all_samples = []
    skipped_by_band = 0
    for raw_skill in skills_to_inject:
        # OLD: used raw skill_id directly against manifest — missed if manifest
        #      has legacy keys and user passes canonical, or vice versa
        # NEW: try both the raw key and its canonical/legacy variants
        skill_id = raw_skill
        if skill_id not in manifest["skills"]:
            # Try resolving as alias → canonical
            try:
                canonical = get_skill_bucket(skill_id).id
                if canonical in manifest["skills"]:
                    skill_id = canonical
            except ValueError:
                pass
        if skill_id not in manifest["skills"]:
            # Try reverse: check if any legacy alias of this canonical is in manifest
            from common.skills import SKILL_ALIASES
            for legacy, canon in SKILL_ALIASES.items():
                if canon == raw_skill and legacy in manifest["skills"]:
                    skill_id = legacy
                    break
        if skill_id not in manifest["skills"]:
            logger.warning("  [SKIP] %s not in bank", raw_skill)
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
            logger.info("  [FILTER] %s: %d samples excluded (band not in %s)", skill_id, skipped, args.stage)

        max_samples = int(len(samples) * args.max_pct / 100)
        samples = samples[:max_samples]
        all_samples.extend(samples)
        logger.info("  [OK] %s: %d samples", skill_id, len(samples))

    if skipped_by_band > 0:
        logger.info("  Total skipped by band restriction: %d", skipped_by_band)
    
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

    # Summary with band distribution and injection caps
    band_counts = {}
    for item in formatted:
        b = item.get("band", "unknown")
        band_counts[b] = band_counts.get(b, 0) + 1

    logger.info("DONE: %d samples -> %s", len(formatted), output_path)
    logger.info("  Band Distribution & Injection Caps:")
    for band_name in sorted(band_counts.keys()):
        count = band_counts[band_name]
        try:
            band_enum = Band[band_name]
            cap_pct = get_injection_cap(stage_enum, band_enum) * 100
            logger.info("    %s: %d samples (cap: %.0f%% of training mix)", band_name, count, cap_pct)
        except (KeyError, ValueError):
            logger.info("    %s: %d samples", band_name, count)


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

LOGGING:
  --log-level DEBUG     Show all debug messages on console
  --log-level INFO      Default: info + warnings + errors
  --log-dir ./my_logs   Custom log directory (default: ./logs/)
  Logs always written to: logs/<command>_<datetime>.log
        """,
    )
    
    # Global logging args
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Console log level (default: INFO). File always captures DEBUG.")
    parser.add_argument("--log-dir", default=None,
                        help="Directory for log files (default: ./logs/)")
    
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

    # diagnose
    p = sub.add_parser("diagnose", help="Run diagnostics (Phase 2)")
    p.add_argument("--model", "-m", default="qwen3:4b")
    p.add_argument("--skill", "-s")
    p.add_argument("--band", "-b")
    p.add_argument("--threshold", "-t", type=float, default=0.7)
    p.add_argument("--bank-dir", default="./synth_data_bank")
    p.add_argument("--output", "-o")
    p.add_argument("--quiet", "-q", action="store_true")

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

    # import-synth
    p = sub.add_parser("import-synth", help="Import samples from PleIAs/SYNTH dataset")
    p.add_argument(
        "--num", "-n", type=int, default=1000, help="Number of samples to import"
    )
    p.add_argument("--bank-dir", default="./synth_data_bank", help="Output directory")
    p.add_argument("--language", "-l", default="en", help="Language filter")
    p.add_argument("--skills", nargs="*", help="Filter by skill buckets")
    p.add_argument("--bands", nargs="*", help="Filter by bands")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize logging BEFORE running any command
    log_path = setup_logging(
        command=args.command,
        console_level=args.log_level,
        log_dir=args.log_dir,
    )
    logger.info("Log file: %s", log_path)
    
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
    }
    
    try:
        cmds[args.command](args)
    except Exception as e:
        logger.critical("Pipeline failed with error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
