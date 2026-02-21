#!/usr/bin/env python3
"""
Phase 1 — Script 1: Build and FREEZE the reduced MMLU subset.

Combines English MMLU (cais/mmlu) with optional Indian language MMLU
(sarvamai/mmlu-indic) into a single frozen evaluation file.

Usage:
    # English only (default)
    python scripts/build_mmlu_subset.py

    # With specific Indian languages
    python scripts/build_mmlu_subset.py --indic-langs hi ta bn

    # With all 10 Indian languages
    python scripts/build_mmlu_subset.py --indic-langs hi bn ta te mr gu kn ml pa or

    # Force rebuild even if file exists
    python scripts/build_mmlu_subset.py --force

Once generated, the subset file must NOT be modified mid-training so that
checkpoint-to-checkpoint trends remain comparable.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.config import load_config, set_seed


MMLU_SPLITS = ["validation", "test"]   # try validation first, then test

# sarvamai/mmlu-indic language codes and their display names
INDIC_LANG_NAMES = {
    "hi":        "Hindi",
    "bn":        "Bengali",
    "ta":        "Tamil",
    "te":        "Telugu",
    "mr":        "Marathi",
    "gu":        "Gujarati",
    "kn":        "Kannada",
    "ml":        "Malayalam",
    "pa":        "Punjabi",
    "or":        "Odia",
    "hi_roman":  "Hindi (Romanized)",
    "bn_roman":  "Bengali (Romanized)",
    "ta_roman":  "Tamil (Romanized)",
    "te_roman":  "Telugu (Romanized)",
    "mr_roman":  "Marathi (Romanized)",
    "gu_roman":  "Gujarati (Romanized)",
    "kn_roman":  "Kannada (Romanized)",
    "ml_roman":  "Malayalam (Romanized)",
    "pa_roman":  "Punjabi (Romanized)",
    "or_roman":  "Odia (Romanized)",
}


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    parser = argparse.ArgumentParser(
        description="Build frozen MMLU + Indic MMLU subset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_mmlu_subset.py                            # English only
  python scripts/build_mmlu_subset.py --indic-langs hi ta        # + Hindi + Tamil
  python scripts/build_mmlu_subset.py --indic-langs hi bn ta te mr gu kn ml pa or
  python scripts/build_mmlu_subset.py --force                    # rebuild even if exists
        """
    )
    parser.add_argument(
        "--n",
        type=int,
        default=cfg["mmlu"]["questions_per_category"],
        help="Questions per category (default from config)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / cfg["mmlu"]["subset_file"],
        help="Output JSON path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg["seed"],
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Specific English MMLU categories (default: all from config)",
    )
    parser.add_argument(
        "--indic-langs",
        nargs="*",
        default=None,
        metavar="LANG",
        help=(
            "Indian language codes to include from sarvamai/mmlu-indic. "
            "Supported: " + ", ".join(INDIC_LANG_NAMES.keys())
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if output file already exists",
    )
    return parser.parse_args()


# ── English MMLU loader ────────────────────────────────────────────────────

def load_mmlu_category(category: str, domain: str = "general_knowledge") -> list[dict]:
    """Load a single English MMLU category from cais/mmlu."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` not installed. Run: pip install datasets")
        sys.exit(1)

    samples = []
    for split in MMLU_SPLITS:
        try:
            ds = load_dataset("cais/mmlu", category, split=split, trust_remote_code=True)
            for row in ds:
                samples.append({
                    "category": category,
                    "domain": domain,
                    "language": "en",
                    "language_name": "English",
                    "source_dataset": "cais/mmlu",
                    "split": split,
                    "question": row["question"],
                    "choices": list(row["choices"]),
                    "answer": int(row["answer"]),
                })
        except Exception as exc:
            print(f"  [WARN] Could not load cais/mmlu/{category}/{split}: {exc}")
    return samples


# ── Indic MMLU loader ──────────────────────────────────────────────────────

def load_indic_mmlu_language(lang_code: str, indic_dataset: str,
                              english_categories: list[str]) -> list[dict]:
    """
    Load questions for one Indian language from sarvamai/mmlu-indic.

    The Indic dataset contains the same 57 MMLU categories translated into each
    language. We load all categories and match them to English category names.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` not installed. Run: pip install datasets")
        sys.exit(1)

    lang_name = INDIC_LANG_NAMES.get(lang_code, lang_code)
    print(f"\n  [{lang_code}] {lang_name} — loading from {indic_dataset} ...")

    samples = []
    for split in MMLU_SPLITS:
        try:
            ds = load_dataset(indic_dataset, lang_code, split=split, trust_remote_code=True)
            loaded = 0
            for row in ds:
                # sarvamai/mmlu-indic uses 'subject' field for the category
                category = row.get("subject", row.get("category", "unknown"))
                # Normalise: lowercase, underscores
                category_norm = category.lower().replace(" ", "_").replace("-", "_")

                # Match against our known English category list
                matched_cat = category_norm
                for known in english_categories:
                    if category_norm == known or known in category_norm or category_norm in known:
                        matched_cat = known
                        break

                choices = row.get("choices", row.get("options", []))
                answer_raw = row.get("answer", row.get("label", 0))
                # answer can be 0-3 int or 'A'/'B'/'C'/'D' string
                if isinstance(answer_raw, str) and answer_raw.upper() in "ABCD":
                    answer_idx = "ABCD".index(answer_raw.upper())
                else:
                    try:
                        answer_idx = int(answer_raw)
                    except (ValueError, TypeError):
                        answer_idx = 0

                samples.append({
                    "category": matched_cat,
                    "domain": "general_knowledge",  # filled in by caller after domain_map lookup
                    "language": lang_code,
                    "language_name": lang_name,
                    "source_dataset": indic_dataset,
                    "split": split,
                    "question": row["question"],
                    "choices": list(choices),
                    "answer": answer_idx,
                })
                loaded += 1

            print(f"    {split}: {loaded} questions loaded")
            if loaded > 0:
                break  # got data from this split, skip the rest
        except Exception as exc:
            print(f"    [WARN] {split}: {exc}")

    return samples


# ── Sampling ───────────────────────────────────────────────────────────────

def stratified_sample(samples: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Return up to n items, balanced across answer labels (0-3 = A/B/C/D)."""
    if len(samples) <= n:
        return list(samples)

    by_label: dict[int, list[dict]] = {}
    for s in samples:
        by_label.setdefault(s["answer"], []).append(s)

    result: list[dict] = []
    per_label = max(1, n // len(by_label))
    for label, group in by_label.items():
        rng.shuffle(group)
        result.extend(group[:per_label])

    remaining = [s for s in samples if s not in result]
    rng.shuffle(remaining)
    result.extend(remaining[: n - len(result)])

    rng.shuffle(result)
    return result[:n]


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = load_config()

    # Guard: don't rebuild silently if already frozen
    if args.out.exists() and not args.force:
        size_kb = args.out.stat().st_size // 1024
        print(f"MMLU subset already exists ({size_kb} KB): {args.out}")
        print("Use --force to rebuild (warning: this will break trend comparability!).")
        return

    set_seed(args.seed)
    rng = random.Random(args.seed)

    # Resolve English categories
    english_categories: list[str] = args.categories or cfg["mmlu"]["categories"]

    # Resolve Indic languages: CLI flag > config file > empty
    if args.indic_langs is not None:
        indic_langs = args.indic_langs
    elif cfg["mmlu"].get("indic_languages"):
        indic_langs = list(cfg["mmlu"]["indic_languages"])
    else:
        indic_langs = []

    # Validate Indic codes
    invalid = [l for l in indic_langs if l not in INDIC_LANG_NAMES]
    if invalid:
        print(f"WARNING: Unknown Indic language codes ignored: {invalid}")
        print(f"  Valid codes: {list(INDIC_LANG_NAMES.keys())}")
        indic_langs = [l for l in indic_langs if l in INDIC_LANG_NAMES]

    indic_dataset = cfg["mmlu"].get("indic_dataset", "sarvamai/mmlu-indic")

    print("=" * 65)
    print("  Building MMLU Subset  (will be FROZEN after this run)")
    print("=" * 65)
    print(f"  Seed                 : {args.seed}")
    print(f"  Questions/category   : {args.n}")
    print(f"  English categories   : {len(english_categories)}")
    if indic_langs:
        names = [INDIC_LANG_NAMES[l] for l in indic_langs]
        print(f"  Indian languages     : {', '.join(names)}")
    else:
        print(f"  Indian languages     : none (English only)")
    print(f"  Output               : {args.out}")
    print("=" * 65)

    # Load domain map from config (category → domain label)
    domain_map: dict[str, str] = cfg["mmlu"].get("domain_map", {})

    subset: list[dict] = []
    failed_en: list[str] = []
    failed_indic: list[str] = []

    # ── English MMLU ───────────────────────────────────────────────────────
    print(f"\n[English] Loading {len(english_categories)} categories from cais/mmlu ...")
    for i, cat in enumerate(english_categories, 1):
        print(f"  [{i:02d}/{len(english_categories)}] {cat} ...", end=" ", flush=True)
        domain = domain_map.get(cat, "general_knowledge")
        samples = load_mmlu_category(cat, domain)
        if not samples:
            failed_en.append(cat)
            print("SKIPPED (no data)")
            continue
        chosen = stratified_sample(samples, args.n, rng)
        subset.extend(chosen)
        print(f"selected {len(chosen)}/{len(samples)}")

    # ── Indic MMLU ─────────────────────────────────────────────────────────
    indic_counts: dict[str, int] = {}
    if indic_langs:
        print(f"\n[Indic] Loading {len(indic_langs)} language(s) from {indic_dataset} ...")
        for lang_code in indic_langs:
            all_indic = load_indic_mmlu_language(
                lang_code, indic_dataset, english_categories
            )
            if not all_indic:
                failed_indic.append(lang_code)
                print(f"    SKIPPED (no data for {lang_code})")
                continue

            # Backfill domain using the same domain_map (Indic questions share categories)
            for q in all_indic:
                q["domain"] = domain_map.get(q["category"], "general_knowledge")

            # Group by category and sample per-category
            by_cat: dict[str, list[dict]] = {}
            for q in all_indic:
                by_cat.setdefault(q["category"], []).append(q)

            lang_chosen = 0
            for cat_samples in by_cat.values():
                chosen = stratified_sample(cat_samples, args.n, rng)
                subset.extend(chosen)
                lang_chosen += len(chosen)

            indic_counts[lang_code] = lang_chosen
            lang_name = INDIC_LANG_NAMES[lang_code]
            print(f"    {lang_name}: {lang_chosen} questions across {len(by_cat)} categories")

    # ── Save ───────────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)

    en_count = sum(1 for q in subset if q["language"] == "en")
    indic_total = len(subset) - en_count

    metadata = {
        "version": 1,
        "seed": args.seed,
        "questions_per_category": args.n,
        "total_questions": len(subset),
        "english_questions": en_count,
        "indic_questions": indic_total,
        "indic_languages_included": indic_langs,
        "indic_language_counts": indic_counts,
        "english_categories_included": len(english_categories) - len(failed_en),
        "english_categories_failed": failed_en,
        "indic_languages_failed": failed_indic,
        "FROZEN": True,
        "WARNING": (
            "DO NOT MODIFY THIS FILE mid-training. "
            "Only rebuild with --force if you restart training from scratch."
        ),
    }
    output = {"metadata": metadata, "questions": subset}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(f"  Subset saved to: {args.out}")
    print(f"  Total questions : {len(subset)}")
    print(f"    English       : {en_count}")
    print(f"    Indic         : {indic_total}")
    if indic_counts:
        for lang, count in indic_counts.items():
            print(f"      {INDIC_LANG_NAMES[lang]:<22}: {count}")
    if failed_en:
        print(f"  English failures: {failed_en}")
    if failed_indic:
        print(f"  Indic failures  : {failed_indic}")
    print("=" * 65)
    print("\n  IMPORTANT: Commit this file so all team members use the same")
    print("  frozen evaluation set.\n")


if __name__ == "__main__":
    main()
