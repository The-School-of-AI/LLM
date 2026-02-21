#!/usr/bin/env python3
"""
Team 16 Early Warning — Master Pipeline Orchestrator
=====================================================

Edit the CHECKPOINTS list below to add your model(s), then run:

    python run_pipeline.py                        # run all checkpoints
    python run_pipeline.py --name gpt2_baseline   # run one specific checkpoint
    python run_pipeline.py --list                 # show registered checkpoints
    python run_pipeline.py --skip-eval            # trend + report only (no new eval)

The script will:
  1.  (Once) Build the frozen MMLU subset if it doesn't exist yet
  2.  For each checkpoint: load model → run probes + MMLU → save results/raw/<run>.json
  3.  Aggregate all raw results → results/aggregated_results.json
  4.  Track trends → detect regressions/plateaus/instability → results/trend_db.json
  5.  Generate plots (PNG + interactive HTML) → results/plots/
  6.  Write early-warning report (MD + HTML) → results/reports/

──────────────────────────────────────────────────────────────────
SECTION 1 — REGISTER YOUR CHECKPOINTS HERE
──────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EDIT THIS SECTION — add / remove checkpoints freely                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

CHECKPOINTS: list[dict] = [
    # ── Open-source baselines (used for smoke-testing while training runs) ──
    #
    # GPT-2 small (117M) — tiny, fast, always available on HuggingFace.
    # No download needed; transformers caches it automatically.
    {
        "name":     "gpt2_baseline",
        "path":     "gpt2",                 # HuggingFace model ID
        "backend":  "hf",                   # plain HF (no quantization needed for tiny model)
        "quant":    "fp32",
        "enabled":  True,
        "notes":    "GPT-2 117M — smoke-test / absolute baseline",
    },

    # GPT-2 medium (345M) — slightly larger baseline.
    {
        "name":     "gpt2_medium_baseline",
        "path":     "gpt2-medium",
        "backend":  "hf",
        "quant":    "fp16",
        "enabled":  False,                  # set True to include
        "notes":    "GPT-2 345M — intermediate baseline",
    },

    # SmolLM2-135M — very small, HF-native, great for local testing.
    {
        "name":     "smollm2_135m",
        "path":     "HuggingFaceTB/SmolLM2-135M",
        "backend":  "hf",
        "quant":    "fp32",
        "enabled":  False,
        "notes":    "SmolLM2 135M — fast on CPU, good smoke-test",
    },

    # ── YOUR MODEL CHECKPOINTS (fill these in as training progresses) ───────
    #
    # Replace the path values with your actual checkpoint directories or
    # GGUF files. Set enabled=True when a checkpoint is ready.

    {
        "name":     "my_model_step_0",      # rename to match your step count
        "path":     "/path/to/your/model/checkpoint-0",   # ← UPDATE THIS
        "backend":  "auto",                 # auto-detects GGUF vs HF
        "quant":    "int4",
        "enabled":  False,                  # ← set True when checkpoint exists
        "notes":    "Initial checkpoint — before any training",
    },
    {
        "name":     "my_model_step_500",
        "path":     "/path/to/your/model/checkpoint-500",  # ← UPDATE THIS
        "backend":  "auto",
        "quant":    "int4",
        "enabled":  False,
        "notes":    "Step 500",
    },
    {
        "name":     "my_model_step_1000",
        "path":     "/path/to/your/model/checkpoint-1000", # ← UPDATE THIS
        "backend":  "auto",
        "quant":    "int4",
        "enabled":  False,
        "notes":    "Step 1000",
    },
    {
        "name":     "my_model_step_2000",
        "path":     "/path/to/your/model/checkpoint-2000", # ← UPDATE THIS
        "backend":  "auto",
        "quant":    "int4",
        "enabled":  False,
        "notes":    "Step 2000",
    },
    # Add more checkpoints by copying the block above ↑
]

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PIPELINE OPTIONS — change these if needed                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

PIPELINE_OPTIONS = {
    # Skip probes/MMLU for a checkpoint if a results file already exists
    # for it (avoids re-running expensive evals on unchanged checkpoints).
    "skip_if_already_evaluated": True,

    # Plot format: "png", "html", or "both"
    "plot_format": "both",

    # Show per-sample tqdm bars during eval
    "verbose_eval": True,

    # Submit results to a Flask collector after each eval.
    # Set to None to disable.
    "collector_url": None,           # e.g. "http://192.168.1.100:5001"

    # Build MMLU subset automatically if not present (requires internet).
    "auto_build_mmlu": True,
}

# ══════════════════════════════════════════════════════════════════════════
#   ↓  DO NOT EDIT BELOW THIS LINE  ↓
# ══════════════════════════════════════════════════════════════════════════

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ── ANSI colours (graceful degradation on Windows) ────────────────────────
try:
    import sys as _sys
    _USE_COLOR = _sys.stdout.isatty()
except Exception:
    _USE_COLOR = False

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

GREEN  = lambda t: _c("32",   t)
YELLOW = lambda t: _c("33",   t)
RED    = lambda t: _c("31",   t)
CYAN   = lambda t: _c("96",   t)
BOLD   = lambda t: _c("1",    t)
DIM    = lambda t: _c("2",    t)


# ── Helpers ────────────────────────────────────────────────────────────────

def banner(text: str, width: int = 68) -> None:
    line = "─" * width
    print(f"\n{CYAN(line)}")
    print(f"  {BOLD(text)}")
    print(f"{CYAN(line)}")


def section(step: int, total: int, text: str) -> None:
    print(f"\n{BOLD(CYAN(f'[{step}/{total}]'))}  {text}")


def result_exists(checkpoint_name: str, raw_dir: Path) -> Path | None:
    """Return the first existing result file for this checkpoint, or None."""
    for f in raw_dir.glob(f"{checkpoint_name}_*.json"):
        return f
    return None


def load_existing_result(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def print_metrics(results: dict) -> None:
    er = results.get("eval_results", {})
    rows = []
    if "mmlu"               in er: rows.append(("MMLU Accuracy",   f"{er['mmlu']['overall_accuracy']:.4f}",              "↑"))
    if "language_modeling"  in er: rows.append(("LM Perplexity",   f"{er['language_modeling']['mean_perplexity']:.2f}",  "↓"))
    if "code_continuation"  in er: rows.append(("Code Pass Rate",  f"{er['code_continuation']['pass_rate']:.4f}",        "↑"))
    if "math_prose"         in er: rows.append(("Math Accuracy",   f"{er['math_prose']['accuracy']:.4f}",                "↑"))
    if "consistency"        in er: rows.append(("Consistency",     f"{er['consistency']['mean_agreement_rate']:.4f}",    "↑"))
    for label, value, direction in rows:
        print(f"    {DIM(label + ':'): <28}  {GREEN(value)}  {DIM(direction)}")

    # Per-domain MMLU breakdown
    if "mmlu" in er and "domain_accuracies" in er["mmlu"]:
        DOMAIN_ICONS = {
            "math":              "  ∑  Math",
            "reasoning":         "  ⚙  Reasoning",
            "science":           "  🔬 Science",
            "coding":            "  💻 Coding",
            "general_knowledge": "  🌐 General Knowledge",
        }
        dom_acc = er["mmlu"]["domain_accuracies"]
        print(f"    {DIM('MMLU by domain:')}")
        for dom, info in dom_acc.items():
            label = DOMAIN_ICONS.get(dom, f"  {dom}")
            acc = info["accuracy"] if isinstance(info, dict) else info
            total = info.get("total", "?") if isinstance(info, dict) else "?"
            print(f"      {DIM(label + ':'): <32}  {GREEN(f'{acc:.4f}')}  {DIM(f'({total} q)')}")

    # Per-language breakdown (only when Indic languages present)
    if "mmlu" in er:
        lang_acc = er["mmlu"].get("language_accuracies", {})
        if len(lang_acc) > 1:  # more than just English
            print(f"    {DIM('MMLU by language:')}")
            for lang, info in lang_acc.items():
                acc = info["accuracy"] if isinstance(info, dict) else info
                total = info.get("total", "?") if isinstance(info, dict) else "?"
                print(f"      {DIM(lang + ':'): <12}  {GREEN(f'{acc:.4f}')}  {DIM(f'({total} q)')}")


# ── Step implementations ───────────────────────────────────────────────────

def step_build_mmlu(cfg_path: Path) -> None:
    from utils.config import load_config
    cfg = load_config(cfg_path)
    subset_path = ROOT / cfg["mmlu"]["subset_file"]
    if subset_path.exists():
        log.info(f"MMLU subset already exists ({subset_path.stat().st_size // 1024} KB) — skipping rebuild.")
        return
    log.info("Building frozen MMLU subset (requires internet) ...")
    import scripts.build_mmlu_subset as bm
    # Patch sys.argv so argparse inside the module sees no args
    _orig = sys.argv
    sys.argv = ["build_mmlu_subset.py"]
    try:
        bm.main()
    finally:
        sys.argv = _orig


def step_run_eval(ckpt: dict, cfg_path: Path, raw_dir: Path) -> dict:
    """Import and run the eval pipeline in-process (no subprocess overhead)."""
    from utils.config import load_config, set_seed
    from evals.model_loader import ModelConfig, load_model
    import platform, socket

    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])

    checkpoint_name = ckpt["name"]
    checkpoint_path = ckpt["path"]
    quant_mode      = ckpt.get("quant", "int4")
    backend         = ckpt.get("backend", "auto")

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id    = f"{checkpoint_name}_{quant_mode}_{timestamp}"
    out_file  = raw_dir / f"{run_id}.json"

    log.info(f"Loading model: {checkpoint_path}  [{backend} / {quant_mode}]")
    model_cfg = ModelConfig(
        checkpoint_path=checkpoint_path,
        backend=backend,
        quant_mode=quant_mode,
        max_new_tokens=cfg["evaluation"]["max_new_tokens"],
        temperature=cfg["evaluation"]["temperature"],
        seed=cfg["seed"],
        n_ctx=cfg["quantization"]["llama_cpp"]["n_ctx"],
        n_threads=cfg["quantization"]["llama_cpp"]["n_threads"],
        n_gpu_layers=cfg["quantization"]["llama_cpp"]["n_gpu_layers"],
    )
    t0 = time.time()
    model = load_model(model_cfg)
    load_time = round(time.time() - t0, 2)
    log.info(f"Model loaded in {load_time}s")

    all_results: dict = {
        "run_id": run_id,
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": str(checkpoint_path),
        "backend": backend,
        "quant_mode": quant_mode,
        "timestamp_utc": timestamp,
        "load_time_s": load_time,
        "system_info": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": sys.version,
        },
        "eval_results": {},
    }

    verbose = PIPELINE_OPTIONS["verbose_eval"]

    # MMLU
    subset_path = ROOT / cfg["mmlu"]["subset_file"]
    if subset_path.exists():
        from evals.mmlu.run_mmlu import run_mmlu_eval
        log.info("  → MMLU ...")
        mmlu_r = run_mmlu_eval(model, subset_path, verbose=verbose)
        all_results["eval_results"]["mmlu"] = mmlu_r
        log.info(f"    MMLU accuracy: {mmlu_r['overall_accuracy']:.4f}")
    else:
        log.warning("  MMLU subset not found — skipping MMLU eval")

    # Probes
    probe_cfg = cfg["probes"]
    if probe_cfg["language_modeling"]["enabled"]:
        from evals.probes.language_modeling import run_lm_probe
        log.info("  → Language modeling probes ...")
        lm_r = run_lm_probe(model, ROOT / probe_cfg["language_modeling"]["probes_file"], verbose=verbose)
        all_results["eval_results"]["language_modeling"] = lm_r
        log.info(f"    Perplexity: {lm_r['mean_perplexity']:.2f}")

    if probe_cfg["code_continuation"]["enabled"]:
        from evals.probes.code_continuation import run_code_probe
        log.info("  → Code continuation probes ...")
        code_r = run_code_probe(model, ROOT / probe_cfg["code_continuation"]["probes_file"], verbose=verbose)
        all_results["eval_results"]["code_continuation"] = code_r
        log.info(f"    Code pass rate: {code_r['pass_rate']:.4f}")

    if probe_cfg["math_prose"]["enabled"]:
        from evals.probes.math_prose import run_math_probe
        log.info("  → Math prose probes ...")
        math_r = run_math_probe(model, ROOT / probe_cfg["math_prose"]["probes_file"], verbose=verbose)
        all_results["eval_results"]["math_prose"] = math_r
        log.info(f"    Math accuracy: {math_r['accuracy']:.4f}")

    if probe_cfg["consistency"]["enabled"]:
        from evals.probes.consistency import run_consistency_probe
        log.info("  → Consistency probes ...")
        cons_r = run_consistency_probe(model, ROOT / probe_cfg["consistency"]["probes_file"], verbose=verbose)
        all_results["eval_results"]["consistency"] = cons_r
        log.info(f"    Agreement rate: {cons_r['mean_agreement_rate']:.4f}")

    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"  Results saved → {out_file.name}")
    return all_results


def step_aggregate(raw_dir: Path, agg_file: Path) -> dict:
    from collector.collect_results import load_all_results, aggregate
    records = load_all_results(raw_dir)
    agg = aggregate(records)
    agg_file.parent.mkdir(parents=True, exist_ok=True)
    with open(agg_file, "w") as f:
        json.dump(agg, f, indent=2)
    return agg


def step_track_trends(agg_file: Path, trend_db: Path, plots_dir: Path, plot_format: str,
                      cfg_path: Path) -> dict:
    from scripts.track_trends import (
        load_aggregated, extract_metric_series, compute_deltas,
        detect_anomalies, build_trend_db, plot_matplotlib, plot_plotly,
    )
    from utils.config import load_config
    cfg = load_config(cfg_path)
    runs   = load_aggregated(agg_file)
    series = extract_metric_series(runs)
    deltas = compute_deltas(series)
    anomalies = detect_anomalies(series, cfg)
    db = build_trend_db(runs, series, deltas, anomalies)
    trend_db.parent.mkdir(parents=True, exist_ok=True)
    with open(trend_db, "w") as f:
        json.dump(db, f, indent=2)
    plots_dir.mkdir(parents=True, exist_ok=True)
    if plot_format in ("png", "both"):
        plot_matplotlib(series, anomalies, plots_dir)
    if plot_format in ("html", "both"):
        plot_plotly(series, anomalies, plots_dir)
    return db


def step_generate_report(trend_db_path: Path, reports_dir: Path, cfg_path: Path) -> Path:
    from scripts.generate_report import (
        build_warning_items, confidence_level,
        generate_markdown_report, generate_html_report,
    )
    from utils.config import load_config
    import json as _json

    cfg = load_config(cfg_path)
    with open(trend_db_path) as f:
        trend_db = _json.load(f)

    checkpoints   = trend_db.get("checkpoints", [])
    anomalies     = trend_db.get("anomalies", {})
    warning_items = build_warning_items(anomalies, checkpoints)
    conf_label, conf_score = confidence_level(len(warning_items), len(checkpoints), cfg)

    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    # JSON
    json_report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "confidence": {"label": conf_label, "score": conf_score},
        "num_checkpoints": len(checkpoints),
        "checkpoints": checkpoints,
        "num_warnings": len(warning_items),
        "warnings": warning_items,
        "notify_teams": cfg["early_warning"]["notify_teams"],
    }
    json_path = reports_dir / f"early_warning_{ts}.json"
    with open(json_path, "w") as f:
        _json.dump(json_report, f, indent=2)

    # Markdown
    md = generate_markdown_report(trend_db, warning_items, conf_label, conf_score, cfg)
    md_path = reports_dir / f"early_warning_{ts}.md"
    with open(md_path, "w") as f:
        f.write(md)

    # HTML
    plots_dir = ROOT / "results" / "plots"
    html = generate_html_report(md, warning_items, plots_dir, reports_dir)
    html_path = reports_dir / f"early_warning_{ts}.html"
    with open(html_path, "w") as f:
        f.write(html)

    # Stable "latest" copies
    for src, name in [(json_path, "latest_early_warning.json"),
                      (md_path,   "latest_early_warning.md"),
                      (html_path, "latest_early_warning.html")]:
        with open(src) as fr, open(reports_dir / name, "w") as fw:
            fw.write(fr.read())

    return md_path


def step_submit(result_file: Path, url: str) -> None:
    try:
        import urllib.request
        with open(result_file) as f:
            payload = f.read()
        req = urllib.request.Request(
            url.rstrip("/") + "/submit",
            data=payload.encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            log.info(f"  Submitted to collector: {resp.read().decode()}")
    except Exception as e:
        log.warning(f"  Could not submit to collector: {e}")


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Team 16 Early Warning — master pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                      # run all enabled checkpoints
  python run_pipeline.py --name gpt2_baseline # run one checkpoint only
  python run_pipeline.py --list               # show registered checkpoints
  python run_pipeline.py --skip-eval          # aggregate + trend + report only
  python run_pipeline.py --force              # re-run even if result exists
        """,
    )
    p.add_argument("--name",      help="Run only the checkpoint with this name")
    p.add_argument("--list",      action="store_true", help="List all registered checkpoints and exit")
    p.add_argument("--skip-eval", action="store_true", help="Skip model loading; only aggregate + trend + report")
    p.add_argument("--force",     action="store_true", help="Re-run eval even if result file already exists")
    p.add_argument("--config",    default=str(ROOT / "configs" / "eval_config.yaml"))
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg_path  = Path(args.config)
    raw_dir   = ROOT / "results" / "raw"
    agg_file  = ROOT / "results" / "aggregated_results.json"
    trend_db  = ROOT / "results" / "trend_db.json"
    plots_dir = ROOT / "results" / "plots"
    reports_dir = ROOT / "results" / "reports"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ── --list ──────────────────────────────────────────────────────────────
    if args.list:
        banner("Registered Checkpoints")
        enabled  = [c for c in CHECKPOINTS if c.get("enabled")]
        disabled = [c for c in CHECKPOINTS if not c.get("enabled")]
        print(f"\n  {GREEN('ENABLED')}  ({len(enabled)} checkpoints)")
        for c in enabled:
            print(f"    ✓  {BOLD(c['name']):30}  {DIM(c['path'])}")
            print(f"       {DIM('backend='+ c.get('backend','auto') +'  quant='+ c.get('quant','int4'))}")
            if c.get("notes"):
                print(f"       {DIM(c['notes'])}")
        print(f"\n  {YELLOW('DISABLED')}  ({len(disabled)} checkpoints)")
        for c in disabled:
            print(f"    ✗  {DIM(c['name'])}")
        print()
        return

    # ── Filter checkpoints ──────────────────────────────────────────────────
    candidates = [c for c in CHECKPOINTS if c.get("enabled", False)]
    if args.name:
        candidates = [c for c in CHECKPOINTS if c["name"] == args.name]
        if not candidates:
            print(RED(f"No checkpoint named '{args.name}' found (check enabled flag)."))
            sys.exit(1)

    if not candidates and not args.skip_eval:
        print(YELLOW("No enabled checkpoints found. Edit CHECKPOINTS in run_pipeline.py."))
        sys.exit(0)

    total_steps = 1 + len(candidates) + 3   # mmlu-build + evals + agg + trend + report

    banner(f"Team 16 Early Warning Pipeline  —  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Checkpoints to evaluate : {BOLD(str(len(candidates)))}")
    print(f"  Skip eval               : {args.skip_eval}")
    print(f"  Config                  : {cfg_path}")
    print(f"  Results dir             : {raw_dir}")

    step = 0

    # ── Step 1: MMLU subset ─────────────────────────────────────────────────
    if not args.skip_eval and PIPELINE_OPTIONS["auto_build_mmlu"]:
        step += 1
        section(step, total_steps, "MMLU Subset")
        try:
            step_build_mmlu(cfg_path)
        except Exception as e:
            log.warning(f"MMLU build failed (continuing without it): {e}")

    # ── Step 2+: Evaluate each checkpoint ───────────────────────────────────
    run_summaries: list[dict] = []

    for ckpt in candidates:
        step += 1
        section(step, total_steps, f"Evaluating  {BOLD(ckpt['name'])}  [{ckpt.get('quant','int4')} / {ckpt.get('backend','auto')}]")
        if ckpt.get("notes"):
            print(f"  {DIM(ckpt['notes'])}")

        if args.skip_eval:
            existing = result_exists(ckpt["name"], raw_dir)
            if existing:
                print(f"  {DIM('Using existing result:')} {existing.name}")
                run_summaries.append(load_existing_result(existing))
            else:
                print(f"  {YELLOW('No existing result found — skipping.')}")
            continue

        # Check if already evaluated
        if PIPELINE_OPTIONS["skip_if_already_evaluated"] and not args.force:
            existing = result_exists(ckpt["name"], raw_dir)
            if existing:
                print(f"  {GREEN('Already evaluated')} → {DIM(existing.name)}")
                print(f"  {DIM('(use --force to re-run)')}")
                result = load_existing_result(existing)
                print_metrics(result)
                run_summaries.append(result)
                continue

        t_start = time.time()
        try:
            result = step_run_eval(ckpt, cfg_path, raw_dir)
            elapsed = round(time.time() - t_start, 1)
            print(f"\n  {GREEN('✓ Completed')} in {elapsed}s")
            print_metrics(result)
            run_summaries.append(result)

            # Optional: submit to Flask collector
            if PIPELINE_OPTIONS["collector_url"]:
                out_files = sorted(raw_dir.glob(f"{ckpt['name']}_*.json"), key=lambda f: f.stat().st_mtime)
                if out_files:
                    step_submit(out_files[-1], PIPELINE_OPTIONS["collector_url"])

        except Exception as e:
            log.error(f"  {RED('FAILED')}: {e}")
            import traceback
            traceback.print_exc()
            print(f"  {YELLOW('Continuing to next checkpoint ...')}")

    # ── Aggregate ───────────────────────────────────────────────────────────
    step += 1
    section(step, total_steps, "Aggregating results")
    try:
        agg = step_aggregate(raw_dir, agg_file)
        print(f"  {GREEN('✓')} {agg['total_runs']} runs aggregated → {agg_file.name}")
    except Exception as e:
        log.error(f"Aggregation failed: {e}")
        return

    # ── Trend tracking ──────────────────────────────────────────────────────
    step += 1
    section(step, total_steps, "Tracking trends & detecting anomalies")
    try:
        trend_data = step_track_trends(agg_file, trend_db, plots_dir,
                                       PIPELINE_OPTIONS["plot_format"], cfg_path)
        anom_summary = trend_data.get("anomaly_summary", {})
        total_anoms  = sum(len(v) for v in anom_summary.values())
        if total_anoms:
            print(f"  {YELLOW(f'⚠  {total_anoms} anomaly/ies detected:')}")
            for metric, types in anom_summary.items():
                print(f"    {metric}: {', '.join(types)}")
        else:
            print(f"  {GREEN('✓ No anomalies detected')}")
        print(f"  Plots → {plots_dir}")
    except Exception as e:
        log.error(f"Trend tracking failed: {e}")
        import traceback; traceback.print_exc()

    # ── Early warning report ────────────────────────────────────────────────
    step += 1
    section(step, total_steps, "Generating early warning report")
    try:
        if trend_db.exists():
            md_path = step_generate_report(trend_db, reports_dir, cfg_path)
            print(f"  {GREEN('✓')} Report → {md_path.name}")

            # Print the report inline
            print()
            with open(md_path) as f:
                for line in f:
                    print("  " + line, end="")
            print()
        else:
            print(f"  {YELLOW('No trend DB yet — skipping report.')}")
    except Exception as e:
        log.error(f"Report generation failed: {e}")
        import traceback; traceback.print_exc()

    # ── Final summary ───────────────────────────────────────────────────────
    banner("Pipeline Complete")
    print(f"  Raw results  : {raw_dir}")
    print(f"  Aggregated   : {agg_file}")
    print(f"  Trend DB     : {trend_db}")
    print(f"  Plots        : {plots_dir}")
    print(f"  Report (MD)  : {reports_dir / 'latest_early_warning.md'}")
    print(f"  Report (HTML): {reports_dir / 'latest_early_warning.html'}")
    print()


if __name__ == "__main__":
    main()
