"""
Quantized Evaluation Pipeline — main entrypoint.

Loads a checkpoint (INT4/INT8/fp16), runs all enabled probe suites and the
MMLU subset, and writes a JSON result file to results/raw/.

Usage:
    python evals/quantized/run_eval.py \\
        --checkpoint /path/to/model_or_gguf \\
        --checkpoint-name "step_500" \\
        [--backend bitsandbytes|llama_cpp|hf|auto] \\
        [--quant int4|int8|fp16|fp32] \\
        [--config configs/eval_config.yaml] \\
        [--skip-mmlu] [--skip-probes] \\
        [--verbose]
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.config import load_config, set_seed
from evals.model_loader import ModelConfig, load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run quantized checkpoint evaluation")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint (HF dir or GGUF file)")
    p.add_argument("--checkpoint-name", default=None,
                   help="Human-readable checkpoint label (e.g. step_500, epoch_2)")
    p.add_argument("--backend", default="auto",
                   choices=["auto", "bitsandbytes", "llama_cpp", "hf"],
                   help="Inference backend")
    p.add_argument("--quant", default="int4",
                   choices=["int4", "int8", "fp16", "fp32"],
                   help="Quantization mode")
    p.add_argument("--config", default=None, help="Path to YAML config")
    p.add_argument("--out-dir", default=None, help="Override output directory")
    p.add_argument("--skip-mmlu", action="store_true", help="Skip MMLU evaluation")
    p.add_argument("--skip-probes", action="store_true", help="Skip probe evaluations")
    p.add_argument("--verbose", action="store_true", help="Show per-sample progress bars")
    return p.parse_args()


def get_system_info() -> dict:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "cpu_count": platform.os.cpu_count() if hasattr(platform, "os") else None,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    checkpoint_name = args.checkpoint_name or Path(args.checkpoint).name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{checkpoint_name}_{args.quant}_{timestamp}"

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / cfg["evaluation"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{run_id}.json"

    logger.info(f"Run ID: {run_id}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Backend: {args.backend}  |  Quant: {args.quant}")

    # Load model
    model_cfg = ModelConfig(
        checkpoint_path=args.checkpoint,
        backend=args.backend,
        quant_mode=args.quant,
        max_new_tokens=cfg["evaluation"]["max_new_tokens"],
        temperature=cfg["evaluation"]["temperature"],
        seed=cfg["seed"],
        n_ctx=cfg["quantization"]["llama_cpp"]["n_ctx"],
        n_threads=cfg["quantization"]["llama_cpp"]["n_threads"],
        n_gpu_layers=cfg["quantization"]["llama_cpp"]["n_gpu_layers"],
    )

    t_load_start = time.time()
    model = load_model(model_cfg)
    t_load = time.time() - t_load_start
    logger.info(f"Model loaded in {t_load:.1f}s")

    all_results: dict = {
        "run_id": run_id,
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": str(args.checkpoint),
        "backend": args.backend,
        "quant_mode": args.quant,
        "timestamp_utc": timestamp,
        "load_time_s": round(t_load, 2),
        "system_info": get_system_info(),
        "eval_results": {},
    }

    # --- MMLU ---
    if not args.skip_mmlu:
        from evals.mmlu.run_mmlu import run_mmlu_eval
        subset_path = ROOT / cfg["mmlu"]["subset_file"]
        if not subset_path.exists():
            logger.warning(f"MMLU subset not found at {subset_path}. Run scripts/build_mmlu_subset.py first.")
        else:
            logger.info("Running MMLU evaluation ...")
            mmlu_result = run_mmlu_eval(model, subset_path, verbose=args.verbose)
            all_results["eval_results"]["mmlu"] = mmlu_result
            logger.info(f"MMLU accuracy: {mmlu_result['overall_accuracy']:.4f}")

    # --- Custom Probes ---
    if not args.skip_probes:
        probe_cfg = cfg["probes"]
        probe_data_dir = ROOT / "data" / "probes"

        if probe_cfg["language_modeling"]["enabled"]:
            from evals.probes.language_modeling import run_lm_probe
            lm_path = ROOT / probe_cfg["language_modeling"]["probes_file"]
            logger.info("Running language modeling probes ...")
            lm_result = run_lm_probe(model, lm_path, verbose=args.verbose)
            all_results["eval_results"]["language_modeling"] = lm_result
            logger.info(f"LM mean perplexity: {lm_result['mean_perplexity']:.2f}")

        if probe_cfg["code_continuation"]["enabled"]:
            from evals.probes.code_continuation import run_code_probe
            code_path = ROOT / probe_cfg["code_continuation"]["probes_file"]
            logger.info("Running code continuation probes ...")
            code_result = run_code_probe(model, code_path, verbose=args.verbose)
            all_results["eval_results"]["code_continuation"] = code_result
            logger.info(f"Code pass rate: {code_result['pass_rate']:.4f}")

        if probe_cfg["math_prose"]["enabled"]:
            from evals.probes.math_prose import run_math_probe
            math_path = ROOT / probe_cfg["math_prose"]["probes_file"]
            logger.info("Running math prose probes ...")
            math_result = run_math_probe(model, math_path, verbose=args.verbose)
            all_results["eval_results"]["math_prose"] = math_result
            logger.info(f"Math accuracy: {math_result['accuracy']:.4f}")

        if probe_cfg["consistency"]["enabled"]:
            from evals.probes.consistency import run_consistency_probe
            cons_path = ROOT / probe_cfg["consistency"]["probes_file"]
            logger.info("Running consistency probes ...")
            cons_result = run_consistency_probe(model, cons_path, verbose=args.verbose)
            all_results["eval_results"]["consistency"] = cons_result
            logger.info(f"Consistency rate: {cons_result['consistency_rate']:.4f}  |  mean agreement: {cons_result['mean_agreement_rate']:.4f}")

    # Save results
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Results saved to: {out_file}")
    er = all_results["eval_results"]
    SEP = "=" * 64
    print(f"\n{SEP}")
    print(f"  Run ID       : {run_id}")
    print(f"  Checkpoint   : {checkpoint_name}")
    print(f"  Backend      : {args.backend}  |  Quant: {args.quant}")
    print(f"  Output       : {out_file.name}")
    print(f"{'-' * 64}")
    if "mmlu" in er:
        mmlu = er["mmlu"]
        print(f"  MMLU Overall : {mmlu['overall_accuracy']:.4f}")
        dom_acc = mmlu.get("domain_accuracies", {})
        DOMAIN_LABELS = {
            "math":              "    Math",
            "reasoning":         "    Reasoning",
            "science":           "    Science",
            "coding":            "    Coding",
            "general_knowledge": "    General Knowledge",
        }
        for dom, info in dom_acc.items():
            acc = info["accuracy"] if isinstance(info, dict) else info
            total = info.get("total", "?") if isinstance(info, dict) else "?"
            label = DOMAIN_LABELS.get(dom, f"    {dom}")
            print(f"  {label:<28}: {acc:.4f}  ({total} q)")
        lang_acc = mmlu.get("language_accuracies", {})
        if len(lang_acc) > 1:
            print(f"  {'  Languages':<28}:")
            for lang, info in lang_acc.items():
                acc = info["accuracy"] if isinstance(info, dict) else info
                total = info.get("total", "?") if isinstance(info, dict) else "?"
                print(f"    {lang:<12}: {acc:.4f}  ({total} q)")
    if "language_modeling" in er:
        print(f"  LM Perplexity: {er['language_modeling']['mean_perplexity']:.2f}")
    if "code_continuation" in er:
        print(f"  Code PassRate: {er['code_continuation']['pass_rate']:.4f}")
    if "math_prose" in er:
        print(f"  Math Accuracy: {er['math_prose']['accuracy']:.4f}")
    if "consistency" in er:
        print(f"  Consistency  : {er['consistency']['consistency_rate']:.4f}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
