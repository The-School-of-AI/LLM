#!/usr/bin/env python3
"""
IDFT Smoke Test Orchestrator
Team 18: SFT, RL-Style Alignment & Final Post-Training Benchmarks

Runs all 4 phases of the IDFT smoke test:
  Phase 0: Setup & validation
  Phase 1: DDT (phi distribution) validation — go/no-go gate
  Phase 2: Training runs (SFT x3 LRs + IDFT x3 LRs)
  Phase 3: Evaluation on 6 benchmarks
  Phase 4: Decision framework & recommendation

Usage:
    python run_idft_smoke_test.py --config idft_smoke_config.yaml
    python run_idft_smoke_test.py --config idft_smoke_config.yaml --skip_phase1
    python run_idft_smoke_test.py --config idft_smoke_config.yaml --phase 2
"""

import argparse
import gc
import json
import logging
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("idft_smoke_test.log"),
    ],
)
logger = logging.getLogger(__name__)


def phase0_setup(config_path: str) -> Dict[str, Any]:
    """
    Phase 0: Setup and validation.

    - Load and validate config
    - Check model availability
    - Check VRAM / hardware
    - Create output directories

    Returns:
        Dict with config and setup metadata.
    """
    from qlora_config import QLoRAConfig

    logger.info("=" * 70)
    logger.info("PHASE 0: SETUP")
    logger.info("=" * 70)

    config = QLoRAConfig.from_yaml(config_path)
    config.auto_configure_hardware()
    warnings = config.validate()

    for w in warnings:
        logger.warning(w)

    # Create output directories
    base_output = Path(config.training.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)

    for subdir in ["sft_runs", "idft_runs", "eval_results", "phi_diagnostic"]:
        (base_output / subdir).mkdir(exist_ok=True)

    config.print_config()

    return {
        "config": config,
        "config_path": config_path,
        "output_dir": str(base_output),
        "timestamp": datetime.now().isoformat(),
    }


def phase1_ddt_validation(setup: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 1: DDT Validation — phi distribution analysis.

    Go/no-go gate. If phi distribution shows no separation,
    IDFT is incompatible with this MoE and we abort.

    Returns:
        Dict with phi stats and go/no-go decision.
    """
    from phi_diagnostic import compute_phi_distribution, evaluate_phi_results

    logger.info("=" * 70)
    logger.info("PHASE 1: DDT VALIDATION (phi distribution)")
    logger.info("=" * 70)

    config = setup["config"]

    # Load model for diagnostic (lighter than training)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model for phi diagnostic: {config.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load dataset
    from datasets import load_dataset

    dataset = load_dataset(config.data.dataset_name, split=config.data.dataset_split)
    if config.data.max_samples and len(dataset) > config.data.max_samples:
        dataset = dataset.select(range(config.data.max_samples))

    if "text" not in dataset.column_names:
        for col in ["content", "prompt", "instruction"]:
            if col in dataset.column_names:
                dataset = dataset.rename_column(col, "text")
                break

    # Run diagnostic
    phi_results = compute_phi_distribution(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_batches=100,
        batch_size=4,
    )

    decision = evaluate_phi_results(phi_results)

    # Save results
    output_path = Path(setup["output_dir"]) / "phi_diagnostic" / "phase1_results.json"
    with open(output_path, "w") as f:
        json.dump({"phi_stats": phi_results, "decision": decision}, f, indent=2)

    logger.info(f"Phi diagnostic results saved to {output_path}")
    logger.info(f"Decision: {'GO' if decision['go'] else 'NO-GO'}")

    # Free model memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"phi_stats": phi_results, "decision": decision}


def phase2_training_runs(setup: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 2: Training runs.

    Run Standard SFT and IDFT at each LR in the grid.
    Select best LR for each condition based on eval loss.

    Returns:
        Dict with training results and best checkpoints.
    """
    logger.info("=" * 70)
    logger.info("PHASE 2: TRAINING RUNS")
    logger.info("=" * 70)

    config = setup["config"]
    learning_rates = config.training.idft.learning_rates
    base_output = Path(setup["output_dir"])

    results = {"sft_runs": [], "idft_runs": []}

    # --- Standard SFT Runs ---
    for lr in learning_rates:
        logger.info(f"\n--- SFT Run: LR={lr} ---")
        run_config = deepcopy(config)
        run_config.training.method = "sft"
        run_config.training.learning_rate = lr
        run_config.training.output_dir = str(base_output / "sft_runs" / f"lr_{lr}")

        try:
            from train_qlora import train

            trainer = train(run_config)
            eval_loss = _get_best_eval_loss(trainer)
            results["sft_runs"].append(
                {
                    "lr": lr,
                    "output_dir": run_config.training.output_dir,
                    "eval_loss": eval_loss,
                    "status": "success",
                }
            )
            del trainer
        except Exception as e:
            logger.error(f"SFT run failed at LR={lr}: {e}")
            results["sft_runs"].append({"lr": lr, "status": "failed", "error": str(e)})
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- IDFT Runs ---
    for lr in learning_rates:
        logger.info(f"\n--- IDFT Run: LR={lr} ---")
        run_config = deepcopy(config)
        run_config.training.method = "idft"
        run_config.training.learning_rate = lr
        run_config.training.output_dir = str(base_output / "idft_runs" / f"lr_{lr}")

        try:
            from train_qlora import train

            trainer = train(run_config)
            eval_loss = _get_best_eval_loss(trainer)
            results["idft_runs"].append(
                {
                    "lr": lr,
                    "output_dir": run_config.training.output_dir,
                    "eval_loss": eval_loss,
                    "status": "success",
                }
            )
            del trainer
        except Exception as e:
            logger.error(f"IDFT run failed at LR={lr}: {e}")
            results["idft_runs"].append({"lr": lr, "status": "failed", "error": str(e)})
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Select best checkpoint per condition
    results["sft_best"] = _select_best_run(results["sft_runs"])
    results["idft_best"] = _select_best_run(results["idft_runs"])

    logger.info(f"\nBest SFT: LR={results['sft_best'].get('lr')}")
    logger.info(f"Best IDFT: LR={results['idft_best'].get('lr')}")

    # Save results
    output_path = base_output / "phase2_training_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def phase3_evaluation(
    setup: Dict[str, Any],
    training_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Phase 3: Evaluation on benchmarks.

    Evaluate best SFT, best IDFT, and base model on all benchmarks.

    Returns:
        Dict with evaluation results for all conditions.
    """
    from evaluate_smoke_test import evaluate_checkpoint

    logger.info("=" * 70)
    logger.info("PHASE 3: EVALUATION")
    logger.info("=" * 70)

    config = setup["config"]
    base_output = Path(setup["output_dir"]) / "eval_results"

    sft_best = training_results.get("sft_best", {})
    idft_best = training_results.get("idft_best", {})

    eval_results = {}

    # Evaluate base model
    logger.info("\nEvaluating base model...")
    eval_results["base"] = evaluate_checkpoint(
        checkpoint_path=config.model.name,
        label="base",
    )

    # Evaluate best SFT
    if sft_best.get("output_dir"):
        logger.info("\nEvaluating best SFT checkpoint...")
        eval_results["sft"] = evaluate_checkpoint(
            checkpoint_path=sft_best["output_dir"],
            label="sft",
            use_peft=True,
            base_model=config.model.name,
        )

    # Evaluate best IDFT
    if idft_best.get("output_dir"):
        logger.info("\nEvaluating best IDFT checkpoint...")
        eval_results["idft"] = evaluate_checkpoint(
            checkpoint_path=idft_best["output_dir"],
            label="idft",
            use_peft=True,
            base_model=config.model.name,
        )

    # Save results
    output_path = base_output / "phase3_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=2, default=str)

    return eval_results


def phase4_decision(
    setup: Dict[str, Any],
    eval_results: Dict[str, Any],
    phi_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Phase 4: Analysis and decision.

    Apply decision framework and print recommendation.

    Returns:
        Dict with final decision and recommendation.
    """
    from evaluate_smoke_test import compare_conditions, print_results_table

    logger.info("=" * 70)
    logger.info("PHASE 4: ANALYSIS & DECISION")
    logger.info("=" * 70)

    sft_results = eval_results.get("sft", {})
    idft_results = eval_results.get("idft", {})
    base_results = eval_results.get("base")

    comparison = compare_conditions(sft_results, idft_results, base_results)

    # Print table
    print_results_table(sft_results, idft_results, base_results)

    # Add phi diagnostic context
    if phi_results:
        comparison["phi_context"] = phi_results.get("phi_stats", {})

    # Save final report
    base_output = Path(setup["output_dir"])
    report_path = base_output / "final_report.json"
    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    logger.info(f"\nFinal report saved to {report_path}")

    return comparison


def _get_best_eval_loss(trainer) -> Optional[float]:
    """Extract best eval loss from trainer state."""
    try:
        if hasattr(trainer.state, "best_metric"):
            return trainer.state.best_metric
        if hasattr(trainer.state, "log_history"):
            eval_losses = [
                entry["eval_loss"]
                for entry in trainer.state.log_history
                if "eval_loss" in entry
            ]
            return min(eval_losses) if eval_losses else None
    except Exception:
        return None


def _select_best_run(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select the run with lowest eval loss."""
    successful = [r for r in runs if r.get("status") == "success"]
    if not successful:
        return {"status": "no_successful_runs"}

    # Prefer runs with eval_loss, fallback to first successful
    with_loss = [r for r in successful if r.get("eval_loss") is not None]
    if with_loss:
        return min(with_loss, key=lambda r: r["eval_loss"])
    return successful[0]


def main():
    parser = argparse.ArgumentParser(description="IDFT Smoke Test Orchestrator")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="idft_smoke_config.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--phase", type=int, default=None, help="Run only a specific phase (0-4)"
    )
    parser.add_argument(
        "--skip_phase1",
        action="store_true",
        help="Skip Phase 1 DDT validation (not recommended)",
    )
    parser.add_argument(
        "--training_results_json",
        type=str,
        default=None,
        help="Path to Phase 2 results JSON (to skip training and go to eval)",
    )
    args = parser.parse_args()

    start_time = time.time()

    # Phase 0: Setup
    setup = phase0_setup(args.config)

    if args.phase is not None and args.phase == 0:
        logger.info("Phase 0 complete. Exiting.")
        return

    # Phase 1: DDT Validation
    phi_results = None
    if not args.skip_phase1 and (args.phase is None or args.phase == 1):
        phi_results = phase1_ddt_validation(setup)

        if not phi_results["decision"]["go"]:
            logger.error(
                "PHASE 1 FAILED: DDT validation indicates IDFT is not "
                "compatible with this model. Aborting."
            )
            for reason in phi_results["decision"]["reasons"]:
                logger.error(f"  - {reason}")
            sys.exit(1)

        if args.phase == 1:
            logger.info("Phase 1 complete. Exiting.")
            return

    # Phase 2: Training Runs
    training_results = None
    if args.training_results_json:
        with open(args.training_results_json) as f:
            training_results = json.load(f)
        logger.info(f"Loaded training results from {args.training_results_json}")
    elif args.phase is None or args.phase == 2:
        training_results = phase2_training_runs(setup)

        if args.phase == 2:
            logger.info("Phase 2 complete. Exiting.")
            return

    # Phase 3: Evaluation
    eval_results = None
    if training_results and (args.phase is None or args.phase == 3):
        eval_results = phase3_evaluation(setup, training_results)

        if args.phase == 3:
            logger.info("Phase 3 complete. Exiting.")
            return

    # Phase 4: Decision
    if eval_results and (args.phase is None or args.phase == 4):
        phase4_decision(setup, eval_results, phi_results)

    elapsed = time.time() - start_time
    logger.info(f"\nTotal elapsed time: {elapsed / 3600:.1f} hours")


if __name__ == "__main__":
    main()
