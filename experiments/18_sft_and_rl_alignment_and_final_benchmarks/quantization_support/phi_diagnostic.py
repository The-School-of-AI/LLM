#!/usr/bin/env python3
"""
Phi Distribution Diagnostic — Phase 1 DDT Validation
Team 18: SFT, RL-Style Alignment & Final Post-Training Benchmarks

Validates that the CLL discriminant (phi) separates in-distribution
vs OOD tokens on MoE model outputs. This is a go/no-go gate before
running the full IDFT smoke test.

Usage:
    python phi_diagnostic.py --config idft_smoke_config.yaml
    python phi_diagnostic.py --model_name "Qwen/Qwen2.5-MoE" --max_batches 50
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_phi_distribution(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    max_batches: int = 100,
    batch_size: int = 4,
    max_seq_length: int = 2048,
) -> Dict[str, float]:
    """
    Run forward passes and collect phi statistics on dataset responses.

    Args:
        model: The base model (not fine-tuned).
        tokenizer: Tokenizer.
        dataset: HuggingFace dataset with 'text' column.
        max_batches: Number of batches to process.
        batch_size: Batch size for forward passes.
        max_seq_length: Maximum sequence length.

    Returns:
        Dict with phi statistics.
    """
    model.eval()
    all_phi = []

    def collate_fn(examples):
        texts = [ex["text"] for ex in examples]
        encodings = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )
        return encodings

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift for causal LM
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            shift_mask = attention_mask[:, 1:]

            log_probs = F.log_softmax(shift_logits, dim=-1)
            probs = log_probs.exp()

            token_log_probs = log_probs.gather(
                -1, shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            entropy = -(probs * log_probs).sum(dim=-1)

            phi = token_log_probs + entropy
            valid_phi = phi[shift_mask.bool()]
            all_phi.append(valid_phi.cpu())

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{max_batches} batches")

    all_phi = torch.cat(all_phi)

    results = {
        "phi_mean": all_phi.mean().item(),
        "phi_std": all_phi.std().item(),
        "phi_median": all_phi.median().item(),
        "phi_below_neg1_pct": (all_phi < -1).float().mean().item() * 100,
        "phi_below_neg3_pct": (all_phi < -3).float().mean().item() * 100,
        "phi_below_neg5_pct": (all_phi < -5).float().mean().item() * 100,
        "phi_above_0_pct": (all_phi > 0).float().mean().item() * 100,
        "total_tokens": len(all_phi),
    }
    return results


def evaluate_phi_results(results: Dict[str, float]) -> Dict[str, Any]:
    """
    Apply go/no-go decision based on phi distribution.

    Expected (from paper):
    - Dataset responses: mean phi around -0.07 to -0.33
    - 3-12% of tokens with phi < -3

    Returns:
        Dict with decision and reasoning.
    """
    decision = {
        "go": True,
        "reasons": [],
        "warnings": [],
    }

    # Check 1: phi mean should be negative (indicates OOD presence)
    if results["phi_mean"] > 0.5:
        decision["warnings"].append(
            f"phi_mean={results['phi_mean']:.4f} is positive — "
            "data may already be in-distribution, IDFT gains may be small."
        )

    # Check 2: Some tokens should be strongly OOD
    if results["phi_below_neg3_pct"] < 1.0:
        decision["warnings"].append(
            f"Only {results['phi_below_neg3_pct']:.1f}% tokens have phi < -3 — "
            "very few OOD tokens, IDFT may not provide benefit."
        )

    # Check 3: Distribution should have meaningful spread
    if results["phi_std"] < 0.5:
        decision["go"] = False
        decision["reasons"].append(
            f"phi_std={results['phi_std']:.4f} is very low — "
            "CLL discriminant is not separating tokens. "
            "IDFT may be incompatible with this MoE architecture."
        )

    # Check 4: If nearly all tokens are OOD, something is wrong
    if results["phi_below_neg5_pct"] > 50:
        decision["warnings"].append(
            f"{results['phi_below_neg5_pct']:.1f}% tokens have phi < -5 — "
            "extremely high OOD fraction, consider reducing clip_B."
        )

    if not decision["reasons"]:
        decision["reasons"].append("Phi distribution shows meaningful spread.")

    return decision


def main():
    parser = argparse.ArgumentParser(description="Phi Distribution Diagnostic")
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2")
    parser.add_argument("--dataset_name", type=str, default="OpenAssistant/oasst1")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_json", type=str, default="phi_diagnostic_results.json")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}
    if args.device == "auto":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = args.device

    # Try loading with bf16 to save memory
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, **model_kwargs
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, **model_kwargs
        )

    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    if args.max_samples and len(dataset) > args.max_samples:
        dataset = dataset.select(range(args.max_samples))

    # Ensure text column exists
    if "text" not in dataset.column_names:
        # Try to find a suitable text column
        for col in ["content", "prompt", "instruction"]:
            if col in dataset.column_names:
                dataset = dataset.rename_column(col, "text")
                break

    logger.info(f"Dataset size: {len(dataset)} samples")
    logger.info(f"Running phi diagnostic ({args.max_batches} batches)...")

    results = compute_phi_distribution(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_batches=args.max_batches,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )

    decision = evaluate_phi_results(results)

    # Print results
    print("\n" + "=" * 70)
    print("PHI DISTRIBUTION DIAGNOSTIC RESULTS")
    print("=" * 70)
    print(f"  Model:          {args.model_name}")
    print(f"  Tokens analyzed: {results['total_tokens']:,}")
    print(f"\n  phi mean:    {results['phi_mean']:.4f}")
    print(f"  phi std:     {results['phi_std']:.4f}")
    print(f"  phi median:  {results['phi_median']:.4f}")
    print(f"\n  phi < -1:    {results['phi_below_neg1_pct']:.1f}%")
    print(f"  phi < -3:    {results['phi_below_neg3_pct']:.1f}%")
    print(f"  phi < -5:    {results['phi_below_neg5_pct']:.1f}%")
    print(f"  phi > 0:     {results['phi_above_0_pct']:.1f}%")

    print(f"\n  DECISION: {'GO' if decision['go'] else 'NO-GO'}")
    for reason in decision["reasons"]:
        print(f"    - {reason}")
    for warning in decision["warnings"]:
        print(f"    WARNING: {warning}")
    print("=" * 70)

    # Save results
    output = {"phi_stats": results, "decision": decision}
    output_path = Path(args.output_json)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Exit with code based on decision
    sys.exit(0 if decision["go"] else 1)


if __name__ == "__main__":
    main()
