"""
proxy_validation.py — Before/After Delta Validation Pipeline

This is MANDATORY. You cannot claim synthetic data works without testing.

Pipeline:
  1. Run diagnostic tests on BASE model → baseline scores
  2. Fine-tune on synthetic data (LoRA)
  3. Run same diagnostic tests → post-finetune scores
  4. Compute delta → publish report

Requirements:
  • Uses small proxy models (1B-8B)
  • LoRA fine-tuning for speed
  • Same tests before and after
  • Publishes delta report
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Add parent to path
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))


@dataclass
class ValidationConfig:
    """Configuration for proxy validation."""

    # Models
    proxy_model: str = "Qwen/Qwen2.5-1.5B"  # HuggingFace model ID

    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Training config
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation: int = 4
    max_seq_length: int = 512
    warmup_ratio: float = 0.1

    # Paths
    output_dir: str = "./validation_output"
    synthetic_data_path: str = "./synthetic_data.jsonl"

    # Evaluation
    eval_batch_size: int = 8


@dataclass
class ValidationReport:
    """Report from a validation run."""

    # Metadata
    timestamp: str
    proxy_model: str
    synthetic_data_path: str
    num_samples: int

    # Scores
    baseline_scores: dict  # {test_id: {passed: bool, score: float}}
    post_finetune_scores: dict

    # Aggregates
    baseline_pass_rate: float
    post_finetune_pass_rate: float
    delta: float  # post - baseline

    # By skill
    skill_deltas: dict  # {skill_id: delta}

    # By band
    band_deltas: dict  # {band: delta}

    # Regressions (tests that got WORSE)
    regressions: list[str]

    # Improvements
    improvements: list[str]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "proxy_model": self.proxy_model,
            "synthetic_data_path": self.synthetic_data_path,
            "num_samples": self.num_samples,
            "baseline_pass_rate": self.baseline_pass_rate,
            "post_finetune_pass_rate": self.post_finetune_pass_rate,
            "delta": self.delta,
            "skill_deltas": self.skill_deltas,
            "band_deltas": self.band_deltas,
            "num_regressions": len(self.regressions),
            "num_improvements": len(self.improvements),
            "regressions": self.regressions,
            "improvements": self.improvements,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "  PROXY VALIDATION REPORT",
            "=" * 60,
            f"  Timestamp: {self.timestamp}",
            f"  Model: {self.proxy_model}",
            f"  Synthetic samples: {self.num_samples}",
            "",
            f"  Baseline Pass Rate:     {self.baseline_pass_rate*100:.1f}%",
            f"  Post-Finetune Rate:     {self.post_finetune_pass_rate*100:.1f}%",
            f"  DELTA:                  {self.delta*100:+.1f}%",
            "",
            "  By Skill:",
        ]

        for skill, delta in sorted(self.skill_deltas.items(), key=lambda x: -x[1]):
            sign = "+" if delta >= 0 else ""
            lines.append(f"    {skill:20s} {sign}{delta*100:.1f}%")

        lines.append("")
        lines.append("  By Band:")
        for band, delta in sorted(self.band_deltas.items()):
            sign = "+" if delta >= 0 else ""
            lines.append(f"    {band}: {sign}{delta*100:.1f}%")

        if self.regressions:
            lines.append("")
            lines.append(f"  ⚠️  REGRESSIONS ({len(self.regressions)}):")
            for r in self.regressions[:5]:
                lines.append(f"    - {r}")

        if self.improvements:
            lines.append("")
            lines.append(f"  ✓ Improvements ({len(self.improvements)}):")
            for i in self.improvements[:5]:
                lines.append(f"    + {i}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ================================================================
# LORA FINE-TUNING (using transformers + peft)
# ================================================================


def prepare_training_data(
    synthetic_data_path: str,
    output_path: str,
    max_samples: int | None = None,
) -> int:
    """Prepare synthetic data for training.

    Converts dual-view samples to instruction format.
    """
    samples = []

    with open(synthetic_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                samples.append(sample)

    if max_samples:
        samples = samples[:max_samples]

    # Convert to instruction format
    training_data = []
    for s in samples:
        # Already formatted by inject (has instruction/output) — use directly
        if "instruction" in s and "output" in s:
            training_data.append(
                {
                    "instruction": s["instruction"],
                    "output": s["output"],
                    "skill": s.get("skill_bucket", ""),
                    "band": s.get("band", ""),
                }
            )
            continue

        # Raw bank format — expand into distilled + think views
        training_data.append(
            {
                "instruction": s["question"],
                "output": s["distilled_view"],
                "skill": s.get("skill_bucket", ""),
                "band": s.get("band", ""),
            }
        )
        if s.get("think_view"):
            training_data.append(
                {
                    "instruction": s["question"],
                    "output": f"<think>\n{s['think_view']}\n</think>\n\n{s['distilled_view']}",
                    "skill": s.get("skill_bucket", ""),
                    "band": s.get("band", ""),
                }
            )

    with open(output_path, "w", encoding="utf-8") as f:
        for d in training_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    return len(training_data)


def run_lora_finetune(config: ValidationConfig) -> str:
    """Run LoRA fine-tuning on proxy model.

    Returns path to fine-tuned model.
    """
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install transformers peft datasets torch")
        raise

    print(f"\n[LoRA] Loading model: {config.proxy_model}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.proxy_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.proxy_model,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="auto",
    )

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare data
    training_data_path = Path(config.output_dir) / "training_data.jsonl"
    training_data_path.parent.mkdir(parents=True, exist_ok=True)

    num_samples = prepare_training_data(
        config.synthetic_data_path,
        str(training_data_path),
    )
    print(f"[LoRA] Prepared {num_samples} training samples")

    # Load dataset
    dataset = load_dataset("json", data_files=str(training_data_path), split="train")

    def tokenize(example):
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        return tokenizer(
            text,
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
        )

    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("[LoRA] Starting training...")
    trainer.train()

    # Save
    output_path = Path(config.output_dir) / "lora_adapter"
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print(f"[LoRA] Saved to: {output_path}")
    return str(output_path)


# ================================================================
# EVALUATION WITH OLLAMA
# ================================================================


def evaluate_with_ollama(
    model_name: str,
    tests: list,
) -> dict:
    """Run diagnostic tests using Ollama."""

    # Import from diagnostics
    from diagnostics.run_diagnostics import run_all_tests

    results = run_all_tests(
        model=model_name,
        tests=tests,
        verbose=False,
    )

    return results


def evaluate_with_hf(
    model_path: str,
    tests: list,
    base_model: str | None = None,
) -> dict:
    """Run diagnostic tests using HuggingFace model."""

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("Install transformers, peft, torch")

    from diagnostics.diagnostic_tests import evaluate_test

    # Load model
    print(f"[Eval] Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if it's a LoRA adapter
    if (Path(model_path) / "adapter_config.json").exists():
        assert base_model, "Must specify base_model for LoRA adapter"
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.float16,
            device_map="auto",
        )

    model.eval()

    # Run tests
    results = []
    for test in tests:
        inputs = tokenizer(
            test.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=test.max_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        output_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        eval_result = evaluate_test(test, output_text)
        eval_result.update(
            {
                "test_id": test.id,
                "band": test.band,
                "skill_buckets": test.skill_buckets,
            }
        )
        results.append(eval_result)

    # Compute stats
    passed = sum(1 for r in results if r["passed"])
    pass_rate = passed / len(results) if results else 0

    return {
        "total_tests": len(results),
        "passed": passed,
        "pass_rate": pass_rate,
        "results": results,
    }


# ================================================================
# MAIN VALIDATION PIPELINE
# ================================================================


def run_validation(
    config: ValidationConfig,
    use_ollama_baseline: bool = True,
    baseline_ollama_model: str = "qwen2.5:1.5b",
) -> ValidationReport:
    """Run complete validation pipeline.

    1. Baseline evaluation
    2. LoRA fine-tuning
    3. Post-finetune evaluation
    4. Compute deltas
    5. Generate report
    """

    from diagnostics.diagnostic_tests import DIAGNOSTIC_TESTS

    timestamp = datetime.now().isoformat()

    print("\n" + "=" * 60)
    print("  PROXY VALIDATION PIPELINE")
    print("=" * 60)
    print(f"  Model: {config.proxy_model}")
    print(f"  Synthetic data: {config.synthetic_data_path}")

    # ── Step 1: Baseline ────────────────────────────────────
    print("\n[Step 1] Running baseline evaluation...")

    if use_ollama_baseline:
        baseline_results = evaluate_with_ollama(
            baseline_ollama_model,
            DIAGNOSTIC_TESTS,
        )
    else:
        baseline_results = evaluate_with_hf(
            config.proxy_model,
            DIAGNOSTIC_TESTS,
        )

    baseline_pass_rate = baseline_results["pass_rate"]
    print(f"  Baseline pass rate: {baseline_pass_rate*100:.1f}%")

    # ── Step 2: LoRA Fine-tuning ────────────────────────────
    print("\n[Step 2] Running LoRA fine-tuning...")

    adapter_path = run_lora_finetune(config)

    # ── Step 3: Post-finetune evaluation ────────────────────
    print("\n[Step 3] Running post-finetune evaluation...")

    post_results = evaluate_with_hf(
        adapter_path,
        DIAGNOSTIC_TESTS,
        base_model=config.proxy_model,
    )

    post_pass_rate = post_results["pass_rate"]
    print(f"  Post-finetune pass rate: {post_pass_rate*100:.1f}%")

    # ── Step 4: Compute deltas ──────────────────────────────
    print("\n[Step 4] Computing deltas...")

    delta = post_pass_rate - baseline_pass_rate

    # Build lookup for comparison
    baseline_lookup = {r["test_id"]: r for r in baseline_results["results"]}
    post_lookup = {r["test_id"]: r for r in post_results["results"]}

    # By skill
    skill_baseline = {}
    skill_post = {}
    for r in baseline_results["results"]:
        for skill in r["skill_buckets"]:
            if skill not in skill_baseline:
                skill_baseline[skill] = {"passed": 0, "total": 0}
            skill_baseline[skill]["total"] += 1
            if r["passed"]:
                skill_baseline[skill]["passed"] += 1

    for r in post_results["results"]:
        for skill in r["skill_buckets"]:
            if skill not in skill_post:
                skill_post[skill] = {"passed": 0, "total": 0}
            skill_post[skill]["total"] += 1
            if r["passed"]:
                skill_post[skill]["passed"] += 1

    skill_deltas = {}
    for skill in set(skill_baseline.keys()) | set(skill_post.keys()):
        b_rate = skill_baseline.get(skill, {"passed": 0, "total": 1})
        p_rate = skill_post.get(skill, {"passed": 0, "total": 1})
        b = b_rate["passed"] / b_rate["total"] if b_rate["total"] > 0 else 0
        p = p_rate["passed"] / p_rate["total"] if p_rate["total"] > 0 else 0
        skill_deltas[skill] = p - b

    # By band
    band_baseline = {}
    band_post = {}
    for r in baseline_results["results"]:
        band = r["band"]
        if band not in band_baseline:
            band_baseline[band] = {"passed": 0, "total": 0}
        band_baseline[band]["total"] += 1
        if r["passed"]:
            band_baseline[band]["passed"] += 1

    for r in post_results["results"]:
        band = r["band"]
        if band not in band_post:
            band_post[band] = {"passed": 0, "total": 0}
        band_post[band]["total"] += 1
        if r["passed"]:
            band_post[band]["passed"] += 1

    band_deltas = {}
    for band in set(band_baseline.keys()) | set(band_post.keys()):
        b_rate = band_baseline.get(band, {"passed": 0, "total": 1})
        p_rate = band_post.get(band, {"passed": 0, "total": 1})
        b = b_rate["passed"] / b_rate["total"] if b_rate["total"] > 0 else 0
        p = p_rate["passed"] / p_rate["total"] if p_rate["total"] > 0 else 0
        band_deltas[band] = p - b

    # Find regressions and improvements
    regressions = []
    improvements = []

    for test_id in baseline_lookup:
        b = baseline_lookup[test_id]["passed"]
        p = post_lookup.get(test_id, {}).get("passed", False)

        if b and not p:
            regressions.append(test_id)
        elif not b and p:
            improvements.append(test_id)

    # Count samples
    with open(config.synthetic_data_path, "r", encoding="utf-8") as f:
        num_samples = sum(1 for _ in f)

    # ── Generate report ─────────────────────────────────────
    report = ValidationReport(
        timestamp=timestamp,
        proxy_model=config.proxy_model,
        synthetic_data_path=config.synthetic_data_path,
        num_samples=num_samples,
        baseline_scores={r["test_id"]: r for r in baseline_results["results"]},
        post_finetune_scores={r["test_id"]: r for r in post_results["results"]},
        baseline_pass_rate=baseline_pass_rate,
        post_finetune_pass_rate=post_pass_rate,
        delta=delta,
        skill_deltas=skill_deltas,
        band_deltas=band_deltas,
        regressions=regressions,
        improvements=improvements,
    )

    # Save report
    report_path = Path(config.output_dir) / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    print(report.summary())
    print(f"\n[Done] Report saved to: {report_path}")

    return report


# ================================================================
# CLI
# ================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run proxy validation on synthetic data"
    )
    parser.add_argument("synthetic_data", help="Path to synthetic data JSONL")
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-1.5B", help="HuggingFace proxy model ID"
    )
    parser.add_argument(
        "--output-dir", default="./validation_output", help="Output directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument(
        "--baseline-ollama",
        default="qwen2.5:1.5b",
        help="Ollama model for baseline (faster)",
    )
    parser.add_argument(
        "--skip-finetune",
        action="store_true",
        help="Skip finetuning, only run baseline",
    )

    args = parser.parse_args()

    config = ValidationConfig(
        proxy_model=args.model,
        synthetic_data_path=args.synthetic_data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lora_r=args.lora_r,
    )

    if args.skip_finetune:
        # Just run baseline
        from diagnostics.diagnostic_tests import DIAGNOSTIC_TESTS

        results = evaluate_with_ollama(args.baseline_ollama, DIAGNOSTIC_TESTS)
        print(f"\nBaseline pass rate: {results['pass_rate']*100:.1f}%")
    else:
        run_validation(
            config,
            baseline_ollama_model=args.baseline_ollama,
        )


if __name__ == "__main__":
    main()
