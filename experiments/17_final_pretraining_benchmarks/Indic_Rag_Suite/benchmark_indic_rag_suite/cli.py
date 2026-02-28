"""
CLI: argparse and config overrides, then run_benchmark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmark_indic_rag_suite.runner import run_benchmark


def _build_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if args.split is not None:
        overrides.setdefault("data", {})["split"] = args.split
    if args.dataset is not None:
        overrides.setdefault("data", {})["dataset_name"] = args.dataset
    if args.lang is not None:
        overrides.setdefault("data", {})["languages"] = [args.lang] if args.lang != "all" else ["all"]
    if args.max_samples is not None:
        overrides.setdefault("data", {})["max_samples_per_lang"] = args.max_samples
    if args.shard_index is not None:
        overrides.setdefault("data", {})["shard_index"] = args.shard_index
    if args.shard_total is not None:
        overrides.setdefault("data", {})["shard_total"] = args.shard_total
    if args.retrieval_backend is not None:
        overrides.setdefault("model", {})["retrieval_backend"] = args.retrieval_backend
    if args.generation_backend is not None:
        overrides.setdefault("model", {})["generation_backend"] = args.generation_backend
    if args.generation_model is not None:
        overrides.setdefault("model", {})["generation_model_name_or_path"] = args.generation_model
    if args.retrieval_model is not None:
        overrides.setdefault("model", {})["retrieval_model_name_or_path"] = args.retrieval_model
    if args.device is not None:
        overrides.setdefault("model", {})["device"] = args.device
    if args.batch_size is not None:
        overrides.setdefault("model", {})["retrieval_batch_size"] = args.batch_size
    if args.tasks is not None:
        overrides.setdefault("run", {})["tasks"] = args.tasks
    if args.output is not None:
        overrides.setdefault("run", {})["output_file"] = args.output
    if args.output_dir is not None:
        overrides.setdefault("run", {})["output_dir"] = args.output_dir
    if args.seed is not None:
        overrides.setdefault("run", {})["seed"] = args.seed
    if args.log_level is not None:
        overrides.setdefault("run", {})["log_level"] = args.log_level
    if getattr(args, "paper_retrieval", False):
        overrides.setdefault("run", {})["retrieval_add_cross_lang_negatives"] = True
    if getattr(args, "mrr_at_k", None) is not None:
        overrides.setdefault("run", {})["retrieval_mrr_at_k"] = args.mrr_at_k
    if getattr(args, "recall_at_k", None) is not None:
        overrides.setdefault("run", {})["recall_at_k_list"] = args.recall_at_k
    if getattr(args, "ndcg_at_k", None) is not None:
        overrides.setdefault("run", {})["ndcg_at_k_list"] = args.ndcg_at_k
    if getattr(args, "use_f1", None) is not None:
        overrides.setdefault("run", {})["use_f1"] = args.use_f1
    if getattr(args, "use_squad_normalize", False):
        overrides.setdefault("run", {})["use_squad_normalize"] = True
    if getattr(args, "use_bleu", False):
        overrides.setdefault("run", {})["use_bleu"] = True
    if getattr(args, "use_rouge", False):
        overrides.setdefault("run", {})["use_rouge"] = True
    if getattr(args, "save_predictions", False):
        overrides.setdefault("run", {})["save_predictions"] = True
    if getattr(args, "generation_evaluator", None) is not None:
        overrides.setdefault("run", {})["generation_evaluator"] = args.generation_evaluator
    return overrides


def run() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Indic-Rag-Suite & IndicMSMARCO: retrieval (MRR@10, Hit@1, Recall@k, NDCG) and generation (EM).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    # Data
    parser.add_argument(
        "--dataset",
        default=None,
        help="ai4bharat/Indic-Rag-Suite (default) or ai4bharat/IndicMSMARCO",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test", "train"],
        default="dev",
        help="dev = quick check; test = full. Use --split test for paper-comparable IndicMSMARCO.",
    )
    parser.add_argument("--lang", default="hi", help="Language code or 'all'")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap samples per language")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index for distributed runs")
    parser.add_argument("--shard-total", type=int, default=1, help="Total shards")
    # Model
    parser.add_argument("--retrieval-backend", default="small", help="small | hf (requires --retrieval-model)")
    parser.add_argument("--generation-backend", default="small", help="small | hf (e.g. --generation-model google/gemma-2-1b)")
    parser.add_argument("--generation-model", default=None, help="Model path for generation (e.g. google/gemma-2-1b)")
    parser.add_argument("--retrieval-model", default=None, help="Model path for retrieval (e.g. BGE-M3)")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0")
    parser.add_argument("--batch-size", type=int, default=16, help="Retrieval encoding batch size")
    # Run
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["retrieval", "generation"],
        choices=["retrieval", "generation"],
        help="Tasks to run",
    )
    parser.add_argument("--mrr-at-k", type=int, default=10, help="MRR cutoff (10 = paper standard for IndicMSMARCO)")
    parser.add_argument("--recall-at-k", type=int, nargs="+", default=None, help="Recall@k and Precision@k values (e.g. 1 5 10 20)")
    parser.add_argument("--ndcg-at-k", type=int, nargs="+", default=None, help="NDCG@k values (e.g. 5 10)")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file")
    parser.add_argument("--output-dir", default=None, help="Output directory (writes results.json)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--use-f1", action="store_true", default=None, help="Report token F1 for generation (default: True)")
    parser.add_argument("--no-use-f1", action="store_false", dest="use_f1", help="Do not report token F1")
    parser.add_argument("--use-squad-normalize", action="store_true", help="Use SQuAD-style normalization for EM/F1")
    parser.add_argument("--use-bleu", action="store_true", help="Report BLEU for generation (requires nltk)")
    parser.add_argument("--use-rouge", action="store_true", help="Report ROUGE-L for generation (requires rouge-score)")
    parser.add_argument("--save-predictions", action="store_true", help="Save per-sample predictions to JSON")
    parser.add_argument(
        "--generation-evaluator",
        choices=["default", "ragas"],
        default=None,
        help="Extra generation evaluator: default (EM/F1 only) or ragas (optional)",
    )
    parser.add_argument(
        "--paper-retrieval",
        action="store_true",
        help="Add other languages as distractors (not paper protocol). Paper uses monolingual retrieval; default is monolingual.",
    )

    args = parser.parse_args()
    config_path = Path(args.config) if args.config else None
    overrides = _build_overrides(args)
    run_benchmark(config_path=config_path, overrides=overrides)
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    sys.exit(run())
