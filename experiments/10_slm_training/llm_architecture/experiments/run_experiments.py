"""
Experiment Runner
==================

Automated testing of different LLM configurations.

Runs experiments for:
1. Base model (GQA)
2. GSA attention
3. DeepSeek Sparse attention
4. mHC connections
5. Multi-token prediction
6. YaRN extended context
7. Combined configurations

Generates comparison reports.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import (
    AttentionConfig,
    AttentionType,
    ConnectionConfig,
    ConnectionType,
    FFNConfig,
    FFNType,
    HeadConfig,
    ModelConfig,
    PositionConfig,
    PositionEmbeddingType,
)
from models.llm import create_model_from_config
from torch.utils.data import DataLoader
from training.train import RandomTextDataset, Trainer, TrainingConfig


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    name: str
    description: str
    model_config: ModelConfig
    training_steps: int = 10000
    batch_size: int = 8
    seq_length: int = 1024


@dataclass
class ExperimentResult:
    """Results from a single experiment."""

    name: str
    final_loss: float
    best_loss: float
    avg_tokens_per_second: float
    total_time_seconds: float
    tokens_seen: int
    parameters: int
    config_summary: Dict[str, str]


class ExperimentRunner:
    """
    Runs multiple experiments and compares results.
    """

    def __init__(
        self,
        output_dir: str = "./experiments",
        training_steps: int = 10000,
        batch_size: int = 8,
        seq_length: int = 1024,
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.training_steps = training_steps
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.seed = seed

        self.results: List[ExperimentResult] = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def create_experiments(self) -> List[ExperimentConfig]:
        """
        Create all experiment configurations.

        Following the 7-step plan:
        1. Base dense model (GQA)
        2. GSA attention
        3. DeepSeek Sparse attention
        4. mHC connections
        5. Multi-token prediction
        6. YaRN extended context
        7. Best combination
        """
        experiments = []

        # Step 1: Base model with GQA (like Qwen3/SmolLM2)
        base_config = self._create_base_config()
        experiments.append(
            ExperimentConfig(
                name="step1_base_gqa",
                description="Base 1B model with Grouped Query Attention",
                model_config=base_config,
            )
        )

        # Step 3: GSA (Gated Sparse Attention)
        gsa_config = self._create_base_config()
        gsa_config.model_name = "LLM-1B-GSA"
        gsa_config.attention.attention_type = AttentionType.GATED_SPARSE
        gsa_config.attention.gsa_num_slots = 64
        gsa_config.attention.gsa_sparse_topk = 32
        experiments.append(
            ExperimentConfig(
                name="step3_gsa",
                description="Model with Gated Sparse Attention (paper 2601.15305v1)",
                model_config=gsa_config,
            )
        )

        # Step 4: DeepSeek Sparse Attention
        ds_config = self._create_base_config()
        ds_config.model_name = "LLM-1B-DeepSeek"
        ds_config.attention.attention_type = AttentionType.DEEPSEEK_SPARSE
        ds_config.attention.ds_compressed_dim = 512
        ds_config.attention.ds_rope_head_dim = 32
        experiments.append(
            ExperimentConfig(
                name="step4_deepseek_sparse",
                description="Model with DeepSeek V3 Sparse Attention (MLA)",
                model_config=ds_config,
            )
        )

        # Step 5: mHC (Manifold Hyper-Connections)
        mhc_config = self._create_base_config()
        mhc_config.model_name = "LLM-1B-mHC"
        mhc_config.connection.connection_type = ConnectionType.MHC
        mhc_config.connection.mhc_expansion_rate = 4.0
        mhc_config.connection.mhc_num_connections = 2
        experiments.append(
            ExperimentConfig(
                name="step5_mhc",
                description="Model with Manifold Hyper-Connections (paper 2512.24880)",
                model_config=mhc_config,
            )
        )

        # Step 6: Multi-Token Prediction
        mtp_config = self._create_base_config()
        mtp_config.model_name = "LLM-1B-MTP"
        mtp_config.head.use_multi_token_prediction = True
        mtp_config.head.num_predict_tokens = 4
        mtp_config.head.mtp_loss_weight = 0.3
        experiments.append(
            ExperimentConfig(
                name="step6_mtp",
                description="Model with Multi-Token Prediction (DeepSeek style)",
                model_config=mtp_config,
            )
        )

        # Step 7: YaRN for extended context
        yarn_config = self._create_base_config()
        yarn_config.model_name = "LLM-1B-YaRN"
        yarn_config.max_position_embeddings = 16384  # Extended from 4096
        yarn_config.position.position_type = PositionEmbeddingType.YARN
        yarn_config.position.yarn_original_max_position = 4096
        yarn_config.position.yarn_scale = 4.0
        experiments.append(
            ExperimentConfig(
                name="step7_yarn",
                description="Model with YaRN for extended context (4k->16k)",
                model_config=yarn_config,
            )
        )

        # Combination experiments

        # GSA + mHC
        gsa_mhc_config = self._create_base_config()
        gsa_mhc_config.model_name = "LLM-1B-GSA-mHC"
        gsa_mhc_config.attention.attention_type = AttentionType.GATED_SPARSE
        gsa_mhc_config.connection.connection_type = ConnectionType.MHC
        experiments.append(
            ExperimentConfig(
                name="combo_gsa_mhc",
                description="GSA attention + mHC connections",
                model_config=gsa_mhc_config,
            )
        )

        # DeepSeek + MTP
        ds_mtp_config = self._create_base_config()
        ds_mtp_config.model_name = "LLM-1B-DS-MTP"
        ds_mtp_config.attention.attention_type = AttentionType.DEEPSEEK_SPARSE
        ds_mtp_config.head.use_multi_token_prediction = True
        ds_mtp_config.head.num_predict_tokens = 4
        experiments.append(
            ExperimentConfig(
                name="combo_deepseek_mtp",
                description="DeepSeek Sparse + Multi-Token Prediction",
                model_config=ds_mtp_config,
            )
        )

        # Full combination: GSA + mHC + MTP + YaRN
        full_config = self._create_base_config()
        full_config.model_name = "LLM-1B-Full"
        full_config.attention.attention_type = AttentionType.GATED_SPARSE
        full_config.attention.gsa_num_slots = 64
        full_config.connection.connection_type = ConnectionType.MHC
        full_config.head.use_multi_token_prediction = True
        full_config.head.num_predict_tokens = 4
        full_config.max_position_embeddings = 16384
        full_config.position.position_type = PositionEmbeddingType.YARN
        full_config.position.yarn_original_max_position = 4096
        experiments.append(
            ExperimentConfig(
                name="combo_full",
                description="Full combination: GSA + mHC + MTP + YaRN",
                model_config=full_config,
            )
        )

        return experiments

    def _create_base_config(self) -> ModelConfig:
        """Create base 1B model configuration."""
        return ModelConfig(
            model_name="LLM-1B-Base",
            model_version="1.0.0",
            vocab_size=50304,  # Divisible by 64
            hidden_size=2048,
            num_hidden_layers=24,
            max_position_embeddings=4096,
            rms_norm_eps=1e-6,
            initializer_range=0.02,
            attention=AttentionConfig(
                attention_type=AttentionType.GROUPED_QUERY,
                num_attention_heads=16,
                num_key_value_heads=4,  # GQA ratio 4:1
                head_dim=128,  # 2048 / 16 = 128
                attention_dropout=0.0,
                attention_bias=False,
            ),
            position=PositionConfig(
                position_type=PositionEmbeddingType.ROPE, rope_theta=10000.0
            ),
            ffn=FFNConfig(
                ffn_type=FFNType.SWIGLU,
                intermediate_size=5504,  # ~2.7x hidden for SwiGLU
                ffn_dropout=0.0,
                ffn_bias=False,
            ),
            connection=ConnectionConfig(connection_type=ConnectionType.RESIDUAL),
            head=HeadConfig(use_multi_token_prediction=False, tie_word_embeddings=True),
        )

    def run_experiment(self, experiment: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment."""
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {experiment.name}")
        print(f"Description: {experiment.description}")
        print(f"{'='*70}\n")

        # Create model
        model = create_model_from_config(experiment.model_config)

        # Create dataset
        dataset = RandomTextDataset(
            vocab_size=experiment.model_config.vocab_size,
            seq_length=self.seq_length,
            num_samples=self.training_steps * self.batch_size * 2,
            seed=self.seed,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        # Training config
        training_config = TrainingConfig(
            max_steps=self.training_steps,
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            experiment_name=experiment.name,
            checkpoint_dir=str(self.output_dir / experiment.name),
            seed=self.seed,
            log_interval=100,
            save_interval=self.training_steps,  # Only save at end
        )

        # Train
        trainer = Trainer(
            model=model,
            train_dataloader=dataloader,
            training_config=training_config,
            model_config=experiment.model_config,
        )

        start_time = time.time()
        final_metrics = trainer.train()
        total_time = time.time() - start_time

        # Calculate average tokens/sec from logs
        avg_tps = final_metrics.tokens_seen / total_time

        # Create result
        config_summary = {
            "attention": experiment.model_config.attention.attention_type.value,
            "connection": experiment.model_config.connection.connection_type.value,
            "position": experiment.model_config.position.position_type.value,
            "mtp": str(experiment.model_config.head.use_multi_token_prediction),
            "max_pos": str(experiment.model_config.max_position_embeddings),
        }

        result = ExperimentResult(
            name=experiment.name,
            final_loss=final_metrics.loss,
            best_loss=trainer.best_loss,
            avg_tokens_per_second=avg_tps,
            total_time_seconds=total_time,
            tokens_seen=final_metrics.tokens_seen,
            parameters=getattr(
                model, "num_parameters", sum(p.numel() for p in model.parameters())
            ),
            config_summary=config_summary,
        )

        self.results.append(result)

        # Clean up
        del model
        del trainer
        torch.cuda.empty_cache()

        return result

    def run_all(self, experiments: Optional[List[ExperimentConfig]] = None):
        """Run all experiments."""
        if experiments is None:
            experiments = self.create_experiments()

        print(f"\n{'#'*70}")
        print(f"# EXPERIMENT SUITE: {len(experiments)} experiments")
        print(f"# Training steps per experiment: {self.training_steps}")
        print(f"# Output directory: {self.output_dir}")
        print(f"{'#'*70}\n")

        for i, experiment in enumerate(experiments):
            print(f"\n[{i+1}/{len(experiments)}] Running: {experiment.name}")

            try:
                result = self.run_experiment(experiment)
                print(
                    f"  ✓ Complete: Loss={result.best_loss:.4f}, "
                    f"Tok/s={result.avg_tokens_per_second:,.0f}"
                )
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                import traceback

                traceback.print_exc()

        # Generate report
        self.generate_report()

    def generate_report(self):
        """Generate comparison report."""
        report_path = self.output_dir / f"experiment_report_{self.timestamp}.md"

        report = []
        report.append("# LLM Architecture Experiment Report")
        report.append(f"\nGenerated: {datetime.now().isoformat()}")
        report.append(f"\nTraining steps per experiment: {self.training_steps}")
        report.append(f"Batch size: {self.batch_size}")
        report.append(f"Sequence length: {self.seq_length}")

        report.append("\n## Results Summary\n")
        report.append(
            "| Experiment | Loss | Tok/s | Params | Attention | Connection | MTP |"
        )
        report.append(
            "|------------|------|-------|--------|-----------|------------|-----|"
        )

        # Sort by loss
        sorted_results = sorted(self.results, key=lambda x: x.best_loss)

        for r in sorted_results:
            report.append(
                f"| {r.name} | {r.best_loss:.4f} | {r.avg_tokens_per_second:,.0f} | "
                f"{r.parameters/1e9:.2f}B | {r.config_summary['attention']} | "
                f"{r.config_summary['connection']} | {r.config_summary['mtp']} |"
            )

        report.append("\n## Best Configuration\n")
        best = sorted_results[0]
        report.append(f"**{best.name}** achieved the best loss of {best.best_loss:.4f}")
        report.append("\nConfiguration:")
        for k, v in best.config_summary.items():
            report.append(f"- {k}: {v}")

        report.append("\n## Detailed Results\n")
        for r in self.results:
            report.append(f"### {r.name}")
            report.append(f"- Final Loss: {r.final_loss:.4f}")
            report.append(f"- Best Loss: {r.best_loss:.4f}")
            report.append(f"- Tokens/sec: {r.avg_tokens_per_second:,.0f}")
            report.append(f"- Total time: {r.total_time_seconds:.1f}s")
            report.append(f"- Parameters: {r.parameters:,}")
            report.append("")

        # Write report
        with open(report_path, "w") as f:
            f.write("\n".join(report))

        print(f"\n📊 Report saved to: {report_path}")

        # Also save JSON
        json_path = self.output_dir / f"experiment_results_{self.timestamp}.json"
        with open(json_path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        print(f"📊 Results saved to: {json_path}")


# Need to import these for _create_base_config


def main():
    parser = argparse.ArgumentParser(description="Run LLM architecture experiments")

    parser.add_argument("--output-dir", type=str, default="./experiments")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        help="Specific experiments to run (default: all)",
    )

    args = parser.parse_args()

    runner = ExperimentRunner(
        output_dir=args.output_dir,
        training_steps=args.steps,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        seed=args.seed,
    )

    experiments = runner.create_experiments()

    if args.experiments:
        experiments = [e for e in experiments if e.name in args.experiments]

    runner.run_all(experiments)


if __name__ == "__main__":
    main()
