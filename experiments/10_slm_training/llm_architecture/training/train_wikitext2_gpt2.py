"""
Quick Wikitext-2 training using GPT-2 tokenizer.

This script is a lightweight alternative to training/train.py so you can
smoke-test the model while a custom tokenizer is being built.

Supports two configuration modes:
1. Preset mode: --preset 1b-base (uses Python preset configs)
2. YAML mode: --config configs/1b_base.yaml (uses YAML config files)

CLI arguments always override config file values.
"""

import argparse
import random

# Add repo root to path
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import PRESET_CONFIGS, ModelConfig, get_preset_config
from models.llm import create_model_from_config
from training.train import Trainer, TrainingConfig


class SystemMonitor:
    """Monitor GPU and CPU resource usage during training."""

    def __init__(self):
        # GPU tracking
        self.gpu_memory_allocated_samples = []
        self.gpu_memory_reserved_samples = []
        self.gpu_utilization_samples = []
        self.has_gpu = torch.cuda.is_available()

        # CPU tracking
        self.cpu_percent_samples = []
        self.cpu_memory_percent_samples = []
        self.cpu_memory_gb_samples = []

        # Try to import psutil for CPU monitoring
        try:
            import psutil

            self.psutil_available = True
            self.process = psutil.Process()

            # Detect container memory limit (cgroup v1 and v2)
            container_memory_gb = None
            try:
                # Try cgroup v2 first (newer Docker/Kubernetes)
                with open("/sys/fs/cgroup/memory.max", "r") as f:
                    limit = f.read().strip()
                    if limit != "max":
                        container_memory_gb = int(limit) / (1024**3)
            except (FileNotFoundError, ValueError, PermissionError):
                try:
                    # Try cgroup v1 (older systems)
                    with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
                        limit = int(f.read().strip())
                        # Ignore if set to max value (not actually limited)
                        if limit < (1 << 63):
                            container_memory_gb = limit / (1024**3)
                except (FileNotFoundError, ValueError, PermissionError):
                    pass

            # Use container limit if available, otherwise system total
            if container_memory_gb:
                self.system_total_memory_gb = container_memory_gb
                self.is_containerized = True
            else:
                self.system_total_memory_gb = psutil.virtual_memory().total / (1024**3)
                self.is_containerized = False

        except ImportError:
            self.psutil_available = False
            self.system_total_memory_gb = 0
            self.is_containerized = False
            print("\n" + "=" * 60)
            print("⚠️  CPU Monitoring Unavailable")
            print("=" * 60)
            print("psutil is not installed. Only GPU tracking is available.")
            print("To enable CPU monitoring, run:")
            print("  pip install psutil")
            print("=" * 60 + "\n")

        if self.has_gpu:
            # Reset peak memory stats at start
            torch.cuda.reset_peak_memory_stats()

            try:
                import pynvml

                pynvml.nvmlInit()
                self.nvml_available = True
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

                # Get GPU name
                self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
                if isinstance(self.gpu_name, bytes):
                    self.gpu_name = self.gpu_name.decode("utf-8")

                # Get total memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                self.total_gpu_memory_gb = mem_info.total / (1024**3)

            except (ImportError, Exception):
                self.nvml_available = False
                self.gpu_name = torch.cuda.get_device_name(0)
                self.total_gpu_memory_gb = torch.cuda.get_device_properties(
                    0
                ).total_memory / (1024**3)
                print("\n" + "=" * 60)
                print("⚠️  GPU Utilization Tracking Unavailable")
                print("=" * 60)
                print("pynvml is not installed. Only GPU memory tracking is available.")
                print("To enable GPU utilization tracking, run:")
                print("  pip install nvidia-ml-py3")
                print("=" * 60 + "\n")
        else:
            self.nvml_available = False
            self.gpu_name = "No GPU"
            self.total_gpu_memory_gb = 0

    def sample(self):
        """Collect current GPU and CPU metrics."""
        # GPU metrics
        if self.has_gpu:
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB

            self.gpu_memory_allocated_samples.append(memory_allocated)
            self.gpu_memory_reserved_samples.append(memory_reserved)

            # GPU utilization (requires pynvml)
            if self.nvml_available:
                try:
                    import pynvml

                    utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    self.gpu_utilization_samples.append(utilization.gpu)
                except Exception:
                    pass

        # CPU metrics (system-wide to capture DataLoader workers)
        if self.psutil_available:
            try:
                import psutil

                # System-wide CPU utilization (captures all processes including workers)
                cpu_percent = psutil.cpu_percent(interval=None)
                if cpu_percent > 0:  # Skip initial 0.0% reading
                    self.cpu_percent_samples.append(cpu_percent)

                # Process memory usage (main process only)
                mem_info = self.process.memory_info()
                mem_gb = mem_info.rss / (1024**3)  # Resident set size in GB
                self.cpu_memory_gb_samples.append(mem_gb)

                # System memory percent
                sys_mem = psutil.virtual_memory()
                self.cpu_memory_percent_samples.append(sys_mem.percent)
            except Exception:
                pass

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics summary."""
        stats = {}

        # GPU stats
        if self.has_gpu:
            stats["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / (
                1024**3
            )
            stats["max_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / (
                1024**3
            )
            stats["total_gpu_memory_gb"] = self.total_gpu_memory_gb
            stats["gpu_name"] = self.gpu_name

        if self.gpu_memory_allocated_samples:
            stats["gpu_memory_allocated_max"] = max(self.gpu_memory_allocated_samples)
            stats["gpu_memory_allocated_min"] = min(self.gpu_memory_allocated_samples)
            stats["gpu_memory_allocated_mean"] = sum(
                self.gpu_memory_allocated_samples
            ) / len(self.gpu_memory_allocated_samples)

        if self.gpu_memory_reserved_samples:
            stats["gpu_memory_reserved_max"] = max(self.gpu_memory_reserved_samples)
            stats["gpu_memory_reserved_min"] = min(self.gpu_memory_reserved_samples)
            stats["gpu_memory_reserved_mean"] = sum(
                self.gpu_memory_reserved_samples
            ) / len(self.gpu_memory_reserved_samples)

        if self.gpu_utilization_samples:
            stats["gpu_utilization_max"] = max(self.gpu_utilization_samples)
            stats["gpu_utilization_min"] = min(self.gpu_utilization_samples)
            stats["gpu_utilization_mean"] = sum(self.gpu_utilization_samples) / len(
                self.gpu_utilization_samples
            )

        # CPU stats
        if self.cpu_percent_samples:
            stats["cpu_percent_max"] = max(self.cpu_percent_samples)
            stats["cpu_percent_min"] = min(self.cpu_percent_samples)
            stats["cpu_percent_mean"] = sum(self.cpu_percent_samples) / len(
                self.cpu_percent_samples
            )

        if self.cpu_memory_gb_samples:
            stats["cpu_memory_gb_max"] = max(self.cpu_memory_gb_samples)
            stats["cpu_memory_gb_min"] = min(self.cpu_memory_gb_samples)
            stats["cpu_memory_gb_mean"] = sum(self.cpu_memory_gb_samples) / len(
                self.cpu_memory_gb_samples
            )

        if self.cpu_memory_percent_samples:
            stats["system_memory_percent_max"] = max(self.cpu_memory_percent_samples)
            stats["system_memory_percent_min"] = min(self.cpu_memory_percent_samples)
            stats["system_memory_percent_mean"] = sum(
                self.cpu_memory_percent_samples
            ) / len(self.cpu_memory_percent_samples)

        if self.psutil_available:
            stats["system_total_memory_gb"] = self.system_total_memory_gb

        return stats

    def cleanup(self):
        """Cleanup NVML resources."""
        if self.nvml_available:
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                pass

    def print_summary(
        self,
        seq_length: Optional[int] = None,
        model_config: Optional[ModelConfig] = None,
    ):
        """Print system statistics summary with configuration details."""
        stats = self.get_stats()

        print(f"\n{'='*60}")
        print("Training Summary")
        print(f"{'='*60}")

        # Configuration details
        if seq_length is not None:
            print(f"Sequence Length: {seq_length}")

        if model_config is not None:
            print("\nModel Configuration:")
            print(f"  Model Name: {model_config.model_name}")
            print(f"  Vocab Size: {model_config.vocab_size:,}")
            print(f"  Hidden Size: {model_config.hidden_size}")
            print(f"  Num Layers: {model_config.num_hidden_layers}")
            print(f"  Num Heads: {model_config.attention.num_attention_heads}")
            print(f"  Max Position: {model_config.max_position_embeddings}")

            # Attention config
            if hasattr(model_config, "attention"):
                att = model_config.attention
                print(
                    f"\n  Attention Type: {att.attention_type.value if hasattr(att.attention_type, 'value') else att.attention_type}"
                )
                if att.attention_type.value in ["gated_sparse", "deepseek_gsa"]:
                    print(f"    GSA k_base: {att.gsa_k_base}")
                    print(f"    GSA k_max: {att.gsa_k_max}")
                    if hasattr(att, "gsa_r_base"):
                        print(f"    GSA r_base: {att.gsa_r_base}")
                    print(f"    GSA use_triton: {att.gsa_use_triton_kernels}")

            # FFN config
            if hasattr(model_config, "ffn"):
                ffn = model_config.ffn
                ffn_type = (
                    ffn.ffn_type.value
                    if hasattr(ffn.ffn_type, "value")
                    else ffn.ffn_type
                )
                print(f"\n  FFN Type: {ffn_type}")
                print(f"    Intermediate Size: {ffn.intermediate_size}")
                if ffn_type == "moe":
                    print(f"    MoE Num Experts: {ffn.moe_num_experts}")
                    print(f"    MoE Top-K: {ffn.moe_num_experts_per_tok}")

            # Position encoding
            if hasattr(model_config, "position"):
                pos = model_config.position
                pos_type = (
                    pos.position_type.value
                    if hasattr(pos.position_type, "value")
                    else pos.position_type
                )
                print(f"\n  Position Encoding: {pos_type}")
                if pos_type == "rope":
                    print(f"    RoPE Base: {pos.rope_theta}")
                elif pos_type == "yarn":
                    print(f"    YaRN Scale: {pos.yarn_scale}")
                    print(
                        f"    YaRN Original Max Pos: {pos.yarn_original_max_position}"
                    )

        # GPU Statistics
        if stats and "gpu_name" in stats:
            print(f"\n{'─'*60}")
            print("GPU Statistics:")
            print(f"{'─'*60}")

            print(f"  Device: {stats['gpu_name']}")
            print(f"  Total Memory: {stats['total_gpu_memory_gb']:.2f} GB")

            # Peak memory usage
            if "max_memory_allocated_gb" in stats:
                print("\n  Peak Memory Usage:")
                print(f"    Allocated: {stats['max_memory_allocated_gb']:.2f} GB")
                print(f"    Reserved:  {stats['max_memory_reserved_gb']:.2f} GB")
                if "total_gpu_memory_gb" in stats and stats["total_gpu_memory_gb"] > 0:
                    utilization = (
                        stats["max_memory_reserved_gb"] / stats["total_gpu_memory_gb"]
                    ) * 100
                    print(f"    Utilization: {utilization:.1f}%")

            # Sampled memory statistics (allocated)
            if "gpu_memory_allocated_max" in stats:
                print("\n  Memory Allocated Over Time (GB):")
                print(f"    Max:  {stats['gpu_memory_allocated_max']:.2f}")
                print(f"    Min:  {stats['gpu_memory_allocated_min']:.2f}")
                print(f"    Mean: {stats['gpu_memory_allocated_mean']:.2f}")

            # Sampled memory statistics (reserved)
            if "gpu_memory_reserved_max" in stats:
                print("\n  Memory Reserved Over Time (GB):")
                print(f"    Max:  {stats['gpu_memory_reserved_max']:.2f}")
                print(f"    Min:  {stats['gpu_memory_reserved_min']:.2f}")
                print(f"    Mean: {stats['gpu_memory_reserved_mean']:.2f}")

            # GPU utilization
            if "gpu_utilization_max" in stats:
                print("\n  GPU Utilization Over Time (%):")
                print(f"    Max:  {stats['gpu_utilization_max']}")
                print(f"    Min:  {stats['gpu_utilization_min']}")
                print(f"    Mean: {stats['gpu_utilization_mean']:.1f}")
            else:
                print("\n  GPU Utilization: Not available (install nvidia-ml-py3)")

        # CPU Statistics
        if stats and ("cpu_percent_max" in stats or "cpu_memory_gb_max" in stats):
            print(f"\n{'─'*60}")
            print("CPU Statistics:")
            print(f"{'─'*60}")

            if "system_total_memory_gb" in stats:
                memory_label = (
                    "Container Memory"
                    if self.is_containerized
                    else "System Total Memory"
                )
                print(f"  {memory_label}: {stats['system_total_memory_gb']:.2f} GB")

            # CPU utilization
            if "cpu_percent_max" in stats:
                print("\n  System-Wide CPU Utilization (%):")
                print(f"    Max:  {stats['cpu_percent_max']:.1f}")
                print(f"    Min:  {stats['cpu_percent_min']:.1f}")
                print(f"    Mean: {stats['cpu_percent_mean']:.1f}")

            # Process memory usage
            if "cpu_memory_gb_max" in stats:
                print("\n  Process Memory Usage (GB):")
                print(f"    Max:  {stats['cpu_memory_gb_max']:.2f}")
                print(f"    Min:  {stats['cpu_memory_gb_min']:.2f}")
                print(f"    Mean: {stats['cpu_memory_gb_mean']:.2f}")

            # System memory utilization
            if "system_memory_percent_max" in stats:
                print("\n  System Memory Utilization (%):")
                print(f"    Max:  {stats['system_memory_percent_max']:.1f}")
                print(f"    Min:  {stats['system_memory_percent_min']:.1f}")
                print(f"    Mean: {stats['system_memory_percent_mean']:.1f}")
        elif not self.psutil_available:
            print(f"\n{'─'*60}")
            print("CPU Statistics: Not available (install psutil)")
            print(f"{'─'*60}")

        print(f"\n{'='*60}\n")


try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError as exc:
    raise ImportError(
        "Missing dependencies. Install with: pip install datasets transformers"
    ) from exc


def get_optimal_num_workers() -> int:
    """
    Determine optimal number of DataLoader workers based on CPU cores.

    Returns:
        Recommended number of workers (typically 75% of available cores)
    """
    import os

    try:
        # Try to get CPU count
        cpu_count = os.cpu_count()
        if cpu_count is None:
            return 4  # Fallback default

        # Use 75% of cores, min 4, max 16
        optimal = max(4, min(16, int(cpu_count * 0.75)))
        return optimal
    except Exception:
        return 4  # Safe fallback


def load_config_from_yaml(config_path: str) -> Tuple[ModelConfig, Dict[str, Any]]:
    """
    Load model and training config from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Tuple of (ModelConfig, training_config_dict)
    """
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    # Extract training config (if present)
    training_data = config_data.pop("training", {})

    # Load model config
    model_config = ModelConfig.from_dict(config_data)

    return model_config, training_data


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def align_model_context_to_seq_length(
    model_config: ModelConfig, seq_length: int
) -> None:
    """
    Ensure model positional capacity matches requested training sequence length.

    This prevents out-of-bounds indexing in RoPE/YaRN caches when training with
    seq_length > model_config.max_position_embeddings.
    """
    if seq_length <= model_config.max_position_embeddings:
        return

    old_max = model_config.max_position_embeddings
    model_config.max_position_embeddings = seq_length
    print(
        f"[Context] Increasing max_position_embeddings: {old_max} -> {seq_length} "
        f"to match training seq_length."
    )

    # If YaRN is used, ensure configured scale can cover seq_length.
    pos = getattr(model_config, "position", None)
    if pos is None:
        return

    pos_type = getattr(pos, "position_type", None)
    pos_type_value = pos_type.value if hasattr(pos_type, "value") else str(pos_type)
    if pos_type_value != "yarn":
        return

    original_max = int(getattr(pos, "yarn_original_max_position", old_max) or old_max)
    original_max = max(1, original_max)
    required_scale = seq_length / float(original_max)
    current_scale = float(getattr(pos, "yarn_scale", 1.0))
    if required_scale > current_scale:
        pos.yarn_scale = required_scale
        print(
            f"[Context] Increasing YaRN scale: {current_scale:.4g} -> {required_scale:.4g} "
            f"(original_max={original_max}, target_seq={seq_length})."
        )


class TokenBlockDataset(Dataset):
    """Simple fixed-length token block dataset."""

    def __init__(self, token_ids: List[int], seq_length: int, stride: int):
        self.token_ids = token_ids
        self.seq_length = seq_length
        self.stride = stride
        if len(token_ids) < seq_length:
            self.num_blocks = 0
        else:
            self.num_blocks = (len(token_ids) - seq_length) // stride + 1

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, idx: int):
        start = idx * self.stride
        end = start + self.seq_length
        block = self.token_ids[start:end]
        input_ids = torch.tensor(block, dtype=torch.long)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


def build_token_ids(
    split: str, tokenizer, add_eos: bool = True, max_tokens: Optional[int] = None
) -> List[int]:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    token_ids: List[int] = []
    eos_id = tokenizer.eos_token_id

    for text in dataset["text"]:
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        token_ids.extend(ids)
        if add_eos and eos_id is not None:
            token_ids.append(eos_id)
        if max_tokens is not None and len(token_ids) >= max_tokens:
            token_ids = token_ids[:max_tokens]
            break

    return token_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 1B LLM on WikiText-2 with GPT-2 tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using YAML config file (recommended)
  python train_wikitext2_gpt2.py --config ../configs/1b_deepseek_gsa.yaml
  python train_wikitext2_gpt2.py --config ../configs/1b_base.yaml --batch-size 4
  
  # Using preset (legacy mode)
  python train_wikitext2_gpt2.py --preset 1b-base --device cuda
  
  # YAML config with CLI overrides
  python train_wikitext2_gpt2.py --config ../configs/1b_gsa.yaml --seq-length 512 --max-steps 500

Note: CLI arguments always override config file values.
        """,
    )

    # Configuration source
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g., configs/1b_base.yaml). Takes precedence over --preset.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="1b-base",
        choices=list(PRESET_CONFIGS.keys()),
        help="Model preset (used if --config not provided)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Hugging Face tokenizer name (default: gpt2)",
    )

    # Dataset
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="WikiText-2 split",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=None,
        help="Sequence length (overrides config)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride between blocks (default: seq-length)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Cap total tokens for a tiny smoke test",
    )

    # Training (can override config file)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")

    # Device selection
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use: auto (best available), cuda, mps (Apple Silicon), or cpu",
    )

    # GSA-specific overrides for memory tuning
    parser.add_argument(
        "--gsa-k-base",
        type=int,
        default=None,
        help="Override GSA k_base (default: from preset, try 256-512 for long sequences)",
    )
    parser.add_argument(
        "--gsa-k-max",
        type=int,
        default=None,
        help="Override GSA k_max (default: from preset, try 512-1024 for long sequences)",
    )
    parser.add_argument(
        "--no-triton",
        action="store_true",
        help="Disable Triton kernels (use PyTorch fallback for GSA)",
    )

    # torch.compile
    parser.add_argument(
        "--use-torch-compile",
        action="store_true",
        help="Enable torch.compile() for graph optimization (requires PyTorch 2.0+)",
    )
    parser.add_argument(
        "--torch-compile-mode",
        type=str,
        default=None,
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        help="torch.compile mode: default (safe), max-autotune-no-cudagraphs (recommended), max-autotune (+ CUDAGraphs), reduce-overhead (CUDAGraphs only)",
    )
    parser.add_argument(
        "--torch-compile-fullgraph",
        action="store_true",
        help="Enforce full-graph compilation (no graph breaks)",
    )
    parser.add_argument(
        "--torch-compile-dynamic",
        action="store_true",
        help="Enable dynamic shapes in torch.compile",
    )

    # Experiment
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # Logging / DataLoader optimization
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of DataLoader workers (default: auto-detect based on CPU cores)",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Number of batches to prefetch per worker (default: 2)",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader workers alive between epochs (recommended for multi-epoch training)",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    # Load configuration
    if args.config:
        # YAML config mode
        print(f"Loading configuration from: {args.config}")
        model_config, training_dict = load_config_from_yaml(args.config)

        # Build training config from YAML
        training_config = TrainingConfig(
            max_steps=training_dict.get("max_steps", 200),
            batch_size=training_dict.get("batch_size", 2),
            gradient_accumulation_steps=training_dict.get(
                "gradient_accumulation_steps", 1
            ),
            seq_length=training_dict.get("seq_length", 256),
            learning_rate=training_dict.get("learning_rate", 3e-4),
            warmup_steps=training_dict.get("warmup_steps", 20),
            device=training_dict.get("device", "auto"),
            experiment_name=training_dict.get("experiment_name", "wikitext2_gpt2"),
            checkpoint_dir=training_dict.get(
                "checkpoint_dir", "./checkpoints/wikitext2_gpt2"
            ),
            seed=args.seed,
            log_interval=training_dict.get("log_interval", 10),
            save_interval=training_dict.get("save_interval", 200),
            use_amp=training_dict.get("use_amp", True),
            use_torch_compile=training_dict.get("use_torch_compile", False),
            torch_compile_mode=training_dict.get(
                "torch_compile_mode", "max-autotune-no-cudagraphs"
            ),
            torch_compile_fullgraph=training_dict.get("torch_compile_fullgraph", False),
            torch_compile_dynamic=training_dict.get("torch_compile_dynamic", False),
            torch_compile_backend=training_dict.get(
                "torch_compile_backend", "inductor"
            ),
        )
    else:
        # Preset mode (legacy)
        print(f"Using preset: {args.preset}")
        model_config = get_preset_config(args.preset)
        training_config = TrainingConfig(
            max_steps=200,
            batch_size=2,
            gradient_accumulation_steps=1,
            seq_length=256,
            learning_rate=3e-4,
            warmup_steps=20,
            device="auto",
            experiment_name="wikitext2_gpt2",
            checkpoint_dir="./checkpoints/wikitext2_gpt2",
            seed=args.seed,
            log_interval=10,
            save_interval=200,
        )

    # CLI overrides (only if explicitly provided)
    if args.max_steps is not None:
        training_config.max_steps = args.max_steps
    if args.batch_size is not None:
        training_config.batch_size = args.batch_size
    if args.gradient_accumulation is not None:
        training_config.gradient_accumulation_steps = args.gradient_accumulation
    if args.seq_length is not None:
        training_config.seq_length = args.seq_length
    if args.learning_rate is not None:
        training_config.learning_rate = args.learning_rate
    if args.warmup_steps is not None:
        training_config.warmup_steps = args.warmup_steps
    if args.device is not None:
        training_config.device = args.device
    if args.experiment_name is not None:
        training_config.experiment_name = args.experiment_name
    if args.checkpoint_dir is not None:
        training_config.checkpoint_dir = args.checkpoint_dir
    if args.log_interval is not None:
        training_config.log_interval = args.log_interval
    if args.save_interval is not None:
        training_config.save_interval = args.save_interval
    if args.no_amp:
        training_config.use_amp = False
    if args.use_torch_compile:
        training_config.use_torch_compile = True
    if args.torch_compile_mode is not None:
        training_config.torch_compile_mode = args.torch_compile_mode
    if args.torch_compile_fullgraph:
        training_config.torch_compile_fullgraph = True
    if args.torch_compile_dynamic:
        training_config.torch_compile_dynamic = True

    # Keep model positional capacity aligned with requested training length.
    align_model_context_to_seq_length(model_config, training_config.seq_length)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_ids = build_token_ids(
        split=args.dataset_split,
        tokenizer=tokenizer,
        add_eos=True,
        max_tokens=args.max_tokens,
    )

    stride = training_config.seq_length if args.stride is None else args.stride
    dataset = TokenBlockDataset(
        token_ids, seq_length=training_config.seq_length, stride=stride
    )
    if len(dataset) == 0:
        raise ValueError(
            "Not enough tokens for the chosen seq-length. "
            "Reduce --seq-length or increase --max-tokens."
        )

    # Determine if we should pin memory (only for CUDA)
    pin_memory = training_config.device == "cuda" or (
        training_config.device == "auto" and torch.cuda.is_available()
    )

    # Auto-detect optimal number of workers if not specified
    num_workers = (
        args.num_workers if args.num_workers is not None else get_optimal_num_workers()
    )

    # Persistent workers only makes sense with num_workers > 0
    use_persistent_workers = args.persistent_workers and num_workers > 0

    print(f"\n{'='*60}")
    print("DataLoader Configuration")
    print(f"{'='*60}")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Prefetch factor: {args.prefetch_factor}")
    print(f"  Persistent workers: {use_persistent_workers}")
    print(f"  Pin memory: {pin_memory}")
    print(f"{'='*60}\n")

    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
        persistent_workers=use_persistent_workers,
    )

    # Update vocab size from tokenizer
    model_config.vocab_size = len(tokenizer)

    # Override GSA k values if provided (for memory tuning)
    if args.gsa_k_base is not None:
        model_config.attention.gsa_k_base = args.gsa_k_base
        print(f"[GSA] Overriding k_base to {args.gsa_k_base}")
    if args.gsa_k_max is not None:
        model_config.attention.gsa_k_max = args.gsa_k_max
        print(f"[GSA] Overriding k_max to {args.gsa_k_max}")
    if args.no_triton:
        model_config.attention.gsa_use_triton_kernels = False
        print("[GSA] Triton kernels disabled, using PyTorch fallback")

    # Training config
    # training_config = TrainingConfig(
    #     max_steps=args.max_steps,
    #     batch_size=args.batch_size,
    #     gradient_accumulation_steps=args.gradient_accumulation,
    #     seq_length=args.seq_length,
    #     learning_rate=args.learning_rate,
    #     warmup_steps=args.warmup_steps,
    #     device=args.device,
    #     experiment_name=args.experiment_name,
    #     checkpoint_dir=args.checkpoint_dir,
    #     seed=args.seed,
    #     log_interval=args.log_interval,
    #     save_interval=args.save_interval,
    #     use_amp=not args.no_amp
    # )

    model = create_model_from_config(model_config, tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        training_config=training_config,
        model_config=model_config,
    )

    # Initialize system monitor (GPU + CPU)
    system_monitor = SystemMonitor()

    # Monkey-patch the trainer's _print_progress to also sample system metrics
    original_print_progress = trainer._print_progress

    def _print_progress_with_monitoring(metrics):
        system_monitor.sample()
        original_print_progress(metrics)

    trainer._print_progress = _print_progress_with_monitoring

    # Run training
    try:
        trainer.train()
    finally:
        # Print system statistics summary with configuration details
        system_monitor.print_summary(
            seq_length=training_config.seq_length, model_config=model_config
        )
        system_monitor.cleanup()


if __name__ == "__main__":
    main()
