"""
QLoRA Configuration Module
Team 18: SFT, RL-Style Alignment & Final Post-Training Benchmarks

This module provides type-safe configuration management for QLoRA training,
with support for YAML files and CLI argument overrides.
"""

import os
import re
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal, Any, Dict
from pathlib import Path

import torch
import yaml


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "microsoft/phi-2"
    trust_remote_code: bool = True
    torch_dtype: str = "auto"
    device_map: str = "auto"
    attn_implementation: str = "eager"
    max_seq_length: int = 512
    max_prompt_length: int = 256
    max_completion_length: int = 256
    
    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_map = {
            "auto": None,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype, None)


@dataclass
class QuantizationConfig:
    """Quantization configuration for bitsandbytes."""
    enabled: bool = True
    bits: Literal[4, 8] = 4
    quant_type: Literal["nf4", "fp4"] = "nf4"
    compute_dtype: str = "bfloat16"
    double_quant: bool = True
    exclude_modules: List[str] = field(default_factory=lambda: [
        "lm_head", "embed_tokens", ".*layernorm.*", ".*norm.*"
    ])
    modules_to_save: List[str] = field(default_factory=list)
    
    def get_compute_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.compute_dtype, torch.bfloat16)
    
    def to_bnb_config(self):
        """Convert to BitsAndBytesConfig if enabled."""
        if not self.enabled:
            return None
        
        from transformers import BitsAndBytesConfig
        
        if self.bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.quant_type,
                bnb_4bit_compute_dtype=self.get_compute_dtype(),
                bnb_4bit_use_double_quant=self.double_quant,
            )
        elif self.bits == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            raise ValueError(f"Unsupported quantization bits: {self.bits}")
    
    def should_quantize_module(self, module_name: str) -> bool:
        """Check if a module should be quantized based on exclusion patterns."""
        for pattern in self.exclude_modules:
            if re.search(pattern, module_name, re.IGNORECASE):
                return False
        return True


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    def to_peft_config(self):
        """Convert to PEFT LoraConfig."""
        from peft import LoraConfig as PeftLoraConfig
        
        return PeftLoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            bias=self.bias,
            task_type=self.task_type,
            target_modules=self.target_modules,
        )


@dataclass
class GRPOSettings:
    """GRPO-specific training settings."""
    num_generations: int = 4
    beta: float = 0.0
    temperature: float = 0.7
    top_p: float = 0.9
    epsilon: float = 0.2


@dataclass
class DPOSettings:
    """DPO-specific training settings."""
    beta: float = 0.1
    label_smoothing: float = 0.0


@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = "./outputs"
    method: Literal["sft", "grpo", "dpo"] = "sft"
    
    # Batch settings
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    
    # Learning rate
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Duration
    num_train_epochs: int = 1
    max_steps: int = -1
    
    # Precision
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    eval_strategy: str = "steps"
    eval_steps: int = 100
    report_to: str = "none"
    
    # Misc
    seed: int = 42
    dataloader_num_workers: int = 0
    
    # Method-specific settings
    grpo: GRPOSettings = field(default_factory=GRPOSettings)
    dpo: DPOSettings = field(default_factory=DPOSettings)


@dataclass
class DataFilters:
    """Data filtering configuration."""
    language: str = "en"
    min_quality: float = 0.5


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_name: str = "OpenAssistant/oasst1"
    dataset_split: str = "train"
    max_samples: Optional[int] = None
    val_split_ratio: float = 0.1
    text_column: str = "text"
    prompt_template: str = "User: {text}\nAssistant:"
    filters: DataFilters = field(default_factory=DataFilters)


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    auto_detect: bool = True
    fallback_to_cpu: bool = True
    mps_fallback_to_bf16: bool = True
    force_device: Optional[str] = None


@dataclass
class HubConfig:
    """HuggingFace Hub configuration."""
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    private: bool = False


@dataclass
class QLoRAConfig:
    """
    Master configuration combining all sub-configs.
    
    This is the main configuration class that should be used throughout
    the training pipeline.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "QLoRAConfig":
        """
        Load configuration from a YAML file.
        
        Args:
            path: Path to the YAML configuration file
            
        Returns:
            QLoRAConfig instance
        """
        with open(path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        return cls.from_dict(yaml_config)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QLoRAConfig":
        """
        Create configuration from a dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            QLoRAConfig instance
        """
        # Parse nested configurations
        model_config = ModelConfig(**config_dict.get("model", {}))
        quantization_config = QuantizationConfig(**config_dict.get("quantization", {}))
        lora_config = LoRAConfig(**config_dict.get("lora", {}))
        
        # Training config with nested GRPO/DPO settings
        training_dict = config_dict.get("training", {})
        grpo_settings = GRPOSettings(**training_dict.pop("grpo", {}))
        dpo_settings = DPOSettings(**training_dict.pop("dpo", {}))
        training_config = TrainingConfig(**training_dict, grpo=grpo_settings, dpo=dpo_settings)
        
        # Data config with nested filters
        data_dict = config_dict.get("data", {})
        filters = DataFilters(**data_dict.pop("filters", {}))
        data_config = DataConfig(**data_dict, filters=filters)
        
        hardware_config = HardwareConfig(**config_dict.get("hardware", {}))
        hub_config = HubConfig(**config_dict.get("hub", {}))
        
        return cls(
            model=model_config,
            quantization=quantization_config,
            lora=lora_config,
            training=training_config,
            data=data_config,
            hardware=hardware_config,
            hub=hub_config,
        )
    
    @classmethod
    def from_args(cls, args: argparse.Namespace, base_config: Optional["QLoRAConfig"] = None) -> "QLoRAConfig":
        """
        Create configuration from CLI arguments, optionally merging with a base config.
        
        Args:
            args: Parsed CLI arguments
            base_config: Optional base configuration to merge with
            
        Returns:
            QLoRAConfig instance
        """
        if base_config is None:
            config = cls()
        else:
            config = base_config
        
        # Override with CLI arguments if provided
        if hasattr(args, 'model_name') and args.model_name:
            config.model.name = args.model_name
        
        if hasattr(args, 'quantization_bits') and args.quantization_bits is not None:
            if args.quantization_bits == 0:
                config.quantization.enabled = False
            else:
                config.quantization.bits = args.quantization_bits
        
        if hasattr(args, 'no_quantization') and args.no_quantization:
            config.quantization.enabled = False
        
        if hasattr(args, 'quant_type') and args.quant_type:
            config.quantization.quant_type = args.quant_type
        
        if hasattr(args, 'lora_r') and args.lora_r:
            config.lora.r = args.lora_r
        
        if hasattr(args, 'lora_alpha') and args.lora_alpha:
            config.lora.alpha = args.lora_alpha
        
        if hasattr(args, 'lora_target_modules') and args.lora_target_modules:
            config.lora.target_modules = args.lora_target_modules
        
        if hasattr(args, 'learning_rate') and args.learning_rate:
            config.training.learning_rate = args.learning_rate
        
        if hasattr(args, 'max_steps') and args.max_steps:
            config.training.max_steps = args.max_steps
        
        if hasattr(args, 'method') and args.method:
            config.training.method = args.method
        
        if hasattr(args, 'num_generations') and args.num_generations:
            config.training.grpo.num_generations = args.num_generations
        
        if hasattr(args, 'output_dir') and args.output_dir:
            config.training.output_dir = args.output_dir
        
        if hasattr(args, 'device') and args.device:
            config.hardware.force_device = args.device
            config.hardware.auto_detect = False
        
        if hasattr(args, 'push_to_hub') and args.push_to_hub:
            config.hub.push_to_hub = True
        
        if hasattr(args, 'hub_model_id') and args.hub_model_id:
            config.hub.hub_model_id = args.hub_model_id
        
        if hasattr(args, 'dataset_name') and args.dataset_name:
            config.data.dataset_name = args.dataset_name
        
        if hasattr(args, 'max_samples') and args.max_samples:
            config.data.max_samples = args.max_samples
        
        return config
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of warnings.
        
        Returns:
            List of warning messages (empty if no issues)
        """
        warnings = []
        
        # Check quantization compatibility
        if self.quantization.enabled:
            if self.hardware.force_device == "mps":
                warnings.append(
                    "Quantization is enabled but MPS device is selected. "
                    "bitsandbytes has limited MPS support. Consider setting "
                    "quantization.enabled=false or using --no_quantization."
                )
            
            if self.quantization.bits == 4 and self.quantization.compute_dtype == "float16":
                warnings.append(
                    "Using FP16 compute dtype with 4-bit quantization. "
                    "BF16 is recommended for better stability."
                )
        
        # Check LoRA configuration
        if self.lora.r > 256:
            warnings.append(
                f"LoRA rank {self.lora.r} is very high. "
                "Consider using r <= 128 for better efficiency."
            )
        
        if self.lora.alpha < self.lora.r:
            warnings.append(
                f"LoRA alpha ({self.lora.alpha}) < rank ({self.lora.r}). "
                "Typically alpha >= rank. Consider alpha = 2 * r."
            )
        
        # Check training configuration
        if self.training.bf16 and self.training.fp16:
            warnings.append(
                "Both bf16 and fp16 are enabled. Only one should be true."
            )
        
        if self.training.method == "grpo" and self.training.grpo.num_generations < 2:
            warnings.append(
                "GRPO requires at least 2 generations per prompt for "
                "meaningful advantage computation."
            )
        
        return warnings
    
    def auto_configure_hardware(self) -> None:
        """
        Auto-detect hardware and adjust settings accordingly.
        
        This method modifies the configuration in-place based on
        detected hardware capabilities.
        """
        if not self.hardware.auto_detect:
            return
        
        # Detect available hardware
        if torch.cuda.is_available():
            device = "cuda"
            capability = torch.cuda.get_device_capability()
            
            # Ampere or newer (SM 8.0+)
            if capability[0] >= 8:
                # Full 4-bit support, BF16 compute
                self.quantization.compute_dtype = "bfloat16"
                self.training.bf16 = True
                self.training.fp16 = False
            else:
                # Pre-Ampere: limited 4-bit, use FP16
                if self.quantization.bits == 4:
                    print("Warning: Pre-Ampere GPU detected. 4-bit support may be limited.")
                self.quantization.compute_dtype = "float16"
                self.training.bf16 = False
                self.training.fp16 = True
        
        elif torch.backends.mps.is_available():
            device = "mps"
            
            if self.hardware.mps_fallback_to_bf16:
                # Disable quantization on MPS
                print("Apple Silicon detected. Disabling quantization, using BF16.")
                self.quantization.enabled = False
                self.model.torch_dtype = "bfloat16"
                self.model.device_map = "mps"
                self.model.attn_implementation = "eager"  # No Flash Attention on MPS
                self.training.bf16 = True
                self.training.fp16 = False
                self.training.dataloader_num_workers = 0
        
        elif self.hardware.fallback_to_cpu:
            device = "cpu"
            print("No GPU detected. Falling back to CPU (training will be slow).")
            self.quantization.enabled = False
            self.model.torch_dtype = "float32"
            self.model.device_map = "cpu"
            self.training.bf16 = False
            self.training.fp16 = False
        
        else:
            raise RuntimeError("No compatible hardware found and fallback_to_cpu is disabled.")
        
        # Set device if not forced
        if self.hardware.force_device is None:
            self.model.device_map = device if device != "cuda" else "auto"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def print_config(self) -> None:
        """Print configuration summary."""
        print("=" * 70)
        print("QLoRA Training Configuration")
        print("=" * 70)
        
        print(f"\n[Model]")
        print(f"  Name: {self.model.name}")
        print(f"  Device: {self.model.device_map}")
        print(f"  Dtype: {self.model.torch_dtype}")
        
        print(f"\n[Quantization]")
        print(f"  Enabled: {self.quantization.enabled}")
        if self.quantization.enabled:
            print(f"  Bits: {self.quantization.bits}")
            print(f"  Type: {self.quantization.quant_type}")
            print(f"  Compute dtype: {self.quantization.compute_dtype}")
            print(f"  Double quant: {self.quantization.double_quant}")
        
        print(f"\n[LoRA]")
        print(f"  Rank (r): {self.lora.r}")
        print(f"  Alpha: {self.lora.alpha}")
        print(f"  Dropout: {self.lora.dropout}")
        print(f"  Target modules: {self.lora.target_modules}")
        
        print(f"\n[Training]")
        print(f"  Method: {self.training.method}")
        print(f"  Batch size: {self.training.per_device_train_batch_size}")
        print(f"  Gradient accumulation: {self.training.gradient_accumulation_steps}")
        print(f"  Effective batch: {self.training.per_device_train_batch_size * self.training.gradient_accumulation_steps}")
        print(f"  Learning rate: {self.training.learning_rate}")
        print(f"  Max steps: {self.training.max_steps}")
        print(f"  Output dir: {self.training.output_dir}")
        
        print(f"\n[Data]")
        print(f"  Dataset: {self.data.dataset_name}")
        print(f"  Max samples: {self.data.max_samples or 'all'}")
        
        print("=" * 70)


# =============================================================================
# CLI Argument Parser
# =============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for CLI overrides.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="QLoRA Training for SFT/GRPO/DPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Configuration file
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )
    
    # Model settings
    parser.add_argument(
        "--model_name",
        type=str,
        help="HuggingFace model name or path"
    )
    
    # Quantization settings
    parser.add_argument(
        "--quantization_bits",
        type=int,
        choices=[0, 4, 8],
        help="Quantization bits (0 to disable)"
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        choices=["nf4", "fp4"],
        help="Quantization type for 4-bit"
    )
    parser.add_argument(
        "--no_quantization",
        action="store_true",
        help="Disable quantization entirely"
    )
    
    # LoRA settings
    parser.add_argument(
        "--lora_r",
        type=int,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        help="Target modules for LoRA"
    )
    
    # Training settings
    parser.add_argument(
        "--learning_rate", "--lr",
        type=float,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["sft", "grpo", "dpo"],
        help="Training method"
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        help="Number of generations for GRPO"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory"
    )
    
    # Hardware settings
    parser.add_argument(
        "--device",
        type=str,
        help="Device override (cuda, cuda:0, mps, cpu)"
    )
    
    # Data settings
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name from HuggingFace Hub"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Maximum number of training samples"
    )
    
    # Hub settings
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push model to HuggingFace Hub"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="HuggingFace Hub model ID"
    )
    
    return parser


def load_config(args: argparse.Namespace) -> QLoRAConfig:
    """
    Load configuration from YAML file and/or CLI arguments.
    
    Priority: CLI args > Custom YAML > Default config
    
    Args:
        args: Parsed CLI arguments
        
    Returns:
        QLoRAConfig instance
    """
    # Start with defaults or load from YAML
    if args.config:
        config = QLoRAConfig.from_yaml(args.config)
    else:
        # Try to load default config if it exists
        default_path = Path(__file__).parent / "default_config.yaml"
        if default_path.exists():
            config = QLoRAConfig.from_yaml(str(default_path))
        else:
            config = QLoRAConfig()
    
    # Override with CLI arguments
    config = QLoRAConfig.from_args(args, config)
    
    # Auto-configure hardware
    config.auto_configure_hardware()
    
    # Validate
    warnings = config.validate()
    for warning in warnings:
        print(f"Warning: {warning}")
    
    return config


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    # Test configuration loading
    parser = create_argument_parser()
    args = parser.parse_args()
    
    config = load_config(args)
    config.print_config()
    
    # Print any warnings
    warnings = config.validate()
    if warnings:
        print("\nValidation Warnings:")
        for w in warnings:
            print(f"  - {w}")
