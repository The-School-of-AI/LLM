#!/usr/bin/env python3
"""
QLoRA Training Script
Team 18: SFT, RL-Style Alignment & Final Post-Training Benchmarks

This script provides a unified interface for training LLMs using QLoRA with
support for SFT, GRPO, and DPO training methods.

Usage:
    # Basic training with defaults
    python train_qlora.py
    
    # With custom config
    python train_qlora.py --config my_config.yaml
    
    # Override specific parameters
    python train_qlora.py --model_name "meta-llama/Llama-2-7b-hf" --method grpo
    
    # Disable quantization for Apple Silicon
    python train_qlora.py --no_quantization --device mps
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Callable, List, Any

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import get_peft_model, prepare_model_for_kbit_training

# Local imports
from qlora_config import (
    QLoRAConfig,
    create_argument_parser,
    load_config,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Model Loading
# =============================================================================

def load_model_and_tokenizer(config: QLoRAConfig) -> Tuple[Any, Any]:
    """
    Load model and tokenizer with optional quantization and LoRA.
    
    Args:
        config: QLoRA configuration
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {config.model.name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        trust_remote_code=config.model.trust_remote_code,
        padding_side="left",
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Get quantization config
    bnb_config = config.quantization.to_bnb_config()
    
    # Determine torch dtype
    torch_dtype = config.model.get_torch_dtype()
    if torch_dtype is None and bnb_config is None:
        torch_dtype = torch.bfloat16
    
    # Load model
    model_kwargs = {
        "trust_remote_code": config.model.trust_remote_code,
        "device_map": config.model.device_map,
        "attn_implementation": config.model.attn_implementation,
    }
    
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        logger.info(f"Loading with {config.quantization.bits}-bit quantization ({config.quantization.quant_type})")
    else:
        model_kwargs["torch_dtype"] = torch_dtype
        logger.info(f"Loading without quantization (dtype: {torch_dtype})")
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        **model_kwargs,
    )
    
    logger.info(f"Model loaded. Parameters: {model.num_parameters():,}")
    
    # Prepare for k-bit training if quantized
    if bnb_config is not None:
        logger.info("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.training.gradient_checkpointing,
        )
    elif config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Apply LoRA
    logger.info("Applying LoRA adapters...")
    peft_config = config.lora.to_peft_config()
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    trainable_params, total_params = model.get_nb_trainable_parameters()
    logger.info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
    
    return model, tokenizer


# =============================================================================
# Data Preparation
# =============================================================================

def prepare_dataset(
    config: QLoRAConfig,
    tokenizer: Any,
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load and prepare dataset for training.
    
    Args:
        config: QLoRA configuration
        tokenizer: Tokenizer for the model
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    logger.info(f"Loading dataset: {config.data.dataset_name}")
    
    # Load dataset
    dataset = load_dataset(
        config.data.dataset_name,
        split=config.data.dataset_split,
    )
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Apply filters if dataset has the required columns
    if config.data.filters.language and "lang" in dataset.column_names:
        dataset = dataset.filter(lambda x: x.get("lang") == config.data.filters.language)
        logger.info(f"After language filter: {len(dataset)} samples")
    
    # Limit samples if specified
    if config.data.max_samples and len(dataset) > config.data.max_samples:
        dataset = dataset.select(range(config.data.max_samples))
        logger.info(f"Limited to {len(dataset)} samples")
    
    # Split into train and eval
    if config.data.val_split_ratio > 0:
        split = dataset.train_test_split(test_size=config.data.val_split_ratio, seed=config.training.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None
    
    # Format dataset for training method
    if config.training.method == "sft":
        train_dataset = format_sft_dataset(train_dataset, config, tokenizer)
        if eval_dataset:
            eval_dataset = format_sft_dataset(eval_dataset, config, tokenizer)
    elif config.training.method == "grpo":
        train_dataset = format_grpo_dataset(train_dataset, config)
        if eval_dataset:
            eval_dataset = format_grpo_dataset(eval_dataset, config)
    elif config.training.method == "dpo":
        train_dataset = format_dpo_dataset(train_dataset, config)
        if eval_dataset:
            eval_dataset = format_dpo_dataset(eval_dataset, config)
    
    return train_dataset, eval_dataset


def format_sft_dataset(dataset: Dataset, config: QLoRAConfig, tokenizer: Any) -> Dataset:
    """Format dataset for SFT training."""
    
    def format_example(example):
        # Get text from the appropriate column
        text = example.get(config.data.text_column, "")
        if not text and "text" in example:
            text = example["text"]
        elif not text and "content" in example:
            text = example["content"]
        elif not text and "prompt" in example:
            text = example["prompt"]
        
        # Apply prompt template if provided
        if config.data.prompt_template and "{text}" in config.data.prompt_template:
            formatted = config.data.prompt_template.format(text=text)
        else:
            formatted = text
        
        return {"text": formatted}
    
    return dataset.map(format_example, remove_columns=dataset.column_names)


def format_grpo_dataset(dataset: Dataset, config: QLoRAConfig) -> Dataset:
    """Format dataset for GRPO training (prompts only)."""
    
    def format_example(example):
        # Get text/prompt
        text = example.get("text", "") or example.get("prompt", "") or example.get("content", "")
        
        # GRPO needs prompts, not full conversations
        if config.data.prompt_template and "{text}" in config.data.prompt_template:
            prompt = config.data.prompt_template.format(text=text)
        else:
            prompt = f"User: {text}\nAssistant:"
        
        return {"prompt": prompt}
    
    return dataset.map(format_example, remove_columns=dataset.column_names)


def format_dpo_dataset(dataset: Dataset, config: QLoRAConfig) -> Dataset:
    """Format dataset for DPO training (requires chosen/rejected pairs)."""
    # DPO requires chosen/rejected pairs
    # This is a basic implementation - may need customization for specific datasets
    
    required_columns = ["prompt", "chosen", "rejected"]
    if all(col in dataset.column_names for col in required_columns):
        return dataset
    
    logger.warning(
        "Dataset doesn't have required DPO columns (prompt, chosen, rejected). "
        "Please provide a properly formatted DPO dataset."
    )
    return dataset


# =============================================================================
# Reward Functions (for GRPO)
# =============================================================================

def create_default_reward_function() -> Callable:
    """
    Create a default reward function for GRPO training.
    
    Returns:
        Reward function that scores completions
    """
    
    def reward_func(completions: List[Any], **kwargs) -> List[float]:
        """
        Combined reward function for GRPO.
        
        Rewards based on:
        - Response length (not too short, not too long)
        - Formatting (complete sentences, proper structure)
        - Non-repetition
        """
        rewards = []
        
        for completion in completions:
            # Handle different completion formats
            if isinstance(completion, list):
                text = completion[0].get("content", "") if completion else ""
            else:
                text = str(completion)
            
            text = text.strip()
            reward = 0.0
            
            if not text:
                rewards.append(0.0)
                continue
            
            # Length reward (20-300 chars is optimal)
            length = len(text)
            if length < 20:
                reward += 0.1 * (length / 20)
            elif length <= 300:
                reward += 0.3
            else:
                reward += max(0.1, 0.3 - (length - 300) / 1000)
            
            # Format reward (ends with punctuation, proper start)
            if text.endswith(('.', '!', '?', ':', '"', "'")):
                reward += 0.2
            if text[0].isupper() or text[0].isdigit():
                reward += 0.1
            
            # Structure reward (has multiple words)
            words = text.split()
            if len(words) >= 5:
                reward += 0.2
            
            # Non-repetition reward
            if len(words) > 3:
                unique_ratio = len(set(words)) / len(words)
                reward += 0.2 * unique_ratio
            
            rewards.append(min(1.0, reward))
        
        return rewards
    
    return reward_func


# =============================================================================
# Training Functions
# =============================================================================

def train_sft(
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    config: QLoRAConfig,
) -> Any:
    """
    Train using Supervised Fine-Tuning (SFT).
    
    Args:
        model: The model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        config: Configuration
        
    Returns:
        Trainer instance
    """
    from trl import SFTTrainer, SFTConfig
    
    logger.info("Starting SFT training...")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{config.training.output_dir}/sft_{timestamp}"
    
    # Training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        
        # Batch settings
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        
        # Learning rate
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        
        # Duration
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        
        # Precision
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        
        # Logging and saving
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        eval_strategy=config.training.eval_strategy if eval_dataset else "no",
        eval_steps=config.training.eval_steps if eval_dataset else None,
        
        # Misc
        seed=config.training.seed,
        dataloader_num_workers=config.training.dataloader_num_workers,
        report_to=config.training.report_to,
        
        # SFT specific
        max_seq_length=config.model.max_seq_length,
        gradient_checkpointing=config.training.gradient_checkpointing,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # Train
    trainer.train()
    
    # Save
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"SFT training complete. Model saved to: {output_dir}")
    
    return trainer


def train_grpo(
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    config: QLoRAConfig,
    reward_func: Optional[Callable] = None,
) -> Any:
    """
    Train using Group Relative Policy Optimization (GRPO).
    
    Args:
        model: The model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        config: Configuration
        reward_func: Custom reward function (optional)
        
    Returns:
        Trainer instance
    """
    from trl import GRPOTrainer, GRPOConfig
    
    logger.info("Starting GRPO training...")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{config.training.output_dir}/grpo_{timestamp}"
    
    # Use default reward function if not provided
    if reward_func is None:
        reward_func = create_default_reward_function()
    
    # Training arguments
    training_args = GRPOConfig(
        output_dir=output_dir,
        
        # Batch settings
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        
        # Learning rate
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        
        # Duration
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        
        # GRPO specific
        num_generations=config.training.grpo.num_generations,
        max_completion_length=config.model.max_completion_length,
        max_prompt_length=config.model.max_prompt_length,
        temperature=config.training.grpo.temperature,
        beta=config.training.grpo.beta,
        
        # Precision
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        
        # Logging and saving
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        
        # Misc
        seed=config.training.seed,
        report_to=config.training.report_to,
        gradient_checkpointing=config.training.gradient_checkpointing,
    )
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_func,
    )
    
    # Train
    trainer.train()
    
    # Save
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"GRPO training complete. Model saved to: {output_dir}")
    
    return trainer


def train_dpo(
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    config: QLoRAConfig,
) -> Any:
    """
    Train using Direct Preference Optimization (DPO).
    
    Args:
        model: The model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        config: Configuration
        
    Returns:
        Trainer instance
    """
    from trl import DPOTrainer, DPOConfig
    
    logger.info("Starting DPO training...")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{config.training.output_dir}/dpo_{timestamp}"
    
    # Training arguments
    training_args = DPOConfig(
        output_dir=output_dir,
        
        # Batch settings
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        
        # Learning rate
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_ratio=config.training.warmup_ratio,
        
        # Duration
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        
        # DPO specific
        beta=config.training.dpo.beta,
        
        # Precision
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        
        # Logging and saving
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        
        # Misc
        seed=config.training.seed,
        report_to=config.training.report_to,
        gradient_checkpointing=config.training.gradient_checkpointing,
    )
    
    # Create trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # Train
    trainer.train()
    
    # Save
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"DPO training complete. Model saved to: {output_dir}")
    
    return trainer


# =============================================================================
# Main Training Function
# =============================================================================

def train(config: QLoRAConfig, reward_func: Optional[Callable] = None) -> Any:
    """
    Main training function.
    
    Args:
        config: QLoRA configuration
        reward_func: Optional custom reward function for GRPO
        
    Returns:
        Trainer instance
    """
    # Print configuration
    config.print_config()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(config, tokenizer)
    
    # Train based on method
    if config.training.method == "sft":
        trainer = train_sft(model, tokenizer, train_dataset, eval_dataset, config)
    elif config.training.method == "grpo":
        trainer = train_grpo(model, tokenizer, train_dataset, eval_dataset, config, reward_func)
    elif config.training.method == "dpo":
        trainer = train_dpo(model, tokenizer, train_dataset, eval_dataset, config)
    else:
        raise ValueError(f"Unknown training method: {config.training.method}")
    
    # Push to hub if requested
    if config.hub.push_to_hub and config.hub.hub_model_id:
        logger.info(f"Pushing to HuggingFace Hub: {config.hub.hub_model_id}")
        trainer.push_to_hub(config.hub.hub_model_id)
    
    return trainer


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args)
    
    # Run training
    try:
        trainer = train(config)
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"Output directory: {config.training.output_dir}")
        print("\nNext steps:")
        print("1. Validate the model with: python validate_quantization.py --check inference")
        print("2. Test locally with the trained adapter")
        print("3. Push to HuggingFace Hub if desired")
        print("=" * 70)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
