#!/usr/bin/env python3
"""
MoE Architecture - Main Entry Point
====================================

Team 8 - Expert Expansion & Routing

This script provides the main entry point for:
1. Creating models from configurations
2. Training with proper telemetry
3. Model expansion (1B→3B→8B→70B)
4. Evaluation and inference

Usage Examples:
--------------

# Create a model from config
python main.py --config 3b_moe --action create

# Train a model
python main.py --config 3b_moe --action train --checkpoint path/to/checkpoint.pt

# Expand model (e.g., 3B MoE → 8B MoE)
python main.py --config 8b_moe --action expand --source-checkpoint 3b_moe.pt

# Run inference
python main.py --config 3b_moe --action inference --checkpoint model.pt --prompt "Hello"

# Print configuration summary
python main.py --config 70b_moe --action summary

Author: Team 8 - MoE Architecture
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from configs import (
    get_config,
    CONFIGS,
)
from model.transformer import MoETransformer, create_model, load_model
from model.config import MoEModelConfig


# =============================================================================
# Model Creation
# =============================================================================

def create_model_from_config(config_name: str, device: str = 'cuda') -> MoETransformer:
    """
    Create a new model from configuration.
    
    Args:
        config_name: Configuration name (1b_dense, 3b_moe, 8b_moe, 70b_moe)
        device: Device to create model on
        
    Returns:
        Initialized MoETransformer model
    """
    config = get_config(config_name)
    print(config.summary())
    
    model = create_model(config)
    model = model.to(device)
    
    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total Parameters: {total_params / 1e9:.2f}B")
    print(f"   Trainable Parameters: {trainable_params / 1e9:.2f}B")
    print(f"   Device: {device}")
    
    return model


# =============================================================================
# Model Expansion
# =============================================================================

def expand_model(
    source_checkpoint: str,
    target_config_name: str,
    output_path: str,
    device: str = 'cuda'
) -> MoETransformer:
    """
    Expand a model to a larger configuration.
    
    Supports transitions:
    - 1B Dense → 3B MoE (expert explosion)
    - 3B MoE → 8B MoE (dimension scaling)
    - 8B MoE → 70B MoE (expert expansion)
    
    Args:
        source_checkpoint: Path to source model checkpoint
        target_config_name: Target configuration name
        output_path: Path to save expanded model
        device: Device for computation
        
    Returns:
        Expanded model
    """
    print(f"\n🔄 Expanding Model")
    print(f"   Source: {source_checkpoint}")
    print(f"   Target: {target_config_name}")
    
    # Load source model
    source_checkpoint_data = torch.load(source_checkpoint, map_location=device)
    source_config = MoEModelConfig.from_dict(source_checkpoint_data['config'])
    source_model = create_model(source_config)
    source_model.load_state_dict(source_checkpoint_data['model_state_dict'])
    
    # Create target model
    target_config = get_config(target_config_name)
    target_model = create_model(target_config).to(device)
    
    # Determine expansion type
    source_stage = source_config.stage
    target_stage = target_config.stage
    
    print(f"   Transition: {source_stage} → {target_stage}")
    
    if source_stage == 1 and target_stage == 2:
        # Expert explosion: 1B Dense → 3B MoE
        print("   Type: Expert Explosion (1→8 experts)")
        noise_std = target_config.expert.noise_std_for_expansion
        target_model.init_from_dense(source_model, noise_std=noise_std)
        
    elif source_stage == 2 and target_stage == 3:
        # Dimension scaling: 3B MoE → 8B MoE
        print("   Type: Dimension Scaling (same 8 experts, bigger)")
        # This requires weight interpolation - simplified version
        _scale_model_dimensions(source_model, target_model)
        
    elif source_stage == 3 and target_stage == 4:
        # Expert expansion: 8B MoE → 70B MoE
        print("   Type: Expert Expansion (8→64 experts)")
        children_per_parent = 8
        noise_std = target_config.expert.noise_std_for_expansion
        target_model.expand_experts(source_model, children_per_parent, noise_std)
        
    else:
        raise ValueError(f"Unsupported expansion: {source_stage} → {target_stage}")
    
    # Save expanded model
    print(f"   Saving to: {output_path}")
    save_checkpoint(target_model, target_config, output_path)
    
    return target_model


def _scale_model_dimensions(source: MoETransformer, target: MoETransformer):
    """
    Scale model dimensions via interpolation.
    
    This is a simplified implementation - production should use
    proper weight interpolation techniques.
    """
    # Copy embeddings with interpolation
    with torch.no_grad():
        # Simple approach: initialize from scratch for larger dimensions
        # In production, use proper interpolation
        
        # Copy what we can directly
        source_vocab, source_hidden = source.embed_tokens.weight.shape
        target_vocab, target_hidden = target.embed_tokens.weight.shape
        
        min_hidden = min(source_hidden, target_hidden)
        target.embed_tokens.weight.data[:, :min_hidden] = source.embed_tokens.weight.data[:, :min_hidden]
        
        print(f"   Note: Dimension scaling uses partial initialization")
        print(f"   Source hidden: {source_hidden}, Target hidden: {target_hidden}")


# =============================================================================
# Training
# =============================================================================

def train_model(
    model: MoETransformer,
    config: MoEModelConfig,
    train_dataloader: DataLoader,
    num_epochs: int = 1,
    learning_rate: float = 1e-4,
    checkpoint_dir: str = 'checkpoints',
    log_interval: int = 100
) -> Dict[str, Any]:
    """
    Train the MoE model with proper telemetry.
    
    Args:
        model: MoETransformer model
        config: Model configuration
        train_dataloader: Training data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        checkpoint_dir: Directory for checkpoints
        log_interval: Steps between logging
        
    Returns:
        Training metrics dictionary
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.train()
    global_step = 0
    metrics = {
        'losses': [],
        'router_metrics': [],
    }
    
    print(f"\n🚀 Starting Training")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Checkpoint Dir: {checkpoint_dir}")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch.get('labels', input_ids).to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                return_router_info=True
            )
            
            loss = outputs['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update MoE biases (loss-free load balancing)
            router_metrics = model.post_training_step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Logging
            if global_step % log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"   Step {global_step} | Loss: {avg_loss:.4f}")
                
                # Log router info if available
                if outputs.get('router_info'):
                    _log_router_info(outputs['router_info'])
                
                metrics['losses'].append(avg_loss)
                metrics['router_metrics'].append(router_metrics)
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"\n📊 Epoch {epoch + 1} Complete | Avg Loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"{config.model_name}_epoch{epoch + 1}.pt"
        )
        save_checkpoint(model, config, checkpoint_path)
    
    return metrics


def _log_router_info(router_info: list):
    """Log router statistics."""
    if not router_info:
        return
    
    # Aggregate across layers
    null_rates = []
    entropies = []
    
    for layer_info in router_info:
        if 'overall_null_rate' in layer_info:
            null_rates.append(layer_info['overall_null_rate'])
        if 'health' in layer_info and 'metrics' in layer_info['health']:
            if 'normalized_entropy' in layer_info['health']['metrics']:
                entropies.append(layer_info['health']['metrics']['normalized_entropy'])
    
    if null_rates:
        avg_null = sum(null_rates) / len(null_rates)
        print(f"      Avg Null Rate: {avg_null:.2%}")
    
    if entropies:
        avg_entropy = sum(entropies) / len(entropies)
        print(f"      Avg Router Entropy: {avg_entropy:.3f}")


# =============================================================================
# Inference
# =============================================================================

def run_inference(
    model: MoETransformer,
    prompt: str,
    tokenizer,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9
) -> str:
    """
    Run inference with the model.
    
    Args:
        model: MoETransformer model
        prompt: Input prompt text
        tokenizer: Tokenizer for encoding/decoding
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        
    Returns:
        Generated text
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            logits = outputs['logits'][:, -1, :]  # Last token logits
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    return generated_text


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(
    model: MoETransformer,
    config: MoEModelConfig,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None
):
    """Save model checkpoint."""
    checkpoint = {
        'config': config.to_dict() if hasattr(config, 'to_dict') else vars(config),
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if global_step is not None:
        checkpoint['global_step'] = global_step
    
    torch.save(checkpoint, path)
    print(f"   ✅ Saved checkpoint: {path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='MoE Architecture - Team 8',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print configuration summary
  python main.py --config 3b_moe --action summary
  
  # Create a new model
  python main.py --config 3b_moe --action create --output model.pt
  
  # Expand model from 3B to 8B
  python main.py --config 8b_moe --action expand --source checkpoint_3b.pt --output model_8b.pt
"""
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        choices=list(CONFIGS.keys()),
        help='Configuration name'
    )
    
    parser.add_argument(
        '--action',
        type=str,
        required=True,
        choices=['summary', 'create', 'expand', 'train', 'inference'],
        help='Action to perform'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        help='Source checkpoint for expansion'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='model.pt',
        help='Output path for checkpoint'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Checkpoint to load for train/inference'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        default='Hello, world!',
        help='Prompt for inference'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f" MoE Architecture - Team 8")
    print(f"{'='*60}")
    print(f" Config: {args.config}")
    print(f" Action: {args.action}")
    print(f" Device: {args.device}")
    print(f"{'='*60}\n")
    
    # Execute action
    if args.action == 'summary':
        config = get_config(args.config)
        print(config.summary())
        
    elif args.action == 'create':
        model = create_model_from_config(args.config, args.device)
        config = get_config(args.config)
        save_checkpoint(model, config, args.output)
        
    elif args.action == 'expand':
        if not args.source:
            parser.error("--source required for expand action")
        expand_model(args.source, args.config, args.output, args.device)
        
    elif args.action == 'train':
        print("Training requires data loader - see train_model() function")
        print("Example usage in code:")
        print("  from main import train_model, create_model_from_config")
        print("  model = create_model_from_config('3b_moe')")
        print("  train_model(model, config, dataloader)")
        
    elif args.action == 'inference':
        print("Inference requires tokenizer - see run_inference() function")
        print("Example usage in code:")
        print("  from main import run_inference, load_model")
        print("  model = load_model('checkpoint.pt')")
        print("  output = run_inference(model, 'Hello', tokenizer)")


if __name__ == '__main__':
    main()
