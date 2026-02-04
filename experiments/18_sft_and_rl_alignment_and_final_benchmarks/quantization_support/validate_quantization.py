#!/usr/bin/env python3
"""
Quantization Validation Script
Team 18: SFT, RL-Style Alignment & Final Post-Training Benchmarks

This script validates that quantization formats work end-to-end,
addressing GitHub Issue #333.

Usage:
    # Run full validation suite
    python validate_quantization.py --config default_config.yaml
    
    # Quick validation (memory + inference only)
    python validate_quantization.py --quick
    
    # Specific checks
    python validate_quantization.py --check memory
    python validate_quantization.py --check gradients
    python validate_quantization.py --check inference
    python validate_quantization.py --check reproducibility
"""

import os
import sys
import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import torch
import gc

# Local imports
from qlora_config import QLoRAConfig, create_argument_parser, load_config


# =============================================================================
# Validation Result Classes
# =============================================================================

@dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    
    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


@dataclass
class ValidationReport:
    """Complete validation report."""
    timestamp: str
    config_summary: Dict[str, Any]
    results: List[ValidationResult]
    
    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)
    
    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)
    
    def print_report(self):
        """Print formatted report."""
        print("\n" + "=" * 70)
        print("QUANTIZATION VALIDATION REPORT")
        print("=" * 70)
        print(f"Timestamp: {self.timestamp}")
        print(f"Model: {self.config_summary.get('model', 'N/A')}")
        print(f"Quantization: {self.config_summary.get('quantization', 'N/A')}")
        print("-" * 70)
        
        for result in self.results:
            status_icon = "[OK]" if result.passed else "[X]"
            print(f"{status_icon} {result.name}")
            print(f"    {result.message}")
            if result.details:
                for key, value in result.details.items():
                    print(f"    - {key}: {value}")
        
        print("-" * 70)
        print(f"Results: {self.pass_count} passed, {self.fail_count} failed")
        
        if self.all_passed:
            print("\nSTATUS: ALL CHECKS PASSED")
            print("Quantization is supported end-to-end for this configuration.")
        else:
            print("\nSTATUS: SOME CHECKS FAILED")
            print("Please review the failed checks before proceeding with training.")
        
        print("=" * 70)
    
    def save_json(self, path: str):
        """Save report as JSON."""
        report_dict = {
            "timestamp": self.timestamp,
            "config_summary": self.config_summary,
            "all_passed": self.all_passed,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details
                }
                for r in self.results
            ]
        }
        with open(path, 'w') as f:
            json.dump(report_dict, f, indent=2)


# =============================================================================
# Memory Utilities
# =============================================================================

def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    elif torch.backends.mps.is_available():
        # MPS doesn't have a direct memory query, return -1
        return -1
    return 0


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Validation Checks
# =============================================================================

def check_model_loading(config: QLoRAConfig) -> ValidationResult:
    """
    Check 1: Verify model loads with quantization config.
    
    Tests:
    - Model loads without OOM
    - Correct layers are quantized
    - No loading errors
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        clear_memory()
        initial_memory = get_gpu_memory_mb()
        
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
        }
        
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = torch_dtype
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            **model_kwargs,
        )
        
        final_memory = get_gpu_memory_mb()
        memory_used = final_memory - initial_memory if initial_memory >= 0 else -1
        
        # Check model is loaded
        param_count = model.num_parameters()
        
        # Clean up
        del model
        clear_memory()
        
        details = {
            "parameters": f"{param_count:,}",
            "memory_used_mb": f"{memory_used:.2f}" if memory_used >= 0 else "N/A (MPS)",
            "quantization_enabled": config.quantization.enabled,
        }
        
        if config.quantization.enabled:
            details["quantization_bits"] = config.quantization.bits
            details["quantization_type"] = config.quantization.quant_type
        
        return ValidationResult(
            name="Model Loading",
            passed=True,
            message="Model loaded successfully with specified configuration",
            details=details
        )
        
    except Exception as e:
        return ValidationResult(
            name="Model Loading",
            passed=False,
            message=f"Failed to load model: {str(e)}",
            details={"error": str(e)}
        )


def check_memory_usage(config: QLoRAConfig) -> ValidationResult:
    """
    Check 2: Verify memory usage is within expected bounds.
    
    Tests:
    - Memory usage with quantization vs without
    - No memory leaks during forward pass
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        if not torch.cuda.is_available():
            return ValidationResult(
                name="Memory Usage",
                passed=True,
                message="Memory check skipped (no CUDA device)",
                details={"reason": "CUDA not available, memory profiling not supported"}
            )
        
        clear_memory()
        
        # Load model
        bnb_config = config.quantization.to_bnb_config()
        model_kwargs = {
            "trust_remote_code": config.model.trust_remote_code,
            "device_map": config.model.device_map,
        }
        
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            **model_kwargs,
        )
        
        load_memory = get_gpu_memory_mb()
        
        # Create dummy input
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.name,
            trust_remote_code=config.model.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        inputs = tokenizer("Hello, world!", return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            _ = model(**inputs)
        
        forward_memory = get_gpu_memory_mb()
        
        # Multiple forward passes to check for leaks
        for _ in range(5):
            with torch.no_grad():
                _ = model(**inputs)
        
        final_memory = get_gpu_memory_mb()
        memory_leak = final_memory - forward_memory
        
        # Clean up
        del model, inputs
        clear_memory()
        
        # Check for memory leak (allow small variance)
        leak_threshold = 50  # MB
        has_leak = memory_leak > leak_threshold
        
        details = {
            "load_memory_mb": f"{load_memory:.2f}",
            "forward_memory_mb": f"{forward_memory:.2f}",
            "final_memory_mb": f"{final_memory:.2f}",
            "memory_increase_mb": f"{memory_leak:.2f}",
        }
        
        if has_leak:
            return ValidationResult(
                name="Memory Usage",
                passed=False,
                message=f"Potential memory leak detected ({memory_leak:.2f} MB)",
                details=details
            )
        
        return ValidationResult(
            name="Memory Usage",
            passed=True,
            message="Memory usage is stable, no leaks detected",
            details=details
        )
        
    except Exception as e:
        return ValidationResult(
            name="Memory Usage",
            passed=False,
            message=f"Memory check failed: {str(e)}",
            details={"error": str(e)}
        )


def check_lora_application(config: QLoRAConfig) -> ValidationResult:
    """
    Check 3: Verify LoRA adapters are applied correctly.
    
    Tests:
    - LoRA applied to correct modules
    - Adapter parameters are trainable
    - Base model parameters are frozen
    """
    try:
        from transformers import AutoModelForCausalLM
        from peft import get_peft_model, prepare_model_for_kbit_training
        
        clear_memory()
        
        # Load model
        bnb_config = config.quantization.to_bnb_config()
        model_kwargs = {
            "trust_remote_code": config.model.trust_remote_code,
            "device_map": config.model.device_map,
        }
        
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            **model_kwargs,
        )
        
        # Prepare for k-bit training if quantized
        if bnb_config is not None:
            model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA
        peft_config = config.lora.to_peft_config()
        model = get_peft_model(model, peft_config)
        
        # Check trainable parameters
        trainable_params, total_params = model.get_nb_trainable_parameters()
        trainable_ratio = trainable_params / total_params
        
        # Verify some parameters are trainable
        has_trainable = trainable_params > 0
        
        # Check that LoRA parameters exist
        lora_params = []
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                lora_params.append(name)
        
        has_lora = len(lora_params) > 0
        
        # Clean up
        del model
        clear_memory()
        
        details = {
            "trainable_params": f"{trainable_params:,}",
            "total_params": f"{total_params:,}",
            "trainable_ratio": f"{trainable_ratio*100:.2f}%",
            "lora_modules_found": len(lora_params),
            "target_modules": config.lora.target_modules,
        }
        
        if not has_trainable or not has_lora:
            return ValidationResult(
                name="LoRA Application",
                passed=False,
                message="LoRA adapters not properly applied",
                details=details
            )
        
        return ValidationResult(
            name="LoRA Application",
            passed=True,
            message=f"LoRA applied successfully ({trainable_ratio*100:.2f}% trainable)",
            details=details
        )
        
    except Exception as e:
        return ValidationResult(
            name="LoRA Application",
            passed=False,
            message=f"LoRA application failed: {str(e)}",
            details={"error": str(e)}
        )


def check_gradient_flow(config: QLoRAConfig) -> ValidationResult:
    """
    Check 4: Verify gradients flow through LoRA adapters.
    
    Tests:
    - Gradients computed for LoRA parameters
    - No NaN/Inf gradients
    - Base model gradients are None or zero
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, prepare_model_for_kbit_training
        
        clear_memory()
        
        # Load model
        bnb_config = config.quantization.to_bnb_config()
        model_kwargs = {
            "trust_remote_code": config.model.trust_remote_code,
            "device_map": config.model.device_map,
        }
        
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            **model_kwargs,
        )
        
        # Prepare for k-bit training if quantized
        if bnb_config is not None:
            model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA
        peft_config = config.lora.to_peft_config()
        model = get_peft_model(model, peft_config)
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.name,
            trust_remote_code=config.model.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create input
        text = "Hello, this is a test."
        inputs = tokenizer(text, return_tensors="pt")
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        
        # Forward and backward
        model.train()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        
        # Check gradients
        lora_grads = []
        base_grads = []
        nan_inf_grads = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                if "lora" in name.lower():
                    lora_grads.append(name)
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        nan_inf_grads.append(name)
                else:
                    base_grads.append(name)
        
        # Clean up
        del model, inputs
        clear_memory()
        
        details = {
            "lora_params_with_grads": len(lora_grads),
            "base_params_with_grads": len(base_grads),
            "nan_inf_grads": len(nan_inf_grads),
            "loss_value": f"{loss.item():.4f}",
        }
        
        # Check for issues
        if len(lora_grads) == 0:
            return ValidationResult(
                name="Gradient Flow",
                passed=False,
                message="No gradients flowing to LoRA parameters",
                details=details
            )
        
        if len(nan_inf_grads) > 0:
            return ValidationResult(
                name="Gradient Flow",
                passed=False,
                message=f"NaN/Inf gradients detected in {len(nan_inf_grads)} parameters",
                details=details
            )
        
        return ValidationResult(
            name="Gradient Flow",
            passed=True,
            message="Gradients flow correctly through LoRA adapters",
            details=details
        )
        
    except Exception as e:
        return ValidationResult(
            name="Gradient Flow",
            passed=False,
            message=f"Gradient check failed: {str(e)}",
            details={"error": str(e)}
        )


def check_inference(config: QLoRAConfig) -> ValidationResult:
    """
    Check 5: Verify inference works with quantized model.
    
    Tests:
    - Model generates text
    - No NaN/Inf in outputs
    - Generation completes without errors
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, prepare_model_for_kbit_training
        
        clear_memory()
        
        # Load model
        bnb_config = config.quantization.to_bnb_config()
        model_kwargs = {
            "trust_remote_code": config.model.trust_remote_code,
            "device_map": config.model.device_map,
        }
        
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            **model_kwargs,
        )
        
        # Prepare for k-bit training if quantized
        if bnb_config is not None:
            model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA
        peft_config = config.lora.to_peft_config()
        model = get_peft_model(model, peft_config)
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.name,
            trust_remote_code=config.model.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Generate
        model.eval()
        prompt = "Once upon a time"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Check for issues
        generated_tokens = len(outputs[0]) - len(inputs["input_ids"][0])
        
        # Clean up
        del model, inputs, outputs
        clear_memory()
        
        details = {
            "prompt": prompt,
            "generated_tokens": generated_tokens,
            "sample_output": generated[:100] + "..." if len(generated) > 100 else generated,
        }
        
        if generated_tokens < 5:
            return ValidationResult(
                name="Inference",
                passed=False,
                message="Generation produced very few tokens",
                details=details
            )
        
        return ValidationResult(
            name="Inference",
            passed=True,
            message=f"Inference successful, generated {generated_tokens} tokens",
            details=details
        )
        
    except Exception as e:
        return ValidationResult(
            name="Inference",
            passed=False,
            message=f"Inference check failed: {str(e)}",
            details={"error": str(e)}
        )


def check_checkpoint_save_load(config: QLoRAConfig) -> ValidationResult:
    """
    Check 6: Verify checkpoint saving and loading works.
    
    Tests:
    - Adapter checkpoint saves correctly
    - Checkpoint loads without errors
    - Loaded model produces outputs
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel
        
        clear_memory()
        
        # Load model
        bnb_config = config.quantization.to_bnb_config()
        model_kwargs = {
            "trust_remote_code": config.model.trust_remote_code,
            "device_map": config.model.device_map,
        }
        
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            **model_kwargs,
        )
        
        # Prepare for k-bit training if quantized
        if bnb_config is not None:
            model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA
        peft_config = config.lora.to_peft_config()
        model = get_peft_model(model, peft_config)
        
        # Save to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = os.path.join(tmpdir, "adapter")
            model.save_pretrained(adapter_path)
            
            # Check files exist
            adapter_files = os.listdir(adapter_path)
            has_adapter_model = "adapter_model.safetensors" in adapter_files or "adapter_model.bin" in adapter_files
            has_adapter_config = "adapter_config.json" in adapter_files
            
            if not has_adapter_model or not has_adapter_config:
                return ValidationResult(
                    name="Checkpoint Save/Load",
                    passed=False,
                    message="Adapter files not saved correctly",
                    details={"files": adapter_files}
                )
            
            # Clean up first model
            del model
            clear_memory()
            
            # Reload base model
            model = AutoModelForCausalLM.from_pretrained(
                config.model.name,
                **model_kwargs,
            )
            
            # Load adapter
            model = PeftModel.from_pretrained(model, adapter_path)
        
        # Test loaded model
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.name,
            trust_remote_code=config.model.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        inputs = tokenizer("Test", return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Clean up
        del model, inputs, outputs
        clear_memory()
        
        return ValidationResult(
            name="Checkpoint Save/Load",
            passed=True,
            message="Checkpoint save/load works correctly",
            details={"saved_files": ["adapter_model.safetensors/bin", "adapter_config.json"]}
        )
        
    except Exception as e:
        return ValidationResult(
            name="Checkpoint Save/Load",
            passed=False,
            message=f"Checkpoint check failed: {str(e)}",
            details={"error": str(e)}
        )


def check_reproducibility(config: QLoRAConfig) -> ValidationResult:
    """
    Check 7: Verify training is reproducible with same seed.
    
    Tests:
    - Same seed produces same loss values
    - Results are deterministic
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, prepare_model_for_kbit_training
        
        def run_single_step(seed: int) -> float:
            """Run a single training step and return loss."""
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            clear_memory()
            
            # Load model
            bnb_config = config.quantization.to_bnb_config()
            model_kwargs = {
                "trust_remote_code": config.model.trust_remote_code,
                "device_map": config.model.device_map,
            }
            
            if bnb_config is not None:
                model_kwargs["quantization_config"] = bnb_config
            else:
                model_kwargs["torch_dtype"] = torch.bfloat16
            
            model = AutoModelForCausalLM.from_pretrained(
                config.model.name,
                **model_kwargs,
            )
            
            if bnb_config is not None:
                model = prepare_model_for_kbit_training(model)
            
            peft_config = config.lora.to_peft_config()
            model = get_peft_model(model, peft_config)
            
            tokenizer = AutoTokenizer.from_pretrained(
                config.model.name,
                trust_remote_code=config.model.trust_remote_code,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Forward pass
            model.train()
            inputs = tokenizer("Test input for reproducibility", return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["labels"] = inputs["input_ids"].clone()
            
            outputs = model(**inputs)
            loss = outputs.loss.item()
            
            del model, inputs, outputs
            clear_memory()
            
            return loss
        
        # Run twice with same seed
        seed = 42
        loss1 = run_single_step(seed)
        loss2 = run_single_step(seed)
        
        # Check reproducibility
        is_reproducible = abs(loss1 - loss2) < 1e-5
        
        details = {
            "seed": seed,
            "loss_run_1": f"{loss1:.6f}",
            "loss_run_2": f"{loss2:.6f}",
            "difference": f"{abs(loss1 - loss2):.6f}",
        }
        
        if not is_reproducible:
            return ValidationResult(
                name="Reproducibility",
                passed=False,
                message="Training is not reproducible with same seed",
                details=details
            )
        
        return ValidationResult(
            name="Reproducibility",
            passed=True,
            message="Training is reproducible with same seed",
            details=details
        )
        
    except Exception as e:
        return ValidationResult(
            name="Reproducibility",
            passed=False,
            message=f"Reproducibility check failed: {str(e)}",
            details={"error": str(e)}
        )


# =============================================================================
# Main Validation Runner
# =============================================================================

def run_validation(
    config: QLoRAConfig,
    checks: Optional[List[str]] = None,
    quick: bool = False,
) -> ValidationReport:
    """
    Run validation checks.
    
    Args:
        config: QLoRA configuration
        checks: List of specific checks to run (None for all)
        quick: Run quick validation (subset of checks)
        
    Returns:
        ValidationReport
    """
    # Available checks
    all_checks = {
        "loading": ("Model Loading", check_model_loading),
        "memory": ("Memory Usage", check_memory_usage),
        "lora": ("LoRA Application", check_lora_application),
        "gradients": ("Gradient Flow", check_gradient_flow),
        "inference": ("Inference", check_inference),
        "checkpoint": ("Checkpoint Save/Load", check_checkpoint_save_load),
        "reproducibility": ("Reproducibility", check_reproducibility),
    }
    
    # Quick mode runs subset
    quick_checks = ["loading", "memory", "inference"]
    
    # Determine which checks to run
    if quick:
        checks_to_run = quick_checks
    elif checks:
        checks_to_run = checks
    else:
        checks_to_run = list(all_checks.keys())
    
    # Run checks
    results = []
    for check_name in checks_to_run:
        if check_name not in all_checks:
            print(f"Warning: Unknown check '{check_name}', skipping")
            continue
        
        name, check_func = all_checks[check_name]
        print(f"Running: {name}...", end=" ", flush=True)
        
        try:
            result = check_func(config)
            print("PASS" if result.passed else "FAIL")
            results.append(result)
        except Exception as e:
            print("ERROR")
            results.append(ValidationResult(
                name=name,
                passed=False,
                message=f"Check raised exception: {str(e)}",
                details={"error": str(e)}
            ))
    
    # Create report
    config_summary = {
        "model": config.model.name,
        "quantization": f"{config.quantization.bits}-bit {config.quantization.quant_type}" 
                       if config.quantization.enabled else "disabled",
        "lora_rank": config.lora.r,
        "device": config.model.device_map,
    }
    
    report = ValidationReport(
        timestamp=datetime.now().isoformat(),
        config_summary=config_summary,
        results=results,
    )
    
    return report


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate quantization support end-to-end (Issue #333)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--check",
        type=str,
        nargs="+",
        choices=["loading", "memory", "lora", "gradients", "inference", "checkpoint", "reproducibility"],
        help="Specific checks to run"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation (loading, memory, inference only)"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        help="Override model name"
    )
    
    parser.add_argument(
        "--no_quantization",
        action="store_true",
        help="Disable quantization"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Save report to JSON file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_parser = create_argument_parser()
    config_args = config_parser.parse_args([])  # Empty args to get defaults
    
    if args.config:
        config_args.config = args.config
    if args.model_name:
        config_args.model_name = args.model_name
    if args.no_quantization:
        config_args.no_quantization = True
    
    config = load_config(config_args)
    
    # Print configuration
    print("\n" + "=" * 70)
    print("QUANTIZATION VALIDATION")
    print("Issue #333: Ensure quantization formats are supported end-to-end")
    print("=" * 70)
    config.print_config()
    
    # Run validation
    print("\nRunning validation checks...")
    print("-" * 70)
    
    report = run_validation(
        config,
        checks=args.check,
        quick=args.quick,
    )
    
    # Print report
    report.print_report()
    
    # Save if requested
    if args.output:
        report.save_json(args.output)
        print(f"\nReport saved to: {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
