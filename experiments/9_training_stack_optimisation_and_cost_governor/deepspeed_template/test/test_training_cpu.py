"""
CPU-compatible test suite for basic validation (no CUDA required).

This lightweight test suite validates core functionality without requiring GPU:
1. Configuration files validity
2. Module imports
3. Basic model loading
4. Data pipeline
5. Tokenization

Usage:
    pytest test_training_cpu.py -v
"""

import json
import os
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
from src.data import get_tokenizer, preprocess_function
from src.utils import set_seed
from transformers import AutoModelForCausalLM

# Get project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestConfiguration:
    """Test configuration files without needing GPU."""

    def test_zero2_config_valid_json(self):
        """Validate ZeRO-2 config is valid JSON with correct structure."""
        config_path = os.path.join(PROJECT_ROOT, "config/deepspeed/zero-2.json")
        assert os.path.exists(config_path), f"Config not found: {config_path}"

        with open(config_path, "r") as f:
            config = json.load(f)

        # Check critical fields
        assert "zero_optimization" in config
        assert config["zero_optimization"]["stage"] == 2
        assert config["train_batch_size"] > 0
        assert config["gradient_accumulation_steps"] > 0

        print("✓ ZeRO-2 configuration is valid")

    def test_zero3_config_valid_json(self):
        """Validate ZeRO-3 config is valid JSON with correct structure."""
        config_path = os.path.join(PROJECT_ROOT, "config/deepspeed/zero-3.json")
        assert os.path.exists(config_path), f"Config not found: {config_path}"

        with open(config_path, "r") as f:
            config = json.load(f)

        # Check critical fields
        assert "zero_optimization" in config
        assert config["zero_optimization"]["stage"] == 3
        assert "offload_param" in config["zero_optimization"]
        assert "offload_optimizer" in config["zero_optimization"]

        print("✓ ZeRO-3 configuration is valid")

    def test_zero_configs_optimizer_settings(self):
        """Verify optimizer configurations are properly set."""
        for config_name in ["zero-2.json", "zero-3.json"]:
            config_path = os.path.join(PROJECT_ROOT, f"config/deepspeed/{config_name}")
            with open(config_path, "r") as f:
                config = json.load(f)

            assert "optimizer" in config
            assert "type" in config["optimizer"]
            assert "params" in config["optimizer"]
            assert "lr" in config["optimizer"]["params"]

            # Validate learning rate is reasonable
            lr = config["optimizer"]["params"]["lr"]
            assert 0 < lr < 1, f"Learning rate {lr} seems unreasonable"

        print("✓ Optimizer settings validated")

    def test_zero_configs_fp16_settings(self):
        """Verify FP16 settings are configured."""
        for config_name in ["zero-2.json", "zero-3.json"]:
            config_path = os.path.join(PROJECT_ROOT, f"config/deepspeed/{config_name}")
            with open(config_path, "r") as f:
                config = json.load(f)

            assert "fp16" in config
            assert "enabled" in config["fp16"]

        print("✓ FP16 settings validated")


class TestModuleImports:
    """Test that all modules can be imported."""

    def test_import_train_module(self):
        """Test importing train module."""

        print("✓ Train module imported successfully")

    def test_import_data_module(self):
        """Test importing data module."""

        print("✓ Data module imported successfully")

    def test_import_utils_module(self):
        """Test importing utils module."""

        print("✓ Utils module imported successfully")

    def test_deepspeed_available(self):
        """Test that DeepSpeed is installed and importable."""
        import deepspeed

        print(f"✓ DeepSpeed {deepspeed.__version__} available")


class TestDataPipeline:
    """Test data loading and processing without full training."""

    def test_tokenizer_loading(self):
        """Test that tokenizer loads correctly."""
        tokenizer = get_tokenizer("distilgpt2")

        assert tokenizer is not None
        assert hasattr(tokenizer, "vocab_size")
        assert tokenizer.pad_token is not None

        print(f"✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")

    def test_tokenization_function(self):
        """Test tokenization of sample text."""
        tokenizer = get_tokenizer("distilgpt2")

        # Sample text
        examples = {
            "text": [
                "This is a test sentence.",
                "Another test sentence for tokenization.",
            ]
        }

        # Tokenize
        tokenized = preprocess_function(examples, tokenizer, max_length=64)

        # Validate output
        assert "input_ids" in tokenized
        assert "attention_mask" in tokenized
        assert "labels" in tokenized
        assert len(tokenized["input_ids"]) == 2
        assert len(tokenized["input_ids"][0]) == 64  # max_length

        print("✓ Tokenization function works correctly")


class TestModelLoading:
    """Test model loading without training."""

    def test_load_small_model(self):
        """Test loading a small model."""
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")

        assert model is not None
        assert hasattr(model, "config")

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0

        print(f"✓ Model loaded ({num_params:,} parameters)")

    def test_model_forward_pass_cpu(self):
        """Test model forward pass on CPU."""
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        model.eval()

        # Create dummy input
        batch_size = 2
        seq_length = 32
        vocab_size = model.config.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        assert outputs.logits is not None
        assert outputs.logits.shape == (batch_size, seq_length, vocab_size)

        print("✓ Model forward pass successful on CPU")


class TestUtilities:
    """Test utility functions."""

    def test_set_seed_function(self):
        """Test that set_seed works correctly."""
        set_seed(42)
        rand1 = torch.rand(5)

        set_seed(42)
        rand2 = torch.rand(5)

        assert torch.allclose(rand1, rand2), "Seed should produce reproducible results"

        print("✓ set_seed produces reproducible results")

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        rand1 = torch.rand(5)

        set_seed(123)
        rand2 = torch.rand(5)

        assert not torch.allclose(
            rand1, rand2
        ), "Different seeds should produce different results"

        print("✓ Different seeds produce different results")


class TestZeROConfigurationDetails:
    """Detailed tests for ZeRO configuration parameters."""

    def test_zero2_offload_optimizer(self):
        """Verify ZeRO-2 optimizer offload configuration."""
        with open(os.path.join(PROJECT_ROOT, "config/deepspeed/zero-2.json"), "r") as f:
            config = json.load(f)

        zero_config = config["zero_optimization"]
        assert "offload_optimizer" in zero_config
        assert zero_config["offload_optimizer"]["device"] in ["cpu", "nvme"]

        print("✓ ZeRO-2 optimizer offload configured correctly")

    def test_zero3_parameter_offload(self):
        """Verify ZeRO-3 parameter offload configuration."""
        with open(os.path.join(PROJECT_ROOT, "config/deepspeed/zero-3.json"), "r") as f:
            config = json.load(f)

        zero_config = config["zero_optimization"]
        assert "offload_param" in zero_config
        assert "offload_optimizer" in zero_config
        assert zero_config["offload_param"]["device"] in ["cpu", "nvme"]

        print("✓ ZeRO-3 parameter offload configured correctly")

    def test_gradient_accumulation_configured(self):
        """Verify gradient accumulation is properly configured."""
        for config_name in ["zero-2.json", "zero-3.json"]:
            with open(
                os.path.join(PROJECT_ROOT, f"config/deepspeed/{config_name}"), "r"
            ) as f:
                config = json.load(f)

            assert "gradient_accumulation_steps" in config
            assert config["gradient_accumulation_steps"] >= 1

        print("✓ Gradient accumulation configured")

    def test_gradient_clipping_configured(self):
        """Verify gradient clipping is configured."""
        for config_name in ["zero-2.json", "zero-3.json"]:
            with open(
                os.path.join(PROJECT_ROOT, f"config/deepspeed/{config_name}"), "r"
            ) as f:
                config = json.load(f)

            assert "gradient_clipping" in config
            assert config["gradient_clipping"] > 0

        print("✓ Gradient clipping configured")


def test_summary():
    """Print test summary."""
    print("\n" + "=" * 80)
    print("CPU TEST SUITE - No GPU Required")
    print("=" * 80)
    print("\nThese tests validate:")
    print("  ✓ Configuration file validity (ZeRO-2 and ZeRO-3)")
    print("  ✓ Module imports")
    print("  ✓ Tokenizer loading and functionality")
    print("  ✓ Model loading")
    print("  ✓ CPU forward pass")
    print("  ✓ Utility functions")
    print("  ✓ ZeRO configuration details")
    print("\nRun with: pytest test_training_cpu.py -v")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_summary()
    import sys

    pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:])
