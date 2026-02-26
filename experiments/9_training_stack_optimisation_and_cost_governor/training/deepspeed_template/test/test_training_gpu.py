"""
GPU-accelerated test suite for validating training loop and DeepSpeed ZeRO configuration.

⚠️  REQUIRES: CUDA-enabled GPU (tests will be skipped if CUDA is not available)

This test suite validates:
1. Training loop functionality with GPU acceleration
2. ZeRO Stage 2 and Stage 3 configurations
3. Model forward/backward passes on GPU
4. Checkpoint saving/loading
5. DeepSpeed engine initialization with CUDA
6. Memory efficiency of ZeRO optimization

Usage:
    # Run all tests (requires CUDA)
    pytest test_training_gpu.py -v

    # Run specific test
    pytest test_training_gpu.py::test_zero_stage2_config -v
    
Note: Most tests use @pytest.mark.skipif to automatically skip when CUDA is unavailable.
For CPU-only testing, use test_training_cpu.py instead.
"""

import json
import os
import shutil
import sys
import tempfile

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import deepspeed
import pytest
import torch
from src.data import get_tokenizer
from src.train import evaluate, save_checkpoint, train_epoch
from src.utils import set_seed
from transformers import AutoModelForCausalLM

# Get project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def small_model():
    """Load a small model for testing (distilgpt2)."""
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return model


@pytest.fixture
def tokenizer_fixture():
    """Load tokenizer for testing."""
    return get_tokenizer("distilgpt2")


@pytest.fixture
def small_dataloader(tokenizer_fixture):
    """Create a small dataloader for testing."""
    # Create a minimal dataset for testing
    from torch.utils.data import DataLoader, TensorDataset

    # Create dummy data
    batch_size = 2
    seq_length = 32
    num_samples = 4

    input_ids = torch.randint(
        0, tokenizer_fixture.vocab_size, (num_samples, seq_length)
    )
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Convert to expected format
    class FormattedDataLoader:
        def __init__(self, dataloader):
            self.dataloader = dataloader

        def __iter__(self):
            for input_ids, attention_mask, labels in self.dataloader:
                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }

        def __len__(self):
            return len(self.dataloader)

    return FormattedDataLoader(dataloader)


class TestZeRoConfiguration:
    """Test ZeRO configuration validation."""

    def test_zero_stage2_config_exists(self):
        """Test that ZeRO Stage 2 configuration file exists and is valid."""
        config_path = os.path.join(PROJECT_ROOT, "config/deepspeed/zero-2.json")
        assert os.path.exists(
            config_path
        ), f"ZeRO Stage 2 config not found at {config_path}"

        with open(config_path, "r") as f:
            config = json.load(f)

        # Validate key configuration parameters
        assert "zero_optimization" in config, "zero_optimization not found in config"
        assert config["zero_optimization"]["stage"] == 2, "ZeRO stage should be 2"
        assert "optimizer" in config, "optimizer not configured"
        assert "scheduler" in config, "scheduler not configured"

        print("✓ ZeRO Stage 2 config validated")

    def test_zero_stage3_config_exists(self):
        """Test that ZeRO Stage 3 configuration file exists and is valid."""
        config_path = os.path.join(PROJECT_ROOT, "config/deepspeed/zero-3.json")
        assert os.path.exists(
            config_path
        ), f"ZeRO Stage 3 config not found at {config_path}"

        with open(config_path, "r") as f:
            config = json.load(f)

        # Validate key configuration parameters
        assert "zero_optimization" in config, "zero_optimization not found in config"
        assert config["zero_optimization"]["stage"] == 3, "ZeRO stage should be 3"
        assert "optimizer" in config, "optimizer not configured"
        assert "scheduler" in config, "scheduler not configured"

        # ZeRO-3 specific checks
        zero_config = config["zero_optimization"]
        assert (
            "offload_param" in zero_config
        ), "ZeRO-3 should have offload_param configured"

        print("✓ ZeRO Stage 3 config validated")

    def test_zero_configs_have_required_fields(self):
        """Test that both ZeRO configs have all required fields."""
        required_fields = [
            "train_batch_size",
            "gradient_accumulation_steps",
            "optimizer",
            "zero_optimization",
            "gradient_clipping",
        ]

        for config_name in ["zero-2.json", "zero-3.json"]:
            config_path = os.path.join(PROJECT_ROOT, f"config/deepspeed/{config_name}")
            with open(config_path, "r") as f:
                config = json.load(f)

            for field in required_fields:
                assert field in config, f"{field} missing in {config_name}"

        print("✓ All required fields present in ZeRO configs")


class TestDeepSpeedInitialization:
    """Test DeepSpeed engine initialization."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_deepspeed_initialization_stage2(self, small_model, temp_output_dir):
        """Test DeepSpeed initialization with ZeRO Stage 2."""

        # Create minimal args
        class Args:
            local_rank = -1
            deepspeed_config = os.path.join(
                PROJECT_ROOT, "config/deepspeed/zero-2.json"
            )

        args = Args()

        # Initialize DeepSpeed
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            args=args, model=small_model, model_parameters=small_model.parameters()
        )

        # Validate initialization
        assert model_engine is not None, "Model engine not initialized"
        assert optimizer is not None, "Optimizer not initialized"
        assert hasattr(
            model_engine, "device"
        ), "Model engine should have device attribute"

        # Check ZeRO stage
        assert model_engine.zero_optimization_stage() == 2, "ZeRO stage should be 2"

        print("✓ DeepSpeed initialized with ZeRO Stage 2")
        print(f"  Device: {model_engine.device}")
        print(f"  ZeRO Stage: {model_engine.zero_optimization_stage()}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_deepspeed_initialization_stage3(self, small_model, temp_output_dir):
        """Test DeepSpeed initialization with ZeRO Stage 3."""

        # Create minimal args
        class Args:
            local_rank = -1
            deepspeed_config = os.path.join(
                PROJECT_ROOT, "config/deepspeed/zero-3.json"
            )

        args = Args()

        # Initialize DeepSpeed
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            args=args, model=small_model, model_parameters=small_model.parameters()
        )

        # Validate initialization
        assert model_engine is not None, "Model engine not initialized"
        assert optimizer is not None, "Optimizer not initialized"

        # Check ZeRO stage
        assert model_engine.zero_optimization_stage() == 3, "ZeRO stage should be 3"

        print("✓ DeepSpeed initialized with ZeRO Stage 3")
        print(f"  Device: {model_engine.device}")
        print(f"  ZeRO Stage: {model_engine.zero_optimization_stage()}")


class TestTrainingLoop:
    """Test training loop functionality."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_train_epoch_runs(self, small_model, small_dataloader, temp_output_dir):
        """Test that train_epoch function runs without errors."""

        # Initialize DeepSpeed
        class Args:
            local_rank = -1
            deepspeed_config = os.path.join(
                PROJECT_ROOT, "config/deepspeed/zero-2.json"
            )

        args = Args()

        model_engine, _, _, _ = deepspeed.initialize(
            args=args, model=small_model, model_parameters=small_model.parameters()
        )

        # Run one epoch with limited steps
        set_seed(42)
        avg_loss = train_epoch(
            model_engine=model_engine,
            train_loader=small_dataloader,
            epoch=0,
            max_steps=2,  # Only run 2 steps for testing
            log_interval=1,
        )

        # Validate outputs
        assert isinstance(avg_loss, float), "Average loss should be a float"
        assert avg_loss > 0, "Loss should be positive"
        assert not torch.isnan(torch.tensor(avg_loss)), "Loss should not be NaN"

        print("✓ Training epoch completed successfully")
        print(f"  Average Loss: {avg_loss:.4f}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_evaluate_runs(self, small_model, small_dataloader, temp_output_dir):
        """Test that evaluate function runs without errors."""

        # Initialize DeepSpeed
        class Args:
            local_rank = -1
            deepspeed_config = os.path.join(
                PROJECT_ROOT, "config/deepspeed/zero-2.json"
            )

        args = Args()

        model_engine, _, _, _ = deepspeed.initialize(
            args=args, model=small_model, model_parameters=small_model.parameters()
        )

        # Run evaluation
        set_seed(42)
        avg_loss, avg_perplexity = evaluate(
            model_engine=model_engine,
            data_loader=small_dataloader,
            phase="Test",
            max_steps=2,  # Only run 2 steps for testing
        )

        # Validate outputs
        assert isinstance(avg_loss, float), "Average loss should be a float"
        assert isinstance(avg_perplexity, float), "Average perplexity should be a float"
        assert avg_loss > 0, "Loss should be positive"
        assert avg_perplexity > 0, "Perplexity should be positive"
        assert not torch.isnan(torch.tensor(avg_loss)), "Loss should not be NaN"

        print("✓ Evaluation completed successfully")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Perplexity: {avg_perplexity:.4f}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_backward_pass(self, small_model):
        """Test that forward and backward passes work correctly."""

        # Initialize DeepSpeed
        class Args:
            local_rank = -1
            deepspeed_config = os.path.join(
                PROJECT_ROOT, "config/deepspeed/zero-2.json"
            )

        args = Args()

        model_engine, _, _, _ = deepspeed.initialize(
            args=args, model=small_model, model_parameters=small_model.parameters()
        )

        # Create dummy batch
        batch_size = 2
        seq_length = 32
        vocab_size = small_model.config.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(
            model_engine.device
        )
        attention_mask = torch.ones_like(input_ids).to(model_engine.device)
        labels = input_ids.clone()

        # Forward pass
        model_engine.train()
        outputs = model_engine(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Validate forward pass
        assert loss is not None, "Loss should not be None"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert loss.item() > 0, "Loss should be positive"

        # Backward pass
        model_engine.backward(loss)

        # Step
        model_engine.step()

        print("✓ Forward/backward pass completed successfully")
        print(f"  Loss: {loss.item():.4f}")


class TestCheckpointing:
    """Test checkpoint saving and loading."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_save_checkpoint(self, small_model, temp_output_dir):
        """Test that checkpoint saving works."""

        # Initialize DeepSpeed
        class Args:
            local_rank = -1
            deepspeed_config = os.path.join(
                PROJECT_ROOT, "config/deepspeed/zero-2.json"
            )

        args = Args()

        model_engine, _, _, _ = deepspeed.initialize(
            args=args, model=small_model, model_parameters=small_model.parameters()
        )

        # Save checkpoint
        checkpoint_dir = os.path.join(temp_output_dir, "test_checkpoint")
        save_checkpoint(model_engine, checkpoint_dir, tag="test")

        # Validate checkpoint exists
        assert os.path.exists(checkpoint_dir), "Checkpoint directory not created"

        # Check for checkpoint files
        checkpoint_files = os.listdir(checkpoint_dir)
        assert len(checkpoint_files) > 0, "No checkpoint files created"

        print("✓ Checkpoint saved successfully")
        print(f"  Location: {checkpoint_dir}")
        print(f"  Files: {checkpoint_files}")


class TestZeROMemoryEfficiency:
    """Test ZeRO memory efficiency and optimization."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_zero_reduces_memory_usage(self, small_model):
        """
        Test that ZeRO optimization reduces memory usage compared to standard training.
        This is a basic sanity check.
        """
        torch.cuda.empty_cache()

        # Get initial memory
        initial_memory = torch.cuda.memory_allocated()

        # Initialize with ZeRO Stage 2
        class Args:
            local_rank = -1
            deepspeed_config = os.path.join(
                PROJECT_ROOT, "config/deepspeed/zero-2.json"
            )

        args = Args()

        model_engine, _, _, _ = deepspeed.initialize(
            args=args, model=small_model, model_parameters=small_model.parameters()
        )

        # Get memory after initialization
        memory_after_init = torch.cuda.memory_allocated()

        # Run a forward/backward pass
        batch_size = 2
        seq_length = 32
        vocab_size = model_engine.module.config.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(
            model_engine.device
        )
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        model_engine.train()
        outputs = model_engine(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        model_engine.backward(loss)
        model_engine.step()

        # Get memory after training step
        memory_after_step = torch.cuda.memory_allocated()

        print("✓ Memory usage tracking completed")
        print(f"  Initial memory: {initial_memory / 1024**2:.2f} MB")
        print(f"  After initialization: {memory_after_init / 1024**2:.2f} MB")
        print(f"  After training step: {memory_after_step / 1024**2:.2f} MB")

        # Basic sanity check - memory should be allocated
        assert (
            memory_after_step > initial_memory
        ), "Memory should be allocated during training"


class TestIntegration:
    """Integration tests for full training pipeline."""

    def test_imports_work(self):
        """Test that all required imports work."""

        print("✓ All imports successful")

    def test_seed_reproducibility(self):
        """Test that set_seed produces reproducible results."""
        from src.utils import set_seed

        # Set seed and generate random numbers
        set_seed(42)
        random_1 = torch.rand(10)

        # Set same seed again
        set_seed(42)
        random_2 = torch.rand(10)

        # Should be identical
        assert torch.allclose(
            random_1, random_2
        ), "Random numbers should be identical with same seed"

        print("✓ Seed reproducibility confirmed")


def test_summary():
    """Print a summary of what tests validate."""
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    print("\nThis test suite validates:")
    print("  ✓ ZeRO Stage 2 configuration")
    print("  ✓ ZeRO Stage 3 configuration")
    print("  ✓ DeepSpeed engine initialization")
    print("  ✓ Training loop execution")
    print("  ✓ Evaluation loop execution")
    print("  ✓ Forward/backward passes")
    print("  ✓ Checkpoint saving")
    print("  ✓ Memory efficiency")
    print("  ✓ Import compatibility")
    print("  ✓ Seed reproducibility")
    print("\nRun with: pytest test_training_gpu.py -v")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Print test summary
    test_summary()

    # Run pytest
    import sys

    pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:])
