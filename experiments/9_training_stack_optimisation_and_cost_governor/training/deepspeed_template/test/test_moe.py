"""
Test script for MoE (Mixture of Experts) utilities and configuration.

This test verifies that:
1. MoE detection works correctly
2. MoE parameter groups are created properly
3. DeepSpeed configs are valid for MoE

NOTE: Full training tests require a GPU. This script includes both
CPU-only tests (for CI) and GPU tests (for actual verification).

Usage:
    # Run all tests (CPU tests will run, GPU tests will skip if no CUDA)
    pytest test/test_moe.py -v
    
    # Run directly
    python test/test_moe.py
"""

import json
import os
import sys

import pytest
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import directly from moe_utils to avoid __init__.py import chain issues
# (src/__init__.py imports utils.py which may not exist in all setups)
import importlib.util
moe_utils_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "moe_utils.py"
)
spec = importlib.util.spec_from_file_location("moe_utils", moe_utils_path)
moe_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moe_utils)

is_moe_model = moe_utils.is_moe_model
create_moe_param_groups = moe_utils.create_moe_param_groups
get_moe_config_recommendations = moe_utils.get_moe_config_recommendations


class SimpleDenseModel(nn.Module):
    """A simple dense model (no MoE) for testing."""
    
    def __init__(self, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class SimpleMoEBlock(nn.Module):
    """A simple MoE block that mimics DeepSpeed's _z3_leaf pattern."""
    
    def __init__(self, hidden_size=64, num_experts=4):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)
        ])
        # This flag tells DeepSpeed ZeRO-3 to treat this as a leaf module
        self._z3_leaf = True
    
    def forward(self, x):
        # Simplified routing (not actual MoE logic, just for testing)
        router_logits = self.gate(x)
        # Apply all experts (simplified)
        for expert in self.experts:
            x = expert(x)
        return x, router_logits


class SimpleMoEModel(nn.Module):
    """A simple model containing MoE blocks for testing."""
    
    def __init__(self, hidden_size=64, num_experts=4):
        super().__init__()
        self.embed = nn.Linear(hidden_size, hidden_size)
        self.moe_block = SimpleMoEBlock(hidden_size, num_experts)
        self.output = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        x = self.embed(x)
        x, router_logits = self.moe_block(x)
        return self.output(x), router_logits


class TestMoEDetection:
    """Test MoE model detection."""
    
    def test_dense_model_not_detected_as_moe(self):
        """Dense models should not be detected as MoE."""
        model = SimpleDenseModel()
        assert is_moe_model(model) == False
    
    def test_moe_model_detected(self):
        """MoE models with _z3_leaf flag should be detected."""
        model = SimpleMoEModel()
        assert is_moe_model(model) == True
    
    def test_moe_block_detected(self):
        """Standalone MoE block should be detected."""
        block = SimpleMoEBlock()
        assert is_moe_model(block) == True


class TestMoEParamGroups:
    """Test MoE parameter group creation."""
    
    def test_dense_model_param_groups(self):
        """Dense model should get standard param groups."""
        model = SimpleDenseModel()
        params = create_moe_param_groups(model)
        # Should return params without error
        assert params is not None
    
    def test_moe_model_param_groups(self):
        """MoE model should get proper param groups."""
        model = SimpleMoEModel()
        params = create_moe_param_groups(model)
        # Should return params without error
        assert params is not None


class TestMoEConfigFiles:
    """Test MoE DeepSpeed configuration files."""
    
    @pytest.fixture
    def config_dir(self):
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "deepspeed"
        )
    
    def test_zero2_moe_config_exists(self, config_dir):
        """ZeRO-2 MoE config should exist."""
        config_path = os.path.join(config_dir, "zero-2-moe.json")
        assert os.path.exists(config_path), f"Config not found: {config_path}"
    
    def test_zero3_moe_config_exists(self, config_dir):
        """ZeRO-3 MoE config should exist."""
        config_path = os.path.join(config_dir, "zero-3-moe.json")
        assert os.path.exists(config_path), f"Config not found: {config_path}"
    
    def test_zero2_moe_config_valid(self, config_dir):
        """ZeRO-2 MoE config should be valid JSON with correct settings."""
        config_path = os.path.join(config_dir, "zero-2-moe.json")
        with open(config_path) as f:
            config = json.load(f)
        
        assert config["zero_optimization"]["stage"] == 2
        assert config["fp16"]["enabled"] == True
        assert config["fp16"]["fp16_master_weights_and_grads"] == True
    
    def test_zero3_moe_config_valid(self, config_dir):
        """ZeRO-3 MoE config should be valid JSON with correct settings."""
        config_path = os.path.join(config_dir, "zero-3-moe.json")
        with open(config_path) as f:
            config = json.load(f)
        
        assert config["zero_optimization"]["stage"] == 3
        assert config["fp16"]["enabled"] == True
        assert config["fp16"]["fp16_master_weights_and_grads"] == True


class TestMoERecommendations:
    """Test MoE configuration recommendations."""
    
    def test_recommendations_structure(self):
        """Recommendations should have expected structure."""
        recs = get_moe_config_recommendations()
        
        assert "recommended_zero_stage" in recs
        assert "reason" in recs
        assert "memory_optimization" in recs
        assert "known_issues" in recs
    
    def test_recommended_stage_is_2(self):
        """ZeRO-2 should be recommended for stability."""
        recs = get_moe_config_recommendations()
        assert recs["recommended_zero_stage"] == 2
    
    def test_known_issues_documented(self):
        """Known issues should be documented."""
        recs = get_moe_config_recommendations()
        issues = recs["known_issues"]
        
        assert len(issues) > 0
        assert any("race condition" in issue["issue"].lower() for issue in issues)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMoEWithDeepSpeed:
    """GPU tests for MoE with DeepSpeed.
    
    These tests require a GPU and will be skipped on CPU-only machines.
    """
    
    def test_moe_model_forward_pass(self):
        """Test forward pass with MoE model on GPU."""
        model = SimpleMoEModel().cuda()
        x = torch.randn(2, 64).cuda()
        output, router_logits = model(x)
        
        assert output.shape == (2, 64)
        assert router_logits.shape[0] == 2


def run_all_tests():
    """Run all tests and print summary."""
    print("=" * 60)
    print("MoE Utilities Test Suite")
    print("=" * 60)
    
    # Run tests
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    
    return exit_code


if __name__ == "__main__":
    run_all_tests()
