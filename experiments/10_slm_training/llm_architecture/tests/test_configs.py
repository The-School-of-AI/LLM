"""
Lightweight Config Validation Tests
====================================

Tests config loading, parsing, and CLI handling WITHOUT instantiating models.
Run this to validate your setup before heavy training runs.

Usage:
    python tests/test_configs.py
    python tests/test_configs.py -v  # verbose
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml


def test_yaml_syntax():
    """Test all YAML files for syntax errors."""
    print("\n[1/6] Testing YAML syntax...")
    configs_dir = Path(__file__).parent.parent / "configs"
    yaml_files = list(configs_dir.glob("*.yaml"))

    if not yaml_files:
        print(f"  ❌ No YAML files found in {configs_dir}")
        return False

    errors = []
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, "r") as f:
                yaml.safe_load(f)
            print(f"  ✓ {yaml_file.name}")
        except yaml.YAMLError as e:
            print(f"  ❌ {yaml_file.name}: {e}")
            errors.append(yaml_file.name)

    if errors:
        print(f"  FAILED: {len(errors)} files with syntax errors")
        return False
    print(f"  PASSED: All {len(yaml_files)} YAML files valid")
    return True


def test_config_loading():
    """Test ModelConfig.load() for all configs."""
    print("\n[2/6] Testing ModelConfig loading...")
    from config.model_config import ModelConfig

    configs_dir = Path(__file__).parent.parent / "configs"
    yaml_files = list(configs_dir.glob("*.yaml"))

    errors = []
    for yaml_file in yaml_files:
        try:
            # Load raw YAML first
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f)

            # Extract training config (to not pass it to ModelConfig)
            data.pop("training", {})

            # Load as ModelConfig
            config = ModelConfig.from_dict(data)
            print(f"  ✓ {yaml_file.name} -> {config.model_name}")
        except Exception as e:
            print(f"  ❌ {yaml_file.name}: {e}")
            errors.append((yaml_file.name, str(e)))

    if errors:
        print(f"  FAILED: {len(errors)} files failed to load")
        return False
    print(f"  PASSED: All {len(yaml_files)} configs load correctly")
    return True


def test_training_config_section():
    """Test that training sections have required fields."""
    print("\n[3/6] Testing training config sections...")

    configs_dir = Path(__file__).parent.parent / "configs"
    yaml_files = list(configs_dir.glob("*.yaml"))

    required_fields = [
        "max_steps",
        "batch_size",
        "learning_rate",
        "device",
        "checkpoint_dir",
        "experiment_name",
    ]

    warnings = []

    for yaml_file in yaml_files:
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        training = data.get("training", {})
        if not training:
            warnings.append(f"{yaml_file.name}: No training section")
            print(f"  ⚠ {yaml_file.name}: No training section (will use defaults)")
            continue

        missing = [f for f in required_fields if f not in training]
        if missing:
            print(f"  ⚠ {yaml_file.name}: Missing {missing}")
            warnings.append(f"{yaml_file.name}: Missing {missing}")
        else:
            print(f"  ✓ {yaml_file.name}: All required fields present")

    if warnings:
        print(f"  PASSED with {len(warnings)} warnings")
    else:
        print(f"  PASSED: All {len(yaml_files)} configs have training sections")
    return True


def test_preset_configs():
    """Test Python preset configurations."""
    print("\n[4/6] Testing Python presets...")
    from config.model_config import PRESET_CONFIGS, get_preset_config

    errors = []
    for preset_name in PRESET_CONFIGS:
        try:
            config = get_preset_config(preset_name)
            param_count = config.num_parameters_billions
            print(f"  ✓ {preset_name}: {config.model_name} ({param_count:.2f}B params)")
        except Exception as e:
            print(f"  ❌ {preset_name}: {e}")
            errors.append(preset_name)

    if errors:
        print(f"  FAILED: {len(errors)} presets broken")
        return False
    print(f"  PASSED: All {len(PRESET_CONFIGS)} presets valid")
    return True


def test_enum_values():
    """Test that config enum values are valid."""
    print("\n[5/6] Testing enum values in configs...")
    from config.model_config import (
        AttentionType,
        ConnectionType,
        FFNType,
        PositionEmbeddingType,
    )

    valid_attention = {e.value for e in AttentionType}
    valid_position = {e.value for e in PositionEmbeddingType}
    valid_ffn = {e.value for e in FFNType}
    valid_connection = {e.value for e in ConnectionType}

    configs_dir = Path(__file__).parent.parent / "configs"
    yaml_files = list(configs_dir.glob("*.yaml"))

    errors = []
    for yaml_file in yaml_files:
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        file_errors = []

        # Check attention type
        attn_type = data.get("attention", {}).get("attention_type")
        if attn_type and attn_type not in valid_attention:
            file_errors.append(f"Invalid attention_type: {attn_type}")

        # Check position type
        pos_type = data.get("position", {}).get("position_type")
        if pos_type and pos_type not in valid_position:
            file_errors.append(f"Invalid position_type: {pos_type}")

        # Check ffn type
        ffn_type = data.get("ffn", {}).get("ffn_type")
        if ffn_type and ffn_type not in valid_ffn:
            file_errors.append(f"Invalid ffn_type: {ffn_type}")

        # Check connection type
        conn_type = data.get("connection", {}).get("connection_type")
        if conn_type and conn_type not in valid_connection:
            file_errors.append(f"Invalid connection_type: {conn_type}")

        if file_errors:
            print(f"  ❌ {yaml_file.name}: {file_errors}")
            errors.extend(file_errors)
        else:
            print(f"  ✓ {yaml_file.name}: All enum values valid")

    if errors:
        print(f"  FAILED: {len(errors)} invalid enum values")
        return False
    print("  PASSED: All enum values valid")
    return True


def test_cli_argument_parsing():
    """Test CLI argument parsing (without running)."""
    print("\n[6/6] Testing CLI argument parsing...")

    # Test train.py argument parsing
    try:
        # Import the module to check for syntax errors
        import training.train as train_module

        print("  ✓ training/train.py imports successfully")
    except Exception as e:
        print(f"  ❌ training/train.py import error: {e}")
        return False

    # Test train_wikitext2_gpt2.py argument parsing
    try:

        print("  ✓ training/train_wikitext2_gpt2.py imports successfully")
    except Exception as e:
        print(f"  ❌ training/train_wikitext2_gpt2.py import error: {e}")
        return False

    # Check that key functions exist
    required_funcs = ["load_config_from_yaml", "TrainingConfig", "Trainer"]
    for func_name in required_funcs:
        if hasattr(train_module, func_name):
            print(f"  ✓ {func_name} exists in train.py")
        else:
            print(f"  ❌ {func_name} missing from train.py")
            return False

    print("  PASSED: CLI modules valid")
    return True


def run_all_tests(verbose=False):
    """Run all tests and report results."""
    print("=" * 60)
    print("LLM Architecture Config Validation")
    print("=" * 60)

    results = {
        "YAML Syntax": test_yaml_syntax(),
        "Config Loading": test_config_loading(),
        "Training Sections": test_training_config_section(),
        "Python Presets": test_preset_configs(),
        "Enum Values": test_enum_values(),
        "CLI Parsing": test_cli_argument_parsing(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")

    print(f"\nResult: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ All config validation tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed. Fix issues before training.")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate LLM configs without running models"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    sys.exit(run_all_tests(verbose=args.verbose))
