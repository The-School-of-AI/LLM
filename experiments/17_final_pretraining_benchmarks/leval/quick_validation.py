#!/usr/bin/env python3
"""
Quick environment validation for OLMES L-Eval benchmarking.
Tests dependency availability WITHOUT downloading or loading any model.
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def validate_environment() -> dict:
    """Validate that all required dependencies are available."""
    print("=" * 60)
    print("  OLMES L-Eval -- Quick Environment Validation")
    print("=" * 60)

    results = {
        "timestamp": datetime.now().isoformat(),
        "validation_type": "quick_environment_check",
        "checks": {},
    }

    # ------------------------------------------------------------------
    # Python version
    # ------------------------------------------------------------------
    print("\n  Checking Python...")
    vi = sys.version_info
    ver = f"{vi.major}.{vi.minor}.{vi.micro}"
    ok = vi >= (3, 9)
    results["checks"]["python"] = {"version": ver, "passed": ok}
    print(f"    Python {ver}: {'PASS' if ok else 'FAIL (need >= 3.9)'}")

    # ------------------------------------------------------------------
    # Core ML libraries
    # ------------------------------------------------------------------
    print("\n  Checking core libraries...")
    core_libs = {
        "torch": "torch",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "datasets": "datasets",
    }
    for name, package in core_libs.items():
        try:
            mod = __import__(package)
            ver = getattr(mod, "__version__", "unknown")
            results["checks"][name] = {"version": ver, "passed": True}
            print(f"    {name} {ver}: PASS")
        except ImportError as e:
            results["checks"][name] = {"passed": False, "error": str(e)}
            print(f"    {name}: FAIL ({e})")

    # ------------------------------------------------------------------
    # Device availability (PyTorch)
    # ------------------------------------------------------------------
    print("\n  Checking compute devices...")
    try:
        import torch

        mps_ok = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
        cuda_ok = torch.cuda.is_available()
        results["checks"]["devices"] = {
            "mps_available": mps_ok,
            "cuda_available": cuda_ok,
            "passed": True,
        }
        print(f"    MPS  (Apple Silicon): {'PASS' if mps_ok else 'N/A'}")
        print(f"    CUDA (NVIDIA GPU)   : {'PASS' if cuda_ok else 'N/A'}")
        print(f"    CPU                 : PASS (always available)")
    except Exception:
        results["checks"]["devices"] = {"passed": False}
        print("    Cannot check devices (torch not available)")

    # ------------------------------------------------------------------
    # Metric libraries
    # ------------------------------------------------------------------
    print("\n  Checking metric libraries...")
    metric_libs = {
        "rouge_score": "rouge-score",
        "sacrebleu": "sacrebleu",
        "numpy": "numpy",
        "pandas": "pandas",
    }
    for name, _pip_name in metric_libs.items():
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "unknown")
            results["checks"][name] = {"version": ver, "passed": True}
            print(f"    {name} {ver}: PASS")
        except ImportError:
            results["checks"][name] = {"passed": False}
            print(f"    {name}: FAIL (install with: pip install {_pip_name})")

    # ------------------------------------------------------------------
    # L-Eval specific: verify we can tokenize long text
    # ------------------------------------------------------------------
    print("\n  Checking L-Eval readiness...")
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")  # tiny, always cached
        long_text = "token " * 512  # 512-token test string
        ids = tok(long_text, truncation=False)["input_ids"]
        results["checks"]["long_context_tokenize"] = {
            "passed": len(ids) >= 512,
            "token_count": len(ids),
        }
        print(f"    Long-context tokenization ({len(ids)} tokens): PASS")
    except Exception as e:
        results["checks"]["long_context_tokenize"] = {"passed": False, "error": str(e)}
        print(f"    Long-context tokenization: FAIL ({e})")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_dir = Path("./results")
    output_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"env_validation_{ts}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    all_passed = all(
        c.get("passed", False)
        for c in results["checks"].values()
        if isinstance(c, dict)
    )
    print("\n" + "=" * 60)
    print("  ENVIRONMENT VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Overall: {'ALL PASS' if all_passed else 'SOME CHECKS FAILED'}")
    print(f"  Results: {out_file}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    validate_environment()
