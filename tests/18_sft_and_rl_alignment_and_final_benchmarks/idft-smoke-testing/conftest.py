"""Conftest for Team 18 IDFT smoke tests — adds quantization_support to sys.path."""

import sys
from pathlib import Path

# Add quantization_support directory to sys.path so tests can import modules directly
_qs_dir = str(
    Path(__file__).resolve().parent.parent.parent.parent
    / "experiments"
    / "18_sft_and_rl_alignment_and_final_benchmarks"
    / "quantization_support"
)
if _qs_dir not in sys.path:
    sys.path.insert(0, _qs_dir)
