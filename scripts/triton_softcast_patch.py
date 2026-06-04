#!/usr/bin/env python3
"""Patch Triton's tl.dot semantic check for the 120B B300 runtime.

The 120B B300 environment used a nightly PyTorch/cu130 stack with Triton
3.6.x-era sources. Some FLA kernels pass mixed bf16/fp32 operands into tl.dot.
Unpatched Triton asserts that both operands have identical dtype and can crash
on the first backward path that reaches those kernels.

Usage:
    python scripts/triton_softcast_patch.py \
      /path/to/site-packages/triton/language/semantic.py

After patching, clear the Triton cache before launching training.
"""
import sys
from pathlib import Path

OLD = """        else:
            assert lhs.dtype in (tl.int8, tl.uint8, tl.float16, tl.bfloat16, tl.float32,
                                 tl.float64), f"Unsupported lhs dtype {lhs.dtype}"
            assert rhs.dtype in (tl.int8, tl.uint8, tl.float16, tl.bfloat16, tl.float32,
                                 tl.float64), f"Unsupported rhs dtype {rhs.dtype}"
            assert lhs.dtype == rhs.dtype, f"Both operands must be same dtype. Got {lhs.dtype} and {rhs.dtype}"""

NEW = """        else:
            assert lhs.dtype in (tl.int8, tl.uint8, tl.float16, tl.bfloat16, tl.float32,
                                 tl.float64), f"Unsupported lhs dtype {lhs.dtype}"
            assert rhs.dtype in (tl.int8, tl.uint8, tl.float16, tl.bfloat16, tl.float32,
                                 tl.float64), f"Unsupported rhs dtype {rhs.dtype}"
            # SOFTCAST-2026-05-20: auto-cast on dtype mismatch.
            if lhs.dtype != rhs.dtype:
                if lhs.dtype.primitive_bitwidth < rhs.dtype.primitive_bitwidth:
                    lhs = self.cast(lhs, rhs.dtype)
                else:
                    rhs = self.cast(rhs, lhs.dtype)"""


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: triton_softcast_patch.py /path/to/triton/language/semantic.py")
        return 2

    path = Path(sys.argv[1])
    src = path.read_text()
    if "SOFTCAST-2026-05-20" in src:
        print("[skip] already patched")
        return 0
    if OLD not in src:
        print("[fail] anchor not found")
        return 1
    path.write_text(src.replace(OLD, NEW, 1))
    print("[ok] patch applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
