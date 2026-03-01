from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import get_safe_filename


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the Hindi full report markdown (Trans vs Ground).")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--trans_dir", default="")
    ap.add_argument("--ground_dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--paper_json", default="")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    safe_model = get_safe_filename(args.model)

    trans_dir = Path(args.trans_dir) if args.trans_dir else (repo_root / "results" / "hf" / safe_model / "hi_trans_full")
    ground_dir = Path(args.ground_dir) if args.ground_dir else (repo_root / "results" / "hf" / safe_model / "hi_ground_full")
    out_path = Path(args.out) if args.out else (repo_root / "results" / "hf" / safe_model / "hi_full_report.md")

    script = repo_root / "scripts" / "generate_hi_full_report.py"

    cmd = [
        sys.executable,
        str(script),
        "--model",
        args.model,
        "--trans_dir",
        str(trans_dir),
        "--ground_dir",
        str(ground_dir),
        "--out",
        str(out_path),
    ]
    if args.paper_json:
        cmd += ["--paper_json", args.paper_json]

    print("Generating Hindi full report...")
    print(f"Model:   {args.model}")
    print(f"Trans:   {trans_dir}")
    print(f"Ground:  {ground_dir}")
    print(f"OutFile: {out_path}")
    if args.paper_json:
        print(f"PaperJson: {args.paper_json}")

    rc = subprocess.call(cmd, cwd=str(repo_root))
    if rc != 0:
        raise SystemExit(
            "Report generation failed. Make sure TransDir and GroundDir each contain a results.json. "
            f"(exit code: {rc})"
        )

    print(f"Done. Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
