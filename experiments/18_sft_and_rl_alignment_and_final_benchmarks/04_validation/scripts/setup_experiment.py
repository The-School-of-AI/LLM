#!/usr/bin/env python3
"""
Experiment Setup Script
========================
Downloads Qwen/Qwen2-0.5B (base) and Qwen/Qwen2-0.5B-Instruct (SFT) from
HuggingFace Hub into local model directories, then updates config/config.yaml
with paths for both API mode (local server endpoints) and local inference mode.

Model selection rationale
--------------------------
- Qwen2-0.5B  : ~1 GB download, runs on CPU without quantization.
  Clear base / instruction-tuned pair from the same model family.
- Base         : Qwen/Qwen2-0.5B          (pre-trained, no RLHF/SFT)
- SFT (Chat)   : Qwen/Qwen2-0.5B-Instruct (instruction-tuned — our "SFT" model)

Usage
-----
  # Full setup (download + config update)
  python scripts/setup_experiment.py

  # Custom download directory
  python scripts/setup_experiment.py --models-dir /data/models

  # Skip download (models already present), only update config
  python scripts/setup_experiment.py --skip-download

  # Use a different device for local inference
  python scripts/setup_experiment.py --device cuda

After running this script:
  1. Start inference servers:
       python scripts/inference_server.py --model-path models/base --port 8001
       python scripts/inference_server.py --model-path models/sft  --port 8002
  2. Run Option B (API mode):
       cd evaluation
       python generate_outputs.py --config ../config/config.yaml \\
           --prompt-ids IF_001 IF_002 FQ_001 RN_001 EC_001 --output ../data/model_outputs_test.csv
  3. Validate end-to-end:
       python scripts/validate_option_b.py
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

# Resolve project root regardless of where the script is called from
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH  = ROOT / "config" / "config.yaml"
MODELS_DIR   = ROOT / "models"

# Default model pair: SmolLM2-135M (base) vs SmolLM2-135M-Instruct (SFT).
# Both are ~270 MB each, run on CPU without quantization, and form a clean
# pre-trained / instruction-tuned pair for integration testing.
DEFAULT_BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_SFT_MODEL_ID  = "HuggingFaceTB/SmolLM2-135M-Instruct"

BASE_API_PORT = 8001
SFT_API_PORT  = 8002


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOLD  = "\033[1m"
GREEN = "\033[92m"
CYAN  = "\033[96m"
RESET = "\033[0m"


def info(msg: str) -> None:
    print(f"  {CYAN}→{RESET} {msg}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def header(msg: str) -> None:
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}  {msg}{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}")


# ---------------------------------------------------------------------------
# Step 1 — Install / check packages
# ---------------------------------------------------------------------------

def ensure_packages() -> None:
    """Verify that required packages are importable (no silent installs)."""
    header("Checking required packages")
    required = {
        "transformers": "transformers>=4.38.0",
        "huggingface_hub": "huggingface_hub>=0.20.0",
        "yaml": "pyyaml>=6.0.1",
        "fastapi": "fastapi>=0.110.0",
        "uvicorn": "uvicorn>=0.29.0",
        "openai": "openai>=1.12.0",
    }
    missing = []
    for module, pkg in required.items():
        try:
            __import__(module)
            ok(f"{module}")
        except ImportError:
            print(f"  ✗ {module}  (install: pip install {pkg})")
            missing.append(pkg)

    if missing:
        print(
            f"\n  Missing packages detected. Run:\n"
            f"  pip install {' '.join(missing)}\n"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step 2 — Download models
# ---------------------------------------------------------------------------

def download_model(model_id: str, local_dir: Path, skip: bool) -> None:
    """Download a HuggingFace model to local_dir."""
    if skip and local_dir.exists() and any(local_dir.iterdir()):
        ok(f"Already present: {local_dir}  (skipping download)")
        return

    from huggingface_hub import snapshot_download  # type: ignore

    info(f"Downloading {model_id}  →  {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        ignore_patterns=["*.msgpack", "flax_model*", "rust_model*", "tf_model*"],
    )
    ok(f"Downloaded: {local_dir}")


# ---------------------------------------------------------------------------
# Step 3 — Update config.yaml
# ---------------------------------------------------------------------------

def update_config(
    base_model_id: str,
    sft_model_id: str,
    base_local_path: Path,
    sft_local_path: Path,
    base_port: int,
    sft_port: int,
    device: str,
) -> None:
    """Rewrite relevant sections of config.yaml in-place."""
    import yaml  # type: ignore

    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # API mode fields
    cfg["models"]["base"]["model_name"] = base_model_id
    cfg["models"]["base"]["endpoint"]   = f"http://localhost:{base_port}/v1"
    cfg["models"]["sft"]["model_name"]  = sft_model_id
    cfg["models"]["sft"]["endpoint"]    = f"http://localhost:{sft_port}/v1"

    # Local mode fields
    cfg["models"]["base"]["local"]["model_path"]        = str(base_local_path)
    cfg["models"]["base"]["local"]["device"]            = device
    cfg["models"]["base"]["local"]["trust_remote_code"] = False

    cfg["models"]["sft"]["local"]["model_path"]         = str(sft_local_path)
    cfg["models"]["sft"]["local"]["device"]             = device
    cfg["models"]["sft"]["local"]["trust_remote_code"]  = False

    # Reduce max_tokens for small model (avoids slow CPU generation)
    cfg["generation"]["max_tokens"] = 256

    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)

    ok(f"config.yaml updated  ({CONFIG_PATH})")


# ---------------------------------------------------------------------------
# Step 4 — Print next steps
# ---------------------------------------------------------------------------

def print_next_steps(base_dir: Path, sft_dir: Path) -> None:
    header("Setup complete — next steps")
    print(f"""
  {BOLD}1. Start inference servers (two separate terminals or use start_servers.py):{RESET}

     # Terminal A — base model
     python scripts/inference_server.py --model-path "{base_dir}" --port {BASE_API_PORT}

     # Terminal B — SFT model
     python scripts/inference_server.py --model-path "{sft_dir}" --port {SFT_API_PORT}

     # OR — start both at once (background processes)
     python scripts/start_servers.py

  {BOLD}2. Run Option B — API mode inference (5 test prompts):{RESET}

     cd evaluation
     python generate_outputs.py \\
         --config ../config/config.yaml \\
         --prompt-ids IF_001 IF_002 FQ_001 RN_001 EC_001 \\
         --output ../data/model_outputs_test.csv

  {BOLD}3. Run evaluation on the outputs:{RESET}

     python run_evaluation.py \\
         --mode csv \\
         --input ../data/model_outputs_test.csv \\
         --prompts ../prompts/evaluation_prompts.json \\
         --output-dir ../results

  {BOLD}4. End-to-end validation checklist:{RESET}

     python scripts/validate_option_b.py

""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Qwen2-0.5B base+SFT models and update config.yaml.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-model-id",
        default=DEFAULT_BASE_MODEL_ID,
        help=f"HuggingFace hub ID for the base model (default: {DEFAULT_BASE_MODEL_ID}).",
    )
    parser.add_argument(
        "--sft-model-id",
        default=DEFAULT_SFT_MODEL_ID,
        help=f"HuggingFace hub ID for the SFT model  (default: {DEFAULT_SFT_MODEL_ID}).",
    )
    parser.add_argument(
        "--models-dir",
        default=str(MODELS_DIR),
        help=f"Directory to download models into (default: {MODELS_DIR})",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip model download; only update config.yaml.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device for local inference mode in config (default: cpu).",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=BASE_API_PORT,
        help=f"Port for the base model API server (default: {BASE_API_PORT}).",
    )
    parser.add_argument(
        "--sft-port",
        type=int,
        default=SFT_API_PORT,
        help=f"Port for the SFT model API server (default: {SFT_API_PORT}).",
    )
    args = parser.parse_args()

    models_dir   = Path(args.models_dir)
    base_dir     = models_dir / "base"
    sft_dir      = models_dir / "sft"
    base_model_id = args.base_model_id
    sft_model_id  = args.sft_model_id

    header("SFT Evaluation — Experiment Setup")
    print(f"  Base model : {base_model_id}  →  {base_dir}")
    print(f"  SFT  model : {sft_model_id}  →  {sft_dir}")
    print(f"  Config     : {CONFIG_PATH}")
    print(f"  Device     : {args.device}")
    print(f"  API ports  : base={args.base_port}  sft={args.sft_port}")

    # Step 1 — packages
    ensure_packages()

    # Step 2 — download
    header("Downloading models")
    download_model(base_model_id, base_dir, skip=args.skip_download)
    download_model(sft_model_id,  sft_dir,  skip=args.skip_download)

    # Step 3 — config
    header("Updating config.yaml")
    update_config(base_model_id, sft_model_id, base_dir, sft_dir, args.base_port, args.sft_port, args.device)

    # Step 4 — instructions
    print_next_steps(base_dir, sft_dir)


if __name__ == "__main__":
    main()
