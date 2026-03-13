"""
Generate Model Outputs — Inference Script
==========================================
Reads evaluation_prompts.json, calls the base model and the SFT model for
every prompt, and writes (or updates) model_outputs.csv so it can be fed
directly into run_evaluation.py --mode csv.

Inference modes
---------------
  api   (default) — OpenAI-compatible REST API (remote or local servers
                    such as vLLM, Ollama, LM Studio, Together AI, OpenAI).
  local           — HuggingFace Transformers loaded directly into memory.
                    No network calls; models run on your GPU/CPU.

Features
--------
- Resume support      : prompts already present in the CSV are skipped
                        (override with --force-rerun)
- Parallel inference  : API mode fires base + SFT calls concurrently;
                        local mode runs sequentially on one GPU by default
                        (use --parallel-local for multi-GPU setups)
- Quantization        : --load-in-4bit / --load-in-8bit via bitsandbytes
- Rate limiting       : configurable delay between prompts
- Checkpoint          : CSV flushed every N prompts (default 5)
- Filters             : --categories / --difficulties / --prompt-ids
- Dry-run             : preview plan without touching any model

Usage examples
--------------
# --- API mode (default) ---
python generate_outputs.py --config ../config/config.yaml

# Override model names / endpoints from CLI
python generate_outputs.py \\
  --base-model meta-llama/Llama-3.1-8B \\
  --base-endpoint http://localhost:8000/v1 \\
  --sft-model  meta-llama/Llama-3.1-8B-SFT-v1 \\
  --sft-endpoint  http://localhost:8001/v1

# --- Local mode (HuggingFace Transformers) ---
python generate_outputs.py \\
  --inference-mode local \\
  --base-model-path /models/llama-3.1-8b \\
  --sft-model-path  /models/llama-3.1-8b-sft \\
  --device cuda

# Local with 4-bit quantization (saves GPU VRAM)
python generate_outputs.py \\
  --inference-mode local \\
  --base-model-path meta-llama/Llama-3.1-8B \\
  --sft-model-path  ./checkpoints/sft-v1 \\
  --device cuda --load-in-4bit

# Resume interrupted run / dry-run / filters — same flags as before
python generate_outputs.py --inference-mode local ... --dry-run
python generate_outputs.py --inference-mode local ... --categories factual_qa
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "prompt_id",
    "category",
    "sub_category",
    "difficulty",
    "prompt_text",
    "base_output",
    "sft_output",
    "base_model",
    "sft_model",
    "timestamp",
    "base_error",
    "sft_error",
    "annotator_name",
    "annotation_date",
    "notes",
]


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load config.yaml; return empty dict if file missing."""
    p = Path(config_path)
    if not p.exists():
        logger.warning("Config not found at %s — using defaults / env vars.", config_path)
        return {}
    try:
        import yaml  # type: ignore
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        logger.warning("PyYAML not installed — reading config as plain JSON fallback.")
        return {}


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

def load_prompts(prompts_path: str) -> list[dict]:
    """Load prompts JSON and return the list of prompt dicts."""
    path = Path(prompts_path)
    if not path.exists():
        # Try relative to this file
        path = Path(__file__).parent.parent / "prompts" / "evaluation_prompts.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = data.get("prompts", [])
    logger.info("Loaded %d prompts from %s", len(prompts), path)
    return prompts


# ---------------------------------------------------------------------------
# Existing CSV loader (for resume support)
# ---------------------------------------------------------------------------

def load_existing_outputs(csv_path: str) -> set[str]:
    """
    Return the set of prompt_ids already present in the CSV.
    Used to skip already-completed prompts when resuming.
    """
    p = Path(csv_path)
    if not p.exists():
        return set()
    done: set[str] = set()
    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("prompt_id", "").strip()
            # Only count as done if both outputs are non-empty and not error placeholders
            base_ok = row.get("base_output", "").strip() and not row.get("base_output", "").startswith("[ERROR]")
            sft_ok = row.get("sft_output", "").strip() and not row.get("sft_output", "").startswith("[ERROR]")
            if pid and base_ok and sft_ok:
                done.add(pid)
    logger.info("Found %d previously completed prompts in %s", len(done), csv_path)
    return done


# ---------------------------------------------------------------------------
# API caller
# ---------------------------------------------------------------------------

def call_model_api(
    prompt_text: str,
    model_name: str,
    endpoint: str,
    api_key: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    retries: int = 3,
    backoff: float = 2.0,
) -> tuple[str, str]:
    """
    Call an OpenAI-compatible chat completions API.

    Returns
    -------
    (output_text, error_message)  —  error_message is "" on success.
    """
    try:
        import openai  # type: ignore
    except ImportError:
        msg = "openai package not installed. Run: pip install openai"
        logger.error(msg)
        return "", f"[ERROR] {msg}"

    client_kwargs: dict[str, Any] = {"api_key": api_key or "dummy"}
    if endpoint:
        client_kwargs["base_url"] = endpoint

    client = openai.OpenAI(**client_kwargs)

    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = resp.choices[0].message.content or ""
            return text.strip(), ""
        except Exception as exc:
            logger.warning(
                "  [%s] Attempt %d/%d failed: %s", model_name, attempt, retries, exc
            )
            if attempt < retries:
                sleep_time = backoff * (2 ** (attempt - 1))
                logger.info("  Retrying in %.1f s …", sleep_time)
                time.sleep(sleep_time)
            else:
                err = f"[ERROR] Failed after {retries} attempts: {exc}"
                return "", err


# ---------------------------------------------------------------------------
# Local HuggingFace model runner
# ---------------------------------------------------------------------------

# Global registry so the same model is never loaded more than once per run.
_local_runners: dict[str, "LocalModelRunner"] = {}


class LocalModelRunner:
    """
    Wraps a HuggingFace causal-LM model + tokenizer for local inference.

    The model is loaded lazily on the first call to .generate().
    Subsequent calls reuse the already-loaded model (no double-loading).

    Parameters
    ----------
    model_path      : HF hub model ID or local directory path.
    device          : 'cuda', 'cpu', 'mps', or 'auto' (default).
    load_in_4bit    : Enable 4-bit NF4 quantization (requires bitsandbytes).
    load_in_8bit    : Enable 8-bit LLM.int8 quantization (requires bitsandbytes).
    trust_remote_code: Pass trust_remote_code=True to from_pretrained.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.trust_remote_code = trust_remote_code
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        """Load model and tokenizer into memory (called once)."""
        try:
            from transformers import (  # type: ignore
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
            import torch  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for local inference. "
                "Install them with: pip install torch transformers accelerate"
            ) from exc

        logger.info("Loading tokenizer from: %s", self.model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        # Ensure a pad token exists
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Build quantization config
        quant_cfg = None
        if self.load_in_4bit:
            logger.info("Using 4-bit NF4 quantization (bitsandbytes).")
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif self.load_in_8bit:
            logger.info("Using 8-bit LLM.int8 quantization (bitsandbytes).")
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

        # Determine device map
        device_map: Any = "auto"
        if self.device not in ("auto", ""):
            device_map = self.device

        logger.info("Loading model from: %s  (device=%s)", self.model_path, device_map)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=device_map,
            quantization_config=quant_cfg,
            trust_remote_code=self.trust_remote_code,
            torch_dtype="auto" if quant_cfg is None else None,
        )
        self._model.eval()
        logger.info("Model loaded: %s", self.model_path)

    def generate(
        self,
        prompt_text: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> tuple[str, str]:
        """
        Run inference. Returns (output_text, error_message).
        """
        if self._model is None:
            try:
                self._load()
            except Exception as exc:
                return "", f"[ERROR] Model load failed: {exc}"

        try:
            import torch  # type: ignore

            # Use chat template if tokenizer supports it, else raw prompt
            if hasattr(self._tokenizer, "chat_template") and self._tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt_text}]
                inputs_text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                inputs_text = prompt_text

            inputs = self._tokenizer(
                inputs_text, return_tensors="pt", truncation=True, max_length=4096
            )
            # Move input tensors to the same device as the model
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self._tokenizer.eos_token_id,
                "do_sample": temperature > 0.0,
            }
            if temperature > 0.0:
                gen_kwargs["temperature"] = temperature

            with torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs, **gen_kwargs
                )

            # Strip the prompt tokens from the output
            input_len = inputs["input_ids"].shape[1]
            new_ids = output_ids[0][input_len:]
            decoded = self._tokenizer.decode(new_ids, skip_special_tokens=True)
            return decoded.strip(), ""

        except Exception as exc:
            return "", f"[ERROR] Inference failed: {exc}"


def _get_local_runner(
    model_path: str,
    device: str,
    load_in_4bit: bool,
    load_in_8bit: bool,
    trust_remote_code: bool,
) -> LocalModelRunner:
    """Return a cached LocalModelRunner, creating it on first use."""
    key = model_path
    if key not in _local_runners:
        _local_runners[key] = LocalModelRunner(
            model_path=model_path,
            device=device,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            trust_remote_code=trust_remote_code,
        )
    return _local_runners[key]


# ---------------------------------------------------------------------------
# Inference dispatcher — API or Local
# ---------------------------------------------------------------------------

def infer_single_prompt(
    prompt: dict,
    base_cfg: dict,
    sft_cfg: dict,
    gen_cfg: dict,
    inference_mode: str = "api",
    parallel_local: bool = False,
) -> dict:
    """
    Call base and SFT models for one prompt.

    Parameters
    ----------
    inference_mode  : 'api' or 'local'
    parallel_local  : If True and mode is 'local', attempt to run both
                      models in parallel threads (useful for multi-GPU).
                      Default False (sequential; safe for single GPU).

    Returns
    -------
    dict with CSV row fields.
    """
    prompt_text = prompt.get("prompt", "")
    max_tokens = gen_cfg.get("max_tokens", 1024)
    temperature = gen_cfg.get("temperature", 0.0)

    if inference_mode == "local":
        base_output, base_error = _run_local(
            prompt_text=prompt_text,
            model_cfg=base_cfg,
            max_tokens=max_tokens,
            temperature=temperature,
            label="base",
        )
        if not parallel_local:
            # Sequential: free activations from base before loading SFT
            sft_output, sft_error = _run_local(
                prompt_text=prompt_text,
                model_cfg=sft_cfg,
                max_tokens=max_tokens,
                temperature=temperature,
                label="sft",
            )
        else:
            # Parallel threads — only sensible with separate GPUs
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                sft_future = pool.submit(
                    _run_local, prompt_text, sft_cfg, max_tokens, temperature, "sft"
                )
                sft_output, sft_error = sft_future.result()

        base_label = base_cfg.get("local_model_path") or base_cfg.get("model_name", "base")
        sft_label = sft_cfg.get("local_model_path") or sft_cfg.get("model_name", "sft")

    else:  # api
        retries = gen_cfg.get("retries", 3)
        backoff = gen_cfg.get("backoff_seconds", 2.0)
        base_key = base_cfg.get("api_key") or os.environ.get("BASE_MODEL_API_KEY", "dummy")
        sft_key = sft_cfg.get("api_key") or os.environ.get("SFT_MODEL_API_KEY", "dummy")

        # Fire both API calls at the same time
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            base_future = pool.submit(
                call_model_api,
                prompt_text,
                base_cfg["model_name"],
                base_cfg.get("endpoint", ""),
                base_key,
                max_tokens,
                temperature,
                retries,
                backoff,
            )
            sft_future = pool.submit(
                call_model_api,
                prompt_text,
                sft_cfg["model_name"],
                sft_cfg.get("endpoint", ""),
                sft_key,
                max_tokens,
                temperature,
                retries,
                backoff,
            )
            base_output, base_error = base_future.result()
            sft_output, sft_error = sft_future.result()

        base_label = base_cfg.get("model_name", "base")
        sft_label = sft_cfg.get("model_name", "sft")

    return {
        "prompt_id": prompt["id"],
        "category": prompt.get("category", ""),
        "sub_category": prompt.get("sub_category", ""),
        "difficulty": prompt.get("difficulty", ""),
        "prompt_text": prompt_text,
        "base_output": base_output,
        "sft_output": sft_output,
        "base_model": base_label,
        "sft_model": sft_label,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_error": base_error,
        "sft_error": sft_error,
        "annotator_name": "",
        "annotation_date": "",
        "notes": "",
    }


def _run_local(
    prompt_text: str,
    model_cfg: dict,
    max_tokens: int,
    temperature: float,
    label: str,
) -> tuple[str, str]:
    """Resolve local config fields and delegate to LocalModelRunner."""
    model_path = (
        model_cfg.get("local_model_path")
        or model_cfg.get("model_name", "")
    )
    if not model_path:
        return "", f"[ERROR] No model path configured for {label} model."

    device = model_cfg.get("local_device", "auto")
    load_in_4bit = bool(model_cfg.get("load_in_4bit", False))
    load_in_8bit = bool(model_cfg.get("load_in_8bit", False))
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))

    runner = _get_local_runner(model_path, device, load_in_4bit, load_in_8bit, trust_remote_code)
    return runner.generate(prompt_text, max_new_tokens=max_tokens, temperature=temperature)


# ---------------------------------------------------------------------------
# CSV writer helpers
# ---------------------------------------------------------------------------

def _open_csv_writer(csv_path: str, append: bool) -> tuple[Any, Any]:
    """Open the CSV for writing and return (file_handle, DictWriter)."""
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    f = open(csv_path, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if not append:
        writer.writeheader()
    return f, writer


def _flush_rows(rows: list[dict], csv_path: str, append: bool) -> None:
    """Write a batch of rows to the CSV and flush."""
    if not rows:
        return
    f, writer = _open_csv_writer(csv_path, append=append)
    try:
        writer.writerows(rows)
        f.flush()
    finally:
        f.close()


# ---------------------------------------------------------------------------
# Progress printer
# ---------------------------------------------------------------------------

def _print_row_summary(row: dict, idx: int, total: int) -> None:
    base_status = "✓" if not row["base_error"] else "✗"
    sft_status = "✓" if not row["sft_error"] else "✗"
    logger.info(
        "[%d/%d] %-12s  base=%s  sft=%s",
        idx, total, row["prompt_id"], base_status, sft_status,
    )
    if row["base_error"]:
        logger.warning("  BASE error: %s", row["base_error"][:120])
    if row["sft_error"]:
        logger.warning("  SFT  error: %s", row["sft_error"][:120])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Call base and SFT models for every prompt in evaluation_prompts.json "
            "and write outputs to model_outputs.csv."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="../config/config.yaml",
        help="Path to config.yaml with model endpoints and generation settings.",
    )
    parser.add_argument(
        "--prompts",
        default="../prompts/evaluation_prompts.json",
        help="Path to the evaluation prompts JSON.",
    )
    parser.add_argument(
        "--output",
        default="../data/model_outputs.csv",
        help="Path of the CSV to write (or update) with model outputs.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run inference even for prompts already in the CSV.",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        metavar="CATEGORY",
        help="Only run prompts in these categories (space-separated).",
    )
    parser.add_argument(
        "--difficulties",
        nargs="*",
        choices=["easy", "medium", "hard"],
        metavar="DIFFICULTY",
        help="Only run prompts at these difficulty levels.",
    )
    parser.add_argument(
        "--prompt-ids",
        nargs="*",
        metavar="ID",
        help="Run only specific prompt IDs (e.g. IF_001 FQ_003).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        metavar="N",
        help="Flush CSV after every N prompts (default: 5).",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=0.5,
        metavar="SECONDS",
        help="Sleep N seconds between prompts to avoid rate limiting (default: 0.5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which prompts would be called without making any API calls.",
    )
    parser.add_argument(
        "--base-model",
        default="",
        help="Override base model name (takes priority over config.yaml).",
    )
    parser.add_argument(
        "--sft-model",
        default="",
        help="Override SFT model name (takes priority over config.yaml).",
    )
    parser.add_argument(
        "--base-endpoint",
        default="",
        help="Override base model API endpoint URL.",
    )
    parser.add_argument(
        "--sft-endpoint",
        default="",
        help="Override SFT model API endpoint URL.",
    )

    # ------------------------------------------------------------------
    # Local inference arguments
    # ------------------------------------------------------------------
    local_group = parser.add_argument_group(
        "local inference",
        "Options for --inference-mode local (HuggingFace Transformers).",
    )
    local_group.add_argument(
        "--inference-mode",
        choices=["api", "local"],
        default="api",
        help="'api' = OpenAI-compatible endpoint (default); 'local' = HF Transformers in-process.",
    )
    local_group.add_argument(
        "--base-model-path",
        default="",
        help="Local path or HF hub ID for the base model (local mode). "
             "Falls back to --base-model if not set.",
    )
    local_group.add_argument(
        "--sft-model-path",
        default="",
        help="Local path or HF hub ID for the SFT model (local mode). "
             "Falls back to --sft-model if not set.",
    )
    local_group.add_argument(
        "--device",
        default="auto",
        help="Device for local inference: 'cuda', 'cpu', 'mps', or 'auto' (default).",
    )
    local_group.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Enable 4-bit NF4 quantization via bitsandbytes (requires CUDA). "
             "Reduces VRAM usage by ~75%%.",
    )
    local_group.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Enable 8-bit LLM.int8 quantization via bitsandbytes (requires CUDA). "
             "Reduces VRAM usage by ~50%%.",
    )
    local_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to HuggingFace from_pretrained (needed for "
             "some models like Falcon, Phi, etc.).",
    )
    local_group.add_argument(
        "--parallel-local",
        action="store_true",
        help="Run base and SFT local models in parallel threads. Only useful for "
             "multi-GPU setups. Single-GPU users should leave this off (default: sequential).",
    )

    args = parser.parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        parser.error("--load-in-4bit and --load-in-8bit are mutually exclusive.")

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    cfg = load_config(args.config)
    base_cfg: dict = cfg.get("models", {}).get("base", {})
    sft_cfg: dict = cfg.get("models", {}).get("sft", {})
    gen_cfg: dict = cfg.get("generation", {})

    # ------------------------------------------------------------------
    # CLI overrides — API fields
    # ------------------------------------------------------------------
    if args.base_model:
        base_cfg["model_name"] = args.base_model
    if args.sft_model:
        sft_cfg["model_name"] = args.sft_model
    if args.base_endpoint:
        base_cfg["endpoint"] = args.base_endpoint
    if args.sft_endpoint:
        sft_cfg["endpoint"] = args.sft_endpoint

    # ------------------------------------------------------------------
    # CLI overrides — local inference fields
    # ------------------------------------------------------------------
    inference_mode: str = args.inference_mode

    if inference_mode == "local":
        # model path: CLI flag > config local section > fall back to model_name
        local_base_cfg = cfg.get("models", {}).get("base", {}).get("local", {})
        local_sft_cfg  = cfg.get("models", {}).get("sft",  {}).get("local", {})

        base_cfg["local_model_path"] = (
            args.base_model_path
            or local_base_cfg.get("model_path", "")
            or base_cfg.get("model_name", "")
        )
        sft_cfg["local_model_path"] = (
            args.sft_model_path
            or local_sft_cfg.get("model_path", "")
            or sft_cfg.get("model_name", "")
        )
        base_cfg["local_device"]     = args.device or local_base_cfg.get("device", "auto")
        sft_cfg["local_device"]      = args.device or local_sft_cfg.get("device", "auto")
        base_cfg["load_in_4bit"]     = args.load_in_4bit or local_base_cfg.get("load_in_4bit", False)
        sft_cfg["load_in_4bit"]      = args.load_in_4bit or local_sft_cfg.get("load_in_4bit", False)
        base_cfg["load_in_8bit"]     = args.load_in_8bit or local_base_cfg.get("load_in_8bit", False)
        sft_cfg["load_in_8bit"]      = args.load_in_8bit or local_sft_cfg.get("load_in_8bit", False)
        base_cfg["trust_remote_code"]= args.trust_remote_code or local_base_cfg.get("trust_remote_code", False)
        sft_cfg["trust_remote_code"] = args.trust_remote_code or local_sft_cfg.get("trust_remote_code", False)

        if not base_cfg["local_model_path"]:
            parser.error(
                "Base model path not set. Use --base-model-path or set "
                "models.base.local.model_path in config.yaml."
            )
        if not sft_cfg["local_model_path"]:
            parser.error(
                "SFT model path not set. Use --sft-model-path or set "
                "models.sft.local.model_path in config.yaml."
            )
    else:
        # API mode validation
        if not base_cfg.get("model_name"):
            parser.error(
                "Base model name not configured. Set models.base.model_name in config.yaml "
                "or pass --base-model."
            )
        if not sft_cfg.get("model_name"):
            parser.error(
                "SFT model name not configured. Set models.sft.model_name in config.yaml "
                "or pass --sft-model."
            )

    # ------------------------------------------------------------------
    # Load prompts
    # ------------------------------------------------------------------
    prompts = load_prompts(args.prompts)

    # Apply filters
    if args.prompt_ids:
        prompts = [p for p in prompts if p["id"] in args.prompt_ids]
        logger.info("Filtered to %d prompts by ID.", len(prompts))
    if args.categories:
        prompts = [p for p in prompts if p.get("category") in args.categories]
        logger.info("Filtered to %d prompts by category.", len(prompts))
    if args.difficulties:
        prompts = [p for p in prompts if p.get("difficulty") in args.difficulties]
        logger.info("Filtered to %d prompts by difficulty.", len(prompts))

    if not prompts:
        logger.error("No prompts match the given filters. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Resume: skip already-done prompts
    # ------------------------------------------------------------------
    done_ids: set[str] = set()
    if not args.force_rerun:
        done_ids = load_existing_outputs(args.output)

    pending = [p for p in prompts if p["id"] not in done_ids]
    skipped = len(prompts) - len(pending)

    logger.info(
        "Prompts: %d total | %d pending | %d skipped (already done)",
        len(prompts), len(pending), skipped,
    )

    # ------------------------------------------------------------------
    # Dry-run
    # ------------------------------------------------------------------
    if args.dry_run:
        print(f"\n{'=' * 60}")
        print(f"  DRY RUN — {len(pending)} prompts would be called")
        print(f"  Inference mode : {inference_mode.upper()}")
        if inference_mode == "local":
            quant = "4-bit" if args.load_in_4bit else ("8-bit" if args.load_in_8bit else "none")
            print(f"  Base model     : {base_cfg['local_model_path']}")
            print(f"  SFT  model     : {sft_cfg['local_model_path']}")
            print(f"  Device         : {base_cfg.get('local_device', 'auto')}")
            print(f"  Quantization   : {quant}")
            print(f"  Parallel       : {'yes' if args.parallel_local else 'no (sequential)'}")
        else:
            print(f"  Base model     : {base_cfg['model_name']}")
            print(f"    endpoint     : {base_cfg.get('endpoint') or '(OpenAI default)'}")
            print(f"  SFT  model     : {sft_cfg['model_name']}")
            print(f"    endpoint     : {sft_cfg.get('endpoint') or '(OpenAI default)'}")
        print(f"  Output CSV     : {args.output}")
        print(f"{'=' * 60}")
        for p in pending:
            print(f"  {p['id']:<12}  [{p.get('difficulty','?'):6}]  {p.get('category','')}")
        print(f"\nRe-run without --dry-run to execute inference.")
        sys.exit(0)

    if not pending:
        logger.info("All prompts already completed. Use --force-rerun to re-run.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    csv_exists = Path(args.output).exists()
    # If we're appending, preserve existing rows; otherwise start fresh
    append_mode = csv_exists and not args.force_rerun

    if args.force_rerun and csv_exists:
        # Back up the existing CSV before overwriting
        backup_path = args.output + ".bak"
        import shutil
        shutil.copy2(args.output, backup_path)
        logger.info("Existing CSV backed up to %s", backup_path)
        append_mode = False

    print(f"\n{'=' * 60}")
    print(f"  SFT Inference Run — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Mode       : {inference_mode.upper()}")
    if inference_mode == "local":
        quant = "4-bit" if args.load_in_4bit else ("8-bit" if args.load_in_8bit else "fp16/auto")
        print(f"  Base model : {base_cfg['local_model_path']}")
        print(f"  SFT  model : {sft_cfg['local_model_path']}")
        print(f"  Device     : {base_cfg.get('local_device', 'auto')}  Quant: {quant}")
        print(f"  Parallel   : {'yes' if args.parallel_local else 'sequential (single GPU safe)'}")
    else:
        print(f"  Base model : {base_cfg['model_name']}")
        print(f"  SFT  model : {sft_cfg['model_name']}")
    print(f"  Prompts    : {len(pending)}")
    print(f"  Output     : {args.output}")
    print(f"  Resume     : {'yes' if append_mode else 'no'}")
    print(f"{'=' * 60}\n")

    buffer: list[dict] = []
    errors_base = 0
    errors_sft = 0

    for idx, prompt in enumerate(pending, 1):
        logger.info(
            "[%d/%d] Calling models for prompt: %s", idx, len(pending), prompt["id"]
        )

        row = infer_single_prompt(
            prompt, base_cfg, sft_cfg, gen_cfg,
            inference_mode=inference_mode,
            parallel_local=args.parallel_local,
        )
        buffer.append(row)

        if row["base_error"]:
            errors_base += 1
        if row["sft_error"]:
            errors_sft += 1

        _print_row_summary(row, idx, len(pending))

        # Checkpoint flush
        if len(buffer) >= args.checkpoint_every:
            _flush_rows(buffer, args.output, append=append_mode or idx > args.checkpoint_every)
            logger.info("  → Flushed %d rows to %s", len(buffer), args.output)
            buffer.clear()
            append_mode = True  # subsequent writes always append

        # Rate-limit delay between prompts (not after the last one)
        if idx < len(pending) and args.rate_limit_delay > 0:
            time.sleep(args.rate_limit_delay)

    # Final flush
    if buffer:
        _flush_rows(buffer, args.output, append=append_mode)
        logger.info("  → Flushed final %d rows to %s", len(buffer), args.output)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_written = len(pending)
    print(f"\n{'=' * 60}")
    print(f"  INFERENCE COMPLETE")
    print(f"  Prompts written  : {total_written}")
    print(f"  Base errors      : {errors_base}")
    print(f"  SFT  errors      : {errors_sft}")
    print(f"  Output CSV       : {args.output}")
    print(f"\n  Next step:")
    print(f"    cd evaluation")
    print(f"    python run_evaluation.py --mode csv --input {args.output}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
