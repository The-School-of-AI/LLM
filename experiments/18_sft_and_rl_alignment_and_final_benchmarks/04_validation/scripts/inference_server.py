#!/usr/bin/env python3
"""
Minimal OpenAI-Compatible Inference Server
==========================================
Wraps a HuggingFace causal-LM model in a FastAPI server that exposes
POST /v1/chat/completions and GET /v1/models — the subset of the OpenAI API
that generate_outputs.py and run_evaluation.py rely on.

Designed for local experimentation. Not intended for production use.

Dependencies
------------
  pip install fastapi uvicorn transformers torch accelerate

Usage
-----
  # Serve Qwen2-0.5B (base) on port 8001
  python scripts/inference_server.py --model-path models/base --port 8001

  # Serve Qwen2-0.5B-Instruct (SFT) on port 8002
  python scripts/inference_server.py --model-path models/sft --port 8002

  # Use GPU
  python scripts/inference_server.py --model-path models/base --port 8001 --device cuda

  # 4-bit quantization
  python scripts/inference_server.py --model-path models/base --port 8001 --load-in-4bit

Health check
------------
  curl http://localhost:8001/health
  curl http://localhost:8002/health
"""

import argparse
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app (declared before model loading so startup event can populate it)
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException  # type: ignore
    from fastapi.responses import JSONResponse   # type: ignore
    from pydantic import BaseModel              # type: ignore
    import uvicorn                              # type: ignore
except ImportError:
    print(
        "ERROR: fastapi and uvicorn are required.\n"
        "       pip install fastapi uvicorn\n",
        file=sys.stderr,
    )
    sys.exit(1)

app = FastAPI(title="SFT Experiment Inference Server")

# Will be populated at startup
_STATE: dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "model_id": "",
    "device": "cpu",
}


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: List[Message]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: str, load_in_4bit: bool, load_in_8bit: bool) -> None:
    """Load tokenizer and model into _STATE. Called once at startup."""
    logger.info("Loading tokenizer from: %s", model_path)

    try:
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        import torch  # type: ignore
    except ImportError as exc:
        logger.error("transformers / torch not installed: %s", exc)
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    if load_in_4bit:
        logger.info("4-bit NF4 quantization enabled.")
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif load_in_8bit:
        logger.info("8-bit quantization enabled.")
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

    device_map: Any = "auto" if device in ("auto", "") else device
    logger.info("Loading model: %s  (device_map=%s)", model_path, device_map)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quant_cfg,
        torch_dtype="auto" if quant_cfg is None else None,
        trust_remote_code=False,
    )
    model.eval()
    logger.info("Model ready.")

    _STATE["tokenizer"] = tokenizer
    _STATE["model"]     = model
    _STATE["device"]    = device


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def _generate(messages: List[dict], max_new_tokens: int, temperature: float) -> str:
    """Run a forward pass and return decoded text (new tokens only)."""
    import torch  # type: ignore

    tokenizer = _STATE["tokenizer"]
    model     = _STATE["model"]

    # Apply chat template when available, else concatenate as raw text
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = "\n".join(
            f"[{m['role'].upper()}] {m['content']}" for m in messages
        )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max(1, max_new_tokens),
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": temperature > 0.0,
    }
    if temperature > 0.0:
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    new_ids   = output_ids[0][input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": _STATE["model_id"],
        "ready": _STATE["model"] is not None,
    }


@app.get("/v1/models")
def list_models() -> dict:
    model_id = _STATE["model_id"]
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest) -> dict:
    if _STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    max_new_tokens = req.max_tokens or 256
    temperature    = req.temperature if req.temperature is not None else 0.0

    t0 = time.perf_counter()
    try:
        output_text = _generate(messages, max_new_tokens, temperature)
    except Exception as exc:
        logger.error("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    elapsed = time.perf_counter() - t0

    logger.info("Generated %d chars in %.2f s", len(output_text), elapsed)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _STATE["model_id"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": -1,
            "completion_tokens": -1,
            "total_tokens": -1,
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal OpenAI-compatible inference server for HuggingFace models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="HuggingFace hub model ID or local model directory.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to listen on (default: 8001).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Inference device (default: cpu).",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="4-bit NF4 quantization (requires bitsandbytes + CUDA).",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="8-bit LLM.int8 quantization (requires bitsandbytes + CUDA).",
    )
    args = parser.parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        parser.error("--load-in-4bit and --load-in-8bit are mutually exclusive.")

    _STATE["model_id"] = args.model_path

    logger.info(
        "Starting inference server: model=%s  port=%d  device=%s",
        args.model_path,
        args.port,
        args.device,
    )

    # Load model before accepting requests
    load_model(
        model_path=args.model_path,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    logger.info(
        "Server ready — listening on http://%s:%d/v1",
        args.host,
        args.port,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
