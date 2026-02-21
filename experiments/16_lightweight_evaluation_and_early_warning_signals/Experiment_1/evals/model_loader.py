"""
Model loader with quantization support.

Supports:
  - bitsandbytes (INT8 / INT4, HuggingFace Transformers)
  - llama-cpp-python (GGUF INT4/INT8, CPU/GPU)
  - plain HuggingFace (fp16 / fp32 for reference)
"""
from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

Backend = Literal["hf", "bitsandbytes", "llama_cpp"]


@dataclass
class ModelConfig:
    checkpoint_path: str                      # HF model ID or local path / GGUF file
    backend: Backend = "bitsandbytes"
    quant_mode: Literal["int8", "int4", "fp16", "fp32"] = "int4"
    device_map: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.0                  # 0 = greedy
    seed: int = 42
    # llama-cpp specific
    n_ctx: int = 2048
    n_threads: int = 8
    n_gpu_layers: int = -1                    # -1 = auto
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


class ModelWrapper:
    """Thin wrapper around loaded model for unified generation interface."""

    def __init__(self, model: Any, tokenizer: Any | None, config: ModelConfig,
                 backend: Backend) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.backend = backend

    def generate(self, prompt: str) -> str:
        if self.backend == "llama_cpp":
            return self._generate_llama_cpp(prompt)
        return self._generate_hf(prompt)

    def _generate_hf(self, prompt: str) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": self.config.max_new_tokens,
                "do_sample": False,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        # Decode only newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        new_ids = output_ids[0][input_len:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def _generate_llama_cpp(self, prompt: str) -> str:
        response = self.model(
            prompt,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature if self.config.temperature > 0 else 0,
            echo=False,
        )
        return response["choices"][0]["text"]

    def log_likelihood(self, text: str) -> float:
        """Return average token NLL (negative log-likelihood) for the text."""
        if self.backend == "llama_cpp":
            return self._nll_llama_cpp(text)
        return self._nll_hf(text)

    def _nll_hf(self, text: str) -> float:
        import torch
        inputs = self.tokenizer(text, return_tensors="pt")
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
        return float(outputs.loss.item())  # cross-entropy loss = avg NLL per token

    def _nll_llama_cpp(self, text: str) -> float:
        # llama.cpp exposes logprobs per token
        response = self.model(
            text,
            max_tokens=1,
            logprobs=True,
            echo=True,
            temperature=0,
        )
        token_logprobs = [
            t.get("logprob", 0.0)
            for t in response.get("choices", [{}])[0].get("logprobs", {}).get("token_logprobs", [])
            if t is not None
        ]
        if not token_logprobs:
            return float("nan")
        import math
        return -sum(token_logprobs) / len(token_logprobs)


def detect_backend(checkpoint_path: str, preferred: str = "auto") -> Backend:
    """Heuristically pick a backend."""
    path = Path(checkpoint_path)
    if path.is_file() and path.suffix in {".gguf", ".bin"}:
        return "llama_cpp"
    if preferred == "bitsandbytes":
        return "bitsandbytes"
    if preferred == "llama_cpp":
        return "llama_cpp"
    # Default: bitsandbytes for HF checkpoints
    return "bitsandbytes"


def load_model(config: ModelConfig) -> ModelWrapper:
    """Load a model with the specified quantization backend."""
    backend = config.backend
    if backend == "auto":
        backend = detect_backend(config.checkpoint_path)

    logger.info(f"Loading model: {config.checkpoint_path} | backend={backend} | quant={config.quant_mode}")

    if backend == "llama_cpp":
        return _load_llama_cpp(config)
    elif backend == "bitsandbytes":
        return _load_bitsandbytes(config)
    else:
        return _load_hf(config)


def _load_bitsandbytes(config: ModelConfig) -> ModelWrapper:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
    except ImportError as e:
        raise ImportError(f"transformers / bitsandbytes not installed: {e}") from e

    bnb_config = None
    if config.quant_mode == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif config.quant_mode == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        config.checkpoint_path,
        quantization_config=bnb_config,
        device_map=config.device_map,
        torch_dtype=torch.float16 if config.quant_mode == "fp16" else None,
        **config.extra_kwargs,
    )
    model.eval()
    return ModelWrapper(model, tokenizer, config, "bitsandbytes")


def _load_llama_cpp(config: ModelConfig) -> ModelWrapper:
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise ImportError(f"llama-cpp-python not installed: {e}") from e

    n_gpu = config.n_gpu_layers
    # On Apple Silicon, use Metal by default
    if platform.system() == "Darwin" and platform.machine() == "arm64" and n_gpu == -1:
        n_gpu = 999  # offload all layers to Metal GPU

    model = Llama(
        model_path=config.checkpoint_path,
        n_ctx=config.n_ctx,
        n_threads=config.n_threads,
        n_gpu_layers=n_gpu,
        seed=config.seed,
        verbose=False,
        **config.extra_kwargs,
    )
    return ModelWrapper(model, None, config, "llama_cpp")


def _load_hf(config: ModelConfig) -> ModelWrapper:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError as e:
        raise ImportError(f"transformers not installed: {e}") from e

    dtype = torch.float16 if config.quant_mode == "fp16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        config.checkpoint_path,
        torch_dtype=dtype,
        device_map=config.device_map,
        **config.extra_kwargs,
    )
    model.eval()
    return ModelWrapper(model, tokenizer, config, "hf")
