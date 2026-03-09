"""HuggingFace causal LM backend for generation."""

from __future__ import annotations

from typing import Any

from benchmark_indicgenbench.models.base import GenerationModelBase
from benchmark_indicgenbench.models.registry import register_model


@register_model("hf")
def _create_hf_model(
    model_name_or_path: str | None = None,
    device: str = "cpu",
    torch_dtype: str = "auto",
    **kwargs: Any,
) -> GenerationModelBase:
    if not model_name_or_path:
        raise ValueError("backend=hf requires model_name_or_path (e.g. --model-name google/gemma-3-1b-it)")
    return HfGenerationModel(model_name_or_path=model_name_or_path, device=device, torch_dtype=torch_dtype)


class HfGenerationModel(GenerationModelBase):
    """HuggingFace AutoModelForCausalLM wrapper."""

    def __init__(self, model_name_or_path: str, device: str = "cpu", torch_dtype: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device if device != "cpu" else None,
        )
        self._device = device
        if device == "cpu" or "cuda" not in device:
            self._model = self._model.to(device)
        self._model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        # Decode only the generated tokens (strip the prompt)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
