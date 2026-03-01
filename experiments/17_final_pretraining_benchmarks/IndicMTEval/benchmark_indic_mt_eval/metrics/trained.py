"""Trained MT metrics: COMET."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from comet import download_model, load_from_checkpoint

    HAS_COMET = True
except ImportError:
    HAS_COMET = False

from benchmark_indic_mt_eval.metrics.registry import register_metric

_comet_model = None


def _get_comet_model(model_name: str = "Unbabel/wmt22-comet-da"):
    global _comet_model
    if _comet_model is None:
        model_path = download_model(model_name)
        _comet_model = load_from_checkpoint(model_path)
    return _comet_model


@register_metric("comet")
def compute_comet(
    hypothesis: str, reference: str, source: str = "", **kwargs
) -> float:
    if not HAS_COMET:
        raise ImportError(
            "unbabel-comet not installed. Install with: pip install unbabel-comet"
        )
    model = _get_comet_model(kwargs.get("comet_model", "Unbabel/wmt22-comet-da"))
    data = [{"src": source, "mt": hypothesis, "ref": reference}]
    output = model.predict(data, batch_size=1, gpus=0, num_workers=1)
    return float(output.scores[0])
