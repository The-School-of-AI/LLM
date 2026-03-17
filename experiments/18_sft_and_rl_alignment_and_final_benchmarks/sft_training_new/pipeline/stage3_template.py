"""
Stage 3 — Chat Template Application.

Applies the configured chat template to each conversation and outputs:
  - formatted_text: the full rendered string
  - role_spans: list of {"role", "start", "end"} character-level offsets

These two fields form the CONTRACT that Stage 4 (tokenisation) and
Stage 5 (loss masking) depend on. Do not modify without updating both
downstream stages.

System prompt selection supports three strategies (config):
  - fixed:      always use prompts[0]
  - random:     sample from prompts list per example (seeded)
  - per_source: pick by _source field (cycles through prompts)
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from pipeline.config import PipelineConfig, Stage3Config, SystemPromptsConfig
from pipeline.funnel_tracker import FunnelTracker
from pipeline.io.readers import iter_records
from pipeline.templates.base import TemplateBase

logger = logging.getLogger(__name__)


def _build_template(cfg: Stage3Config) -> TemplateBase:
    if cfg.template == "chatml":
        from pipeline.templates.chatml import ChatMLTemplate
        return ChatMLTemplate()
    if cfg.template == "llama3":
        from pipeline.templates.llama3 import Llama3Template
        return Llama3Template()
    if cfg.template == "tokenizer_native":
        if not cfg.tokenizer_name_or_path:
            raise ValueError("stage3.tokenizer_name_or_path must be set when template=tokenizer_native")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name_or_path, trust_remote_code=True)
        from pipeline.templates.tokenizer_native import TokenizerNativeTemplate
        return TokenizerNativeTemplate(tok)
    if cfg.template == "custom":
        if not cfg.custom_template_path:
            raise ValueError("stage3.custom_template_path must be set when template=custom")
        raise NotImplementedError(
            "Custom Jinja2 template support: implement CustomJinja2Template in templates/custom.py"
        )
    raise ValueError(f"Unknown template type: '{cfg.template}'")


class SystemPromptSelector:
    def __init__(self, cfg: SystemPromptsConfig, seed: int = 42) -> None:
        self._cfg = cfg
        self._prompts = cfg.prompts or ["You are a helpful assistant."]
        self._rng = random.Random(seed)
        self._source_map: dict[str, str] = {}

    def select(self, record: dict) -> str | None:
        if not self._prompts:
            return None
        strategy = self._cfg.variation_strategy
        if strategy == "fixed":
            return self._prompts[0]
        if strategy == "random":
            return self._rng.choice(self._prompts)
        if strategy == "per_source":
            source = record.get("_source", "default")
            if source not in self._source_map:
                idx = len(self._source_map) % len(self._prompts)
                self._source_map[source] = self._prompts[idx]
            return self._source_map[source]
        return self._prompts[0]


def run(cfg: PipelineConfig, tracker: FunnelTracker | None = None) -> None:
    stage_cfg = cfg.stage3
    if not stage_cfg.enabled:
        logger.info("Stage 3 (template) disabled — skipping")
        return

    input_path  = cfg.work_path(stage_cfg.input_file)
    output_path = cfg.work_path(stage_cfg.output_file)

    if cfg.globals.save_intermediates and output_path.exists() and cfg.globals.resume_from_stage > 3:
        logger.info("Stage 3: output %s exists, resuming from next stage", output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    template = _build_template(stage_cfg)
    selector = SystemPromptSelector(stage_cfg.system_prompts, seed=cfg.globals.seed)

    total = kept = skipped = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for record in iter_records(input_path):
            total += 1
            conversations = record.get("conversations", [])
            if not conversations:
                skipped += 1
                if tracker:
                    tracker.record_drop("stage3", "empty_conversations", "no_turns")
                continue

            system_prompt = selector.select(record)

            try:
                formatted_text, role_spans = template.apply(conversations, system_prompt=system_prompt)
            except Exception as exc:
                logger.warning("Template application failed for record: %s", exc)
                skipped += 1
                if tracker:
                    tracker.record_drop("stage3", "template_error", str(exc))
                continue

            record["formatted_text"] = formatted_text
            record["role_spans"] = role_spans
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    if tracker:
        tracker.record_stage_output("stage3", kept)

    logger.info(
        "Stage 3 complete: total=%d, kept=%d, skipped=%d → %s",
        total, kept, skipped, output_path,
    )
