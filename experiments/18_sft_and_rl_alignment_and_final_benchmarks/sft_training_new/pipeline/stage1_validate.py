"""
Stage 1 — Schema Validation.

Loads raw data (any supported format), validates every example against
the schema defined in config, and routes passing examples to the output
file and failing examples (with reasons) to the rejected file.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pipeline.config import PipelineConfig, Stage1Config
from pipeline.funnel_tracker import FunnelTracker
from pipeline.io.readers import iter_records

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class ConversationValidator:
    def __init__(self, cfg: Stage1Config) -> None:
        self._cfg = cfg

    def validate(self, record: dict) -> tuple[bool, str]:
        """Return (ok, reject_reason). reject_reason is empty when ok=True."""
        conversations = record.get("conversations")
        if not isinstance(conversations, list):
            return False, "missing_or_invalid_conversations_field"

        if len(conversations) < self._cfg.min_turns:
            return False, f"too_few_turns:{len(conversations)}<{self._cfg.min_turns}"

        if len(conversations) > self._cfg.max_turns:
            return False, f"too_many_turns:{len(conversations)}>{self._cfg.max_turns}"

        seen_roles: set[str] = set()
        for i, turn in enumerate(conversations):
            role = turn.get("role")
            content = turn.get("content")

            if not isinstance(role, str) or not role:
                return False, f"turn[{i}]:missing_role"

            if role == "system" and not self._cfg.allow_system_turn:
                return False, f"turn[{i}]:system_turn_not_allowed"

            if role not in ("system", "user", "assistant"):
                return False, f"turn[{i}]:unknown_role:{role}"

            if not isinstance(content, str):
                return False, f"turn[{i}]:content_must_be_string"

            if len(content) < self._cfg.min_content_chars:
                return False, f"turn[{i}]:content_too_short:{len(content)}"

            seen_roles.add(role)

        # Check required roles are present
        for required in self._cfg.required_roles:
            if required not in seen_roles:
                return False, f"missing_required_role:{required}"

        # Last turn check
        if self._cfg.require_last_turn_is_assistant:
            if conversations[-1].get("role") != "assistant":
                return False, f"last_turn_not_assistant:{conversations[-1].get('role')}"

        # Multi-turn check: only check if we don't allow multi-turn
        if not self._cfg.allow_multi_turn:
            user_count = sum(1 for t in conversations if t.get("role") == "user")
            if user_count > 1:
                return False, f"multi_turn_not_allowed:user_count={user_count}"

        return True, ""


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def run(cfg: PipelineConfig, tracker: FunnelTracker | None = None) -> None:
    stage_cfg = cfg.stage1
    if not stage_cfg.enabled:
        logger.info("Stage 1 (validate) disabled — skipping")
        return

    input_path  = cfg.work_path(stage_cfg.input_file)
    output_path = cfg.work_path(stage_cfg.output_file)
    reject_path = cfg.work_path(stage_cfg.rejected_file)

    if cfg.globals.save_intermediates and output_path.exists() and cfg.globals.resume_from_stage > 1:
        logger.info("Stage 1: output %s exists, resuming from next stage", output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    validator = ConversationValidator(stage_cfg)

    total = kept = rejected = 0
    with (
        open(output_path, "w", encoding="utf-8") as fout,
        open(reject_path, "w", encoding="utf-8") as frej,
    ):
        for record in iter_records(input_path):
            total += 1
            ok, reason = validator.validate(record)
            if ok:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            else:
                record["_reject_reason"] = reason
                frej.write(json.dumps(record, ensure_ascii=False) + "\n")
                rejected += 1
                if tracker:
                    tracker.record_drop("stage1", "schema_validation", reason)

    if tracker:
        tracker.record_stage_output("stage1", kept)

    logger.info(
        "Stage 1 complete: total=%d, kept=%d, rejected=%d → %s",
        total, kept, rejected, output_path,
    )
