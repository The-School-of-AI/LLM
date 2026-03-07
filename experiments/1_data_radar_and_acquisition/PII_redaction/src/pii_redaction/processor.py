from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .config import PipelineConfig
from .detectors import detect, placeholder_for
from .models import DocumentDecision, EntityMatch, FieldRedactionResult


def run_pipeline(
    input_paths: list[str | Path],
    output_dir: str | Path,
    config: PipelineConfig,
    resume: bool | None = None,
) -> dict[str, Any]:
    started_at = time.time()
    resolved_inputs = _expand_inputs(input_paths)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    should_resume = config.runtime.resume if resume is None else resume

    aggregate_entities: Counter[str] = Counter()
    aggregate_drop_reasons: Counter[str] = Counter()
    aggregate_detector_hits: Counter[str] = Counter()
    aggregate_summary: Counter[str] = Counter()
    audit_samples: list[dict[str, Any]] = []
    file_summaries: list[dict[str, Any]] = []

    for input_path in resolved_inputs:
        stem = _stable_stem(input_path)
        file_dir = out_dir / stem
        completion_marker = file_dir / "_SUCCESS.json"
        if should_resume and completion_marker.exists():
            with completion_marker.open("r", encoding="utf-8") as handle:
                prior_summary = json.load(handle)
            file_summaries.append(prior_summary)
            aggregate_entities.update(prior_summary["entities_by_type"])
            aggregate_drop_reasons.update(prior_summary["drop_reasons"])
            aggregate_detector_hits.update(prior_summary["detector_hits"])
            aggregate_summary.update(prior_summary["summary"])
            audit_samples.extend(prior_summary.get("audit_samples", []))
            continue

        file_summary = process_file(input_path, file_dir, config)
        file_summaries.append(file_summary)
        aggregate_entities.update(file_summary["entities_by_type"])
        aggregate_drop_reasons.update(file_summary["drop_reasons"])
        aggregate_detector_hits.update(file_summary["detector_hits"])
        aggregate_summary.update(file_summary["summary"])
        audit_samples.extend(file_summary.get("audit_samples", []))

    manifest = {
        "run_completed_at_epoch_s": int(time.time()),
        "processing_seconds": round(time.time() - started_at, 3),
        "input_files": [str(path) for path in resolved_inputs],
        "output_dir": str(out_dir),
        "summary": dict(aggregate_summary),
        "entities_by_type": dict(aggregate_entities),
        "drop_reasons": dict(aggregate_drop_reasons),
        "detector_hits": dict(aggregate_detector_hits),
        "audit_samples": audit_samples[: config.runtime.max_audit_samples],
        "files": file_summaries,
    }
    with (out_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=False)
    return manifest


def process_file(input_path: str | Path, output_dir: str | Path, config: PipelineConfig) -> dict[str, Any]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    redacted_path = output_dir / "redacted.jsonl"
    dropped_path = output_dir / "dropped.jsonl"
    metrics_path = output_dir / "metrics.json"
    completion_marker = output_dir / "_SUCCESS.json"

    summary: Counter[str] = Counter()
    entities_by_type: Counter[str] = Counter()
    detector_hits: Counter[str] = Counter()
    drop_reasons: Counter[str] = Counter()
    audit_samples: list[dict[str, Any]] = []

    with (
        input_path.open("r", encoding="utf-8-sig") as source,
        redacted_path.open("w", encoding="utf-8") as redacted_out,
        dropped_path.open("w", encoding="utf-8") as dropped_out,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            summary["records_seen"] += 1
            record = json.loads(line)
            decision = process_record(record, config)
            for field in decision.field_results:
                for entity in field.entities:
                    entities_by_type[entity.label] += 1
                    detector_hits[entity.detector] += 1
            if decision.action == "drop":
                summary["records_dropped"] += 1
                drop_reasons[decision.drop_reason or "policy_drop"] += 1
                dropped_out.write(
                    json.dumps(
                        {
                            "id": decision.doc_id,
                            "drop_reason": decision.drop_reason,
                            "entities_by_type": decision.entities_by_type,
                            "line_number": line_number,
                        },
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            if decision.total_entities:
                summary["records_redacted"] += 1
            else:
                summary["records_unchanged"] += 1
            summary["records_kept"] += 1
            redacted_out.write(json.dumps(decision.record, sort_keys=True, ensure_ascii=False) + "\n")

            if decision.total_entities and len(audit_samples) < config.runtime.max_audit_samples:
                audit_samples.append(
                    {
                        "id": decision.doc_id,
                        "entities_by_type": decision.entities_by_type,
                        "field_paths": [field.field_path for field in decision.field_results if field.entities],
                    }
                )

    summary["entities_detected"] = sum(entities_by_type.values())
    metrics = {
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "summary": dict(summary),
        "entities_by_type": dict(entities_by_type),
        "drop_reasons": dict(drop_reasons),
        "detector_hits": dict(detector_hits),
        "audit_samples": audit_samples,
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True, ensure_ascii=False)
    with completion_marker.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True, ensure_ascii=False)
    return metrics


def process_record(record: dict[str, Any], config: PipelineConfig) -> DocumentDecision:
    cloned = json.loads(json.dumps(record, ensure_ascii=False))
    doc_id = str(_get_path(cloned, config.schema.id_field) or _hash_record(cloned))
    field_results: list[FieldRedactionResult] = []

    total_redacted_chars = 0
    total_original_chars = 0
    high_risk_hits = 0
    high_risk_labels = set(config.drop_policy.high_risk_labels)

    for field_path in config.schema.text_fields:
        for resolved_path, container, key, value in _iter_text_targets(cloned, field_path):
            entities = detect(value, config)
            redacted_text = _apply_redactions(value, entities, config)
            field_result = FieldRedactionResult(
                field_path=resolved_path,
                original_length=len(value),
                redacted_length=len(redacted_text),
                text_changed=redacted_text != value,
                entities=entities,
            )
            field_results.append(field_result)
            total_original_chars += len(value)
            if entities:
                total_redacted_chars += sum(entity.length for entity in entities)
                high_risk_hits += sum(1 for entity in entities if entity.label in high_risk_labels)
                container[key] = redacted_text

    total_entities = sum(field.entity_count for field in field_results)
    entity_density = (total_redacted_chars / total_original_chars) if total_original_chars else 0.0

    drop_reason = None
    if total_entities >= config.drop_policy.drop_if_total_entities_gte:
        drop_reason = "total_entity_threshold"
    elif high_risk_hits >= config.drop_policy.drop_if_high_risk_entities_gte:
        drop_reason = "high_risk_entity_threshold"
    elif (
        total_entities >= config.drop_policy.drop_if_entity_density_min_entities
        and entity_density >= config.drop_policy.drop_if_entity_density_gte
    ):
        drop_reason = "entity_density_threshold"

    action = "drop" if drop_reason else "keep"
    if action == "keep":
        cloned.setdefault("_redaction", {})
        cloned["_redaction"].update(
            {
                "status": "redacted" if total_entities else "unchanged",
                "entities_by_type": _count_entities(field_results),
            }
        )

    return DocumentDecision(
        record=cloned,
        doc_id=doc_id,
        action=action,
        drop_reason=drop_reason,
        field_results=field_results,
    )


def _apply_redactions(text: str, entities: Iterable[EntityMatch], config: PipelineConfig) -> str:
    cursor = 0
    parts: list[str] = []
    for entity in entities:
        parts.append(text[cursor : entity.start])
        parts.append(entity.replacement or placeholder_for(entity.label, config))
        cursor = entity.end
    parts.append(text[cursor:])
    return "".join(parts)


def _count_entities(field_results: list[FieldRedactionResult]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for field in field_results:
        for entity in field.entities:
            counts[entity.label] += 1
    return dict(counts)


def _expand_inputs(input_paths: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw_path in input_paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.jsonl")))
        elif any(symbol in str(path) for symbol in ["*", "?"]):
            files.extend(sorted(Path().glob(str(path))))
        else:
            files.append(path)
    deduped = sorted({file.resolve() for file in files})
    return deduped


def _stable_stem(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}-{digest}"


def _hash_record(record: dict[str, Any]) -> str:
    payload = json.dumps(record, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _get_path(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _iter_text_targets(payload: dict[str, Any], dotted_path: str) -> Iterable[tuple[str, Any, Any, str]]:
    parts = dotted_path.split(".")
    yield from _iter_text_targets_inner(payload, parts, prefix="")


def _iter_text_targets_inner(
    current: Any,
    parts: list[str],
    prefix: str,
) -> Iterable[tuple[str, Any, Any, str]]:
    if not parts:
        return

    part = parts[0]
    rest = parts[1:]

    if isinstance(current, list):
        for index, item in enumerate(current):
            next_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from _iter_text_targets_inner(item, parts, next_prefix)
        return

    if not isinstance(current, dict) or part not in current:
        return

    value = current[part]
    next_prefix = f"{prefix}.{part}" if prefix else part
    if rest:
        yield from _iter_text_targets_inner(value, rest, next_prefix)
        return

    if isinstance(value, str):
        yield next_prefix, current, part, value
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, str):
                yield f"{next_prefix}[{index}]", value, index, item


def _set_path(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    current = payload
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value
