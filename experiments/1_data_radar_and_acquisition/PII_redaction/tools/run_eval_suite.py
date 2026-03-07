from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pii_redaction.config import load_config
from pii_redaction.processor import process_record


def evaluate_file(path: Path, config_path: str | None, text_field: str) -> dict[str, object]:
    config = load_config(config_path)
    config.schema.text_fields = [text_field]
    summary = {"records": 0, "action_mismatches": 0, "entity_mismatches": 0}
    mismatches: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            expected_action = record.pop("expected_action")
            expected_entities = record.pop("expected_entities")
            decision = process_record(record, config)
            summary["records"] += 1
            if decision.action != expected_action:
                summary["action_mismatches"] += 1
                mismatches.append({"id": decision.doc_id, "type": "action", "expected": expected_action, "actual": decision.action})
            for label, expected_count in expected_entities.items():
                actual_count = decision.entities_by_type.get(label, 0)
                if actual_count != expected_count:
                    summary["entity_mismatches"] += 1
                    mismatches.append({
                        "id": decision.doc_id,
                        "type": "entity",
                        "label": label,
                        "expected": expected_count,
                        "actual": actual_count,
                    })
            unexpected_labels = set(decision.entities_by_type) - set(expected_entities)
            if unexpected_labels:
                summary["entity_mismatches"] += len(unexpected_labels)
                for label in sorted(unexpected_labels):
                    mismatches.append({
                        "id": decision.doc_id,
                        "type": "unexpected_entity",
                        "label": label,
                        "expected": 0,
                        "actual": decision.entities_by_type[label],
                    })
    return {"summary": summary, "mismatches": mismatches}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the redaction pipeline on generated synthetic corpora.")
    parser.add_argument("--config", help="Optional pipeline config path.")
    parser.add_argument(
        "--dataset-dir",
        default=str(ROOT / "datasets" / "synthetic"),
        help="Directory containing generated evaluation corpora.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    files = [
        (dataset_dir / "llm_multilingual_structured.jsonl", "text"),
        (dataset_dir / "llm_nested_records.jsonl", "payload.body"),
    ]
    results = []
    exit_code = 0
    for path, text_field in files:
        result = evaluate_file(path, args.config, text_field)
        result["path"] = str(path)
        results.append(result)
        if result["summary"]["action_mismatches"] or result["summary"]["entity_mismatches"]:
            exit_code = 1
    print(json.dumps({"results": results}, indent=2, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
