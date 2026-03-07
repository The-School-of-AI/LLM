from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasets import load_dataset

from pii_redaction.config import load_config
from pii_redaction.processor import process_record


LABEL_MAP: dict[str, str] = {
    "EMAIL": "EMAIL_ADDRESS",
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "EMAILADDRESS": "EMAIL_ADDRESS",
    "PHONE": "PHONE_NUMBER",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "PHONENUMBER": "PHONE_NUMBER",
    "TELEPHONENUM": "PHONE_NUMBER",
    "TELEPHONE_NUMBER": "PHONE_NUMBER",
    "MOBILEPHONE": "PHONE_NUMBER",
    "MOBILE_PHONE_NUMBER": "PHONE_NUMBER",
    "IP": "IP_ADDRESS",
    "IP_ADDRESS": "IP_ADDRESS",
    "IPADDRESS": "IP_ADDRESS",
    "ADDRESS": "PHYSICAL_ADDRESS",
    "PHYSICAL_ADDRESS": "PHYSICAL_ADDRESS",
    "STREET_ADDRESS": "PHYSICAL_ADDRESS",
    "STREET": "PHYSICAL_ADDRESS",
    "STREET_ADDRESS": "PHYSICAL_ADDRESS",
    "CREDIT_CARD": "CREDIT_CARD_NUMBER",
    "CREDIT_CARD_NUMBER": "CREDIT_CARD_NUMBER",
    "CREDITCARDNUMBER": "CREDIT_CARD_NUMBER",
    "CARD_NUMBER": "CREDIT_CARD_NUMBER",
    "IBAN": "ACCOUNT_NUMBER",
    "ACCOUNT": "ACCOUNT_NUMBER",
    "ACCOUNT_NUMBER": "ACCOUNT_NUMBER",
    "BANK_ACCOUNT": "ACCOUNT_NUMBER",
    "SSN": "SSN",
    "AADHAAR": "AADHAAR_NUMBER",
    "AADHAAR_NUMBER": "AADHAAR_NUMBER",
    "PAN": "PAN_NUMBER",
    "PAN_NUMBER": "PAN_NUMBER",
    "URL": "SENSITIVE_URL",
    "LINK": "SENSITIVE_URL",
}

PRESENCE_ONLY_LABELS = {"PHYSICAL_ADDRESS", "PERSON_NAME"}


DEFAULT_SUPPORTED = {
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "IP_ADDRESS",
    "PHYSICAL_ADDRESS",
    "CREDIT_CARD_NUMBER",
    "ACCOUNT_NUMBER",
    "SSN",
    "AADHAAR_NUMBER",
    "PAN_NUMBER",
    "SENSITIVE_URL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or evaluate the AI4Privacy Open PII dataset.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect language and label distributions.")
    _add_common_args(inspect_parser)

    eval_parser = subparsers.add_parser("eval", help="Evaluate the pipeline on supported labels.")
    _add_common_args(eval_parser)
    eval_parser.add_argument("--config", default="configs/default_config.json", help="Pipeline config path.")
    eval_parser.add_argument(
        "--text-field",
        default="source_text",
        help="Dataset field to process as text. Default: source_text",
    )
    eval_parser.add_argument(
        "--include-person-name",
        action="store_true",
        help="Include PERSON/NAME-like labels in mapping. Disabled by default because the pipeline is intentionally conservative on free-form names.",
    )
    eval_parser.add_argument(
        "--output",
        default="reports/ai4privacy_eval.json",
        help="Path to write evaluation JSON.",
    )

    return parser.parse_args()


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", default="validation", help="Dataset split to use. Default: validation")
    parser.add_argument("--limit", type=int, default=1000, help="Max rows to inspect/evaluate. Default: 1000")
    parser.add_argument(
        "--languages",
        nargs="*",
        help="Optional language filter, e.g. --languages en hi te",
    )


def load_rows(split: str, limit: int, languages: list[str] | None) -> list[dict[str, Any]]:
    ds = load_dataset("ai4privacy/open-pii-masking-500k-ai4privacy", split=split)
    rows: list[dict[str, Any]] = []
    allowed = set(languages or [])
    for row in ds:
        if allowed and row.get("language") not in allowed:
            continue
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def normalize_privacy_mask(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        if value and isinstance(value[0], str):
            flattened: list[dict[str, Any]] = []
            for item in value:
                try:
                    parsed = json.loads(item)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, list):
                    flattened.extend([entry for entry in parsed if isinstance(entry, dict)])
            return flattened
        return [entry for entry in value if isinstance(entry, dict)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [entry for entry in parsed if isinstance(entry, dict)]
    return []


def canonical_label(raw_label: str, include_person_name: bool) -> str | None:
    cleaned = raw_label.upper().replace("-", "_").replace(" ", "_")
    if include_person_name and cleaned in {"PERSON", "PERSON_NAME", "NAME", "FIRST_NAME", "LAST_NAME", "FULL_NAME", "GIVENNAME", "SURNAME"}:
        return "PERSON_NAME"
    return LABEL_MAP.get(cleaned)


def inspect_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    languages = Counter()
    labels = Counter()
    mapped = Counter()
    for row in rows:
        languages[row.get("language", "<missing>")] += 1
        for item in normalize_privacy_mask(row.get("privacy_mask")):
            raw_label = str(item.get("label", "<missing>"))
            labels[raw_label] += 1
            canonical = canonical_label(raw_label, include_person_name=True)
            if canonical:
                mapped[canonical] += 1
    return {
        "rows": len(rows),
        "languages": dict(languages.most_common()),
        "raw_labels": dict(labels.most_common()),
        "mapped_labels": dict(mapped.most_common()),
    }


def evaluate_rows(
    rows: list[dict[str, Any]],
    config_path: str,
    text_field: str,
    include_person_name: bool,
) -> dict[str, Any]:
    config = load_config(config_path)
    config.schema.text_fields = [text_field]
    supported = set(DEFAULT_SUPPORTED)
    if include_person_name:
        supported.add("PERSON_NAME")

    totals = Counter()
    per_label = defaultdict(Counter)
    mismatches: list[dict[str, Any]] = []

    for row in rows:
        text = row.get(text_field)
        if not isinstance(text, str):
            continue
        gold_items = normalize_privacy_mask(row.get("privacy_mask"))
        gold = Counter()
        for item in gold_items:
            canonical = canonical_label(str(item.get("label", "")), include_person_name)
            if canonical in supported:
                gold[canonical] += 1
        for label in list(gold):
            if label in PRESENCE_ONLY_LABELS:
                gold[label] = 1

        if not gold:
            continue

        record = {
            "id": row.get("uid"),
            "lang": row.get("language"),
            text_field: text,
        }
        decision = process_record(record, config)
        predicted = Counter({label: count for label, count in decision.entities_by_type.items() if label in supported})
        for label in list(predicted):
            if label in PRESENCE_ONLY_LABELS:
                predicted[label] = 1

        totals["records_evaluated"] += 1
        if decision.total_entities:
            totals["records_with_prediction"] += 1

        all_labels = set(gold) | set(predicted)
        for label in all_labels:
            gold_count = gold.get(label, 0)
            pred_count = predicted.get(label, 0)
            matched = min(gold_count, pred_count)
            per_label[label]["gold"] += gold_count
            per_label[label]["predicted"] += pred_count
            per_label[label]["matched"] += matched
            totals["gold_entities"] += gold_count
            totals["predicted_entities"] += pred_count
            totals["matched_entities"] += matched
            if gold_count != pred_count:
                mismatches.append(
                    {
                        "uid": row.get("uid"),
                        "language": row.get("language"),
                        "label": label,
                        "gold": gold_count,
                        "predicted": pred_count,
                        "text_preview": text[:240],
                    }
                )

    summary = {
        "records_evaluated": totals["records_evaluated"],
        "gold_entities": totals["gold_entities"],
        "predicted_entities": totals["predicted_entities"],
        "matched_entities": totals["matched_entities"],
        "precision": _safe_div(totals["matched_entities"], totals["predicted_entities"]),
        "recall": _safe_div(totals["matched_entities"], totals["gold_entities"]),
    }
    label_metrics = {}
    for label, counts in sorted(per_label.items()):
        label_metrics[label] = {
            "gold": counts["gold"],
            "predicted": counts["predicted"],
            "matched": counts["matched"],
            "precision": _safe_div(counts["matched"], counts["predicted"]),
            "recall": _safe_div(counts["matched"], counts["gold"]),
        }

    return {
        "summary": summary,
        "per_label": label_metrics,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:200],
    }


def _safe_div(num: int, den: int) -> float:
    if den == 0:
        return 0.0
    return round(num / den, 4)


def main() -> int:
    args = parse_args()
    rows = load_rows(args.split, args.limit, args.languages)
    if args.command == "inspect":
        print(json.dumps(inspect_rows(rows), indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    result = evaluate_rows(
        rows=rows,
        config_path=args.config,
        text_field=args.text_field,
        include_person_name=args.include_person_name,
    )
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False, sort_keys=True))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
