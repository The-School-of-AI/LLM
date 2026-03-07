from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "synthetic"


def build_records() -> list[dict[str, object]]:
    return [
        {
            "id": "en-chat-001",
            "lang": "en",
            "source": "synthetic_chat",
            "text": "Customer name: Jane Doe email jane.doe@example.com phone +1 415 555 0101",
            "expected_action": "keep",
            "expected_entities": {"PERSON_NAME": 1, "EMAIL_ADDRESS": 1, "PHONE_NUMBER": 1},
        },
        {
            "id": "en-web-001",
            "lang": "en",
            "source": "synthetic_web",
            "text": "Reset at https://example.com/reset?email=jane%40example.com&token=abc123&lang=en",
            "expected_action": "keep",
            "expected_entities": {"SENSITIVE_URL": 1},
        },
        {
            "id": "en-drop-001",
            "lang": "en",
            "source": "synthetic_web",
            "text": "Contact jane@example.com phone +1 415 555 0101 ssn 123-45-6789 card 4111 1111 1111 1111",
            "expected_action": "drop",
            "expected_entities": {"EMAIL_ADDRESS": 1, "PHONE_NUMBER": 1, "SSN": 1, "CREDIT_CARD_NUMBER": 1},
        },
        {
            "id": "hi-001",
            "lang": "hi",
            "source": "synthetic_profile",
            "text": "\u0928\u093e\u092e: \u0930\u0935\u093f \u0915\u0941\u092e\u093e\u0930 \u0906\u0927\u093e\u0930 1234 5678 9123",
            "expected_action": "keep",
            "expected_entities": {"PERSON_NAME": 1, "AADHAAR_NUMBER": 1},
        },
        {
            "id": "hi-002",
            "lang": "hi",
            "source": "synthetic_profile",
            "text": "\u092a\u0924\u093e: 221B Baker Street, Delhi",
            "expected_action": "keep",
            "expected_entities": {"PHYSICAL_ADDRESS": 1},
        },
        {
            "id": "bn-001",
            "lang": "bn",
            "source": "synthetic_profile",
            "text": "Customer name: \u0985\u09a8\u09a8\u09cd\u09af\u09be \u09a6\u09a4\u09cd\u09a4\u09be email anya@example.com",
            "expected_action": "keep",
            "expected_entities": {"PERSON_NAME": 1, "EMAIL_ADDRESS": 1},
        },
        {
            "id": "ta-001",
            "lang": "ta",
            "source": "synthetic_profile",
            "text": "Customer name: \u0bae\u0bc0\u0ba9\u0bbe \u0bb0\u0bbe\u0b9c\u0bcd account number 7788990011",
            "expected_action": "keep",
            "expected_entities": {"PERSON_NAME": 1, "ACCOUNT_NUMBER": 1},
        },
        {
            "id": "te-001",
            "lang": "te",
            "source": "synthetic_profile",
            "text": "Name: \u0c05\u0c30\u0c4d\u0c1c\u0c41\u0c28\u0c4d \u0c30\u0c46\u0c21\u0c4d\u0c21\u0c3f Aadhaar 1234 5678 9123",
            "expected_action": "keep",
            "expected_entities": {"PERSON_NAME": 1, "AADHAAR_NUMBER": 1},
        },
        {
            "id": "kn-001",
            "lang": "kn",
            "source": "synthetic_profile",
            "text": "Customer name: \u0c85\u0ca8\u0cbf\u0cb2\u0ccd \u0c95\u0cc1\u0cae\u0cbe\u0cb0 phone +91 98765 43210",
            "expected_action": "keep",
            "expected_entities": {"PERSON_NAME": 1, "PHONE_NUMBER": 1},
        },
        {
            "id": "ml-001",
            "lang": "ml",
            "source": "synthetic_profile",
            "text": "Customer name: \u0d05\u0d28\u0d3f\u0d32\u0d4d \u0d15\u0d41\u0d2e\u0d3e\u0d30\u0d4d email anil@example.com",
            "expected_action": "keep",
            "expected_entities": {"PERSON_NAME": 1, "EMAIL_ADDRESS": 1},
        },
        {
            "id": "gu-001",
            "lang": "gu",
            "source": "synthetic_profile",
            "text": "Customer name: \u0a85\u0aa8\u0a3f\u0ab2 \u0a15\u0ac1\u0aae\u0abe\u0ab0 account number 001122334455",
            "expected_action": "keep",
            "expected_entities": {"PERSON_NAME": 1, "ACCOUNT_NUMBER": 1},
        },
        {
            "id": "mr-001",
            "lang": "mr",
            "source": "synthetic_profile",
            "text": "\u0928\u093e\u0935: \u0938\u093e\u0915\u0947\u0924 \u092a\u093e\u091f\u0940\u0932 PAN ABCDE1234F",
            "expected_action": "keep",
            "expected_entities": {"PAN_NUMBER": 1},
        },
        {
            "id": "pa-001",
            "lang": "pa",
            "source": "synthetic_profile",
            "text": "Customer name: \u0a17\u0a41\u0a30\u0a2a\u0a4d\u0a30\u0a40\u0a24 \u0a38\u0a3f\u0a70\u0a18 phone +91 98123 45678",
            "expected_action": "keep",
            "expected_entities": {"PERSON_NAME": 1, "PHONE_NUMBER": 1},
        },
        {
            "id": "ur-001",
            "lang": "ur",
            "source": "synthetic_profile",
            "text": "Customer name: \u0639\u0644\u06cc \u062d\u0633\u0646 email ali@example.com",
            "expected_action": "keep",
            "expected_entities": {"PERSON_NAME": 1, "EMAIL_ADDRESS": 1},
        },
        {
            "id": "log-001",
            "lang": "en",
            "source": "synthetic_log",
            "text": "2026-03-06T12:01:00Z INFO user_email=ops@example.com client_ip=203.0.113.10",
            "expected_action": "keep",
            "expected_entities": {"EMAIL_ADDRESS": 1, "IP_ADDRESS": 1},
        },
        {
            "id": "markdown-001",
            "lang": "en",
            "source": "synthetic_markdown",
            "text": "Please ship to address: 742 Evergreen Road, Springfield",
            "expected_action": "keep",
            "expected_entities": {"PHYSICAL_ADDRESS": 1},
        },
        {
            "id": "json-001",
            "lang": "en",
            "source": "synthetic_code",
            "text": '{"email": "json.user@example.com", "callback": "https://api.example.com/cb?token=secret"}',
            "expected_action": "keep",
            "expected_entities": {"EMAIL_ADDRESS": 1, "SENSITIVE_URL": 1},
        },
        {
            "id": "clean-001",
            "lang": "en",
            "source": "synthetic_clean",
            "text": "Barack Obama discussed open-source LLM training in Delhi.",
            "expected_action": "keep",
            "expected_entities": {},
        },
    ]


def build_nested_records() -> list[dict[str, object]]:
    return [
        {
            "id": "nested-001",
            "lang": "en",
            "source": "synthetic_nested",
            "payload": {"body": "Reach support at nested@example.com or +1 212 555 0110"},
            "expected_action": "keep",
            "expected_entities": {"EMAIL_ADDRESS": 1, "PHONE_NUMBER": 1},
        },
        {
            "id": "nested-002",
            "lang": "hi",
            "source": "synthetic_nested",
            "payload": {"body": "\u0928\u093e\u092e: \u0930\u093e\u091c \u0936\u0930\u094d\u092e\u093e email raj@example.com"},
            "expected_action": "keep",
            "expected_entities": {"PERSON_NAME": 1, "EMAIL_ADDRESS": 1},
        },
    ]


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    main_records = build_records()
    nested_records = build_nested_records()
    write_jsonl(DATASET_DIR / "llm_multilingual_structured.jsonl", main_records)
    write_jsonl(DATASET_DIR / "llm_nested_records.jsonl", nested_records)
    manifest = {
        "files": [
            {
                "path": str(DATASET_DIR / "llm_multilingual_structured.jsonl"),
                "record_count": len(main_records),
                "notes": "Mixed LLM-style web, chat, log, markdown, code, and multilingual structured PII samples.",
            },
            {
                "path": str(DATASET_DIR / "llm_nested_records.jsonl"),
                "record_count": len(nested_records),
                "notes": "Nested-schema samples for payload.body style inputs.",
            },
        ]
    }
    with (DATASET_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
