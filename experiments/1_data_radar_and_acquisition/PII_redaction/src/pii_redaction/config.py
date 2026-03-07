from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SchemaConfig:
    id_field: str = "id"
    text_fields: list[str] = field(default_factory=lambda: ["text"])
    lang_field: str = "lang"
    source_field: str = "source"


@dataclass
class DetectionConfig:
    enable_email: bool = True
    enable_phone: bool = True
    enable_ip_address: bool = True
    enable_physical_address: bool = True
    enable_government_id: bool = True
    enable_payment_card: bool = True
    enable_account_number: bool = True
    enable_url_sanitization: bool = True
    name_detection_mode: str = "anchored"
    person_name_anchors: list[str] = field(
        default_factory=lambda: [
            "name",
            "contact",
            "employee",
            "patient",
            "customer",
            "user",
            "नाम",
        ]
    )
    account_anchors: list[str] = field(
        default_factory=lambda: [
            "account",
            "acct",
            "a/c",
            "bank account",
            "खाता",
            "खाते",
        ]
    )
    address_anchors: list[str] = field(
        default_factory=lambda: [
            "address",
            "addr",
            "location",
            "located at",
            "residence",
            "resident at",
            "पता",
        ]
    )
    aadhaar_anchors: list[str] = field(
        default_factory=lambda: [
            "aadhaar",
            "uid",
            "uidai",
            "आधार",
        ]
    )
    phone_context_blocklist: list[str] = field(
        default_factory=lambda: [
            "account",
            "acct",
            "aadhaar",
            "uid",
            "card",
            "ssn",
            "pan",
            "खाता",
            "आधार",
        ]
    )
    sensitive_query_params: list[str] = field(
        default_factory=lambda: [
            "access_token",
            "account",
            "acct",
            "aadhaar",
            "api_key",
            "apikey",
            "auth",
            "authorization",
            "card",
            "card_number",
            "code",
            "cvv",
            "dob",
            "email",
            "key",
            "name",
            "otp",
            "pan",
            "password",
            "phone",
            "session",
            "ssn",
            "token",
            "userid"
        ]
    )


@dataclass
class RedactionConfig:
    placeholder_style: str = "typed"
    generic_placeholder: str = "[REDACTED]"
    preserve_separator_spacing: bool = True


@dataclass
class DropPolicyConfig:
    drop_if_total_entities_gte: int = 4
    drop_if_high_risk_entities_gte: int = 2
    drop_if_entity_density_gte: float = 0.18
    drop_if_entity_density_min_entities: int = 5
    high_risk_labels: list[str] = field(
        default_factory=lambda: [
            "ACCOUNT_NUMBER",
            "AADHAAR_NUMBER",
            "CREDIT_CARD_NUMBER",
            "PAN_NUMBER",
            "SSN"
        ]
    )


@dataclass
class RuntimeConfig:
    max_audit_samples: int = 50
    write_per_file_metrics: bool = True
    resume: bool = True


@dataclass
class PipelineConfig:
    schema: SchemaConfig = field(default_factory=SchemaConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    redaction: RedactionConfig = field(default_factory=RedactionConfig)
    drop_policy: DropPolicyConfig = field(default_factory=DropPolicyConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PipelineConfig":
        return cls(
            schema=SchemaConfig(**payload.get("schema", {})),
            detection=DetectionConfig(**payload.get("detection", {})),
            redaction=RedactionConfig(**payload.get("redaction", {})),
            drop_policy=DropPolicyConfig(**payload.get("drop_policy", {})),
            runtime=RuntimeConfig(**payload.get("runtime", {}))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema.__dict__,
            "detection": self.detection.__dict__,
            "redaction": self.redaction.__dict__,
            "drop_policy": self.drop_policy.__dict__,
            "runtime": self.runtime.__dict__
        }


def load_config(path: str | Path | None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    return PipelineConfig.from_dict(payload)
