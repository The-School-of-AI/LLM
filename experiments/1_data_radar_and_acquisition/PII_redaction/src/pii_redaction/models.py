from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EntityMatch:
    start: int
    end: int
    label: str
    detector: str
    priority: int = 100
    replacement: str | None = None
    score: float = 1.0

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class FieldRedactionResult:
    field_path: str
    original_length: int
    redacted_length: int
    text_changed: bool
    entities: list[EntityMatch] = field(default_factory=list)

    @property
    def entity_count(self) -> int:
        return len(self.entities)


@dataclass
class DocumentDecision:
    record: dict[str, Any]
    doc_id: str
    action: str
    drop_reason: str | None
    field_results: list[FieldRedactionResult]

    @property
    def total_entities(self) -> int:
        return sum(field.entity_count for field in self.field_results)

    @property
    def entities_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for field in self.field_results:
            for entity in field.entities:
                counts[entity.label] = counts.get(entity.label, 0) + 1
        return counts
