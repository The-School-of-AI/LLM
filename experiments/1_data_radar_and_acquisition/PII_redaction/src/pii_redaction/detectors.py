from __future__ import annotations

import functools
import re
import unicodedata
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Iterable
from urllib.parse import parse_qsl, quote, urlparse, urlunparse

from .config import PipelineConfig
from .models import EntityMatch

EMAIL_RE = re.compile(r"(?<![\w.+-])([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})(?![\w-])", re.IGNORECASE)
PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,5}\)?[\s.-]?)?\d[\d\s().-]{7,}\d(?!\w)"
)
IPV4_RE = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
IPV6_RE = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}(?![\w:])")
SSN_RE = re.compile(r"(?<!\d)(\d{3}-\d{2}-\d{4})(?!\d)")
AADHAAR_RE = re.compile(r"(?<!\d)(\d{4}[ -]?\d{4}[ -]?\d{4})(?!\d)")
PAN_RE = re.compile(r"(?<![A-Z0-9])([A-Z]{5}\d{4}[A-Z])(?![A-Z0-9])")
CARD_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
ACCOUNT_RE = re.compile(
    r"(?i)\b(?:account|acct|a/c|bank\s+account|खाता|खाते)\b(?:\s*(?:number|no\.?|#|:|-))?\s*([0-9][0-9 -]{5,33})\b"
)
IBAN_RE = re.compile(r"(?<![A-Z0-9])([A-Z]{2}\d{2}[A-Z0-9]{11,30})(?![A-Z0-9])")
URL_RE = re.compile(r"https?://[^\s<>'\"`]+", re.IGNORECASE)
STREET_NAME_RE = re.compile(
    r"(?<!\w)([A-Z?-??-??-?][\w?-??-??-??-?'\-]{1,}(?:\s+[A-Z?-??-??-?0-9][\w?-??-??-??-?'\-]{0,}){0,4}\s+(?:Street|St\.?|Road|Rd\.?|Lane|Ln\.?|Avenue|Ave\.?|Boulevard|Blvd\.?|Drive|Dr\.?|Way|Place|Plaza|Court|Terrace|Stra?e|Strasse|Rue|Via|Calle|Camino|Chemin|Allee|Weg|Gasse|Expressway|Setu|Steeg))(?=$|[\s,.;:)])",
    re.UNICODE,
)
STREET_PREFIX_RE = re.compile(
    r"(?<!\w)((?:Via|Rue|Calle|Chemin|Camino|Concession|Sector|Ring)\s+[A-Z?-??-??-?0-9][\w?-??-??-??-?'\-]{1,}(?:\s+[A-Z?-??-??-?0-9][\w?-??-??-??-?'\-]{1,}){0,4})(?=$|[\s,.;:)])",
    re.UNICODE,
)
STREET_SINGLE_TOKEN_RE = re.compile(
    r"(?<!\w)([A-Z?-??-??-?][\w?-??-??-??-?'\-]{2,}(?:strasse|stra?e|weg|steeg|setu|expressway))(?=$|[\s,.;:)])",
    re.IGNORECASE | re.UNICODE,
)
ADDRESS_HINTS = (
    "street",
    "st",
    "road",
    "rd",
    "lane",
    "ln",
    "avenue",
    "ave",
    "nagar",
    "colony",
    "sector",
    "block",
    "apt",
    "apartment",
    "floor",
    "city",
    "district",
    "state",
    "zip",
    "postal code",
    "pincode",
)


@dataclass(frozen=True)
class DetectorContext:
    config: PipelineConfig


def detect(text: str, config: PipelineConfig) -> list[EntityMatch]:
    context = DetectorContext(config=config)
    candidates: list[EntityMatch] = []
    if config.detection.enable_url_sanitization:
        candidates.extend(_detect_sensitive_urls(text, context))
    if config.detection.enable_email:
        candidates.extend(_detect_email(text))
    if config.detection.enable_phone:
        candidates.extend(_detect_phone(text, config))
    if config.detection.enable_ip_address:
        candidates.extend(_detect_ip(text))
    if config.detection.enable_government_id:
        candidates.extend(_detect_government_ids(text, config))
    if config.detection.enable_payment_card:
        candidates.extend(_detect_payment_cards(text))
    if config.detection.enable_account_number:
        candidates.extend(_detect_account_numbers(text))
    if config.detection.enable_physical_address:
        candidates.extend(_detect_addresses(text, config))
        candidates.extend(_detect_street_names(text))
    if config.detection.name_detection_mode == "anchored":
        candidates.extend(_detect_anchored_names(text, config))
    return _resolve_overlaps(candidates)


def placeholder_for(label: str, config: PipelineConfig) -> str:
    if config.redaction.placeholder_style == "generic":
        return config.redaction.generic_placeholder
    return f"<{label}>"


def _detect_email(text: str) -> Iterable[EntityMatch]:
    for match in EMAIL_RE.finditer(text):
        yield EntityMatch(match.start(1), match.end(1), "EMAIL_ADDRESS", "regex_email", priority=90)


def _detect_phone(text: str, config: PipelineConfig) -> Iterable[EntityMatch]:
    for match in PHONE_RE.finditer(text):
        candidate = match.group(0)
        digits = re.sub(r"\D", "", candidate)
        looks_like_ipv4 = candidate.count(".") == 3 and re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", candidate) is not None
        anchor_window = text[max(0, match.start() - 24) : match.start()].lower()
        has_structured_anchor = any(anchor.lower() in anchor_window for anchor in config.detection.phone_context_blocklist)
        if 10 <= len(digits) <= 15 and not has_structured_anchor and not looks_like_ipv4:
            yield EntityMatch(match.start(), match.end(), "PHONE_NUMBER", "regex_phone", priority=80)


def _detect_ip(text: str) -> Iterable[EntityMatch]:
    for match in IPV4_RE.finditer(text):
        candidate = match.group(0)
        try:
            ip_address(candidate)
        except ValueError:
            continue
        yield EntityMatch(match.start(), match.end(), "IP_ADDRESS", "regex_ipv4", priority=85)
    for match in IPV6_RE.finditer(text):
        candidate = match.group(0)
        try:
            ip_address(candidate)
        except ValueError:
            continue
        yield EntityMatch(match.start(), match.end(), "IP_ADDRESS", "regex_ipv6", priority=85)


def _detect_government_ids(text: str, config: PipelineConfig) -> Iterable[EntityMatch]:
    for match in SSN_RE.finditer(text):
        yield EntityMatch(match.start(1), match.end(1), "SSN", "regex_ssn", priority=95)
    for match in AADHAAR_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(1))
        window_start = max(0, match.start(1) - 24)
        anchor_window = text[window_start : match.start(1)].lower()
        if len(digits) == 12 and any(anchor.lower() in anchor_window for anchor in config.detection.aadhaar_anchors):
            yield EntityMatch(match.start(1), match.end(1), "AADHAAR_NUMBER", "regex_aadhaar", priority=95)
    for match in PAN_RE.finditer(text):
        yield EntityMatch(match.start(1), match.end(1), "PAN_NUMBER", "regex_pan", priority=92)


def _detect_payment_cards(text: str) -> Iterable[EntityMatch]:
    for match in CARD_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if not 13 <= len(digits) <= 19:
            continue
        has_card_shape = len(digits) in {15, 16} and digits[0] in {"3", "4", "5", "6"} and len(set(digits)) >= 4
        if _passes_luhn(digits) or has_card_shape:
            yield EntityMatch(
                match.start(),
                match.end(),
                "CREDIT_CARD_NUMBER",
                "regex_payment_card",
                priority=96,
            )


def _detect_account_numbers(text: str) -> Iterable[EntityMatch]:
    for match in ACCOUNT_RE.finditer(text):
        candidate = match.group(1).strip()
        stripped = re.sub(r"[\s-]", "", candidate)
        if 6 <= len(stripped) <= 34 and any(ch.isdigit() for ch in stripped):
            yield EntityMatch(
                match.start(1),
                match.end(1),
                "ACCOUNT_NUMBER",
                "regex_account_number",
                priority=93,
            )
    for match in IBAN_RE.finditer(text):
        yield EntityMatch(
            match.start(1),
            match.end(1),
            "ACCOUNT_NUMBER",
            "regex_iban",
            priority=94,
        )


def _detect_addresses(text: str, config: PipelineConfig) -> Iterable[EntityMatch]:
    anchor_pattern = _compiled_anchor_regex(tuple(config.detection.address_anchors))
    for match in anchor_pattern.finditer(text):
        span = _extract_address_span(text, match.end())
        if span is None:
            continue
        start, end = span
        yield EntityMatch(start, end, "PHYSICAL_ADDRESS", "anchored_address_parser", priority=70)


def _detect_street_names(text: str) -> Iterable[EntityMatch]:
    for pattern in (STREET_NAME_RE, STREET_PREFIX_RE, STREET_SINGLE_TOKEN_RE):
        for match in pattern.finditer(text):
            yield EntityMatch(
                match.start(1),
                match.end(1),
                "PHYSICAL_ADDRESS",
                "regex_street_name",
                priority=68,
            )


def _detect_anchored_names(text: str, config: PipelineConfig) -> Iterable[EntityMatch]:
    anchor_pattern = _compiled_anchor_regex(tuple(config.detection.person_name_anchors))
    for match in anchor_pattern.finditer(text):
        span = _extract_name_span(text, match.end())
        if span is None:
            continue
        start, end = span
        yield EntityMatch(start, end, "PERSON_NAME", "anchored_name_parser", priority=40)


def _detect_sensitive_urls(text: str, context: DetectorContext) -> Iterable[EntityMatch]:
    sensitive_keys = {key.lower() for key in context.config.detection.sensitive_query_params}
    for match in URL_RE.finditer(text):
        url = match.group(0)
        parsed = urlparse(url)
        if not parsed.query:
            continue
        query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
        changed = False
        redacted_pairs: list[tuple[str, str]] = []
        for key, value in query_pairs:
            if key.lower() in sensitive_keys:
                changed = True
                redacted_pairs.append((key, "<SENSITIVE_VALUE>"))
            else:
                redacted_pairs.append((key, value))
        if not changed:
            continue
        sanitized_query = "&".join(
            f"{quote(key, safe='')}={quote(value, safe='<>')}" for key, value in redacted_pairs
        )
        sanitized = urlunparse(parsed._replace(query=sanitized_query))
        yield EntityMatch(
            match.start(),
            match.end(),
            "SENSITIVE_URL",
            "url_query_sanitizer",
            priority=99,
            replacement=sanitized,
        )


def _passes_luhn(number: str) -> bool:
    total = 0
    parity = len(number) % 2
    for idx, char in enumerate(number):
        digit = ord(char) - ord("0")
        if idx % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _resolve_overlaps(candidates: list[EntityMatch]) -> list[EntityMatch]:
    accepted: list[EntityMatch] = []
    occupied: list[tuple[int, int]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (-item.priority, -item.length, item.start, item.end, item.label),
    ):
        if any(not (candidate.end <= start or candidate.start >= end) for start, end in occupied):
            continue
        accepted.append(candidate)
        occupied.append((candidate.start, candidate.end))
    return sorted(accepted, key=lambda item: (item.start, item.end))


@functools.lru_cache(maxsize=32)
def _compiled_anchor_regex(anchors: tuple[str, ...]) -> re.Pattern[str]:
    escaped = [re.escape(anchor) for anchor in sorted(anchors, key=len, reverse=True)]
    return re.compile(r"(?i)(?<!\w)(?:" + "|".join(escaped) + r")(?!\w)\s*[:\-]?\s*")


def _extract_name_span(text: str, start_idx: int) -> tuple[int, int] | None:
    idx = start_idx
    while idx < len(text) and text[idx].isspace():
        idx += 1
    token_spans: list[tuple[int, int]] = []
    current = idx
    while current < len(text) and len(token_spans) < 4:
        while current < len(text) and text[current].isspace():
            current += 1
        token_start = current
        while current < len(text) and _is_name_char(text[current]):
            current += 1
        token = text[token_start:current]
        if not token:
            break
        if not _looks_like_name_token(token):
            break
        token_spans.append((token_start, current))
        while current < len(text) and text[current].isspace():
            current += 1
        if current < len(text) and not _is_name_char(text[current]):
            break
    if not 2 <= len(token_spans) <= 4:
        return None
    return token_spans[0][0], token_spans[-1][1]


def _extract_address_span(text: str, start_idx: int) -> tuple[int, int] | None:
    idx = start_idx
    while idx < len(text) and text[idx].isspace():
        idx += 1
    end = idx
    while end < len(text) and text[end] not in "\n\r;":
        if text[end] == ".":
            break
        end += 1
    candidate = text[idx:end].strip(" ,")
    lowered = candidate.lower()
    if not (12 <= len(candidate) <= 160):
        return None
    if not any(char.isdigit() for char in candidate):
        return None
    if not any(hint in lowered for hint in ADDRESS_HINTS):
        return None
    leading_trim = len(text[idx:end]) - len(text[idx:end].lstrip(" ,"))
    start = idx + leading_trim
    return start, start + len(candidate)


def _is_name_char(char: str) -> bool:
    return char.isalpha() or unicodedata.category(char).startswith("M") or char in {"'", "-", "."}


def _looks_like_name_token(token: str) -> bool:
    letters = [char for char in token if char.isalpha()]
    if len(letters) < 2:
        return False
    if all(ord(char) < 128 for char in letters):
        return token[0].isupper()
    return True
