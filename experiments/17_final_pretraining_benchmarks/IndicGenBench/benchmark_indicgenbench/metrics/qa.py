"""QA metrics: Exact Match and Token F1."""

from __future__ import annotations

import re
import string


def _normalize(s: str) -> str:
    return " ".join(s.split()).lower()


def _normalize_squad(s: str) -> str:
    def remove_articles(t: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", t, flags=re.IGNORECASE)

    def remove_punc(t: str) -> str:
        return "".join(ch for ch in t if ch not in string.punctuation)

    return " ".join(remove_articles(remove_punc(s.lower())).split())


def exact_match(pred: str, gold: str) -> bool:
    pred_n = _normalize_squad(pred)
    gold_n = _normalize_squad(gold)
    if pred_n == gold_n:
        return True
    # Containment check for extractive QA
    if gold_n in pred_n or pred_n in gold_n:
        return True
    return False


def token_f1(pred: str, gold: str) -> float:
    pred_tok = set(_normalize(pred).split())
    gold_tok = set(_normalize(gold).split())
    if not gold_tok:
        return 1.0 if not pred_tok else 0.0
    common = len(pred_tok & gold_tok)
    if common == 0:
        return 0.0
    prec = common / len(pred_tok)
    rec = common / len(gold_tok)
    return 2 * prec * rec / (prec + rec)


def compute_qa_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    assert len(predictions) == len(references)
    n = len(predictions)
    if n == 0:
        return {"exact_match": 0.0, "token_f1": 0.0, "n": 0.0}
    em = sum(1 for p, r in zip(predictions, references) if exact_match(p, r)) / n
    f1 = sum(token_f1(p, r) for p, r in zip(predictions, references)) / n
    return {"exact_match": em, "token_f1": f1, "n": float(n)}
