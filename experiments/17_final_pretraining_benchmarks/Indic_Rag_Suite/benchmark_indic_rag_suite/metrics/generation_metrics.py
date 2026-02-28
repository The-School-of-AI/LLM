"""
Generation metrics: Exact Match and Token F1 for answer comparison.
"""

from __future__ import annotations


def _normalize(s: str) -> str:
    return " ".join(s.split()).lower()


def exact_match(pred: str, gold: str, normalize: bool = True) -> bool:
    if normalize:
        pred = _normalize(pred)
        gold = _normalize(gold)
    if pred == gold:
        return True
    if gold in pred or pred in gold:
        return True
    return False


def token_f1(pred: str, gold: str, normalize: bool = True) -> float:
    if normalize:
        pred = _normalize(pred)
        gold = _normalize(gold)
    pred_tok = set(pred.split())
    gold_tok = set(gold.split())
    if not gold_tok:
        return 1.0 if not pred_tok else 0.0
    common = len(pred_tok & gold_tok)
    prec = common / len(pred_tok) if pred_tok else 0.0
    rec = common / len(gold_tok) if gold_tok else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def compute_generation_metrics(
    predictions: list[str],
    references: list[str],
    use_f1: bool = False,
) -> dict[str, float]:
    assert len(predictions) == len(references)
    n = len(predictions)
    em_count = sum(1 for p, r in zip(predictions, references) if exact_match(p, r))
    result: dict[str, float] = {
        "exact_match": em_count / n if n else 0.0,
        "n": float(n),
    }
    if use_f1:
        result["token_f1"] = sum(token_f1(p, r) for p, r in zip(predictions, references)) / n if n else 0.0
    return result
