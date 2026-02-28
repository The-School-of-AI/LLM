"""
Generation metrics: Exact Match, Token F1, optional SQuAD-style normalization, BLEU, ROUGE-L.
"""

from __future__ import annotations

import re
import string


def _normalize(s: str) -> str:
    """Simple normalization: lowercase, collapse whitespace."""
    return " ".join(s.split()).lower()


def _normalize_squad(s: str) -> str:
    """SQuAD-style: remove articles (a/an/the), punctuation, lowercase, collapse whitespace."""
    def remove_articles(t: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", t, flags=re.IGNORECASE)

    def remove_punc(t: str) -> str:
        return "".join(ch for ch in t if ch not in string.punctuation)

    return " ".join(remove_articles(remove_punc(s.lower())).split())


def exact_match(pred: str, gold: str, normalize: bool = True, use_squad_normalize: bool = False) -> bool:
    if use_squad_normalize:
        pred = _normalize_squad(pred)
        gold = _normalize_squad(gold)
    elif normalize:
        pred = _normalize(pred)
        gold = _normalize(gold)
    if pred == gold:
        return True
    if gold in pred or pred in gold:
        return True
    return False


def token_f1(pred: str, gold: str, normalize: bool = True, use_squad_normalize: bool = False) -> float:
    if use_squad_normalize:
        pred = _normalize_squad(pred)
        gold = _normalize_squad(gold)
    elif normalize:
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


def _bleu_score(pred: str, ref: str) -> float:
    """Sentence BLEU (pred as hypothesis, ref as reference). Returns 0..1 or 0 if error."""
    try:
        import nltk
        nltk.data.find("tokenizers/punkt")
    except (ImportError, LookupError):
        try:
            import nltk
            nltk.download("punkt", quiet=True)
        except Exception:
            return 0.0
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tok = ref.split()
        pred_tok = pred.split()
        if not ref_tok:
            return 1.0 if not pred_tok else 0.0
        smooth = SmoothingFunction()
        return sentence_bleu([ref_tok], pred_tok, smoothing_function=smooth.method1)
    except Exception:
        return 0.0


def _rouge_l_score(pred: str, ref: str) -> float:
    """ROUGE-L F1 (longest common subsequence). Returns 0..1 or 0 if rouge_score not installed."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        s = scorer.score(ref, pred)
        return s["rougeL"].fmeasure
    except ImportError:
        return 0.0
    except Exception:
        return 0.0


def compute_generation_metrics(
    predictions: list[str],
    references: list[str],
    use_f1: bool = True,
    use_squad_normalize: bool = False,
    use_bleu: bool = False,
    use_rouge: bool = False,
) -> dict[str, float]:
    assert len(predictions) == len(references)
    n = len(predictions)
    em_count = sum(
        1 for p, r in zip(predictions, references)
        if exact_match(p, r, use_squad_normalize=use_squad_normalize)
    )
    result: dict[str, float] = {
        "exact_match": em_count / n if n else 0.0,
        "n": float(n),
    }
    if use_f1:
        result["token_f1"] = sum(
            token_f1(p, r, use_squad_normalize=use_squad_normalize)
            for p, r in zip(predictions, references)
        ) / n if n else 0.0
    if use_bleu:
        result["bleu"] = sum(_bleu_score(p, r) for p, r in zip(predictions, references)) / n if n else 0.0
    if use_rouge:
        result["rouge_l"] = sum(_rouge_l_score(p, r) for p, r in zip(predictions, references)) / n if n else 0.0
    return result
