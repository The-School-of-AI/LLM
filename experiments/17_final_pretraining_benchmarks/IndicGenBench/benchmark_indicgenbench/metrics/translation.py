"""Translation metrics: BLEU, chrF, METEOR via sacrebleu and nltk."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def compute_translation_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    assert len(predictions) == len(references)
    n = len(predictions)
    if n == 0:
        return {"bleu": 0.0, "chrf": 0.0, "meteor": 0.0, "n": 0.0}

    result: dict[str, float] = {"n": float(n)}

    # BLEU and chrF via sacrebleu (corpus-level)
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        result["bleu"] = bleu.score
        chrf = sacrebleu.corpus_chrf(predictions, [references])
        result["chrf"] = chrf.score
    except ImportError:
        logger.warning("sacrebleu not installed, skipping BLEU/chrF")
        result["bleu"] = 0.0
        result["chrf"] = 0.0

    # METEOR via nltk (sentence-level averaged)
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score as nltk_meteor
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        total_meteor = 0.0
        for pred, ref in zip(predictions, references):
            try:
                total_meteor += nltk_meteor([ref.split()], pred.split())
            except Exception:
                pass
        result["meteor"] = total_meteor / n
    except ImportError:
        logger.warning("nltk not installed, skipping METEOR")
        result["meteor"] = 0.0

    return result
