#!/usr/bin/env python3
"""
Generate Statement 1: Spelling (అక్షరక్రమం) questions - Telugu
Target: 30,000 pairs (15% of 200,000)
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_telugu.telugu_grammar import get_telugu_aksharas  # noqa: E402
from group1_telugu.telugu_vocabulary import (  # noqa: E402
    EASY_WORDS_UNIQUE,
    HARD_WORDS_UNIQUE,
    MEDIUM_WORDS_UNIQUE,
)
from group1_telugu.prompt_utils_telugu import format_qa_pair_telugu  # noqa: E402

# Expand word lists to reach target count
EASY_WORDS = EASY_WORDS_UNIQUE * 50
MEDIUM_WORDS = MEDIUM_WORDS_UNIQUE * 60
HARD_WORDS = HARD_WORDS_UNIQUE * 70


def get_telugu_grapheme_clusters(word: str) -> list[str]:
    """
    Get aksharas (syllabic units) for Telugu word.
    Per Telugu linguistics: Conjuncts = 1 unit, Anusvara = part of preceding.
    Used for: Counting, length, position, spelling (S1-S4, S7, S9, S10).
    """
    return get_telugu_aksharas(word)


def generate_spelling_answer(word: str) -> str:
    """Generate spelling answer as hyphen-separated aksharas (e.g. పు-స్త-కం)"""
    aksharas = get_telugu_aksharas(word)
    return "-".join(aksharas)


# Spelling templates: sequence of aksharas in a word
TEMPLATES_SPELLING = [
    '"{word}" పదం యొక్క అక్షరక్రమం ఏమిటి?',
    '"{word}" పదం యొక్క సరైన అక్షరక్రమం ఏది?',
    '"{word}" అనే పదాన్ని అక్షరాలుగా విడదీయండి?',
    '"{word}" పదం యొక్క స్పెల్లింగ్ చెప్పండి?',
    '"{word}" పదాన్ని అక్షరాల వారీగా వ్రాయండి?',
    '"{word}" పదాన్ని తప్పు లేకుండా ఎలా వ్రాయాలి?',
    '"{word}" పదం యొక్క స్పెల్లింగ్ సమాచారం ఇవ్వండి?',
    '"{word}" పదాన్ని అక్షరాలుగా విభజించండి?',
    '"{word}" పదం యొక్క అక్షర నిర్మాణం ఏమిటి?',
    '"{word}" పదం యొక్క స్పెల్లింగ్ చెప్పగలరా?',
]

# Listing templates: list aksharas separately
TEMPLATES_LISTING = [
    '"{word}" పదంలోని అన్ని అక్షరాలను జాబితా చేయండి?',
    '"{word}" పదంలోని అక్షరాలను వేరువేరుగా వ్రాయండి?',
    '"{word}" పదంలోని అక్షరాలను క్రమంలో చూపించండి?',
    '"{word}" పదంలోని అక్షరాల జాబితా ఇవ్వండి?',
    '"{word}" పదంలోని అక్షరాలను ఒక్కొక్కటిగా చెప్పండి?',
    '"{word}" పదంలోని అక్షరాలను వ్రాయండి?',
    '"{word}" పదంలో ఏయే అక్షరాలు ఉన్నాయి?',
    '"{word}" పదంలోని అక్షరాలను క్రమాంకంలో ఇవ్వండి?',
    '"{word}" పదంలోని అక్షరాలను విడదీసి జాబితా చేయండి?',
    '"{word}" పదంలోని అక్షరాలను ప్రత్యేకంగా చెప్పండి?',
    '"{word}" పదంలోని స్వరాలు మరియు వ్యంజనాలను జాబితా చేయండి?',
    '"{word}" పదంలోని అక్షరాలను అనుక్రమంగా వ్రాయండి?',
    '"{word}" పదంలోని ఒక్కొక్క అక్షరాన్ని పేర్కొనండి?',
    '"{word}" పదంలోని అక్షరాలను విభజించి వ్రాయండి?',
    '"{word}" పదంలోని అక్షర భాగాలను జాబితా చేయండి?',
    '"{word}" పదంలోని అక్షరాలను విడదీసి జాబితా రూపంలో ఇవ్వండి?',
    '"{word}" పదంలోని అక్షరాలను చూపించండి?',
    '"{word}" పదంలోని అక్షరాలను ప్రత్యేకంగా తెలియజేయండి?',
    '"{word}" పదంలోని అక్షరాలు ఏవి?',
    '"{word}" పదంలోని అక్షరాలను అనుక్రమంగా వ్రాయండి?',
]

TEMPLATES = TEMPLATES_SPELLING + TEMPLATES_LISTING


def generate_listing_answer(word: str, template: str) -> str:
    """Generate listing answer based on specific template rules (akshara-level)"""
    clusters = get_telugu_aksharas(word)

    if "అనుక్రమంగా వ్రాయండి" in template or "క్రమంలో చూపించండి" in template or "ప్రత్యేకంగా తెలియజేయండి" in template:
        return ", ".join(clusters)
    elif "అక్షర భాగాలను జాబితా చేయండి" in template:
        return ", ".join(clusters)
    elif "విభజించి వ్రాయండి" in template:
        return "-".join(clusters)
    elif "విడదీసి జాబితా రూపంలో ఇవ్వండి" in template:
        return ", ".join(clusters)
    elif "అక్షరాలు ఏవి" in template:
        if len(clusters) == 2:
            return f"{clusters[0]} మరియు {clusters[1]}"
        return ", ".join(clusters)
    elif "క్రమాంకంలో ఇవ్వండి" in template:
        return ", ".join([f"{i+1}: {c}" for i, c in enumerate(clusters)])
    elif "స్వరాలు మరియు వ్యంజనాలను జాబితా చేయండి" in template:
        return ", ".join(clusters)
    else:
        return ", ".join(clusters)


if __name__ == "__main__":
    all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
    samples = []
    target_count = 30000

    # Generate all unique combinations first
    unique_combinations = {}
    for word in set(all_words):
        for template_idx, template in enumerate(TEMPLATES):
            query = template.format(word=word)
            if template in TEMPLATES_SPELLING:
                answer = generate_spelling_answer(word)
            else:
                answer = generate_listing_answer(word, template)
            unique_combinations[(word, template_idx)] = (query, answer)

    if len(unique_combinations) >= target_count:
        samples = list(unique_combinations.values())[:target_count]
    else:
        samples = list(unique_combinations.values())
        while len(samples) < target_count:
            word = random.choice(list(set(all_words)))
            template_idx = random.randint(0, len(TEMPLATES) - 1)
            template = TEMPLATES[template_idx]
            query = template.format(word=word)
            if template in TEMPLATES_SPELLING:
                answer = generate_spelling_answer(word)
            else:
                answer = generate_listing_answer(word, template)
            samples.append((query, answer))

    random.shuffle(samples)

    output_file = os.path.join(os.path.dirname(__file__), "group1_s1.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_telugu(query, answer) + "\n")

    print(f"S1 Spelling (Telugu): Generated {len(samples)} samples")
