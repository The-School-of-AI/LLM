# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility library of instructions."""

import functools
import os
import random
import re
from importlib.metadata import version

import immutabledict

from packaging.version import parse as parse_version


# Downloading 'punkt' with nltk<3.9 has a remote code vuln.
# see  https://github.com/EleutherAI/lm-evaluation-harness/issues/2210
# and https://github.com/nltk/nltk/issues/3266
# for more information.

RANK = os.environ.get("LOCAL_RANK", "0")


WORD_LIST = [
    # Nouns (నామవాచకాలు)
    'పుస్తకం', 'పిల్లవాడు', 'చెట్టు', 'రహదారి', 'ఇల్లు',
    # Verbs (క్రియలు)
    'తినడం', 'వెళ్ళడం', 'చూడటం', 'ఆలోచించడం', 'మాట్లాడటం',
    # Adjectives (విశేషణాలు)
    'పొడవైన', 'తీయని', 'వేగవంతమైన', 'అందమైన', 'పాత'
]

# ISO 639-1 codes to language names in Telugu.
LANGUAGE_CODES = immutabledict.immutabledict(
    {
        "en": "ఇంగ్లీష్",
        "es": "స్పానిష్",
        "pt": "పోర్చుగీస్",
        "ar": "అరబిక్",
        "hi": "హిందీ",
        "fr": "ఫ్రెంచ్",
        "ru": "రష్యన్",
        "de": "జర్మన్",
        "ja": "జపనీస్",
        "it": "ఇటాలియన్",
        "bn": "బెంగాలీ",
        "uk": "ఉక్రేనియన్",
        "th": "థాయ్",
        "ur": "ఉర్దూ",
        "ta": "తమిళం",
        "te": "తెలుగు",
        "bg": "బల్గేరియన్",
        "ko": "కొరియన్",
        "pl": "పోలిష్",
        "he": "హిబ్రూ",
        "fa": "పర్షియన్",
        "vi": "వియత్నామీస్",
        "ne": "నేపాలీ",
        "sw": "స్వాహిలి",
        "kn": "కన్నడ",
        "mr": "మరాఠీ",
        "gu": "గుజరాతీ",
        "pa": "పంజాబీ",
        "ml": "మలయాళం",
        "fi": "ఫిన్నిష్",
    }
)


_ALPHABETS = r"([\u0C00-\u0C7F])"  # Telugu script Unicode range
_PREFIXES = r"(శ్రీ|శ్రీమతి|కుమారి)[.]"
_SUFFIXES = r"(లిమిటెడ్|ప్రైవేట్|కో|జూనియర్|సీనియర్)"
_STARTERS = r"(అతను|ఆమె|వారు|ఇది|మేము|కానీ|అయినప్పటికీ|అని|ఇక్కడ|ఎక్కడ)"
_ACRONYMS = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = r"[.](com|net|org|io|gov|edu|me)"
_DIGITS = r"([\u0C66-\u0C6F])"  # Telugu digits Unicode range
_MULTIPLE_DOTS = r"\.{2,}"


from indicnlp.tokenize import indic_tokenize
from indicnlp.tokenize.sentence_tokenize import sentence_split

def split_into_sentences(text):
    """Split the text into sentences.

    Args:
      text: A string that consists of more than or equal to one sentences.

    Returns:
      A list of strings where each string is a sentence.
    """
    input_paragraphs = text.split("\n\n")
    
    input_sentences = []
    
    for input_paragraph in input_paragraphs:
        # Use indicnlp sentence splitter for Telugu
        input_sentences.extend(sentence_split(
            input_paragraph, lang="tel_Telu"
        ))
    
    if input_sentences and input_sentences[0].strip().endswith(":"):
        input_sentences = input_sentences[1:]
    
    return input_sentences

def is_telugu_word(token):
    """Check if token is a Telugu word (excluding punctuation)."""
    return re.fullmatch(r'[\u0C00-\u0C7F]+', token) is not None

def is_word(token):
    """Check if token is a word (excluding punctuation like '.', ';', etc.)."""
    # Define regex to match pure punctuation
    if re.fullmatch(r'[^\w\s]', token):
        return False
    return True

def tokenize_only_words(text, lang='te'):
    """Tokenizes text and returns only word tokens for the specified language."""
    tokens = indic_tokenize.trivial_tokenize(text, lang)
    words_only = [tok for tok in tokens if is_word(tok)]
    return words_only

def count_words(text):
    """Counts the number of words."""
    tokens = tokenize_only_words(text, lang='te')
    num_words = len(tokens)
    return num_words


@functools.lru_cache(maxsize=None)
def _get_sentence_tokenizer():
    # This function is retained for potential future use or compatibility,
    # but split_into_sentences is now the primary method.
    import nltk
    return nltk.data.load("nltk:tokenizers/punkt/english.pickle")

def count_sentences(text):
    """Count the number of sentences."""
    input_sentences = split_into_sentences(text)
    return len(input_sentences)


def generate_keywords(num_keywords):
    """Randomly generates a few keywords."""
    return random.sample(WORD_LIST, k=num_keywords)
