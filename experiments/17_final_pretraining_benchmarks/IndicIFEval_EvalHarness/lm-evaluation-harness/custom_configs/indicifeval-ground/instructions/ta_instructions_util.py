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
    # Nouns (பெயர்ச்சொற்கள்)
    'புத்தகம்', 'குழந்தை', 'மரம்', 'சாலை', 'வீடு',
    # Verbs (வினைச்சொற்கள்)
    'சாப்பிடு', 'போ', 'பார்', 'சிந்தி', 'பேசு',
    # Adjectives (உரிச்சொற்கள்)
    'நீளமான', 'இனிமையான', 'வேகமான', 'அழகான', 'பழைய'
]

# ISO 639-1 codes to language names in Tamil.
LANGUAGE_CODES = immutabledict.immutabledict(
    {
        "en": "ஆங்கிலம்",
        "es": "ஸ்பானிஷ்",
        "pt": "போர்த்துகீசியம்",
        "ar": "அரபு",
        "hi": "இந்தி",
        "fr": "பிரெஞ்சு",
        "ru": "ரஷ்யன்",
        "de": "ஜெர்மன்",
        "ja": "ஜப்பானிய",
        "it": "இத்தாலியன்",
        "bn": "பெங்காலி",
        "uk": "உக்ரேனியன்",
        "th": "தாய்",
        "ur": "உருது",
        "ta": "தமிழ்",
        "te": "தெலுங்கு",
        "bg": "பல்கேரியன்",
        "ko": "கொரியன்",
        "pl": "போலிஷ்",
        "he": "ஹீப்ரு",
        "fa": "பாரசீக",
        "vi": "வியட்நாமிய",
        "ne": "நேபாளி",
        "sw": "சுவாஹிலி",
        "kn": "கன்னடம்",
        "mr": "மராத்தி",
        "gu": "குஜராத்தி",
        "pa": "பஞ்சாபி",
        "ml": "மலையாளம்",
        "fi": "ஃபின்னிஷ்",
    }
)


_ALPHABETS = r"([\u0B80-\u0BFF])"  # Tamil script Unicode range
_PREFIXES = r"(திரு|திருமதி|செல்வி)[.]"
_SUFFIXES = r"(லிமிடெட்|பிரைவேட்|கோ|ஜூனியர்|சீனியர்)"
_STARTERS = r"(அவர்|அவர்கள்|இது|இவை|நாம்|ஆனால்|இருப்பினும்|என்று|இங்கே|எங்கே)"
_ACRONYMS = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = r"[.](com|net|org|io|gov|edu|me)"
_DIGITS = r"([\u0BE6-\u0BEF])"  # Tamil digits Unicode range
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
        # Use indicnlp sentence splitter for Tamil
        input_sentences.extend(sentence_split(
            input_paragraph, lang="tam_Taml"
        ))
    
    if input_sentences and input_sentences[0].strip().endswith(":"):
        input_sentences = input_sentences[1:]
    
    return input_sentences

def is_tamil_word(token):
    """Check if token is a Tamil word (excluding punctuation)."""
    return re.fullmatch(r'[\u0B80-\u0BFF]+', token) is not None

def is_word(token):
    """Check if token is a word (excluding punctuation like '.', ';', etc.)."""
    # Define regex to match pure punctuation
    if re.fullmatch(r'[^\w\s]', token):
        return False
    return True

def tokenize_only_words(text, lang='ta'):
    """Tokenizes text and returns only word tokens for the specified language."""
    tokens = indic_tokenize.trivial_tokenize(text, lang)
    words_only = [tok for tok in tokens if is_word(tok)]
    return words_only

def count_words(text):
    """Counts the number of words."""
    tokens = tokenize_only_words(text, lang='ta')
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