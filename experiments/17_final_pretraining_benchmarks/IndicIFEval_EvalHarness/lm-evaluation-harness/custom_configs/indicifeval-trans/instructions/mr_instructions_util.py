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
    # Nouns (नाम)
    'पुस्तक', 'मुलगा', 'झाड', 'रस्ता', 'घर',
    # Verbs (क्रियापद)
    'खाणे', 'जाणे', 'पाहणे', 'विचार करणे', 'बोलणे',
    # Adjectives (विशेषण)
    'लांब', 'गोड', 'वेगवान', 'सुंदर', 'जुने'
]

# ISO 639-1 codes to language names in Marathi.
LANGUAGE_CODES = immutabledict.immutabledict(
    {
        "en": "इंग्रजी",
        "es": "स्पॅनिश",
        "pt": "पोर्तुगीज",
        "ar": "अरबी",
        "hi": "हिंदी",
        "fr": "फ्रेंच",
        "ru": "रशियन",
        "de": "जर्मन",
        "ja": "जपानी",
        "it": "इटालियन",
        "bn": "बंगाली",
        "uk": "युक्रेनियन",
        "th": "थाई",
        "ur": "उर्दू",
        "ta": "तमिळ",
        "te": "तेलुगु",
        "bg": "बल्गेरियन",
        "ko": "कोरियन",
        "pl": "पोलिश",
        "he": "हिब्रू",
        "fa": "पर्शियन",
        "vi": "व्हिएतनामी",
        "ne": "नेपाळी",
        "sw": "स्वाहिली",
        "kn": "कन्नड",
        "mr": "मराठी",
        "gu": "गुजराती",
        "pa": "पंजाबी",
        "ml": "मल्याळम",
        "fi": "फिन्निश",
    }
)


_ALPHABETS = r"([\u0900-\u097F])"  # Devanagari script Unicode range
_PREFIXES = r"(श्री|सौ|कु)[.]"
_SUFFIXES = r"(लि|प्रा|सीओ|ज्युनियर|सिनियर)"
_STARTERS = r"(तो|ती|ते|हे|आम्ही|पण|तरीही|की|येथे|कुठे)"
_ACRONYMS = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = r"[.](com|net|org|io|gov|edu|me)"
_DIGITS = r"([\u0966-\u096F])"  # Devanagari digits Unicode range
_MULTIPLE_DOTS = r"।{2,}|\.{2,}"


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
        # Use indicnlp sentence splitter for Marathi
        input_sentences.extend(sentence_split(
            input_paragraph, lang="mar_Deva"
        ))
    
    if input_sentences and input_sentences[0].strip().endswith(":"):
        input_sentences = input_sentences[1:]
    
    return input_sentences

def is_marathi_word(token):
    """Check if token is a Devanagari word (excluding punctuation like '।', '॥')."""
    if token in {'।', '॥'}:
        return False
    return re.fullmatch(r'[\u0900-\u097F]+', token) is not None

def is_word(token):
    """Check if token is a word (excluding punctuation like '।', '॥', '.', ';', etc.)."""
    # Define regex to match pure punctuation (including English and Devanagari punctuation)
    if token in {'।', '॥'} or re.fullmatch(r'[^\w\s]', token):
        return False
    return True

def tokenize_only_words(text, lang='mr'):
    """Tokenizes text and returns only word tokens for the specified language."""
    tokens = indic_tokenize.trivial_tokenize(text, lang)
    words_only = [tok for tok in tokens if is_word(tok)]
    return words_only

def count_words(text):
    """Counts the number of words."""
    tokens = tokenize_only_words(text, lang='mr')
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
