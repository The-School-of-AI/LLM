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
from indicnlp.tokenize import indic_tokenize
from indicnlp.tokenize.sentence_tokenize import DELIM_PAT_NO_DANDA, sentence_split
from packaging.version import parse as parse_version

# Downloading 'punkt' with nltk<3.9 has a remote code vuln.
# see  https://github.com/EleutherAI/lm-evaluation-harness/issues/2210
# and https://github.com/nltk/nltk/issues/3266
# for more information.

RANK = os.environ.get("LOCAL_RANK", "0")


WORD_LIST = [
    # Nouns
    'ਕਿਤਾਬ', 'ਬੱਚਾ', 'ਰੁੱਖ', 'ਸੜਕ', 'ਘਰ',
    # Verbs
    'ਖਾਣਾ', 'ਜਾਣਾ', 'ਦੇਖਣਾ', 'ਸੋਚਣਾ', 'ਬੋਲਣਾ',
    # Adjectives
    'ਲੰਬਾ', 'ਮਿੱਠਾ', 'ਤੇਜ਼', 'ਸੁੰਦਰ', 'ਪੁਰਾਣਾ'
]

# ISO 639-1 codes to language names.
LANGUAGE_CODES = immutabledict.immutabledict(
    {
        "en": "ਅੰਗਰੇਜ਼ੀ",
        "es": "ਸਪੇਨੀ",
        "pt": "ਪੁਰਤਗਾਲੀ",
        "ar": "ਅਰਬੀ",
        "hi": "ਹਿੰਦੀ",
        "fr": "ਫਰਾਂਸੀਸੀ",
        "ru": "ਰੂਸੀ",
        "de": "ਜਰਮਨ",
        "ja": "ਜਾਪਾਨੀ",
        "it": "ਇਤਾਲਵੀ",
        "bn": "ਬੰਗਾਲੀ",
        "uk": "ਯੂਕਰੇਨੀ",
        "th": "ਥਾਈ",
        "ur": "ਉਰਦੂ",
        "ta": "ਤਮਿਲ",
        "te": "ਤੇਲਗੂ",
        "bg": "ਬੁਲਗਾਰੀਆਈ",
        "ko": "ਕੋਰੀਆਈ",
        "pl": "ਪੋਲਿਸ਼",
        "he": "ਹਿਬਰੂ",
        "fa": "ਫ਼ਾਰਸੀ",
        "vi": "ਵੀਅਤਨਾਮੀ",
        "ne": "ਨੇਪਾਲੀ",
        "sw": "ਸਵਾਹਿਲੀ",
        "kn": "ਕੰਨੜ",
        "mr": "ਮਰਾਠੀ",
        "gu": "ਗੁਜਰਾਤੀ",
        "pa": "ਪੰਜਾਬੀ",
        "ml": "ਮਲਿਆਲਮ",
        "fi": "ਫਿਨਿਸ਼",
    }
)


_ALPHABETS = r"([\u0A00-\u0A7F])"
_PREFIXES = r"(ਡਾ|ਸ੍ਰੀ|ਸ੍ਰੀਮਤੀ|ਸੁਸ਼੍ਰੀ)[.]"
_SUFFIXES = r"(ਲਿ|ਪ੍ਰਾ|ਸੀਓ|ਜੂਨੀਅਰ|ਸੀਨੀਅਰ)"
_STARTERS = r"(ਉਹ|ਇਹ|ਅਸੀਂ|ਪਰ|ਹਾਲਾਂਕਿ|ਕਿ|ਇੱਥੇ|ਜਿੱਥੇ)"
_ACRONYMS = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = r"[.](com|net|org|io|gov|edu|me)"
_DIGITS = r"([\u0A66-\u0A6F])"
_MULTIPLE_DOTS = r"।{2,}|\.{2,}"


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
    
        input_sentences.extend(sentence_split(
            input_paragraph, lang="pan_Gurm", delim_pat=DELIM_PAT_NO_DANDA
        ))
    
    if input_sentences and input_sentences[0].strip().endswith(":"):
        input_sentences = input_sentences[1:]
    
    return input_sentences

def is_punjabi_word(token):
    """Check if token is a Gurmukhi word (excluding punctuation like '।', '॥')."""
    if token in {'।', '॥'}:
        return False
    return re.fullmatch(r'[\u0A00-\u0A7F]+', token) is not None

def is_word(token):
    """Check if token is a word (excluding punctuation like '।', '॥', '.', ';', etc.)."""
    # Define regex to match pure punctuation (including English and Punjabi punctuation)
    if token in {'।', '॥'} or re.fullmatch(r'[^\w\s]', token):
        return False
    return True

def tokenize_only_words(text, lang='pa'):
    tokens = indic_tokenize.trivial_tokenize(text, lang)
    words_only = [tok for tok in tokens if is_word(tok)]
    return words_only

def count_words(text):
    """Counts the number of words."""
    tokens = tokenize_only_words(text)
    num_words = len(tokens)
    return num_words


@functools.lru_cache(maxsize=None)
def _get_sentence_tokenizer():
    # This part is for English and remains unchanged.
    import nltk.data
    return nltk.data.load("nltk:tokenizers/punkt/english.pickle")

def count_sentences(text):
    """Count the number of sentences."""
    input_sentences = split_into_sentences(text)
    return len(input_sentences)


def generate_keywords(num_keywords):
    """Randomly generates a few keywords."""
    return random.sample(WORD_LIST, k=num_keywords)