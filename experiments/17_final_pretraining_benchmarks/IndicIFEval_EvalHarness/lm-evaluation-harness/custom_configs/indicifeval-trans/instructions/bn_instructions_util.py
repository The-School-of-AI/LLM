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
    'বই', 'শিশু', 'গাছ', 'রাস্তা', 'বাড়ি',
    # Verbs
    'খাওয়া', 'যাওয়া', 'দেখা', 'ভাবা', 'বলা',
    # Adjectives
    'লম্বা', 'মিষ্টি', 'দ্রুত', 'সুন্দর', 'পুরানো'
]

# ISO 639-1 codes to language names.
LANGUAGE_CODES = immutabledict.immutabledict(
    {
        "en": "ইংরেজি",
        "es": "স্প্যানিশ",
        "pt": "পর্তুগিজ",
        "ar": "আরবি",
        "hi": "হিন্দি",
        "fr": "ফরাসি",
        "ru": "রাশিয়ান",
        "de": "জার্মান",
        "ja": "জাপানি",
        "it": "ইতালীয়",
        "bn": "বাংলা",
        "uk": "ইউক্রেনীয়",
        "th": "থাই",
        "ur": "উর্দু",
        "ta": "তামিল",
        "te": "তেলুগু",
        "bg": "বুলগেরীয়",
        "ko": "কোরিয়ান",
        "pl": "পোলিশ",
        "he": "হিব্রু",
        "fa": "ফার্সি",
        "vi": "ভিয়েতনামী",
        "ne": "নেপালি",
        "sw": "সোয়াহিলি",
        "kn": "কন্নড়",
        "mr": "মারাঠি",
        "gu": "গুজরাটি",
        "pa": "পাঞ্জাবি",
        "ml": "মালায়ালাম",
        "fi": "ফিনিশ",
    }
)


_ALPHABETS = r"([\u0980-\u09FF])"
_PREFIXES = r"(ডঃ|শ্রী|শ্রীমতী|সুশ্রী)[.]"
_SUFFIXES = r"(লিঃ|প্রাঃ|কোং|জুনিয়র|সিনিয়র)"
_STARTERS = r"(সে|তারা|এই|এরা|আমরা|কিন্তু|যদিও|যে|এখানে|যেখানে)"
_ACRONYMS = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = r"[.](com|net|org|io|gov|edu|me)"
_DIGITS = r"([\u09E6-\u09EF])"
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
            input_paragraph, lang="ben_Beng", delim_pat=DELIM_PAT_NO_DANDA
        ))
    
    if input_sentences and input_sentences[0].strip().endswith(":"):
        input_sentences = input_sentences[1:]
    
    return input_sentences

def is_bengali_word(token):
    """Check if token is a Bengali word (excluding punctuation like '।', '॥')."""
    if token in {'।', '॥'}:
        return False
    return re.fullmatch(r'[\u0980-\u09FF]+', token) is not None

def is_word(token):
    """Check if token is a word (excluding punctuation like '।', '॥', '.', ';', etc.)."""
    # Define regex to match pure punctuation (including English and Bengali punctuation)
    if token in {'।', '॥'} or re.fullmatch(r'[^\w\s]', token):
        return False
    return True

def tokenize_only_words(text, lang='bn'):
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