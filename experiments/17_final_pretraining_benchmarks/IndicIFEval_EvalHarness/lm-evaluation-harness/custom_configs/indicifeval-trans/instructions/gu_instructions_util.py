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
    'પુસ્તક', 'બાળક', 'વૃક્ષ', 'રસ્તો', 'ઘર',
    # Verbs
    'ખાવું', 'જવું', 'જોવું', 'વિચારવું', 'બોલવું',
    # Adjectives
    'લાંબુ', 'મીઠું', 'ઝડપી', 'સુંદર', 'જૂનું'
]

# ISO 639-1 codes to language names.
LANGUAGE_CODES = immutabledict.immutabledict(
    {
        "en": "અંગ્રેજી",
        "es": "સ્પેનિશ",
        "pt": "પોર્ટુગીઝ",
        "ar": "અરબી",
        "hi": "હિન્દી",
        "fr": "ફ્રેન્ચ",
        "ru": "રશિયન",
        "de": "જર્મન",
        "ja": "જાપાનીઝ",
        "it": "ઇટાલિયન",
        "bn": "બંગાળી",
        "uk": "યુક્રેનિયન",
        "th": "થાઈ",
        "ur": "ઉર્દૂ",
        "ta": "તમિલ",
        "te": "તેલુગુ",
        "bg": "બલ્ગેરિયન",
        "ko": "કોરિયન",
        "pl": "પોલિશ",
        "he": "હિબ્રુ",
        "fa": "પર્શિયન",
        "vi": "વિયેતનામીસ",
        "ne": "નેપાળી",
        "sw": "સ્વાહિલી",
        "kn": "કન્નડ",
        "mr": "મરાઠી",
        "gu": "ગુજરાતી",
        "pa": "પંજાબી",
        "ml": "મલયાલમ",
        "fi": "ફિનિશ",
    }
)


_ALPHABETS = r"([\u0A80-\u0AFF])"
_PREFIXES = r"(ડૉ|શ્રી|શ્રીમતી|સુશ્રી)[.]"
_SUFFIXES = r"(લિ|પ્રા|કું|જુનિયર|સિનિયર)"
_STARTERS = r"(તે|આ|અમે|પરંતુ|જોકે|કે|અહીં|જ્યાં)"
_ACRONYMS = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = r"[.](com|net|org|io|gov|edu|me)"
_DIGITS = r"([\u0AE6-\u0AEF])"
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
            input_paragraph, lang="guj_Gujr", delim_pat=DELIM_PAT_NO_DANDA
        ))
    
    if input_sentences and input_sentences[0].strip().endswith(":"):
        input_sentences = input_sentences[1:]
    
    return input_sentences

def is_gujarati_word(token):
    """Check if token is a Gujarati word (excluding punctuation like '।', '॥')."""
    if token in {'।', '॥'}:
        return False
    return re.fullmatch(r'[\u0A80-\u0AFF]+', token) is not None

def is_word(token):
    """Check if token is a word (excluding punctuation like '।', '॥', '.', ';', etc.)."""
    # Define regex to match pure punctuation (including English and Gujarati punctuation)
    if token in {'।', '॥'} or re.fullmatch(r'[^\w\s]', token):
        return False
    return True

def tokenize_only_words(text, lang='gu'):
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