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
    # Nouns (നാമങ്ങൾ)
    'പുസ്തകം', 'കുട്ടി', 'മരം', 'റോഡ്', 'വീട്',
    # Verbs (ക്രിയകൾ)
    'കഴിക്കുക', 'പോകുക', 'കാണുക', 'ചിന്തിക്കുക', 'സംസാരിക്കുക',
    # Adjectives (നാമവിശേഷണങ്ങൾ)
    'നീളമുള്ള', 'മധുരമുള്ള', 'വേഗതയേറിയ', 'മനോഹരമായ', 'പഴയ'
]

# ISO 639-1 codes to language names in Malayalam.
LANGUAGE_CODES = immutabledict.immutabledict(
    {
        "en": "ഇംഗ്ലീഷ്",
        "es": "സ്പാനിഷ്",
        "pt": "പോർച്ചുഗീസ്",
        "ar": "അറബിക്",
        "hi": "ഹിന്ദി",
        "fr": "ഫ്രഞ്ച്",
        "ru": "റഷ്യൻ",
        "de": "ജർമ്മൻ",
        "ja": "ജാപ്പനീസ്",
        "it": "ഇറ്റാലിയൻ",
        "bn": "ബംഗാളി",
        "uk": "ഉക്രേനിയൻ",
        "th": "തായ്",
        "ur": "ഉറുദു",
        "ta": "തമിഴ്",
        "te": "തെലുങ്ക്",
        "bg": "ബൾഗേറിയൻ",
        "ko": "കൊറിയൻ",
        "pl": "പോളിഷ്",
        "he": "ഹീബ്രു",
        "fa": "പേർഷ്യൻ",
        "vi": "വിയറ്റ്നാമീസ്",
        "ne": "നേപ്പാളി",
        "sw": "സ്വാഹിലി",
        "kn": "കന്നഡ",
        "mr": "മറാത്തി",
        "gu": "ഗുജറാത്തി",
        "pa": "പഞ്ചാബി",
        "ml": "മലയാളം",
        "fi": "ഫിന്നിഷ്",
    }
)


_ALPHABETS = r"([\u0D00-\u0D7F])"  # Malayalam script Unicode range
_PREFIXES = r"(ശ്രീ|ശ്രീമതി)[.]"
_SUFFIXES = r"(ലിമിറ്റഡ്|പ്രൈവറ്റ്|കോ|ജൂനിയർ|സീനിയർ)"
_STARTERS = r"(അവൻ|അവൾ|അവർ|ഇത്|ഞങ്ങൾ|എന്നാൽ|എങ്കിലും|എന്ന്|ഇവിടെ|എവിടെ)"
_ACRONYMS = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = r"[.](com|net|org|io|gov|edu|me)"
_DIGITS = r"([\u0D66-\u0D6F])"  # Malayalam digits Unicode range
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
        # Use indicnlp sentence splitter for Malayalam
        input_sentences.extend(sentence_split(
            input_paragraph, lang="mal_Mlym"
        ))
    
    if input_sentences and input_sentences[0].strip().endswith(":"):
        input_sentences = input_sentences[1:]
    
    return input_sentences

def is_malayalam_word(token):
    """Check if token is a Malayalam word (excluding punctuation)."""
    return re.fullmatch(r'[\u0D00-\u0D7F]+', token) is not None

def is_word(token):
    """Check if token is a word (excluding punctuation like '.', ';', etc.)."""
    # Define regex to match pure punctuation
    if re.fullmatch(r'[^\w\s]', token):
        return False
    return True

def tokenize_only_words(text, lang='ml'):
    """Tokenizes text and returns only word tokens for the specified language."""
    tokens = indic_tokenize.trivial_tokenize(text, lang)
    words_only = [tok for tok in tokens if is_word(tok)]
    return words_only

def count_words(text):
    """Counts the number of words."""
    tokens = tokenize_only_words(text, lang='ml')
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