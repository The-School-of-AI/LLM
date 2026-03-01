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

"""Library of instructions."""

import collections
import json
import logging
import random
import re
import string
from typing import Dict, Optional, Sequence, Union

import langdetect

from . import bn_instructions_util as instructions_util
from nltk import word_tokenize as en_word_tokenize


logger = logging.getLogger(__name__)

_InstructionArgsDtype = Optional[Dict[str, Union[int, str, Sequence[str]]]]

_LANGUAGES = instructions_util.LANGUAGE_CODES

# The relational operation for comparison.
_COMPARISON_RELATION = ("less than", "at least")

# The maximum number of sentences.
_MAX_NUM_SENTENCES = 20

# The number of placeholders.
_NUM_PLACEHOLDERS = 4

# The number of bullet lists.
_NUM_BULLETS = 5

# The options of constrained response.
_CONSTRAINED_RESPONSE_OPTIONS = (
    "আমার উত্তর 'হ্যাঁ'।",
    "আমার উত্তর 'না'।",
    "আমার উত্তর 'সম্ভবত'।",
)

# The options of starter keywords.
_STARTER_OPTIONS = (
    "আমি বলব",
    "আমার উত্তর হল",
    "আমার বিশ্বাস",
    "আমার মতে",
    "আমার মনে হয়",
    "আমার অনুমান",
    "আমার অনুভব হচ্ছে",
    "আমার দৃষ্টিকোণ থেকে",
    "আমি যেভাবে দেখছি",
    "আমার অনুসারে",
    "যতদূর আমার প্রশ্ন",
    "আমার বোঝামতে",
    "আমার विचारे",
    "এই বিষয়ে আমার মতামত",
    "আমার ধারণা অনুযায়ী",
)

# The options of ending keywords.
# TODO(jeffreyzhou) add more ending options
_ENDING_OPTIONS = ("আর কোনো প্রশ্ন?", "আমি কি আপনাকে আর কোনো সাহায্য করতে পারি?")

# The number of highlighted sections.
_NUM_HIGHLIGHTED_SECTIONS = 4

# The section splitter.
_SECTION_SPLITER = ("খণ্ড", "বিভাগ")

# The number of sections.
_NUM_SECTIONS = 5

# The number of paragraphs.
_NUM_PARAGRAPHS = 5

# The postscript marker.
_POSTSCRIPT_MARKER = ("পুনশ্চ", "P.S")

# The number of keywords.
_NUM_KEYWORDS = 2

# The occurrences of a single keyword.
_KEYWORD_FREQUENCY = 3

# The occurrences of a single letter.
_LETTER_FREQUENCY = 10

# The occurrences of words rth all capital letters.
_ALL_CAPITAL_WORD_FREQUENCY = 20

# The number of words in the response.
_NUM_WORDS_LOWER_LIMIT = 100
_NUM_WORDS_UPPER_LIMIT = 500


class Instruction:
    """An instruction template."""

    def __init__(self, instruction_id):
        self.id = instruction_id

    def build_description(self, **kwargs):
        raise NotImplementedError("`build_description` not implemented.")

    def get_instruction_args(self):
        raise NotImplementedError("`get_instruction_args` not implemented.")

    def get_instruction_args_keys(self):
        raise NotImplementedError("`get_instruction_args_keys` not implemented.")

    def check_following(self, value):
        raise NotImplementedError("`check_following` not implemented.")


class ResponseLanguageChecker(Instruction):
    """Check the language of the entire response."""

    def build_description(self, *, language=None):
        """Build the instruction description.

        Args:
          language: A string representing the expected language of the response. The
            language has to comply to the 97 types defined in
            `langid.py` (https://pypi.org/project/langid/1.1.5/), which follows
            ISO 639-1 codes (https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes);
            for example, `en` for English, `zh` for Chinese, `fr` for French.

        Returns:
          A string representing the instruction description.
        """
        self._language = language
        if self._language is None:
            self._language = random.choice(list(_LANGUAGES.keys()))
        # TODO(tianjianlu): opens the description generation to more choices.
        self._description_pattern = (
            "আপনার সম্পূর্ণ প্রতিক্রিয়া {language} ভাষায় হওয়া উচিত, অন্য কোনো ভাষার অনুমতি নেই।"
        )
        return self._description_pattern.format(language=_LANGUAGES[self._language])

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"language": self._language}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["language"]

    def check_following(self, value):
        """Check if the language of the entire response follows the instruction.

        Args:
          value: A string representing the response.

        Returns:
          True if the language of `value` follows instruction; otherwise False.
        """
        assert isinstance(value, str)

        try:
            return langdetect.detect(value) == self._language
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logging.error(
                "Unable to detect language for text %s due to %s", value, e
            )  # refex: disable=pytotw.037
            return True


class NumberOfSentences(Instruction):
    """Check the number of sentences."""

    def build_description(self, *, num_sentences=None, relation=None):
        """Build the instruction description.

        Args:
          num_sentences: An integer specifying the number of sentences as a
            threshold.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of sentences < the threshold;
            if 'at least', the actual number of sentences >= the threshold.

        Returns:
          A string representing the instruction description.
        """
        # The number of sentences as a threshold for comparison.
        self._num_sentences_threshold = num_sentences
        if self._num_sentences_threshold is None or self._num_sentences_threshold < 0:
            self._num_sentences_threshold = random.randint(1, _MAX_NUM_SENTENCES)

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = (
            "আপনার প্রতিক্রিয়ায় {relation} {num_sentences} বাক্য থাকতে হবে।"
        )
        return self._description_pattern.format(
            relation=self._comparison_relation,
            num_sentences=self._num_sentences_threshold,
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "num_sentences": self._num_sentences_threshold,
            "relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_sentences", "relation"]

    def check_following(self, value):
        """Check if the number of sentences follows the instruction.

        Args:
          value: A string representing the response.

        Returns:
          True if the response follows the instruction.

        Raise:
            ValueError if the string in `instruction_args` is not in
            [`less_than`, `at_least`].
        """
        num_sentences = instructions_util.count_sentences(value)
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return num_sentences < self._num_sentences_threshold
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return num_sentences >= self._num_sentences_threshold


class PlaceholderChecker(Instruction):
    """Check the placeholders in template writing."""

    def build_description(self, *, num_placeholders=None):
        """Build the instruction description.

        Args:
          num_placeholders: An integer denoting the minimum number of
            placeholders required in the response.

        Returns:
          A string representing the instruction description.
        """
        self._num_placeholders = num_placeholders
        if self._num_placeholders is None or self._num_placeholders < 0:
            self._num_placeholders = random.randint(1, _NUM_PLACEHOLDERS)
        self._description_pattern = (
            "প্রতিক্রিয়াতে কমপক্ষে {num_placeholders} প্লেসহোল্ডার থাকতে হবে যা বর্গাকার বন্ধনী দ্বারা দেখানো হয়েছে, যেমন [ঠিকানা]।"
        )
        return self._description_pattern.format(num_placeholders=self._num_placeholders)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_placeholders": self._num_placeholders}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_placeholders"]

    def check_following(self, value):
        """Check if the number of placeholders follows the instruction.

        Args:
          value: A string representing the response.

        Returns:
          True if the actual number of placeholders in the response is greater than
          or equal to `num_placeholders`; otherwise, False.
        """
        placeholders = re.findall(r"\[.*?\]", value)
        num_placeholders = len(placeholders)
        return num_placeholders >= self._num_placeholders


class BulletListChecker(Instruction):
    """Checks the bullet list in the prompt."""

    def build_description(self, *, num_bullets=None):
        """Build the instruction description.

        Args:
          num_bullets: An integer specifying the exact number of bullet lists
            that is required to appear in the response.

        Returns:
          A string representing the instruction description.
        """
        self._num_bullets = num_bullets
        if self._num_bullets is None or self._num_bullets < 0:
            self._num_bullets = random.randint(1, _NUM_BULLETS)
        self._description_pattern = (
            "আপনার উত্তরে ঠিক {num_bullets} বুলেট পয়েন্ট থাকতে হবে। "
            + "মার্কডাউন বুলেট পয়েন্ট ব্যবহার করুন যেমন:\n"
            + "* এটি পয়েন্ট 1।\n"
            + "* এটি পয়েন্ট 2।"
        )
        return self._description_pattern.format(num_bullets=self._num_bullets)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_bullets": self._num_bullets}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_bullets"]

    def check_following(self, value):
        r"""Check if the number of bullet lists meets the requirement.

        Args:
          value: A string representing the response. The response is expected to
            contain some bullet lists that start with `\*`.

        Returns:
          True if the actual number of bullet lists in the response meets the
          requirement.
        """
        bullet_lists = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
        bullet_lists_2 = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
        num_bullet_lists = len(bullet_lists) + len(bullet_lists_2)
        return num_bullet_lists == self._num_bullets


class ConstrainedResponseChecker(Instruction):
    """Checks the constrained response."""

    def build_description(self):
        """Build the instruction description."""
        # A sequence of string(s) representing the options of the expected response.
        self._constrained_responses = _CONSTRAINED_RESPONSE_OPTIONS
        self._description_pattern = (
            "নিম্নলিখিত বিকল্পগুলির মধ্যে যেকোনো একটি দিয়ে উত্তর দিন: {response_options}"
        )
        return self._description_pattern.format(
            response_options=self._constrained_responses
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response matches the constrained options.

        Args:
          value: A string representing the response.

        Returns:
          True if the actual response contains one of the options in the constrained
          responses; otherwise False.
        """
        value = value.strip()
        for constrained_response in self._constrained_responses:
            if constrained_response in value:
                return True
        return False


class ConstrainedStartChecker(Instruction):
    """Checks the response start."""

    def build_description(self, *, starter=None):
        """Build the instruction description.

        Args:
          starter: A string representing the keyword that the response should start
            with.

        Returns:
          A string representing the instruction description.
        """
        self._starter = starter.strip() if isinstance(starter, str) else starter
        if self._starter is None:
            self._starter = random.choice(_STARTER_OPTIONS)
        self._description_pattern = (
            "কথোপকথনের সময়, যখন আপনার পালা আসবে, অনুগ্রহ করে সর্বদা {starter} দিয়ে শুরু করুন"
        )
        return self._description_pattern.format(starter=self._starter)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"starter": self._starter}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["starter"]

    def check_following(self, value):
        """Checks if the response starts with the constrained keyword or phrase.

        Args:
          value: A string representing the response.

        Returns:
          True if the response starts with the given phrase or keyword that is
          contained in `instruction_args`; otherwise, False.
        """
        response_pattern = r"^\s*" + self._starter + r".*$"
        response_with_constrained_start = re.search(
            response_pattern, value, flags=re.MULTILINE
        )
        return True if response_with_constrained_start else False


class HighlightSectionChecker(Instruction):
    """Checks the highlighted section."""

    def build_description(self, *, num_highlights=None):
        """Build the instruction description.

        Args:
          num_highlights: An integer specifying the minimum number of highlighted
            sections.

        Returns:
          A string representing the instruction description.
        """
        self._num_highlights = num_highlights
        if self._num_highlights is None or self._num_highlights < 0:
            self._num_highlights = random.randint(1, _NUM_HIGHLIGHTED_SECTIONS)

        self._description_pattern = (
            "মার্কডাউন দিয়ে আপনার উত্তরে কমপক্ষে {num_highlights}টি বিভাগ হাইলাইট করুন, "
            + "অর্থাৎ *হাইলাইট করা বিভাগ*।"
        )

        return self._description_pattern.format(num_highlights=self._num_highlights)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_highlights": self._num_highlights}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_highlights"]

    def check_following(self, value):
        """Checks if the number of highlighted sections meets the requirement.

        Args:
          value: a string representing the response. The response is expected to
            contain highlighted sections in the format of *highlighted*.

        Returns:
          True if the actual number of highlighted sections in the format of
          *highlighted sections* meets the minimum requirement; otherwise False.
        """
        num_highlights = 0
        highlights = re.findall(r"\*[^\n\*]*\*", value)
        double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", value)
        for highlight in highlights:
            if highlight.strip("*").strip():
                num_highlights += 1
        for highlight in double_highlights:
            if highlight.removeprefix("**").removesuffix("**").strip():
                num_highlights += 1

        return num_highlights >= self._num_highlights


class SectionChecker(Instruction):
    """Checks the sections."""

    def build_description(self, *, section_spliter=None, num_sections=None):
        """Build the instruction description.

        Args:
          section_spliter: A string represents the section spliter keyword that
            marks a new section, i.e., `Section` or `SECTION`.
          num_sections: An integer specifying the number of sections.

        Returns:
          A string representing the instruction description.
        """
        self._section_spliter = (
            section_spliter.strip()
            if isinstance(section_spliter, str)
            else section_spliter
        )
        if self._section_spliter is None:
            self._section_spliter = random.choice(_SECTION_SPLITER)

        self._num_sections = num_sections
        if self._num_sections is None or self._num_sections < 0:
            self._num_sections = random.randint(1, _NUM_SECTIONS)

        self._description_pattern = (
            "আপনার প্রতিক্রিয়ায় {num_sections}টি বিভাগ থাকতে হবে। প্রতিটি বিভাগের শুরু "
            + "{section_spliter} X দিয়ে চিহ্নিত করুন, যেমন:\n"
            + "{section_spliter} 1\n"
            + "[বিভাগ 1-এর বিষয়বস্তু]\n"
            + "{section_spliter} 2\n"
            + "[বিভাগ 2-এর বিষয়বস্তু]"
        )

        return self._description_pattern.format(
            num_sections=self._num_sections, section_spliter=self._section_spliter
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "section_spliter": self._section_spliter,
            "num_sections": self._num_sections,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["section_spliter", "num_sections"]

    def check_following(self, value):
        """Checks the response contains multiple sections.

        Args:
          value: A string representing the response. The response is expected
            to contain multiple sections (number of sections is greater than 1).
            A new section starts with `Section 1`, where the number denotes the
            section index.

        Returns:
          True if the number of sections in the response is greater than or equal to
          the minimum number of sections; otherwise, False.
        """
        section_splitter_patten = r"\s?" + self._section_spliter + r"\s?\d+\s?"
        sections = re.split(section_splitter_patten, value)
        num_sections = len(sections) - 1
        return num_sections >= self._num_sections


class ParagraphChecker(Instruction):
    """Checks the paragraphs."""

    def build_description(self, *, num_paragraphs=None):
        """Build the instruction description.

        Args:
          num_paragraphs: An integer specifying the number of paragraphs.

        Returns:
          A string representing the instruction description.
        """
        self._num_paragraphs = num_paragraphs
        if self._num_paragraphs is None or self._num_paragraphs < 0:
            self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

        self._description_pattern = (
            "সেখানে {num_paragraphs}টি অনুচ্ছেদ থাকতে হবে। "
            + "অনুচ্ছেদগুলি মার্কডাউন ডিভাইডার দ্বারা বিভক্ত: ***"
        )

        return self._description_pattern.format(num_paragraphs=self._num_paragraphs)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_paragraphs": self._num_paragraphs}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_paragraphs"]

    def check_following(self, value):
        """Checks the response contains required number of paragraphs.

        Args:
          value: A string representing the response. The response may contain
            paragraphs that are separated by the markdown divider: `***`.

        Returns:
          True if the actual number of paragraphs is the same as required;
          otherwise, False.
        """
        paragraphs = re.split(r"\s?\*\*\*\s?", value)
        num_paragraphs = len(paragraphs)

        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index == 0 or index == len(paragraphs) - 1:
                    num_paragraphs -= 1
                else:
                    return False

        return num_paragraphs == self._num_paragraphs


class PostscriptChecker(Instruction):
    """Checks the postscript."""

    def build_description(self, *, postscript_marker=None):
        """Build the instruction description.

        Args:
          postscript_marker: A string containing the keyword that marks the start
            of the postscript section.

        Returns:
          A string representing the instruction description.
        """
        self._postscript_marker = (
            postscript_marker.strip()
            if isinstance(postscript_marker, str)
            else postscript_marker
        )
        if self._postscript_marker is None:
            self._postscript_marker = random.choice(_POSTSCRIPT_MARKER)

        self._description_pattern = (
            "আপনার প্রতিক্রিয়ার শেষে, অনুগ্রহ করে স্পষ্টভাবে {postscript} দিয়ে শুরু হওয়া একটি পোস্টস্ক্রিপ্ট যোগ করুন"
        )

        return self._description_pattern.format(postscript=self._postscript_marker)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"postscript_marker": self._postscript_marker}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["postscript_marker"]

    def check_following(self, value):
        """Checks if the response follows the postscript format.

        Args:
          value: a string representing the response. The response is expected to
            contain a postscript section.

        Returns:
          True if the response contains a postscript section starting with
          the keyword containing in the `instruction_args`; otherwise False.
        """
        value = value.lower()
        if self._postscript_marker == "P.P.S":
            postscript_pattern = r"\s*p\.\s?p\.\s?s.*$"
        elif self._postscript_marker == "P.S.":
            postscript_pattern = r"\s*p\.\s?s\..*$"
        else:
            postscript_pattern = r"\s*" + self._postscript_marker.lower() + r".*$"
        postscript = re.findall(postscript_pattern, value, flags=re.MULTILINE)
        return True if postscript else False


class RephraseChecker(Instruction):
    """Checks the rephrase."""

    def build_description(self, *, original_message):
        """Build the instruction description.

        Args:
          original_message: A string representing the original message. The
            rephrased response should only change its words/sentences in between
            its two asterisks, for example, *change me*. Both original and rephrased
            messages should contain the changes in the form of *change me*.

        Returns:
          A string representing the instruction description.
        """
        if not self.is_change(original_message):
            raise ValueError(
                f"Message {original_message} does not contain changes "
                "in the form of *change me*."
            )

        self._reference_without_change = original_message
        self._description = (
            "পুনর্লিখন: আপনার পুনর্লিখিত প্রতিক্রিয়ায় শুধুমাত্র "
            + "দুটি তারকা চিহ্নের মাঝখানের শব্দ/বাক্য পরিবর্তন করা উচিত "
            + "যেমন *আমাকে পরিবর্তন করুন*।"
        )
        return self._description

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"original_message": self._reference_without_change}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["original_message"]

    def check_following(self, value):
        r"""Checks if the rephrasing follows the instruction.

        Args:
          value: A string representing the response, which is expected to rephras
            the string of `instruction_args`.

        Returns:
          True if `value` and `instruction_args` only differ by the words/sentences
          in between two asterisks such as *change me*; otherwise, False.
        """

        if not self.is_change(value):
            raise ValueError(
                f"value {value} does not contain changes in the form of *change me*."
            )

        response_without_changes = self.strip_changes(value)
        reference_without_changes = self.strip_changes(self._reference_without_change)

        return response_without_changes == reference_without_changes

    def is_change(self, response):
        """Check if there is change in the response in the form of *change me*."""
        return re.search(r"\*.*\*", response)

    def strip_changes(self, response):
        """Strips off the changes."""
        return re.sub(r"\*.*\*", "", response)


class KeywordChecker(Instruction):
    """Check the existence of certain keywords."""

    def build_description(self, *, keywords=None):
        """Build the instruction description.

        Args:
          keywords: A sequence of strings representing the keywords that are
            expected in the response.

        Returns:
          A string representing the instruction description.
        """

        if not keywords:
            self._keywords = instructions_util.generate_keywords(
                num_keywords=_NUM_KEYWORDS
            )
        else:
            self._keywords = keywords
        self._keywords = sorted(self._keywords)

        self._description_pattern = "প্রতিক্রিয়াতে কীওয়ার্ড {keywords} অন্তর্ভুক্ত করুন।"

        return self._description_pattern.format(keywords=self._keywords)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"keywords": self._keywords}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keywords"]

    def check_following(self, value):
        """Check if the response contain the expected keywords."""
        for keyword in self._keywords:
            if not re.search(keyword, value, flags=re.IGNORECASE):
                return False
        return True


class KeywordFrequencyChecker(Instruction):
    """Check the keyword frequency."""

    def build_description(self, *, keyword=None, frequency=None, relation=None):
        """Build the instruction description.

        Args:
          keyword: A string representing a keyword that is expected in the response.
          frequency: An integer specifying the number of times `keyword` is expected
            to appear in the response.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of occurrences < frequency;
            if 'at least', the actual number of occurrences >= frequency.

        Returns:
          A string representing the instruction description.
        """
        if not keyword:
            self._keyword = instructions_util.generate_keywords(num_keywords=1)[0]
        else:
            self._keyword = keyword.strip()

        self._frequency = frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, _KEYWORD_FREQUENCY)

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = (
            "আপনার প্রতিক্রিয়ায়, '{keyword}' শব্দটি {relation} "
            + "{frequency} বার উপস্থিত থাকতে হবে।"
        )

        return self._description_pattern.format(
            keyword=self._keyword,
            relation=self._comparison_relation,
            frequency=self._frequency,
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "keyword": self._keyword,
            "frequency": self._frequency,
            "relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keyword", "frequency", "relation"]

    def check_following(self, value):
        """Checks if the response contain the keyword with required frequency."""
        actual_occurrences = len(re.findall(self._keyword, value, flags=re.IGNORECASE))

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return actual_occurrences < self._frequency
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return actual_occurrences >= self._frequency


class NumberOfWords(Instruction):
    """Checks the number of words."""

    def build_description(self, *, num_words=None, relation=None):
        """Build the instruction description.

        Args:
          num_words: An integer specifying the number of words contained in the
            response.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of words < num_words;
            if 'at least', the actual number of words >= num_words.

        Returns:
          A string representing the instruction description.
        """

        self._num_words = num_words
        if self._num_words is None or self._num_words < 0:
            self._num_words = random.randint(
                _NUM_WORDS_LOWER_LIMIT, _NUM_WORDS_UPPER_LIMIT
            )

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = "{relation} {num_words} শব্দে উত্তর দিন।"

        return self._description_pattern.format(
            relation=self._comparison_relation, num_words=self._num_words
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_words": self._num_words, "relation": self._comparison_relation}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_words", "relation"]

    def check_following(self, value):
        """Checks if the response contains the expected number of words."""
        num_words = instructions_util.count_words(value)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return num_words < self._num_words
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return num_words >= self._num_words


class JsonFormat(Instruction):
    """Check the Json format."""

    def build_description(self):
        self._description_pattern = (
            "সম্পূর্ণ আউটপুট JSON ফরম্যাটে মোড়ানো উচিত। আপনি মার্কডাউন টিক ব্যবহার করতে পারেন যেমন ```।"
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        value = (
            value.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            json.loads(value)
        except ValueError:
            return False
        return True


class ParagraphFirstWordCheck(Instruction):
    """Check the paragraph and the first word of the nth paragraph."""

    def build_description(
        self, num_paragraphs=None, nth_paragraph=None, first_word=None
    ):
        r"""Build the instruction description.

        Args:
          num_paragraphs: An integer indicating the number of paragraphs expected
            in the response. A paragraph is a subset of the string that is
            expected to be separated by '\n\n'.
          nth_paragraph: An integer indicating the paragraph number that we look at.
            Note that n starts from 1.
          first_word: A string that represent the first word of the bth paragraph.

        Returns:
          A string representing the instruction description.
        """
        self._num_paragraphs = num_paragraphs
        if self._num_paragraphs is None or self._num_paragraphs < 0:
            self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

        self._nth_paragraph = nth_paragraph
        if (
            self._nth_paragraph is None
            or self._nth_paragraph <= 0
            or self._nth_paragraph > self._num_paragraphs
        ):
            self._nth_paragraph = random.randint(1, self._num_paragraphs + 1)

        self._nth_paragraph = int(self._nth_paragraph)
        self._first_word = first_word
        if self._first_word is None:
            self._first_word = instructions_util.generate_keywords(num_keywords=1)[0]
        self._first_word = self._first_word.lower()

        self._description_pattern = (
            "সেখানে {num_paragraphs}টি অনুচ্ছেদ থাকতে হবে। "
            + "অনুচ্ছেদ এবং শুধুমাত্র অনুচ্ছেদগুলি একে অপরের থেকে দুটি নতুন লাইন দ্বারা পৃথক করা হয় যেমনটি পাইথনে '\\n\\n' ছিল। "
            + "{nth_paragraph}তম অনুচ্ছেদটি '{first_word}' শব্দ দিয়ে শুরু হওয়া উচিত।"
        )

        return self._description_pattern.format(
            num_paragraphs=self._num_paragraphs,
            nth_paragraph=self._nth_paragraph,
            first_word=self._first_word,
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "num_paragraphs": self._num_paragraphs,
            "nth_paragraph": self._nth_paragraph,
            "first_word": self._first_word,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_paragraphs", "nth_paragraph", "first_word"]

    def check_following(self, value):
        """Checks for required number of paragraphs and correct first word.

        Args:
          value: a string representing the response. The response may contain
            paragraphs that are separated by two new lines and the first word of
            the nth paragraph will have to match a specified word.

        Returns:
          True if the number of paragraphs is the same as required and the first
          word of the specified paragraph is the same as required. Otherwise, false.
        """

        paragraphs = re.split(r"\n\n", value)
        num_paragraphs = len(paragraphs)

        for paragraph in paragraphs:
            if not paragraph.strip():
                num_paragraphs -= 1

        # check that index doesn't go out of bounds
        if self._nth_paragraph <= num_paragraphs:
            paragraph = paragraphs[self._nth_paragraph - 1].strip()
            if not paragraph:
                return False
        else:
            return False

        first_word = ""
        punctuation = {".", ",", "?", "!", "'", '"', "*"}

        # get first word and remove punctuation
        word = paragraph.split()[0].strip()
        # TODO(jeffrey): make more complex?
        word = word.lstrip("'")
        word = word.lstrip('"')

        for letter in word:
            if letter in punctuation:
                break
            first_word += letter.lower()

        return first_word == self._first_word


# TODO(jeffrey) add relation - at least/at most?
class KeySentenceChecker(Instruction):
    """Check the existence of certain key sentences."""

    def build_description(self, key_sentences=None, num_sentences=None):
        """Build the instruction description.

        Args:
          key_sentences: A sequences of strings representing the key sentences that
            are expected in the response.
          num_sentences: The number of key sentences that are expected to be seen in
            the response.

        Returns:
          A string representing the instruction description.
        """

        if not key_sentences:
            # TODO(jeffrey) make a generate sentences function? wonderwords package
            self._key_sentences = set(["আপাতত, এটা ঠিক আছে।"])
        else:
            self._key_sentences = key_sentences

        if not num_sentences:
            self._num_sentences = random.randint(1, len(self._key_sentences))
        else:
            self._num_sentences = num_sentences

        self._description_pattern = (
            "নিম্নলিখিত বাক্যগুলি থেকে {num_sentences}টি অন্তর্ভুক্ত করুন {key_sentences}"
        )

        return self._description_pattern.format(
            num_sentences=self._num_sentences, key_sentences=self._key_sentences
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "num_sentences": self._num_sentences,
            "key_sentences": list(self._key_sentences),
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_sentences", "key_sentences"]

    def check_following(self, value):
        """Checks if the response contains the expected key sentences."""
        count = 0
        sentences = instructions_util.split_into_sentences(value)
        for sentence in self._key_sentences:
            if sentence in sentences:
                count += 1

        return count == self._num_sentences


class ForbiddenWords(Instruction):
    """Checks that specified words are not used in response."""

    def build_description(self, forbidden_words=None):
        """Build the instruction description.

        Args:
          forbidden_words: A sequences of strings representing words that are not
            allowed in the response.

        Returns:
          A string representing the instruction description.
        """

        if not forbidden_words:
            self._forbidden_words = instructions_util.generate_keywords(
                num_keywords=_NUM_KEYWORDS
            )
        else:
            self._forbidden_words = list(set(forbidden_words))
        self._forbidden_words = sorted(self._forbidden_words)
        self._description_pattern = (
            "প্রতিক্রিয়াতে কীওয়ার্ড {forbidden_words} অন্তর্ভুক্ত করবেন না।"
        )

        return self._description_pattern.format(forbidden_words=self._forbidden_words)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"forbidden_words": self._forbidden_words}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["forbidden_words"]

    def check_following(self, value):
        """Check if the response does not contain the expected keywords."""
        for word in self._forbidden_words:
            if re.search(r"\b" + word + r"\b", value, flags=re.IGNORECASE):
                return False
        return True


class RephraseParagraph(Instruction):
    """Checks that the paragraph is rephrased."""

    def build_description(self, *, original_paragraph, low, high):
        """Builds the instruction description.

        Args:
          original_paragraph: A string presenting the original paragraph. The
            rephrases response should have between low-high words in common.
          low: An integer presenting the lower bound of similar words.
          high: An integer representing the upper bound of similar words.

        Returns:
          A string representing the instruction description.
        """
        # TODO(jeffrey) make more encompassing
        self._original_paragraph = original_paragraph
        self._low = low
        self._high = high

        self._description = (
            "নিম্নলিখিত অনুচ্ছেদটি পুনর্লিখন করুন: "
            + "{original_paragraph}\nআপনার প্রতিক্রিয়ায় "
            + "{low} এবং {high}-এর মধ্যে একই রকম শব্দ থাকতে হবে। "
            + "শব্দগুলি তখনই এক হয় যখন এবং শুধুমাত্র যখন সমস্ত "
            + "অক্ষর, কেস উপেক্ষা করে, একই হয়। উদাহরণস্বরূপ, "
            + "'run' হল 'Run'-এর সমান কিন্তু 'ran'-এর থেকে ভিন্ন।"
        )

        return self._description.format(
            original_paragraph=original_paragraph, low=self._low, high=self._high
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "original_paragraph": self._original_paragraph,
            "low": self._low,
            "high": self._high,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["original_paragraph", "low", "high"]

    def check_following(self, value):
        val_words = re.findall(r"\w+", value.lower())
        original_words = re.findall(r"\w+", self._original_paragraph.lower())
        similar_words = 0

        dict_val = collections.Counter(val_words)
        dict_original = collections.Counter(original_words)

        for word in dict_original:
            similar_words += min(dict_original[word], dict_val[word])

        return similar_words >= self._low and similar_words <= self._high


class TwoResponsesChecker(Instruction):
    """Check that two responses were given."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "দুটি ভিন্ন প্রতিক্রিয়া দিন। প্রতিক্রিয়া এবং শুধুমাত্র প্রতিক্রিয়াগুলি "
            "6টি তারকা চিহ্ন দ্বারা পৃথক করা উচিত: ******।"
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response has two different answers.

        Args:
          value: A string representing the response.

        Returns:
          True if two responses are detected and false otherwise.
        """
        valid_responses = list()
        responses = value.split("******")
        for index, response in enumerate(responses):
            if not response.strip():
                if index != 0 and index != len(responses) - 1:
                    return False
            else:
                valid_responses.append(response)
        return (
            len(valid_responses) == 2
            and valid_responses[0].strip() != valid_responses[1].strip()
        )


class RepeatPromptThenAnswer(Instruction):
    """Checks that Prompt is first repeated then answered."""

    def build_description(self, *, prompt_to_repeat=None):
        """Build the instruction description.

        Args:
          prompt_to_repeat: The prompt that is meant to be repeated.

        Returns:
          A string representing the instruction description.
        """
        if not prompt_to_repeat:
            raise ValueError("prompt_to_repeat must be set.")
        else:
            self._prompt_to_repeat = prompt_to_repeat
        self._description_pattern = (
            "প্রথমে কোনো পরিবর্তন ছাড়াই শব্দে শব্দে অনুরোধটি পুনরাবৃত্তি করুন, "
            "তারপর আপনার উত্তর দিন (1. অনুরোধ পুনরাবৃত্তি করার আগে কোনো শব্দ বা অক্ষর বলবেন না; "
            "2. যে অনুরোধটি আপনার পুনরাবৃত্তি করতে হবে তাতে এই বাক্যটি অন্তর্ভুক্ত নয়)"
        )
        return self._description_pattern

    def get_instruction_args(self):
        return {"prompt_to_repeat": self._prompt_to_repeat}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["prompt_to_repeat"]

    def check_following(self, value):
        if value.strip().lower().startswith(self._prompt_to_repeat.strip().lower()):
            return True
        return False


class EndChecker(Instruction):
    """Checks that the prompt ends with a given phrase."""

    def build_description(self, *, end_phrase=None):
        """Build the instruction description.

        Args:
          end_phrase: A string representing the phrase the response should end with.

        Returns:
          A string representing the instruction description.
        """
        self._end_phrase = (
            end_phrase.strip() if isinstance(end_phrase, str) else end_phrase
        )
        if self._end_phrase is None:
            self._end_phrase = random.choice(_ENDING_OPTIONS)
        self._description_pattern = (
            "আপনার প্রতিক্রিয়াটি এই সঠিক বাক্যাংশ '{ender}' দিয়ে শেষ করুন। "
            "এই বাক্যাংশের পরে অন্য কোনো শব্দ থাকা উচিত নয়।"
        )
        return self._description_pattern.format(ender=self._end_phrase)

    def get_instruction_args(self):
        return {"end_phrase": self._end_phrase}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["end_phrase"]

    def check_following(self, value):
        """Checks if the response ends with the expected phrase."""
        value = value.strip().strip('"').lower()
        self._end_phrase = self._end_phrase.strip().lower()
        return value.endswith(self._end_phrase)


class TitleChecker(Instruction):
    """Checks the response for a title."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "আপনার উত্তরে একটি শিরোনাম থাকতে হবে, যা ডাবল অ্যাঙ্গেল ব্র্যাকেটে মোড়ানো, "
            "যেমন <<আনন্দের কবিতা>>।"
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response contains a title."""
        pattern = r"<<[^\n]+>>"
        re_pattern = re.compile(pattern)
        titles = re.findall(re_pattern, value)

        for title in titles:
            if title.lstrip("<").rstrip(">").strip():
                return True
        return False


class LetterFrequencyChecker(Instruction):
    """Checks letter frequency."""

    def build_description(self, *, letter=None, let_frequency=None, let_relation=None):
        """Build the instruction description.

        Args:
          letter: A string representing a letter that is expected in the response.
          let_frequency: An integer specifying the number of times `keyword` is
            expected to appear in the response.
          let_relation: A string in (`less than`, `at least`), defining the
            relational operator for comparison. Two relational comparisons are
            supported for now; if 'less than', the actual number of
            occurrences < frequency; if 'at least', the actual number of
            occurrences >= frequency.

        Returns:
          A string representing the instruction description.
        """
        if (
            not letter
            or len(letter) > 1
            or ord(letter.lower()) < 97
            or ord(letter.lower()) > 122
        ):
            self._letter = random.choice(list(string.ascii_letters))
        else:
            self._letter = letter.strip()
        self._letter = self._letter.lower()

        self._frequency = let_frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, _LETTER_FREQUENCY)

        if let_relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif let_relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {let_relation} is given."
            )
        else:
            self._comparison_relation = let_relation

        self._description_pattern = (
            "আপনার প্রতিক্রিয়ায়, '{letter}' অক্ষরটি {let_relation} "
            "{let_frequency} বার উপস্থিত থাকতে হবে।"
        )

        return self._description_pattern.format(
            letter=self._letter,
            let_frequency=self._frequency,
            let_relation=self._comparison_relation,
        )

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return {
            "letter": self._letter,
            "let_frequency": self._frequency,
            "let_relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["letter", "let_frequency", "let_relation"]

    def check_following(self, value):
        """Checks that the response contains the letter at the right frequency."""
        value = value.lower()
        letters = collections.Counter(value)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return letters[self._letter] < self._frequency
        else:
            return letters[self._letter] >= self._frequency


class CapitalLettersEnglishChecker(Instruction):
    """Checks that the response is in english and is in all capital letters."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "আপনার সম্পূর্ণ প্রতিক্রিয়া ইংরেজিতে এবং সমস্ত বড় হাতের অক্ষরে হওয়া উচিত।"
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response is in English and in all capital letters."""
        assert isinstance(value, str)

        try:
            return value.isupper() and langdetect.detect(value) == "en"
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logging.error(
                "Unable to detect language for text %s due to %s", value, e
            )  # refex: disable=pytotw.037
            return True


class LowercaseLettersEnglishChecker(Instruction):
    """Checks that the response is in english and is in all lowercase letters."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "আপনার সম্পূর্ণ প্রতিক্রিয়া ইংরেজিতে এবং সমস্ত ছোট হাতের অক্ষরে হওয়া উচিত। "
            "কোনো বড় হাতের অক্ষরের অনুমতি নেই।"
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response is in English and in all lowercase letters."""
        assert isinstance(value, str)

        try:
            return value.islower() and langdetect.detect(value) == "en"
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logging.error(
                "Unable to detect language for text %s due to %s", value, e
            )  # refex: disable=pytotw.037
            return True


class CommaChecker(Instruction):
    """Checks the response for no commas."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "আপনার সম্পূর্ণ প্রতিক্রিয়ায়, কোনো কমা ব্যবহার করা থেকে বিরত থাকুন।"
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response does not contain commas."""
        return not re.search(r"\,", value)


class CapitalWordFrequencyChecker(Instruction):
    """Checks frequency of words with all capital letters."""

    def build_description(
        self,
        capital_frequency=None,
        capital_relation=None,
    ):
        """Build the instruction description.

        Args:
          capital_frequency: An integer that represents the number of words that
            should be in all capital letters.
          capital_relation: A string that is 'at least' or 'at most' that refers to
            the frequency.

        Returns:
          A string representing the instruction description.
        """
        self._frequency = capital_frequency
        if self._frequency is None:
            self._frequency = random.randint(1, _ALL_CAPITAL_WORD_FREQUENCY)

        self._comparison_relation = capital_relation
        if capital_relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif capital_relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {capital_relation} is given."
            )

        self._description_pattern = (
            "আপনার প্রতিক্রিয়ায়, সমস্ত বড় হাতের অক্ষর সহ শব্দগুলি {relation} {frequency} বার আসা উচিত।"
        )

        return self._description_pattern.format(
            frequency=self._frequency, relation=self._comparison_relation
        )

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return {
            "capital_frequency": self._frequency,
            "capital_relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["capital_frequency", "capital_relation"]

    def check_following(self, value):
        """Checks the frequency of words with all capital letters."""
        # Hyphenated words will count as one word
        words = en_word_tokenize(value)
        capital_words = [word for word in words if word.isupper()]

        capital_words = len(capital_words)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return capital_words < self._frequency
        else:
            return capital_words >= self._frequency


class QuotationChecker(Instruction):
    """Checks response is wrapped with double quotation marks."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "আপনার সম্পূর্ণ প্রতিক্রিয়াটি ডবল উদ্ধৃতি চিহ্নের মধ্যে রাখুন।"
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response is wrapped with double quotation marks."""
        value = value.strip()
        return len(value) > 1 and value[0] == '"' and value[-1] == '"'