#!/usr/bin/env python3
"""
Tokenizer Tools - Common Utilities

This module contains all shared code used across tokenizer tools:
- Language ranges (27 languages/scripts)
- Character detection functions
- Token categorization logic
- Vocabulary wrapper class
- Byte-level BPE decoding

All tokenizer tools import from this module to ensure consistency.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

# ============================================================================
# LANGUAGE RANGES - Hardcoded for consistency across all tools
# ============================================================================

LANGUAGE_RANGES = {
    # Indic Scripts
    "Devanagari (Hindi/Sanskrit/Marathi/Nepali)": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "Bengali/Assamese": [(0x0980, 0x09FF)],
    "Tamil": [(0x0B80, 0x0BFF)],
    "Telugu": [(0x0C00, 0x0C7F)],
    "Kannada": [(0x0C80, 0x0CFF)],
    "Malayalam": [(0x0D00, 0x0D7F)],
    "Gujarati": [(0x0A80, 0x0AFF)],
    "Gurmukhi (Punjabi)": [(0x0A00, 0x0A7F)],
    "Odia/Oriya": [(0x0B00, 0x0B7F)],
    "Sinhala": [(0x0D80, 0x0DFF)],
    # East Asian Scripts
    "Chinese (CJK Unified)": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF)],
    "Japanese Hiragana": [(0x3040, 0x309F)],
    "Japanese Katakana": [(0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    "Korean Hangul": [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)],
    # Middle Eastern Scripts
    "Arabic": [
        (0x0600, 0x06FF),
        (0x0750, 0x077F),
        (0x08A0, 0x08FF),
        (0xFB50, 0xFDFF),
        (0xFE70, 0xFEFF),
    ],
    "Hebrew": [(0x0590, 0x05FF), (0xFB1D, 0xFB4F)],
    "Persian (Farsi)": [(0x0600, 0x06FF)],  # Uses Arabic script with additions
    # European Scripts
    "Cyrillic (Russian/Ukrainian/Bulgarian)": [(0x0400, 0x04FF), (0x0500, 0x052F)],
    "Greek": [(0x0370, 0x03FF), (0x1F00, 0x1FFF)],
    # Southeast Asian Scripts
    "Thai": [(0x0E00, 0x0E7F)],
    "Lao": [(0x0E80, 0x0EFF)],
    "Myanmar (Burmese)": [(0x1000, 0x109F)],
    "Khmer (Cambodian)": [(0x1780, 0x17FF)],
    # Other Scripts
    "Armenian": [(0x0530, 0x058F)],
    "Georgian": [(0x10A0, 0x10FF)],
    "Ethiopic": [(0x1200, 0x137F)],
}


# ============================================================================
# CHARACTER DETECTION FUNCTIONS
# ============================================================================


def check_language(char: str) -> Tuple[bool, str]:
    """
    Check if a character belongs to a non-Latin script.

    Args:
        char: Single character to check

    Returns:
        Tuple of (is_language, language_name)
        - is_language: True if character belongs to any language in LANGUAGE_RANGES
        - language_name: Name of the language/script (empty string if not found)
    """
    code_point = ord(char)

    for language_name, ranges in LANGUAGE_RANGES.items():
        for start, end in ranges:
            if start <= code_point <= end:
                return True, language_name

    return False, ""


def is_english_char(char: str) -> bool:
    """
    Check if a character is English (a-z, A-Z).

    Args:
        char: Single character to check

    Returns:
        True if character is ASCII alphabetic (a-z, A-Z)
    """
    return char.isalpha() and ord(char) < 128


# ============================================================================
# TOKEN CATEGORIZATION
# ============================================================================


def categorize_token(decoded_token: str, check_language_func=None) -> str:
    """
    Categorize a token based on its content.

    Args:
        decoded_token: The decoded token text
        check_language_func: Function to check character language (defaults to module's check_language)

    Returns:
        Category string: special, english, code, indic, east_asian, middle_eastern,
                        european, multilingual, numeric, symbols, whitespace, other
    """
    if check_language_func is None:
        check_language_func = check_language

    if not decoded_token or not decoded_token.strip():
        return "whitespace"

    # Check for special tokens
    if (
        decoded_token.startswith("<") and decoded_token.endswith(">")
    ) or decoded_token in [
        "<s>",
        "</s>",
        "<pad>",
        "<unk>",
        "<|endoftext|>",
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
    ]:
        return "special"

    # Count character types
    has_language = False
    has_english = False
    has_digit = False
    has_code = False
    detected_languages = []

    for char in decoded_token:
        code_point = ord(char)

        # Check if it's a language character
        is_lang, lang_name = check_language_func(char)
        if is_lang:
            has_language = True
            if lang_name not in detected_languages:
                detected_languages.append(lang_name)

        # Check English
        if char.isalpha() and code_point < 128:
            has_english = True

        # Check digits
        if char.isdigit():
            has_digit = True

        # Check code-like characters
        if char in "{}[]()<>=+-*/:;,.!@#$%^&|\\`~'\"":
            has_code = True

    # Categorize based on content (priority order)
    if has_language:
        # If it contains multilingual characters
        if len(detected_languages) == 1:
            # Single language - categorize by language family
            lang = detected_languages[0]
            if any(
                indic in lang
                for indic in [
                    "Devanagari",
                    "Bengali",
                    "Tamil",
                    "Telugu",
                    "Kannada",
                    "Malayalam",
                    "Gujarati",
                    "Gurmukhi",
                    "Odia",
                    "Sinhala",
                ]
            ):
                return "indic"
            elif any(asian in lang for asian in ["Chinese", "Japanese", "Korean"]):
                return "east_asian"
            elif any(me in lang for me in ["Arabic", "Hebrew", "Persian"]):
                return "middle_eastern"
            elif any(eu in lang for eu in ["Cyrillic", "Greek"]):
                return "european"
            else:
                return "multilingual"
        else:
            # Multiple languages
            return "multilingual"

    if has_code and has_english:
        return "code"
    elif has_english:
        return "english"
    elif has_digit:
        return "numeric"
    elif has_code:
        return "symbols"
    else:
        return "other"


# ============================================================================
# VOCABULARY WRAPPER CLASS
# ============================================================================


class VocabularyWrapper:
    """
    Wrapper for tokenizer vocabulary with proper byte-level BPE decoding.

    Handles multiple tokenizer formats and provides consistent decoding interface.
    """

    def __init__(self, name: str, vocab: Dict[str, int]):
        """
        Initialize vocabulary wrapper.

        Args:
            name: Name of the tokenizer
            vocab: Dictionary mapping token strings to token IDs
        """
        self.name = name
        self._vocab = vocab
        self.vocab_size = len(vocab)
        # Create reverse mapping (id -> token) for decoding
        self._id_to_token = {v: k for k, v in vocab.items()}

        # Create byte decoder for handling byte-level BPE tokens
        self._byte_decoder = self._build_byte_decoder()

    def _build_byte_decoder(self) -> Dict[str, int]:
        """
        Build byte decoder for byte-level BPE tokens.

        Many modern tokenizers (GPT-2, GPT-3, etc.) use byte-level encoding
        where each token is a sequence of bytes that needs to be decoded as UTF-8.

        Returns:
            Dictionary mapping character to byte value
        """
        # Standard byte decoder used by GPT-2 and similar tokenizers
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1

        byte_decoder = {chr(c): bytes([b]) for c, b in zip(cs, bs)}
        return byte_decoder

    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary dictionary (token -> ID)."""
        return self._vocab

    def decode(self, token_ids: List[int], skip_special_tokens=False) -> str:
        """
        Decode token IDs to text with proper byte-level BPE handling.

        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens (not implemented)

        Returns:
            Decoded text string
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self._id_to_token:
                tokens.append(self._id_to_token[token_id])

        if not tokens:
            return ""

        # Join tokens
        text = "".join(tokens)

        # Try byte-level decoding
        try:
            # Replace special markers
            text = text.replace("Ġ", " ")  # GPT-style space marker
            text = text.replace("▁", " ")  # SentencePiece space marker

            # Only remove ## if it's at the start (BERT-style continuation marker)
            if text.startswith("##"):
                text = text[2:]

            # Try to decode as byte-level BPE
            byte_string = bytearray()
            for char in text:
                if char in self._byte_decoder:
                    byte_string.extend(self._byte_decoder[char])
                else:
                    # Not a byte-level token, return as-is
                    return text

            # Decode UTF-8
            decoded = byte_string.decode("utf-8", errors="replace")
            return decoded
        except Exception:
            # If decoding fails, return original
            return text


# ============================================================================
# BYTE-LEVEL TOKEN DECODING
# ============================================================================


def decode_byte_token(token: str) -> str:
    """
    Decode byte-level BPE token to actual text.

    Simple decoder for individual tokens (not using full vocabulary).
    For more robust decoding, use VocabularyWrapper.decode().

    Args:
        token: Raw token string

    Returns:
        Decoded token text
    """
    try:
        # Remove special markers
        clean = token.replace("▁", "").replace("Ġ", " ").replace("##", "")
        clean = clean.replace("<|", "").replace("|>", "").strip()

        # For byte-level BPE tokenizers (like GPT-2, many modern tokenizers)
        # The tokens are already UTF-8 characters that may represent bytes
        # Try to decode if it looks like byte encoding

        # Method 1: Direct interpretation (works for many tokenizers)
        return clean

    except Exception:
        return token


# ============================================================================
# JSON LOADING UTILITIES
# ============================================================================


def load_vocab_from_json(data: dict) -> dict:
    """
    Extract vocabulary from JSON data in multiple formats.

    Supports:
    - Nested format: {"model": {"vocab": {...}}}
    - Semi-nested: {"vocab": {...}}
    - Flat format: {"token": id, ...}

    Args:
        data: Parsed JSON data

    Returns:
        Dictionary mapping tokens to IDs

    Raises:
        ValueError: If vocabulary cannot be extracted
    """
    if not isinstance(data, dict):
        raise ValueError("JSON data must be a dictionary")

    # Check for nested format: {"model": {"vocab": {...}}}
    if "model" in data and isinstance(data["model"], dict) and "vocab" in data["model"]:
        return data["model"]["vocab"]

    # Check for semi-nested format: {"vocab": {...}}
    # Must verify it's actually a dict (not a token named "vocab")
    if "vocab" in data and isinstance(data["vocab"], dict):
        return data["vocab"]

    # Check for flat format: {"token": id, ...}
    # Verify that values are integers (token IDs)
    sample_values = list(data.values())[:10]
    if sample_values and all(isinstance(v, int) for v in sample_values):
        return data

    raise ValueError("Could not extract vocabulary from JSON data")


# ============================================================================
# LANGUAGE FAMILY UTILITIES
# ============================================================================


def get_language_family(language_name: str) -> str:
    """
    Get the family/category for a language name.

    Args:
        language_name: Language name from LANGUAGE_RANGES

    Returns:
        Family name: 'indic', 'east_asian', 'middle_eastern', 'european', 'southeast_asian', 'other'
    """
    if any(
        indic in language_name
        for indic in [
            "Devanagari",
            "Bengali",
            "Tamil",
            "Telugu",
            "Kannada",
            "Malayalam",
            "Gujarati",
            "Gurmukhi",
            "Odia",
            "Sinhala",
        ]
    ):
        return "indic"
    elif any(asian in language_name for asian in ["Chinese", "Japanese", "Korean"]):
        return "east_asian"
    elif any(me in language_name for me in ["Arabic", "Hebrew", "Persian"]):
        return "middle_eastern"
    elif any(eu in language_name for eu in ["Cyrillic", "Greek"]):
        return "european"
    elif any(sea in language_name for sea in ["Thai", "Lao", "Myanmar", "Khmer"]):
        return "southeast_asian"
    elif any(other in language_name for other in ["Armenian", "Georgian", "Ethiopic"]):
        return "other"
    else:
        return "other"


def get_indic_languages() -> List[str]:
    """Get list of all Indic language names."""
    return [
        name for name in LANGUAGE_RANGES.keys() if get_language_family(name) == "indic"
    ]


def get_all_language_families() -> Dict[str, List[str]]:
    """
    Get all languages organized by family.

    Returns:
        Dictionary mapping family names to lists of language names
    """
    families = defaultdict(list)
    for lang_name in LANGUAGE_RANGES.keys():
        family = get_language_family(lang_name)
        families[family].append(lang_name)
    return dict(families)


# ============================================================================
# MODULE INFO
# ============================================================================

__version__ = "1.0.0"
__all__ = [
    "LANGUAGE_RANGES",
    "check_language",
    "is_english_char",
    "categorize_token",
    "VocabularyWrapper",
    "decode_byte_token",
    "load_vocab_from_json",
    "get_language_family",
    "get_indic_languages",
    "get_all_language_families",
]


if __name__ == "__main__":
    # Test the module
    print("Tokenizer Utils Module")
    print("=" * 60)
    print(f"Version: {__version__}")
    print(f"Languages supported: {len(LANGUAGE_RANGES)}")
    print("\nLanguage families:")
    for family, langs in get_all_language_families().items():
        print(f"  {family}: {len(langs)} languages")
    print("\n✅ Module loaded successfully")
