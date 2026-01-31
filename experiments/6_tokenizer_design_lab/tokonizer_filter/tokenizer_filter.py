#!/usr/bin/env python3
"""
Tokenizer Filter Tool

Filter and adapt tokenizer vocabularies based on category priorities and constraints.
Creates a 128k-token subset optimized for:
- Indic languages (strong Devanagari support)
- Code optimization (Python, JS/TS, C/C++)
- JSON and tool calling
- English baseline
- Max token length <= 32

Usage:
    # Single tokenizer - remove filtered tokens only (keeps all remaining)
    python tokenizer_filter.py --input tokenizer_json_files/qwen_tokenizer.json --output filtered_tokenizer/qwen_filtered.json
    
    # Process ALL tokenizers at once (default: no size limit)
    python tokenizer_filter.py --input-dir tokenizer_json_files --output-dir filtered_tokenizer
    
    # Remove specific categories (parameter-driven)
    python tokenizer_filter.py --input-dir tokenizer_json_files --remove-categories east_asian middle_eastern european
    
    # Custom token length (parameter-driven, default is 31)
    python tokenizer_filter.py --input-dir tokenizer_json_files --max-token-length 24
    
    # Filter low-frequency English tokens (unusual patterns, jargon, archaic, long compounds)
    python tokenizer_filter.py --input-dir tokenizer_json_files --filter-low-freq-english --max-english-length 12
    
    # Enforce 128k size limit (optional)
    python tokenizer_filter.py --input-dir tokenizer_json_files --target-size 128000 --enforce-size
    
    # Combine: remove categories + filter English + custom length (default max is 31)
    python tokenizer_filter.py --input-dir tokenizer_json_files --remove-categories east_asian middle_eastern --filter-low-freq-english --max-token-length 30
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from datetime import datetime
import unicodedata

# Import common utilities
from utils import (
    LANGUAGE_RANGES
)


# Keep LANGUAGE_RANGES visible for backward compatibility
# (already imported from utils above)
class TokenizerFilter:
    """Filter and adapt tokenizer vocabularies based on priorities."""
    
    def __init__(self, config_path: str = "tokenizer_config.json", 
                 max_token_length: int = None, 
                 remove_categories: List[str] = None,
                 filter_low_freq_english: bool = False,
                 max_english_length: int = 12):
        """
        Initialize filter with configuration.
        
        Args:
            config_path: Path to config file
            max_token_length: Override max token length from config (e.g., 32, 24, 30)
            remove_categories: Override categories to remove from config (e.g., ['east_asian', 'middle_eastern'])
            filter_low_freq_english: Enable low-frequency English token filtering
            max_english_length: Max length for English tokens (default: 12, only used if filter_low_freq_english=True)
        """
        self.config = self._load_config(config_path)
        self.language_ranges = LANGUAGE_RANGES  # Use hardcoded language ranges
        self.category_priorities = self.config["category_priorities"]
        self.filtering_rules = self.config["filtering_rules"]
        self.stats = defaultdict(int)
        
        # Low-frequency English filtering settings
        self.filter_low_freq_english = filter_low_freq_english
        self.max_english_length = max_english_length
        
        # Apply parameter overrides
        if max_token_length is not None:
            self.filtering_rules["max_token_length"] = max_token_length
            print(f"  ⚙️  Override: max_token_length = {max_token_length}")
        
        if remove_categories is not None:
            # Clear existing remove categories and set new ones
            self.category_priorities["remove"] = {}
            for category in remove_categories:
                self.category_priorities["remove"][category] = {
                    "description": f"Removed via command-line parameter",
                    "reason": "User-specified removal"
                }
            print(f"  ⚙️  Override: remove_categories = {remove_categories}")
        
        if filter_low_freq_english:
            print(f"  ⚙️  Low-frequency English filtering enabled (max_english_length={max_english_length})")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
            print("Using default configuration...")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Return default configuration if config file not found."""
        return {
            "selection_criteria": {
                "target_vocab_size": 128000,
                "max_token_length": 31
            },
            "category_priorities": {
                "keep": {
                    "english": {"priority": 10},
                    "code": {"priority": 9},
                    "indic": {"priority": 8},
                    "special": {"priority": 7},
                    "numeric": {"priority": 6},
                    "symbols": {"priority": 5}
                },
                "remove": {
                    "east_asian": {},
                    "middle_eastern": {},
                    "european": {},
                    "multilingual": {},
                    "whitespace": {},
                    "other": {},
                    "long_tokens": {}
                }
            },
            "filtering_rules": {
                "max_token_length": 32,
                "preserve_special_tokens": True
            }
        }
    
    def check_language(self, char: str) -> Tuple[bool, str]:
        """Check if character belongs to a non-Latin script and return the script name."""
        code_point = ord(char)
        
        for language_name, ranges in self.language_ranges.items():
            for start, end in ranges:
                if start <= code_point <= end:
                    return True, language_name
        
        return False, ''
    
    def categorize_token(self, token_value: str, decoded_text: str) -> str:
        """Categorize a token based on its content and the config."""
        # Check token length first
        if len(decoded_text) > self.filtering_rules["max_token_length"]:
            return "long_tokens"
        
        if not decoded_text or not decoded_text.strip():
            return "whitespace"
        
        # Check for special tokens
        if (decoded_text.startswith('<') and decoded_text.endswith('>')) or \
           decoded_text in ['<s>', '</s>', '<pad>', '<unk>', '<|endoftext|>', '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']:
            return "special"
        
        # Count character types
        has_language = False
        has_english = False
        has_digit = False
        has_code = False
        detected_languages = []
        
        for char in decoded_text:
            code_point = ord(char)
            
            # Check if it's a language character
            is_lang, lang_name = self.check_language(char)
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
            if char in '{}[]()<>=+-*/:;,.!@#$%^&|\\`~\'"':
                has_code = True
        
        # Categorize based on content (priority order)
        if has_language:
            # If it contains multilingual characters
            if len(detected_languages) == 1:
                # Single language - categorize by language family
                lang = detected_languages[0]
                if any(indic in lang for indic in ['Devanagari', 'Bengali', 'Tamil', 'Telugu', 
                                                    'Kannada', 'Malayalam', 'Gujarati', 'Gurmukhi', 
                                                    'Odia', 'Sinhala']):
                    return "indic"
                elif any(asian in lang for asian in ['Chinese', 'Japanese', 'Korean']):
                    return "east_asian"
                elif any(me in lang for me in ['Arabic', 'Hebrew', 'Persian']):
                    return "middle_eastern"
                elif any(eu in lang for eu in ['Cyrillic', 'Greek']):
                    return "european"
                else:
                    return "multilingual"
            else:
                # Multiple languages
                return "multilingual"
        elif has_english and has_code:
            return "code"
        elif has_english:
            return "english"
        elif has_digit:
            return "numeric"
        elif has_code:
            return "symbols"
        else:
            return "other"
    
    def is_low_frequency_english(self, decoded_text: str, category: str) -> Tuple[bool, str]:
        """
        Check if an English token should be filtered out as low-frequency.
        
        Filters tokens with:
        1. Unusual character patterns (consonant clusters, rare vowel patterns)
        2. Very long compound words (>max_english_length chars)
        3. Archaic/obsolete patterns
        4. Domain-specific jargon patterns
        
        Args:
            decoded_text: The decoded token text
            category: The token category
        
        Returns:
            Tuple of (is_low_freq, reason)
        """
        # Only apply to English tokens
        if category not in ['english', 'code']:
            return False, ""
        
        # Strip whitespace and common markers
        text = decoded_text.strip()
        if not text:
            return False, ""
        
        # Get only alphabetic part (remove punctuation/numbers for analysis)
        alpha_text = ''.join(c for c in text if c.isalpha()).lower()
        if len(alpha_text) < 3:
            return False, ""  # Too short to analyze
        
        # Rule 1: Very long compound words (>max_english_length chars)
        if len(alpha_text) > self.max_english_length:
            return True, f"long_compound_word (len={len(alpha_text)})"
        
        # Rule 2: Unusual consonant clusters (4+ consonants in a row)
        consonants = 'bcdfghjklmnpqrstvwxyz'
        consonant_cluster = 0
        max_consonant_cluster = 0
        for char in alpha_text:
            if char in consonants:
                consonant_cluster += 1
                max_consonant_cluster = max(max_consonant_cluster, consonant_cluster)
            else:
                consonant_cluster = 0
        
        if max_consonant_cluster >= 5:
            return True, f"unusual_consonant_cluster (len={max_consonant_cluster})"
        
        # Rule 3: Unusual vowel patterns (4+ vowels in a row, excluding common patterns)
        vowels = 'aeiou'
        vowel_cluster = 0
        max_vowel_cluster = 0
        for char in alpha_text:
            if char in vowels:
                vowel_cluster += 1
                max_vowel_cluster = max(max_vowel_cluster, vowel_cluster)
            else:
                vowel_cluster = 0
        
        if max_vowel_cluster >= 4:
            # Exclude common patterns
            common_vowel_patterns = ['aeou', 'aeio', 'ueue', 'ieee']
            is_common = any(pattern in alpha_text for pattern in common_vowel_patterns)
            if not is_common:
                return True, f"unusual_vowel_cluster (len={max_vowel_cluster})"
        
        # Rule 4: Rare letter combinations (heuristic patterns)
        rare_patterns = [
            'xz', 'qx', 'zx', 'qz', 'xq', 'zq',  # Rare consonant pairs
            'yyy', 'xxx', 'zzz',  # Triple same letters (except common ones)
            'ww', 'vv', 'jj', 'kk',  # Double rare consonants
        ]
        
        for pattern in rare_patterns:
            if pattern in alpha_text:
                return True, f"rare_pattern ({pattern})"
        
        # Rule 5: Low vowel density (< 20% vowels in words > 6 chars)
        if len(alpha_text) > 6:
            vowel_count = sum(1 for c in alpha_text if c in vowels)
            vowel_ratio = vowel_count / len(alpha_text)
            if vowel_ratio < 0.2:
                return True, f"low_vowel_density ({vowel_ratio:.2f})"
        
        # Rule 6: Archaic/obsolete patterns (heuristic)
        archaic_patterns = [
            'ye', 'thee', 'thou', 'hath', 'doth', 'whence', 'whilst',
            'shalt', 'unto', 'thy', 'thine'
        ]
        
        # Only flag if it starts/ends with or is exactly these patterns
        for pattern in archaic_patterns:
            if alpha_text == pattern or alpha_text.startswith(pattern + ' ') or alpha_text.endswith(' ' + pattern):
                return True, f"archaic_pattern ({pattern})"
        
        # Rule 7: Domain jargon patterns (medical, legal, chemical suffixes)
        jargon_suffixes = [
            'ectomy', 'otomy', 'itis', 'osis', 'emia',  # Medical
            'oxazole', 'benzene', 'amine', 'azine',  # Chemical
            'aceae', 'idae', 'inae',  # Biological taxonomy
        ]
        
        for suffix in jargon_suffixes:
            if alpha_text.endswith(suffix) and len(alpha_text) > len(suffix) + 3:
                return True, f"domain_jargon ({suffix})"
        
        return False, ""
    
    def should_keep_token(self, category: str) -> bool:
        """Determine if a token should be kept based on its category."""
        # First check if explicitly marked for removal
        if category in self.category_priorities.get("remove", {}):
            return False
        # Then check if it's in the keep list
        return category in self.category_priorities.get("keep", {})
    
    def get_token_priority(self, category: str) -> int:
        """Get priority score for a token category."""
        if category in self.category_priorities["keep"]:
            return self.category_priorities["keep"][category]["priority"]
        return 0
    
    def load_tokenizer_vocab(self, json_path: Path) -> Dict[str, int]:
        """Load tokenizer vocabulary from JSON file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle nested structure
            if isinstance(data, dict) and 'model' in data:
                if isinstance(data['model'], dict) and 'vocab' in data['model']:
                    vocab = data['model']['vocab']
                else:
                    vocab = data
            else:
                vocab = data
            
            if not isinstance(vocab, dict):
                raise ValueError("Vocabulary is not a dictionary")
            
            return vocab
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise
    
    def decode_token(self, token: str, token_id: int) -> str:
        """Decode token with byte-level BPE handling."""
        # Build byte decoder
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        
        byte_decoder = {chr(c): bytes([b]) for c, b in zip(cs, bs)}
        
        text = token
        try:
            # Replace special markers
            text = text.replace('Ġ', ' ')  # GPT-style space marker
            text = text.replace('▁', ' ')  # SentencePiece space marker
            
            # Only remove ## if it's at the start (BERT-style continuation marker)
            if text.startswith('##'):
                text = text[2:]
            
            # Try to decode as byte-level BPE
            byte_string = bytearray()
            for char in text:
                if char in byte_decoder:
                    byte_string.extend(byte_decoder[char])
                else:
                    # Not a byte-level token, return as-is
                    return text
            
            # Decode UTF-8
            decoded = byte_string.decode('utf-8', errors='replace')
            return decoded
        except:
            return text
    
    def filter_tokenizer(self, vocab: Dict[str, int], target_size: int = None, enforce_size_limit: bool = False) -> Dict:
        """
        Filter tokenizer vocabulary based on filter criteria.
        
        Args:
            vocab: Input vocabulary dictionary
            target_size: Target vocabulary size (for reference only, unless enforce_size_limit=True)
            enforce_size_limit: If True, enforce target_size limit. If False, keep all non-filtered tokens.
        
        Returns:
            Dictionary with 'filtered_vocab', 'removed_tokens', 'stats'
        """
        print(f"\nFiltering tokenizer vocabulary...")
        print(f"  Original size: {len(vocab):,}")
        if target_size and enforce_size_limit:
            print(f"  Target size: {target_size:,} (enforced)")
        elif target_size:
            print(f"  Target size: {target_size:,} (reference only, will keep all non-filtered tokens)")
        else:
            print(f"  Mode: Remove filtered categories only (no size limit)")
        
        # Import VocabularyWrapper for proper decoding
        try:
            from generate_tokenizer_report import VocabularyWrapper
            vocab_wrapper = VocabularyWrapper("filter_temp", vocab)
            use_wrapper_decode = True
        except:
            vocab_wrapper = None
            use_wrapper_decode = False
        
        # Categorize all tokens
        categorized_tokens = defaultdict(list)
        removed_tokens = defaultdict(list)
        
        for token_value, token_id in vocab.items():
            # Decode token using VocabularyWrapper for consistency with report
            if use_wrapper_decode:
                try:
                    decoded = vocab_wrapper.decode([token_id])
                except:
                    decoded = self.decode_token(token_value, token_id)
            else:
                decoded = self.decode_token(token_value, token_id)
            
            # Categorize
            category = self.categorize_token(token_value, decoded)
            
            # Track statistics
            self.stats[f"category_{category}"] += 1
            
            # Check if token should be kept based on category
            should_keep = self.should_keep_token(category)
            
            # Apply low-frequency English filtering if enabled
            if should_keep and self.filter_low_freq_english:
                is_low_freq, reason = self.is_low_frequency_english(decoded, category)
                if is_low_freq:
                    should_keep = False
                    # Track as a special removal category
                    removed_tokens[f"low_freq_english_{reason}"].append({
                        "token_value": token_value,
                        "token_id": token_id,
                        "decoded": decoded,
                        "reason": f"Low-frequency English: {reason}"
                    })
                    self.stats["low_freq_english_removed"] += 1
                    continue
            
            if should_keep:
                priority = self.get_token_priority(category)
                categorized_tokens[category].append({
                    "token_value": token_value,
                    "token_id": token_id,
                    "decoded": decoded,
                    "priority": priority,
                    "length": len(decoded)
                })
            else:
                removed_tokens[category].append({
                    "token_value": token_value,
                    "token_id": token_id,
                    "decoded": decoded,
                    "reason": self.category_priorities["remove"].get(category, {}).get("description", "Low priority")
                })
        
        # Sort tokens by priority and collect all kept tokens
        all_kept_tokens = []
        for category, tokens in categorized_tokens.items():
            print(f"  {category}: {len(tokens):,} tokens")
            all_kept_tokens.extend(tokens)
        
        # Sort by priority (descending) and then by token_id (ascending for stability)
        all_kept_tokens.sort(key=lambda x: (-x["priority"], x["token_id"]))
        
        # Decide what to keep based on enforce_size_limit flag
        if enforce_size_limit and target_size and len(all_kept_tokens) > target_size:
            selected_tokens = all_kept_tokens[:target_size]
            overflow_tokens = all_kept_tokens[target_size:]
            
            print(f"\n  Selected: {len(selected_tokens):,} tokens (size limit enforced)")
            print(f"  Overflow (kept categories but exceeded limit): {len(overflow_tokens):,} tokens")
            
            # Add overflow to removed
            for token in overflow_tokens:
                removed_tokens["overflow"].append({
                    "token_value": token["token_value"],
                    "token_id": token["token_id"],
                    "decoded": token["decoded"],
                    "reason": "Exceeded target vocabulary size"
                })
        else:
            selected_tokens = all_kept_tokens
            if target_size and len(selected_tokens) > target_size:
                print(f"\n  Kept: {len(selected_tokens):,} tokens (no size limit enforced, exceeds target by {len(selected_tokens) - target_size:,})")
            else:
                print(f"\n  Kept: {len(selected_tokens):,} tokens")
        
        # Build filtered vocabulary
        filtered_vocab = {}
        for token in selected_tokens:
            filtered_vocab[token["token_value"]] = token["token_id"]
        
        # Print removal summary
        print(f"\n  Removed categories:")
        total_removed = 0
        for category, tokens in removed_tokens.items():
            print(f"    {category}: {len(tokens):,} tokens")
            total_removed += len(tokens)
        
        print(f"\n  Total removed: {total_removed:,} tokens")
        print(f"  Final vocab size: {len(filtered_vocab):,}")
        
        return {
            "filtered_vocab": filtered_vocab,
            "removed_tokens": removed_tokens,
            "stats": dict(self.stats),
            "metadata": {
                "original_size": len(vocab),
                "filtered_size": len(filtered_vocab),
                "target_size": target_size if target_size else "N/A",
                "size_limit_enforced": enforce_size_limit,
                "removed_count": total_removed,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def save_filtered_tokenizer(self, filtered_vocab: Dict[str, int], output_path: Path):
        """Save filtered tokenizer to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_vocab, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Filtered tokenizer saved to: {output_path}")
    
    def save_filtering_report(self, result: Dict, output_path: Path):
        """Save detailed filtering report."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a serializable report (exclude full vocab)
        report = {
            "metadata": result["metadata"],
            "stats": result["stats"],
            "removed_summary": {
                category: len(tokens) 
                for category, tokens in result["removed_tokens"].items()
            },
            "config": {
                "target_vocab_size": self.config["selection_criteria"]["target_vocab_size"],
                "max_token_length": self.filtering_rules["max_token_length"],
                "category_priorities": self.category_priorities
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Filtering report saved to: {output_path}")
    
    def save_removed_tokens_log(self, removed_tokens: Dict, output_path: Path):
        """Save detailed log of removed tokens."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(removed_tokens, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Removed tokens log saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter tokenizer vocabulary based on category priorities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter single tokenizer (default: keeps all non-filtered tokens)
  python tokenizer_filter.py --input tokenizer_json_files/qwen_tokenizer.json --output filtered_tokenizer/qwen_filtered.json
  
  # Filter ALL tokenizers at once (no size limit)
  python tokenizer_filter.py --input-dir tokenizer_json_files --output-dir filtered_tokenizer
  
  # Remove specific categories via parameters
  python tokenizer_filter.py --input-dir tokenizer_json_files --remove-categories east_asian middle_eastern european
  
  # Custom token length limit
  python tokenizer_filter.py --input-dir tokenizer_json_files --max-token-length 32
  
  # Filter low-frequency English tokens (removes unusual patterns, jargon, archaic terms, long compounds)
  python tokenizer_filter.py --input-dir tokenizer_json_files --filter-low-freq-english --max-english-length 12
  
  # Combine: remove categories + low-freq English filtering
  python tokenizer_filter.py --input-dir tokenizer_json_files --remove-categories east_asian middle_eastern --filter-low-freq-english
  
  # Remove categories AND set max length (no size limit)
  python tokenizer_filter.py --input-dir tokenizer_json_files --remove-categories east_asian middle_eastern --max-token-length 31
  
  # Enforce 128k size limit (optional)
  python tokenizer_filter.py --input-dir tokenizer_json_files --target-size 128000 --enforce-size --remove-categories east_asian
  
  # Full pipeline: all filters enabled
  python tokenizer_filter.py --input-dir tokenizer_json_files --max-token-length 31 --remove-categories east_asian middle_eastern european --filter-low-freq-english --max-english-length 10
        """
    )
    
    parser.add_argument("--input", "-i", help="Input tokenizer JSON file")
    parser.add_argument("--input-dir", help="Input directory with tokenizer JSON files (processes ALL at once)")
    parser.add_argument("--output", "-o", help="Output filtered tokenizer JSON file")
    parser.add_argument("--output-dir", default="filtered_tokenizer", help="Output directory (default: filtered_tokenizer/)")
    parser.add_argument("--config", "-c", default="tokenizer_config.json", help="Configuration file (default: tokenizer_config.json in current dir)")
    parser.add_argument("--target-size", "-t", type=int, help="Target vocabulary size (optional, for reference only unless --enforce-size is used)")
    parser.add_argument("--enforce-size", action="store_true", help="Enforce target size limit (by default, keeps all non-filtered tokens)")
    parser.add_argument("--max-token-length", type=int, help="Maximum token length (overrides config, default: 31, e.g., 31, 24, 30)")
    parser.add_argument("--remove-categories", nargs="+", 
                       help="Categories to remove (overrides config). Options: east_asian, middle_eastern, european, multilingual, whitespace, other, long_tokens")
    parser.add_argument("--filter-low-freq-english", action="store_true", 
                       help="Enable low-frequency English token filtering (removes unusual patterns, jargon, archaic terms, long compounds)")
    parser.add_argument("--max-english-length", type=int, default=12,
                       help="Maximum length for English tokens when --filter-low-freq-english is enabled (default: 12)")
    parser.add_argument("--save-removed", action="store_true", help="Save detailed log of removed tokens")
    parser.add_argument("--parallel", action="store_true", help="Process tokenizers in parallel (faster for multiple files)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input and not args.input_dir:
        parser.error("Either --input or --input-dir must be specified")
    
    print("=" * 80)
    print("TOKENIZER FILTER TOOL")
    print("=" * 80)
    
    # Show parameter overrides and mode
    print("\n⚙️  CONFIGURATION:")
    if args.max_token_length or args.remove_categories or args.target_size or args.enforce_size:
        if args.max_token_length:
            print(f"  Max Token Length: {args.max_token_length}")
        if args.remove_categories:
            print(f"  Remove Categories: {', '.join(args.remove_categories)}")
        if args.target_size:
            if args.enforce_size:
                print(f"  Target Size: {args.target_size:,} (ENFORCED - will limit to this size)")
            else:
                print(f"  Target Size: {args.target_size:,} (reference only)")
        if not args.enforce_size:
            print(f"  Mode: Keep all non-filtered tokens (no size limit)")
    else:
        print(f"  Mode: Remove filtered categories only, keep all remaining tokens")
    
    # Initialize filter with parameter overrides
    filter_tool = TokenizerFilter(
        config_path=args.config,
        max_token_length=args.max_token_length,
        remove_categories=args.remove_categories,
        filter_low_freq_english=args.filter_low_freq_english,
        max_english_length=args.max_english_length
    )
    
    # Process single file
    if args.input:
        input_path = Path(args.input)
        
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return 1
        
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path(args.output_dir) / f"{input_path.stem}_filtered.json"
        
        print(f"\nProcessing: {input_path.name}")
        print("-" * 80)
        
        # Load vocab
        vocab = filter_tool.load_tokenizer_vocab(input_path)
        
        # Filter
        result = filter_tool.filter_tokenizer(vocab, target_size=args.target_size, enforce_size_limit=args.enforce_size)
        
        # Save filtered tokenizer
        filter_tool.save_filtered_tokenizer(result["filtered_vocab"], output_path)
        
        # Save report
        report_path = Path(args.output_dir) / "tokenizer_results" / f"{input_path.stem}_filtering_report.json"
        filter_tool.save_filtering_report(result, report_path)
        
        # Save removed tokens if requested
        if args.save_removed:
            removed_path = Path(args.output_dir) / "tokenizer_results" / f"{input_path.stem}_removed_tokens.json"
            filter_tool.save_removed_tokens_log(result["removed_tokens"], removed_path)
    
    # Process directory (ALL tokenizers at once)
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}")
            return 1
        
        # Find all JSON files
        json_files = list(input_dir.glob("*.json"))
        
        if not json_files:
            print(f"Error: No JSON files found in {input_dir}")
            return 1
        
        print(f"\n📂 Found {len(json_files)} tokenizer files to process")
        print("-" * 80)
        for f in json_files:
            print(f"  • {f.name}")
        print("-" * 80)
        
        # Track results
        successful = []
        failed = []
        
        # Process each tokenizer
        for idx, json_file in enumerate(json_files, 1):
            print(f"\n[{idx}/{len(json_files)}] Processing: {json_file.name}")
            print("-" * 80)
            
            try:
                # Load vocab
                vocab = filter_tool.load_tokenizer_vocab(json_file)
                
                # Filter
                result = filter_tool.filter_tokenizer(vocab, target_size=args.target_size, enforce_size_limit=args.enforce_size)
                
                # Save filtered tokenizer
                output_path = Path(args.output_dir) / f"{json_file.stem}_filtered.json"
                filter_tool.save_filtered_tokenizer(result["filtered_vocab"], output_path)
                
                # Save report
                report_path = Path(args.output_dir) / "tokenizer_results" / f"{json_file.stem}_filtering_report.json"
                filter_tool.save_filtering_report(result, report_path)
                
                # Save removed tokens if requested
                if args.save_removed:
                    removed_path = Path(args.output_dir) / "tokenizer_results" / f"{json_file.stem}_removed_tokens.json"
                    filter_tool.save_removed_tokens_log(result["removed_tokens"], removed_path)
                
                successful.append(json_file.name)
                print(f"  ✅ Successfully processed {json_file.name}")
                
            except Exception as e:
                print(f"  ❌ Error processing {json_file.name}: {e}")
                failed.append((json_file.name, str(e)))
                continue
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 BATCH PROCESSING SUMMARY")
        print("=" * 80)
        print(f"\n✅ Successful: {len(successful)}/{len(json_files)}")
        for name in successful:
            print(f"  • {name}")
        
        if failed:
            print(f"\n❌ Failed: {len(failed)}/{len(json_files)}")
            for name, error in failed:
                print(f"  • {name}: {error}")
        
        print(f"\n📁 Output directory: {Path(args.output_dir).absolute()}")
        print(f"📄 Reports directory: {(Path(args.output_dir) / 'tokenizer_results').absolute()}")
    
    print("\n" + "=" * 80)
    print("✅ FILTERING COMPLETE")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
