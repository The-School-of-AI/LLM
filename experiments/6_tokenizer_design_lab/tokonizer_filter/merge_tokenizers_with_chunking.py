#!/usr/bin/env python3
"""
Tokenizer Merger with Chunking Support - Enhanced Version

Integrates sophisticated merge logic from merge_tokenizers_fixed.py with chunking support.

Features:
- Loads tokens from filtered tokenizer JSON files
- Uses analysis report for statistics
- Loads long tokens (>= 32 length) from CSV
- Chunks repeating patterns > 16 chars into 8 & 16 character chunks
- Filters placeholders and unused tokens
- Category-based prioritization and ID block assignment
- Structural token normalization

Usage:
    python merge_tokenizers_with_chunking.py \
        --input-dir filtered_tokenizer \
        --output merged_tokenizer_128k.json \
        --target-size 128000
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
from datetime import datetime
import re

# Progress bar (optional)
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = lambda x, **kwargs: x

# Import common utilities
from utils import (
    LANGUAGE_RANGES,
    check_language as utils_check_language,
    get_language_family,
    load_vocab_from_json
)


# Keep LANGUAGE_RANGES visible for backward compatibility
class SpecialTokensManager:
    """Manages special tokens from configuration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Load special tokens configuration."""
        self.config = {}
        self.special_tokens = {}  # token -> id
        self.reserved_ids = set()  # IDs reserved for future use
        
        if config_path and config_path.exists():
            self._load_config(config_path)
        else:
            self._load_default_config()
    
    def _load_config(self, config_path: Path):
        """Load configuration from JSON file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.config = data.get('special_tokens_scheme', {})
            self._parse_config()
    
    def _load_default_config(self):
        """Load minimal default configuration."""
        # Minimal base tokens
        self.special_tokens = {
            '</s>': 0,
            '<pad>': 1,
            '<s>': 2,
            '<unk>': 3,
            '<|endoftext|>': 4,
            '<|im_end|>': 5,
            '<|im_start|>': 6,
            '[/INST]': 7,
            '[INST]': 8,
            '[TOOL_CALLS]': 9,
        }
        # Reserve 100-299 for future use
        self.reserved_ids = set(range(100, 300))
    
    def _parse_config(self):
        """Parse configuration and build token mappings."""
        # Process each category
        for category_name, category_data in self.config.items():
            if not isinstance(category_data, dict):
                continue
            
            if category_name == 'future_governance_reserved':
                # Mark these IDs as reserved
                id_range = category_data.get('id_range', [100, 299])
                self.reserved_ids = set(range(id_range[0], id_range[1] + 1))
                continue
            
            if category_name == 'regular_vocabulary_start':
                continue
            
            # Process token list
            tokens = category_data.get('tokens', [])
            if isinstance(tokens, list):
                for token_def in tokens:
                    if isinstance(token_def, dict):
                        token = token_def.get('token')
                        token_id = token_def.get('id')
                        if token and token_id is not None:
                            self.special_tokens[token] = token_id
    
    def get_all_special_tokens(self) -> Dict[str, int]:
        """Return all special tokens."""
        return self.special_tokens.copy()
    
    def is_reserved_id(self, token_id: int) -> bool:
        """Check if ID is reserved for future use."""
        return token_id in self.reserved_ids
    
    def get_regular_vocab_start_id(self) -> int:
        """Get the starting ID for regular vocabulary."""
        # Find the max used ID
        max_special_id = max(self.special_tokens.values()) if self.special_tokens else 0
        max_reserved_id = max(self.reserved_ids) if self.reserved_ids else 0
        return max(max_special_id, max_reserved_id) + 1
    
    def get_summary(self) -> Dict:
        """Get summary of special token allocation."""
        return {
            'total_special_tokens': len(self.special_tokens),
            'reserved_ids_count': len(self.reserved_ids),
            'regular_vocab_start': self.get_regular_vocab_start_id(),
            'categories': self._get_category_summary()
        }
    
    def _get_category_summary(self) -> Dict:
        """Get count per category."""
        summary = {}
        for category_name, category_data in self.config.items():
            if isinstance(category_data, dict) and 'tokens' in category_data:
                tokens = category_data.get('tokens', [])
                if isinstance(tokens, list):
                    summary[category_name] = len(tokens)
        return summary


class TokenNormalizer:
    """Normalize tokens from different tokenizer formats to canonical form."""
    
    def __init__(self):
        self._byte_decoder = self._build_byte_decoder()
    
    def _build_byte_decoder(self) -> Dict[str, int]:
        """Build byte decoder for byte-level BPE tokens."""
        bs = list(range(ord("!"), ord("~") + 1)) + \
             list(range(ord("¡"), ord("¬") + 1)) + \
             list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return dict(zip([chr(c) for c in cs], bs))
    
    def normalize(self, token: str) -> str:
        """Normalize token to canonical form for comparison."""
        # Remove special markers
        normalized = token.replace('Ġ', ' ').replace('▁', ' ')
        
        # Remove ## only from start (BERT continuation)
        if normalized.startswith('##'):
            normalized = normalized[2:]
        
        # Normalize structural tokens (HTML tags, types) to lowercase
        normalized = self._normalize_structural(normalized)
        
        # Try to decode as byte-level BPE
        try:
            byte_array = bytearray()
            for char in normalized:
                if char in self._byte_decoder:
                    byte_array.append(self._byte_decoder[char])
                else:
                    return normalized
            
            decoded = byte_array.decode('utf-8', errors='replace')
            return decoded
        except:
            return normalized
    
    def _normalize_structural(self, token: str) -> str:
        """Normalize case for structural tokens to reduce duplicates."""
        # HTML tags: <tag> -> lowercase
        html_tag_pattern = re.compile(r'^<([a-zA-Z][a-zA-Z0-9]*)>$')
        match = html_tag_pattern.match(token)
        if match:
            return f"<{match.group(1).lower()}>"
        
        # Type annotations: <type> -> lowercase
        type_pattern = re.compile(r'^<([a-zA-Z]+)>$')
        match = type_pattern.match(token)
        if match:
            type_name = match.group(1).lower()
            if type_name in ['int', 'long', 'double', 'float', 'string', 'bool', 
                            'char', 'byte', 'short', 'any', 'void', 'object']:
                return f"<{type_name}>"
        
        return token


class PlaceholderFilter:
    """Filter out placeholder and unused tokens from source tokenizers."""
    
    def __init__(self):
        self.special_numbered_pattern = re.compile(r'^<SPECIAL_\d+>$', re.IGNORECASE)
        self.plhd_pattern = re.compile(r'PLHD', re.IGNORECASE)
        self.unused_pattern = re.compile(r'\bunused\b', re.IGNORECASE)
        self.never_used_pattern = re.compile(r'never[_\s-]used', re.IGNORECASE)
    
    def is_placeholder(self, token: str) -> Tuple[bool, str]:
        """Check if token is a placeholder/unused token."""
        if self.unused_pattern.search(token):
            return True, "[unused*] pattern"
        
        if self.never_used_pattern.search(token):
            return True, "never_used pattern"
        
        if self.plhd_pattern.search(token):
            return True, "PLHD placeholder"
        
        if self.special_numbered_pattern.match(token):
            return True, "numbered SPECIAL token"
        
        token_lower = token.lower()
        placeholder_indicators = ['placeholder', '_placeholder_', 'plchld', 'rsrvd', 'reserved_']
        
        for indicator in placeholder_indicators:
            if indicator in token_lower:
                return True, f"'{indicator}' indicator"
        
        return False, ""


class TokenCategorizer:
    """Categorize tokens using hardcoded language ranges."""
    
    def __init__(self, config_path: str = "tokenizer_config.json"):
        self.config = self._load_config(config_path)
        self.language_ranges = LANGUAGE_RANGES  # Use hardcoded language ranges
        self._language_check_cache = {}
        self._compile_patterns()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Config not found: {config_path}, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default config with basic language ranges."""
        return {
            "language_ranges": {
                "indic": {
                    "devanagari": {"ranges": [[2304, 2431]], "priority": "high"},
                    "bengali": {"ranges": [[2432, 2559]], "priority": "medium"},
                },
                "english": {"ranges": [[65, 90], [97, 122]], "priority": "critical"}
            }
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for categorization."""
        self.special_tokens = {
            '<s>', '</s>', '<pad>', '<unk>', '<|endoftext|>',
            '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]',
            '<|im_start|>', '<|im_end|>', '<|system|>', '<|user|>', '<|assistant|>',
            '[INST]', '[/INST]', '[TOOL_CALLS]', '[/TOOL_CALLS]'
        }
        self.code_char_pattern = re.compile(r'[{}()\[\]<>=+\-*/;:,.]')
        self.json_structural = {'{', '}', '[', ']', ':', ',', '"', "'"}
    
    def check_language(self, char: str) -> Tuple[bool, str, str]:
        """Check if character belongs to a script."""
        if char in self._language_check_cache:
            return self._language_check_cache[char]
        
        code_point = ord(char)
        
        for language_name, ranges in self.language_ranges.items():
            for start, end in ranges:
                if start <= code_point <= end:
                    # Determine family from language name
                    if any(indic in language_name for indic in ['Devanagari', 'Bengali', 'Tamil', 'Telugu', 
                                                                  'Kannada', 'Malayalam', 'Gujarati', 'Gurmukhi', 
                                                                  'Odia', 'Sinhala']):
                        family = 'indic'
                    elif any(asian in language_name for asian in ['Chinese', 'Japanese', 'Korean']):
                        family = 'east_asian'
                    elif any(me in language_name for me in ['Arabic', 'Hebrew', 'Persian']):
                        family = 'middle_eastern'
                    elif any(eu in language_name for eu in ['Cyrillic', 'Greek']):
                        family = 'european'
                    else:
                        family = 'other'
                    
                    result = (True, language_name, family)
                    self._language_check_cache[char] = result
                    return result
        
        result = (False, '', '')
        self._language_check_cache[char] = result
        return result
    
    def categorize_token(self, normalized_token: str) -> str:
        """Categorize token into categories."""
        if normalized_token in self.special_tokens:
            return "special"
        
        if len(normalized_token) == 1 and normalized_token in self.json_structural:
            return "json_structural"
        
        if not normalized_token or normalized_token.isspace():
            return "whitespace"
        
        has_english = False
        has_digit = False
        has_code_chars = False
        detected_languages = {}
        
        for char in normalized_token:
            code_point = ord(char)
            
            is_lang, lang_key, family = self.check_language(char)
            if is_lang:
                if family not in detected_languages:
                    detected_languages[family] = []
                if lang_key not in detected_languages[family]:
                    detected_languages[family].append(lang_key)
            
            if char.isalpha() and code_point < 128:
                has_english = True
            
            if char.isdigit():
                has_digit = True
            
            if self.code_char_pattern.search(char):
                has_code_chars = True
        
        # Categorization priority
        if detected_languages:
            non_english_langs = {k: v for k, v in detected_languages.items() 
                                if k != 'english'}
            
            if len(non_english_langs) == 1:
                family = list(non_english_langs.keys())[0]
                return family
            elif len(non_english_langs) > 1:
                return "multilingual"
        
        if has_code_chars and has_english:
            return "code"
        elif has_english:
            return "english"
        elif has_digit:
            return "numeric"
        elif has_code_chars:
            return "symbols"
        else:
            return "other"


class RepeatingPatternDetector:
    """Detects and chunks repeating character patterns."""
    
    def is_repeating(self, text: str, min_length: int = 3) -> Tuple[bool, str]:
        """Check if text is a repeating pattern."""
        if len(text) < min_length:
            return False, ""
        
        # Check for single character repetition
        if len(set(text)) == 1:
            return True, text[0]
        
        # Check for mixed patterns (prefix + repeating suffix)
        mixed_prefixes = ['//', '/*', '<!--', ' ', '\t']
        for prefix in mixed_prefixes:
            if text.startswith(prefix) and len(text) > len(prefix) + 3:
                suffix = text[len(prefix):]
                if len(set(suffix)) == 1 and len(suffix) >= 3:
                    return True, text
        
        # Check for 2-character repetition
        if len(text) >= 4:
            pattern = text[:2]
            if all(text[i:i+2] == pattern for i in range(0, len(text)-1, 2)):
                return True, pattern
        
        # Check for 3-character repetition
        if len(text) >= 6:
            pattern = text[:3]
            if all(text[i:i+3] == pattern for i in range(0, len(text)-2, 3)):
                return True, pattern
        
        # Check for 4-character repetition
        if len(text) >= 8:
            pattern = text[:4]
            if all(text[i:i+4] == pattern for i in range(0, len(text)-3, 4)):
                return True, pattern
        
        return False, ""
    
    def generate_chunks(self, text: str, chunk_sizes: List[int] = [8, 16]) -> List[str]:
        """Generate chunks of a repeating pattern."""
        is_rep, pattern = self.is_repeating(text)
        if not is_rep or len(text) <= 16:
            return []
        
        chunks = []
        
        # Check if it's a mixed pattern
        mixed_prefixes = ['//', '/*', '<!--', ' ', '\t']
        is_mixed = False
        prefix = ""
        repeat_char = ""
        
        for pref in mixed_prefixes:
            if pattern.startswith(pref) and len(pattern) > len(pref):
                suffix = pattern[len(pref):]
                if len(set(suffix)) == 1:
                    is_mixed = True
                    prefix = pref
                    repeat_char = suffix[0]
                    break
        
        for size in chunk_sizes:
            if size < len(text):
                if is_mixed:
                    repeats_needed = size - len(prefix)
                    if repeats_needed > 0:
                        chunk = prefix + (repeat_char * repeats_needed)
                    else:
                        chunk = prefix
                elif len(pattern) == 1:
                    chunk = pattern * size
                else:
                    chunk = (pattern * (size // len(pattern) + 1))[:size]
                
                if chunk and chunk != text:
                    chunks.append(chunk)
        
        return chunks


class TokenizerMerger:
    """Merge multiple tokenizers with chunking support."""
    
    def __init__(self, master_config_path: Optional[str] = None,
                 config_path: str = "tokenizer_config.json", 
                 special_tokens_config: Optional[str] = None,
                 target_size: int = 128000, 
                 verbose: bool = True):
        """
        Initialize TokenizerMerger.
        
        Args:
            master_config_path: Path to master config JSON (overrides other params)
            config_path: Path to tokenizer categorization config
            special_tokens_config: Path to special tokens config
            target_size: Target vocabulary size
            verbose: Enable verbose output
        """
        # Load master configuration if provided
        self.master_config = None
        if master_config_path:
            self.master_config = self._load_master_config(master_config_path)
        
        # Apply configuration (master config takes precedence)
        if self.master_config:
            self.target_size = self.master_config.get('tokenizer_settings', {}).get('target_size', target_size)
            self.verbose = self.master_config.get('tokenizer_settings', {}).get('verbose', verbose)
            config_path = self.master_config.get('paths', {}).get('tokenizer_config', config_path)
            special_tokens_config = self.master_config.get('paths', {}).get('special_tokens_config', special_tokens_config)
        else:
            self.target_size = target_size
            self.verbose = verbose
        
        # Initialize components
        self.normalizer = TokenNormalizer()
        self.categorizer = TokenCategorizer(config_path)
        self.placeholder_filter = PlaceholderFilter()
        self.pattern_detector = RepeatingPatternDetector()
        
        # Initialize special tokens manager
        special_tokens_path = Path(special_tokens_config) if special_tokens_config else None
        self.special_tokens_manager = SpecialTokensManager(special_tokens_path)
        
        # Storage
        self.tokenizers = {}
        self.token_registry = defaultdict(list)
        self.merged_vocab = {}
        
        # Statistics
        self.stats = defaultdict(int)
        
        # Load tokenizer priority from config or use defaults
        if self.master_config and 'tokenizer_priority' in self.master_config:
            self.tokenizer_priority = self._parse_tokenizer_priority(
                self.master_config['tokenizer_priority']
            )
        else:
            # Default priority (fallback if no config)
            self.tokenizer_priority = {
                'gptoss': {'code': 1, 'english': 1, 'indic': 1},
                'mistral': {'code': 2, 'english': 2, 'indic': 2},
                'byted': {'code': 3, 'english': 3, 'indic': 3},
                'ds': {'code': 4, 'english': 4, 'indic': 4},
                'olmo': {'code': 5, 'english': 5, 'indic': 5},
                'gemma': {'code': 6, 'english': 6, 'indic': 6},
                'qwen': {'code': 7, 'english': 7, 'indic': 7},
                'qwencode': {'code': 7, 'english': 7, 'indic': 7},
                'olmocode': {'code': 7, 'english': 7, 'indic': 7},
                'dscode': {'code': 8, 'english': 8, 'indic': 8},
            }
        
        # Load other config values
        self._load_config_values()
    
    def _load_master_config(self, config_path: str) -> Dict:
        """Load master configuration from JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            if self.verbose:
                print(f"📋 Loaded master config: {config_path}")
                print(f"   Config version: {config.get('config_version', 'unknown')}")
            return config
        except Exception as e:
            print(f"⚠️  Warning: Could not load master config: {e}")
            return None
    
    def _parse_tokenizer_priority(self, priority_config: Dict) -> Dict:
        """Parse tokenizer priority from config."""
        priority = {}
        for tokenizer_name, settings in priority_config.items():
            if tokenizer_name == 'description':
                continue
            if isinstance(settings, dict):
                # Extract only the priority values (code, english, indic)
                priority[tokenizer_name] = {
                    k: v for k, v in settings.items() 
                    if k in ['code', 'english', 'indic']
                }
        return priority
    
    def _load_config_values(self):
        """Load additional configuration values from master config."""
        if not self.master_config:
            # Set defaults
            self.excluded_categories = {'east_asian', 'middle_eastern', 'european', 'multilingual'}
            self.max_token_length = 32
            self.chunk_sizes = [8, 16]
            self.chunk_threshold = 16
            self.unwanted_ranges = [
                (0x0E00, 0x0E7F), (0x0E80, 0x0EFF), (0x1000, 0x109F),
                (0x1780, 0x17FF), (0x1200, 0x137F), (0x10A0, 0x10FF),
                (0x0530, 0x058F), (0x1680, 0x169F), (0x16A0, 0x16FF),
                (0x1700, 0x171F), (0x1720, 0x173F), (0x1740, 0x175F),
                (0x1760, 0x177F)
            ]
            self.category_order = [
                ('indic', 10000), ('code', 9000), ('english', 100000),
                ('numeric', 4000), ('symbols', 3000), ('other', 100),
                ('whitespace', 100)
            ]
            return
        
        # Load from config
        filtering = self.master_config.get('filtering_rules', {})
        self.excluded_categories = set(filtering.get('excluded_categories', []))
        self.max_token_length = filtering.get('max_token_length', 32)
        
        # Parse unwanted Unicode ranges
        unwanted_list = filtering.get('unwanted_unicode_ranges', [])
        self.unwanted_ranges = [
            (item['start'], item['end']) for item in unwanted_list
        ]
        
        # Load chunking rules
        chunking = self.master_config.get('chunking_rules', {})
        self.chunk_sizes = chunking.get('chunk_sizes', [8, 16])
        self.chunk_threshold = chunking.get('chunk_threshold_length', 16)
        
        # Load category allocation order
        cat_alloc = self.master_config.get('category_allocation', {})
        cat_order = cat_alloc.get('order', [])
        self.category_order = [
            (item['category'], item['max_allocation']) 
            for item in cat_order
        ]
    
    def load_tokenizer(self, json_path: Path) -> bool:
        """Load a single tokenizer from JSON file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract tokenizer name from filename
            name = json_path.stem.replace('_filtered', '').replace('_tokenizer', '')
            
            # Handle different JSON formats
            vocab = None
            
            if isinstance(data, dict):
                # Check if it's a nested format (has 'model' or 'vocab' keys)
                if 'model' in data and isinstance(data['model'], dict) and 'vocab' in data['model']:
                    vocab = data['model']['vocab']
                elif 'vocab' in data and isinstance(data['vocab'], dict):
                    vocab = data['vocab']
                else:
                    # Assume it's a flat token->id dictionary
                    # Check if values are integers (token IDs)
                    sample_values = list(data.values())[:5]
                    if sample_values and all(isinstance(v, int) for v in sample_values):
                        vocab = data
                    else:
                        if self.verbose:
                            print(f"  ✗ Unknown format in {json_path.name}")
                        return False
                
                if vocab:
                    self.tokenizers[name] = vocab
                    
                    if self.verbose:
                        print(f"  ✓ Loaded {name}: {len(vocab):,} tokens")
                    
                    return True
            
            return False
        
        except Exception as e:
            if self.verbose:
                print(f"  ✗ Error loading {json_path.name}: {e}")
            return False
    
    def load_directory(self, directory: Path) -> int:
        """Load all tokenizers from a directory."""
        if self.verbose:
            print(f"\n📁 Loading tokenizers from: {directory}")
        
        if not directory.exists():
            print(f"  ✗ Directory not found")
            return 0
        
        json_files = sorted(directory.glob("*.json"))
        
        if not json_files:
            print(f"  ✗ No JSON files found")
            return 0
        
        loaded = 0
        for json_file in json_files:
            if self.load_tokenizer(json_file):
                loaded += 1
        
        return loaded
    
    def load_analysis_report(self, report_path: Path):
        """Load analysis report for reference (optional)."""
        if not report_path.exists():
            if self.verbose:
                print(f"  ℹ️  Analysis report not found: {report_path}")
            return
        
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            if self.verbose:
                print(f"\n📊 Loaded analysis report: {len(report)} tokenizer(s)")
                for name, stats in report.items():
                    if 'vocab_size' in stats:
                        print(f"  {name}: {stats['vocab_size']:,} tokens, "
                              f"{stats.get('indic_percentage', 0):.2f}% Indic")
        
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  Could not load analysis report: {e}")
    
    def load_long_tokens_csv(self, csv_path: Path) -> int:
        """Load long tokens from CSV and generate chunks."""
        if self.verbose:
            print(f"\n📄 Loading long tokens from CSV: {csv_path}")
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                long_tokens = []
                for row in reader:
                    token_value = row['token_value']
                    decoded_value = row['decoded_value']
                    token_length = int(row['token_length'])
                    category = row['category']
                    
                    long_tokens.append({
                        'original': token_value,
                        'decoded': decoded_value,
                        'length': token_length,
                        'category': category
                    })
                
                if self.verbose:
                    print(f"  ✓ Loaded {len(long_tokens):,} long tokens")
                
                return self._process_long_tokens(long_tokens)
            
        except Exception as e:
            print(f"  ✗ Error loading CSV: {e}")
            return 0
    
    def _process_long_tokens(self, long_tokens: List[Dict]) -> int:
        """Process long tokens and generate chunks."""
        if self.verbose:
            print(f"\n🔧 Processing long tokens and generating chunks...")
        
        added = 0
        chunked = 0
        replaced = 0
        
        iterator = tqdm(long_tokens, desc="  Processing", 
                       disable=not HAS_TQDM or not self.verbose)
        
        for token_data in iterator:
            decoded = token_data['decoded']
            length = token_data['length']
            category = token_data['category']
            
            # Normalize the decoded token
            normalized = self.normalizer.normalize(decoded)
            
            # Check if it's a repeating pattern longer than 16
            is_rep, pattern = self.pattern_detector.is_repeating(normalized)
            
            if is_rep and length > self.chunk_threshold:
                # Generate chunks using configured sizes
                chunks = self.pattern_detector.generate_chunks(normalized, chunk_sizes=self.chunk_sizes)
                
                for chunk in chunks:
                    # Register chunk
                    self.token_registry[chunk].append({
                        'source': 'chunked_long',
                        'original': chunk,
                        'id': -1,
                        'category': category
                    })
                    self.stats[f'chunked_{category}'] += 1
                    chunked += 1
                
                # DON'T add the original - it's been replaced by chunks
                replaced += 1
            else:
                # Add non-repeating long tokens as-is
                self.token_registry[normalized].append({
                    'source': 'csv_long',
                    'original': decoded,
                    'id': -1,
                    'category': category
                })
                self.stats[f'long_{category}'] += 1
                added += 1
        
        if self.verbose:
            print(f"\n  📊 Processing Summary:")
            print(f"    Non-repeating long tokens added: {added:,}")
            print(f"    Repeating patterns replaced by chunks: {replaced:,}")
            print(f"    Chunks generated: {chunked:,}")
        
        return added + chunked
    
    def collect_tokens(self):
        """Phase 1: Collect and normalize all tokens."""
        if self.verbose:
            print(f"\n{'='*80}")
            print("PHASE 1: COLLECTING TOKENS")
            print('='*80)
        
        placeholders_filtered = 0
        
        for tokenizer_name, vocab in self.tokenizers.items():
            if self.verbose:
                print(f"\n  Processing {tokenizer_name}...")
            
            vocab_items = list(vocab.items())
            iterator = tqdm(vocab_items, desc=f"    Collecting", 
                          disable=not HAS_TQDM or not self.verbose)
            
            for original_token, token_id in iterator:
                # Filter placeholders
                is_placeholder, reason = self.placeholder_filter.is_placeholder(original_token)
                if is_placeholder:
                    placeholders_filtered += 1
                    self.stats[f"filtered_{reason}"] += 1
                    continue
                
                # Normalize token
                normalized = self.normalizer.normalize(original_token)
                
                # Categorize
                category = self.categorizer.categorize_token(normalized)
                
                # Register
                self.token_registry[normalized].append({
                    'source': tokenizer_name,
                    'original': original_token,
                    'id': token_id,
                    'category': category
                })
                
                self.stats[f"collected_{tokenizer_name}"] += 1
                self.stats[f"category_{category}"] += 1
        
        if self.verbose:
            print(f"\n  📊 Collection Summary:")
            print(f"    Unique normalized tokens: {len(self.token_registry):,}")
            total_instances = sum(self.stats[k] for k in self.stats if k.startswith('collected_'))
            print(f"    Total instances: {total_instances:,}")
            print(f"    🗑️  Placeholders filtered: {placeholders_filtered:,}")
    
    def select_best_tokens(self):
        """Phase 2: Select best version of each token."""
        if self.verbose:
            print(f"\n{'='*80}")
            print("PHASE 2: SELECTING BEST TOKENS (with strict filtering)")
            print('='*80)
        
        # Categories to EXCLUDE (from config)
        excluded_categories = self.excluded_categories
        
        selected_tokens = {}
        filtered_count = 0
        length_filtered = 0
        script_filtered = 0
        
        registry_items = list(self.token_registry.items())
        iterator = tqdm(registry_items, desc="  Selecting best", 
                       disable=not HAS_TQDM or not self.verbose)
        
        for normalized, instances in iterator:
            best = self._select_best_instance(normalized, instances)
            
            if best:
                # Filter 1: Excluded categories
                if best['category'] in excluded_categories:
                    filtered_count += 1
                    self.stats[f"filtered_{best['category']}"] += 1
                    continue
                
                # Filter 2: Length > max_token_length (from config)
                if len(normalized) > self.max_token_length:
                    length_filtered += 1
                    self.stats[f"filtered_long_token"] += 1
                    continue
                
                # Filter 3: Check for unwanted Unicode ranges (Southeast Asian, Ethiopic, etc.)
                if self._contains_unwanted_scripts(normalized):
                    script_filtered += 1
                    self.stats[f"filtered_unwanted_script"] += 1
                    continue
                
                selected_tokens[normalized] = best
                self.stats[f"selected_{best['category']}"] += 1
        
        if self.verbose:
            print(f"\n  📊 Selection Summary:")
            print(f"    Selected tokens: {len(selected_tokens):,}")
            if filtered_count > 0:
                print(f"    🗑️  Excluded categories: {filtered_count:,} tokens")
            if length_filtered > 0:
                print(f"    🗑️  Length > 32 filtered: {length_filtered:,} tokens")
            if script_filtered > 0:
                print(f"    🗑️  Unwanted scripts filtered: {script_filtered:,} tokens")
            
            category_counts = defaultdict(int)
            for token_data in selected_tokens.values():
                category_counts[token_data['category']] += 1
            
            print(f"\n  📂 Category Breakdown:")
            for category in sorted(category_counts.keys(), key=lambda c: -category_counts[c]):
                count = category_counts[category]
                pct = (count / len(selected_tokens) * 100) if len(selected_tokens) > 0 else 0
                print(f"    {category:20s}: {count:7,} ({pct:5.2f}%)")
        
        return selected_tokens
    
    def _contains_unwanted_scripts(self, token: str) -> bool:
        """Check if token contains unwanted Unicode scripts (from config)."""
        for char in token:
            code_point = ord(char)
            for start, end in self.unwanted_ranges:
                if start <= code_point <= end:
                    return True
        
        return False
    
    def _select_best_instance(self, normalized: str, instances: List[Dict]) -> Optional[Dict]:
        """Select best instance based on priority."""
        if not instances:
            return None
        
        if len(instances) == 1:
            return instances[0]
        
        category = instances[0]['category']
        
        def sort_key(inst):
            source = inst['source']
            token_id = inst['id']
            
            # Prioritize non-generated sources
            if source in ['chunked_long', 'csv_long']:
                return (99, token_id, source)
            
            tokenizer_prio = 99
            if source in self.tokenizer_priority:
                tokenizer_prio = self.tokenizer_priority[source].get(category, 99)
            
            return (tokenizer_prio, token_id, source)
        
        sorted_instances = sorted(instances, key=sort_key)
        return sorted_instances[0]
    
    def assign_merged_ids(self, selected_tokens: Dict) -> Dict:
        """Phase 3: Assign IDs with special tokens, reserved IDs, and regular vocabulary."""
        if self.verbose:
            print(f"\n{'='*80}")
            print("PHASE 3: ASSIGNING IDs (Special Tokens + Reserved + Regular Vocab)")
            print('='*80)
        
        merged_vocab = {}
        
        # STEP 1: Add special tokens at their designated IDs
        special_tokens = self.special_tokens_manager.get_all_special_tokens()
        for token, token_id in special_tokens.items():
            merged_vocab[token] = token_id
        
        if self.verbose:
            print(f"\n  🔐 Special Tokens: {len(special_tokens):,} tokens allocated")
            print(f"     Token ID range: 0-{max(special_tokens.values())}")
        
        # STEP 2: Mark reserved IDs (not allocated, just reserved)
        reserved_count = len(self.special_tokens_manager.reserved_ids)
        if self.verbose and reserved_count > 0:
            reserved_range = self.special_tokens_manager.reserved_ids
            min_reserved = min(reserved_range)
            max_reserved = max(reserved_range)
            print(f"\n  🔒 Reserved IDs: {reserved_count:,} IDs reserved for future governance")
            print(f"     Reserved range: {min_reserved}-{max_reserved}")
        
        # STEP 3: Allocate regular vocabulary starting after special tokens and reserved IDs
        regular_vocab_start = self.special_tokens_manager.get_regular_vocab_start_id()
        if self.verbose:
            print(f"\n  📚 Regular Vocabulary starts at ID: {regular_vocab_start:,}")
        
        # Group tokens by category
        by_category = defaultdict(list)
        for normalized, instance in selected_tokens.items():
            # Skip if token is already in special tokens
            if normalized in special_tokens:
                continue
            by_category[instance['category']].append((normalized, instance))
        
        # Category priority and allocations (from config)
        category_order = self.category_order
        
        # Calculate remaining budget for regular vocabulary
        current_id = regular_vocab_start
        remaining_budget = self.target_size - len(special_tokens)
        
        # First pass: allocate minimum or actual (whichever is smaller)
        for category, max_alloc in category_order:
            if category not in by_category:
                continue
            
            tokens = by_category[category]
            tokens.sort(key=lambda x: x[0])  # Sort alphabetically
            
            # Allocate up to maximum, or all tokens if fewer
            to_allocate = min(len(tokens), max_alloc, remaining_budget)
            
            if self.verbose:
                status = f"(allocated {to_allocate:,}/{len(tokens):,})"
                print(f"     {category:20s}: {len(tokens):7,} tokens {status}")
            
            for i, (normalized, _) in enumerate(tokens[:to_allocate]):
                merged_vocab[normalized] = current_id
                current_id += 1
            
            remaining_budget -= to_allocate
            
            if remaining_budget <= 0:
                if self.verbose:
                    print(f"\n  ⚠️  Reached target size {self.target_size:,}")
                break
        
        # Second pass: backfill remaining budget
        if remaining_budget > 0 and self.verbose:
            print(f"\n  📊 Backfilling {remaining_budget:,} remaining slots...")
        
        for category, max_alloc in category_order:
            if category not in by_category or remaining_budget <= 0:
                continue
            
            tokens = by_category[category]
            remaining_tokens = [t for t, _ in tokens if t not in merged_vocab]
            
            if remaining_tokens:
                to_add = min(len(remaining_tokens), remaining_budget)
                
                if self.verbose and to_add > 0:
                    print(f"     {category:20s}: +{to_add:,} more tokens")
                
                for normalized in remaining_tokens[:to_add]:
                    merged_vocab[normalized] = current_id
                    current_id += 1
                    remaining_budget -= 1
                
                if remaining_budget <= 0:
                    break
        
        if self.verbose:
            print(f"\n  ✓ Total tokens assigned: {len(merged_vocab):,}")
            print(f"     Special tokens: {len(special_tokens):,}")
            print(f"     Regular vocabulary: {len(merged_vocab) - len(special_tokens):,}")
            if remaining_budget > 0:
                print(f"  ℹ️  Unused slots: {remaining_budget:,}")
        
        return merged_vocab
    
    def enforce_target_size(self, merged_vocab: Dict) -> Dict:
        """Phase 4: Ensure exactly target size."""
        if len(merged_vocab) == self.target_size:
            return merged_vocab
        
        if self.verbose:
            print(f"\n{'='*80}")
            print("PHASE 4: ENFORCING TARGET SIZE")
            print('='*80)
        
        if len(merged_vocab) > self.target_size:
            if self.verbose:
                print(f"  Trimming from {len(merged_vocab):,} to {self.target_size:,}")
            
            # Keep first target_size tokens
            sorted_items = sorted(merged_vocab.items(), key=lambda x: x[1])
            merged_vocab = dict(sorted_items[:self.target_size])
        
        if self.verbose:
            print(f"  ✓ Final size: {len(merged_vocab):,}")
        
        return merged_vocab
    
    def merge(self) -> Dict:
        """Execute full merge pipeline."""
        self.collect_tokens()
        selected_tokens = self.select_best_tokens()
        merged_vocab = self.assign_merged_ids(selected_tokens)
        merged_vocab = self.enforce_target_size(merged_vocab)
        
        self.merged_vocab = merged_vocab
        return merged_vocab
    
    def save_vocab(self, output_path: Path):
        """Save merged vocabulary to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sorted_vocab = dict(sorted(self.merged_vocab.items(), key=lambda x: x[1]))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_vocab, f, ensure_ascii=False, indent=2)
        
        if self.verbose:
            print(f"\n💾 Merged tokenizer saved to: {output_path}")
    
    def save_report(self, output_path: Path):
        """Save merge report."""
        report_path = output_path.parent / f"{output_path.stem}_report.json"
        
        # Calculate statistics
        category_distribution = defaultdict(int)
        source_distribution = defaultdict(int)
        
        for token in self.merged_vocab.keys():
            if token in self.token_registry:
                instances = self.token_registry[token]
                if instances:
                    category_distribution[instances[0]['category']] += 1
                    source_distribution[instances[0]['source']] += 1
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'target_size': self.target_size,
            'final_vocab_size': len(self.merged_vocab),
            'category_distribution': dict(category_distribution),
            'source_distribution': dict(source_distribution),
            'statistics': dict(self.stats)
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"📊 Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge tokenizers with chunking support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use master config file (recommended)
  python merge_tokenizers_with_chunking_updated.py --master-config tokenizer_merge_config.json
  
  # Override specific values
  python merge_tokenizers_with_chunking_updated.py --master-config tokenizer_merge_config.json --target-size 64000
  
  # Use individual parameters (legacy mode)
  python merge_tokenizers_with_chunking_updated.py --input-dir filtered_tokenizer --output merged.json
        """
    )
    
    # Master config (takes precedence over individual parameters)
    parser.add_argument("--master-config", "-m",
                       help="Master configuration JSON file (overrides other parameters)")
    
    # Individual parameters (used if --master-config not provided, or to override config values)
    parser.add_argument("--input-dir", "-i",
                       help="Directory with filtered tokenizer JSON files")
    parser.add_argument("--output", "-o",
                       help="Output merged tokenizer JSON file")
    parser.add_argument("--target-size", "-t", type=int,
                       help="Target vocabulary size")
    parser.add_argument("--config",
                       help="Tokenizer categorization config file")
    parser.add_argument("--special-tokens-config", 
                       help="Special tokens configuration JSON")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Quiet mode (minimal output)")
    
    args = parser.parse_args()
    
    # Load master config if provided
    master_config = None
    if args.master_config:
        try:
            with open(args.master_config, 'r', encoding='utf-8') as f:
                master_config = json.load(f)
        except Exception as e:
            print(f"❌ Error loading master config: {e}")
            return 1
    
    # Determine parameter values (command-line overrides config)
    if master_config:
        paths = master_config.get('paths', {})
        settings = master_config.get('tokenizer_settings', {})
        
        input_dir = args.input_dir or paths.get('input_dir')
        output = args.output or paths.get('output')
        target_size = args.target_size or settings.get('target_size', 128000)
        config = args.config or paths.get('tokenizer_config', 'tokenizer_config.json')
        special_tokens_config = args.special_tokens_config or paths.get('special_tokens_config', 'special_tokens_config.json')
        quiet = args.quiet if args.quiet else not settings.get('verbose', True)
    else:
        # No master config - use individual parameters (with defaults)
        input_dir = args.input_dir
        output = args.output or "merged_tokenizer_128k.json"
        target_size = args.target_size or 128000
        config = args.config or "tokenizer_config.json"
        special_tokens_config = args.special_tokens_config or "special_tokens_config.json"
        quiet = args.quiet
    
    # Validate required parameters
    if not input_dir:
        print("❌ Error: --input-dir is required (or specify in master config)")
        return 1
    
    # Create merger
    merger = TokenizerMerger(
        master_config_path=args.master_config,
        config_path=config,
        special_tokens_config=special_tokens_config,
        target_size=target_size,
        verbose=not quiet
    )
    
    if not quiet:
        print("="*80)
        print("TOKENIZER MERGER WITH CHUNKING - ENHANCED")
        print("="*80)
        if args.master_config:
            print(f"📋 Using master config: {args.master_config}")
        print()
    
    # Load tokenizers from directory
    input_dir_path = Path(input_dir)
    loaded = merger.load_directory(input_dir_path)
    
    if loaded == 0:
        print("\n❌ No tokenizers loaded. Exiting.")
        return 1
    
    # Execute merge
    merger.merge()
    
    # Save outputs
    output_path = Path(output)
    merger.save_vocab(output_path)
    merger.save_report(output_path)
    
    if not quiet:
        print("\n" + "="*80)
        print("✅ MERGE COMPLETE")
        print("="*80)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
