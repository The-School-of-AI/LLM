#!/usr/bin/env python3
"""
Generate comprehensive tokenizer analysis report.

Analyzes vocabulary, Indic language support, and token characteristics
from tokenizer JSON files with hardcoded language ranges.

Usage:
    # Basic analysis
    python generate_tokenizer_report.py
    
    # Custom directory and output
    python generate_tokenizer_report.py --dir ../data --output tokenizer_results/tokenizer_analysis_report.json
    
    # Analyze tokens with minimum length
    python generate_tokenizer_report.py --min-token-length 5
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Import common utilities
from utils import (
    LANGUAGE_RANGES,
    check_language,
    is_english_char,
    categorize_token,
    VocabularyWrapper,
    load_vocab_from_json
)



class TokenizerAnalyzer:
    """Analyze tokenizer vocabulary and characteristics."""
    
    def __init__(self):
        self.tokenizers = {}
        self.analyses = {}
        self.category_stats = {}  # Track category counts per tokenizer
    
    def load_tokenizer_from_json(self, json_path: Path, tokenizer_name: str = None) -> bool:
        """
        Load a tokenizer vocabulary from a JSON file.
        
        Args:
            json_path: Path to tokenizer.json file
            tokenizer_name: Optional name for the tokenizer
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if tokenizer_name is None:
                # Extract name from filename
                filename = json_path.stem
                if filename.endswith('_tokenizer'):
                    tokenizer_name = filename[:-10]
                else:
                    tokenizer_name = filename
            
            with open(json_path, 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)
            
            # Extract vocabulary from JSON - support multiple formats
            vocab = {}
            if isinstance(tokenizer_data, dict):
                if 'model' in tokenizer_data and isinstance(tokenizer_data['model'], dict) and 'vocab' in tokenizer_data['model']:
                    # Nested format: {"model": {"vocab": {...}}}
                    vocab = tokenizer_data['model']['vocab']
                elif 'vocab' in tokenizer_data and isinstance(tokenizer_data['vocab'], dict):
                    # Format: {"vocab": {...}} - must be a dict, not a token named "vocab"
                    vocab = tokenizer_data['vocab']
                else:
                    # Flat format: {"token": id, "token2": id2, ...}
                    # Check if it looks like a flat vocab (all values are integers)
                    sample_values = list(tokenizer_data.values())[:10]
                    if sample_values and all(isinstance(v, int) for v in sample_values):
                        vocab = tokenizer_data
                    else:
                        print(f"✗ {tokenizer_name}: Could not find vocabulary")
                        return False
            else:
                print(f"✗ {tokenizer_name}: Invalid JSON format")
                return False
            
            # Create vocabulary wrapper
            tokenizer = VocabularyWrapper(tokenizer_name, vocab)
            self.tokenizers[tokenizer_name] = tokenizer
            
            print(f"✓ {tokenizer_name:15} (vocab: {tokenizer.vocab_size:>8,})")
            return True
            
        except Exception as e:
            print(f"✗ {tokenizer_name or json_path.stem:15} - Error: {e}")
            return False
    
    def load_tokenizers_from_directory(self, directory: Path) -> int:
        """
        Load all tokenizer JSON files from a directory.
        
        Args:
            directory: Path to directory containing tokenizer JSON files
        
        Returns:
            Number of tokenizers loaded successfully
        """
        if not directory.exists():
            print(f"Error: Directory not found: {directory}")
            return 0
        
        # Find all JSON files in directory
        json_files = list(directory.glob("*.json"))
        
        if not json_files:
            print(f"Error: No JSON files found in {directory}")
            return 0
        
        loaded_count = 0
        for json_file in sorted(json_files):
            if self.load_tokenizer_from_json(json_file):
                loaded_count += 1
        
        return loaded_count
    
    def check_language(self, char: str) -> Tuple[bool, str]:
        """Check if a character belongs to a non-Latin script and return the language/script name."""
        code_point = ord(char)
        
        for language_name, ranges in LANGUAGE_RANGES.items():
            for start, end in ranges:
                if start <= code_point <= end:
                    return True, language_name
        
        return False, ''
    
    def is_english_char(self, char: str) -> bool:
        """Check if a character is English (a-z, A-Z)."""
        return char.isalpha() and ord(char) < 128
    
    def decode_byte_token(self, token: str) -> str:
        """Decode byte-level BPE token to actual text."""
        try:
            # Remove special markers
            clean = token.replace('▁', '').replace('Ġ', ' ').replace('##', '')
            clean = clean.replace('<|', '').replace('|>', '').strip()
            
            # For byte-level BPE tokenizers (like GPT-2, many modern tokenizers)
            # The tokens are already UTF-8 characters that may represent bytes
            # Try to decode if it looks like byte encoding
            
            # Method 1: Direct interpretation (works for many tokenizers)
            return clean
            
        except Exception:
            return token
    
    def analyze_token(self, token: str) -> Dict:
        """Analyze a single token."""
        # Decode the token (handles byte-level BPE)
        decoded_token = self.decode_byte_token(token)
        
        # Skip empty tokens
        if not decoded_token:
            return {
                'length': 0,
                'indic_char_count': 0,
                'is_indic_token': False,
                'script_counts': {},
                'original': token,
                'clean': decoded_token
            }
        
        # Count Indic characters
        indic_chars = []
        script_counts = defaultdict(int)
        
        try:
            for char in decoded_token:
                is_indic, script = self.is_indic_char(char)
                if is_indic:
                    indic_chars.append(char)
                    script_counts[script] += 1
        except Exception as e:
            # Handle any Unicode errors
            pass
        
        return {
            'length': len(decoded_token),
            'indic_char_count': len(indic_chars),
            'is_indic_token': len(indic_chars) > 0,
            'script_counts': dict(script_counts),
            'original': token,
            'clean': decoded_token
        }
    
    def analyze_tokenizer(self, name: str, min_token_length: int = 0) -> Dict:
        """Analyze a tokenizer's vocabulary.
        
        Args:
            name: Name of the tokenizer to analyze
            min_token_length: Minimum token length to include in category analysis (default: 0 = all tokens)
        """
        tokenizer = self.tokenizers[name]
        vocab = tokenizer.get_vocab()
        
        stats = {
            'name': name,
            'vocab_size': len(vocab),
            'total_indic_tokens': 0,  # Keep for backwards compatibility
            'indic_percentage': 0.0,
            'total_multilingual_tokens': 0,  # New: all non-English languages
            'multilingual_percentage': 0.0,
            'total_english_tokens': 0,
            'english_percentage': 0.0,
            'long_tokens_count': 0,
            'long_tokens_percentage': 0.0,
            'language_token_counts': defaultdict(int),  # Per-language counts
            'tokens_by_length': defaultdict(int),
            'category_counts': defaultdict(int),  # Category counts
        }
        
        # Initialize category stats for this tokenizer
        self.category_stats[name] = {
            'min_length': min_token_length,
            'categories': defaultdict(int),
            'total_tokens_analyzed': 0
        }
        
        # Analyze each token by decoding using the tokenizer
        for token, token_id in vocab.items():
            # Use tokenizer to decode the token ID to actual text
            try:
                decoded_text = tokenizer.decode([token_id], skip_special_tokens=False)
            except:
                # Fallback to original token string
                decoded_text = token
            
            # Check for non-Latin script characters
            found_languages = set()
            language_chars = []
            english_chars = []
            
            try:
                for char in decoded_text:
                    # Check for non-Latin languages
                    has_lang, language = self.check_language(char)
                    if has_lang:
                        found_languages.add(language)
                        language_chars.append(char)
                    
                    # Check English
                    if self.is_english_char(char):
                        english_chars.append(char)
            except:
                pass
            
            # Update statistics
            length = len(decoded_text)
            stats['tokens_by_length'][length] += 1
            
            if length > 32:
                stats['long_tokens_count'] += 1
            
            # Categorize token (if meets minimum length)
            if len(token) >= min_token_length:
                category = categorize_token(decoded_text, self.check_language)
                stats['category_counts'][category] += 1
                self.category_stats[name]['categories'][category] += 1
                self.category_stats[name]['total_tokens_analyzed'] += 1
            
            # Count tokens with non-Latin scripts
            if found_languages:
                stats['total_multilingual_tokens'] += 1
                
                # Count tokens per language
                for lang in found_languages:
                    stats['language_token_counts'][lang] += 1
                    
                    # Also count Indic tokens for backwards compatibility
                    if any(indic_name in lang for indic_name in [
                        'Devanagari', 'Bengali', 'Tamil', 'Telugu', 'Kannada',
                        'Malayalam', 'Gujarati', 'Gurmukhi', 'Odia', 'Sinhala'
                    ]):
                        stats['total_indic_tokens'] += 1
            
            if english_chars:
                stats['total_english_tokens'] += 1
        
        # Calculate percentages
        stats['indic_percentage'] = (stats['total_indic_tokens'] / stats['vocab_size']) * 100
        stats['multilingual_percentage'] = (stats['total_multilingual_tokens'] / stats['vocab_size']) * 100
        stats['english_percentage'] = (stats['total_english_tokens'] / stats['vocab_size']) * 100
        stats['long_tokens_percentage'] = (stats['long_tokens_count'] / stats['vocab_size']) * 100
        
        return stats
    
    def analyze_all(self, min_token_length: int = 0):
        """Analyze all loaded tokenizers.
        
        Args:
            min_token_length: Minimum token length for category analysis (default: 0 = all tokens)
        """
        total = len(self.tokenizers)
        for idx, name in enumerate(self.tokenizers, 1):
            print(f"[{idx}/{total}] Analyzing {name}...", end=" ", flush=True)
            self.analyses[name] = self.analyze_tokenizer(name, min_token_length)
            indic = self.analyses[name]['total_indic_tokens']
            print(f"✓ (Indic: {indic:,}, Vocab: {self.analyses[name]['vocab_size']:,})")
    
    def print_comparison_report(self):
        """Print a formatted comparison report."""
        if not self.analyses:
            print("No analyses available. Run analyze_all() first.")
            return
        
        # Header
        tokenizer_names = ' | '.join(self.analyses.keys())
        print("\n" + "=" * 140)
        print(f"COMPARISON RESULTS: {tokenizer_names}")
        print("=" * 140)
        
        # Main metrics
        print("\n")
        print(f"{'Metric':<40}", end="")
        for name in self.analyses:
            print(f"{name:>25}", end="")
        print()
        print("-" * 140)
        
        # Vocabulary size
        print(f"{'Total vocabulary size':<40}", end="")
        for name in self.analyses:
            print(f"{self.analyses[name]['vocab_size']:>25,}", end="")
        print()
        
        # Indic tokens
        print(f"{'Total Indic tokens':<40}", end="")
        for name in self.analyses:
            print(f"{self.analyses[name]['total_indic_tokens']:>25,}", end="")
        print()
        
        # Indic percentage
        print(f"{'Indic tokens (%)':<40}", end="")
        for name in self.analyses:
            pct = self.analyses[name]['indic_percentage']
            print(f"{pct:>24.2f}%", end="")
        print()
        
        # English tokens
        print(f"{'Total English tokens':<40}", end="")
        for name in self.analyses:
            print(f"{self.analyses[name]['total_english_tokens']:>25,}", end="")
        print()
        
        # English percentage
        print(f"{'English tokens (%)':<40}", end="")
        for name in self.analyses:
            pct = self.analyses[name]['english_percentage']
            print(f"{pct:>24.2f}%", end="")
        print()
        
        # Multilingual tokens
        print(f"{'Total multilingual tokens':<40}", end="")
        for name in self.analyses:
            print(f"{self.analyses[name]['total_multilingual_tokens']:>25,}", end="")
        print()
        
        # Multilingual percentage
        print(f"{'Multilingual tokens (%)':<40}", end="")
        for name in self.analyses:
            pct = self.analyses[name]['multilingual_percentage']
            print(f"{pct:>24.2f}%", end="")
        print()
        
        # Long tokens count
        print(f"{'Tokens with length > 32':<40}", end="")
        for name in self.analyses:
            print(f"{self.analyses[name]['long_tokens_count']:>25,}", end="")
        print()
        
        # Long tokens percentage
        print(f"{'Long tokens (>32) (%)':<40}", end="")
        for name in self.analyses:
            pct = self.analyses[name]['long_tokens_percentage']
            print(f"{pct:>24.2f}%", end="")
        print()
        
        # Language breakdown by category
        print("\n")
        print("-" * 140)
        print("LANGUAGE BREAKDOWN:")
        print("-" * 140)
        
        # Get all languages across all tokenizers
        all_languages = set()
        for analysis in self.analyses.values():
            all_languages.update(analysis['language_token_counts'].keys())
        
        # Categorize languages
        language_categories = {
            'Indic Scripts': [],
            'East Asian Scripts': [],
            'Middle Eastern Scripts': [],
            'European Scripts': [],
            'Southeast Asian Scripts': [],
            'Other Scripts': []
        }
        
        for lang in all_languages:
            if any(indic in lang for indic in ['Devanagari', 'Bengali', 'Tamil', 'Telugu', 'Kannada', 
                                                'Malayalam', 'Gujarati', 'Gurmukhi', 'Odia', 'Sinhala']):
                language_categories['Indic Scripts'].append(lang)
            elif any(asian in lang for asian in ['Chinese', 'Japanese', 'Korean']):
                language_categories['East Asian Scripts'].append(lang)
            elif any(me in lang for me in ['Arabic', 'Hebrew', 'Persian']):
                language_categories['Middle Eastern Scripts'].append(lang)
            elif any(eu in lang for eu in ['Cyrillic', 'Greek']):
                language_categories['European Scripts'].append(lang)
            elif any(sea in lang for sea in ['Thai', 'Lao', 'Myanmar', 'Khmer']):
                language_categories['Southeast Asian Scripts'].append(lang)
            else:
                language_categories['Other Scripts'].append(lang)
        
        # Print each category
        for category, languages in language_categories.items():
            if not languages:
                continue
            
            print(f"\n{category}:")
            print(f"{'Language':<40}", end="")
            for name in self.analyses:
                print(f"{name:>25}", end="")
            print()
            print("-" * 140)
            
            # Sort languages within category
            language_order = list(LANGUAGE_RANGES.keys())
            sorted_languages = sorted(languages, key=lambda x: language_order.index(x) if x in language_order else 999)
            
            # Print each language
            for language in sorted_languages:
                print(f"{language:<40}", end="")
                for name in self.analyses:
                    count = self.analyses[name]['language_token_counts'].get(language, 0)
                    print(f"{count:>25,}", end="")
                print()
        
        print("\n" + "=" * 140)
    
    def save_report(self, output_path: str = None):
        """Save the analysis to JSON."""
        if output_path is None:
            output_path = "tokenizer_analysis_report.json"
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert defaultdicts to regular dicts for JSON serialization
        serializable_analyses = {}
        for name, analysis in self.analyses.items():
            serializable_analyses[name] = {
                'name': analysis['name'],
                'vocab_size': analysis['vocab_size'],
                'total_indic_tokens': analysis['total_indic_tokens'],
                'indic_percentage': analysis['indic_percentage'],
                'total_multilingual_tokens': analysis['total_multilingual_tokens'],
                'multilingual_percentage': analysis['multilingual_percentage'],
                'total_english_tokens': analysis['total_english_tokens'],
                'english_percentage': analysis['english_percentage'],
                'long_tokens_count': analysis['long_tokens_count'],
                'long_tokens_percentage': analysis['long_tokens_percentage'],
                'language_token_counts': dict(analysis['language_token_counts']),
                'tokens_by_length': {str(k): v for k, v in analysis['tokens_by_length'].items()},
                'category_counts': dict(analysis['category_counts']),
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_analyses, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate tokenizer analysis report from JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python generate_tokenizer_report.py
  
  # Custom directory and output
  python generate_tokenizer_report.py --dir ../data --output my_report.json
  
  # Analyze tokens with minimum length
  python generate_tokenizer_report.py --min-token-length 5
        """
    )
    
    parser.add_argument(
        '--dir', '-d',
        default='../data',
        help='Directory containing tokenizer JSON files (default: ../data/)'
    )
    parser.add_argument(
        '--output', '-o',
        default='tokenizer_results/tokenizer_analysis_report.json',
        help='Output JSON file path (default: tokenizer_results/tokenizer_analysis_report.json)'
    )
    parser.add_argument(
        '--min-token-length', '-m',
        type=int,
        default=0,
        help='Minimum token length for category analysis (default: 0 = all tokens)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 100)
    print("  TOKENIZER ANALYSIS REPORT GENERATOR")
    print("=" * 100)
    print(f"\n📁 Input:  {args.dir}")
    print(f"💾 Output: {args.output}")
    if args.min_token_length > 0:
        print(f"⚙️  Min token length: {args.min_token_length}")
    
    # Create analyzer
    analyzer = TokenizerAnalyzer()
    
    # Load all tokenizers from directory
    print("\n" + "-" * 100)
    print("Loading Tokenizers")
    print("-" * 100)
    
    tokenizer_dir = Path(args.dir)
    loaded = analyzer.load_tokenizers_from_directory(tokenizer_dir)
    
    if loaded == 0:
        print("\n✗ No tokenizers loaded.")
        print(f"Please ensure JSON files exist in: {tokenizer_dir}")
        return 1
    
    print(f"\n✅ Loaded {loaded} tokenizer(s)")
    
    # Analyze vocabularies
    print("\n" + "-" * 100)
    print("Analyzing Vocabularies")
    if args.min_token_length > 0:
        print(f"(Category analysis: min token length = {args.min_token_length})")
    print("-" * 100)
    
    analyzer.analyze_all(min_token_length=args.min_token_length)
    
    # Print comparison report
    analyzer.print_comparison_report()
    
    # Save to JSON
    analyzer.save_report(args.output)
    
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
