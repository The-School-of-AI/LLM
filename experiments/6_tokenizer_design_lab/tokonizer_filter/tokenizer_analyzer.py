#!/usr/bin/env python3
"""
Unified Tokenizer Analysis Tool

Analyze tokenizer vocabularies from JSON files.
Generate reports, find common tokens, and export data for analysis.

Usage Examples:
  python tokenizer_analyzer.py --report
  python tokenizer_analyzer.py --common-tokens --min-length 32
  python tokenizer_analyzer.py --report --common-tokens --export-long-tokens
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import unicodedata
from datetime import datetime
import csv

# Import common utilities
from utils import (
    LANGUAGE_RANGES,
    check_language,
    is_english_char,
    categorize_token,
    VocabularyWrapper,
    load_vocab_from_json
)

# Unicode ranges for world languages (comprehensive coverage)
# Hardcoded language ranges for consistent detection across all tokenizers


class TokenizerAnalyzer:
    """Unified tokenizer analysis tool."""
    
    def __init__(self):
        self.tokenizers = {}
        self.analyses = {}
        self.category_stats = {}
    
    def load_tokenizer_from_json(self, json_path: Path) -> bool:
        """Load a tokenizer vocabulary from JSON file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle multiple JSON formats
            vocab = {}
            if isinstance(data, dict):
                if 'model' in data and isinstance(data['model'], dict) and 'vocab' in data['model']:
                    # Nested format: {"model": {"vocab": {...}}}
                    vocab = data['model']['vocab']
                elif 'vocab' in data and isinstance(data['vocab'], dict):
                    # Format: {"vocab": {...}} - must be a dict, not a token named "vocab"
                    vocab = data['vocab']
                else:
                    # Flat format: {"token": id, "token2": id2, ...}
                    # Check if it looks like a flat vocab (all values are integers)
                    sample_values = list(data.values())[:10]
                    if sample_values and all(isinstance(v, int) for v in sample_values):
                        vocab = data
                    else:
                        print(f"  ✗ Failed to load {json_path.name}: Unknown format")
                        return False
            else:
                print(f"  ✗ Failed to load {json_path.name}: Invalid JSON")
                return False
            
            # Validate vocab is a dict
            if not isinstance(vocab, dict):
                print(f"  ✗ Failed to load {json_path.name}: vocab is not a dictionary")
                return False
            
            # Extract tokenizer name from filename
            name = json_path.stem
            if name.endswith('_tokenizer'):
                name = name[:-10]
            
            # Create wrapper
            wrapper = VocabularyWrapper(name, vocab)
            self.tokenizers[name] = wrapper
            
            print(f"  ✓ Loaded {name} (vocab size: {len(vocab):,})")
            return True
            
        except Exception as e:
            print(f"  ✗ Failed to load {json_path.name}: {e}")
            return False
    
    def load_tokenizers_from_directory(self, directory: Path) -> int:
        """Load all tokenizers from a directory."""
        if not directory.exists():
            print(f"Error: Directory not found: {directory}")
            return 0
        
        # Find JSON files directly in the directory
        json_files = []
        for file in directory.iterdir():
            if file.is_file() and file.suffix == '.json':
                json_files.append(file)
        
        if not json_files:
            print(f"Error: No JSON files found in {directory}")
            return 0
        
        loaded_count = 0
        for json_file in json_files:
            if self.load_tokenizer_from_json(json_file):
                loaded_count += 1
        
        return loaded_count
    
    def analyze_tokenizer(self, name: str, min_token_length: int = 0) -> Dict:
        """Analyze a tokenizer's vocabulary."""
        tokenizer = self.tokenizers[name]
        vocab = tokenizer.get_vocab()
        
        stats = {
            'name': name,
            'vocab_size': len(vocab),
            'total_indic_tokens': 0,
            'indic_percentage': 0.0,
            'total_multilingual_tokens': 0,
            'multilingual_percentage': 0.0,
            'total_english_tokens': 0,
            'english_percentage': 0.0,
            'long_tokens_count': 0,
            'long_tokens_percentage': 0.0,
            'language_token_counts': defaultdict(int),
            'tokens_by_length': defaultdict(int),
            'category_counts': defaultdict(int),
        }
        
        # Initialize category stats
        self.category_stats[name] = {
            'min_length': min_token_length,
            'categories': defaultdict(int),
            'total_tokens_analyzed': 0
        }
        
        for token, token_id in vocab.items():
            # Decode token
            try:
                decoded_text = tokenizer.decode([token_id], skip_special_tokens=False)
            except:
                decoded_text = token
            
            # Update statistics
            length = len(decoded_text)
            stats['tokens_by_length'][length] += 1
            
            if length > 32:
                stats['long_tokens_count'] += 1
            
            # Categorize token (if meets minimum length)
            if len(token) >= min_token_length:
                category = categorize_token(decoded_text, check_language)
                stats['category_counts'][category] += 1
                self.category_stats[name]['categories'][category] += 1
                self.category_stats[name]['total_tokens_analyzed'] += 1
            
            # Check for languages
            found_languages = set()
            english_chars = []
            
            try:
                for char in decoded_text:
                    has_lang, language = check_language(char)
                    if has_lang:
                        found_languages.add(language)
                    
                    if is_english_char(char):
                        english_chars.append(char)
            except:
                pass
            
            # Update language stats
            if found_languages:
                stats['total_multilingual_tokens'] += 1
                
                for lang in found_languages:
                    stats['language_token_counts'][lang] += 1
                    
                    # Count Indic tokens
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
        """Analyze all loaded tokenizers."""
        for name in self.tokenizers:
            print(f"\nAnalyzing {name}...")
            self.analyses[name] = self.analyze_tokenizer(name, min_token_length)
            print(f"  ✓ Analysis complete")
    
    def save_report(self, output_path: Path):
        """Save analysis to JSON."""
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert defaultdicts to regular dicts
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
        
        print(f"\n✓ Report saved to: {output_path}")
    
    def find_common_tokens(self, min_length: int, output_dir: Path):
        """Find common tokens across tokenizers."""
        print("\n" + "=" * 80)
        print(f"ANALYZING COMMON TOKENS (length >= {min_length})")
        print("=" * 80)
        
        # Collect all long tokens
        # Use list() to preserve insertion order (matches analyze_common_tokens.py behavior)
        tokenizer_names = list(self.tokenizers.keys())
        tokenizer_long_tokens = {}
        
        print(f"\nCollecting tokens with length >= {min_length} from each tokenizer:")
        for name in tokenizer_names:
            tokenizer = self.tokenizers[name]
            vocab = tokenizer.get_vocab()
            
            long_tokens = {}
            for token, token_id in vocab.items():
                if len(token) >= min_length:
                    long_tokens[token] = token_id
            
            tokenizer_long_tokens[name] = long_tokens
            print(f"  {name}: {len(long_tokens):,} tokens")
        
        # Find all unique long tokens across all tokenizers
        all_unique_tokens = set()
        for tokens_dict in tokenizer_long_tokens.values():
            all_unique_tokens.update(tokens_dict.keys())
        
        print(f"\n  Total unique long tokens across all tokenizers: {len(all_unique_tokens):,}")
        
        # Build comparison data
        comparison_data = []
        
        for token_value in sorted(all_unique_tokens):
            # Get decoded value from first tokenizer that has it
            decoded_text = None
            for name in tokenizer_names:
                if token_value in tokenizer_long_tokens[name]:
                    tokenizer = self.tokenizers[name]
                    token_id = tokenizer_long_tokens[name][token_value]
                    try:
                        decoded_text = tokenizer.decode([token_id], skip_special_tokens=False)
                        break
                    except:
                        pass
            
            # Categorize the token
            text_to_categorize = decoded_text if decoded_text else token_value
            category = categorize_token(text_to_categorize, check_language)
            
            # Detect languages in the decoded text
            detected_languages = []
            if decoded_text:
                for char in decoded_text:
                    is_lang, lang_name = check_language(char)
                    if is_lang and lang_name not in detected_languages:
                        detected_languages.append(lang_name)
            
            row = {
                'token_value': token_value,
                'decoded_value': decoded_text if decoded_text else token_value,
                'token_length': len(token_value),
                'decoded_length': len(decoded_text) if decoded_text else len(token_value),
                'category': category,
                'languages': '|'.join(detected_languages) if detected_languages else ''
            }
            
            # Add token IDs for each tokenizer (empty if not present)
            present_count = 0
            for tokenizer_name in tokenizer_names:
                if token_value in tokenizer_long_tokens[tokenizer_name]:
                    token_id = tokenizer_long_tokens[tokenizer_name][token_value]
                    row[f'{tokenizer_name}_token_id'] = token_id
                    present_count += 1
                else:
                    row[f'{tokenizer_name}_token_id'] = ''
            
            # Mark if present in all tokenizers
            row['present_in_all'] = 'Yes' if present_count == len(tokenizer_names) else 'No'
            row['present_in_count'] = present_count
            
            comparison_data.append(row)
        
        # Count common tokens and category breakdown
        common_tokens = sum(1 for row in comparison_data if row['present_in_all'] == 'Yes')
        print(f"  Tokens present in ALL tokenizers: {common_tokens}")
        
        # Show category breakdown
        category_counts = defaultdict(int)
        for row in comparison_data:
            category_counts[row['category']] += 1
        
        print(f"\n  Category breakdown:")
        for category in sorted(category_counts.keys()):
            print(f"    {category}: {category_counts[category]}")
        
        # Write to CSV with enhanced columns
        if comparison_data:
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"common_long_tokens_{min_length}plus_{timestamp}.csv"
            
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                # Build fieldnames dynamically
                fieldnames = ['token_value', 'decoded_value', 'token_length', 'decoded_length', 
                             'category', 'languages']
                for tokenizer_name in tokenizer_names:
                    fieldnames.append(f'{tokenizer_name}_token_id')
                fieldnames.extend(['present_in_all', 'present_in_count'])
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for row in comparison_data:
                    writer.writerow(row)
            
            print(f"\n✓ Exported comparison of {len(comparison_data)} long tokens to: {output_path}")
        else:
            print("\n✗ No long tokens found")
    
    def export_long_tokens(self, min_length: int, output_dir: Path):
        """Export all long tokens to CSV."""
        print("\n" + "=" * 80)
        print(f"EXPORTING LONG TOKENS (length >= {min_length})")
        print("=" * 80)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"long_tokens_{min_length}plus_{timestamp}.csv"
        
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['tokenizer_name', 'token_id', 'token_value', 'raw_token',
                         'token_length', 'decoded_length', 'category', 'languages']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            total_exported = 0
            
            for name, tokenizer in self.tokenizers.items():
                vocab = tokenizer.get_vocab()
                count = 0
                
                for token, token_id in vocab.items():
                    if len(token) >= min_length:
                        try:
                            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
                        except:
                            decoded = token
                        
                        category = categorize_token(decoded, check_language)
                        
                        languages = []
                        for char in decoded:
                            is_lang, lang = check_language(char)
                            if is_lang and lang not in languages:
                                languages.append(lang)
                        
                        writer.writerow({
                            'tokenizer_name': name,
                            'token_id': token_id,
                            'token_value': decoded,
                            'raw_token': token,
                            'token_length': len(token),
                            'decoded_length': len(decoded),
                            'category': category,
                            'languages': ', '.join(languages) if languages else 'none',
                        })
                        count += 1
                
                print(f"  ✓ {name}: {count:,} tokens")
                total_exported += count
        
        print(f"\n✓ Exported {total_exported:,} tokens total")
        print(f"✓ CSV saved to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Tokenizer Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick Examples:
  # Generate report
  python tokenizer_analyzer.py --report
  
  # Find common tokens >= 32 chars
  python tokenizer_analyzer.py --common-tokens --min-length 32
  
  # Complete analysis
  python tokenizer_analyzer.py --report --common-tokens --export-long-tokens --min-length 10
        """
    )
    
    # Analysis types
    analysis_group = parser.add_argument_group('Analysis Types (at least one required)')
    analysis_group.add_argument('--report', action='store_true',
                               help='Generate JSON report')
    analysis_group.add_argument('--common-tokens', action='store_true',
                               help='Find common tokens across tokenizers')
    analysis_group.add_argument('--export-long-tokens', action='store_true',
                               help='Export all long tokens to CSV')
    
    # Filtering
    filter_group = parser.add_argument_group('Filtering')
    filter_group.add_argument('--min-length', '-m', type=int, default=0,
                             help='Minimum token length (default: 0)')
    
    # Directories
    dir_group = parser.add_argument_group('Directories')
    dir_group.add_argument('--input-dir', '-i', default='../data',
                          help='Input directory (default: ../data)')
    dir_group.add_argument('--output-dir', '-o', default='tokenizer_results',
                          help='Output directory (default: tokenizer_results/)')
    
    # Output files
    output_group = parser.add_argument_group('Output Files')
    output_group.add_argument('--output', help='Custom JSON report filename')
    
    args = parser.parse_args()
    
    # Validate: at least one analysis type required
    if not any([args.report, args.common_tokens, args.export_long_tokens]):
        parser.error("At least one analysis type required: --report, --common-tokens, or --export-long-tokens")
    
    print("=" * 80)
    print("UNIFIED TOKENIZER ANALYZER")
    print("=" * 80)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    if args.min_length > 0:
        print(f"Filter: tokens with length >= {args.min_length}")
    
    # Create analyzer
    analyzer = TokenizerAnalyzer()
    
    # Load tokenizers
    print("\nLoading tokenizers...")
    print("-" * 80)
    input_dir = Path(args.input_dir)
    loaded = analyzer.load_tokenizers_from_directory(input_dir)
    
    if loaded == 0:
        print("\n✗ No tokenizers loaded")
        return 1
    
    print(f"\n✓ Loaded {loaded} tokenizer(s)")
    
    output_dir = Path(args.output_dir)
    
    # Run requested analyses
    if args.report:
        print("\n" + "=" * 80)
        print("ANALYZING TOKENIZERS")
        print("=" * 80)
        analyzer.analyze_all(min_token_length=args.min_length)
        output_file = args.output if args.output else "tokenizer_analysis_report.json"
        output_path = output_dir / output_file
        analyzer.save_report(output_path)
    
    if args.common_tokens:
        analyzer.find_common_tokens(args.min_length, output_dir)
        
        # Print detailed column explanation (matches analyze_common_tokens.py)
        print("=" * 80)
        print("\nSuccess! CSV file created with the following columns:")
        print("  - token_value: The raw token text from vocabulary")
        print("  - decoded_value: Properly decoded token (handling byte-level BPE)")
        print("  - token_length: Length of the raw token")
        print("  - decoded_length: Length of the decoded text")
        print("  - category: Token category (english, code, indic, east_asian, middle_eastern, etc.)")
        print("  - languages: Detected languages (pipe-separated, e.g., 'Chinese|Japanese')")
        for name in sorted(analyzer.tokenizers.keys()):
            print(f"  - {name}_token_id: Token ID in {name} (empty if not present)")
        print("  - present_in_all: Whether token exists in all tokenizers")
        print("  - present_in_count: Number of tokenizers containing the token")
        print()
    
    if args.export_long_tokens:
        analyzer.export_long_tokens(args.min_length, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
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
