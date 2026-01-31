"""
Special Tokens - Definitions and utilities for special token handling.

Provides utilities for:
- Special token definitions and management
- Encoding/decoding with special tokens
- Validation and testing

Usage:
    from special_tokens import SpecialTokenRegistry

    registry = SpecialTokenRegistry("../config.yaml")
    tokens = registry.get_all_special_tokens()
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class SpecialToken:
    """Container for a special token."""
    name: str
    token: str
    id: int
    category: str
    description: Optional[str] = None


class SpecialTokenRegistry:
    """Registry for managing special tokens."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.tokens = self._load_tokens()
        self.token_to_id = {t.token: t.id for t in self.tokens}
        self.id_to_token = {t.id: t.token for t in self.tokens}
        self.name_to_token = {t.name: t for t in self.tokens}

    def _load_tokens(self) -> List[SpecialToken]:
        """Load all special tokens from config."""
        tokens = []

        for category, token_list in self.config['special_tokens'].items():
            for token_def in token_list:
                token = SpecialToken(
                    name=token_def['name'],
                    token=token_def['token'],
                    id=token_def['id'],
                    category=category,
                    description=token_def.get('description', '')
                )
                tokens.append(token)

        logger.info(f"Loaded {len(tokens)} special tokens from config")
        return tokens

    def get_all_special_tokens(self) -> List[SpecialToken]:
        """Get all special tokens."""
        return self.tokens

    def get_by_category(self, category: str) -> List[SpecialToken]:
        """Get special tokens by category."""
        return [t for t in self.tokens if t.category == category]

    def get_by_name(self, name: str) -> Optional[SpecialToken]:
        """Get special token by name."""
        return self.name_to_token.get(name)

    def get_by_id(self, token_id: int) -> Optional[SpecialToken]:
        """Get special token by ID."""
        return self.id_to_token.get(token_id)

    def is_special_token(self, token: str) -> bool:
        """Check if a token string is a special token."""
        return token in self.token_to_id

    def get_token_id(self, token: str) -> Optional[int]:
        """Get ID for a special token string."""
        return self.token_to_id.get(token)

    def get_token_string(self, token_id: int) -> Optional[str]:
        """Get token string for a special token ID."""
        return self.id_to_token.get(token_id)

    def get_reserved_id_range(self) -> tuple:
        """Get the range of reserved IDs for special tokens."""
        if not self.tokens:
            return (0, 0)
        return (min(t.id for t in self.tokens), max(t.id for t in self.tokens))

    def validate_tokenizer(self, tokenizer_vocab: Dict[str, int]) -> Dict[str, Any]:
        """
        Validate that special tokens are properly included in tokenizer.

        Returns validation report with:
        - missing_tokens: special tokens not in vocab
        - id_conflicts: tokens with mismatched IDs
        - present_tokens: special tokens correctly included
        """
        missing_tokens = []
        id_conflicts = []
        present_tokens = []

        for token in self.tokens:
            if token.token not in tokenizer_vocab:
                missing_tokens.append(token)
            elif tokenizer_vocab[token.token] != token.id:
                id_conflicts.append({
                    'token': token,
                    'expected_id': token.id,
                    'actual_id': tokenizer_vocab[token.token]
                })
            else:
                present_tokens.append(token)

        report = {
            'valid': len(missing_tokens) == 0 and len(id_conflicts) == 0,
            'total_special_tokens': len(self.tokens),
            'present_tokens': len(present_tokens),
            'missing_tokens': [{'name': t.name, 'token': t.token, 'id': t.id} for t in missing_tokens],
            'id_conflicts': id_conflicts
        }

        if report['valid']:
            logger.info("✓ All special tokens validated successfully")
        else:
            logger.warning(f"⚠ Validation issues: {len(missing_tokens)} missing, {len(id_conflicts)} conflicts")

        return report

    def export_special_tokens_config(self, output_path: str):
        """Export special tokens for use with other tokenizer libraries."""
        output = {
            'special_tokens': [
                {
                    'name': t.name,
                    'token': t.token,
                    'id': t.id,
                    'category': t.category
                }
                for t in self.tokens
            ],
            'token_to_id': self.token_to_id,
            'id_to_token': {str(k): v for k, v in self.id_to_token.items()},
            'reserved_range': {
                'min': self.get_reserved_id_range()[0],
                'max': self.get_reserved_id_range()[1]
            }
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        import json
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Exported special tokens config to: {output_file}")

    def print_summary(self):
        """Print a summary of all special tokens."""
        print("\n" + "="*80)
        print("SPECIAL TOKENS SUMMARY")
        print("="*80)

        categories = {}
        for token in self.tokens:
            if token.category not in categories:
                categories[token.category] = []
            categories[token.category].append(token)

        for category, tokens in categories.items():
            print(f"\n{category.upper()} ({len(tokens)} tokens):")
            print("-" * 80)
            for token in sorted(tokens, key=lambda t: t.id):
                print(f"  ID {token.id:3d}: {token.token:25s} ({token.name})")

        print("\n" + "="*80)
        print(f"Total: {len(self.tokens)} special tokens")
        min_id, max_id = self.get_reserved_id_range()
        print(f"Reserved ID range: {min_id} - {max_id}")
        print("="*80 + "\n")


class SpecialTokenEncoder:
    """Helper for encoding/decoding text with special tokens."""

    def __init__(self, registry: SpecialTokenRegistry):
        self.registry = registry

    def wrap_document(self, text: str) -> str:
        """Wrap text with document begin/end tokens."""
        begin = self.registry.get_by_name('begin_of_text')
        end = self.registry.get_by_name('end_of_text')

        if begin and end:
            return f"{begin.token}{text}{end.token}"
        return text

    def wrap_chat_message(self, role: str, content: str) -> str:
        """Wrap a chat message with role tokens."""
        role_token = self.registry.get_by_name(role)

        if role_token:
            return f"{role_token.token}{content}"
        return content

    def wrap_code_block(self, code: str, language: str = "python") -> str:
        """Wrap code with code block and language tokens."""
        code_begin = self.registry.get_by_name('code_begin')
        code_end = self.registry.get_by_name('code_end')
        lang_token = self.registry.get_by_name(f'lang_{language}')

        if code_begin and code_end:
            lang_prefix = lang_token.token if lang_token else ""
            return f"{code_begin.token}{lang_prefix}{code}{code_end.token}"
        return code

    def wrap_json(self, json_str: str) -> str:
        """Wrap JSON with JSON envelope tokens."""
        json_begin = self.registry.get_by_name('json_begin')
        json_end = self.registry.get_by_name('json_end')

        if json_begin and json_end:
            return f"{json_begin.token}{json_str}{json_end.token}"
        return json_str

    def wrap_tool_call(self, tool_call: str) -> str:
        """Wrap a tool call with tool call token."""
        tool_call_token = self.registry.get_by_name('tool_call')

        if tool_call_token:
            return f"{tool_call_token.token}{tool_call}"
        return tool_call

    def wrap_tool_result(self, result: str) -> str:
        """Wrap a tool result with tool result token."""
        tool_result_token = self.registry.get_by_name('tool_result')

        if tool_result_token:
            return f"{tool_result_token.token}{result}"
        return result

    def add_source_tag(self, text: str, source: str) -> str:
        """Add source metadata tag to text."""
        source_token = self.registry.get_by_name(f'source_{source}')

        if source_token:
            return f"{source_token.token}{text}"
        return text


def main():
    """Demo and testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Special Tokens Utility")
    parser.add_argument('--config', type=str, default='../config.yaml',
                        help='Path to config file')
    parser.add_argument('--export', type=str, default=None,
                        help='Export special tokens config to file')
    parser.add_argument('--validate', type=str, default=None,
                        help='Validate tokenizer vocab file (JSON)')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo')

    args = parser.parse_args()

    # Load registry
    registry = SpecialTokenRegistry(args.config)

    # Print summary
    registry.print_summary()

    # Export if requested
    if args.export:
        registry.export_special_tokens_config(args.export)

    # Validate if requested
    if args.validate:
        import json
        with open(args.validate, 'r') as f:
            vocab = json.load(f)
        report = registry.validate_tokenizer(vocab)
        print("\nValidation Report:")
        print(json.dumps(report, indent=2))

    # Demo if requested
    if args.demo:
        encoder = SpecialTokenEncoder(registry)

        print("\n" + "="*80)
        print("SPECIAL TOKEN ENCODING DEMO")
        print("="*80)

        # Document wrapping
        doc = encoder.wrap_document("This is a test document.")
        print(f"\nDocument wrapping:\n{doc}")

        # Chat message
        msg = encoder.wrap_chat_message("user", "Hello, how are you?")
        print(f"\nChat message:\n{msg}")

        # Code block
        code = encoder.wrap_code_block("def hello():\n    print('Hello')", "python")
        print(f"\nCode block:\n{code}")

        # JSON
        json_data = encoder.wrap_json('{"key": "value"}')
        print(f"\nJSON:\n{json_data}")

        # Source tag
        tagged = encoder.add_source_tag("Wikipedia article text...", "wikipedia")
        print(f"\nSource tagged:\n{tagged}")

        print("\n" + "="*80)


if __name__ == "__main__":
    main()
