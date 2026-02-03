"""
Tokenizer Loader - Unified interface for loading tokenizers.

Supports:
- Custom JSON tokenizers
- HuggingFace tokenizers
- tiktoken (GPT models)
- SentencePiece models
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Union


class TokenizerInterface(ABC):
    """Abstract base class for all tokenizer implementations."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        pass
    
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        pass
    
    def tokenize(self, text: str) -> List[str]:
        """Return list of token strings (optional)."""
        token_ids = self.encode(text)
        return [self.decode([tid]) for tid in token_ids]
    
    def get_stats(self, text: str) -> Dict[str, Any]:
        """Get tokenization statistics for a given text."""
        tokens = self.encode(text)
        return {
            "num_tokens": len(tokens),
            "num_chars": len(text),
            "num_bytes": len(text.encode('utf-8')),
            "tokens_per_char": len(tokens) / max(len(text), 1),
            "tokens_per_byte": len(tokens) / max(len(text.encode('utf-8')), 1),
        }


class CustomJSONTokenizer(TokenizerInterface):
    """
    Custom tokenizer loaded from JSON format.
    
    Expected JSON structure:
    {
        "vocab": {"token": id, ...},
        "merges": [...],  # Optional BPE merges
        "special_tokens": {...},  # Optional
        "config": {...}  # Optional configuration
    }
    """
    
    def __init__(self, name: str, json_path: Union[str, Path]):
        super().__init__(name)
        self.json_path = Path(json_path)
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load tokenizer from JSON file."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {self.json_path}")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data.get('vocab', {})
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merges = data.get('merges', [])
        self.special_tokens = data.get('special_tokens', {})
        self.config = data.get('config', {})
        
        # Build merge priority dict if merges exist
        self.merge_priority = {tuple(m.split()): i for i, m in enumerate(self.merges)}
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text using BPE algorithm.
        Falls back to byte-level if token not in vocab.
        """
        if not text:
            return []
        
        # Start with character-level tokens
        tokens = list(text)
        
        # Apply BPE merges iteratively
        while len(tokens) > 1:
            # Find the pair with highest priority (lowest index)
            best_pair = None
            best_priority = float('inf')
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_priority:
                    priority = self.merge_priority[pair]
                    if priority < best_priority:
                        best_priority = priority
                        best_pair = (i, pair)
            
            if best_pair is None:
                break
            
            # Apply the merge
            idx, (t1, t2) = best_pair
            merged = t1 + t2
            tokens = tokens[:idx] + [merged] + tokens[idx + 2:]
        
        # Convert tokens to IDs
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # Fallback: encode as bytes
                for byte in token.encode('utf-8'):
                    byte_token = f"<0x{byte:02X}>"
                    if byte_token in self.vocab:
                        token_ids.append(self.vocab[byte_token])
                    else:
                        # Ultimate fallback: unknown token
                        unk_id = self.special_tokens.get('unk', 0)
                        token_ids.append(unk_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        tokens = []
        for tid in token_ids:
            if tid in self.id_to_token:
                token = self.id_to_token[tid]
                # Handle byte tokens
                if token.startswith('<0x') and token.endswith('>'):
                    try:
                        byte_val = int(token[3:-1], 16)
                        tokens.append(bytes([byte_val]).decode('utf-8', errors='replace'))
                    except:
                        tokens.append(token)
                else:
                    tokens.append(token)
        return ''.join(tokens)
    
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)


class HuggingFaceTokenizer(TokenizerInterface):
    """Wrapper for HuggingFace tokenizers."""
    
    def __init__(self, name: str, model_name: str):
        super().__init__(name)
        self.model_name = model_name
        self._tokenizer = None
    
    def _lazy_load(self):
        """Lazy load the tokenizer on first use."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
            except ImportError:
                raise ImportError("Please install transformers: pip install transformers")
            except Exception as e:
                raise RuntimeError(f"Failed to load tokenizer {self.model_name}: {e}")
    
    def encode(self, text: str) -> List[int]:
        self._lazy_load()
        return self._tokenizer.encode(text, add_special_tokens=False)
    
    def decode(self, token_ids: List[int]) -> str:
        self._lazy_load()
        return self._tokenizer.decode(token_ids)
    
    def vocab_size(self) -> int:
        self._lazy_load()
        return len(self._tokenizer)


class TiktokenTokenizer(TokenizerInterface):
    """Wrapper for tiktoken (OpenAI's tokenizer)."""
    
    def __init__(self, name: str, model: str = "gpt-4o"):
        super().__init__(name)
        self.model = model
        self._encoding = None
    
    def _lazy_load(self):
        """Lazy load tiktoken on first use."""
        if self._encoding is None:
            try:
                import tiktoken
                self._encoding = tiktoken.encoding_for_model(self.model)
            except ImportError:
                raise ImportError("Please install tiktoken: pip install tiktoken")
            except Exception as e:
                raise RuntimeError(f"Failed to load tiktoken for {self.model}: {e}")
    
    def encode(self, text: str) -> List[int]:
        self._lazy_load()
        return self._encoding.encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        self._lazy_load()
        return self._encoding.decode(token_ids)
    
    def vocab_size(self) -> int:
        self._lazy_load()
        return self._encoding.n_vocab


def load_tokenizer(
    name: str,
    backend: str,
    path: Optional[str] = None,
    model: Optional[str] = None
) -> TokenizerInterface:
    """
    Factory function to load a tokenizer.
    
    Args:
        name: Display name for the tokenizer
        backend: One of 'json', 'huggingface', 'tiktoken', 'sentencepiece'
        path: Path to tokenizer file (for json/sentencepiece)
        model: Model name (for huggingface/tiktoken)
    
    Returns:
        TokenizerInterface instance
    """
    backend = backend.lower()
    
    if backend == 'json':
        if not path:
            raise ValueError("Path required for JSON tokenizer")
        return CustomJSONTokenizer(name, path)
    
    elif backend == 'huggingface':
        if not model:
            raise ValueError("Model name required for HuggingFace tokenizer")
        return HuggingFaceTokenizer(name, model)
    
    elif backend == 'tiktoken':
        model = model or 'gpt-4o'
        return TiktokenTokenizer(name, model)
    
    else:
        raise ValueError(f"Unknown backend: {backend}")


# Convenience function to load all tokenizers from config
def load_tokenizers_from_config(config: Dict[str, Any]) -> Dict[str, TokenizerInterface]:
    """Load all tokenizers defined in the configuration."""
    tokenizers = {}
    
    # Load custom tokenizer if defined
    if 'custom_tokenizer' in config:
        custom = config['custom_tokenizer']
        try:
            tokenizers['custom'] = load_tokenizer(
                name=custom.get('name', 'custom'),
                backend=custom.get('type', 'json'),
                path=custom.get('path')
            )
        except FileNotFoundError:
            print(f"Warning: Custom tokenizer not found at {custom.get('path')}")
    
    # Load reference tokenizers
    if 'reference_tokenizers' in config:
        for key, ref in config['reference_tokenizers'].items():
            try:
                tokenizers[key] = load_tokenizer(
                    name=ref.get('name', key),
                    backend=ref.get('backend', 'huggingface'),
                    path=ref.get('path'),
                    model=ref.get('model')
                )
            except Exception as e:
                print(f"Warning: Failed to load {key}: {e}")
    
    return tokenizers
