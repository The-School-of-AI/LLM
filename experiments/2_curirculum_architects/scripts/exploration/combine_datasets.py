"""
Dataset Combination Script
===================================
Combines smaller datasets to meet the minimum 4K token requirement.

Configurable strategies for merge:
1. domain  - Group by metadata['domain']
2. band    - Group by metadata['band'] (B0-B5)
3. strict  - Group by (band, domain)
4. discard - Filter out samples < min_tokens

Usage:
    python combine_datasets.py --config packing_config.yaml
"""

import argparse
import json
from typing import Any, Dict, Generator

import yaml


class MockTokenizer:
    """
    Mock tokenizer for development (proxy for Team 6).
    Estimates ~1 token per 4 characters.
    """

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    @property
    def eos_token(self) -> str:
        return "<EOS>"


class LinearPacker:
    """Packs documents until reaching min_tokens threshold."""

    def __init__(self, tokenizer: MockTokenizer, min_tokens: int, strategy: str):
        self.tokenizer = tokenizer
        self.min_tokens = min_tokens
        self.strategy = strategy
        self.buffers: Dict[Any, Dict[str, Any]] = {}

    def _get_group_key(self, metadata: Dict[str, Any]) -> Any:
        if self.strategy == "domain":
            return metadata.get("domain", "unknown")
        elif self.strategy == "band":
            return metadata.get("band", "unknown")
        elif self.strategy == "strict":
            return (metadata.get("band", "unknown"), metadata.get("domain", "unknown"))
        return "global"

    def process(self, input_stream: Generator) -> Generator[Dict[str, Any], None, None]:
        if self.strategy == "discard":
            yield from self._process_discard(input_stream)
        else:
            yield from self._process_packing(input_stream)

    def _process_discard(self, input_stream):
        """Strategy 4: Discard samples below min_tokens."""
        for item in input_stream:
            text = item.get("text", "")
            if self.tokenizer.count_tokens(text) >= self.min_tokens:
                yield item

    def _process_packing(self, input_stream):
        """Strategies 1-3: Pack documents until threshold reached."""
        for item in input_stream:
            text = item.get("text", "")
            metadata = item.get("metadata", {})

            if not text:
                continue

            key = self._get_group_key(metadata)
            token_count = self.tokenizer.count_tokens(text)

            if key not in self.buffers:
                self.buffers[key] = {
                    "text_parts": [],
                    "current_tokens": 0,
                    "source_ids": [],
                    "reference_metadata": metadata.copy(),
                }

            buf = self.buffers[key]

            # Add separator if buffer not empty
            separator = f" {self.tokenizer.eos_token} " if buf["text_parts"] else ""
            sep_tokens = self.tokenizer.count_tokens(separator)

            buf["text_parts"].append(separator + text)
            buf["current_tokens"] += sep_tokens + token_count

            if "source_id" in metadata:
                buf["source_ids"].append(metadata["source_id"])

            # Flush if threshold reached
            if buf["current_tokens"] >= self.min_tokens:
                yield self._flush_buffer(key)

        # Flush remaining buffers
        for key in list(self.buffers.keys()):
            if self.buffers[key]["text_parts"]:
                yield self._flush_buffer(key)

    def _flush_buffer(self, key) -> Dict[str, Any]:
        buf = self.buffers[key]
        packed_text = "".join(buf["text_parts"])

        out_metadata = buf["reference_metadata"].copy()
        out_metadata["source_ids"] = buf["source_ids"]
        out_metadata.pop("source_id", None)

        del self.buffers[key]

        return {"text": packed_text, "metadata": out_metadata}


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(config: Dict[str, Any]):
    strategy = config.get("strategy", "strict")
    min_tokens = config.get("min_tokens", 4096)
    input_file = config.get("input_file")
    output_file = config.get("output_file")

    print(f"Strategy: {strategy} | Min Tokens: {min_tokens}")

    tokenizer = MockTokenizer()
    packer = LinearPacker(tokenizer, min_tokens, strategy)

    def read_input():
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    count_out = 0
    with open(output_file, "w", encoding="utf-8") as f_out:
        for packed_item in packer.process(read_input()):
            f_out.write(json.dumps(packed_item) + "\n")
            count_out += 1

    print(f"Done. Generated {count_out} sequences -> {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Team 2 Dataset Combiner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(config)
