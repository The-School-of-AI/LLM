#!/usr/bin/env python3
"""
Smoke test for the data cleaning pipeline.

Exercises EVERY cleaning path in clean.py against real data:
  - 10% sample from Tokenizer/indic_tokenizer_samples_by_size (B0-B2 Indic)
  - Downloaded code files: Python, C++, JS (B3-B5 code)
  - Downloaded arxiv papers (B4 academic)
  - Local lock files, minified JS, autogen files from ~/TSAI, ~/Documents
  - Synthetic docs for patterns not found in real data (reference sections)
  - C4 English web (streamed from HuggingFace, optional)
  - open-web-math (streamed from HuggingFace, optional)

Reports per-source stats showing which cleaning steps fired, with
before/after examples saved to test_cleaning_examples/.

Usage:
    python test_cleaning_smoke.py                    # local data only (fast)
    python test_cleaning_smoke.py --include-hf       # + stream C4 & open-web-math
    python test_cleaning_smoke.py --quick             # 1% sample instead of 10%
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Optional

# Add pipeline dir to path
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

from clean import CleaningStats, clean_text, is_valid_document

# ═══════════════════════════════════════════════════════════════════════════
# PATHS — edit these if your layout differs
# ═══════════════════════════════════════════════════════════════════════════
REPO_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, "..", ".."))
TOKENIZER_SAMPLES = os.path.join(
    REPO_ROOT, "Tokenizer", "indic_tokenizer_samples_by_size"
)
DOWNLOADS_DIR = os.path.join(REPO_ROOT, "DataSet", "downloads")
EXAMPLES_DIR = os.path.join(PIPELINE_DIR, "test_cleaning_examples")

# Local directories to scan for lockfiles, minified JS, autogen files
LOCAL_SCAN_DIRS = [
    os.path.expanduser("~/TSAI"),
    os.path.expanduser("~/Documents"),
]
LOCAL_SCAN_MAXDEPTH = 5

RANDOM_SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════


def load_tokenizer_samples(sample_frac: float = 0.1) -> Dict[str, List[str]]:
    """
    Load 10% of parquet files from Tokenizer samples, grouped by source.
    Returns {source_name: [text, text, ...]}.
    """
    import pyarrow.parquet as pq

    if not os.path.isdir(TOKENIZER_SAMPLES):
        print(f"  SKIP: {TOKENIZER_SAMPLES} not found")
        return {}

    source_dirs = sorted(glob.glob(os.path.join(TOKENIZER_SAMPLES, "source=*")))
    result: Dict[str, List[str]] = {}
    total_files = 0
    total_docs = 0

    for sd in source_dirs:
        source = os.path.basename(sd).replace("source=", "")
        parquets = sorted(glob.glob(os.path.join(sd, "*.parquet")))
        if not parquets:
            continue

        # Sample fraction of files (at least 1)
        n_sample = max(1, int(len(parquets) * sample_frac))
        random.seed(RANDOM_SEED)
        sampled = random.sample(parquets, n_sample)
        total_files += n_sample

        texts = []
        for pf in sampled:
            try:
                schema = pq.read_schema(pf)
                t = pq.read_table(pf, schema=schema)
                if "text" in t.column_names:
                    for val in t["text"]:
                        s = val.as_py()
                        if s:
                            texts.append(s)
            except Exception as e:
                print(f"  WARN: Error reading {pf}: {e}")

        if texts:
            result[source] = texts
            total_docs += len(texts)

    print(
        f"  Tokenizer samples: {len(result)} sources, {total_files} files, {total_docs:,} docs"
    )
    return result


def load_downloaded_code(max_per_lang: int = 1000) -> Dict[str, List[str]]:
    """Load downloaded code JSONL files from DataSet/downloads/."""
    result: Dict[str, List[str]] = {}

    file_map = {
        "code_python": "python_data.json",
        "code_cpp": "cpp_data.json",
        "code_javascript": "javeascript_data.json",
    }

    for source_name, filename in file_map.items():
        path = os.path.join(DOWNLOADS_DIR, filename)
        if not os.path.exists(path):
            print(f"  SKIP: {path} not found")
            continue

        texts = []
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= max_per_lang:
                    break
                row = json.loads(line)
                content = row.get("content", "")
                if content:
                    texts.append(content)

        result[source_name] = texts
        print(f"  {source_name}: {len(texts)} docs from {filename}")

    return result


def load_downloaded_arxiv(max_docs: int = 500) -> Dict[str, List[str]]:
    """Load downloaded arxiv parquet from DataSet/downloads/."""
    import pyarrow.parquet as pq

    parquet_files = glob.glob(os.path.join(DOWNLOADS_DIR, "arxiv*.parquet"))
    if not parquet_files:
        print("  SKIP: No arxiv parquet found in downloads/")
        return {}

    t = pq.read_table(parquet_files[0])
    texts = []
    for i in range(min(max_docs, t.num_rows)):
        article = t["article"][i].as_py()
        if article:
            texts.append(article)

    print(f"  arxiv: {len(texts)} docs from {os.path.basename(parquet_files[0])}")
    return {"arxiv_papers": texts}


def load_local_lockfiles(max_files: int = 10) -> Dict[str, List[str]]:
    """Find and load real lock files from local filesystem."""
    texts = []
    found = []

    for scan_dir in LOCAL_SCAN_DIRS:
        if not os.path.isdir(scan_dir):
            continue
        for root, dirs, files in os.walk(scan_dir):
            # Respect maxdepth
            depth = root.replace(scan_dir, "").count(os.sep)
            if depth >= LOCAL_SCAN_MAXDEPTH:
                dirs.clear()
                continue
            for fname in files:
                if fname in (
                    "package-lock.json",
                    "yarn.lock",
                    "Pipfile.lock",
                    "poetry.lock",
                ):
                    fpath = os.path.join(root, fname)
                    try:
                        size = os.path.getsize(fpath)
                        if size > 1024 and size < 5_000_000:  # 1KB - 5MB
                            with open(fpath, errors="replace") as f:
                                # Read first 10K chars (enough to trigger detection)
                                content = f.read(10000)
                            texts.append(content)
                            found.append(f"{fname} ({size // 1024}KB)")
                            if len(texts) >= max_files:
                                break
                    except (OSError, PermissionError):
                        pass
            if len(texts) >= max_files:
                break

    if texts:
        print(f"  lockfiles: {len(texts)} files — {', '.join(found[:5])}")
    return {"local_lockfiles": texts} if texts else {}


def load_local_minified(max_files: int = 10) -> Dict[str, List[str]]:
    """Find and load real minified JS files from local filesystem."""
    texts = []
    found = []

    for scan_dir in LOCAL_SCAN_DIRS:
        if not os.path.isdir(scan_dir):
            continue
        for root, dirs, files in os.walk(scan_dir):
            depth = root.replace(scan_dir, "").count(os.sep)
            if depth >= LOCAL_SCAN_MAXDEPTH:
                dirs.clear()
                continue
            for fname in files:
                if fname.endswith(".min.js") or fname.endswith(".min.css"):
                    fpath = os.path.join(root, fname)
                    try:
                        size = os.path.getsize(fpath)
                        if size > 1024:
                            with open(fpath, errors="replace") as f:
                                content = f.read(50000)  # first 50K
                            texts.append(content)
                            found.append(f"{fname} ({size // 1024}KB)")
                            if len(texts) >= max_files:
                                break
                    except (OSError, PermissionError):
                        pass
            if len(texts) >= max_files:
                break

    if texts:
        print(f"  minified: {len(texts)} files — {', '.join(found[:5])}")
    return {"local_minified": texts} if texts else {}


def load_local_autogen(max_files: int = 10) -> Dict[str, List[str]]:
    """Find and load real auto-generated Python files from local filesystem."""
    import re

    autogen_re = re.compile(
        r"DO NOT EDIT|auto.?generated|Generated by|THIS FILE IS GENERATED",
        re.IGNORECASE,
    )

    texts = []
    found = []

    for scan_dir in LOCAL_SCAN_DIRS:
        if not os.path.isdir(scan_dir):
            continue
        for root, dirs, files in os.walk(scan_dir):
            depth = root.replace(scan_dir, "").count(os.sep)
            if depth >= LOCAL_SCAN_MAXDEPTH:
                dirs.clear()
                continue
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, errors="replace") as f:
                        head = f.read(2000)
                    if autogen_re.search(head):
                        texts.append(head)
                        found.append(fname)
                        if len(texts) >= max_files:
                            break
                except (OSError, PermissionError):
                    pass
            if len(texts) >= max_files:
                break

    if texts:
        print(f"  autogen: {len(texts)} files — {', '.join(found[:5])}")
    return {"local_autogen": texts} if texts else {}


def create_synthetic_docs() -> Dict[str, List[str]]:
    """
    Synthetic documents that exercise cleaning paths not covered by real data.
    These serve as unit tests embedded in the smoke test.
    """
    docs = []

    # 1. Academic paper with References section
    docs.append(
        "This paper presents a novel approach to neural network pruning.\n"
        "We demonstrate that structured pruning achieves 3x speedup with "
        "minimal accuracy loss on ImageNet[1]. Our method builds on prior "
        "work by Han et al.[2] and extends the lottery ticket hypothesis[3].\n"
        "Results show consistent improvement across ResNet[4], VGG[5], and "
        "EfficientNet[6] architectures.\n"
        "The key insight is that weight magnitude alone[7][8] is insufficient; "
        "gradient information[9] provides a more reliable pruning signal.\n\n"
        "References\n"
        "1. K. He et al. Deep residual learning for image recognition. CVPR 2016.\n"
        "2. S. Han et al. Learning both weights and connections. NeurIPS 2015.\n"
        "3. J. Frankle and M. Carlin. The lottery ticket hypothesis. ICLR 2019.\n"
        "4. K. He et al. Deep residual learning. CVPR 2016.\n"
        "5. K. Simonyan and A. Zisserman. Very deep convolutional networks. ICLR 2015.\n"
        "6. M. Tan and Q. Le. EfficientNet. ICML 2019.\n"
        "7. Y. LeCun et al. Optimal brain damage. NeurIPS 1989.\n"
        "8. B. Hassibi and D. Stork. Second order derivatives. NeurIPS 1992.\n"
        "9. P. Molchanov et al. Pruning convolutional neural networks. ICLR 2017.\n"
    )

    # 2. Paper with Bibliography heading (non-English)
    docs.append(
        "इस शोध पत्र में हम हिंदी भाषा प्रसंस्करण के लिए एक नवीन दृष्टिकोण प्रस्तुत करते हैं।\n"
        "हमारे परिणाम दर्शाते हैं कि ट्रांसफॉर्मर आधारित मॉडल[1] हिंदी NLP कार्यों में "
        "उत्कृष्ट प्रदर्शन करते हैं[2][3]।\n\n"
        "संदर्भ\n"
        "1. Vaswani et al. Attention is all you need. NeurIPS 2017.\n"
        "2. Devlin et al. BERT. NAACL 2019.\n"
        "3. Conneau et al. Unsupervised cross-lingual representation learning. ACL 2019.\n"
    )

    # 3. Document with ghost tags (Samvaad format)
    docs.append(
        "[USER] What is the capital of France?\n\n"
        "[ASSISTANT] The capital of France is Paris. It is located in the north-central "
        "part of the country along the Seine River.\n\n"
        "[USER] Tell me more about Paris.\n\n"
        "[ASSISTANT] Paris is known for the Eiffel Tower, the Louvre Museum, and "
        "its rich history dating back to the 3rd century BCE."
    )

    # 4. Document with XML ghost tags (SmolTalk2 format)
    docs.append(
        "<USER>\nHow do neural networks work?\n</USER>\n"
        "<ASSISTANT>\nNeural networks are computational models inspired by "
        "biological neurons. They consist of layers of interconnected nodes "
        "that process information through weighted connections.\n</ASSISTANT>"
    )

    # 5. Document with ChatML ghost tags
    docs.append(
        "<|system|>You are a helpful assistant.\n"
        "<|user|>Explain gradient descent.\n"
        "<|assistant|>Gradient descent is an optimization algorithm used to "
        "minimize a function by iteratively moving in the direction of steepest "
        "descent as defined by the negative of the gradient."
    )

    # 6. Anthropic-style conversation markers
    docs.append(
        "\nHuman: What is machine learning?\n\n"
        "Assistant: Machine learning is a subset of artificial intelligence that "
        "focuses on building systems that learn from data. Rather than being "
        "explicitly programmed, these systems identify patterns in data and make "
        "decisions with minimal human intervention.\n\n"
        "Human: Can you give an example?\n\n"
        "Assistant: A common example is email spam filtering. The system learns "
        "from thousands of labeled emails what constitutes spam versus legitimate "
        "mail, and then applies those patterns to new incoming emails."
    )

    # 7. Python file with Apache license header
    docs.append(
        "# Copyright 2024 The Example Authors.\n"
        "#\n"
        '# Licensed under the Apache License, Version 2.0 (the "License");\n'
        "# you may not use this file except in compliance with the License.\n"
        "# You may obtain a copy of the License at\n"
        "#\n"
        "#     http://www.apache.org/licenses/LICENSE-2.0\n"
        "#\n"
        "# Unless required by applicable law or agreed to in writing, software\n"
        '# distributed under the License is distributed on an "AS IS" BASIS,\n'
        "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
        "#\n"
        "\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "\n"
        "class SimpleModel(nn.Module):\n"
        "    def __init__(self, hidden_size=256):\n"
        "        super().__init__()\n"
        "        self.linear = nn.Linear(hidden_size, hidden_size)\n"
        "        self.relu = nn.ReLU()\n"
        "\n"
        "    def forward(self, x):\n"
        "        return self.relu(self.linear(x))\n"
    )

    # 8. Auto-generated Python file (should be dropped entirely)
    docs.append(
        "# This file is auto-generated by protoc-gen-python. DO NOT EDIT!\n"
        "# source: example.proto\n"
        "\n"
        "from google.protobuf import descriptor as _descriptor\n"
        "from google.protobuf import message as _message\n"
        "\n"
        "DESCRIPTOR = _descriptor.FileDescriptor(\n"
        '    name="example.proto",\n'
        '    package="example",\n'
        ")\n"
        "\n"
        "class ExampleMessage(_message.Message):\n"
        "    DESCRIPTOR = DESCRIPTOR\n"
    )

    # 9. Minified JS (should be dropped entirely)
    docs.append(
        '!function(e,t){"use strict";var n=function(){function e(e,t){for(var n=0;n<t.length;n++)'
        '{var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),'
        "Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}();"
        'var i=function(){function e(){this._listeners={}}return n(e,[{key:"on",value:function(e,t){this._listeners[e]'
        '||(this._listeners[e]=[]),this._listeners[e].push(t)}},{key:"emit",value:function(e){for(var t=arguments.length,'
        "n=Array(t>1?t-1:0),r=1;r<t;r++)n[r-1]=arguments[r];(this._listeners[e]||[]).forEach(function(e){e.apply(void 0,n)})}}]),e}();"
    )

    # 10. Lock file content (should be dropped entirely)
    docs.append(
        "{\n"
        '  "name": "my-project",\n'
        '  "version": "1.0.0",\n'
        '  "lockfileVersion": 2,\n'
        '  "requires": true,\n'
        '  "packages": {\n'
        '    "node_modules/lodash": {\n'
        '      "version": "4.17.21",\n'
        '      "resolved": "https://registry.npmjs.org/lodash/-/lodash-4.17.21.tgz",\n'
        '      "integrity": "sha512-v2kDEe57lecTulaDIuNTPy3Ry4gLGJ6Z1O3vE1krgXZNrsQ+LFTGHVxVjcXPs17LhbZVGedAJv8XZ1tvj5FvSg=="\n'
        "    },\n"
        '    "node_modules/express": {\n'
        '      "version": "4.18.2",\n'
        '      "resolved": "https://registry.npmjs.org/express/-/express-4.18.2.tgz",\n'
        '      "integrity": "sha512-5/PsL6iGPdfQ/lKM1UuielYgv3BUoJfz1aUwU9vHZ+J7gyvwdQXFEBIEIaxeGf0GIcreATNyBExtalisDbuMqQ=="\n'
        "    },\n"
        '    "node_modules/body-parser": {\n'
        '      "version": "1.20.1",\n'
        '      "resolved": "https://registry.npmjs.org/body-parser/-/body-parser-1.20.1.tgz",\n'
        '      "integrity": "sha512-jWi7abTbYwajOytWCQc37VulmWiRae5RyTpaCyDcS5/lMdtwSz5lOpDE67srw/HYe35f1z3fDQw+3txg7gNtWw=="\n'
        "    }\n"
        "  }\n"
        "}\n"
    )

    # 11. Document with mojibake
    docs.append(
        "ThÃ©orie des ensembles est une branche des mathÃ©matiques. "
        "Les travaux de Georg Cantor ont rÃ©volutionnÃ© la comprÃ©hension "
        "de lâ€™infini et des nombres transfinis. Cette thÃ©orie constitue "
        "le fondement de pratiquement toutes les mathÃ©matiques modernes."
    )

    # 12. Document with ZWNJ/ZWJ (Indic, should be preserved)
    docs.append(
        "हिन्दी भाषा में शब्द\u200Cरूप और वाक्य\u200Cरचना महत्वपूर्ण हैं। "
        "संयुक्त\u200Dअक्षर भी देवनागरी लिपि की विशेषता है। "
        "यह एक परीक्षण दस्तावेज़ है जो ZWNJ और ZWJ की जाँच करता है।"
    )

    # 13. Markdown instruction format (MegaScience / Alpaca)
    docs.append(
        "### Instruction:\n"
        "Explain the process of photosynthesis.\n\n"
        "### Response:\n"
        "Photosynthesis is the process by which green plants and some other "
        "organisms use sunlight to synthesize nutrients from carbon dioxide "
        "and water. It generally involves the green pigment chlorophyll and "
        "generates oxygen as a byproduct."
    )

    print(
        f"  synthetic: {len(docs)} test docs (reference sections, ghost tags, license, "
        f"autogen, minified, lockfile, mojibake, ZWNJ, instruction format)"
    )
    return {"synthetic_tests": docs}


def stream_hf_dataset(
    dataset_name: str,
    config: Optional[str],
    split: str,
    text_field: str,
    max_docs: int,
    source_name: str,
) -> Dict[str, List[str]]:
    """Stream docs from HuggingFace. Returns empty dict on failure."""
    try:
        from datasets import load_dataset
    except ImportError:
        print(f"  SKIP: datasets library not installed for {source_name}")
        return {}

    try:
        kwargs = {"split": split, "streaming": True}
        if config:
            ds = load_dataset(dataset_name, config, **kwargs)
        else:
            ds = load_dataset(dataset_name, **kwargs)

        texts = []
        for i, row in enumerate(ds):
            if i >= max_docs:
                break
            text = row.get(text_field, "")
            if text and len(text) > 50:
                texts.append(text)

        print(f"  {source_name}: {len(texts)} docs streamed from {dataset_name}")
        return {source_name: texts} if texts else {}

    except Exception as e:
        print(f"  SKIP: {source_name} — {type(e).__name__}: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SourceReport:
    """Per-source cleaning results."""

    source: str
    docs_in: int = 0
    docs_out: int = 0
    docs_dropped: int = 0
    docs_invalid: int = 0
    stats: CleaningStats = None

    def __post_init__(self):
        if self.stats is None:
            self.stats = CleaningStats()


def run_cleaning(
    all_sources: Dict[str, List[str]],
    save_examples: bool = True,
    max_examples_per_path: int = 3,
) -> List[SourceReport]:
    """
    Run clean_text() on every document from every source.
    Returns per-source reports. Saves before/after examples.
    """
    reports: List[SourceReport] = []

    # Track before/after examples for each cleaning path
    examples: Dict[str, List[dict]] = defaultdict(list)

    total_docs = sum(len(docs) for docs in all_sources.values())
    processed = 0
    t_start = time.time()

    for source_name, docs in sorted(all_sources.items()):
        stats = CleaningStats()
        report = SourceReport(source=source_name, docs_in=len(docs), stats=stats)
        valid_count = 0

        for doc in docs:
            before = doc
            cleaned = clean_text(doc, stats=stats)
            processed += 1

            if cleaned:
                if is_valid_document(cleaned):
                    valid_count += 1
                else:
                    report.docs_invalid += 1
            else:
                report.docs_dropped += 1

            # Capture before/after examples for each stat that changed
            if save_examples and before != cleaned:
                _capture_examples(
                    examples, source_name, before, cleaned, stats, max_examples_per_path
                )

            # Progress every 10K docs
            if processed % 10000 == 0:
                elapsed = time.time() - t_start
                rate = processed / max(0.1, elapsed)
                print(
                    f"    [{processed:,}/{total_docs:,}] {rate:.0f} docs/s "
                    f"({source_name})",
                    flush=True,
                )

        report.docs_out = valid_count
        reports.append(report)

    elapsed = time.time() - t_start
    print(
        f"\n  Processed {processed:,} docs in {elapsed:.1f}s "
        f"({processed / max(0.1, elapsed):.0f} docs/s)"
    )

    if save_examples:
        _save_examples(examples)

    return reports


# Snapshot of stats before each doc — used to detect which fields changed
_prev_stats_snapshot: dict = {}


def _capture_examples(
    examples: Dict[str, List[dict]],
    source: str,
    before: str,
    after: str,
    stats: CleaningStats,
    max_per_path: int,
):
    """Capture before/after for cleaning paths that fired."""
    global _prev_stats_snapshot

    current = {f.name: getattr(stats, f.name) for f in fields(stats)}

    for field_name, current_val in current.items():
        prev_val = _prev_stats_snapshot.get(field_name, 0)
        if current_val > prev_val and field_name not in ("docs_processed",):
            if len(examples[field_name]) < max_per_path:
                examples[field_name].append(
                    {
                        "source": source,
                        "field": field_name,
                        "before_len": len(before),
                        "after_len": len(after),
                        "before_snippet": before[:500],
                        "after_snippet": after[:500] if after else "(DROPPED)",
                    }
                )

    _prev_stats_snapshot = current


def _save_examples(examples: Dict[str, List[dict]]):
    """Save before/after examples to test_cleaning_examples/."""
    os.makedirs(EXAMPLES_DIR, exist_ok=True)

    for field_name, exs in sorted(examples.items()):
        if not exs:
            continue
        path = os.path.join(EXAMPLES_DIR, f"{field_name}.json")
        with open(path, "w") as f:
            json.dump(exs, f, indent=2, ensure_ascii=False)

    summary_path = os.path.join(EXAMPLES_DIR, "README.txt")
    with open(summary_path, "w") as f:
        f.write("Before/after examples from test_cleaning_smoke.py\n")
        f.write("=" * 50 + "\n\n")
        for field_name in sorted(examples.keys()):
            f.write(f"{field_name}: {len(examples[field_name])} example(s)\n")

    print(f"\n  Examples saved to: {EXAMPLES_DIR}/")
    print(f"  Fields with examples: {', '.join(sorted(examples.keys()))}")


# ═══════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════


def print_report(reports: List[SourceReport]):
    """Print a comprehensive per-source stats table."""

    # Aggregate stats
    total_stats = CleaningStats()
    total_in = 0
    total_out = 0
    total_dropped = 0
    total_invalid = 0

    for r in reports:
        total_stats += r.stats
        total_in += r.docs_in
        total_out += r.docs_out
        total_dropped += r.docs_dropped
        total_invalid += r.docs_invalid

    print("\n" + "=" * 90)
    print("CLEANING SMOKE TEST RESULTS")
    print("=" * 90)

    # Per-source summary table
    print(
        f"\n{'Source':<30} {'In':>7} {'Out':>7} {'Drop':>7} {'Invalid':>7} {'Drop%':>7}"
    )
    print("-" * 72)
    for r in sorted(reports, key=lambda x: x.docs_in, reverse=True):
        drop_pct = f"{r.docs_dropped * 100 / max(1, r.docs_in):.1f}%"
        print(
            f"{r.source:<30} {r.docs_in:>7,} {r.docs_out:>7,} "
            f"{r.docs_dropped:>7,} {r.docs_invalid:>7,} {drop_pct:>7}"
        )

    print("-" * 72)
    total_drop_pct = f"{total_dropped * 100 / max(1, total_in):.1f}%"
    print(
        f"{'TOTAL':<30} {total_in:>7,} {total_out:>7,} "
        f"{total_dropped:>7,} {total_invalid:>7,} {total_drop_pct:>7}"
    )

    # Cleaning stats breakdown
    print(f"\n{'Cleaning Step':<40} {'Count':>12} {'Notes'}")
    print("-" * 70)

    stat_notes = {
        "docs_processed": "total docs that entered clean_text()",
        "docs_dropped_empty": "empty after cleaning",
        "docs_dropped_short": "below min_chars threshold",
        "docs_dropped_low_diversity": "below min unique tokens",
        "docs_dropped_mojibake": "mojibake ratio > 2%",
        "docs_dropped_autogenerated": "DO NOT EDIT / Generated by (full drop)",
        "docs_dropped_minified": "single long line, low newline density",
        "docs_dropped_lockfile": "package-lock / yarn.lock content",
        "chars_removed_surrogates": "U+D800-U+DFFF lone surrogates",
        "chars_removed_c0c1": "C0/C1 control chars (null, DEL, 0x80-0x9F)",
        "chars_removed_zw_bidi": "ZWSP, bidi marks, BOM, U+FFFD",
        "chars_removed_pua": "private-use area chars",
        "ghost_tag_removals": "chars removed by ghost tag stripping",
        "license_headers_stripped": "files with license header removed",
        "autogen_warnings_stripped": "autogen warning lines removed (non-dropped files)",
        "citation_markers_stripped": "chars removed by [42] citation stripping",
        "reference_sections_stripped": "Reference/Bibliography tails removed",
    }

    for f in fields(total_stats):
        val = getattr(total_stats, f.name)
        note = stat_notes.get(f.name, "")
        marker = ""
        if val == 0 and f.name.startswith("docs_dropped_"):
            marker = " ⚠ NOT EXERCISED"
        elif val == 0 and f.name in (
            "license_headers_stripped",
            "citation_markers_stripped",
            "reference_sections_stripped",
            "ghost_tag_removals",
        ):
            marker = " ⚠ NOT EXERCISED"

        if val > 0 or marker:
            print(f"  {f.name:<38} {val:>12,}  {note}{marker}")

    # Coverage check
    print(f"\n{'=' * 90}")
    print("COVERAGE CHECK — did every cleaning path fire?")
    print("=" * 90)

    critical_paths = [
        ("docs_dropped_autogenerated", "Autogen file drop"),
        ("docs_dropped_minified", "Minified JS/CSS drop"),
        ("docs_dropped_lockfile", "Lock file drop"),
        ("docs_dropped_mojibake", "Mojibake detection"),
        ("ghost_tag_removals", "Ghost tag stripping"),
        ("license_headers_stripped", "License header stripping"),
        ("citation_markers_stripped", "Citation marker stripping"),
        ("reference_sections_stripped", "Reference/Bibliography tail stripping"),
    ]

    all_covered = True
    for field_name, label in critical_paths:
        val = getattr(total_stats, field_name)
        status = "PASS" if val > 0 else "MISS"
        if status == "MISS":
            all_covered = False
        print(f"  [{status}] {label:<45} ({val:,} hits)")

    print()
    if all_covered:
        print("  ALL CLEANING PATHS EXERCISED — ready for production run.")
    else:
        print("  SOME PATHS NOT EXERCISED — check data sources or add test docs.")

    # Write machine-readable report
    report_path = os.path.join(PIPELINE_DIR, "test_cleaning_report.json")
    report_data = {
        "total_docs_in": total_in,
        "total_docs_out": total_out,
        "total_dropped": total_dropped,
        "total_invalid": total_invalid,
        "all_paths_covered": all_covered,
        "stats": asdict(total_stats),
        "per_source": [
            {
                "source": r.source,
                "docs_in": r.docs_in,
                "docs_out": r.docs_out,
                "docs_dropped": r.docs_dropped,
                "docs_invalid": r.docs_invalid,
                "stats": asdict(r.stats),
            }
            for r in reports
        ],
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n  Report written to: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Smoke test for cleaning pipeline")
    parser.add_argument(
        "--quick", action="store_true", help="1%% sample instead of 10%% (faster)"
    )
    parser.add_argument(
        "--include-hf",
        action="store_true",
        help="Also stream C4 and open-web-math from HuggingFace",
    )
    parser.add_argument(
        "--no-examples", action="store_true", help="Skip saving before/after examples"
    )
    parser.add_argument(
        "--max-code",
        type=int,
        default=1000,
        help="Max docs per code language (default: 1000)",
    )
    parser.add_argument(
        "--max-arxiv", type=int, default=500, help="Max arxiv docs (default: 500)"
    )
    args = parser.parse_args()

    sample_frac = 0.01 if args.quick else 0.10

    print("=" * 70)
    print("DATA CLEANING SMOKE TEST")
    print("=" * 70)
    print(f"  Sample fraction: {sample_frac:.0%}")
    print(f"  Max code docs/lang: {args.max_code}")
    print(f"  Max arxiv docs: {args.max_arxiv}")
    print(f"  HuggingFace streaming: {'yes' if args.include_hf else 'no'}")
    print()

    # ── Load all data sources ─────────────────────────────────────────
    print("Loading data sources...")
    all_sources: Dict[str, List[str]] = {}

    # 1. Tokenizer samples (10% of local Indic data)
    all_sources.update(load_tokenizer_samples(sample_frac))

    # 2. Downloaded code files
    all_sources.update(load_downloaded_code(max_per_lang=args.max_code))

    # 3. Downloaded arxiv
    all_sources.update(load_downloaded_arxiv(max_docs=args.max_arxiv))

    # 4. Local lock files, minified JS, autogen files
    all_sources.update(load_local_lockfiles())
    all_sources.update(load_local_minified())
    all_sources.update(load_local_autogen())

    # 5. Synthetic test docs
    all_sources.update(create_synthetic_docs())

    # 6. HuggingFace streaming (optional)
    if args.include_hf:
        print("\nStreaming from HuggingFace...")
        all_sources.update(
            stream_hf_dataset(
                "allenai/c4",
                "en",
                "validation",
                "text",
                500,
                "hf_c4_en",
            )
        )
        all_sources.update(
            stream_hf_dataset(
                "open-web-math/open-web-math",
                None,
                "train",
                "text",
                300,
                "hf_open_web_math",
            )
        )

    total = sum(len(v) for v in all_sources.values())
    print(f"\n  Total: {len(all_sources)} sources, {total:,} docs")

    if total == 0:
        print("\nERROR: No data loaded. Check paths.")
        sys.exit(1)

    # ── Run cleaning ──────────────────────────────────────────────────
    print("\nRunning clean_text() on all docs...")
    reports = run_cleaning(all_sources, save_examples=not args.no_examples)

    # ── Print report ──────────────────────────────────────────────────
    print_report(reports)


if __name__ == "__main__":
    main()
