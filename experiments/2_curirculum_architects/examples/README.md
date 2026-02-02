# Curriculum Tags Examples

This directory contains example scripts demonstrating how to use the curriculum-tags package.

## Available Examples

### 1. Basic Usage (`basic_usage.py`)

Demonstrates fundamental tagging operations:
- Initializing the tagger
- Tagging individual samples
- Viewing curriculum metadata

```bash
python examples/basic_usage.py
```

### 2. Custom Plugin (`custom_plugin.py`)

Shows how to create and use custom metric plugins:
- Defining a new metric class
- Registering the plugin
- Using it with the tagger

```bash
python examples/custom_plugin.py
```

### 3. Parquet Processing (`parquet_processing.py`)

Demonstrates batch processing of parquet files:
- Creating sample datasets
- Processing with progress tracking
- Viewing tagged results

```bash
python examples/parquet_processing.py
```

## Running Examples

Make sure the package is installed:

```bash
# From the curriculum_tags directory
uv pip install -e .
```

Then run any example:

```bash
python examples/basic_usage.py
```
