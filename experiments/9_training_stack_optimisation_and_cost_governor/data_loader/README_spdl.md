
# SPDL Dataloader

This project implements a high-performance dataloader using Meta's SPDL (Scalable and Performant Data Loading) library for efficient binary (.bin/.idx) token data processing and batching. The loader is optimized for large-scale LLM training and supports streaming, batching, and robust error handling.


## Prerequisites

- Python 3.10+ (tested with 3.11)
- pip or uv for dependency management

### Installing Python 3.11 (on macOS)
```bash
brew install python@3.11
```


## Installation

1. Create a virtual environment with Python 3.11:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```


2. Install project dependencies (recommended: uv):
   ```bash
   pip install uv  # install uv if not already present
   uv pip install -e .  # install all dependencies from pyproject.toml
   ```


   To update all packages to the latest compatible versions (recommended before production runs):
   ```bash
   uv pip install -U -e .
   ```

   This will ensure all dependencies in `pyproject.toml` are upgraded to their latest compatible versions. Re-run this command regularly to keep your environment secure and up to date.


   Or, using pip (if you prefer):
   ```bash
   pip install -r requirements.txt
   ```
   The requirements.txt file is auto-generated and matches all dependencies in pyproject.toml for compatibility with pip-based workflows.

This will automatically install:
- **SPDL 0.2.0** - Meta's high-performance data loading library
- **spdl-core** - Core SPDL functionality
- **spdl-io** - I/O operations for SPDL
- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computing library


## Project Structure

```
data_loader/
├── pyproject.toml              # Project configuration and dependencies
├── spdl_dataloader.py          # Main SPDL dataloader implementation (.bin/.idx)
├── test_spdl_bin_idx_dataloader.py # Test script for .bin/.idx loader
├── README_spdl.md              # This documentation
└── test_result.md              # Latest test results and hardware info
```

## Dependencies

The project uses the following key libraries:

- **SPDL**: Meta's streaming data loading library for high-performance I/O
- **PyTorch**: For tensor operations and model input
- **NumPy**: For efficient binary data handling
#
# Binary .bin/.idx Loader
#

The dataloader reads tokenized data from a binary `.bin` file and uses a corresponding `.idx` file for fast sequence offset lookup. It supports:
- Streaming large datasets from disk
- Efficient batch construction (configurable batch size and sequence length)
- Robust error handling for incomplete/corrupt data
- Torch tensor output for direct model input

See `spdl_dataloader.py` and `test_spdl_bin_idx_dataloader.py` for implementation and usage examples.

## Hardware & Performance

**Tested on:**
- Platform: macOS-26.2-arm64-arm-64bit
- CPU Cores: 10
- Memory: 16 GB
- CUDA: Not available (CPU only)
- Python: 3.11.14
- PyTorch: 2.10.0

**Recent test results:**
- Batches processed: 10
- Tokens processed: 41,943,040
- Batch size: 1024, Sequence length: 4096
- All assertions passed, loader robust to edge cases

See `test_result.md` for full details and metrics.
- **PyTorch**: Deep learning framework for tensor operations
- **PyArrow**: Efficient Parquet file reading and processing
- **NumPy**: Numerical computing support

All dependencies are automatically managed through `pyproject.toml` and installed via pip.

## Usage

Activate the environment:
```bash
source .venv/bin/activate
```

Run the dataloader, passing one or more Parquet file paths as arguments:
```bash
python data_loader_main.py /path/to/shard1.parquet /path/to/shard2.parquet
```

## Testing

### Quick Test Run
Run the test script to generate sample data and verify functionality:
```bash
python test_spdl_dataloader.py
```

### Automated Test Runner
Use the provided test runner script for a complete test with system information:
```bash
./run_test.sh
```

This script will:
- Activate the virtual environment
- Display system information (hardware, CUDA availability, etc.)
- Run the full test suite with performance benchmarking
- Clean up generated test data

**Note:** Make sure the script is executable: `chmod +x run_test.sh`

Expected output (example):
```
==========================================
System Information
==========================================
CUDA available: False
Platform: macOS-26.2-arm64-arm-64bit
CPU cores: 10
Memory: 16 GB
Python version: 3.11.14
PyTorch version: 2.10.0

==========================================
Running SPDL DataLoader Test
==========================================
Data generation time: 13.86 seconds
Processing time: 39.23 seconds
Batches processed: 1
Total records: 1000000
Throughput: 25490.44 records/second
Test completed successfully.
```

## Performance

The dataloader achieves high throughput for large-scale data processing:
- **Throughput**: ~25K records/second on CPU (may vary by hardware)
- **Memory Efficient**: Streaming processing with configurable prefetch buffers
- **Scalable**: Supports multiple Parquet shards and parallel processing


## Configuration

The dataloader is fully parameterized using YAML configuration files. Two example configs are provided:

- `configuration_P4.yaml`: Optimized for NVIDIA Tesla P4 (8GB VRAM)
- `configuration_P5.yaml`: Optimized for NVIDIA A100/P5 (40GB+ VRAM)

You can create your own config file or modify these to match your hardware. Key parameters:

- `batch_size`: Number of samples per batch
- `num_threads`: Number of processing threads
- `prefetch_buffer`: Number of batches to prefetch
- `sequence_length`: Number of tokens per sequence
- `dtype`: Token dtype (e.g., uint16, int32)

The config file is loaded automatically via the `SPDL_CONFIG` environment variable, or defaults to `configuration_P4.yaml`.


## Production Usage

The main entry point for all dataloader runs is `dataloader.py`, which supports argument parsing and logging.

To run the dataloader in production mode with a specific config and token folder:

```bash
./run_spdl_production.sh <CONFIG_FILE> <TOKEN_FOLDER>
```

Example:
```bash
./run_spdl_production.sh configuration_P5.yaml /path/to/token_folder
```

This script will:
- Activate the virtual environment
- Set the config for the run (via SPDL_CONFIG)
- Call `dataloader.py` with the token folder and batch count
- Log progress and results using the logging framework

You can also run the dataloader directly:
```bash
python dataloader.py --token-folder /path/to/token_folder --batches 10 --log-level INFO
```

All pipeline parameters (batch size, threads, buffer, sequence length, dtype) are set from the config file.

## Logging

Logs and audit trails are stored in `/data/dolma/logs`.
