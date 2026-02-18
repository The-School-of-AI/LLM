# SPDL Dataloader

This project implements a high-performance dataloader using Meta's SPDL (Scalable and Performant Data Loading) library for efficient Parquet file processing and batching.

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

2. Install project dependencies:
   ```bash
   pip install -e .
   ```

   Or using uv:
   ```bash
   pip install uv
   uv pip install -e .
   ```

This will automatically install:
- **SPDL 0.2.0** - Meta's high-performance data loading library
- **spdl-core** - Core SPDL functionality
- **spdl-io** - I/O operations for SPDL
- **PyTorch** - Deep learning framework
- **PyArrow** - Apache Arrow Python bindings for Parquet support
- **NumPy** - Numerical computing library

## Project Structure

```
data_loader/
├── pyproject.toml          # Project configuration and dependencies
├── run_test.sh             # Automated test runner script
├── spdl_dataloader.py      # Main SPDL dataloader implementation
├── data_loader_main.py     # Command-line interface
├── test_spdl_dataloader.py # Test script with performance benchmarking
├── common.py              # Shared constants and configuration
├── README_spdl.md         # This documentation
└── test_result.md         # Latest test results and hardware info
```

## Dependencies

The project uses the following key libraries:

- **SPDL**: Meta's streaming data loading library for high-performance I/O
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

Key parameters in `common.py`:
- `BATCH_SIZE`: Number of samples per batch (default: 1024)
- `NUM_THREADS`: Number of processing threads (auto-detected from CPU cores)
- `PREFETCH_BUFFER`: Number of batches to prefetch (default: 8)
- `TOKENS_COLUMN`: Column name for token data in Parquet files

## Logging

Logs and audit trails are stored in `/data/dolma/logs`.
