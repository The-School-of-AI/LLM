# SPDL DataLoader Test Results

## Test Execution Details

**Date:** 15 February 2026
**Test File:** `test_spdl_dataloader.py`
**SPDL Version:** 0.2.0

## Hardware Configuration

- **Platform:** macOS-26.2-arm64-arm-64bit
- **CPU Cores:** 10
- **Memory:** 16 GB
- **CUDA Available:** False
- **Python Version:** 3.11.14

## Test Results

### Data Generation
- **Time:** 13.86 seconds
- **Data Size:** 1,000,000 records (2 Parquet files × 500,000 records each)
- **Record Format:** Tokens (128 integers per record)

### Processing Performance
- **Processing Time:** 39.23 seconds
- **Batches Processed:** 1 batch
- **Throughput:** 25,490.44 records/second
- **Device:** CPU (CUDA not available)

### Test Status
✅ **PASSED** - All assertions successful
- Parquet files created successfully
- Pipeline built and executed without errors
- Data cleanup completed

## Performance Notes

- Test ran on CPU due to CUDA unavailability
- SPDL pipeline processed 10 batches internally for measurement
- Throughput calculation based on total records processed over processing time
- Memory usage was efficient with streaming data loading
- Performance varies slightly between runs due to system load