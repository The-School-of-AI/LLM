
# SPDL DataLoader Test Results

## Test Execution Details

**Date:** 17 February 2026
**Test File:** test_spdl_bin_idx_dataloader.py
**SPDL Version:** 0.2.0
**Config:** configuration_P4.yaml

## Hardware Configuration

- Platform: macOS-26.2-arm64-arm-64bit
- CPU Cores: 10
- Memory: 16 GB
- CUDA Available: False
- Python Version: 3.11.14

## Test Results

Testing SPDL bin/idx dataloader in: Test_data
Step 0: batch shape torch.Size([512, 2048]), output shape torch.Size([512])
Step 1: batch shape torch.Size([512, 2048]), output shape torch.Size([512])
Step 2: batch shape torch.Size([512, 2048]), output shape torch.Size([512])
Test completed: 10 batches, 10485760 tokens processed.
SPDL bin/idx dataloader test PASSED.

## Processing Performance

- Processing Time: 2 seconds
- Batches Processed: 10
- Tokens Processed: 10,485,760
- Throughput: 5,242,880 tokens/second
- Device: CPU (CUDA not available)

## Performance Notes
- Test ran on CPU due to CUDA unavailability
- SPDL dataloader processed 10 batches for measurement
- Memory usage was efficient with streaming binary data loading
- Performance may vary with larger datasets or GPU acceleration
