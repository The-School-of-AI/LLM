# SPDL DataLoader Test Results

## Test Execution Details

**Date:** 18 February 2026
**Test File:** test_spdl_bin_idx_dataloader.py
**SPDL Version:** 0.2.0

## Hardware Configuration

- Darwin MAC-H63J4WGW74 25.2.0 Darwin Kernel Version 25.2.0: Tue Nov 18 21:09:40 PST 2025; root:xnu-12377.61.12~1/RELEASE_ARM64_T6000 arm64
- machdep.cpu.cores_per_package: 10
- machdep.cpu.core_count: 10
- machdep.cpu.logical_per_package: 10
- machdep.cpu.thread_count: 10
- machdep.cpu.brand_string: Apple M1 Pro

## Metrics Summary

- Throughput: 13077432.60 tokens/sec
- Average batch processing time: 0.002150 seconds
- Total Processing Time: 0.80 seconds seconds
- Total tokens processed: 10485760
- Output shape: torch.Size([512])

## Performance Notes
- Test ran on CPU due to CUDA unavailability
- SPDL dataloader processed 10 batches for measurement
- Memory usage was efficient with streaming binary data loading
- Performance may vary with larger datasets or GPU acceleration
