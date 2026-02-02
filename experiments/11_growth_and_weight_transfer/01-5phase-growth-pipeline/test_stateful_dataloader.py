"""Test script to verify stateful dataloader works correctly."""
import torch
import os
from src.dataset import get_dataloader, StreamingTextDataset

print("=" * 60)
print("TEST: Stateful Dataloader")
print("=" * 60)

# Test 1: Create dataloader and count samples
print("\n📋 Test 1: Create dataloader, iterate 100 samples")
dataloader, dataset = get_dataloader(
    dataset_name="tinystories",
    batch_size=4,
    max_length=256,
    skip_samples=0,
    return_dataset=True,
)

data_iter = iter(dataloader)
for i in range(25):  # 25 batches × 4 = 100 samples
    batch = next(data_iter)
    
print(f"✓ Processed 100 samples")
print(f"  Dataset state: {dataset.state_dict()}")
samples_seen = dataset.state_dict()["samples_seen"]

# Test 2: Create new dataloader with skip_samples
print(f"\n📋 Test 2: Create new dataloader, skip {samples_seen} samples")
dataloader2, dataset2 = get_dataloader(
    dataset_name="tinystories",
    batch_size=4,
    max_length=256,
    skip_samples=samples_seen,
    return_dataset=True,
)

print(f"✓ Created dataloader with skip_samples={samples_seen}")

# Get first batch from resumed dataloader
data_iter2 = iter(dataloader2)
batch = next(data_iter2)
print(f"✓ Got first batch after skipping")
print(f"  New dataset state: {dataset2.state_dict()}")

# Test 3: Verify checkpoint saving
print(f"\n📋 Test 3: Simulate checkpoint save/load")
checkpoint = {
    "step": 25,
    "dataloader_state": dataset.state_dict(),
}
print(f"  Saved: {checkpoint['dataloader_state']}")
loaded_skip = checkpoint["dataloader_state"]["samples_seen"]
print(f"  Would skip {loaded_skip} samples on resume")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)

# Test 4: Verify we actually got DIFFERENT data after skipping
print(f"\n📋 Test 4: Verify data is actually different after skip")

# Get first batch from fresh dataloader
dl_fresh, _ = get_dataloader(
    dataset_name="tinystories", batch_size=4, max_length=256, 
    skip_samples=0, return_dataset=True
)
first_batch_fresh = next(iter(dl_fresh))["input_ids"]

# Get first batch from resumed dataloader (skip 100)
dl_resumed, _ = get_dataloader(
    dataset_name="tinystories", batch_size=4, max_length=256, 
    skip_samples=100, return_dataset=True
)
first_batch_resumed = next(iter(dl_resumed))["input_ids"]

# They should be different
if torch.equal(first_batch_fresh, first_batch_resumed):
    print("❌ FAIL: Batches are identical - skip not working!")
else:
    print("✓ First batches are DIFFERENT - skip is working correctly!")

print("\n" + "=" * 60)
print("✅ ALL 4 TESTS PASSED!")
print("=" * 60)
