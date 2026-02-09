
import os
from components.checkpoint_registry import CheckpointRegistry

def test_registry():
    # Use in-memory SQLite for fast testing
    print("Initializing Registry (Local SQLite)...")
    registry = CheckpointRegistry("sqlite:///:memory:")
    
    run_id = "test_run_rego"
    
    # 1. Register a TEMPORARY checkpoint
    print("\n--- Test Case 1: Temporary Checkpoint ---")
    key_temp = "s3://bucket/ckpt_step_100.pt"
    registry.register_checkpoint(run_id, 100, key_temp, 0.5, tag="temporary")
    
    # Check if delete allowed
    allowed = registry.can_delete(key_temp)
    print(f"Can delete temporary? {allowed} (Expected: True)")
    if allowed:
        registry.mark_for_deletion(key_temp)

    # 2. Register a GROWTH checkpoint (Protected)
    print("\n--- Test Case 2: Growth Checkpoint ---")
    key_growth = "s3://bucket/ckpt_step_1000_growth.pt"
    registry.register_checkpoint(run_id, 1000, key_growth, 0.1, tag="growth")
    
    # Check if delete allowed
    allowed = registry.can_delete(key_growth)
    print(f"Can delete growth?    {allowed} (Expected: False)")
    
    # Try forcing deletion
    try:
        registry.mark_for_deletion(key_growth)
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")

if __name__ == "__main__":
    test_registry()
