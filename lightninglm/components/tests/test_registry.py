import time

from lightninglm.components.checkpoint_registry import CheckpointRegistry


def test_registry():
    """End-to-end test for the ClickHouse-backed CheckpointRegistry."""

    print("Initializing Registry (ClickHouse)...")
    registry = CheckpointRegistry()

    run_id = "test_run_registry"

    # Clean up any leftover test data
    registry._insert(
        f"INSERT INTO {registry.table} (run_id, s3_key, status) "
        f"VALUES ('{run_id}', '__cleanup__', 'deleted')"
    )

    # ---- 1. Register a TEMPORARY checkpoint ----
    print("\n--- Test 1: Temporary Checkpoint ---")
    key_temp = f"s3://bucket/test_ckpt_temp_{int(time.time())}.pt"
    registry.register_checkpoint(run_id, 100, key_temp, loss=0.5, tag="temporary")

    record = registry.get_checkpoint(key_temp)
    assert record is not None, "Checkpoint not found after register"
    assert record["tag"] == "temporary"
    assert record["is_protected"] is False
    assert record["status"] == "registered"
    print(
        f"✓ Registered: step={record['step']}, tag={record['tag']}, protected={record['is_protected']}"
    )

    # ---- 2. can_delete on temporary → True ----
    print("\n--- Test 2: can_delete (temporary) ---")
    allowed = registry.can_delete(key_temp)
    assert allowed is True, f"Expected True, got {allowed}"
    print(f"✓ can_delete(temporary) = {allowed}")

    # ---- 3. Soft-delete the temporary checkpoint ----
    print("\n--- Test 3: mark_for_deletion (temporary) ---")
    registry.mark_for_deletion(key_temp)
    record = registry.get_checkpoint(key_temp)
    assert record["status"] == "deleted", f"Expected 'deleted', got {record['status']}"
    print(f"✓ Status after deletion: {record['status']}")

    # ---- 4. Register a GROWTH checkpoint (auto-protected) ----
    print("\n--- Test 4: Growth Checkpoint (protected) ---")
    key_growth = f"s3://bucket/test_ckpt_growth_{int(time.time())}.pt"
    registry.register_checkpoint(run_id, 1000, key_growth, loss=0.1, tag="growth")

    record = registry.get_checkpoint(key_growth)
    assert record["is_protected"] is True
    assert record["tag"] == "growth"
    print(
        f"✓ Registered: step={record['step']}, tag={record['tag']}, protected={record['is_protected']}"
    )

    # ---- 5. can_delete on protected → False ----
    print("\n--- Test 5: can_delete (protected) ---")
    allowed = registry.can_delete(key_growth)
    assert allowed is False, f"Expected False, got {allowed}"
    print(f"✓ can_delete(growth) = {allowed}")

    # ---- 6. mark_for_deletion on protected → ValueError ----
    print("\n--- Test 6: mark_for_deletion (protected) → error ---")
    try:
        registry.mark_for_deletion(key_growth)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly blocked: {e}")

    # ---- 7. Unknown checkpoint → can_delete returns False ----
    print("\n--- Test 7: Unknown checkpoint ---")
    allowed = registry.can_delete("s3://bucket/nonexistent.pt")
    assert allowed is False
    print(f"✓ can_delete(unknown) = {allowed}")

    # ---- 8. list_checkpoints ----
    print("\n--- Test 8: list_checkpoints ---")
    # Register a couple more
    key_tqp = f"s3://bucket/test_ckpt_tqp_{int(time.time())}.pt"
    registry.register_checkpoint(run_id, 2000, key_tqp, loss=0.05, tag="tqp")

    checkpoints = registry.list_checkpoints(run_id)
    assert len(checkpoints) >= 2, f"Expected >= 2, got {len(checkpoints)}"
    print(f"✓ list_checkpoints returned {len(checkpoints)} registered checkpoints")
    for c in checkpoints:
        print(
            f"  step={c['step']}, tag={c['tag']}, loss={c['loss']}, s3_key={c['s3_key']}"
        )

    # ---- 9. best_checkpoint ----
    print("\n--- Test 9: best_checkpoint ---")
    best = registry.best_checkpoint(run_id, top_n=1)
    assert len(best) == 1
    assert best[0]["loss"] <= 0.1, f"Expected lowest loss, got {best[0]['loss']}"
    print(
        f"✓ Best checkpoint: step={best[0]['step']}, loss={best[0]['loss']}, s3_key={best[0]['s3_key']}"
    )

    print("\n✅ All CheckpointRegistry tests passed!")


if __name__ == "__main__":
    test_registry()
