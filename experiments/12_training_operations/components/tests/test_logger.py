import json
import os
import shutil
import time

from components.json_logger import JSONLogger


def test_logger():
    run_id = "test_run_context"
    base_dir = "/tmp/test_logs"

    # Cleanup
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    print("Initializing Logger with Default Context...")
    default_ctx = {
        "source": "growth_team/lora",
        "model_size": "70B",
        "cluster_region": "us-east-1",
    }

    logger = JSONLogger(
        base_dir=base_dir, run_id=run_id, rank=0, default_context=default_ctx
    )

    print("Logging step with additional context...")
    # This step adds 'routing_dist' but should KEEP 'source', 'model_size' etc.
    logger.log_step(step=1, metrics={"loss": 0.4}, context={"routing_dist": [0.1, 0.9]})

    time.sleep(0.1)
    logger.close()

    log_file = f"{base_dir}/{run_id}_rank_0.jsonl"
    print(f"Verifying context in {log_file}...")

    with open(log_file, "r") as f:
        line = json.loads(f.readline())
        ctx = line["context"]

        # Checks
        assert ctx["source"] == "growth_team/lora", "Missing default source"
        assert ctx["model_size"] == "70B", "Missing default model_size"
        assert ctx["routing_dist"] == [0.1, 0.9], "Missing step context"

        print("✓ Context Merge Verified Successfully")
        print(f"  Merged Context: {ctx}")


if __name__ == "__main__":
    test_logger()
