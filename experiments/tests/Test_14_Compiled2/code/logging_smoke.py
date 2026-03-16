import os
import time


def main() -> int:
    run_id = os.environ.get("RUN_ID") or f"logging_smoke_{int(time.time())}"
    rank = int(os.environ.get("RANK", "0"))

    try:
        from components import TrainingOps
    except Exception as e:
        print(f"[ERROR] Failed to import TrainingOps: {e}")
        return 1

    skip_vector_check = os.environ.get("SKIP_VECTOR_CHECK", "0") == "1"
    vector_service_name = os.environ.get("VECTOR_SERVICE_NAME", "t12-vector.service")

    ops = TrainingOps(
        run_id=run_id,
        rank=rank,
        skip_vector_check=skip_vector_check,
        vector_service_name=vector_service_name,
    )

    for step in range(5):
        ops.log_step(
            step=step,
            metrics={"loss": 1.0 / (step + 1), "lr": 3e-4},
            context={"phase": "logging_smoke"},
        )
        time.sleep(1.0)

    ops.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
