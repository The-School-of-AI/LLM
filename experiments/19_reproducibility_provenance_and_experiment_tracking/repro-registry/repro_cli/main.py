from pathlib import Path

from repro.registry import finalize_run, start_run


def main():
    run_id, run_dir = start_run(Path("runs"))
    print("Started run:", run_id)
    finalize_run(run_dir)
    print("Finalized run")


if __name__ == "__main__":
    main()
