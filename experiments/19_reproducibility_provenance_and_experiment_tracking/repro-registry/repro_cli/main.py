from repro.registry import start_run, finalize_run
from pathlib import Path

def main():
    run_id, run_dir = start_run(Path("runs"))
    print("Started run:", run_id)
    finalize_run(run_dir)
    print("Finalized run")

if __name__ == "__main__":
    main()
