"""Run the benchmark from the repo root: python -m benchmark_indic_rag_suite --split dev --lang hi -o out.json"""

from benchmark_indic_rag_suite.cli import run

if __name__ == "__main__":
    exit(run())
