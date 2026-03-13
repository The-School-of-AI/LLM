#!/usr/bin/env python3
"""
Option B End-to-End Validation Script
======================================
Runs a full automated checklist verifying that every component of the
Option B (API mode) pipeline is working correctly:

  Phase 1: Environment & Package Check
  Phase 2: Model Files on Disk
  Phase 3: API Server Health
  Phase 4: API Inference (live single-prompt test)
  Phase 5: generate_outputs.py output (model_outputs_test.csv)
  Phase 6: run_evaluation.py output (results/)
  Phase 7: Metric completeness & correctness

Usage
-----
  # Full validation (assumes servers already running)
  python scripts/validate_option_b.py

  # Custom paths
  python scripts/validate_option_b.py \\
      --csv ../data/model_outputs_test.csv \\
      --results-dir ../results \\
      --base-port 8001 --sft-port 8002

Exit codes
----------
  0  All checks passed
  1  One or more checks failed
"""

import argparse
import csv
import importlib
import importlib.util
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Default paths
DEFAULT_CSV         = ROOT / "data" / "model_outputs_test.csv"
DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_MODELS_DIR  = ROOT / "models"
CONFIG_PATH         = ROOT / "config" / "config.yaml"
PROMPTS_PATH        = ROOT / "prompts" / "evaluation_prompts.json"

BASE_PORT = 8001
SFT_PORT  = 8002

# Test prompt IDs used by the end-to-end run
TEST_PROMPT_IDS = ["IF_001", "IF_002", "FQ_001", "RN_001", "EC_001"]

# Required CSV column names
REQUIRED_CSV_FIELDS = [
    "prompt_id", "category", "difficulty", "prompt_text",
    "base_output", "sft_output", "base_model", "sft_model",
    "timestamp", "base_error", "sft_error",
]

# Required keys in summary JSON
REQUIRED_SUMMARY_KEYS = [
    "total_prompts", "base_if_rate", "sft_if_rate", "if_improvement",
    "base_hallucination_rate", "sft_hallucination_rate",
    "new_hallucination_count",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
SKIP = "\033[93m[SKIP]\033[0m"
BOLD = "\033[1m"
RST  = "\033[0m"


class ChecklistRunner:
    def __init__(self):
        self.results: list[dict] = []

    def check(self, label: str, condition: bool, detail: str = "") -> bool:
        status = "PASS" if condition else "FAIL"
        self.results.append({"label": label, "status": status, "detail": detail})
        icon = PASS if condition else FAIL
        line = f"  {icon} {label}"
        if detail:
            line += f"\n         {detail}"
        print(line)
        return condition

    def skip(self, label: str, reason: str = "") -> None:
        self.results.append({"label": label, "status": "SKIP", "detail": reason})
        line = f"  {SKIP} {label}"
        if reason:
            line += f"\n         reason: {reason}"
        print(line)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r["status"] == "PASS")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r["status"] == "FAIL")

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r["status"] == "SKIP")

    def summary(self) -> bool:
        """Print summary table and return True if all checks passed."""
        print(f"\n{'=' * 65}")
        print(f"{BOLD}  VALIDATION SUMMARY — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RST}")
        print(f"{'=' * 65}")
        print(f"  Total  : {self.total}")
        print(f"  \033[92mPassed\033[0m : {self.passed}")
        if self.failed:
            print(f"  \033[91mFailed\033[0m : {self.failed}")
        if self.skipped:
            print(f"  \033[93mSkipped\033[0m: {self.skipped}")

        if self.failed == 0:
            print(f"\n  {BOLD}✓ ALL CHECKS PASSED — Option B pipeline is working correctly.{RST}")
        else:
            print(f"\n  ✗ {self.failed} check(s) failed. See details above.")
            failed_items = [r["label"] for r in self.results if r["status"] == "FAIL"]
            for item in failed_items:
                print(f"     • {item}")

        print(f"{'=' * 65}\n")
        return self.failed == 0


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: int = 5) -> tuple[int, bytes]:
    """Simple HTTP GET. Returns (status_code, body_bytes)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as exc:
        return -1, str(exc).encode()


def _openai_chat(endpoint: str, model_name: str, message: str) -> tuple[str, str]:
    """Send a single chat completion via the openai package. Returns (output, error)."""
    try:
        import openai  # type: ignore
    except ImportError:
        return "", "openai not installed"
    try:
        client = openai.OpenAI(base_url=endpoint, api_key="dummy")
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": message}],
            max_tokens=64,
            temperature=0.0,
        )
        return resp.choices[0].message.content or "", ""
    except Exception as exc:
        return "", str(exc)


def _load_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _latest_file(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Validation phases
# ---------------------------------------------------------------------------

def phase_environment(cl: ChecklistRunner) -> None:
    print(f"\n{BOLD}Phase 1 — Environment & Packages{RST}")
    print("─" * 50)

    cl.check("Python ≥ 3.10", sys.version_info >= (3, 10), f"Found {sys.version.split()[0]}")

    for pkg in ["yaml", "openai", "transformers", "fastapi", "uvicorn"]:
        available = importlib.util.find_spec(pkg) is not None
        cl.check(f"Package: {pkg}", available, "" if available else f"pip install {pkg}")

    cl.check("config/config.yaml exists", CONFIG_PATH.exists(), str(CONFIG_PATH))
    cl.check("evaluation_prompts.json exists", PROMPTS_PATH.exists(), str(PROMPTS_PATH))


def phase_model_files(cl: ChecklistRunner, models_dir: Path) -> None:
    print(f"\n{BOLD}Phase 2 — Model Files on Disk{RST}")
    print("─" * 50)

    base_dir = models_dir / "base"
    sft_dir  = models_dir / "sft"

    base_exists = base_dir.exists() and any(base_dir.iterdir())
    sft_exists  = sft_dir.exists()  and any(sft_dir.iterdir())

    cl.check("models/base/ exists and non-empty", base_exists, str(base_dir))
    cl.check("models/sft/  exists and non-empty", sft_exists,  str(sft_dir))

    if base_exists:
        files = list(base_dir.rglob("*.json")) + list(base_dir.rglob("*.safetensors")) + list(base_dir.rglob("*.bin"))
        cl.check("Base model has weight files", len(files) > 0, f"{len(files)} weight/config files found")

    if sft_exists:
        files = list(sft_dir.rglob("*.json")) + list(sft_dir.rglob("*.safetensors")) + list(sft_dir.rglob("*.bin"))
        cl.check("SFT model has weight files",  len(files) > 0, f"{len(files)} weight/config files found")


def phase_api_servers(cl: ChecklistRunner, base_port: int, sft_port: int) -> tuple[str, str]:
    """Returns (base_model_name, sft_model_name) for live inference test."""
    print(f"\n{BOLD}Phase 3 — API Server Health{RST}")
    print("─" * 50)

    base_name = ""
    sft_name  = ""

    for port, label in [(base_port, "base"), (sft_port, "sft")]:
        url = f"http://127.0.0.1:{port}/health"
        code, body = _http_get(url)
        ok = code == 200
        cl.check(f"{label.upper()} server /health  (port {port})", ok,
                 f"HTTP {code}" if not ok else "")

        if ok:
            try:
                data = json.loads(body)
                model_id = data.get("model", "")
                cl.check(f"{label.upper()} server reports ready=True", data.get("ready", False),
                         f"model={model_id}")
                if label == "base":
                    base_name = model_id
                else:
                    sft_name = model_id
            except Exception:
                cl.skip(f"{label.upper()} server ready flag", "Could not parse /health JSON")

        urls_ok, _ = _http_get(f"http://127.0.0.1:{port}/v1/models")
        cl.check(f"{label.upper()} /v1/models endpoint", urls_ok == 200,
                 f"HTTP {urls_ok}" if urls_ok != 200 else "")

    return base_name, sft_name


def phase_live_inference(
    cl: ChecklistRunner,
    base_port: int,
    sft_port: int,
    base_name: str,
    sft_name: str,
) -> None:
    print(f"\n{BOLD}Phase 4 — Live Inference via OpenAI Client{RST}")
    print("─" * 50)

    test_message = "Say exactly: Hello World"

    for port, model_name, label in [
        (base_port, base_name or "base", "base"),
        (sft_port,  sft_name  or "sft",  "sft"),
    ]:
        endpoint = f"http://127.0.0.1:{port}/v1"
        output, err = _openai_chat(endpoint, model_name, test_message)
        has_output = bool(output) and not output.startswith("[ERROR]")
        cl.check(
            f"OpenAI client → {label.upper()} model returns non-empty output",
            has_output,
            f"output[:80]={output[:80]!r}" if has_output else f"error={err}",
        )


def phase_csv_outputs(cl: ChecklistRunner, csv_path: Path) -> list[dict]:
    print(f"\n{BOLD}Phase 5 — generate_outputs.py CSV Output{RST}")
    print("─" * 50)

    cl.check("model_outputs_test.csv exists", csv_path.exists(), str(csv_path))
    if not csv_path.exists():
        cl.skip("CSV column completeness",   "file missing")
        cl.skip("CSV row count",             "file missing")
        cl.skip("base_output non-empty",     "file missing")
        cl.skip("sft_output non-empty",      "file missing")
        cl.skip("No base_error entries",     "file missing")
        cl.skip("No sft_error entries",      "file missing")
        cl.skip("prompt_ids match expected", "file missing")
        return []

    rows = _load_csv(csv_path)
    if not rows:
        cl.skip("CSV column completeness", "CSV is empty")
        return []

    # Columns
    actual_cols = set(rows[0].keys())
    missing_cols = [c for c in REQUIRED_CSV_FIELDS if c not in actual_cols]
    cl.check(
        "CSV has all required columns",
        len(missing_cols) == 0,
        f"missing: {missing_cols}" if missing_cols else f"{len(actual_cols)} columns present",
    )

    # Row count
    expected_count = len(TEST_PROMPT_IDS)
    cl.check(
        f"CSV has {expected_count} rows (one per test prompt)",
        len(rows) >= expected_count,
        f"found {len(rows)} rows",
    )

    # Outputs non-empty
    base_empty = [r["prompt_id"] for r in rows if not r.get("base_output", "").strip()]
    sft_empty  = [r["prompt_id"] for r in rows if not r.get("sft_output",  "").strip()]
    cl.check("base_output non-empty for all rows", len(base_empty) == 0,
             f"empty for: {base_empty}" if base_empty else "")
    cl.check("sft_output non-empty for all rows",  len(sft_empty) == 0,
             f"empty for: {sft_empty}"  if sft_empty  else "")

    # Errors
    base_errs = [r["prompt_id"] for r in rows if r.get("base_error", "").startswith("[ERROR]")]
    sft_errs  = [r["prompt_id"] for r in rows if r.get("sft_error",  "").startswith("[ERROR]")]
    cl.check("No [ERROR] in base_error column", len(base_errs) == 0,
             f"errored prompts: {base_errs}" if base_errs else "")
    cl.check("No [ERROR] in sft_error column",  len(sft_errs) == 0,
             f"errored prompts: {sft_errs}"  if sft_errs  else "")

    # prompt_id coverage
    found_ids = {r["prompt_id"] for r in rows}
    covered   = [pid for pid in TEST_PROMPT_IDS if pid in found_ids]
    cl.check(
        f"All test prompt IDs present in CSV",
        len(covered) == len(TEST_PROMPT_IDS),
        f"covered {len(covered)}/{len(TEST_PROMPT_IDS)}: {covered}",
    )

    return rows


def phase_evaluation_results(cl: ChecklistRunner, results_dir: Path) -> dict:
    print(f"\n{BOLD}Phase 6 — run_evaluation.py Output Files{RST}")
    print("─" * 50)

    cl.check("results/ directory exists", results_dir.exists(), str(results_dir))
    if not results_dir.exists():
        for label in ["evaluation_results_*.json", "summary_*.json", "report_*.md", "per_prompt_review_*.csv"]:
            cl.skip(f"{label} produced", "results/ missing")
        return {}

    for pattern, label in [
        ("evaluation_results_*.json", "evaluation_results JSON"),
        ("summary_*.json",            "summary JSON"),
        ("report_*.md",               "Markdown report"),
        ("per_prompt_review_*.csv",   "per-prompt review CSV"),
    ]:
        latest = _latest_file(results_dir, pattern)
        cl.check(f"{label} produced", latest is not None,
                 str(latest) if latest else f"no match for {pattern}")

    summary_path = _latest_file(results_dir, "summary_*.json")
    if not summary_path:
        return {}

    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        cl.check("summary JSON is valid JSON", True, str(summary_path))
        return data
    except Exception as exc:
        cl.check("summary JSON is valid JSON", False, str(exc))
        return {}


def phase_metrics(cl: ChecklistRunner, summary: dict, results_dir: Path) -> None:
    print(f"\n{BOLD}Phase 7 — Metric Completeness & Correctness{RST}")
    print("─" * 50)

    if not summary:
        cl.skip("Required metric keys present", "summary JSON not loaded")
        cl.skip("Numeric metric ranges",         "summary JSON not loaded")
        cl.skip("Verdict present in report",     "summary JSON not loaded")
        return

    # Key presence
    missing_keys = [k for k in REQUIRED_SUMMARY_KEYS if k not in summary]
    cl.check(
        "All required metric keys present in summary",
        len(missing_keys) == 0,
        f"missing: {missing_keys}" if missing_keys else f"all {len(REQUIRED_SUMMARY_KEYS)} keys present",
    )

    # Range checks
    for key in ["base_if_rate", "sft_if_rate", "base_hallucination_rate", "sft_hallucination_rate"]:
        if key in summary:
            val = summary[key]
            in_range = isinstance(val, (int, float)) and 0.0 <= val <= 1.0
            cl.check(f"{key} in [0.0, 1.0]", in_range, f"value={val}")

    cl.check(
        "total_prompts > 0",
        summary.get("total_prompts", 0) > 0,
        f"total_prompts={summary.get('total_prompts')}",
    )

    cl.check(
        "new_hallucination_count ≥ 0",
        isinstance(summary.get("new_hallucination_count", -1), int)
        and summary["new_hallucination_count"] >= 0,
        f"value={summary.get('new_hallucination_count')}",
    )

    # Verdict in report
    report_path = _latest_file(results_dir, "report_*.md")
    if report_path:
        report_text = report_path.read_text(encoding="utf-8")
        has_verdict = "PASS" in report_text or "FAIL" in report_text or "verdict" in report_text.lower()
        cl.check(
            "Verdict present in Markdown report",
            has_verdict,
            str(report_path),
        )
    else:
        cl.skip("Verdict present in Markdown report", "report file not found")

    # Print key metrics for human review
    print(f"\n  {BOLD}Key Metrics:{RST}")
    print(f"    Prompts evaluated     : {summary.get('total_prompts', 'N/A')}")
    print(f"    Base IF rate          : {summary.get('base_if_rate',  'N/A'):.3f}" if 'base_if_rate' in summary else "    Base IF rate          : N/A")
    print(f"    SFT  IF rate          : {summary.get('sft_if_rate',   'N/A'):.3f}" if 'sft_if_rate'  in summary else "    SFT  IF rate          : N/A")
    print(f"    IF improvement        : {summary.get('if_improvement', 'N/A'):.3f}" if 'if_improvement' in summary else "    IF improvement        : N/A")
    print(f"    Base hallucination %  : {summary.get('base_hallucination_rate', 'N/A'):.3f}" if 'base_hallucination_rate' in summary else "    Base hallucination %  : N/A")
    print(f"    SFT  hallucination %  : {summary.get('sft_hallucination_rate',  'N/A'):.3f}" if 'sft_hallucination_rate'  in summary else "    SFT  hallucination %  : N/A")
    print(f"    New hallucinations    : {summary.get('new_hallucination_count', 'N/A')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Option B end-to-end validation checklist.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", default=str(DEFAULT_CSV),
                        help="Path to model_outputs_test.csv produced by generate_outputs.py.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                        help="Directory containing run_evaluation.py outputs.")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR),
                        help="Directory containing models/base and models/sft.")
    parser.add_argument("--base-port", type=int, default=BASE_PORT)
    parser.add_argument("--sft-port",  type=int, default=SFT_PORT)
    parser.add_argument(
        "--skip-server-checks",
        action="store_true",
        help="Skip Phases 3 and 4 (server health + live inference). "
             "Use when servers are not running (e.g., validating offline artifacts).",
    )
    args = parser.parse_args()

    csv_path     = Path(args.csv)
    results_dir  = Path(args.results_dir)
    models_dir   = Path(args.models_dir)

    print("\n" + "=" * 65)
    print(f"{BOLD}  OPTION B — END-TO-END VALIDATION CHECKLIST{RST}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    cl = ChecklistRunner()

    # Phase 1 — Environment
    phase_environment(cl)

    # Phase 2 — Model files
    phase_model_files(cl, models_dir)

    # Phase 3 — Server health
    if args.skip_server_checks:
        print(f"\n{BOLD}Phase 3 — API Server Health{RST}")
        print("─" * 50)
        cl.skip("Base/SFT server health", "--skip-server-checks flag set")
        base_name, sft_name = "", ""
    else:
        base_name, sft_name = phase_api_servers(cl, args.base_port, args.sft_port)

    # Phase 4 — Live inference
    if args.skip_server_checks:
        print(f"\n{BOLD}Phase 4 — Live Inference{RST}")
        print("─" * 50)
        cl.skip("Live inference test", "--skip-server-checks flag set")
    else:
        phase_live_inference(cl, args.base_port, args.sft_port, base_name, sft_name)

    # Phase 5 — CSV
    phase_csv_outputs(cl, csv_path)

    # Phase 6 — Evaluation results
    summary = phase_evaluation_results(cl, results_dir)

    # Phase 7 — Metrics
    phase_metrics(cl, summary, results_dir)

    # Final summary
    all_passed = cl.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
