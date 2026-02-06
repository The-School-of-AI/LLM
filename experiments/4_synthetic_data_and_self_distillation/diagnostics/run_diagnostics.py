"""
run_diagnostics.py — Fast Diagnostic Test Runner

Runs the 50 diagnostic tests against a model.
Designed for speed (< 1 min for all tests on local GPU).

Usage:
  python run_diagnostics.py                          # all tests
  python run_diagnostics.py --model qwen3:4b         # specific model
  python run_diagnostics.py --band B3                # specific band
  python run_diagnostics.py --skill RSN-ARITHMETIC   # specific skill
  python run_diagnostics.py --output results.json    # save results
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# Add parent to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from diagnostics.diagnostic_tests import (
    DIAGNOSTIC_TESTS,
    DiagnosticTest,
    evaluate_test,
    get_tests_for_skill,
    get_tests_for_band,
    get_test_summary,
)


# ================================================================
# OLLAMA CLIENT (minimal)
# ================================================================

OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def ollama_generate(model: str, prompt: str, max_tokens: int = 150) -> str:
    """Generate text from Ollama."""
    # Qwen3 models preamble before answering even with thinking off; enforce a floor
    max_tokens = max(max_tokens, 100)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,  # Top-level: disables Qwen3 thinking mode (ignored if nested in options)
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.1,
        },
    }
    
    url = f"{OLLAMA_BASE}/api/generate"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    
    return result.get("response", "").strip()


def check_ollama() -> bool:
    """Check if Ollama is running."""
    try:
        url = f"{OLLAMA_BASE}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False


# ================================================================
# TEST RUNNER
# ================================================================

def run_single_test(model: str, test: DiagnosticTest, verbose: bool = False) -> dict:
    """Run a single diagnostic test."""
    start = time.time()
    
    try:
        output = ollama_generate(model, test.prompt, test.max_tokens)
        elapsed_ms = (time.time() - start) * 1000
        
        result = evaluate_test(test, output)
        result.update({
            "test_id": test.id,
            "test_name": test.name,
            "band": test.band,
            "skill_buckets": test.skill_buckets,
            "elapsed_ms": round(elapsed_ms, 1),
            "error": None,
        })
        
        if verbose:
            status = "✓" if result["passed"] else "✗"
            print(f"  {status} {test.id}: {test.name} ({elapsed_ms:.0f}ms)")
            if not result["passed"]:
                print(f"      Expected: {test.expected}")
                print(f"      Got: {output[:60]}...")
        
        return result
        
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        if verbose:
            print(f"  ✗ {test.id}: ERROR - {str(e)[:50]}")
        return {
            "test_id": test.id,
            "test_name": test.name,
            "band": test.band,
            "skill_buckets": test.skill_buckets,
            "passed": False,
            "expected": test.expected,
            "actual": None,
            "failure_detected": None,
            "elapsed_ms": round(elapsed_ms, 1),
            "error": str(e),
        }


def run_all_tests(
    model: str,
    tests: list[DiagnosticTest] | None = None,
    verbose: bool = True,
) -> dict:
    """Run all diagnostic tests."""
    
    tests = tests or DIAGNOSTIC_TESTS
    results = []
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  DIAGNOSTIC TEST RUN: {model}")
        print(f"{'='*60}")
        print(f"  Tests to run: {len(tests)}")
    
    start_total = time.time()
    
    for i, test in enumerate(tests):
        result = run_single_test(model, test, verbose)
        results.append(result)
    
    elapsed_total = time.time() - start_total
    
    # Compute statistics
    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed
    pass_rate = passed / len(results) if results else 0
    
    # By band
    by_band = {}
    for r in results:
        band = r["band"]
        if band not in by_band:
            by_band[band] = {"passed": 0, "failed": 0}
        if r["passed"]:
            by_band[band]["passed"] += 1
        else:
            by_band[band]["failed"] += 1
    
    # By skill
    by_skill = {}
    for r in results:
        for skill in r["skill_buckets"]:
            if skill not in by_skill:
                by_skill[skill] = {"passed": 0, "failed": 0}
            if r["passed"]:
                by_skill[skill]["passed"] += 1
            else:
                by_skill[skill]["failed"] += 1
    
    # Failure analysis
    failures = [r for r in results if not r["passed"]]
    failure_modes = {}
    for f in failures:
        if f.get("failure_detected"):
            fm = f["failure_detected"]
            failure_modes[fm] = failure_modes.get(fm, 0) + 1
    
    summary = {
        "model": model,
        "total_tests": len(results),
        "passed": passed,
        "failed": failed,
        "pass_rate": round(pass_rate, 3),
        "elapsed_seconds": round(elapsed_total, 2),
        "by_band": by_band,
        "by_skill": by_skill,
        "failure_modes_detected": failure_modes,
        "results": results,
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"  Total: {len(results)} | Passed: {passed} | Failed: {failed}")
        print(f"  Pass Rate: {pass_rate*100:.1f}%")
        print(f"  Time: {elapsed_total:.1f}s ({elapsed_total/len(results)*1000:.0f}ms/test)")
        
        print(f"\n  By Band:")
        for band in sorted(by_band.keys()):
            stats = by_band[band]
            total = stats["passed"] + stats["failed"]
            rate = stats["passed"] / total if total > 0 else 0
            print(f"    {band}: {stats['passed']}/{total} ({rate*100:.0f}%)")
        
        print(f"\n  By Skill (top failures):")
        skill_rates = [
            (skill, s["passed"] / (s["passed"] + s["failed"]))
            for skill, s in by_skill.items()
        ]
        skill_rates.sort(key=lambda x: x[1])
        for skill, rate in skill_rates[:5]:
            if rate < 1.0:
                print(f"    {skill}: {rate*100:.0f}%")
        
        if failure_modes:
            print(f"\n  Failure Modes Detected:")
            for mode, count in sorted(failure_modes.items(), key=lambda x: -x[1])[:5]:
                print(f"    {mode}: {count}")
    
    return summary


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run diagnostic tests on a model"
    )
    parser.add_argument("--model", "-m", default="qwen3:4b",
                        help="Ollama model name")
    parser.add_argument("--band", "-b", default=None,
                        help="Filter by band (B0-B5)")
    parser.add_argument("--skill", "-s", default=None,
                        help="Filter by skill bucket")
    parser.add_argument("--output", "-o", default=None,
                        help="Save results to JSON file")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Minimal output")
    parser.add_argument("--list", action="store_true",
                        help="List all tests and exit")
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        summary = get_test_summary()
        print(f"\nDiagnostic Tests: {summary['total_tests']}")
        print(f"\nBy Band:")
        for band, count in sorted(summary["by_band"].items()):
            print(f"  {band}: {count}")
        print(f"\nBy Skill:")
        for skill, count in sorted(summary["by_skill"].items()):
            print(f"  {skill}: {count}")
        return
    
    # Check Ollama
    if not check_ollama():
        print("ERROR: Ollama is not running!")
        print("Start it with: ollama serve")
        sys.exit(1)
    
    # Filter tests
    tests = DIAGNOSTIC_TESTS
    if args.band:
        tests = get_tests_for_band(args.band)
    if args.skill:
        tests = [t for t in tests if args.skill in t.skill_buckets]
    
    if not tests:
        print("No tests match the specified filters")
        sys.exit(1)
    
    # Run tests
    results = run_all_tests(
        model=args.model,
        tests=tests,
        verbose=not args.quiet,
    )
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Exit code based on pass rate
    if results["pass_rate"] < 0.5:
        sys.exit(1)


if __name__ == "__main__":
    main()
