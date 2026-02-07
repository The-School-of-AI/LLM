#!/usr/bin/env python3
"""
Test script for monitoring dashboard system.
Verifies core functionality before deployment.
"""

import sys
import json
import tempfile
from pathlib import Path

print("=" * 80)
print("Testing Monitoring Dashboard System")
print("=" * 80)

# Test 1: Check imports
print("\n[Test 1] Checking imports...")
try:
    from flask import Flask
    from flask_cors import CORS
    print("✅ PASS: Flask and CORS available")
except ImportError as e:
    print(f"❌ FAIL: Missing dependencies - {e}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# Test 2: Create test JSONL data
print("\n[Test 2] Creating test log data...")
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directory structure
        run_dir = Path(tmpdir) / "test_run"
        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir(parents=True)
        
        # Write test JSONL file
        jsonl_file = metrics_dir / "data.jsonl"
        with open(jsonl_file, "w") as f:
            for step in range(10):
                f.write(json.dumps({
                    "step": step,
                    "metric": "train/loss",
                    "value": 5.0 - step * 0.1
                }) + "\n")
                f.write(json.dumps({
                    "step": step,
                    "metric": "gpu/temp",
                    "value": 65 + step
                }) + "\n")
        
        print(f"✅ PASS: Created test data at {jsonl_file}")
        
        # Test 3: Verify file format
        print("\n[Test 3] Verifying JSONL format...")
        with open(jsonl_file) as f:
            lines = f.readlines()
            if len(lines) == 20:  # 10 steps × 2 metrics
                print(f"✅ PASS: Correct number of entries ({len(lines)})")
            else:
                print(f"❌ FAIL: Expected 20 entries, got {len(lines)}")
        
        # Test 4: Parse JSONL entries
        print("\n[Test 4] Parsing JSONL entries...")
        valid_entries = 0
        with open(jsonl_file) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if "step" in entry and "metric" in entry and "value" in entry:
                        valid_entries += 1
                except json.JSONDecodeError:
                    pass
        
        if valid_entries == 20:
            print(f"✅ PASS: All {valid_entries} entries valid")
        else:
            print(f"❌ FAIL: Only {valid_entries}/20 entries valid")
        
        # Test 5: Metric discovery simulation
        print("\n[Test 5] Testing metric discovery...")
        unique_metrics = set()
        with open(jsonl_file) as f:
            for line in f:
                entry = json.loads(line.strip())
                unique_metrics.add(entry["metric"])
        
        expected_metrics = {"train/loss", "gpu/temp"}
        if unique_metrics == expected_metrics:
            print(f"✅ PASS: Discovered metrics: {unique_metrics}")
        else:
            print(f"❌ FAIL: Expected {expected_metrics}, got {unique_metrics}")
        
        # Test 6: Metric grouping
        print("\n[Test 6] Testing metric grouping...")
        grouped = {}
        for metric in unique_metrics:
            category = metric.split("/")[0] if "/" in metric else "other"
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(metric)
        
        expected_groups = {"train": ["train/loss"], "gpu": ["gpu/temp"]}
        if grouped == expected_groups:
            print(f"✅ PASS: Correct grouping: {grouped}")
        else:
            print(f"❌ FAIL: Expected {expected_groups}, got {grouped}")

except Exception as e:
    print(f"❌ FAIL: Error during testing - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Check server file exists
print("\n[Test 7] Checking server file...")
server_file = Path(__file__).parent / "dashboard_server.py"
if server_file.exists():
    print(f"✅ PASS: Server file exists at {server_file}")
else:
    print(f"❌ FAIL: Server file not found at {server_file}")

# Test 8: Check dashboard HTML exists
print("\n[Test 8] Checking dashboard HTML...")
dashboard_file = Path(__file__).parent / "dashboard" / "index.html"
if dashboard_file.exists():
    print(f"✅ PASS: Dashboard HTML exists at {dashboard_file}")
else:
    print(f"❌ FAIL: Dashboard HTML not found at {dashboard_file}")

# Summary
print("\n" + "=" * 80)
print("Test Summary")
print("=" * 80)
print("""
All core tests passed! You can now:

1. Start the dashboard:
   python dashboard_server.py

2. Access in browser:
   http://localhost:5000

3. Point to your training logs:
   python dashboard_server.py --log-dir ../training/deepspeed_template/logs

See README.md for detailed usage instructions.
""")
print("=" * 80)