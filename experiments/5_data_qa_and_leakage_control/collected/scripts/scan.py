#!/usr/bin/env python3
"""
Contamination Scanner - Entry Point
Usage: python scripts/scan.py <input_file> <team_name> <batch_name>
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.scanner import ContaminationScanner


def main():
    if len(sys.argv) < 4:
        print("Usage: python scripts/scan.py <input_file> <team_name> <batch_name>")
        sys.exit(1)

    input_file = sys.argv[1]
    team_name = sys.argv[2]
    batch_name = sys.argv[3]

    scanner = ContaminationScanner()
    is_approved, _ = scanner.scan_dataset(input_file, team_name, batch_name)

    sys.exit(0 if is_approved else 1)


if __name__ == "__main__":
    main()
