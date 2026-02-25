#!/usr/bin/env python3
"""
Contamination Scanner - Entry Point

Usage:
    python scripts/scan.py <input_file> <team_name> <batch_name>

Exit codes:
    0 — dataset approved (no contamination detected)
    1 — dataset rejected (contamination found) or error
"""

import sys
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.scanner import ContaminationScanner
from core.utils import get_git_info


def main() -> None:
    """Parse CLI arguments, run the scanner, and exit with the appropriate code."""
    if len(sys.argv) < 4:
        print("Usage: python scripts/scan.py <input_file> <team_name> <batch_name>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    team_name = sys.argv[2]
    batch_name = sys.argv[3]

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    if not input_path.is_file():
        print(f"Error: path is not a file: {input_path}")
        sys.exit(1)

    git_info = get_git_info()
    if git_info["dirty"] == "true":
        print(
            "⚠ Warning: repository has uncommitted changes. "
            "The report will be marked dirty — commit your changes for a clean audit trail."
        )

    try:
        scanner = ContaminationScanner()
        is_approved, _ = scanner.scan_dataset(input_path, team_name, batch_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during scan: {e}")
        sys.exit(1)

    sys.exit(0 if is_approved else 1)


if __name__ == "__main__":
    main()
