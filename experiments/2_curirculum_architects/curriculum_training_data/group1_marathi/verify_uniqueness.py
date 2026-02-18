#!/usr/bin/env python3
import glob
import os


def check_uniqueness():
    files = sorted(glob.glob("group1_s*.txt"))

    print(
        f"{'File Name':<20} | {'Total Lines':>12} | {'Unique Lines':>12} | {'Redundant %':>12}"
    )
    print("-" * 65)

    for file_path in files:
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            total_lines = len(lines)
            unique_lines = len(set(lines))

            if total_lines > 0:
                redundancy = (1 - unique_lines / total_lines) * 100
            else:
                redundancy = 0.0

            print(
                f"{file_name:<20} | {total_lines:>12,} | {unique_lines:>12,} | {redundancy:>11.2f}%"
            )
        except Exception as e:
            print(f"{file_name:<20} | Error: {e}")


if __name__ == "__main__":
    check_uniqueness()
