#!/usr/bin/env python3
"""
Utility script to combine group1_part1.txt and group1_part2.txt back into group1.txt
Useful if you need the full dataset in a single file.
"""

import os


def combine_group1_parts():
    """Combine group1_part1.txt and group1_part2.txt into group1.txt"""
    output_dir = "output"
    part1_path = os.path.join(output_dir, "group1_part1.txt")
    part2_path = os.path.join(output_dir, "group1_part2.txt")
    combined_path = os.path.join(output_dir, "group1.txt")

    if not os.path.exists(part1_path):
        print(f"Error: {part1_path} not found")
        return

    if not os.path.exists(part2_path):
        print(f"Error: {part2_path} not found")
        return

    print(f"Combining {part1_path} and {part2_path}...")

    with open(combined_path, "w", encoding="utf-8") as outfile:
        # Write part 1
        with open(part1_path, "r", encoding="utf-8") as infile:
            outfile.write(infile.read())

        # Write part 2
        with open(part2_path, "r", encoding="utf-8") as infile:
            outfile.write(infile.read())

    combined_size = os.path.getsize(combined_path) / (1024 * 1024)
    print(f"✓ Successfully combined into {combined_path}")
    print(f"  Combined file size: {combined_size:.2f} MB")


if __name__ == "__main__":
    combine_group1_parts()
