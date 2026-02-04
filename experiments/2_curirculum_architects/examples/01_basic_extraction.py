"""Example: Basic usage of curriculum extractor for single record processing.

This example demonstrates:
- Initializing the extractor
- Processing single records (read-only)
- Handling rejections
- Viewing flattened output
"""

from pathlib import Path

from curriculum_extractor import CurriculumExtractor


def main():
    """Demonstrate basic curriculum extraction."""
    curriculum_path = Path(__file__).parent.parent / "curriculum.yaml"

    print("=" * 80)
    print("CURRICULUM EXTRACTOR - Basic Usage")
    print("=" * 80)

    # Initialize extractor with timing enabled
    extractor = CurriculumExtractor(curriculum_path, track_timing=True)

    print(f"\n[OK] Loaded {len(extractor.plugins)} metrics:")
    for plugin in extractor.plugins:
        print(f"  - {plugin.name} (level {plugin.level})")

    # Sample records - these are NEVER modified
    samples = [
        {
            "id": "sample_1",
            "text": "Hello world! This is a simple sentence for testing.",
            "source": "example",
            "lang": "en",
        },
        {
            "id": "sample_2",
            "text": """
def fibonacci(n):
    '''Calculate fibonacci number recursively.'''
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Example usage
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
            """,
            "source": "code_example",
        },
        {
            "id": "sample_3",
            "text": """
The implementation of quantum entanglement phenomena requires 
sophisticated mathematical frameworks incorporating Hilbert space 
representations and non-commutative operator algebras. Recent 
advancements in quantum error correction have demonstrated the 
viability of fault-tolerant quantum computation using topological 
codes such as the surface code.
            """,
            "source": "academic",
        },
    ]

    print("\n" + "=" * 80)
    print("EXTRACTING METADATA (records are read-only)")
    print("=" * 80)

    for sample in samples:
        print(f"\n--- Sample: {sample['id']} ---")
        print(f"Source: {sample['source']}")
        print(f"Text preview: {sample['text'][:50].strip()}...")

        # Extract metadata - record is NOT modified
        metadata, rejection = extractor.extract_record(
            sample, source_file="example.parquet"
        )

        if rejection:
            print("\n  ❌ REJECTED")
            print(f"     Reason: {rejection.rejected_reason}")
            print(f"     Rejected at: {rejection.rejected_at}")
        else:
            print("\n  ✓ EXTRACTED")
            # Show key metrics
            for key, value in metadata.items():
                if not key.startswith("opt_"):  # Skip optional columns
                    print(f"     {key}: {value}")

    # Show timing stats
    timing = extractor.get_timing_stats()
    if timing:
        print("\n" + "=" * 80)
        print("TIMING STATISTICS")
        print("=" * 80)
        for metric_name, stats in timing.items():
            print(f"  {metric_name}:")
            print(f"    - calls: {stats['count']}")
            print(f"    - avg: {stats['mean_ms']:.3f} ms")
            print(f"    - total: {stats['total_seconds']:.3f} s")

    print("\n[OK] Basic extraction complete.")
    print("\nNote: Original records were NOT modified (read-only processing)")


if __name__ == "__main__":
    main()
