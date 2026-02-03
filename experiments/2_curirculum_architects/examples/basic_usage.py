"""Simple example showing curriculum tagging with auto-discovery."""

import json
from pathlib import Path

from curriculum_tags import CurriculumTagger


def main():
    """Demonstrate curriculum tagging."""
    curriculum_path = Path(__file__).parent.parent / "curriculum.yaml"

    # Simple! Auto-loads metrics from metrics_config.yaml
    print("=" * 80)
    print("Auto-Loading Metrics from metrics_config.yaml")
    print("=" * 80)
    tagger = CurriculumTagger(curriculum_path)
    print(f"[OK] Loaded {len(tagger.plugins)} metrics automatically")
    for plugin in tagger.plugins:
        print(f"  - {plugin.name}")

    # Tag some samples
    samples = [
        {
            "id": "sample_1",
            "text": "Hello world! This is a simple sentence.",
            "source": "example",
            "lang": "en",
        },
        {
            "id": "sample_2",
            "text": """
            def fibonacci(n):
                if n <= 1:
                    return n
                return fibonacci(n-1) + fibonacci(n-2)
            """,
            "source": "code_example",
        },
        {
            "id": "sample_3",
            "text": """
            The implementation of quantum entanglement phenomena requires 
            sophisticated mathematical frameworks incorporating Hilbert space 
            representations and non-commutative operator algebras.
            """,
            "source": "academic",
        },
    ]

    print("\n" + "=" * 80)
    print("Tagging Samples")
    print("=" * 80)

    for sample in samples:
        tagged = tagger.tag_sample(sample)
        print(json.dumps(tagged, indent=2))
        print("-" * 80)
    print("\n[OK] Tagging complete.")


if __name__ == "__main__":
    main()
