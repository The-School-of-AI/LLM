import sys
from pathlib import Path

# Add current directory to path so we can import curriculum_tags
sys.path.append(str(Path(__file__).parent.parent))

from curriculum_tags import CurriculumTagger


def test_metrics():
    print("Initializing CurriculumTagger...")
    # Initialize tagger (will load metrics from metrics_config.yaml)
    # We assume we are running from experiments/2_curirculum_architects/
    tagger = CurriculumTagger("curriculum.yaml")

    # Sample text that should trigger various metrics
    sample_text = """
    Let's think step by step.
    First, we define the function:
    
    def calculate_pi(n_terms):
        pi = 0
        for i in range(n_terms):
            pi += 4 * (-1)**i / (2*i + 1)
        return pi
        
    Therefore, the value of pi is approximately 3.14159.
    Action: Calculator.calculate(3.14 * 2)
    Observation: 6.28
    
    ∑ x_i = 100
    """

    sample = {"text": sample_text, "id": "test_1"}

    print("\nProcessing sample...")
    tagged = tagger.tag_sample(sample)

    tags = tagged.get("curriculum_tags", {})
    print("\nGenerated Tags:")
    import json

    print(json.dumps(tags, indent=2))

    # assertions to verify metrics ran
    print("\nVerifying metrics...")

    # Check Tokenizer Difficulty
    # Check Tokenizer Difficulty
    if "tokenizer_difficulty" in tags:
        print("[OK] TokenizerDifficultyMetric ran")
        td = tags["tokenizer_difficulty"]
        # Updated to check for top-level stats since banding logic was removed
        if "avg_token_id" in td:
            print(f"   - Avg Token ID: {td['avg_token_id']}")
    else:
        print("[FAIL] TokenizerDifficultyMetric missing")

    # Check Structural Density
    if "structural_density" in tags:
        print("[OK] StructuralDensityMetric ran")
        sd = tags["structural_density"]
        print(f"   - Structural Density: {sd.get('structural_density')}")
        print(f"   - Symbolic Density: {sd.get('symbolic_density')}")
    else:
        print("[FAIL] StructuralDensityMetric missing")

    # Check CoT Scanner
    if "cot_scanner" in tags:
        print("[OK] CoTScannerMetric ran")
        cot = tags["cot_scanner"]
        print(f"   - Has CoT: {cot.get('has_cot')}")
        print(f"   - Has Agentic: {cot.get('has_agentic')}")
    else:
        print("[FAIL] CoTScannerMetric missing")

    # Check Band Assignment
    if "band_assignment" in tags:
        print("[OK] BandAssignmentMetric ran")
        ba = tags["band_assignment"]
        print(f"   - Band: {ba.get('band')}")
        print(f"   - Reason: {ba.get('reason')}")
    else:
        print("[FAIL] BandAssignmentMetric missing")


if __name__ == "__main__":
    test_metrics()
