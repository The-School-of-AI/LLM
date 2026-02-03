from coreset_engine.scoring.perplexity import DifficultyScorer


def test_difficulty_scorer_defaults():
    scorer = DifficultyScorer()

    # CASE 1: Simple/Repetitive Text (Should be EASY -> Low Score)
    simple_text = "hello world " * 100
    score_simple = scorer.score({"text": simple_text})

    # CASE 2: Complex/Random Text (Should be HARD -> High Score)
    # Using a sentence with unique words and more entropy
    # artificially make it random-ish to ensure high entropy for test
    import random

    random_text = "".join([chr(random.randint(65, 90)) for _ in range(1000)])

    score_complex = scorer.score({"text": random_text})

    print(f"Simple Score: {score_simple}")
    print(f"Complex Score: {score_complex}")

    assert (
        score_complex > score_simple
    ), "Random text should have higher entropy (difficulty) than repetitive text"


def test_precomputed_perplexity():
    scorer = DifficultyScorer()
    record = {"text": "ignore me", "perplexity": 150.5}
    assert scorer.score(record) == 150.5
