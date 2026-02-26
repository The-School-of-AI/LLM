import json
import random
from pathlib import Path

import repro.seeds as seeds

# Test set_all_seeds() sets Python random deterministically


def test_set_all_seeds_sets_python_random():
    seeds.set_all_seeds(42)

    value1 = random.random()

    seeds.set_all_seeds(42)
    value2 = random.random()

    assert value1 == value2


# Test NumPy seeding (only if NumPy is available)


def test_set_all_seeds_sets_numpy_random():
    if seeds.np is None:
        return  # skip if numpy not installed

    seeds.set_all_seeds(123)
    v1 = seeds.np.random.rand()

    seeds.set_all_seeds(123)
    v2 = seeds.np.random.rand()

    assert v1 == v2


# Test Torch seeding (only if Torch is available)


def test_set_all_seeds_sets_torch_random():
    if seeds.torch is None:
        return  # skip if torch not installed

    seeds.set_all_seeds(999)

    v1 = seeds.torch.rand(1).item()

    seeds.set_all_seeds(999)
    v2 = seeds.torch.rand(1).item()

    assert v1 == v2


# Test capture_seeds() creates directories and file


def test_capture_seeds_creates_file(tmp_path: Path):
    output_path = tmp_path / "a/b/seeds.json"

    seeds.capture_seeds(7, output_path)

    assert output_path.exists()


# Test capture_seeds() JSON content


def test_capture_seeds_writes_correct_json(tmp_path: Path):
    output_path = tmp_path / "seeds.json"

    seeds.capture_seeds(101, output_path)

    data = json.loads(output_path.read_text())

    assert data["global_seed"] == 101
    assert data["python_random"] == 101
    assert data["numpy"] == (101 if seeds.np else None)
    assert data["torch"] == (101 if seeds.torch else None)


# Test JSON formatting (pretty-printed)


def test_capture_seeds_json_is_pretty_printed(tmp_path: Path):
    output_path = tmp_path / "seeds.json"

    seeds.capture_seeds(1, output_path)

    text = output_path.read_text()
    assert "\n  " in text
