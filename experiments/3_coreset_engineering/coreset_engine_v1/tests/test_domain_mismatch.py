# Test what happens with domain filtering
from src.core.types import DifficultyBand
from src.curriculum.loader import CurriculumLoader

loader = CurriculumLoader('config/curriculum.yaml')
ok, errors = loader.load()

# Check B3 allowed domains from curriculum
b3_band = loader.bands.get(DifficultyBand.B3)
print(f'B3 allowed_domains from curriculum: {b3_band.allowed_domains}')

# Check what domains exist in data
import json
from pathlib import Path
from collections import defaultdict

chunks_file = Path('data/datasets/large_sample_chunks.jsonl')
domains_by_band = defaultdict(set)

with open(chunks_file) as f:
    for line in f:
        chunk = json.loads(line)
        band = chunk.get('band')
        domain = chunk.get('domain')
        domains_by_band[band].add(domain)

print(f'B3 domains in data: {sorted(domains_by_band["B3"])}')

# Now simulate what the bucket creation does
allowed_b3_domains = set(b3_band.allowed_domains)
actual_b3_domains = domains_by_band.get('B3', set())
overlap = allowed_b3_domains & actual_b3_domains
missing = allowed_b3_domains - actual_b3_domains

print(f'\nOverlap (bucket creation will use these): {overlap}')
print(f'Missing from data (will be skipped): {missing}')

# Check all bands
print("\n=== Domain Mismatch Across All Bands ===")
for band_name in ['B0', 'B1', 'B2', 'B3', 'B4', 'B5']:
    band = DifficultyBand(band_name)
    band_def = loader.bands.get(band)
    if band_def:
        allowed = set(band_def.allowed_domains)
        actual = domains_by_band.get(band_name, set())
        overlap = allowed & actual
        missing = allowed - actual
        print(f"\n{band_name}:")
        print(f"  Curriculum allows: {sorted(allowed)}")
        print(f"  Data has: {sorted(actual)}")
        print(f"  Usable overlap: {sorted(overlap)}")
        if missing:
            print(f"  Missing from data: {sorted(missing)}")
