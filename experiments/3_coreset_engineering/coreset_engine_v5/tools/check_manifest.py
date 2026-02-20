import json

manifest_file = "output/coresets/1B/manifest.json"
with open(manifest_file) as f:
    manifest = json.load(f)

# Get band-domain distribution
band_domain_dist = {}
for entry in manifest["distribution"]:
    key = f"{entry['band']}_{entry['domain']}"
    if key not in band_domain_dist:
        band_domain_dist[key] = {"count": 0, "tokens": 0}
    band_domain_dist[key]["count"] += entry["count"]
    band_domain_dist[key]["tokens"] += entry["tokens"]

print("Band-Domain distribution in selected 1B:")
for key in sorted(band_domain_dist.keys()):
    info = band_domain_dist[key]
    pct = 100 * info["tokens"] / manifest["total_tokens"]
    print(f'{key}: {info["count"]} chunks, {info["tokens"]:,} tokens ({pct:.2f}%)')

# Check if reasoning is there
reasoning_found = [k for k in band_domain_dist.keys() if "reasoning" in k]
print(f"\nReasoning found: {reasoning_found}")
