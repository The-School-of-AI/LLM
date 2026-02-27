import yaml

with open("config/curriculum.yaml") as f:
    curr = yaml.safe_load(f)

secondary = curr.get("languages", {}).get("secondary", [])
print("Secondary languages from YAML:")
for lang_spec in secondary:
    code = lang_spec.get("code")
    max_share = lang_spec.get("max_token_share")
    earliest = lang_spec.get("earliest_stage")
    print(
        f"  Code: {code}, Max share: {max_share}, Earliest: {earliest}, Type of max_share: {type(max_share)}"
    )
