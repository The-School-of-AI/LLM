#!/bin/bash
# Validate generated coreset outputs

set -euo pipefail

MANIFEST_DIR=${1:-outputs/manifests}

echo "Validating coresets in: $MANIFEST_DIR"

# Check if any JSON files exist
shopt -s nullglob
manifests=("$MANIFEST_DIR"/*.json)

if [ ${#manifests[@]} -eq 0 ]; then
    echo "⚠️  No manifest files found in $MANIFEST_DIR"
    echo "Run 'make run-all' to generate manifests first."
    exit 0
fi

for manifest in "${manifests[@]}"; do
    echo ""
    echo "Validating: $(basename "$manifest")"
    python -m src.validation.validate --manifest "$manifest"
done

echo ""
echo "Validation complete!"
