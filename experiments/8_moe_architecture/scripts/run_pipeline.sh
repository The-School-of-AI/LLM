#!/bin/bash
# Main coreset generation pipeline

set -euo pipefail

STAGES=("stage_1b" "stage_3b" "stage_8b" "stage_moe")
SEED=${SEED:-42}
OUTPUT_DIR=${OUTPUT_DIR:-outputs}

echo "Starting coreset generation pipeline..."
echo "Seed: $SEED"
echo "Output directory: $OUTPUT_DIR"

for stage in "${STAGES[@]}"; do
    echo ""
    echo "========================================="
    echo "Processing stage: $stage"
    echo "========================================="
    
    python -m src.coreset_builder.main \
        --config "config/${stage}.yaml" \
        --seed "$SEED" \
        --output-dir "$OUTPUT_DIR"
    
    echo "Stage $stage completed."
done

echo ""
echo "Pipeline completed successfully!"
echo "Manifests saved to: $OUTPUT_DIR/manifests/"
