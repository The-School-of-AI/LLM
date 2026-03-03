#!/usr/bin/env bash
# =============================================================================
# patch_configs.sh — Patches RULER/scripts config files for Gemma models.
# Run once after cloning the RULER repo.
# Usage: bash patch_configs.sh
# =============================================================================

set -euo pipefail

SCRIPTS_DIR="$(pwd)/RULER/scripts"

if [ ! -d "${SCRIPTS_DIR}" ]; then
    echo "[ERROR] RULER not found. Clone it first:"
    echo "  git clone https://github.com/NVIDIA/RULER.git RULER"
    exit 1
fi

echo "[1/3] Patching config_models.sh — adding gemma-1b and gemma-2b entries..."

# Check if already patched
if grep -q "gemma-2b)" "${SCRIPTS_DIR}/config_models.sh"; then
    echo "  -> Already patched, skipping."
else
    # Insert gemma entries before the closing 'esac'
    python3 - <<'PYEOF'
import re, os

path = os.path.join("RULER", "scripts", "config_models.sh")
with open(path) as f:
    content = f.read()

gemma_block = '''
        gemma-2b)
            MODEL_PATH="${MODEL_DIR}/google/gemma-2b"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="vllm"
            TOKENIZER_PATH="${MODEL_DIR}/google/gemma-2b"
            TOKENIZER_TYPE="hf"
            ;;

        gemma-1b)
            MODEL_PATH="${MODEL_DIR}/google/gemma-1b"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="vllm"
            TOKENIZER_PATH="${MODEL_DIR}/google/gemma-1b"
            TOKENIZER_TYPE="hf"
            ;;

        gemma-2b-it)
            MODEL_PATH="${MODEL_DIR}/google/gemma-2b-it"
            MODEL_TEMPLATE_TYPE="gemma"
            MODEL_FRAMEWORK="vllm"
            TOKENIZER_PATH="${MODEL_DIR}/google/gemma-2b-it"
            TOKENIZER_TYPE="hf"
            ;;

        gemma-1b-it)
            MODEL_PATH="${MODEL_DIR}/google/gemma-1b-it"
            MODEL_TEMPLATE_TYPE="gemma"
            MODEL_FRAMEWORK="vllm"
            TOKENIZER_PATH="${MODEL_DIR}/google/gemma-1b-it"
            TOKENIZER_TYPE="hf"
            ;;
'''

# Insert before 'esac'
content = content.replace('\n    esac', gemma_block + '\n    esac')

with open(path, "w") as f:
    f.write(content)

print("  -> gemma-1b, gemma-2b, gemma-1b-it, gemma-2b-it added.")
PYEOF
fi

echo "[2/3] Patching data/template.py — adding gemma chat template..."

TEMPLATE_FILE="${SCRIPTS_DIR}/data/template.py"
if grep -q '"gemma"' "${TEMPLATE_FILE}" 2>/dev/null; then
    echo "  -> Already patched, skipping."
elif [ -f "${TEMPLATE_FILE}" ]; then
    python3 - <<'PYEOF'
import os

path = os.path.join("RULER", "scripts", "data", "template.py")
with open(path) as f:
    content = f.read()

gemma_template = '''
    "gemma": (
        "<start_of_turn>user\\n{context}\\n\\n{query}<end_of_turn>\\n"
        "<start_of_turn>model\\n"
    ),
'''

# Insert after the "base" entry if it exists
if '"base"' in content:
    content = content.replace(
        '"base": "{context}\\n\\n{query}",',
        '"base": "{context}\\n\\n{query}",' + gemma_template
    )
    with open(path, "w") as f:
        f.write(content)
    print("  -> gemma template added.")
else:
    print("  -> WARNING: Could not find 'base' template to anchor insertion.")
    print("     Manually add the gemma template to template.py (see SETUP.md).")
PYEOF
else
    echo "  -> WARNING: ${TEMPLATE_FILE} not found. Skipping template patch."
    echo "     This is expected if the RULER repo is newer. Refer to SETUP.md."
fi

echo "[3/3] Setting SEQ_LENGTHS in config_models.sh to L4-friendly defaults..."

python3 - <<'PYEOF'
import re, os

path = os.path.join("RULER", "scripts", "config_models.sh")
with open(path) as f:
    content = f.read()

# Replace SEQ_LENGTHS block with L4-friendly values
content = re.sub(
    r'SEQ_LENGTHS=\(\s*[^)]*\)',
    'SEQ_LENGTHS=(\n    4096\n    8192\n)',
    content,
    flags=re.DOTALL
)

with open(path, "w") as f:
    f.write(content)

print("  -> SEQ_LENGTHS set to (4096, 8192). Edit RULER/scripts/config_models.sh for more.")
PYEOF

echo ""
echo "[OK] All patches applied. You can now run:"
echo ""
echo "  bash run_ruler.sh gemma-2b synthetic"
echo ""
