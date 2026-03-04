#!/usr/bin/env bash
# =============================================================================
# patch_configs.sh — Patches RULER/scripts config files for Gemma models.
# Run once after cloning the RULER repo. Safe to re-run (idempotent per model).
# Usage: bash patch_configs.sh
# =============================================================================

set -euo pipefail

SCRIPTS_DIR="$(pwd)/RULER/scripts"

if [ ! -d "${SCRIPTS_DIR}" ]; then
    echo "[ERROR] RULER not found. Clone it first:"
    echo "  git clone https://github.com/NVIDIA/RULER.git RULER"
    exit 1
fi

CONFIG_MODELS="${SCRIPTS_DIR}/config_models.sh"

echo "[1/3] Patching config_models.sh — adding Gemma model entries..."

# Each model is added individually — safe to re-run (skips if already present)

add_model() {
    local model_name="$1"
    local model_path_suffix="$2"
    local template_type="$3"

    if grep -q "${model_name})" "${CONFIG_MODELS}"; then
        echo "  -> ${model_name}: already present, skipping."
    else
        # Insert before 'esac' using awk
        awk -v name="${model_name}" -v path="${model_path_suffix}" -v tmpl="${template_type}" '
        /^    esac/ {
            print "        " name ")"
            print "            MODEL_PATH=\"${MODEL_DIR}/" path "\""
            print "            MODEL_TEMPLATE_TYPE=\"" tmpl "\""
            print "            MODEL_FRAMEWORK=\"vllm\""
            print "            TOKENIZER_PATH=\"${MODEL_DIR}/" path "\""
            print "            TOKENIZER_TYPE=\"hf\""
            print "            ;;"
            print ""
        }
        { print }
        ' "${CONFIG_MODELS}" > "${CONFIG_MODELS}.tmp" && mv "${CONFIG_MODELS}.tmp" "${CONFIG_MODELS}"
        echo "  -> ${model_name}: added (template=${template_type})."
    fi
}

add_model "gemma-1b"    "google/gemma-1b"    "base"
add_model "gemma-2b"    "google/gemma-2b"    "base"
add_model "gemma-3b"    "google/gemma-3b"    "base"
add_model "gemma-1b-it" "google/gemma-1b-it" "gemma"
add_model "gemma-2b-it" "google/gemma-2b-it" "gemma"
add_model "gemma-3b-it"   "google/gemma-3b-it"   "gemma"
add_model "gemma-3-1b-pt" "google/gemma-3-1b-pt" "base"

echo ""
echo "[2/3] Patching data/template.py — adding gemma chat template..."

TEMPLATE_FILE="${SCRIPTS_DIR}/data/template.py"

if [ ! -f "${TEMPLATE_FILE}" ]; then
    echo "  -> WARNING: ${TEMPLATE_FILE} not found."
    echo "     The RULER version you cloned may not have this file — skipping."
elif grep -q '"gemma"' "${TEMPLATE_FILE}"; then
    echo "  -> Already patched, skipping."
else
    python3 - <<'PYEOF'
import os, re

path = os.path.join("RULER", "scripts", "data", "template.py")
with open(path) as f:
    content = f.read()

gemma_entry = (
    '    "gemma": (\n'
    '        "<start_of_turn>user\\n{context}\\n\\n{query}<end_of_turn>\\n"\n'
    '        "<start_of_turn>model\\n"\n'
    '    ),\n'
)

# Try to insert after "base" entry (handles various quote styles and spacing)
new_content = re.sub(
    r'("base"\s*:.+?\n)',
    lambda m: m.group(0) + "\n" + gemma_entry,
    content,
    count=1,
    flags=re.DOTALL
)

if new_content == content:
    # Fallback: insert before closing brace of the template dict
    new_content = re.sub(r'(\n\})', "\n" + gemma_entry + r"\1", content, count=1)

if new_content != content:
    with open(path, "w") as f:
        f.write(new_content)
    print("  -> gemma template added.")
else:
    print("  -> WARNING: Could not auto-patch template.py.")
    print("     Add this manually inside the template dict in:")
    print("     RULER/scripts/data/template.py")
    print('     "gemma": (')
    print('         "<start_of_turn>user\\n{context}\\n\\n{query}<end_of_turn>\\n"')
    print('         "<start_of_turn>model\\n"')
    print("     ),")
PYEOF
fi

echo ""
echo "[3/3] Setting SEQ_LENGTHS in config_models.sh to L4-friendly defaults..."

python3 - <<'PYEOF'
import re, os

path = os.path.join("RULER", "scripts", "config_models.sh")
with open(path) as f:
    content = f.read()

content = re.sub(
    r'SEQ_LENGTHS=\(\s*[^)]*\)',
    'SEQ_LENGTHS=(\n    4096\n    8192\n)',
    content,
    flags=re.DOTALL
)

with open(path, "w") as f:
    f.write(content)

print("  -> SEQ_LENGTHS set to (4096, 8192).")
PYEOF

echo ""
echo "[OK] All patches applied. You can now run:"
echo ""
echo "  bash run_ruler.sh gemma-3b-it synthetic"
echo ""
