#!/usr/bin/env bash
# AWS B200/B300 setup helper for LightningLM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

WORK="${LIGHTNINGLM_ROOT:-$ROOT_DIR}"
VENV="${LIGHTNINGLM_VENV:-$WORK/.venv-b300}"
S3_BUCKET="${LIGHTNINGLM_PREBUILT_S3:-s3://t1-dataacquisition-checkpoints-2/training/8b_growth}"
PYTHON_BIN="${PYTHON_BIN:-python3.13}"

echo "=== AWS B200/B300 setup ==="
echo "Root: $WORK"
echo "Venv: $VENV"

cd "$WORK"
"$PYTHON_BIN" -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip setuptools wheel

echo "Installing PyTorch nightly cu130"
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu130

echo "Installing runtime packages"
pip install \
  deepspeed==0.18.9 \
  transformers==5.5.4 \
  datasets \
  accelerate \
  einops==0.8.2 \
  safetensors==0.7.0 \
  psutil \
  pynvml \
  requests \
  boto3 \
  pyyaml \
  huggingface_hub \
  pandas \
  numpy \
  scipy \
  tqdm \
  ninja

if command -v aws >/dev/null 2>&1; then
  SITE_PACKAGES="$VENV/lib64/python3.13/site-packages"
  if aws s3 ls "$S3_BUCKET/env_backup/b300_prebuilt_packages.tar.gz" --region us-west-2 >/dev/null 2>&1; then
    echo "Installing optional B300 prebuilt packages from S3"
    aws s3 cp "$S3_BUCKET/env_backup/b300_prebuilt_packages.tar.gz" /tmp/b300_prebuilt_packages.tar.gz \
      --region us-west-2 --only-show-errors
    cd "$SITE_PACKAGES"
    tar xzf /tmp/b300_prebuilt_packages.tar.gz
    rm /tmp/b300_prebuilt_packages.tar.gz
  else
    echo "Optional prebuilt package archive not found; continuing with pip-installed packages."
  fi
else
  echo "AWS CLI not found; skipping optional prebuilt packages."
fi

cd "$WORK"
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
import triton; print(f'Triton: {triton.__version__}')
import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')
"

python3 "$WORK/scripts/doctor.py"

echo ""
echo "Setup complete."
echo "Sync or generate data into data/training_shards_8k, then launch:"
echo "  source $VENV/bin/activate"
echo "  NUM_GPUS=8 bash scripts/run_120b_tqp.sh"
